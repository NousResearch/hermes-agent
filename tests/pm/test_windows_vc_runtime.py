"""Windows MSVC builds PM stages carry the VC++ runtime their DLLs import.

Every upstream Windows llama.cpp build (ggml-base.dll, llama-server-impl.dll, ggml-vulkan.dll, ...)
imports MSVCP140.dll and VCRUNTIME140.dll, and the archives ship neither. A fresh Windows has no
MSVCP140.dll, so the in-process device probe could not load ggml-base.dll and budgeted a
discrete GPU from system RAM (#148249).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from pm import Lockfile, get_package, paths
from pm.install import _prepare_artifacts
from pm.packages import _unpack_vc_runtime

_MACHINE = {"x64": 0x8664, "arm64": 0xAA64}
_CRT = "Contents/VC/Redist/MSVC/14.51.36231/{arch}/Microsoft.VC145.CRT/"
_DEBUG = "Contents/VC/Redist/MSVC/14.51.36231/debug_nonredist/{arch}/Microsoft.VC145.DebugCRT/"


def _pe(arch: str) -> bytes:
    return b"MZ" + bytes(58) + (64).to_bytes(4, "little") + b"PE\0\0" + _MACHINE[arch].to_bytes(2, "little")


def _zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return path


def _redist(path: Path, arch: str) -> Path:
    """Visual Studio's CRT redist payload: release DLLs, debug DLLs, and package metadata."""
    crt, debug = _CRT.format(arch=arch), _DEBUG.format(arch=arch)
    return _zip(path, {
        "manifest.json": b"{}",
        "[Content_Types].xml": b"<Types/>",
        crt + "msvcp140.dll": _pe(arch),
        crt + "vcruntime140.dll": _pe(arch),
        # The ARM64 payload's CRT folder carries the x64 SEH helper for emulated code.
        crt + "vcruntime140_1.dll": _pe("x64"),
        debug + "msvcp140d.dll": _pe(arch),
    })


class _LocalArchives:
    def __init__(self, archives: list[Path]):
        self.archives = archives

    def fetch_many(self, artifacts, scratch, **_kwargs):
        return self.archives[:len(artifacts)]


def _stage(tmp_path: Path, name: str, target: str, members) -> Path:
    """Stage the locked artifacts through PM's unpack/merge/stage, from local stand-ins named as
    the lock names them: the redist payload for a .vsix, ``members(filename)`` for the rest."""
    package = get_package(name)
    lock = Lockfile(paths.lockfile_path())
    artifacts = lock.artifacts(name, target)
    arch = target.split("-")[1]
    archives = []
    for index, artifact in enumerate(artifacts):
        local = tmp_path / str(index) / artifact["url"].rsplit("/", 1)[-1]
        local.parent.mkdir()
        archives.append(_redist(local, arch) if local.name.endswith(".vsix") else _zip(local, members(local.name)))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    staged = _prepare_artifacts(package, _LocalArchives(archives), scratch, artifacts, lock.version(name), target)
    return package.binary(staged, target)


@pytest.mark.parametrize("name,target", [
    ("llamacpp-cpu", "win32-x64"),
    ("llamacpp-cpu", "win32-arm64"),
    ("llamacpp-vulkan", "win32-x64"),
    ("llamacpp-hip", "win32-x64"),
    ("llamacpp-cuda", "win32-x64"),
    ("llamacpp-cuda", "win32-arm64"),
])
def test_windows_engine_stages_the_vc_runtime_beside_llama_server(tmp_path, name, target):
    arch = target.split("-")[1]

    server = _stage(tmp_path, name, target, lambda filename: (
        {"cudart64_13.dll": _pe(arch)} if filename.startswith("cudart-")
        else {"llama-server.exe": _pe(arch), "ggml-base.dll": _pe(arch)}))

    assert server.is_file()
    beside = {p.name for p in server.parent.iterdir()}
    assert {"msvcp140.dll", "vcruntime140.dll", "ggml-base.dll"} <= beside
    assert ("vcruntime140_1.dll" in beside) is (arch == "x64")
    assert "msvcp140d.dll" not in beside
    assert not {"Contents", "manifest.json", "[Content_Types].xml"} & beside


def test_arm64_whisper_cpp_stages_the_vc_runtime_beside_whisper_cli(tmp_path):
    """whisper-cli.exe and whisper.dll import MSVCP140.dll too; upstream nests them under Release/."""
    cli = _stage(tmp_path, "whispercpp-cpu", "win32-arm64", lambda filename: {
        "Release/whisper-cli.exe": _pe("arm64"), "Release/whisper.dll": _pe("arm64")})

    assert cli.is_file()
    assert {"msvcp140.dll", "vcruntime140.dll", "whisper.dll"} <= {p.name for p in cli.parent.iterdir()}


def test_arm64_whisper_cpp_pin_carries_the_vc_runtime_its_fetch_urls_name():
    """`pm lock --bump whispercpp-cpu` rebuilds the row from fetch_urls; keep the two in step."""
    lock = Lockfile(paths.lockfile_path())
    urls = [artifact["url"] for artifact in lock.artifacts("whispercpp-cpu", "win32-arm64")]

    assert urls == get_package("whispercpp-cpu").fetch_urls(lock.version("whispercpp-cpu"), "win32-arm64")
    assert urls[-1].endswith(".vsix")


def test_vc_runtime_unpack_stages_only_plain_dll_names(tmp_path):
    """A member name is zip text, not a path: on Windows `..\\x.dll` under the CRT folder would land
    outside the staged tree, which PM's own extract() refuses for every other archive."""
    crt = _CRT.format(arch="x64")
    payload = _zip(tmp_path / "redist.vsix", {
        crt + "msvcp140.dll": _pe("x64"),
        crt + "..\\..\\escaped.dll": _pe("x64"),
    })
    staged = tmp_path / "out" / "tree"

    _unpack_vc_runtime("llamacpp-cpu", payload, staged, "win32-x64")

    assert sorted(p.name for p in staged.parent.rglob("*") if p.is_file()) == ["msvcp140.dll"]

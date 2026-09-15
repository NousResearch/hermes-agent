"""Native/cross-target execution and Windows directory-hold staging contracts."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import pytest

from pm.package import InstallError, Package
from pm.lock import Lockfile
from pm.store import Store

ensure_mod = importlib.import_module("pm.ensure")


class _FakePackage(Package):
    """unpack() turns the verified archive bytes into one binary file."""

    name = "stage-test"

    def missing_reason(self, target: str):
        return None

    def unpack(self, archive: Path, staged: Path, target: str) -> None:
        staged.mkdir(parents=True, exist_ok=True)
        (staged / "bin").mkdir(exist_ok=True)
        (staged / "bin" / "tool").write_bytes(archive.read_bytes())

    def verify(self, entry: Path, target: str) -> str:
        return "" if (entry / "bin" / "tool").is_file() else "bin/tool missing"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _seed_fetch_cache(store: Store, payload: bytes) -> None:
    """Pre-populate the store's download cache so store.fetch() proves the
    digest from local bytes instead of touching the network."""
    entry = store.entry(f"fetch-{_sha(payload)}")
    entry.mkdir(parents=True)
    (entry / "payload.bin").write_bytes(payload)


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(runtime))
    store = Store(runtime)
    package = _FakePackage()
    monkeypatch.setattr(ensure_mod, "get_package", lambda name: package)
    monkeypatch.setattr(ensure_mod, "_store", lambda: store)
    return store


def _arm_lock(monkeypatch, artifacts):
    from pm import paths
    lock = Lockfile(paths.store_root().parent / "stage-lock.json")
    lock.set_pin("node", "1.0", {"any": artifacts})
    lock.set_pin("stage-test", "1.0", {"any": artifacts})
    lock.save()
    monkeypatch.setattr(ensure_mod, "_lockfile", lambda: lock)
    return lock


TARGET = "linux-arm64-bionic"
ENTRY = "stage-test-1.0-linux-arm64-bionic"


@pytest.mark.parametrize("name,relative", [
    ("python", "bin/python3.14"), ("uv", "bin/uv"), ("node", "bin/node"),
])
def test_bionic_deb_stages_real_packages_without_host_execution(tmp_path, monkeypatch, name, relative):
    from pm import paths
    from tests.pm.test_deb_safety import _build_deb
    from pm.registry import get_package

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "store"))
    lock = Lockfile(tmp_path / "lock.json")
    monkeypatch.setattr(paths, "lockfile_path", lambda: lock.path)
    deb = tmp_path / "tool.deb"
    main = "data/data/com.termux/files/usr/" + relative
    _build_deb(deb, [(main, b"bionic fixture"), ("data/data/com.termux/files/usr/lib/keep", b"library")])
    digest = _sha(deb.read_bytes())
    lock.set_pin(name, "1.0", {TARGET: {"url": "https://example.invalid/tool.deb", "sha256": digest}})
    lock.save()
    cached = paths.store_root() / f"fetch-{digest}"
    cached.mkdir(parents=True)
    (cached / "tool.deb").write_bytes(deb.read_bytes())
    paths.facts_path().write_bytes(b'{"schema":1,"packages":{}}')
    before = paths.facts_path().read_bytes()
    monkeypatch.setattr("pm.packages.subprocess.run", lambda *a, **kw: pytest.fail("bionic executed on host"))
    entry = ensure_mod.stage_only(name, TARGET)
    assert get_package(name).binary(entry, TARGET) == entry / main
    assert (entry / main).read_bytes() == b"bionic fixture"
    assert paths.facts_path().read_bytes() == before
    (entry / main).unlink()
    assert get_package(name).verify(entry, TARGET)


@pytest.mark.platforms("linux", arch="x86_64")
def test_real_node_foreign_stage_checks_bytes_without_exec(tmp_path, monkeypatch):
    import io
    import zipfile

    from pm import paths
    from pm.package import machine_matches_binary
    from pm.registry import get_package
    from pm.store import current_target

    assert current_target() == "linux-x64"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "runtime"))
    store = Store(paths.store_root())
    # Real Node package/unpacker/verifier, with a hash-verified offline archive.
    elf = bytearray(b"\x7fELF" + b"\0" * 60)
    elf[4:7] = b"\x02\x01\x01"  # ELF64, little endian, current ELF version
    elf[18:20] = (0xB7).to_bytes(2, "little")  # AArch64
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as payload:
        payload.writestr("node-v1.0-linux-arm64/bin/node", elf)
    data = archive.getvalue()
    _arm_lock(monkeypatch, [{"url": "https://example.test/node.zip", "sha256": _sha(data)}])
    cached = store.entry(f"fetch-{_sha(data)}")
    cached.mkdir(parents=True)
    (cached / "node.zip").write_bytes(data)

    def refuse_exec(*args, **kwargs):
        pytest.fail(f"cross-target stage attempted execution: {args}")

    monkeypatch.setattr("pm.packages.subprocess.run", refuse_exec)
    entry = ensure_mod.stage_only("node", "linux-arm64")
    node = entry / "bin/node"
    assert node.read_bytes() == elf
    assert machine_matches_binary(node, "linux-arm64") is True
    assert ensure_mod.stage_only("node", "linux-arm64") == entry
    assert not paths.facts_path().exists()
    assert not cached.exists()

    # The no-exec path must still diagnose wrong-architecture and missing bytes.
    elf[18:20] = (0x3E).to_bytes(2, "little")  # x86-64
    node.write_bytes(elf)
    assert "not a linux-arm64 binary" in get_package("node").verify(entry, "linux-arm64")
    node.unlink()
    assert get_package("node").verify(entry, "linux-arm64")


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("compression", ["gz", "xz"])
def test_real_node_native_install_keeps_smoke_validation(tmp_path, sandbox, monkeypatch, compression):
    import io
    import tarfile

    from pm.packages import Nodejs
    from pm.store import current_target

    # An executable fixture makes native verification observable without a Node download.
    probe = tmp_path / "native-probes"
    script = f'#!/bin/sh\nprintf "%s\\n" "$1" >> "{probe}"\nexit 0\n'.encode()
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode=f"w:{compression}") as payload:
        member = tarfile.TarInfo("node-v1.0/bin/node")
        member.mode = 0o755
        member.size = len(script)
        payload.addfile(member, io.BytesIO(script))
    data = archive.getvalue()
    _arm_lock(monkeypatch, [{"url": f"https://example.test/node.tar.{compression}", "sha256": _sha(data)}])
    cached = sandbox.entry(f"fetch-{_sha(data)}")
    cached.mkdir(parents=True)
    (cached / f"node.tar.{compression}").write_bytes(data)
    monkeypatch.setenv("PATH", "")
    package, facts = Nodejs(), ensure_mod._facts()
    entry = ensure_mod._install(package, ensure_mod._lockfile(), facts, sandbox, current_target())
    assert probe.read_text().splitlines() == ["--version", "--version"]
    assert facts.get("node")["entry"] == entry.name
    (entry / "bin/node").write_text("#!/bin/sh\nexit 23\n")
    assert "23" in package.verify(entry, current_target())


@pytest.mark.platforms("windows")
def test_stage_repin_refuses_a_native_directory_hold_then_recovers(sandbox, monkeypatch):
    import ctypes
    from ctypes import wintypes

    original, replacement = b"original", b"replacement"
    _arm_lock(monkeypatch, [{"url": "https://example.test/tool", "sha256": _sha(original)}])
    _seed_fetch_cache(sandbox, original)
    entry = ensure_mod.stage_only("stage-test", TARGET)
    marker = (entry / ".pm-stage-pin.json").read_bytes()
    _arm_lock(monkeypatch, [{"url": "https://example.test/tool", "sha256": _sha(replacement)}])
    _seed_fetch_cache(sandbox, replacement)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    # FILE_FLAG_BACKUP_SEMANTICS opens a directory; omit FILE_SHARE_DELETE.
    handle = kernel.CreateFileW(str(entry), 0, 3, None, 3, 0x02000000, None)
    assert handle != wintypes.HANDLE(-1).value, ctypes.get_last_error()
    try:
        with pytest.raises(InstallError):
            ensure_mod.stage_only("stage-test", TARGET)
        assert (entry / "bin/tool").read_bytes() == original
        assert (entry / ".pm-stage-pin.json").read_bytes() == marker
    finally:
        kernel.CloseHandle(handle)
    ensure_mod.stage_only("stage-test", TARGET)
    assert (entry / "bin/tool").read_bytes() == replacement

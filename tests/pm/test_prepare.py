"""Prepared archives must reproduce the verified PM store entry, not an upstream unpack."""
from __future__ import annotations

import hashlib
import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from pm import paths
from pm.lock import Lockfile
from pm.store import Store, extract, tree_digest


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_staging_source_change_invalidates_prepared_identity(tmp_path):
    from pm.prepare import _source_fingerprints

    stage = tmp_path / "stage.py"
    stage.write_text("def stage(): return 1\n", encoding="utf-8")
    before = _source_fingerprints({"fixture.stage": stage})
    stage.write_text("def stage(): return 2\n", encoding="utf-8")
    assert _source_fingerprints({"fixture.stage": stage}) != before


@pytest.mark.platforms("posix")
def test_real_foreign_stage_roundtrips_posix_archive(tmp_path, monkeypatch):
    """A real locked Node archive, fetched by hash, stages without executing AArch64."""
    from pm.prepare import prepare
    from pm.registry import get_package

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "runtime"))
    lock = Lockfile(tmp_path / "lock.json")
    monkeypatch.setattr(paths, "lockfile_path", lambda: lock.path)
    elf = bytearray(b"\x7fELF" + b"\0" * 60)
    elf[4:7] = b"\x02\x01\x01"
    elf[18:20] = (0xB7).to_bytes(2, "little")
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as payload:
        member = tarfile.TarInfo("node-v1.0-linux-arm64/bin/node")
        member.mode, member.size = 0o755, len(elf)
        payload.addfile(member, io.BytesIO(elf))
        link = tarfile.TarInfo("node-v1.0-linux-arm64/bin/node-alias")
        link.type, link.linkname = tarfile.SYMTYPE, "node"
        payload.addfile(link)
    raw = archive.getvalue()
    lock.set_pin("node", "1.0", {"linux-arm64": {
        "url": "https://example.invalid/node.tar.gz", "sha256": _sha(raw),
    }})
    lock.save()
    cache = Store(paths.store_root()).entry(f"fetch-{_sha(raw)}")
    cache.mkdir(parents=True)
    (cache / "node.tar.gz").write_bytes(raw)
    output = tmp_path / "prepared.tar.gz"
    pin = prepare("node", "linux-arm64", output)
    staged = Store(paths.store_root()).entry(get_package("node").store_entry("1.0", "linux-arm64"))
    extracted = tmp_path / "extracted"
    extract(output, extracted)
    from pm.prepare import source_identity
    assert pin == {"sha256": _sha(output.read_bytes()), "digest": tree_digest(extracted),
                   "source": source_identity(lock, "node", "linux-arm64")}
    assert (extracted / "bin/node").stat().st_mode & 0o111
    assert (extracted / "bin/node-alias").is_symlink()
    assert (staged / ".pm-stage-pin.json").is_file()
    assert not (extracted / ".pm-stage-pin.json").exists()
    assert get_package("node").verify(extracted, "linux-arm64") == ""
    # A fresh stage uses raw pins even when the lock already has a prepared row.
    lock.set_prepared("node", "linux-arm64", {**pin, "sha256": "f" * 64})
    lock.save()
    cache.mkdir(parents=True)
    (cache / "node.tar.gz").write_bytes(raw)
    assert prepare("node", "linux-arm64", output) == pin


def test_multi_archive_package_prepares_one_merged_target_tree(tmp_path, monkeypatch):
    from pm.prepare import prepare
    from pm.registry import get_package

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "runtime"))
    target = "win32-x64"
    lock = Lockfile(tmp_path / "lock.json")
    monkeypatch.setattr(paths, "lockfile_path", lambda: lock.path)
    # A PE header allows the real LlamaCppCuda verifier to inspect the foreign binary.
    exe = bytearray(b"MZ" + b"\0" * 126)
    exe[60:64] = (64).to_bytes(4, "little")
    exe[64:70] = b"PE\0\0" + (0x8664).to_bytes(2, "little")
    inputs = []
    for filename, member, payload in (
        ("engine.zip", "llama-server.exe", bytes(exe)),
        ("runtime.zip", "cudart64.dll", b"CUDA runtime"),
    ):
        raw = io.BytesIO()
        with zipfile.ZipFile(raw, "w") as archive:
            archive.writestr(member, payload)
        data = raw.getvalue()
        digest = _sha(data)
        inputs.append({"url": "https://example.invalid/" + filename, "sha256": digest})
        cache = Store(paths.store_root()).entry(f"fetch-{digest}")
        cache.mkdir(parents=True)
        (cache / filename).write_bytes(data)
    lock.set_pin("llamacpp-cuda", "1", {target: inputs})  # type: ignore[arg-type] -- lock accepts artifact lists
    lock.save()
    output = tmp_path / "prepared.zip"
    pin = prepare("llamacpp-cuda", target, output)
    staged = Store(paths.store_root()).entry(get_package("llamacpp-cuda").store_entry("1", target))
    extracted = tmp_path / "extracted"
    extract(output, extracted)
    assert tree_digest(extracted) == pin["digest"]
    assert (staged / ".pm-stage-pin.json").is_file()
    assert not (extracted / ".pm-stage-pin.json").exists()
    assert (extracted / "llama-server.exe").read_bytes() == bytes(exe)
    assert (extracted / "cudart64.dll").read_bytes() == b"CUDA runtime"
    assert get_package("llamacpp-cuda").verify(extracted, target) == ""
    from pm.prepare import source_identity
    lock.set_pin("llamacpp-cuda", "1", {target: [inputs[0], {**inputs[1], "sha256": "f" * 64}]})  # type: ignore[arg-type]
    assert source_identity(lock, "llamacpp-cuda", target) != pin["source"]


def test_zip_roundtrip_uses_stock_compatible_members(tmp_path):
    from pm.prepare import archive_tree

    entry = tmp_path / "entry"
    (entry / "cmd").mkdir(parents=True)
    (entry / "cmd/git.exe").write_bytes(b"MZ\0test")
    (entry / "empty").mkdir()
    (entry / "__pycache__").mkdir()
    (entry / "__pycache__/module.pyc").write_bytes(b"valid store content")
    output = tmp_path / "git.zip"
    pin = archive_tree(entry, "win32-x64", output)
    with zipfile.ZipFile(output) as zf:
        assert all(zf.getinfo(name).compress_type == zipfile.ZIP_DEFLATED
                   for name in zf.namelist())
        assert "cmd/git.exe" in zf.namelist()
        assert "empty/" in zf.namelist()
        assert "__pycache__/module.pyc" in zf.namelist()
    extracted = tmp_path / "unzipped"
    extract(output, extracted)
    assert pin == {"sha256": _sha(output.read_bytes()), "digest": tree_digest(entry)}
    assert tree_digest(extracted) == pin["digest"]
    assert (extracted / "empty").is_dir()
    assert archive_tree(entry, "win32-x64", output) == pin


@pytest.mark.platforms("posix")
def test_windows_zip_rejects_symlinks_stock_extractor_cannot_restore(tmp_path):
    from pm.prepare import archive_tree

    entry = tmp_path / "entry"
    entry.mkdir()
    (entry / "file").write_bytes(b"x")
    (entry / "link").symlink_to("file")
    output = tmp_path / "prepared.zip"
    with pytest.raises(ValueError, match="symlink"):
        archive_tree(entry, "win32-x64", output)
    assert not output.exists()


def test_source_identity_invalidates_when_dependency_pin_changes(tmp_path):
    from pm.prepare import source_identity

    lock = Lockfile(tmp_path / "pins.json")
    target = "win32-arm64"
    lock.set_pin("python", "1", {target: {"url": "https://example.invalid/python.zip", "sha256": "a" * 64}})
    lock.set_pin("ripgrep", "1", {target: {"url": "https://example.invalid/rg.zip", "sha256": "b" * 64}})
    before = source_identity(lock, "ripgrep", target)
    assert before == source_identity(lock, "ripgrep", target)
    lock.set_pin("python", "1", {target: {"url": "https://example.invalid/python.zip", "sha256": "c" * 64}})
    assert source_identity(lock, "ripgrep", target) != before
    lock.set_pin("python", "1", {target: {"url": "https://example.invalid/python.zip", "sha256": "a" * 64}})
    assert source_identity(lock, "ripgrep", target) == before
    lock.set_pin("ripgrep", "2", {target: {"url": "https://example.invalid/rg.zip", "sha256": "b" * 64}})
    assert source_identity(lock, "ripgrep", target) != before


def test_prepare_refuses_host_bound_operations_before_staging(tmp_path):
    from pm.prepare import preparation_requirements
    from pm.store import current_target

    host = current_target()
    assert preparation_requirements("git", "win32-x64", host="linux-x64")
    assert preparation_requirements("python", "darwin-arm64", host="linux-x64")
    assert preparation_requirements("npm", "win32-x64", host="linux-x64")
    assert preparation_requirements("ripgrep", "win32-arm64", host="linux-x64")
    assert preparation_requirements("node", "win32-x64", host="linux-x64") == ()
    assert preparation_requirements("npm", "linux-arm64-bionic", host="linux-x64") == ()
    assert preparation_requirements("npm", host, host=host) == ("installed node for " + host,)
    assert preparation_requirements("ripgrep", "win32-arm64", host="win32-arm64") == (
        "installed python for win32-arm64",)
    assert preparation_requirements("git", "win32-x64", host="win32-x64") == ()
    assert preparation_requirements("python", "darwin-arm64", host="darwin-arm64") == ()

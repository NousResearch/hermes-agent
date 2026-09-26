"""A Python archive must run before PM selects it as the host interpreter."""
import hashlib
import io
import subprocess
import zipfile
from pathlib import Path

import pytest

from pm import paths
from pm.install import ensure, stage_only
from pm.lock import Facts, Lockfile
from pm.packages import Python
from pm.store import current_target, tree_digest


@pytest.fixture
def python_store(tmp_path, monkeypatch):
    root = tmp_path / "store"
    root.mkdir()
    lock = Lockfile(tmp_path / "lock.json")
    monkeypatch.setattr(paths, "store_root", lambda: root)
    monkeypatch.setattr(paths, "writable_store_root", lambda: root)
    monkeypatch.setattr(paths, "facts_path", lambda: root / "facts.json")
    monkeypatch.setattr(paths, "lockfile_path", lambda: lock.path)
    return root, lock


def cache_archive(root, lock, target, files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    data = stream.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    cached = root / f"fetch-{digest}"
    cached.mkdir()
    (cached / "python.zip").write_bytes(data)
    lock.set_pin("python", "candidate", {target: {
        "url": "https://example.invalid/python.zip", "sha256": digest,
    }})
    lock.save()


def test_unexecutable_python_never_replaces_selected_interpreter(python_store):
    root, lock = python_store
    target = current_target()
    package = Python()
    previous = root / "previous-python"
    old_binary = package.binary(previous, target)
    old_binary.parent.mkdir(parents=True)
    old_binary.write_bytes(b"previous selected interpreter")
    facts = Facts(root / "facts.json")
    facts.record("python", "previous", previous.name, {}, root,
                 target=target, artifacts=["a" * 64], digest=tree_digest(previous))
    before = facts.path.read_bytes()
    relative = package.binary(Path("."), target).as_posix()
    cache_archive(root, lock, target, {relative: b"not executable on this host", "keep.txt": b"keep layout"})

    from pm.package import InstallError
    with pytest.raises(InstallError, match="staged entry failed verification"):
        ensure("python", explicit=True)

    assert facts.path.read_bytes() == before
    assert old_binary.read_bytes() == b"previous selected interpreter"
    assert not (root / package.store_entry("candidate", target)).exists()


def test_cross_target_python_remains_file_only(python_store, monkeypatch):
    root, lock = python_store
    target = "linux-arm64" if current_target() != "linux-arm64" else "win32-x64"
    relative = Python().binary(Path("."), target).as_posix()
    cache_archive(root, lock, target, {relative: b"cross target bytes", "keep.txt": b"keep layout"})
    monkeypatch.setattr("pm.packages.subprocess.run", lambda *a, **k: pytest.fail("cross-target execution"))
    entry = stage_only("python", target)
    assert Python().binary(entry, target).read_bytes() == b"cross target bytes"
    assert not (root / "facts.json").exists()


@pytest.mark.parametrize("outcome, expected", [
    (subprocess.CompletedProcess([], 7, b"", b"missing runtime"), "exited 7: missing runtime"),
    (OSError("loader refused"), "loader refused"),
    (subprocess.TimeoutExpired([], 60), "timed out"),
])
def test_native_python_reports_probe_failure(tmp_path, monkeypatch, outcome, expected):
    package = Python()
    target = current_target()
    binary = package.binary(tmp_path, target)
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"fixture")
    def run(argv, **kwargs):
        assert argv == [str(binary), "--version"]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    monkeypatch.setattr("pm.packages.subprocess.run", run)
    assert expected in package.verify(tmp_path, target)


def test_native_python_success(tmp_path, monkeypatch):
    package = Python()
    target = current_target()
    binary = package.binary(tmp_path, target)
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"fixture")
    seen = []
    def run(argv, **kwargs):
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, b"Python test", b"")
    monkeypatch.setattr("pm.packages.subprocess.run", run)
    assert package.verify(tmp_path, target) == ""
    assert seen == [[str(binary), "--version"]]

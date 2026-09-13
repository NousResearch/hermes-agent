"""Dependency freshness follows PM's interpreter identity, not unrelated tools."""
from pm.lock import Lockfile
from pm.packages import Venv
from pm.store import current_target


def test_venv_stamp_tracks_the_python_pin(tmp_path, monkeypatch):
    from pm import paths

    core = tmp_path / "core"
    core.mkdir()
    (core / "uv.lock").write_bytes(b"unchanged dependency lock")
    lock = Lockfile(tmp_path / "pm-lock.json")
    target = current_target()
    artifact = {"url": "https://example.invalid/python", "sha256": "1" * 64}
    lock.set_pin("python", "test.1", {target: artifact})
    lock.save()
    monkeypatch.setattr(paths, "repo_root", lambda: core)
    monkeypatch.setattr(paths, "lockfile_path", lambda: lock.path)
    venv = Venv()
    original = venv.expected_stamp([], plugin_dirs=[])

    lock.set_pin("uv", "unrelated", {target: artifact})
    lock.save()
    assert venv.expected_stamp([], plugin_dirs=[]) == original

    # Repacked interpreter bytes are a new input even at the same version.
    artifact = {**artifact, "sha256": "2" * 64}
    lock.set_pin("python", "test.1", {target: artifact})
    lock.save()
    repinned = venv.expected_stamp([], plugin_dirs=[])
    assert repinned != original

    lock.set_pin("python", "test.2", {target: artifact})
    lock.save()
    assert venv.expected_stamp([], plugin_dirs=[]) != repinned

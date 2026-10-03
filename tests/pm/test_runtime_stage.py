"""The staged runtime installs the committed lock, never re-resolves it."""
from pathlib import Path

import pm.environment
import pm.runtime_stage
from pm.runtime_stage import stage_runtime


def test_staged_snapshot_syncs_frozen_not_locked(tmp_path, monkeypatch):
    """#125323: --locked's up-to-date assertion is environment-brittle in the
    staged copy of the workspace; the snapshot must install via --frozen."""
    captured = {}

    class FakeEnvironment:
        def __init__(self, *, uv, python, destination, cache, env,
                     offline, output, no_config):
            self.destination = destination

        @property
        def executable(self):
            return Path("/python")

        def create(self):
            captured.setdefault("create", 0)
            captured["create"] += 1

        def sync(self, source, **kwargs):
            captured["sync"] = kwargs
            captured["sync_source_files"] = sorted(
                p.name for p in Path(source).iterdir())

        def install_wheelhouse(self, source, wheelhouse, *, timeout):
            captured["wheelhouse"] = True

    import subprocess as _subprocess

    class Ok:
        returncode = 0
        stderr = ""

    monkeypatch.setattr(pm.environment, "PythonEnvironment", FakeEnvironment)
    monkeypatch.setattr(pm.runtime_stage.subprocess, "run", lambda *a, **k: Ok())
    monkeypatch.setattr(pm.runtime_stage.tempfile, "TemporaryDirectory",
                        lambda *a, **k: _fake_dir(tmp_path))

    stage_runtime(Path("/uv"), Path("/python"), tmp_path / "dest")

    assert captured["create"] == 1
    # The engine default is frozen=True; passing locked=True (or frozen=False)
    # re-resolves or re-asserts freshness in a directory PM copied the lock
    # into, which is exactly the Windows python-deps abort of #125323.
    assert captured["sync"].get("locked") is not True
    assert captured["sync"].get("frozen", True) is True
    # Only the committed pair ever enters the snapshot uv runs against.
    assert captured["sync_source_files"] == ["pyproject.toml", "uv.lock"]


class _fake_dir:
    def __init__(self, path):
        self._path = Path(path) / "pm-project-snap"
        self._path.mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        return self._path

    def __exit__(self, *exc):
        return False

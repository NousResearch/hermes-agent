"""The PM runtime stages from the lock; ambient index settings stay transport.

`uv sync --locked` re-resolves against the ambient index to assert lock
freshness, so a bridged mirror (pm/index_config) — any URL differing from the
lock's recorded registry, even a trailing slash — fails the bootstrap (#125321).
The staged sync must use the frozen mode instead: install the exact locked,
hash-verified set with no freshness assertion.
"""
from __future__ import annotations

from pathlib import Path
import subprocess


def test_stage_runtime_syncs_frozen_not_locked(tmp_path, monkeypatch):
    import pm.environment
    import pm.packages
    import pm.runtime
    from pm.runtime_stage import stage_runtime

    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nname = "hermes-pm-runtime"\nversion = "0.0.0"\n', encoding="utf-8")
    (project / "uv.lock").write_text("version = 1\nrevision = 3\n", encoding="utf-8")

    captured: dict = {}

    class _FakeEnvironment:
        executable = tmp_path / "python"

        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def create(self):
            captured["created"] = True

        def sync(self, source, **kwargs):
            captured["sync"] = kwargs

    monkeypatch.setattr(pm.environment, "PythonEnvironment", _FakeEnvironment)
    monkeypatch.setattr(pm.packages, "uv_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(pm.runtime, "runtime_environment", lambda: {"HERMES_HOME": str(tmp_path)})
    monkeypatch.setattr(pm.runtime_stage.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0))

    stage_runtime(tmp_path / "uv", tmp_path / "python", tmp_path / "dest", project=project)

    assert captured.get("created") is True
    # Frozen (the sync default), never `--locked`: mirrors are transport only.
    assert captured["sync"]["locked"] is False

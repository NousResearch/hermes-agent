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


def test_stage_runtime_emits_frozen_in_real_uv_argv(tmp_path, monkeypatch):
    """Argv-level pin: the staging sync's real uv command carries --frozen.

    The kwargs assertion above cannot fail if the ``locked=False`` argument is
    dropped from ``pm/runtime_stage.py`` (False is ``PythonEnvironment.sync``'s
    declared default) — the fake in that test also never runs the flag-picking
    body at ``pm/environment.py``. Here a stub ``uv`` binary records its real
    argv through the un-stubbed ``PythonEnvironment`` stack, so both mutations
    the guard exists for (kwarg dropped → still fine is NOT acceptable;
    ``--locked`` emitted unconditionally) fail this test.
    """
    import pm.packages
    import pm.runtime
    import pm.runtime_stage
    from pm.runtime_stage import stage_runtime

    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nname = "hermes-pm-runtime"\nversion = "0.0.0"\n', encoding="utf-8")
    (project / "uv.lock").write_text("version = 1\nrevision = 3\n", encoding="utf-8")

    record = tmp_path / "uv-argv.log"
    uv_stub = tmp_path / "uv"
    uv_stub.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        f"with open({str(record)!r}, 'a') as fh:\n"
        "    fh.write(' '.join(sys.argv[1:]) + '\\n')\n"
        "sys.exit(0)\n",
        encoding="utf-8")
    uv_stub.chmod(0o755)

    monkeypatch.setattr(pm.packages, "uv_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(pm.runtime, "runtime_environment", lambda: {"HERMES_HOME": str(tmp_path)})
    # Only the post-install dependency validation is stubbed (it wants a real
    # venv python); create/sync run the real PythonEnvironment against the uv stub.
    monkeypatch.setattr(pm.runtime_stage.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0))

    stage_runtime(uv_stub, tmp_path / "python", tmp_path / "dest", project=project)

    lines = record.read_text().splitlines()
    syncs = [line for line in lines if line.split()[0] == "sync"]
    assert len(syncs) == 1, f"expected exactly one uv sync, got: {lines}"
    staging = syncs[0]
    assert "--frozen" in staging.split(), staging
    assert "--locked" not in staging.split(), staging
    # Staging-only markers (the app sync does not pass these): pin them so a
    # future refactor cannot silently point this assertion at the other sync.
    assert "--no-default-groups" in staging.split(), staging
    assert "--no-install-project" in staging.split(), staging

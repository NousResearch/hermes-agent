"""pm.workspace: the generated uv-workspace root for plugin deps.

The workspace root is a pm-GENERATED project (never the committed
pyproject.toml — sealed installs are read-only and member lists are
machine-specific). Its pyproject = core's pyproject verbatim +
``[tool.uv.workspace] members`` pointing at each snapshotted plugin.
``uv lock`` unions core + plugin deps into ONE lock; conflict = loud refusal.
"""

from __future__ import annotations

import os
import subprocess
import shutil
import sys
from pathlib import Path

import pytest

import pm.workspace as ws
from pm.environment import managed_environment
from tests.pm.test_environment_build import locked_project  # noqa: F401


@pytest.fixture(autouse=True)
def isolated_machine_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))


@pytest.fixture
def layout(locked_project, tmp_path, monkeypatch):
    core, uv, _ = locked_project
    manifest = core / "pyproject.toml"
    manifest.write_text(manifest.read_text().replace('[tool.uv.workspace]\nmembers=["member"]\n', ""))
    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (uv, Path(sys.executable)))
    monkeypatch.setattr(ws.paths, "repo_root", lambda: core)
    return tmp_path, core, core / "member", tmp_path / "store"




def test_preparation_refuses_existing_workspace_without_mutating_it(layout):
    from pm.package import InstallError

    tmp, core, plug_a, _ = layout
    root = tmp / "workspace"
    environment = managed_environment(tmp / "env")
    kwargs = dict(root=root, source=core, seed_lock=None,
                  environment=environment)
    ws.lock_and_sync([plug_a], [], **kwargs)
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    (plug_a / "pyproject.toml").write_text('changed after publication')
    with pytest.raises(InstallError, match="fresh"):
        ws.lock_and_sync([], [], **kwargs)
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()} == before


def test_missing_explicit_seed_cannot_silently_resolve_new_versions(layout):
    tmp, core, plug_a, _ = layout
    with pytest.raises(FileNotFoundError):
        ws.lock_and_sync([plug_a], [], root=tmp / "workspace", source=core,
                         seed_lock=tmp / "missing.lock", environment=managed_environment(tmp / "env"))
    assert not (tmp / "env").exists()









# --- classified failures + staging surface (FINAL-RUNTIME-CONTRACT) ---


def test_classify_network_failure_stays_generic():
    from pm.workspace import ResolutionConflict, classify_uv_failure

    err = classify_uv_failure("lock", 1, "error: Failed to fetch https://pypi.org (timed out)")
    assert not isinstance(err, ResolutionConflict)


def test_sync_failure_is_never_a_conflict(layout, monkeypatch):
    from pm.package import InstallError

    tmp, core, _, _ = layout
    environment = managed_environment(tmp / "candidate")
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kwargs:
                        subprocess.CompletedProcess(cmd, 1, "", "Failed to download wheel"))
    with pytest.raises(InstallError) as excinfo:
        ws.lock_and_sync([], [], root=tmp / "workspace", source=core,
                         seed_lock=None, environment=environment, frozen=True)
    assert not isinstance(excinfo.value, ws.ResolutionConflict)


def test_staging_root_and_env_are_honored_without_live_mutation(layout, monkeypatch):
    tmp, core, _, _ = layout
    staging = tmp / "staging-ws"
    monkeypatch.setenv("PM_WORKSPACE_TEST_SENTINEL", "live")
    environment = managed_environment(tmp / "staging-venv", env={
        "PATH": "/staged/bin", "PM_WORKSPACE_TEST_SENTINEL": "staged",
    })
    # The prepared environment is authoritative; workspace never discovers tools.
    monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: pytest.fail("PATH discovery"))
    seen = []
    def run(cmd, **kwargs):
        seen.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(subprocess, "run", run)
    ws.lock_and_sync([], [], root=staging, source=core, seed_lock=None, environment=environment)
    assert [cmd[1] for cmd, _ in seen] == ["lock", "sync"]
    for cmd, kwargs in seen:
        assert Path(cmd[0]) == environment.uv
        assert Path(kwargs["cwd"]) == staging
        assert kwargs["env"]["PM_WORKSPACE_TEST_SENTINEL"] == "staged"
        assert kwargs["env"]["UV_CACHE_DIR"] == str(environment.cache)
        assert kwargs["env"]["UV_PROJECT_ENVIRONMENT"] == str(environment.destination)
        assert kwargs["env"]["UV_PYTHON"] == str(environment.python)
    assert os.environ["PM_WORKSPACE_TEST_SENTINEL"] == "live"

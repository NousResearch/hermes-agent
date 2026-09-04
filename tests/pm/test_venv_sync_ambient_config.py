"""Desired behavior: ambient user uv configuration must not affect pm's
venv sync, while the project's own [tool.uv] settings stay honored.

This is the contract the installers' old run_locked_uv_sync() used to
guarantee (#82446, #21269, #83914). Now that install.sh / install.ps1
delegate to ``python -m pm.cli install``, the contract lives in pm's
venv-sync env. We assert the behavior (a locked project sync succeeds
under pm's env even when the ambient environment is poisoned), not the
names of helper functions.

The negative control is the load-bearing half: the same poisoned ambient
env fed to uv RAW must break the sync — otherwise the test could pass for
the wrong reason and never catch a regression.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


def _write_project(root: Path) -> None:
    """A project whose [tool.uv] the lock depends on: hide it and uv
    rejects the lock (the #82446 shape)."""
    (root / "pyproject.toml").write_text(
        """[project]
name = "venv-sync-regression"
version = "0.1.0"
requires-python = ">=3.11"

[project.optional-dependencies]
all = []

[tool.uv]
package = false
exclude-newer = "14 days"
""",
        encoding="utf-8",
    )


@pytest.fixture
def real_uv() -> str:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is unavailable")
    return uv


@pytest.fixture
def locked_project(real_uv, tmp_path: Path) -> Path:
    """A temp project with a lock generated in a clean ambient env."""
    project = tmp_path / "project"
    project.mkdir()
    _write_project(project)
    clean = os.environ.copy()
    for key in ("UV_NO_CONFIG", "UV_CONFIG_FILE", "UV_DEFAULT_INDEX", "XDG_CONFIG_HOME", "XDG_CONFIG_DIRS"):
        clean.pop(key, None)
    clean["UV_OFFLINE"] = "1"
    locked = subprocess.run(
        [real_uv, "lock"], cwd=project, env=clean,
        capture_output=True, text=True, check=False,
    )
    assert locked.returncode == 0, locked.stderr
    return project


def _poisoned_env() -> dict:
    env = os.environ.copy()
    env["UV_NO_CONFIG"] = "1"
    env["UV_CONFIG_FILE"] = "/poison/uv.toml"
    # A hostile index: resolution against it can only fail, so if it ever
    # reaches uv the sync is observably broken.
    env["UV_DEFAULT_INDEX"] = "https://poison.invalid/simple"
    env["XDG_CONFIG_HOME"] = "/poison/user"
    env["XDG_CONFIG_DIRS"] = "/poison/system"
    return env


def test_ambient_uv_config_does_not_affect_pm_venv_sync(real_uv, locked_project, monkeypatch):
    """Poison the ambient env, then sync through pm's venv-sync env. The
    sync must succeed with the project's [tool.uv] honored — identical to
    the clean-env outcome."""
    poisoned = _poisoned_env()
    monkeypatch.setenv("UV_NO_CONFIG", "1")
    monkeypatch.setenv("UV_CONFIG_FILE", "/poison/uv.toml")
    monkeypatch.setenv("UV_DEFAULT_INDEX", "https://poison.invalid/simple")

    # Negative control: uv with the RAW poisoned ambient env rejects the
    # lock (project config hidden by UV_NO_CONFIG). Proves the poison is a
    # real threat the test would catch.
    raw = subprocess.run(
        [real_uv, "lock", "--check"], cwd=locked_project,
        env=poisoned, capture_output=True, text=True, check=False,
    )
    assert raw.returncode != 0, "negative control: raw poisoned env must fail"

    # Behavior: pm's venv-sync env makes the ambient config a no-op.
    from pm.packages import uv_env

    env = uv_env(base_env=poisoned)
    # The venv-sync shape restores project-config visibility (pm_uv with a
    # venv pops UV_NO_CONFIG); XDG config discovery is isolated to an
    # empty dir so ambient uv.toml can't steer either.
    env.pop("UV_NO_CONFIG", None)
    assert env.get("XDG_CONFIG_HOME") not in ("/poison/user", None)

    with_pm = subprocess.run(
        [real_uv, "lock", "--check"], cwd=locked_project,
        env=env, capture_output=True, text=True, check=False,
    )
    assert with_pm.returncode == 0, with_pm.stderr


def test_uv_env_strips_ambient_uv_overrides(monkeypatch):
    """The env pm hands uv carries none of the ambient UV_* overrides."""
    monkeypatch.setenv("UV_NO_CONFIG", "1")
    monkeypatch.setenv("UV_CONFIG_FILE", "/poison/uv.toml")
    monkeypatch.setenv("UV_DEFAULT_INDEX", "https://poison.invalid/simple")
    monkeypatch.setenv("XDG_CONFIG_HOME", "/poison/user")
    monkeypatch.setenv("XDG_CONFIG_DIRS", "/poison/system")

    from pm.packages import uv_env

    env = uv_env()
    assert "UV_NO_CONFIG" in env  # hermetic default for non-project calls
    assert "UV_CONFIG_FILE" not in env
    assert "UV_DEFAULT_INDEX" not in env
    # Config discovery is redirected off the ambient dirs.
    assert env.get("XDG_CONFIG_HOME") not in ("/poison/user", None)
    assert env.get("XDG_CONFIG_DIRS") not in ("/poison/system", None)

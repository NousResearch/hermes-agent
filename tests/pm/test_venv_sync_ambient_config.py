"""Ambient uv settings cannot steer PM; project configuration still can."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture(autouse=True)
def isolated_machine_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))


def test_ambient_uv_config_does_not_affect_pm_venv_sync(tmp_path, monkeypatch):
    from pm.environment import managed_environment

    uv = shutil.which("uv")
    assert uv, "the isolation contract requires real uv"
    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (Path(uv), Path(sys.executable)))
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nname="venv-sync-regression"\nversion="1"\nrequires-python=">=3.11"\n'
        '[tool.uv]\npackage=false\nexclude-newer="14 days"\n', encoding="utf-8",
    )
    environment = managed_environment(tmp_path / "candidate", offline=True)
    environment.lock(project)
    locked = (project / "uv.lock").read_bytes()
    config = tmp_path / "config"
    (config / "uv").mkdir(parents=True)
    (config / "uv" / "uv.toml").write_text('required-version="<0.0.1"\n')
    for key, value in {
        "UV_NO_CONFIG": "1", "UV_CONFIG_FILE": "/poison/uv.toml",
        "UV_DEFAULT_INDEX": "https://poison.invalid/simple", "UV_PYTHON": "/poison/python",
        "UV_PROJECT_ENVIRONMENT": str(tmp_path / "unrelated-environment"),
        "UV_PROJECT": "/poison/project", "VIRTUAL_ENV": "/poison/venv",
        "PYTHONPATH": "/poison/imports", "XDG_CONFIG_HOME": str(config), "XDG_CONFIG_DIRS": str(config),
    }.items():
        monkeypatch.setenv(key, value)
    before = dict(os.environ)
    raw = subprocess.run([uv, "lock", "--check"], cwd=project,
                         capture_output=True, text=True, timeout=30)
    assert raw.returncode != 0, "negative control: raw poisoned environment must fail"
    environment = managed_environment(tmp_path / "candidate", offline=True)
    environment.create()
    environment.sync(project, locked=True)
    environment.check()
    assert environment.executable.is_file()
    assert not (tmp_path / "unrelated-environment").exists()
    assert (project / "uv.lock").read_bytes() == locked
    assert dict(os.environ) == before

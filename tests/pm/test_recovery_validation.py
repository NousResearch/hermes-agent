"""PM validates real imports before it publishes a recovered dependency tree."""
from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import shutil
import sys

import pytest

from pm.package import InstallError
from pm.runtime import runtime_environment
from pm.recovery import validate_environment

@pytest.fixture(autouse=True)
def isolated_machine_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))



@pytest.mark.parametrize("damage", ["module", "distribution"])
def test_validation_rejects_a_missing_required_import(tmp_path, monkeypatch, damage):
    import pm.paths as paths
    from hermes_cli.runtime_paths import site_packages

    core = tmp_path / "core"
    core.mkdir()
    version = importlib.metadata.version("python-dotenv")
    (core / "pyproject.toml").write_text(
        '[project]\nname="validation-proof"\nversion="1"\nrequires-python=">=3.11"\n'
        f'dependencies=["python-dotenv=={version}"]\n[tool.uv]\npackage=false\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(paths, "repo_root", lambda: core)
    uv = shutil.which("uv")
    assert uv, "validation integration requires real uv"
    env = {**runtime_environment(), "UV_PYTHON": sys.executable}
    env.pop("UV_NO_CONFIG", None)
    workspace = tmp_path / "workspace"
    candidate = tmp_path / "venv"
    monkeypatch.setattr("pm._uv._toolchain", lambda **kwargs: (Path(uv), Path(sys.executable)))
    from pm.workspace import lock_and_sync

    lock_and_sync([], [], root=workspace, venv_dir=candidate)
    python = candidate / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    validate_environment(python, env=env, cwd=workspace)
    shutil.rmtree(site_packages(candidate) / "dotenv")
    if damage == "distribution":
        for metadata in site_packages(candidate).glob("python_dotenv-*.dist-info"):
            shutil.rmtree(metadata)
    with pytest.raises(InstallError, match="startup validation failed"):
        validate_environment(python, env=env, cwd=workspace)

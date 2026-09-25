"""Regression for #122183: a legacy venv must not shadow PM's selected wheels."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gateway import run as gateway_run


@pytest.mark.platforms("windows")
def test_pm_gateway_keeps_committed_imports_ahead_of_legacy_venv(tmp_path, monkeypatch):
    import pydantic_core
    from pm.environments import committed_venv, install_state_dir, runtime_facts_path

    project = Path(gateway_run.__file__).resolve().parent.parent
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    generation = install_state_dir(project) / "environments" / "selected"
    (generation / "pyvenv.cfg").parent.mkdir(parents=True)
    (generation / "pyvenv.cfg").write_text("version = 3.11\n", encoding="utf-8")
    selected = generation / "Lib" / "site-packages"
    selected.mkdir(parents=True)

    facts = runtime_facts_path(project)
    facts.parent.mkdir(parents=True, exist_ok=True)
    facts.write_text(json.dumps({"packages": {"venv": {"environment": str(generation)}}}), encoding="utf-8")
    assert committed_venv(project) == generation

    legacy = tmp_path / "legacy"
    legacy_package = legacy / "Lib" / "site-packages" / "pydantic_core"
    legacy_package.mkdir(parents=True)
    (legacy_package / "__init__.py").write_text(
        "from . import _pydantic_core\n", encoding="utf-8"
    )
    monkeypatch.setenv("VIRTUAL_ENV", str(legacy))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(project), str(selected))))
    monkeypatch.setattr(gateway_run.sys, "path", [str(project), str(selected), *sys.path])
    before = list(sys.path)
    gateway_run._ensure_windows_gateway_venv_imports()
    assert sys.path == before
    assert os.environ["PYTHONPATH"] == os.pathsep.join((str(project), str(selected)))
    assert str(legacy / "Lib" / "site-packages") not in sys.path

    code = """
import shutil
import sys
from pathlib import Path
selected, legacy, installed_package = map(Path, sys.argv[1:])
shutil.copytree(installed_package, selected / 'pydantic_core')
assert selected in map(Path, sys.path)
import pydantic_core._pydantic_core as native
assert Path(native.__file__).is_relative_to(selected), native.__file__
assert str(legacy / 'Lib' / 'site-packages') not in sys.path
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(selected), str(legacy),
         str(Path(pydantic_core.__file__).parent)],
        env=os.environ.copy(), cwd=project, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.platforms("windows")
def test_unmanaged_gateway_still_activates_existing_venv(tmp_path, monkeypatch):
    venv = tmp_path / "venv"
    site_packages = venv / "Lib" / "site-packages"
    site_packages.mkdir(parents=True)
    monkeypatch.setattr("hermes_cli._launchers.resolve_store_python", lambda root: None)
    monkeypatch.setenv("VIRTUAL_ENV", str(venv))
    monkeypatch.setattr(gateway_run.sys, "path", [str(tmp_path)])

    gateway_run._ensure_windows_gateway_venv_imports()

    assert str(site_packages) in sys.path
    assert os.environ["VIRTUAL_ENV"] == str(venv.resolve())

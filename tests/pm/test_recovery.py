"""PM repair replays the selected dependency graph without broken app imports."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from pm.lock import Facts
from pm.packages import uv_env
from tests.pm.test_workspace_build_inputs import _wheel


@pytest.mark.parametrize("failure", [None, "missing_distribution", "broken_module"])
def test_startup_validation_checks_real_ruamel_dependency(tmp_path, failure):
    from importlib.metadata import distribution
    import venv

    import ruamel.yaml

    from hermes_cli.runtime_paths import site_packages
    from pm.package import InstallError
    from pm.recovery import validate_environment

    candidate = tmp_path / "candidate"
    venv.EnvBuilder(with_pip=False).create(candidate)
    python = candidate / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    target = site_packages(candidate)
    package = target / "ruamel" / "yaml"
    # Copy the real parser and distribution metadata into an isolated candidate.
    # A fake YAML class would not detect a broken install or missing dependency.
    ignored = ["__pycache__"] + (["main.py"] if failure == "broken_module" else [])
    shutil.copytree(Path(ruamel.yaml.__file__).parent, package, ignore=shutil.ignore_patterns(*ignored))
    installed = distribution("ruamel.yaml")
    assert installed.files
    metadata = next(Path(installed.locate_file(path)).parent for path in installed.files if path.name == "METADATA")
    if failure != "missing_distribution":
        shutil.copytree(metadata, target / metadata.name)

    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="yaml-recovery"\ndependencies=["ruamel.yaml"]\n', encoding="utf-8",
    )

    if failure:
        with pytest.raises(InstallError, match="ruamel"):
            validate_environment(python, env=dict(os.environ), cwd=tmp_path)
    else:
        validate_environment(python, env=dict(os.environ), cwd=tmp_path)


@pytest.mark.parametrize("failure", [None, "missing_lock", "corrupt_facts", "empty_environment", "missing_extras", "validation", "publication"])
def test_repair_restores_recorded_plugin_dependencies_without_config(tmp_path, monkeypatch, failure):
    import pm.paths as paths
    import pm.workspace as workspace
    from hermes_cli.runtime_paths import selected_venv, site_packages

    engine = importlib.import_module("pm.ensure")
    uv = shutil.which("uv")
    assert uv, "recovery integration requires real uv"
    core = tmp_path / "core"
    core.mkdir()
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    _wheel(wheels, "core_dep", "1.0")
    _wheel(wheels, "plugin_dep", "1.0")
    (core / "pyproject.toml").write_text(
        '[project]\nname="repair-core"\nversion="1"\nrequires-python=">=3.11"\n'
        'dependencies=["core-dep==1.0"]\n[project.optional-dependencies]\nall=[]\n'
        '[tool.uv]\npackage=false\nno-index=true\n'
        f'find-links=[{json.dumps(wheels.as_posix())}]\n', encoding="utf-8",
    )
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    (plugin / "pyproject.toml").write_text(
        '[project]\nname="repair-plugin"\nversion="1"\nrequires-python=">=3.11"\n'
        'dependencies=["plugin-dep==1.0"]\n[tool.uv]\npackage=false\n', encoding="utf-8",
    )
    monkeypatch.setattr(paths, "repo_root", lambda: core)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(engine, "uv", lambda **kwargs: (
        uv, {**uv_env(kwargs.get("base_env")), "UV_PYTHON": sys.executable, "UV_OFFLINE": "1"},
    ))
    monkeypatch.setattr(engine, "lazy_installs_allowed", lambda: True)
    monkeypatch.setattr(workspace, "enabled_member_dirs", lambda: [plugin])
    # The committed core lock is independent of the plugin union.
    env = {**uv_env(), "UV_PYTHON": sys.executable, "UV_OFFLINE": "1"}
    env.pop("UV_NO_CONFIG")
    subprocess.run([uv, "lock"], cwd=core, env=env, capture_output=True, check=True, timeout=60)
    engine.sync_venv([], explicit=True)
    old = selected_venv(core)
    old_fact = Facts(paths.runtime_facts_path()).get("venv")
    old_lock = Path(old_fact["resolved_lock"]).read_bytes()
    config = tmp_path / "home" / "config.yaml"
    config.write_bytes(b"plugins:\n  enabled: [repair-plugin]\n")
    config_before = config.read_bytes()
    shutil.rmtree(site_packages(old) / "core_dep")
    shutil.rmtree(site_packages(old) / "plugin_dep")

    def broken_config(*args, **kwargs):
        raise AssertionError("repair must use the recorded graph, not parse config")
    monkeypatch.setattr(engine, "lazy_installs_allowed", broken_config)
    monkeypatch.setattr(workspace, "enabled_member_dirs", broken_config)
    if failure:
        from pm import recovery
        from pm.package import InstallError

        if failure == "corrupt_facts":
            paths.runtime_facts_path().write_text("invalid recorded state", encoding="utf-8")
        elif failure in {"empty_environment", "missing_extras"}:
            data = json.loads(paths.runtime_facts_path().read_text(encoding="utf-8"))
            recorded = data["packages"]["venv"]
            if failure == "empty_environment":
                recorded["environment"] = ""
            else:
                recorded.pop("extras")
            paths.runtime_facts_path().write_text(json.dumps(data), encoding="utf-8")
        old_facts = paths.runtime_facts_path().read_bytes()
        def fail(*args, **kwargs):
            raise InstallError("venv", "injected validation or publication failure")
        if failure == "missing_lock":
            Path(old_fact["resolved_lock"]).unlink()
        elif failure == "validation":
            monkeypatch.setattr(recovery, "validate_environment", fail)
        elif failure == "publication":
            monkeypatch.setattr(Facts, "record_state", fail)
        with pytest.raises((InstallError, ValueError)):
            engine.sync_venv(repair=True)
        assert paths.runtime_facts_path().read_bytes() == old_facts
        if failure not in {"corrupt_facts", "empty_environment"}:
            assert selected_venv(core) == old
        assert config.read_bytes() == config_before
        assert old.is_dir()
        return

    engine.sync_venv(repair=True)
    restored = selected_venv(core)
    assert restored != old
    python = restored / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    result = subprocess.run(
        [str(python), "-I", "-c", "import core_dep, plugin_dep; print(core_dep.__version__, plugin_dep.__version__)"],
        cwd=tmp_path, capture_output=True, text=True, check=True, timeout=30,
    )
    assert result.stdout.strip() == "1.0 1.0"
    new_fact = Facts(paths.runtime_facts_path()).get("venv")
    assert new_fact["stamp"] == old_fact["stamp"]
    assert Path(new_fact["resolved_lock"]).read_bytes() == old_lock
    assert old.is_dir() and not (site_packages(old) / "plugin_dep").exists()
    assert config.read_bytes() == config_before


def test_uncertain_profile_selection_refuses_sync_but_not_recorded_repair(tmp_path, monkeypatch):
    import pm.paths as paths
    from hermes_cli.plugins_admission import AdmissionRefused, admit_plugin_set_change
    from hermes_cli.runtime_paths import install_state_dir, selected_venv, site_packages

    engine = importlib.import_module("pm.ensure")
    # Use the same engine for admission and repair with the offline uv fixture.
    monkeypatch.setattr("pm.client.sync_venv", engine.sync_venv)
    uv = shutil.which("uv")
    assert uv
    core = tmp_path / "core"
    core.mkdir()
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    _wheel(wheels, "core_dep", "1.0")
    _wheel(wheels, "plugin_dep", "1.0")
    (core / "pyproject.toml").write_text(
        '[project]\nname="uncertain-core"\nversion="1"\nrequires-python=">=3.11"\n'
        'dependencies=["core-dep==1.0"]\n[tool.uv]\npackage=false\nno-index=true\n'
        f'find-links=[{json.dumps(wheels.as_posix())}]\n', encoding="utf-8",
    )
    home = tmp_path / "home"
    sibling = home / "profiles" / "worker"
    plugin = sibling / "plugins" / "worker-deps"
    plugin.mkdir(parents=True)
    (plugin / "pyproject.toml").write_text(
        '[project]\nname="worker-deps"\nversion="1"\nrequires-python=">=3.11"\n'
        'dependencies=["plugin-dep==1.0"]\n[tool.uv]\npackage=false\n', encoding="utf-8",
    )
    active_config = home / "config.yaml"
    active_config.write_text("plugins:\n  enabled: []\n", encoding="utf-8")
    sibling_config = sibling / "config.yaml"
    sibling_config.write_text("plugins:\n  enabled: [worker-deps]\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(paths, "repo_root", lambda: core)
    monkeypatch.setattr(engine, "uv", lambda **kwargs: (
        uv, {**uv_env(kwargs.get("base_env")), "UV_PYTHON": sys.executable, "UV_OFFLINE": "1"},
    ))
    monkeypatch.setattr(engine, "lazy_installs_allowed", lambda: True)
    env = {**uv_env(), "UV_PYTHON": sys.executable, "UV_OFFLINE": "1"}
    env.pop("UV_NO_CONFIG")
    subprocess.run([uv, "lock"], cwd=core, env=env, check=True, capture_output=True, timeout=60)
    engine.sync_venv([], explicit=True)
    old = selected_venv(core)
    before_facts = paths.runtime_facts_path().read_bytes()
    before_config = active_config.read_bytes()
    before_generations = set((install_state_dir(core) / "environments").iterdir())
    old_lock = Path(Facts(paths.runtime_facts_path()).get("venv")["resolved_lock"]).read_bytes()
    sibling_config.write_text("plugins:\n  enabled: [unclosed\n", encoding="utf-8")
    damaged_config = sibling_config.read_bytes()

    with pytest.raises(ValueError, match="config.yaml"):
        engine.sync_venv(explicit=True)
    with pytest.raises(AdmissionRefused, match="config.yaml"):
        admit_plugin_set_change(set(), set(), active_plugins_dir=home / "plugins")
    assert paths.runtime_facts_path().read_bytes() == before_facts
    assert active_config.read_bytes() == before_config
    assert selected_venv(core) == old
    assert set((install_state_dir(core) / "environments").iterdir()) == before_generations

    shutil.rmtree(site_packages(old) / "plugin_dep")
    engine.sync_venv(repair=True)
    repaired = selected_venv(core)
    assert repaired != old
    python = repaired / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    result = subprocess.run(
        [str(python), "-I", "-c", "import plugin_dep; print(plugin_dep.__version__)"],
        cwd=tmp_path, capture_output=True, text=True, check=True, timeout=30,
    )
    assert result.stdout.strip() == "1.0"
    assert Path(Facts(paths.runtime_facts_path()).get("venv")["resolved_lock"]).read_bytes() == old_lock
    assert active_config.read_bytes() == before_config
    assert sibling_config.read_bytes() == damaged_config

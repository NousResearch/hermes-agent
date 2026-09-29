"""Enabling a plugin whose dependencies only a newer PM generation carries asks for a restart.

A running backend keeps the dependency generation it booted on. When enabling a plugin publishes a
new generation with the plugin's Python dependencies, loading the plugin in this process would run
its ``register`` without them, so the enable must report ``restart_required`` and load nothing.
Real temp HERMES_HOME, real PM generation records under the temp installs root.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import plugins_cmd as cmd
from hermes_cli.config import load_config, save_config

DEP = "restart_fixture_only_dep"

INIT = f'''from pathlib import Path

import {DEP}


def register(ctx):
    (Path(__file__).parent / "registered").write_text({DEP}.VALUE)
'''


@pytest.fixture
def world(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    (home / "plugins").mkdir(parents=True)
    (home / "config.yaml").write_text("plugins:\n  enabled: []\n  disabled: []\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    reloads = []

    def reload_gateway_plugins(*args, **kwargs):
        reloads.append(args)
        return {"reloaded": True, "activations": []}

    monkeypatch.setattr("gateway.control_socket.reload_gateway_plugins", reload_gateway_plugins)
    return home, reloads


def _plugin(home, *, deps=True):
    plugin = home / "plugins" / "dep-fixture"
    plugin.mkdir()
    requirement = f"python_dependencies:\n  - {DEP.replace('_', '-')}\n" if deps else ""
    (plugin / "plugin.yaml").write_text(f"name: dep-fixture\nversion: '1.0'\n{requirement}", encoding="utf-8")
    (plugin / "__init__.py").write_text(INIT if deps else INIT.replace(f"import {DEP}\n", "").replace(
        f"{DEP}.VALUE", "'loaded'"), encoding="utf-8")
    return plugin


def _generation(name, *, with_dep):
    """A real venv under this checkout's PM generations dir (temp installs root)."""
    from pm.environments import install_state_dir, site_packages
    from pm.paths import repo_root

    venv = install_state_dir(repo_root()) / "environments" / name / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True, timeout=120)
    packages = site_packages(venv)
    packages.mkdir(parents=True, exist_ok=True)
    if with_dep:
        (packages / f"{DEP}.py").write_text("VALUE = 'from the generation'\n")
        info = packages / f"{DEP}-1.0.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {DEP}\nVersion: 1.0\n")
    return venv


def _commit(venv):
    from pm.environments import runtime_facts_path
    from pm.paths import repo_root

    runtime_facts_path(repo_root()).write_text(json.dumps({"packages": {"venv": {"environment": str(venv)}}}))


def _boot(monkeypatch, venv):
    """What boot activation does: the committed generation's site-packages on sys.path."""
    from pm.environments import site_packages

    _commit(venv)
    monkeypatch.syspath_prepend(str(site_packages(venv)))


def _admission(publish=None):
    """PM's admission: commit the selection and, when given, publish a new generation."""
    def admit(enabled, disabled, **_kwargs):
        cfg = load_config()
        cfg["plugins"] = {"enabled": sorted(enabled), "disabled": sorted(disabled)}
        save_config(cfg)
        if publish is not None:
            _commit(publish)
    return admit


def test_enable_that_publishes_a_new_generation_asks_for_a_restart_and_loads_nothing(world, monkeypatch):
    home, reloads = world
    plugin = _plugin(home)
    _boot(monkeypatch, _generation("gen1", with_dep=False))
    newer = _generation("gen2", with_dep=True)
    monkeypatch.setattr("hermes_cli.plugins_admission.admit_plugin_set_change", _admission(publish=newer))

    result = cmd.dashboard_set_agent_plugin_enabled("dep-fixture", enabled=True)

    assert result["ok"] and not result["unchanged"], result
    assert result["restart_required"] is True
    assert result["activation"] is None and result["gateway_reloaded"] is False
    assert "dep-fixture" in cmd._get_enabled_set()
    assert not (plugin / "registered").exists(), "register ran without its dependencies"
    assert DEP not in sys.modules
    assert reloads == [], "a gateway on the old generation was asked to load it"


def test_enable_without_a_new_generation_still_loads_the_plugin_now(world, monkeypatch):
    home, reloads = world
    plugin = _plugin(home)
    _boot(monkeypatch, _generation("gen1", with_dep=True))
    monkeypatch.setattr("hermes_cli.plugins_admission.admit_plugin_set_change", _admission())

    result = cmd.dashboard_set_agent_plugin_enabled("dep-fixture", enabled=True)

    assert result["ok"] and result["restart_required"] is False, result
    assert result["gateway_reloaded"] is True and result["activation"]["key"] == "dep-fixture"
    assert (plugin / "registered").read_text() == "from the generation"
    assert len(reloads) == 1
    sys.modules.pop(DEP, None)


def test_enable_of_a_plugin_without_dependencies_loads_now_even_after_a_publication(world, monkeypatch):
    home, _reloads = world
    plugin = _plugin(home, deps=False)
    _boot(monkeypatch, _generation("gen1", with_dep=False))
    newer = _generation("gen2", with_dep=False)
    monkeypatch.setattr("hermes_cli.plugins_admission.admit_plugin_set_change", _admission(publish=newer))

    result = cmd.dashboard_set_agent_plugin_enabled("dep-fixture", enabled=True)

    assert result["ok"] and result["restart_required"] is False, result
    assert (plugin / "registered").read_text() == "loaded"

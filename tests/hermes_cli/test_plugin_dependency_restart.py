"""A plugin whose Python dependencies were installed after a process booted needs a restart there.

``hermes plugins enable`` commits a new dependency generation and nudges the running gateway to
reload its plugins. Activation is a boot decision, so that gateway (and a running dashboard) keeps
the generation it booted with and cannot import the new dependencies. Enable used to report
"Gateway reloaded plugins; nothing of this plugin needs a session or restart", and discovery
logged "run hermes pm repair" — on a hosted container the terminal backend then failed with
``No module named '<dep>'`` until something restarted the gateway. The honest answer is: restart.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import plugins_activation
from hermes_cli.plugins_loader import installed_in_selected_environment, missing_python_dependencies

DIST = "hermes-depprobe-fixture"
REQ = f"{DIST}>=1.0,<2"


def _dist_info(site: Path, version: str) -> None:
    info = site / f"{DIST.replace('-', '_')}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {DIST}\nVersion: {version}\n", encoding="utf-8")


def _select_generation(home: Path, *, with_dist: bool, version: str = "1.0") -> Path:
    """Commit a PM selection for this checkout under *home*, the way a sync leaves it."""
    from pm.environments import install_state_dir, runtime_facts_path, site_packages
    from pm.paths import repo_root

    venv = install_state_dir(repo_root()) / "environments" / "gen1" / "venv"
    venv.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /nonexistent\n", encoding="utf-8")
    site = site_packages(venv)
    site.mkdir(parents=True)
    if with_dist:
        _dist_info(site, version)
    facts = runtime_facts_path(repo_root())
    facts.write_text(json.dumps({"schema": 1, "packages": {"venv": {"environment": str(venv)}}}),
                     encoding="utf-8")
    return site


@pytest.fixture
def home(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    plugin = hermes_home / "plugins" / "depprobe"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        f"name: depprobe\nversion: '1.0'\npython_dependencies:\n  - '{REQ}'\n", encoding="utf-8")
    (plugin / "__init__.py").write_text("def register(ctx):\n    pass\n", encoding="utf-8")
    (hermes_home / "config.yaml").write_text("plugins:\n  enabled: [depprobe]\n  disabled: []\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


def _discover(caplog):
    from hermes_cli import plugins

    plugins._reset_plugin_managers_for_tests()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
        plugins.discover_plugins(force=True)
    manager = plugins.get_plugin_manager()
    messages = [r.getMessage() for r in caplog.records if "depprobe" in r.getMessage()]
    return manager, messages


def test_discovery_says_restart_when_the_selected_generation_has_the_dependency(home, caplog):
    _select_generation(home, with_dist=True)
    try:
        manager, messages = _discover(caplog)
        assert any("installed after this process started" in m and "restart" in m for m in messages), messages
        assert not any("pm repair" in m for m in messages), messages
        summary = plugins_activation.plugin_activation_summary(manager, "depprobe")
        assert summary["missing_dependencies"] == [REQ]
    finally:
        from hermes_cli import plugins
        plugins._reset_plugin_managers_for_tests()


def test_discovery_keeps_the_repair_remedy_when_nothing_installed_it(home, caplog):
    _select_generation(home, with_dist=False)
    try:
        _manager, messages = _discover(caplog)
        assert any("not installed" in m and "pm repair" in m for m in messages), messages
    finally:
        from hermes_cli import plugins
        plugins._reset_plugin_managers_for_tests()


def test_an_importable_dependency_is_not_reported(home, monkeypatch):
    site = _select_generation(home, with_dist=True)
    monkeypatch.syspath_prepend(str(site))
    from hermes_cli.plugins import PluginManifest

    manifest = PluginManifest(name="depprobe", python_dependencies=[REQ])
    assert missing_python_dependencies(manifest) == []


def _gateway_answers(monkeypatch, activations):
    monkeypatch.setattr("gateway.control_socket.reload_gateway_plugins",
                        lambda target, *, profile_home=None, timeout=30.0:
                        {"reloaded": True, "plugins": ["depprobe"], "activations": activations})
    monkeypatch.setattr(plugins_activation, "notify_serve_backend", lambda name, home: None)


def test_enable_reports_restart_when_the_running_gateway_cannot_import_the_dependency(home, monkeypatch):
    """The live symptom: the gateway reloaded, but it booted before the dependency existed."""
    _select_generation(home, with_dist=True)
    _gateway_answers(monkeypatch, [{"name": "depprobe", "key": "depprobe", "activated_now": {},
                                    "deferred": {}, "missing_dependencies": [REQ]}])

    result = plugins_activation.activate_plugin_now("depprobe", in_process=False)
    hint = plugins_activation.activation_hint(result)

    assert result["gateway_reloaded"] is True
    assert result["restart_required"] is True
    assert result["activation"]["missing_dependencies"] == [REQ]
    assert result["activation"]["missing_in"] == ["gateway"]
    assert "nothing of this plugin needs" not in hint
    assert "installed after the running gateway" in hint and "hermes gateway restart" in hint


def test_restart_names_only_the_process_that_cannot_import(home, monkeypatch):
    """A gateway restarted since the install imports the dependency; the Desktop backend booted
    before it does not. The remedy is reopening the app, not restarting the gateway."""
    _select_generation(home, with_dist=True)
    _gateway_answers(monkeypatch, [{"name": "depprobe", "key": "depprobe", "activated_now": {}, "deferred": {}}])
    monkeypatch.setattr(plugins_activation, "notify_serve_backend", lambda name, home: {"activation": {
        "name": "depprobe", "key": "depprobe", "activated_now": {}, "deferred": {}, "missing_dependencies": [REQ]}})

    result = plugins_activation.activate_plugin_now("depprobe", in_process=False)
    hint = plugins_activation.activation_hint(result)

    assert result["restart_required"] is True and result["activation"]["missing_in"] == ["backend"]
    assert "desktop app" in hint and "hermes gateway restart" not in hint


def test_enable_reports_repair_when_the_dependency_was_never_installed(home, monkeypatch):
    _select_generation(home, with_dist=False)
    _gateway_answers(monkeypatch, [{"name": "depprobe", "key": "depprobe", "activated_now": {},
                                    "deferred": {}, "missing_dependencies": [REQ]}])

    hint = plugins_activation.activation_hint(plugins_activation.activate_plugin_now("depprobe", in_process=False))

    assert "pm repair" in hint


def test_enable_needs_no_restart_when_the_gateway_imports_everything(home, monkeypatch):
    _select_generation(home, with_dist=True)
    _gateway_answers(monkeypatch, [{"name": "depprobe", "key": "depprobe", "activated_now": {},
                                    "deferred": {}}])

    result = plugins_activation.activate_plugin_now("depprobe", in_process=False)

    assert result["restart_required"] is False
    assert "nothing of this plugin needs" in plugins_activation.activation_hint(result)


def test_selected_generation_probe_is_false_without_a_selection(home):
    assert installed_in_selected_environment([REQ]) is False


def test_a_dependency_excluded_by_its_marker_is_never_missing():
    """PM never installs a marker-excluded requirement here; reporting it would demand a restart
    (or repair) on every enable that can never help."""
    from hermes_cli.plugins import PluginManifest

    excluded = f"{DIST}>=1.0,<2; python_version < '3.0'"
    manifest = PluginManifest(name="depprobe", python_dependencies=[excluded, REQ])
    assert missing_python_dependencies(manifest) == [REQ]


def test_a_plugin_importing_its_new_dependency_at_load_still_asks_for_a_restart(home, monkeypatch):
    """The common shape: ``import <dep>`` at module top. In a gateway booted before the install the
    load fails, so the plugin has no activation summary; the reload answer must still carry what
    blocked it, or enable reports "nothing of this plugin needs a session or restart"."""
    from gateway.run_plugin_rewire import reload_plugins_verb
    from hermes_cli import plugins
    from tui_gateway.contracts.tools_mcp_plugins import PluginActivation

    _select_generation(home, with_dist=True)
    (home / "plugins" / "depprobe" / "__init__.py").write_text(
        "import hermes_depprobe_fixture\n\ndef register(ctx):\n    pass\n", encoding="utf-8")
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    try:
        answer = reload_plugins_verb(SimpleNamespace(adapters={}), loop)()  # the gateway's real verb
    finally:
        loop.call_soon_threadsafe(loop.stop)
        plugins._reset_plugin_managers_for_tests()
    monkeypatch.setattr("gateway.control_socket.reload_gateway_plugins",
                        lambda target, *, profile_home=None, timeout=30.0: answer)
    monkeypatch.setattr(plugins_activation, "notify_serve_backend", lambda name, home: None)

    result = plugins_activation.activate_plugin_now("depprobe", in_process=False)

    assert result["restart_required"] is True
    assert result["activation"]["missing_dependencies"] == [REQ]
    assert "nothing of this plugin needs" not in plugins_activation.activation_hint(result)
    PluginActivation.model_validate(result["activation"])  # the plugins.manage wire accepts it


@pytest.mark.parametrize("shape, restart", [
    ("pyproject", True), ("pip_dependencies", True), ("external", False)])
def test_restart_follows_what_pm_installs_for_the_plugin(home, shape, restart):
    """The probe reads the declaration the way PM admission does: an authored pyproject.toml, the
    legacy ``pip_dependencies`` alias, and nothing for a ``python_runtime: external`` sidecar (its
    dependencies never enter Hermes's environment, so no restart or repair can load them)."""
    from hermes_cli import plugins

    _select_generation(home, with_dist=True)
    plugin = home / "plugins" / "depprobe"
    manifest = {"pyproject": "name: depprobe\nversion: '1.0'\n",
                "pip_dependencies": f"name: depprobe\nversion: '1.0'\npip_dependencies:\n  - '{REQ}'\n",
                "external": (f"name: depprobe\nversion: '1.0'\npython_runtime: external\n"
                             f"python_dependencies:\n  - '{REQ}'\n")}[shape]
    (plugin / "plugin.yaml").write_text(manifest, encoding="utf-8")
    if shape == "pyproject":
        (plugin / "pyproject.toml").write_text(
            f'[project]\nname = "depprobe"\nversion = "1.0"\ndependencies = ["{REQ}"]\n', encoding="utf-8")
    if shape != "external":  # an external sidecar's dependency is never importable in-process
        (plugin / "__init__.py").write_text(
            "import hermes_depprobe_fixture\n\ndef register(ctx):\n    pass\n", encoding="utf-8")
    try:
        plugins.discover_plugins(force=True)
        manager = plugins.get_plugin_manager()
        entry = plugins_activation.find_activation(
            plugins_activation.activation_summaries(manager)
            + plugins_activation.dependency_blocked_summaries(manager), "depprobe")
    finally:
        plugins._reset_plugin_managers_for_tests()

    assert bool((entry or {}).get("missing_dependencies")) is restart, entry


def test_an_installed_version_the_requirement_rejects_is_missing(home, monkeypatch, tmp_path):
    """``plugins update`` raising a floor: the running process holds 1.0, the selected generation
    1.7. A name-only probe called that satisfied and asked for no restart."""
    _select_generation(home, with_dist=True, version="1.7")
    running = tmp_path / "running-site"
    _dist_info(running, "1.0")
    monkeypatch.syspath_prepend(str(running))
    from hermes_cli.plugins import PluginManifest

    raised = f"{DIST}>=1.5,<2"
    assert missing_python_dependencies(PluginManifest(name="depprobe", python_dependencies=[raised])) == [raised]
    assert installed_in_selected_environment([raised]) is True
    assert installed_in_selected_environment([f"{DIST}>=1.8"]) is False

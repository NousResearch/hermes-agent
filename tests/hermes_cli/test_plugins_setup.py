"""Real plugin-owned setup subprocesses against isolated profile homes."""
import os
from pathlib import Path

import pytest
import yaml

from hermes_cli import plugins_cmd as cmd


@pytest.fixture
def native(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    plugin = home / "plugins" / "native-fixture"
    plugin.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (plugin / "plugin.yaml").write_text("name: native-fixture\nsetup: {entrypoint: setup.py}\n")
    (plugin / "setup.py").write_text('''from pathlib import Path
import os

def describe(hermes_home):
    home = Path(hermes_home)
    return {"revision": "fixture-v1", "ready": (home / "runtime").exists(),
            "summary": "Install fixture runtime", "details": ["Destination: " + str(home / "runtime")]}

def run(hermes_home):
    home = Path(hermes_home)
    assert os.environ["HERMES_HOME"] == str(home)
    (home / "runtime").write_text("verified fixture")
''')
    return home, plugin


def test_enable_requires_exact_profile_key_revision_before_persist(native):
    home, plugin = native
    before = os.environ.copy()
    result = cmd.dashboard_set_agent_plugin_enabled("native-fixture", enabled=True)
    assert result.get("status") == "consent_required", result
    assert result["setup"]["revision"] == "fixture-v1"
    assert result["consent"] == {"key": "native-fixture", "hermes_home": str(home), "revision": "fixture-v1"}
    assert not (home / "runtime").exists()
    assert "native-fixture" not in cmd._get_enabled_set()
    for field in ("key", "hermes_home", "revision"):
        wrong = {**result["consent"], field: "wrong"}
        refused = cmd.dashboard_set_agent_plugin_enabled("native-fixture", enabled=True, setup_consent=wrong)
        assert refused["status"] == "consent_required"
        assert not (home / "runtime").exists()
    enabled = cmd.dashboard_set_agent_plugin_enabled("native-fixture", enabled=True, setup_consent=result["consent"])
    assert enabled["ok"], enabled
    assert (home / "runtime").read_text() == "verified fixture"
    assert "native-fixture" in cmd._get_enabled_set()
    assert os.environ == before


@pytest.mark.parametrize("surface", ["cli", "composite", "rpc", "rest"])
def test_every_enable_surface_refuses_unconsented_setup(native, monkeypatch, surface):
    home, _ = native
    monkeypatch.setattr(cmd, "_is_tty", lambda: False)
    if surface == "cli":
        with pytest.raises(SystemExit):
            cmd.cmd_enable("native-fixture", allow_tool_override=False)
    elif surface == "composite":
        cmd._persist_plugin_selection(["native-fixture"], {0}, set())
    elif surface == "rpc":
        from tui_gateway import server
        response = server.handle_request({"id": 1, "method": "plugins.manage",
                                          "params": {"action": "toggle", "key": "native-fixture", "enable": True}})
        assert response["error"]["data"]["status"] == "consent_required", response
    else:
        import asyncio
        from fastapi import HTTPException
        from hermes_cli.web_routers import dashboard_ui
        monkeypatch.setattr(dashboard_ui, "_require_token", lambda r: None)
        with pytest.raises(HTTPException) as error:
            # Reuse the sync-test fixture's loop; asyncio.run() would orphan it.
            asyncio.get_event_loop().run_until_complete(
                dashboard_ui.post_agent_plugin_enable(None, "native-fixture")
            )
        assert error.value.detail["status"] == "consent_required"
    assert not (home / "runtime").exists()
    assert "native-fixture" not in cmd._get_enabled_set()


@pytest.mark.parametrize("surface", ["cli", "dashboard", "rpc", "rest"])
def test_install_enable_cannot_bypass_setup(native, tmp_path, monkeypatch, surface):
    import shutil
    import subprocess
    home, plugin = native
    source = tmp_path / "source"
    shutil.move(plugin, source)
    for args in (["init", "-q"], ["add", "."], ["-c", "user.name=Fixture", "-c", "user.email=fixture@example.com", "commit", "-qm", "fixture"]):
        subprocess.run(["git", *args], cwd=source, check=True, capture_output=True)
    monkeypatch.setattr(cmd, "_is_tty", lambda: False)
    if surface == "cli":
        with pytest.raises(SystemExit):
            cmd.cmd_install(source.as_uri(), enable=True)
    elif surface == "dashboard":
        result = cmd.dashboard_install_plugin(source.as_uri(), force=False, enable=True)
        assert result["status"] == "consent_required", result
        assert result["installed"] is True
    elif surface == "rpc":
        from tui_gateway import server
        response = server.handle_request({"id": 1, "method": "plugins.manage", "params": {
            "action": "install", "identifier": source.as_uri(), "enable": True}})
        assert response["error"]["data"]["status"] == "consent_required", response
    else:
        import asyncio
        from fastapi import HTTPException
        from hermes_cli.web_models import _AgentPluginInstallBody
        from hermes_cli.web_routers import dashboard_ui
        monkeypatch.setattr(dashboard_ui, "_require_token", lambda r: None)
        with pytest.raises(HTTPException) as error:
            # Reuse the sync-test fixture's loop; asyncio.run() would orphan it.
            asyncio.get_event_loop().run_until_complete(
                dashboard_ui.post_agent_plugin_install(None, _AgentPluginInstallBody(identifier=source.as_uri()))
            )
        assert error.value.detail["status"] == "consent_required"
    assert plugin.is_dir()
    assert not (home / "runtime").exists()
    assert "native-fixture" not in cmd._get_enabled_set()


@pytest.mark.parametrize("surface", ["cli", "rpc"])
def test_reviewed_consent_flows_through_public_enable_surface(native, surface, monkeypatch):
    import argparse
    import json
    home, _ = native
    consent = cmd.dashboard_set_agent_plugin_enabled("native-fixture", enabled=True)["consent"]
    if surface == "cli":
        from hermes_cli.subcommands.plugins import build_plugins_parser
        parser = argparse.ArgumentParser()
        build_plugins_parser(parser.add_subparsers(), cmd_plugins=cmd.plugins_command)
        args = parser.parse_args(["plugins", "enable", "native-fixture", "--setup-consent", json.dumps(consent), "--no-allow-tool-override"])
        args.func(args)
    else:
        from tui_gateway import server
        response = server.handle_request({"id": 1, "method": "plugins.manage", "params": {
            "action": "toggle", "key": "native-fixture", "enable": True, "setup_consent": consent}})
        assert response["result"]["ok"], response
    assert (home / "runtime").exists()
    assert "native-fixture" in cmd._get_enabled_set()


def test_management_never_loads_runtime_and_manifest_keeps_setup_metadata(native, monkeypatch):
    from hermes_cli import plugins
    from hermes_cli.plugins_manifest import parse_manifest_file
    home, plugin = native
    def forbidden():
        pytest.fail("Enable must not discover/load plugins or touch live cached toolsets")
    monkeypatch.setattr(plugins, "discover_plugins", forbidden)
    manifest = parse_manifest_file(plugin / "plugin.yaml", plugin, "user", "")
    assert manifest.setup == {"entrypoint": "setup.py"}
    consent = cmd.dashboard_set_agent_plugin_enabled("native-fixture", enabled=True)["consent"]
    assert cmd.dashboard_set_agent_plugin_enabled("native-fixture", enabled=True, setup_consent=consent)["ok"]


@pytest.mark.parametrize("mode", ["raise", "unready", "changed", "timeout", "malformed", "traversal", "symlink"])
@pytest.mark.parametrize("was_enabled", [False, True])
def test_setup_failure_preserves_exact_enablement(native, monkeypatch, mode, was_enabled):
    from hermes_cli import plugins_setup
    home, plugin = native
    cmd._save_plugin_sets({"other", *( ["native-fixture"] if was_enabled else [])}, {"disabled-other"})
    before = (home / "config.yaml").read_bytes()
    consent = cmd.dashboard_set_agent_plugin_enabled("native-fixture", enabled=True)["consent"]
    source = (plugin / "setup.py").read_text()
    replacement = {
        "raise": "raise RuntimeError('Install prerequisite missing: fixture-library')",
        "unready": "pass",
        "changed": "(home / 'runtime').write_text('ok')",
        "timeout": "__import__('time').sleep(5)",
    }
    if mode in replacement:
        source = source.replace('(home / "runtime").write_text("verified fixture")', replacement[mode])
        if mode == "changed":
            source = source.replace('"revision": "fixture-v1"', '\"revision\": (\"new\" if (home / \"runtime\").exists() else \"fixture-v1\")')
        (plugin / "setup.py").write_text(source)
    elif mode == "malformed":
        (plugin / "setup.py").write_text("def describe(hermes_home): return {'ready': 'yes'}")
    elif mode == "traversal":
        (plugin / "plugin.yaml").write_text("name: native-fixture\nsetup: {entrypoint: ../setup.py}\n")
    else:
        target = home / "elsewhere.py"
        (plugin / "setup.py").rename(target)
        (plugin / "setup.py").symlink_to(target)
    monkeypatch.setattr(plugins_setup, "RUN_TIMEOUT", 0.2)
    result = cmd.dashboard_set_agent_plugin_enabled("native-fixture", enabled=True, setup_consent=consent)
    assert result["status"] == "setup_failed", result
    assert (home / "config.yaml").read_bytes() == before


def test_rpc_consent_is_bound_to_real_selected_profile(native, tmp_path, monkeypatch):
    import shutil
    from tui_gateway import server
    from hermes_constants import get_hermes_home_override
    home, plugin = native
    from hermes_cli.profiles import get_profile_dir
    selected = get_profile_dir("selected")
    assert selected.is_relative_to(tmp_path)
    shutil.copytree(plugin, selected / "plugins" / "native-fixture")
    def toggle(**params):
        return server.handle_request({"id": 1, "method": "plugins.manage", "params": {
            "action": "toggle", "key": "native-fixture", "enable": True, **params}})
    result = toggle(profile="selected")
    consent = result["error"]["data"]["consent"]
    assert consent["hermes_home"] == str(selected)
    assert toggle(setup_consent=consent)["error"]["data"]["status"] == "consent_required"
    assert toggle(profile="selected", setup_consent=consent)["result"]["plugin"]["status"] == "enabled"
    assert (selected / "runtime").exists()
    assert not (home / "runtime").exists()
    assert "native-fixture" not in cmd._get_enabled_set()
    assert get_hermes_home_override() is None


def test_composite_refusal_preserves_unseen_enablement(native, monkeypatch):
    home, _ = native
    monkeypatch.setattr(cmd, "_is_tty", lambda: False)
    cmd._save_plugin_sets({"unseen-plugin"}, {"disabled-plugin"})
    before = (home / "config.yaml").read_bytes()
    changed, enabled = cmd._persist_plugin_selection(["native-fixture"], {0}, {"disabled-plugin"})
    assert not changed
    assert enabled == {"unseen-plugin"}
    assert (home / "config.yaml").read_bytes() == before


def test_pack_install_does_not_bulk_authorize_native_setup(native, tmp_path, monkeypatch):
    import shutil
    import subprocess
    from hermes_cli.plugin_packs import PluginPack, PackPluginEntry, ResolvedPackPlugin, install_pack_plugins
    home, plugin = native
    source = tmp_path / "source"
    shutil.move(plugin, source)
    def git(*args):
        return subprocess.run(["git", *args], cwd=source, check=True, capture_output=True, text=True).stdout.strip()
    git("init", "-q")
    git("add", ".")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.com", "commit", "-qm", "fixture")
    entry = PackPluginEntry(repo=source.as_uri(), ref=git("rev-parse", "HEAD"))
    monkeypatch.setattr(cmd, "_is_tty", lambda: False)
    results = install_pack_plugins(PluginPack(name="fixture", plugins=[entry]),
                                  [ResolvedPackPlugin(entry=entry, identifier=source.as_uri())], cmd._console(), force=False)
    assert not results[0].ok
    assert "native-fixture" not in cmd._get_enabled_set()
    assert not (home / "runtime").exists()


def test_rest_enable_accepts_exact_reviewed_consent_over_http(native, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from hermes_cli.web_routers import dashboard_ui
    home, _ = native
    monkeypatch.setattr(dashboard_ui, "_require_token", lambda r: None)
    app = FastAPI()
    app.include_router(dashboard_ui.router)
    with TestClient(app) as client:
        refusal = client.post("/api/dashboard/agent-plugins/native-fixture/enable")
        assert refusal.status_code == 409
        consent = refusal.json()["detail"]["consent"]
        result = client.post("/api/dashboard/agent-plugins/native-fixture/enable", json={"setup_consent": consent})
        assert result.status_code == 200, result.text
    assert (home / "runtime").exists()


def test_profile_lock_serializes_consented_setup_processes(native):
    import json
    import subprocess
    import sys
    home, plugin = native
    source = (plugin / "setup.py").read_text().replace(
        '(home / "runtime").write_text("verified fixture")',
        "with (home / 'runs').open('a') as f: f.write('run\\n')\n    __import__('time').sleep(0.2)\n    (home / 'runtime').write_text('verified fixture')")
    (plugin / "setup.py").write_text(source)
    consent = cmd.dashboard_set_agent_plugin_enabled("native-fixture", enabled=True)["consent"]
    script = "from hermes_cli.plugins_cmd import dashboard_set_agent_plugin_enabled as enable; import json,sys; print(json.dumps(enable('native-fixture', enabled=True, setup_consent=json.loads(sys.argv[1]), _toggle_toolsets=False)))"
    children = [subprocess.Popen([sys.executable, "-c", script, json.dumps(consent)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
    for child in children:
        out, err = child.communicate(timeout=30)
        assert child.returncode == 0, err
        assert json.loads(out)["ok"], out
    assert (home / "runs").read_text().splitlines() == ["run"]


@pytest.mark.live_system_guard_bypass
@pytest.mark.linux_only
def test_timeout_retires_setup_descendants_before_returning(native, monkeypatch):
    import psutil
    from hermes_cli import plugins_setup
    home, plugin = native
    source = (plugin / "setup.py").read_text()
    source = source.replace('(home / "runtime").write_text("verified fixture")',
        "import subprocess, sys, time\n    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n    (home / 'child-pid').write_text(str(child.pid))\n    time.sleep(60)")
    (plugin / "setup.py").write_text(source)
    consent = cmd.dashboard_set_agent_plugin_enabled("native-fixture", enabled=True)["consent"]
    monkeypatch.setattr(plugins_setup, "RUN_TIMEOUT", 2)
    result = cmd.dashboard_set_agent_plugin_enabled("native-fixture", enabled=True, setup_consent=consent)
    assert result["status"] == "setup_failed"
    pid = int((home / "child-pid").read_text())
    try:
        child = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    try:
        assert child.status() == psutil.STATUS_ZOMBIE, "Timed-out setup child is still running"
    finally:
        if child.is_running():
            child.kill()


def test_run_return_value_is_not_part_of_the_setup_contract(native):
    home, plugin = native
    with (plugin / "setup.py").open("a") as f:
        f.write("\n    return home / 'runtime'\n")
    consent = cmd.dashboard_set_agent_plugin_enabled("native-fixture", enabled=True)["consent"]
    result = cmd.dashboard_set_agent_plugin_enabled("native-fixture", enabled=True, setup_consent=consent)
    assert result["ok"], result


def test_revision_change_between_describe_and_run_never_executes_unreviewed_setup(native, monkeypatch):
    from hermes_cli import plugins_setup
    home, plugin = native
    consent = cmd.dashboard_set_agent_plugin_enabled("native-fixture", enabled=True)["consent"]
    invoke = plugins_setup._invoke
    def replace_after_describe(path, action, home, **kwargs):
        result = invoke(path, action, home, **kwargs)
        if action == "describe":
            path.write_text(path.read_text().replace("fixture-v1", "fixture-v2"))
        return result
    monkeypatch.setattr(plugins_setup, "_invoke", replace_after_describe)
    result = cmd.dashboard_set_agent_plugin_enabled("native-fixture", enabled=True, setup_consent=consent)
    assert result["status"] == "setup_failed"
    assert not (home / "runtime").exists(), "Setup executed after its revision changed"

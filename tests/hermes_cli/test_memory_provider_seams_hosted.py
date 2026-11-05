"""plugins.isolation: host — the memory-provider seams cross the plugin host: a refused panel save
(``json.JSONDecodeError`` raised in the host) is a 400 with no secret saved, a profile hook runs in
the host with the new profile's home, and a host started only for that hook does not outlive it."""

import json
import os

import hermes_yaml as yaml
import pytest

from hermes_cli import plugins as plugins_mod

_INIT = """
import json
import os
from pathlib import Path

from agent.memory_provider import MemoryProvider


class HostedProv(MemoryProvider):
    name = "hostedprov"

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def get_tool_schemas(self):
        return []

    def on_profile_created(self, profile_name, hermes_home):
        (Path(hermes_home) / "hook.json").write_text(json.dumps({"name": profile_name, "pid": os.getpid()}))

    def read_config_fields(self, hermes_home):
        return {"values": json.loads((Path(hermes_home) / "hostedprov.json").read_text())}

    def write_config_fields(self, values, hermes_home):
        path = Path(hermes_home) / "hostedprov.json"
        cfg = json.loads(path.read_text()) if path.exists() else {}  # JSONDecodeError on a corrupt file
        cfg.update(values)
        path.write_text(json.dumps(cfg))


def _on_session_start(**kwargs):
    return None


def register(ctx):
    ctx.register_memory_provider(HostedProv())
    ctx.register_hook("on_session_start", _on_session_start)
"""

_SCHEMA = """
from plugins.memory.config_schema import KIND_SECRET, KIND_TEXT, STORAGE_PROVIDER, ProviderConfigSchema, ProviderField

CONFIG_SCHEMA = ProviderConfigSchema(
    name="hostedprov", label="Hosted", storage=STORAGE_PROVIDER,
    fields=(ProviderField(key="apiKey", label="Key", kind=KIND_SECRET, description="", env_key="HOSTEDPROV_API_KEY"),
            ProviderField(key="workspace", label="Workspace", kind=KIND_TEXT, description="")),
)
"""


@pytest.fixture
def hosted_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    plugin_dir = home / "plugins" / "hostedprov"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text(_INIT, encoding="utf-8")
    (plugin_dir / "config_schema.py").write_text(_SCHEMA, encoding="utf-8")
    (home / "config.yaml").write_text(yaml.safe_dump(
        {"plugins": {"isolation": "host"}, "memory": {"provider": "hostedprov"}}), encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    plugins_mod._reset_plugin_managers_for_tests()
    yield home
    for manager in plugins_mod._known_plugin_managers():
        host = getattr(manager, "_plugin_host_instance", None)
        if host is not None:
            host.shutdown()
    plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("any")  # the host is a child process
def test_a_refused_save_crosses_the_host_as_a_400_with_no_secret_saved(hosted_home):
    from starlette.testclient import TestClient
    from hermes_cli.config import load_env
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    (hosted_home / "hostedprov.json").write_text("{not json", encoding="utf-8")
    client = TestClient(app, headers={_SESSION_HEADER_NAME: _SESSION_TOKEN})

    resp = client.put("/api/memory/providers/hostedprov/config?surface=declared",
                      json={"values": {"workspace": "w", "apiKey": "hp-key"}})

    assert resp.status_code == 400
    assert "HOSTEDPROV_API_KEY" not in load_env()
    assert (hosted_home / "hostedprov.json").read_text(encoding="utf-8") == "{not json"


@pytest.mark.platforms("any")
def test_a_profile_hook_runs_in_the_host_and_its_host_is_stopped(hosted_home):
    from hermes_cli.profiles import create_profile

    work = create_profile("work", clone_config=True, no_alias=True)

    seen = json.loads((work / "hook.json").read_text())
    assert seen["name"] == "work" and seen["pid"] != os.getpid()
    hosts = [getattr(m, "_plugin_host_instance", None) for m in plugins_mod._known_plugin_managers()]
    assert not any(h is not None and h.alive for h in hosts)
    # The hook-only load's on_session_start proxy pointed at the stopped host: it was dropped too.
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(str(work))
    try:
        assert not plugins_mod.get_plugin_manager()._hooks.get("on_session_start")
    finally:
        reset_hermes_home_override(token)


@pytest.mark.platforms("any")
def test_a_host_already_running_for_that_profile_is_left_alone(hosted_home):
    """Rename fires the hook too: a host the dashboard already runs for that profile keeps running."""
    from unittest.mock import patch

    from hermes_cli.profiles import create_profile, rename_profile
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    create_profile("work", clone_config=True, no_alias=True)

    from hermes_cli import profile_memory_config as pmc  # rename late-imports notify from here
    calls = []
    original = pmc.notify_memory_provider

    def notify_with_host_up(profile_dir, call):
        token = set_hermes_home_override(str(profile_dir))
        try:
            manager = plugins_mod.get_plugin_manager()
            manager._plugin_host().ensure_started()
            calls.append((manager, manager._running_plugin_host().pid))
        finally:
            reset_hermes_home_override(token)
        original(profile_dir, call)

    with patch.object(pmc, "notify_memory_provider", notify_with_host_up), \
            patch("hermes_cli.profiles.check_alias_collision", return_value="skip"):
        play = rename_profile("work", "play")

    (manager, pid), = calls
    host = manager._running_plugin_host()
    assert host is not None and host.pid == pid
    assert json.loads((play / "hook.json").read_text())["name"] == "work"  # created hook, cloned along

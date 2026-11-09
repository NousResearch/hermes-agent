"""plugins.isolation: host — a delegation provider plugin runs out of process and still routes tasks."""

import json
import os
import sys
from pathlib import Path

import hermes_yaml as yaml
import pytest

from hermes_cli import plugins as plugins_mod
from hermes_cli.plugins import PluginManager

HOSTED_PROVIDER = '''
import os
from agent.delegation_provider import DelegationProvider, DelegationRouteError

class Hosted(DelegationProvider):
    @property
    def name(self):
        return "hosted-map"

    def task_parameters(self, config):
        return {"option": {"type": "string", "enum": ["a", "b"], "description": f"pid {os.getpid()}"}}

    def route_task(self, task, config):
        if task.get("option") == "b":
            raise DelegationRouteError("b is closed")
        return {"model": f"model-{task.get('option')}-{os.getpid()}", "label": task.get("option")}

def register(ctx):
    ctx.register_delegation_provider(Hosted())

    def switch(raw_args, session_key=""):
        names = [b["name"] for b in ctx.delegation_backends.list_backends()]
        ctx.delegation_backends.set_global_backend(raw_args.strip(), session_key=session_key, apply_now=True)
        return f"{os.getpid()} {','.join(names)}"
    ctx.register_command("deleg-switch", switch)
'''


@pytest.mark.platforms("posix")  # the host is a child process
def test_hosted_delegation_provider_extends_schema_and_routes_from_the_host(tmp_path, monkeypatch):
    from agent.delegation_provider import DelegationProvider
    from agent.delegation_registry import get_active_provider
    from tools.delegate_tool_routes import resolve_task_routes, schema_extension

    home = tmp_path / ".hermes"
    plugin_dir = home / "plugins" / "hosted-map"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump({"name": "hosted-map", "version": "1.0"}), encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(HOSTED_PROVIDER, encoding="utf-8")
    (home / "config.yaml").write_text(yaml.safe_dump(
        {"plugins": {"enabled": ["hosted-map"], "isolation": "host"}}), encoding="utf-8")
    # Bundled plugins: only the delegation category, so the bundled default strategy is listed and selectable.
    empty_bundled = tmp_path / "bundled"
    empty_bundled.mkdir()
    (empty_bundled / "delegation").symlink_to(Path(__file__).resolve().parents[2] / "plugins" / "delegation",
                                              target_is_directory=True)
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: empty_bundled)
    manager = PluginManager()
    manager.discover_and_load()
    try:
        assert manager._plugins["hosted-map"].error is None
        assert not any("hosted-map" in name or "hosted_map" in name for name in sys.modules)
        config = {"backend": "hosted-map"}
        provider = get_active_provider(config)
        assert isinstance(provider, DelegationProvider) and provider.name == "hosted-map"

        params, _ = schema_extension(config)
        host_pid = int(params["option"]["description"].split()[-1])
        assert params["option"]["enum"] == ["a", "b"] and host_pid != os.getpid()

        routes, err = resolve_task_routes([{"goal": "route me through the host", "option": "a"}], config, None)
        assert err is None
        assert (routes[0].creds["model"], routes[0].label) == (f"model-a-{host_pid}", "a")
        refused, err = resolve_task_routes([{"goal": "refused by the hosted plugin", "option": "b"}], config, None)
        assert err is None and "b is closed" in refused[0].error
        assert json.dumps(params) == json.dumps(schema_extension(config)[0])

        # The /delegation seam acts on THIS process (ctx.delegation_backends is a host remote facade): its
        # config.yaml, and the --now rebuild THIS process's session runs at its next turn.
        from agent import delegation_backends
        from hermes_cli.plugin_command_dispatch import invoke_plugin_command
        out = invoke_plugin_command(plugins_mod.get_plugin_command_handler("deleg-switch"), "default",
                                    session_key="chat-a")
        pid, listed = out.split(" ", 1)
        assert int(pid) != os.getpid() and listed == "default,hosted-map"
        assert yaml.safe_load((home / "config.yaml").read_text())["delegation"]["backend"] == "default"
        assert delegation_backends._rebuild_requests == {"chat-a"}
        delegation_backends._rebuild_requests.clear()
    finally:
        manager.unload()
        manager._plugin_host().shutdown()

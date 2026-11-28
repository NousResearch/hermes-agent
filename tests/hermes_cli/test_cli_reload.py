"""``/reload`` in the classic CLI (#159205): one consented reload of ``.env``, plugins, MCP servers and
skills into the current chat, then one tool rebuild that makes plugin Python tools callable now.
``/reload env`` stays the ``.env``-only reload. Driven through ``HermesCLI.process_command``."""
import threading
from types import SimpleNamespace

import pytest

import hermes_yaml as yaml
from cli import HermesCLI

TOOLSET = "reload_probe"


def _write_plugin(home, tool):
    plugin = home / "plugins" / TOOLSET
    plugin.mkdir(parents=True, exist_ok=True)
    (plugin / "plugin.yaml").write_text(yaml.safe_dump({"name": TOOLSET, "version": "1.0.0"}), encoding="utf-8")
    (plugin / "__init__.py").write_text(
        "def register(ctx):\n"
        f"    ctx.register_tool(name={tool!r}, toolset={TOOLSET!r},\n"
        f"        schema={{'name': {tool!r}, 'description': 'probe', 'parameters': {{'type': 'object', 'properties': {{}}}}}},\n"
        "        handler=lambda args, **kw: 'ok')\n", encoding="utf-8")


def _config(home, **extra):
    # tool_search off: the plugin tool sits in tools[] itself rather than behind the tool_call bridge.
    (home / "config.yaml").write_text(yaml.safe_dump({
        "plugins": {"enabled": [TOOLSET]}, "tools": {"tool_search": {"enabled": "off"}}, **extra}), encoding="utf-8")


@pytest.fixture
def home(monkeypatch):
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    _config(home)
    monkeypatch.setattr("cli._hermes_home", home)  # the CLI config loader's home (the consent gate)
    from hermes_cli import plugins
    plugins._reset_plugin_managers_for_tests()
    yield home
    from tools.registry import registry
    for name in [n for n in registry.get_all_tool_names() if n.startswith("reload_probe_")]:
        registry.deregister(name)
    plugins._reset_plugin_managers_for_tests()


@pytest.fixture
def steps(monkeypatch):
    """Record the step order; MCP and skills are stubbed (no servers), plugins and .env run for real."""
    order = []
    from hermes_cli import config, plugins_activation
    real_env, real_plugins = config.reload_env, plugins_activation.rediscover_plugins
    monkeypatch.setattr(config, "reload_env", lambda: order.append("env") or real_env())
    monkeypatch.setattr(plugins_activation, "rediscover_plugins", lambda: order.append("plugins") or real_plugins())
    monkeypatch.setattr("hermes_cli.chat_reload.reload_mcp_servers",
                        lambda: order.append("mcp") or {"added": [], "removed": [], "reconnected": [], "failed": []})
    monkeypatch.setattr("agent.skill_commands.reload_skills",
                        lambda: order.append("skills") or {"added": [{"name": "fresh", "description": "d"}],
                                                           "removed": [], "total": 1})
    return order


def _cli(monkeypatch, *, running=False):
    monkeypatch.setattr("hermes_cli.gateway_side.connection_for", lambda cli: None)
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "s"
    cli._command_running = False
    cli._agent_running = running
    cli.conversation_history = []
    cli.enabled_toolsets = [TOOLSET]
    cli.agent = SimpleNamespace(tools=[], valid_tool_names=set(), enabled_toolsets=[TOOLSET], disabled_toolsets=None,
                                _agent_tools_lock=threading.RLock())
    cli._busy_command = lambda status: __import__("contextlib").nullcontext()
    return cli


def _names(agent):
    return {tool["function"]["name"] for tool in agent.tools}


def test_reload_now_reloads_in_order_and_publishes_plugin_tools_into_this_chat(home, steps, monkeypatch, capsys):
    """The issue's core contract: a plugin tool registered (or changed) on disk is callable in the
    CURRENT chat after ``/reload now``, with no restart and no new session."""
    _write_plugin(home, "reload_probe_one")
    cli = _cli(monkeypatch)
    assert cli.process_command("/reload now") is True
    assert steps == ["env", "plugins", "mcp", "skills"]
    assert "reload_probe_one" in _names(cli.agent) and "reload_probe_one" in cli.agent.valid_tool_names
    out = capsys.readouterr().out
    assert "added: reload_probe_one" in out and "Plugins:" in out and "fresh" in out
    # One note for the model, on the next user message; never a transcript row of its own.
    assert cli.conversation_history == []
    assert "reload_probe_one" in cli._pending_skills_reload_note and "fresh" in cli._pending_skills_reload_note

    # Changed plugin code is re-imported: the renamed tool replaces the old one in this chat.
    _write_plugin(home, "reload_probe_two")
    import os
    init = home / "plugins" / TOOLSET / "__init__.py"
    os.utime(init, ns=(init.stat().st_mtime_ns + 10**9,) * 2)
    cli.process_command("/reload now")
    assert "reload_probe_two" in _names(cli.agent) and "reload_probe_one" not in _names(cli.agent)
    assert "changed: reload_probe" in capsys.readouterr().out


def test_bare_reload_asks_first_and_a_cancel_changes_nothing(home, steps, monkeypatch, capsys):
    asked = []
    cli = _cli(monkeypatch)
    cli._prompt_text_input_modal = lambda **kw: asked.append(kw) or "cancel"
    cli._normalize_slash_confirm_choice = lambda raw, choices: raw
    cli.process_command("/reload")
    assert asked and "/reload now" in asked[0]["detail"] and "/reload always" in asked[0]["detail"]
    assert steps == [] and cli.agent.tools == []
    assert "Nothing was reloaded" in capsys.readouterr().out


def test_bare_reload_runs_without_asking_when_the_gate_is_off(home, steps, monkeypatch):
    _config(home, approvals={"mcp_reload_confirm": False})
    cli = _cli(monkeypatch)
    cli._prompt_text_input_modal = lambda **kw: pytest.fail("the gate is off: no question")
    cli.process_command("/reload")
    assert steps == ["env", "plugins", "mcp", "skills"]


def test_reload_always_turns_the_gate_off_and_reloads(home, steps, monkeypatch):
    cli = _cli(monkeypatch)
    cli.process_command("/reload always")
    assert steps == ["env", "plugins", "mcp", "skills"]
    saved = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    assert saved["approvals"]["mcp_reload_confirm"] is False


def test_reload_env_is_the_old_env_only_reload(home, steps, monkeypatch, capsys):
    cli = _cli(monkeypatch)
    cli._prompt_text_input_modal = lambda **kw: pytest.fail("/reload env never asks")
    cli.process_command("/reload env")
    assert steps == ["env"]
    assert "Reloaded .env" in capsys.readouterr().out
    assert cli.agent.tools == [] and getattr(cli, "_pending_skills_reload_note", None) is None


def test_reload_is_refused_while_a_turn_runs(home, steps, monkeypatch, capsys):
    cli = _cli(monkeypatch, running=True)
    cli.process_command("/reload now")
    assert steps == [] and cli.agent.tools == []
    assert "unavailable while the agent is running" in capsys.readouterr().out


def test_an_unknown_argument_prints_usage(home, steps, monkeypatch, capsys):
    cli = _cli(monkeypatch)
    cli.process_command("/reload everything")
    assert steps == [] and "/reload [env|now|always]" in capsys.readouterr().out

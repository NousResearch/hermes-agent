"""Every surface a plugin slash command answers on shows a ``CardReply``'s text unless it can render
the card: a platform without ``send_card_message``, a local session's ``slash.exec`` (CLI/TUI/Desktop on
the gateway), the classic CLI in process, the TUI backend's ``command.dispatch``, and a reply that
crossed the plugin-host wire. Contract: ``gateway/plugin_card_actions.py``."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent
from gateway.plugin_card_actions import CardReply, plugin_command_reply, plugin_reply_text
from gateway.run import GatewayRunner
from gateway.session import SessionSource
import hermes_cli.plugins as plugins_mod
from hermes_cli.plugins import PluginManager

CARD = {"type": "AdaptiveCard", "body": [{"type": "TextBlock", "text": "Proposal 42"}],
        "actions": [{"type": "Action.Execute", "title": "Approve", "data": {"hermes_plugin_action": "brain"}}]}
REPLY = CardReply(card=CARD, text="1 proposal waiting: /brain approve 42")


class _TextOnlyAdapter(BasePlatformAdapter):
    """A platform with no ``send_card_message``."""

    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise AssertionError("the reply text goes back through the runner, not an out-of-band send")

    async def get_chat_info(self, chat_id):
        return {}


def _runner(**adapters) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={p: PlatformConfig(enabled=True, token="t") for p in adapters.values()})
    runner._draining = False
    return runner


@pytest.fixture
def brain_command(monkeypatch):
    monkeypatch.setattr(plugins_mod, "get_plugin_command_handler",
                        lambda name: (lambda raw_args: REPLY) if name == "brain" else None)


@pytest.mark.asyncio
async def test_a_platform_without_card_support_replies_with_the_text(brain_command):
    adapter = _TextOnlyAdapter(PlatformConfig(enabled=True, token="t"), Platform.TELEGRAM)
    runner = _runner()
    runner.adapters = {Platform.TELEGRAM: adapter}
    adapter.gateway_runner = runner
    source = adapter.build_source(chat_id="c1", chat_type="dm", user_id="u1")

    handled, reply, _ = await runner._hm_dispatch_quick_and_plugin_commands(
        MessageEvent(text="/brain", source=source, message_id="m1"), source, "brain")

    assert (handled, reply) == (True, REPLY.text)


@pytest.mark.asyncio
async def test_a_local_session_slash_exec_gets_the_text(brain_command):
    from gateway.session_commands import _run_runner_command
    runner = _runner()
    runner.session_store = SimpleNamespace(lookup_by_session_key=lambda key: None)
    source = SessionSource(platform=Platform.LOCAL, chat_id="local-1", chat_type="dm", user_id="u1")
    live = SimpleNamespace(source=source, route="agent:main:local:dm:local-1")

    assert await _run_runner_command(runner, live, "brain", "", "plugin", "brain") == REPLY.text


def test_the_tui_backend_command_dispatch_outputs_the_text(brain_command, monkeypatch):
    from tui_gateway import server
    monkeypatch.setattr(server, "_sessions", {"sid-p": {"session_key": "agent:tui:key-p", "cwd": ""}})

    res = server._methods["command.dispatch"]("d", {"name": "brain", "arg": "", "session_id": "sid-p"})

    assert res["result"] == {"type": "plugin", "output": REPLY.text}


@pytest.fixture
def plugin_home(tmp_path, monkeypatch):
    """A real plugin whose command returns a CardReply, loaded through discovery."""
    home = tmp_path / ".hermes"
    plugin = home / "plugins" / "brain-review"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(json.dumps({"name": "brain-review", "version": "0.1.0"}))
    (plugin / "__init__.py").write_text(
        "from gateway.plugin_card_actions import CardReply\n"
        "def register(ctx):\n"
        f"    reply = CardReply(card={CARD!r}, text={REPLY.text!r})\n"
        "    ctx.register_command('brain', lambda arg: reply)\n")
    (home / "config.yaml").write_text(json.dumps({"plugins": {"enabled": ["brain-review"]}}))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: tmp_path / "empty-bundled")
    monkeypatch.setattr(PluginManager, "_scan_entry_points", lambda self: [])
    plugins_mod._reset_plugin_managers_for_tests()
    plugins_mod.discover_plugins()
    yield home
    plugins_mod._reset_plugin_managers_for_tests()


def test_the_classic_cli_prints_the_text(plugin_home, monkeypatch, capsys):
    from cli import HermesCLI
    monkeypatch.setattr("hermes_cli.gateway_side.connection_for", lambda _cli: None)
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "s1"
    cli._clarify_callback = lambda questions: None

    cli._run_plugin_slash_command("/brain", "")

    out = capsys.readouterr().out
    assert REPLY.text in out
    assert "CardReply" not in out and "AdaptiveCard" not in out


def test_a_card_reply_that_crossed_the_plugin_host_wire_is_rebuilt():
    from hermes_cli.plugin_host_wire import decode, encode
    crossed = decode(encode(REPLY))

    assert plugin_command_reply(crossed) == REPLY
    assert plugin_reply_text(crossed) == REPLY.text


PLAIN_DICT = {"card": CARD, "text": "plain dict"}  # not a CardReply: str() as before


@pytest.mark.parametrize("result, shown", [(None, None), ("", None), ("done", "done"), (42, "42"),
                                           (PLAIN_DICT, str(PLAIN_DICT))])
def test_other_command_results_keep_their_text_form(result, shown):
    assert plugin_reply_text(result) == shown


@pytest.mark.parametrize("card, text", [(CARD, ""), (CARD, "   "), ("{}", "text"), (None, "text"),
                                        ({"type": "AdaptiveCard", "body": [object()]}, "text")])
def test_a_card_reply_requires_a_json_card_and_text(card, text):
    with pytest.raises((TypeError, ValueError)):
        CardReply(card=card, text=text)

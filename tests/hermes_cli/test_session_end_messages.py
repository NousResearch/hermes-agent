"""Plugin ``on_session_finalize`` return values reach the user out-of-band on every surface.

A hook returns ``str`` or ``{"message": str}``; the CLI prints it, the TUI gets it on the
``session.close`` result, the messaging gateway sends it to the owning chat via the adapter.
No surface starts an agent turn with it. ``on_session_end`` (per turn) stays observer-only.
"""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import cli as cli_mod  # module-level, like sibling CLI tests: importing cli runs bootstrap
from hermes_cli import plugins
from hermes_cli.plugins import PluginManager

DIGEST = "Session digest: 3 files touched"


@pytest.fixture
def finalize_hook(monkeypatch):
    """A real PluginManager whose only hook is an on_session_finalize returning a message."""
    calls: list[dict] = []

    def _hook(**kwargs):
        calls.append(kwargs)
        return {"message": DIGEST}

    mgr = PluginManager()
    mgr._discovered = True
    mgr._hooks["on_session_finalize"] = [_hook]
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: mgr)
    return calls


def test_session_end_messages_accepts_str_and_message_dict_only():
    from hermes_cli.lifecycle import session_end_messages

    assert session_end_messages(["  hi  ", {"message": "there"}, {"context": "x"}, {"message": 3}, "", None, 7]) == [
        "hi", "there"]


# ── CLI ──────────────────────────────────────────────────────────────────────────────────────────


def test_cli_new_prints_finalize_message(finalize_hook, capsys):
    cli = cli_mod.HermesCLI()
    cli.agent = MagicMock()
    cli.agent.session_id = "old-session"

    cli.new_session(silent=True)

    assert finalize_hook and finalize_hook[0]["session_id"] == "old-session"
    assert DIGEST in capsys.readouterr().out


def test_cli_exit_prints_finalize_message_after_screen_clear(finalize_hook, capsys, monkeypatch):
    agent = MagicMock()
    agent.session_id = "exit-session"
    monkeypatch.setattr(cli_mod, "_active_agent_ref", agent)
    monkeypatch.setattr(cli_mod, "_cleanup_done", False)
    monkeypatch.setattr(cli_mod, "_session_end_messages", [])

    cli_mod._run_cleanup()
    obj = cli_mod.HermesCLI()
    obj.conversation_history = []
    obj._print_exit_summary(clear_screen=False)

    assert DIGEST in capsys.readouterr().out
    assert cli_mod._session_end_messages == []


def test_one_shot_prints_finalize_message_to_stderr(finalize_hook, capsys, monkeypatch):
    monkeypatch.setattr(cli_mod, "_single_query_finalize_attempted_session_ids", set())
    agent = SimpleNamespace(session_id="oneshot-session", platform="cli")

    cli_mod._notify_single_query_session_finalize(SimpleNamespace(agent=agent, session_id="oneshot-session"))

    out = capsys.readouterr()
    assert DIGEST in out.err
    assert DIGEST not in out.out  # stdout is the answer scripts parse


# ── TUI / Desktop gateway ────────────────────────────────────────────────────────────────────────


def test_tui_session_close_returns_finalize_message(finalize_hook, monkeypatch):
    from tui_gateway import server

    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_TURN_SETTLE_BEFORE_CLOSE_SECONDS", 0.0)
    server._sessions["close-sid"] = {
        "agent": None, "session_key": "close-key", "history": [], "history_lock": threading.Lock(),
        "running": False, "slash_worker": None}

    reply = server.handle_request(
        {"id": "1", "method": "session.close", "params": {"session_id": "close-sid"}})

    assert reply["result"] == {"closed": True, "messages": [DIGEST]}
    assert finalize_hook[0]["session_id"] == "close-key"


# ── messaging gateway ────────────────────────────────────────────────────────────────────────────


def _gateway_runner():
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.run import GatewayRunner
    from gateway.session import SessionEntry, SessionSource, build_session_key

    source = SessionSource(platform=Platform.TELEGRAM, user_id="u1", chat_id="c1", user_name="t", chat_type="dm")
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._session_model_overrides, runner._session_reasoning_overrides = {}, {}
    runner._pending_model_notes, runner._background_tasks = {}, set()
    key = build_session_key(source)
    entry = SessionEntry(session_key=key, session_id="sess-1", created_at=datetime.now(),
                         updated_at=datetime.now(), platform=Platform.TELEGRAM, chat_type="dm", origin=source)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = entry
    runner.session_store.reset_session.return_value = entry
    runner.session_store._entries = {key: entry}
    runner.session_store._generate_session_key.return_value = key
    runner._running_agents, runner._pending_messages, runner._pending_approvals = {}, {}, {}
    runner._session_db = None
    runner._agent_cache_lock = None
    runner._is_user_authorized = lambda _source: True
    runner._format_session_info = lambda: ""
    return runner, adapter, source, key


def _sent_texts(adapter) -> list[str]:
    return [c.args[1] for c in adapter.send.await_args_list]


@pytest.mark.asyncio
async def test_gateway_new_sends_finalize_message_to_owning_chat(finalize_hook):
    from gateway.platforms.event import MessageEvent

    runner, adapter, source, _key = _gateway_runner()

    reply = await runner._handle_reset_command(MessageEvent(text="/new", source=source, message_id="m1"))

    assert finalize_hook[0]["session_id"] == "sess-1"
    assert DIGEST in _sent_texts(adapter)
    assert adapter.send.await_args_list[_sent_texts(adapter).index(DIGEST)].args[0] == "c1"
    assert DIGEST not in str(reply)  # a separate notice, not folded into the /new banner


@pytest.mark.asyncio
async def test_gateway_shutdown_sends_finalize_message_before_teardown(finalize_hook):
    runner, adapter, _source, key = _gateway_runner()
    runner._cleanup_agent_resources_off_loop = AsyncMock()
    runner._flush_agent_transcript_at_shutdown = lambda agent: None

    await runner._finalize_shutdown_agents({key: SimpleNamespace(session_id="sess-1")})

    assert DIGEST in _sent_texts(adapter)


@pytest.mark.asyncio
async def test_gateway_finalize_without_message_sends_nothing(monkeypatch):
    mgr = PluginManager()
    mgr._discovered = True
    mgr._hooks["on_session_finalize"] = [lambda **_: None]
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: mgr)
    runner, adapter, _source, key = _gateway_runner()
    runner._cleanup_agent_resources_off_loop = AsyncMock()
    runner._flush_agent_transcript_at_shutdown = lambda agent: None

    await runner._finalize_shutdown_agents({key: SimpleNamespace(session_id="sess-1")})

    adapter.send.assert_not_awaited()

"""An internal turn hands the agent the chat_name its session already knows.

A Kanban wake and a background-process completion arrive as ``internal=True``
events whose source was rebuilt from a stored route that has no ``chat_name``.
The session-context prompt of such a turn reuses the last human turn's pinned
bytes, but the agent built for it (memory providers' ``initialize`` kwargs, the
session row) and the turn's ``HERMES_SESSION_CHAT_NAME`` read the source
itself. Because the agent signature did not include ``chat_name``, an agent
built on an identity-less internal turn then served every later human turn.

These drive the real ``_handle_message_with_agent`` / ``_run_agent`` path; only
the AIAgent class and config reads are substituted.
"""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.session_context import clear_session_vars, get_session_env, set_session_vars
from hermes_cli import kanban_db as kb
from tests.gateway.test_internal_event_pin_wiring import (
    _ORIGIN,
    KEY,
    _drive,
    _human_source,
    _make_runner,
    _TurnAgent,
)
from tests.gateway.test_kanban_wake_scope import (
    UnscopedAdapter,
    _completed_subscription,
    _one_notifier_tick,
)
from tests.gateway.test_kanban_wake_scope import _runner as _notifier_runner

CHAT_NAME = "Guild / #general"


class _RecordingAgent(_TurnAgent):
    built: list[dict] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        type(self).built.append(kwargs)


def _kanban_wake_source(tmp_path, monkeypatch) -> SessionSource:
    """The source the real Kanban notifier injects for a completed task."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    kb.init_db()
    _completed_subscription(
        platform="discord", chat_id=_ORIGIN["chat_id"], chat_type="group", user_id=_ORIGIN["user_id"],
    )
    adapter = UnscopedAdapter()
    with monkeypatch.context() as tick_patch:  # the tick stubs asyncio.sleep; keep that scoped to it
        asyncio.run(_one_notifier_tick(tick_patch, _notifier_runner(adapter, Platform.DISCORD)))
    assert len(adapter.handled) == 1
    source = adapter.handled[0].source
    assert source.chat_name is None, "the subscription carries no chat name of its own"
    return source


def _turn_runner(monkeypatch, *, durable: dict, origin: SessionSource | None):
    import gateway.run as gr

    monkeypatch.setattr(gr, "_load_gateway_config", lambda: {"model": {"default": "m/default"}})
    monkeypatch.setattr(gr, "_resolve_gateway_model", lambda config=None: "m/default")
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _RecordingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    runner = _make_runner(monkeypatch, durable_prompt_pin=durable)
    runner.session_store._entries = {}
    runner.session_store.get_model_override.return_value = None
    del runner._cache_session_source  # keep the real live-source cache
    runner.session_store.get_or_create_session.return_value.origin = origin
    seen_env: list[str] = []
    runner._set_session_env = lambda context: seen_env.append(context.source.chat_name)
    return runner, seen_env


@pytest.mark.parametrize("known_from", ["live-source", "stored-origin"])
def test_kanban_wake_builds_its_agent_with_the_sessions_chat_name(tmp_path, monkeypatch, known_from):
    wake = _kanban_wake_source(tmp_path, monkeypatch)
    _RecordingAgent.built = []
    durable: dict = {}
    runner, seen_env = _turn_runner(monkeypatch, durable=durable, origin=None)
    asyncio.run(_drive(runner, ((False, _human_source()),)))
    if known_from == "live-source":
        runner._agent_cache.pop(KEY)  # idle eviction: the pins stay, the agent goes
    else:
        # Gateway restart: no live source, no cached agent; the session entry keeps its origin.
        runner, seen_env = _turn_runner(monkeypatch, durable=durable, origin=_human_source())

    asyncio.run(_drive(runner, ((True, wake),)))

    assert _RecordingAgent.built[-1]["chat_name"] == CHAT_NAME
    assert seen_env[-1] == CHAT_NAME


def test_agent_built_on_a_nameless_internal_turn_is_not_reused_for_a_named_turn(monkeypatch):
    _RecordingAgent.built = []
    durable: dict = {}
    runner, _ = _turn_runner(monkeypatch, durable=durable, origin=None)
    asyncio.run(_drive(runner, ((False, _human_source()),)))
    # Restart into a session whose stored origin names no chat: the wake has nothing to inherit.
    runner, _ = _turn_runner(monkeypatch, durable=durable, origin=SessionSource(**_ORIGIN))

    asyncio.run(_drive(runner, ((True, SessionSource(**_ORIGIN)), (False, _human_source()))))

    assert [kw["chat_name"] for kw in _RecordingAgent.built[1:]] == [None, CHAT_NAME]


def test_process_completion_without_stored_origin_keeps_the_spawning_chat_name(monkeypatch):
    import gateway.run as gr
    from tools.process_registry import ProcessSession
    from tools.terminal_tool_background import _register_completion_watcher, _stamp_gateway_routing

    tokens = set_session_vars(
        platform="discord", chat_id=_ORIGIN["chat_id"], chat_type="group", chat_name=CHAT_NAME,
        user_id=_ORIGIN["user_id"], user_name="Ace", session_key=KEY,
    )
    proc = ProcessSession(id="proc_chatname", command="sleep 1", session_key=KEY)
    try:
        _stamp_gateway_routing(proc, get_session_env)
    finally:
        clear_session_vars(tokens)
    registry = types.SimpleNamespace(pending_watchers=[])
    monkeypatch.setattr(gr, "_gateway_runner_ref", lambda: None)
    _register_completion_watcher(registry, proc, KEY)
    proc.exit_code = 0
    evt = GatewayRunner._build_process_completion_event(registry.pending_watchers[0], proc, proc.id)

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.session_store = types.SimpleNamespace(_ensure_loaded=lambda: None, _entries={})
    runner._session_sources = {}
    source = runner._build_process_event_source(evt)

    assert (source.chat_id, source.chat_name, source.user_name) == (_ORIGIN["chat_id"], CHAT_NAME, "Ace")

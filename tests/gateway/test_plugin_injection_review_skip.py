"""#137339 — plugin-injected turns must not trigger the background review fork.

``PluginContext.inject_message`` dispatches a ``MessageEvent`` stamped with
``metadata["hermes_plugin_injection"]``.  Those turns are machine-initiated like
cron turns, so the end-of-turn memory/skill review fork (~30K tokens / event)
has no human-in-the-loop value; the gateway now forwards the stamp as
``skip_background_review``.

Wiring guard: this drives the REAL ``_run_agent`` / ``TurnRunner.run_sync`` path
(only the AIAgent class and config reads are substituted), because the flag is
rewritten on the per-session cached agent every turn — a stub at ``_run_agent``
would not catch the flag latching onto the next human turn's review.
"""

from __future__ import annotations

import sys
import types
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource

KEY = "agent:main:discord:group:1513247605675790346:117431298246705156"
_ORIGIN = dict(
    platform=Platform.DISCORD,
    chat_id="1513247605675790346",
    chat_type="group",
    user_id="117431298246705156",
    scope_id="1480524732964278294",
)


def _human_source() -> SessionSource:
    return SessionSource(
        **_ORIGIN,
        chat_name="Guild / #general",
        user_name="Ace",
        message_id="1552671843494666330",
    )


def _injection_source() -> SessionSource:
    # run_plugin_injection dispatches on a source restored from the session
    # store, so no chat_name / user_name / message_id.
    return SessionSource(**_ORIGIN)


def _make_runner(monkeypatch):
    import agent.model_metadata as mm
    import gateway.run as gr
    import gateway.session as gs

    monkeypatch.setattr(gs, "_discord_tools_loaded", lambda: True)
    monkeypatch.setattr(
        gr, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    monkeypatch.setattr(mm, "get_model_context_length", lambda *a, **k: 100_000)

    r = gr.GatewayRunner(GatewayConfig())
    r.adapters = {}
    r._running_agents = {}
    r._running_agents_ts = {}
    r._pending_messages = {}
    r._pending_approvals = {}
    r._is_user_authorized = lambda s: True
    r._set_session_env = lambda c: None
    r._handle_active_session_busy_message = AsyncMock(return_value=False)
    r._session_db = MagicMock()
    r._recover_telegram_topic_thread_id = lambda s: None
    r._cache_session_source = lambda k, s: None
    r._is_session_run_current = lambda k, g: True
    r._begin_session_run_generation = lambda k: 1
    r._reply_anchor_for_event = lambda e: None
    r._get_guild_id = lambda e: None
    r._should_send_voice_reply = lambda *a, **k: False
    r.hooks = MagicMock()
    r.hooks.emit = AsyncMock()
    # The turn lease is released by the real turn tail; disable leasing so
    # sequential turns can run.
    r._turn_leases = None

    store = MagicMock()
    store.get_or_create_session.return_value = SessionEntry(
        session_key=KEY,
        session_id="sess-injection",
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 2),
        platform=Platform.DISCORD,
        chat_type="group",
    )
    store.load_transcript.return_value = []
    store.has_platform_message_id.return_value = False
    r.session_store = store
    return r


class _TurnAgent:
    """AIAgent stand-in: answers without an API call; mirrors init_agent's
    ``skip_background_review = bool(skip_background_review)`` default."""

    def __init__(self, *args, **kwargs):
        self.model = kwargs.get("model")
        self.provider = kwargs.get("provider")
        self.base_url = kwargs.get("base_url")
        self.session_id = kwargs.get("session_id")
        self.tools = []
        self.skip_background_review = bool(kwargs.get("skip_background_review", False))
        self._primary_runtime = {
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
        }

    def run_conversation(
        self, user_message, conversation_history=None, task_id=None, **kwargs
    ):
        return {
            "final_response": "ok",
            "messages": (conversation_history or [])
            + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": "ok"},
            ],
            "api_calls": 1,
        }


def _install_fake_agent_module(monkeypatch) -> None:
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _TurnAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)


async def _human_turn(runner) -> None:
    src = _human_source()
    event = MessageEvent(text="hi", source=src, message_id=src.message_id)
    await runner._handle_message_with_agent(event, src, KEY, 1)


async def _injected_turn(runner) -> None:
    src = _injection_source()
    # The stamp _dispatch_plugin_message_injection puts on every injected event; the
    # strict gateway_session_key routing pins are unrelated to this fix (and would need
    # the adapter's derived-key check to line up), so only the injection stamp is kept.
    event = MessageEvent(
        text="POKE: say OK",
        source=src,
        internal=True,
        allow_gateway_control=False,
        metadata={"hermes_plugin_id": "poke", "hermes_plugin_injection": True},
    )
    await runner._handle_message_with_agent(event, src, KEY, 1)


def _cached_agent(runner) -> _TurnAgent:
    entry = runner._agent_cache.get(KEY)
    assert entry is not None, "turn evicted the session's cached agent"
    return entry[0]


@pytest.mark.asyncio
async def test_injected_turn_skips_review_and_human_turns_restore_it(monkeypatch):
    _install_fake_agent_module(monkeypatch)
    runner = _make_runner(monkeypatch)
    runner.session_store._entries = {}

    await _human_turn(runner)
    agent = _cached_agent(runner)
    assert agent.skip_background_review is False, "human turn must keep reviews enabled"

    await _injected_turn(runner)
    agent = _cached_agent(runner)
    assert agent.skip_background_review is True, (
        "plugin-injected turn did not set skip_background_review on the turn's agent"
    )

    await _human_turn(runner)
    agent = _cached_agent(runner)
    assert agent.skip_background_review is False, (
        "injected turn left skip_background_review latched for the next human turn's review"
    )

"""Call-site wiring guard for the internal-event session-context pin.

``_pinned_session_context_prompt(..., internal=True)`` reuses the existing pin
verbatim so an internal event (kanban wake, delegation completion, watch
notification) cannot re-key it.  A helper-level unit test would stay green if
the call site in ``_handle_message_with_agent`` stopped forwarding
``event.internal``, so this test drives the REAL handler through
human -> internal -> human on one session and asserts the context prompt that
reaches ``_run_agent`` is byte-identical on all three turns.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionEntry, SessionSource
from gateway.turn_context import TurnContext

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


def _wake_source() -> SessionSource:
    # kanban_watchers._push_wake shape: rebuilt from the persisted origin, so
    # no chat_name / user_name / message_id.
    return SessionSource(**_ORIGIN)


def _make_runner(monkeypatch, config: GatewayConfig | None = None):
    import agent.model_metadata as mm
    import gateway.run as gr
    import gateway.session as gs

    monkeypatch.setattr(gs, "_discord_tools_loaded", lambda: True)
    monkeypatch.setattr(gr, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    monkeypatch.setattr(mm, "get_model_context_length", lambda *a, **k: 100_000)

    r = gr.GatewayRunner(config or GatewayConfig())
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
    # The turn lease is released by the real turn tail, which _run_agent's
    # stub skips; disable leasing so three sequential turns can run.
    r._turn_leases = None

    store = MagicMock()
    store.get_or_create_session.return_value = SessionEntry(
        session_key=KEY,
        session_id="sess-wiring",
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 2),
        platform=Platform.DISCORD,
        chat_type="group",
    )
    store.load_transcript.return_value = []
    store.has_platform_message_id.return_value = False
    r.session_store = store
    return r


def _capture(runner, sink: list):
    async def fake_run_agent(**kw):
        sink.append(kw)
        return {
            "final_response": "ok",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }

    runner._run_agent = fake_run_agent


async def _drive(runner, turns, *, channel_prompt=None):
    for internal, src in turns:
        event = MessageEvent(
            text="[kanban] wake" if internal else "hi",
            source=src,
            message_id=None if internal else src.message_id,
            internal=internal,
            # Adapters resolve channel_prompts onto human events; the kanban
            # wake is built without one.
            channel_prompt=None if internal else channel_prompt,
        )
        await runner._handle_message_with_agent(event, src, KEY, 1)


@pytest.mark.asyncio
async def test_internal_event_reuses_pin_through_real_handler(monkeypatch):
    runner = _make_runner(monkeypatch)
    calls: list[dict] = []
    _capture(runner, calls)

    # Internal-first on a fresh session (kanban/API-server wake, startup resume):
    # with no pin to reuse it must still render AND pin, or every internal-only
    # turn re-renders and loses verbatim reuse.
    await _drive(runner, ((True, _wake_source()),))
    assert runner._peek_session_state(KEY).conversation.ephemeral_pin is not None

    await _drive(runner, ((False, _human_source()), (True, _wake_source()), (False, _human_source())))

    seen = [kw["context_prompt"] for kw in calls[1:]]
    assert len(seen) == 3, f"_run_agent reached {len(seen)}/3 turns"
    # The human render names the chat; the wake-shaped source cannot.  If the
    # internal turn re-rendered, its bytes would differ and the next human
    # turn would re-key back (A->B->A).
    assert "Guild / #general" in seen[0]
    assert seen[0] == seen[1] == seen[2], "internal event re-keyed the session-context pin"


# ---------------------------------------------------------------------------
# The other ephemeral prompt components must not toggle across the sequence either.
# ---------------------------------------------------------------------------

THREAD_ID = "1552000000000000001"
PARENT_ID = "1513247605675790346"


def _effective_ephemeral(runner, kw) -> str:
    """The real TurnRunner combiner applied to what reached ``_run_agent``."""
    ctx = TurnContext(source=kw["source"], context_prompt=kw["context_prompt"], channel_prompt=kw.get("channel_prompt"))
    return TurnRunner(runner, ctx)._combined_ephemeral_prompt()


def _thread_origin() -> dict:
    return dict(
        platform=Platform.DISCORD,
        chat_id=THREAD_ID,
        chat_type="thread",
        thread_id=THREAD_ID,
        user_id="117431298246705156",
        scope_id="1480524732964278294",
    )


def _human_thread_source() -> SessionSource:
    return SessionSource(
        **_thread_origin(),
        parent_chat_id=PARENT_ID,
        chat_name="Guild / #dev / build thread",
        user_name="Ace",
        message_id="1552671843494666331",
    )


def _wake_thread_source() -> SessionSource:
    # _push_wake rebuilds chat_id/thread_id from the subscription; it has no
    # parent_chat_id, chat_name, user_name or message_id.
    return SessionSource(**_thread_origin())


@pytest.mark.asyncio
async def test_internal_event_keeps_channel_prompt_and_parent_override(monkeypatch):
    config = GatewayConfig()
    config.platforms[Platform.DISCORD] = PlatformConfig(
        enabled=True,
        channel_overrides={PARENT_ID: ChannelOverride(system_prompt="Parent persona.")},
    )
    runner = _make_runner(monkeypatch, config)
    calls: list[dict] = []
    _capture(runner, calls)

    await _drive(
        runner,
        ((False, _human_thread_source()), (True, _wake_thread_source()), (False, _human_thread_source())),
        channel_prompt="Channel hint.",
    )

    assert len(calls) == 3
    eph = [_effective_ephemeral(runner, kw) for kw in calls]
    assert "Channel hint." in eph[0] and "Parent persona." in eph[0]
    assert eph[0] == eph[1] == eph[2], "internal event toggled the channel ephemeral components"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result_key", "interrupted"),
    (("pending_steer", False), ("interrupt_message", True)),
    ids=("leftover-steer", "interrupt-text"),
)
@pytest.mark.parametrize("channel_prompt", ["Channel hint.", "", None], ids=("hint", "empty", "none"))
async def test_eventless_followup_keeps_effective_prompt_through_next_human(
    monkeypatch, result_key, interrupted, channel_prompt
):
    config = GatewayConfig()
    config.platforms[Platform.DISCORD] = PlatformConfig(
        enabled=True,
        channel_overrides={PARENT_ID: ChannelOverride(system_prompt="Parent persona.")},
    )
    runner = _make_runner(monkeypatch, config)
    calls: list[dict] = []
    _capture(runner, calls)
    runner._run_agent_deliver_first_response = AsyncMock()
    runner._refresh_agent_cache_message_count = AsyncMock()

    adapter = MagicMock()
    adapter.get_pending_message.return_value = None
    adapter._active_sessions = {}
    source = _human_thread_source()

    await _drive(runner, ((False, source),), channel_prompt=channel_prompt)
    first = calls[0]
    turn_ctx = TurnContext(
        source=first["source"],
        context_prompt=first["context_prompt"],
        channel_prompt=first["channel_prompt"],
        session_key=first["session_key"],
        session_id=first["session_id"],
        run_generation=1,
        history=[],
    )
    result = {
        "final_response": "done",
        "messages": [],
        result_key: "follow up",
        "interrupted": interrupted,
    }

    pending_event, pending = await runner._run_agent_drain_pending(result, adapter, source, KEY)
    assert pending_event is None
    assert pending == "follow up"
    await runner._run_agent_queued_followup(
        turn_ctx, adapter, pending, pending_event, "done", result, None
    )
    await _drive(runner, ((False, source),), channel_prompt=channel_prompt)

    assert [call["channel_prompt"] for call in calls] == [channel_prompt] * 3
    ephemeral = [_effective_ephemeral(runner, call) for call in calls]
    assert "Parent persona." in ephemeral[0]
    assert ephemeral[0] == ephemeral[1] == ephemeral[2]


@pytest.mark.asyncio
async def test_event_backed_followup_overrides_inherited_channel_prompt(monkeypatch):
    runner = _make_runner(monkeypatch)
    calls: list[dict] = []
    _capture(runner, calls)
    runner._run_agent_deliver_first_response = AsyncMock()
    runner._refresh_agent_cache_message_count = AsyncMock()
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="queued")
    runner._session_key_for_source = lambda source: KEY

    source = _human_source()
    adapter = MagicMock()
    adapter._active_sessions = {}
    pending_event = MessageEvent(
        text="queued",
        source=source,
        message_id="queued-message-1",
        channel_prompt="Queued event prompt.",
    )
    turn_ctx = TurnContext(
        source=source,
        context_prompt="ctx",
        channel_prompt="Inherited prompt.",
        session_key=KEY,
        session_id="sess-wiring",
        run_generation=1,
        history=[],
    )
    result = {"final_response": "done", "messages": []}

    await runner._run_agent_queued_followup(
        turn_ctx, adapter, "queued", pending_event, "done", result, None
    )

    assert len(calls) == 1
    assert calls[0]["channel_prompt"] == "Queued event prompt.", "event prompt must win over the inherited one"


@pytest.mark.asyncio
async def test_synthetic_non_internal_event_reuses_channel_pin(monkeypatch):
    """Goal continuation / heartbeat due-prompt / ``/goal`` resume events are non-internal (they
    must still pass authorization, unlike the kanban-wake ``internal=True`` case covered above)
    but carry no channel inputs of their own — ``_synthetic_prompt_event`` always sets
    ``channel_prompt=None``. Regression for #126109: without ``inherit_channel_pin``, that
    ``None`` overwrote the session's ``channel_pin``, and the human turn after it lost its
    channel prompt / parent-override too. Also asserts full ephemeral byte-equality: the
    synthetic turn strips ``message_id`` too, which is a ``_ephemeral_change_key`` input on
    Discord, so without feeding the same ``inherit_channel_pin`` decision into
    ``_pinned_session_context_prompt`` the session-context block re-renders shorter on the
    synthetic turn and back on the human turn after it (A->B->A on the full ephemeral prompt,
    not just ``channel_prompt``)."""
    config = GatewayConfig()
    config.platforms[Platform.DISCORD] = PlatformConfig(
        enabled=True,
        channel_overrides={PARENT_ID: ChannelOverride(system_prompt="Parent persona.")},
    )
    runner = _make_runner(monkeypatch, config)
    calls: list[dict] = []
    _capture(runner, calls)

    human = _human_thread_source()
    # _synthetic_prompt_event() reuses the triggering source verbatim, stripping only
    # message_id — chat_name/user_name/parent_chat_id survive, unlike the rebuilt-from-origin
    # kanban wake source used elsewhere in this file.
    synthetic_source = dataclasses.replace(human, message_id=None)

    for text, source, channel_prompt, inherit_channel_pin in (
        ("hi", human, "Channel hint.", False),
        ("[goal] continuation", synthetic_source, None, True),
        ("hi again", human, "Channel hint.", False),
    ):
        event = MessageEvent(
            text=text, source=source, message_id=source.message_id, internal=False,
            channel_prompt=channel_prompt, inherit_channel_pin=inherit_channel_pin,
        )
        await runner._handle_message_with_agent(event, source, KEY, 1)

    assert len(calls) == 3
    # The synthetic middle turn carries its own channel_prompt=None but must resolve to the
    # pinned "Channel hint." — and so must the human turn after it.
    resolved_channel_prompts = [kw["channel_prompt"] for kw in calls]
    assert resolved_channel_prompts == ["Channel hint."] * 3, (
        f"channel pin flipped across the turn sequence: {resolved_channel_prompts}"
    )
    # Full ephemeral byte-equality: the parent-chat override AND the session-context block must
    # stay identical across human -> synthetic -> human, not just contain the same substrings.
    eph = [_effective_ephemeral(runner, kw) for kw in calls]
    assert "Channel hint." in eph[0] and "Parent persona." in eph[0]
    assert eph[0] == eph[1] == eph[2], "synthetic turn toggled the ephemeral prompt bytes"

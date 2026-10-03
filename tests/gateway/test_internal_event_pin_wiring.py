"""Call-site wiring guard for prompt-pin preserving gateway events.

``_pinned_session_context_prompt(..., preserve_pin=True)`` reuses the existing pin
verbatim so internal wakes and non-internal synthetic continuations cannot re-key it.
A helper-level unit test would stay green if the real handler stopped forwarding the
resolved preservation decision, so these tests drive real turn handling and assert the
effective prompt stays byte-identical.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import ChannelOverride, GatewayConfig, HomeChannel, Platform, PlatformConfig
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


def _make_runner(
    monkeypatch, config: GatewayConfig | None = None, *, durable_prompt_pin: dict | None = None,
):
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
    if durable_prompt_pin is not None:
        def _get_prompt_pin(_key, *, expected_session_id=None):
            if expected_session_id is not None:
                assert expected_session_id == "sess-wiring"
            value = durable_prompt_pin.get("value")
            return dict(value) if isinstance(value, dict) else None

        def _set_prompt_pin(_key, value, *, expected_session_id=None):
            assert expected_session_id == "sess-wiring"
            durable_prompt_pin["value"] = dict(value) if isinstance(value, dict) else None
            return True

        store.get_prompt_pin.side_effect = _get_prompt_pin
        store.set_prompt_pin.side_effect = _set_prompt_pin
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
async def test_first_internal_event_after_restart_rehydrates_durable_prompt_pins(monkeypatch):
    config = GatewayConfig()
    config.platforms[Platform.DISCORD] = PlatformConfig(
        enabled=True,
        channel_overrides={PARENT_ID: ChannelOverride(system_prompt="Parent persona.")},
    )
    durable: dict = {}

    before = _make_runner(monkeypatch, config, durable_prompt_pin=durable)
    calls_before: list[dict] = []
    _capture(before, calls_before)
    await _drive(before, ((False, _human_thread_source()),), channel_prompt="Channel hint.")

    persisted = durable.get("value")
    assert isinstance(persisted, dict)
    assert persisted["channel_prompt"] == "Channel hint."
    assert persisted["parent_chat_id"] == PARENT_ID

    # A brand-new runner has no ConversationState. The first post-restart event is internal.
    after = _make_runner(monkeypatch, config, durable_prompt_pin=durable)
    calls_after: list[dict] = []
    _capture(after, calls_after)
    await _drive(
        after,
        ((True, _wake_thread_source()), (False, _human_thread_source())),
        channel_prompt="Channel hint.",
    )

    human, first_internal, next_human = calls_before[0], calls_after[0], calls_after[1]
    assert after._peek_session_state(KEY).conversation.ephemeral_pin is not None
    assert after._peek_session_state(KEY).conversation.channel_pin is not None
    assert first_internal["context_prompt"] == human["context_prompt"] == next_human["context_prompt"]
    assert first_internal["channel_prompt"] == human["channel_prompt"] == next_human["channel_prompt"]
    assert (
        first_internal["source"].parent_chat_id
        == human["source"].parent_chat_id
        == next_human["source"].parent_chat_id
    )
    assert _effective_ephemeral(after, first_internal) == _effective_ephemeral(before, human)


@pytest.mark.asyncio
async def test_human_first_after_restart_ignores_stale_durable_prompt_pin(monkeypatch):
    durable = {
        "value": {
            "version": 1,
            "context_key": "stale-key",
            "context_prompt": "STALE CONTEXT",
            "redact_pii": False,
            "channel_prompt": "Stale channel prompt.",
            "parent_chat_id": "stale-parent",
        }
    }
    config = GatewayConfig()
    config.platforms[Platform.DISCORD] = PlatformConfig(
        enabled=True,
        channel_overrides={PARENT_ID: ChannelOverride(system_prompt="Parent persona.")},
    )
    runner = _make_runner(monkeypatch, config, durable_prompt_pin=durable)
    calls: list[dict] = []
    _capture(runner, calls)

    await _drive(runner, ((False, _human_thread_source()),), channel_prompt="Fresh channel prompt.")

    assert len(calls) == 1
    assert calls[0]["context_prompt"] != "STALE CONTEXT"
    assert calls[0]["channel_prompt"] == "Fresh channel prompt."
    assert calls[0]["source"].parent_chat_id == PARENT_ID
    assert durable["value"]["context_prompt"] == calls[0]["context_prompt"]
    assert durable["value"]["channel_prompt"] == "Fresh channel prompt."
    assert durable["value"]["parent_chat_id"] == PARENT_ID


@pytest.mark.asyncio
async def test_internal_event_never_reuses_prompt_pin_from_another_privacy_policy(monkeypatch):
    """A pin rendered with redact_pii off must not reach the model once redaction is on, not even
    through the internal-event reuse path after a restart."""
    import gateway.run as gr

    monkeypatch.setattr(gr, "_load_gateway_config", lambda: {"privacy": {"redact_pii": True}})
    durable = {
        "value": {
            "version": 1,
            "context_key": "unredacted-key",
            "context_prompt": "UNREDACTED CONTEXT",
            "redact_pii": False,
            "channel_prompt": "Channel hint.",
            "parent_chat_id": None,
        }
    }
    runner = _make_runner(monkeypatch, durable_prompt_pin=durable)
    calls: list[dict] = []
    _capture(runner, calls)

    await _drive(runner, ((True, _wake_source()),))

    assert calls[0]["context_prompt"] != "UNREDACTED CONTEXT"
    assert calls[0]["channel_prompt"] == "Channel hint."


@pytest.mark.asyncio
async def test_non_internal_synthetic_event_preserves_all_prompt_pins(monkeypatch):
    """#126109: a synthetic continuation reuses both prompt pins without becoming internal."""
    config = GatewayConfig()
    config.platforms[Platform.DISCORD] = PlatformConfig(
        enabled=True,
        channel_overrides={PARENT_ID: ChannelOverride(system_prompt="Parent persona.")},
    )
    runner = _make_runner(monkeypatch, config)
    calls: list[dict] = []
    _capture(runner, calls)

    source = _human_thread_source()
    await _drive(runner, ((False, source),), channel_prompt="Channel hint.")

    synthetic = runner._synthetic_prompt_event(source, "[Continuing toward your standing goal]")
    assert synthetic.internal is False
    assert synthetic.source.message_id is None
    synthetic = dataclasses.replace(synthetic, text=synthetic.text + " (rewritten)")

    await runner._handle_message_with_agent(synthetic, synthetic.source, KEY, 1)
    await _drive(runner, ((False, source),), channel_prompt="Channel hint.")

    assert [call["channel_prompt"] for call in calls] == ["Channel hint."] * 3
    ephemeral = [_effective_ephemeral(runner, call) for call in calls]
    assert ephemeral[0] == ephemeral[1] == ephemeral[2]
    assert runner._peek_session_state(KEY).conversation.channel_pin == ("Channel hint.", PARENT_ID)


@pytest.mark.asyncio
async def test_first_non_internal_synthetic_after_restart_rehydrates_prompt_pins(monkeypatch):
    """A restart must not make the first synthetic continuation temporarily drop prompt identity."""
    config = GatewayConfig()
    config.platforms[Platform.DISCORD] = PlatformConfig(
        enabled=True,
        channel_overrides={PARENT_ID: ChannelOverride(system_prompt="Parent persona.")},
    )
    durable: dict = {}
    source = _human_thread_source()

    before = _make_runner(monkeypatch, config, durable_prompt_pin=durable)
    calls_before: list[dict] = []
    _capture(before, calls_before)
    await _drive(before, ((False, source),), channel_prompt="Channel hint.")
    assert isinstance(durable.get("value"), dict)

    after = _make_runner(monkeypatch, config, durable_prompt_pin=durable)
    calls_after: list[dict] = []
    _capture(after, calls_after)
    synthetic = after._synthetic_prompt_event(source, "[heartbeat] continue")
    await after._handle_message_with_agent(synthetic, synthetic.source, KEY, 1)
    await _drive(after, ((False, source),), channel_prompt="Channel hint.")

    human, first_synthetic, next_human = calls_before[0], calls_after[0], calls_after[1]
    assert first_synthetic["channel_prompt"] == human["channel_prompt"] == next_human["channel_prompt"]
    assert first_synthetic["context_prompt"] == human["context_prompt"] == next_human["context_prompt"]
    assert _effective_ephemeral(after, first_synthetic) == _effective_ephemeral(before, human)
    assert _effective_ephemeral(after, next_human) == _effective_ephemeral(before, human)


@pytest.mark.asyncio
@pytest.mark.parametrize("restart_before_synthetic", [False, True], ids=("live", "after-restart"))
async def test_synthetic_turn_refreshes_home_config_without_losing_pinned_source(
    monkeypatch, restart_before_synthetic
):
    """A pin-preserving continuation must observe /sethome without adopting synthetic source shape."""
    import gateway.slash_commands as slash_commands
    import hermes_cli.config as cli_config

    monkeypatch.setattr(slash_commands, "persist_home_channel", lambda *_a, **_k: None)
    monkeypatch.setattr(cli_config, "save_env_value", lambda *_a, **_k: None)

    old_home = HomeChannel(
        platform=Platform.DISCORD,
        chat_id="111111111111111111",
        name="Old home",
    )
    config = GatewayConfig()
    config.platforms[Platform.DISCORD] = PlatformConfig(enabled=True, home_channel=old_home)
    durable: dict = {}
    source = _human_source()

    before = _make_runner(monkeypatch, config, durable_prompt_pin=durable)
    before_calls: list[dict] = []
    _capture(before, before_calls)
    await _drive(before, ((False, source),), channel_prompt="Channel hint.")

    assert "111111111111111111" in before_calls[0]["context_prompt"]
    assert "Guild / #general" in before_calls[0]["context_prompt"]
    assert durable["value"]["source"]["message_id"] == source.message_id

    runner = (
        _make_runner(monkeypatch, config, durable_prompt_pin=durable)
        if restart_before_synthetic
        else before
    )
    calls = [] if restart_before_synthetic else before_calls
    if restart_before_synthetic:
        _capture(runner, calls)

    new_home_source = dataclasses.replace(
        source,
        chat_id="222222222222222222",
        chat_name="New home",
        message_id="1552671843494666399",
    )
    reply = await runner._handle_set_home_command(
        MessageEvent(
            text="/sethome",
            source=new_home_source,
            message_id=new_home_source.message_id,
        )
    )
    assert reply
    assert config.platforms[Platform.DISCORD].home_channel.chat_id == "222222222222222222"

    synthetic = runner._synthetic_prompt_event(source, "[heartbeat] continue")
    await runner._handle_message_with_agent(synthetic, synthetic.source, KEY, 1)
    await _drive(runner, ((False, source),), channel_prompt="Channel hint.")

    synthetic_call, next_human = calls[-2], calls[-1]
    assert "222222222222222222" in synthetic_call["context_prompt"]
    assert "New home" in synthetic_call["context_prompt"]
    assert "111111111111111111" not in synthetic_call["context_prompt"]
    assert "Guild / #general" in synthetic_call["context_prompt"]
    assert synthetic_call["context_prompt"] == next_human["context_prompt"]

    # The synthetic refresh itself is durable: a crash before another authoritative turn must not
    # resurrect the old home destination on the next continuation.
    assert durable["value"]["context_prompt"] == synthetic_call["context_prompt"]
    restarted = _make_runner(monkeypatch, config, durable_prompt_pin=durable)
    restarted_calls: list[dict] = []
    _capture(restarted, restarted_calls)
    resumed = restarted._synthetic_prompt_event(source, "[heartbeat] continue after restart")
    await restarted._handle_message_with_agent(resumed, resumed.source, KEY, 1)
    assert "222222222222222222" in restarted_calls[0]["context_prompt"]
    assert "111111111111111111" not in restarted_calls[0]["context_prompt"]
    assert restarted_calls[0]["context_prompt"] == synthetic_call["context_prompt"]

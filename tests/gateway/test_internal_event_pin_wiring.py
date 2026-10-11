"""Call-site wiring guard for the internal-event session-context pin.

``_pinned_session_context_prompt(..., preserve_pin=True)`` reuses the existing pin
verbatim so a turn that carries no fresh prompt identity (an internal event, or a
non-internal synthetic) cannot re-key it.  A helper-level unit test would stay
green if the call site in ``_handle_message_with_agent`` stopped forwarding the
predicate, so every case here drives the REAL handler or the REAL merge/admission
boundary and asserts on the prompt bytes that reach ``_run_agent`` / the queue.

On top of the pre-existing call-site cases the kernel adds exactly TWO test
functions, one per invariant, both table-driven so a new entrypoint is a new table
row rather than a new test:

* ``test_non_internal_synthetic_event_preserves_all_prompt_pins`` — prompt identity
  and turn OWNERSHIP through every entrypoint: a turn without a fresh identity reuses
  the session's pins, a turn that owns one is never coalesced into a foreign head, and
  a refusal nothing retries is reported instead of vanishing on a debug line.
* ``test_first_non_internal_synthetic_after_restart_rehydrates_prompt_pins`` —
  restart and teardown recovery: pins rehydrate from the durable snapshot, a turn
  retained in the debounce buffer survives teardown, and a failing spool is visible.
"""

from __future__ import annotations

import json
import logging
import sys
import types
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, TextDebounceState
from gateway.platforms.event import MessageType
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionEntry, SessionSource
from gateway.session_identity import RoutingIdentity
from gateway.turn_context import TurnContext

KEY = "agent:main:discord:group:1513247605675790346:117431298246705156"
PARENT_ID = "1513247605675790346"
THREAD_ID = "1552000000000000001"
_ORIGIN = dict(
    platform=Platform.DISCORD,
    chat_id="1513247605675790346",
    chat_type="group",
    user_id="117431298246705156",
    scope_id="1480524732964278294",
)
# A refusal or a spool failure must be reported above debug on this logger.
_BASE_LOGGER = "gateway.platforms.base"


# ── sources and config ─────────────────────────────────────────────────────
# ``_push_wake`` rebuilds a source from the persisted origin, so a wake carries no
# chat_name / user_name / message_id / parent_chat_id: it cannot re-render the human
# turn's prompt bytes, which is exactly why it must reuse the pin.

def _human_source() -> SessionSource:
    return SessionSource(
        **_ORIGIN,
        chat_name="Guild / #general",
        user_name="Ace",
        message_id="1552671843494666330",
    )


def _wake_source() -> SessionSource:
    return SessionSource(**_ORIGIN)


def _other_sender_source() -> SessionSource:
    """A DIFFERENT sender, so a same-sender merge is never what is under test."""
    return SessionSource(
        **{**_ORIGIN, "user_id": "999888777"},
        chat_name="Someone else", user_name="Other", message_id="O1",
    )


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
    return SessionSource(**_thread_origin())


def _pinned_config() -> GatewayConfig:
    """Discord with a parent-channel override, so the channel ephemeral components differ
    from the context prompt and a flip between them is visible."""
    config = GatewayConfig()
    config.platforms[Platform.DISCORD] = PlatformConfig(
        enabled=True,
        channel_overrides={PARENT_ID: ChannelOverride(system_prompt="Parent persona.")},
    )
    return config


def _provenance_source() -> SessionSource:
    """A source carrying the wire-invisible routing provenance ``replace_source`` copies."""
    source = _human_source()
    source._identity = RoutingIdentity(
        transport_profile="default", runtime_profile="default",
        authorization_home=Path("/auth"), runtime_home=Path("/run"),
    )
    source._transport_adapter_ref = object()
    source._authorization_profile_home = Path("/auth")
    return source


# ── harness ────────────────────────────────────────────────────────────────

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
    """Push ``(internal, source)`` turns through the real handler on one session."""
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


async def _send(runner, event: MessageEvent) -> None:
    """Send one already-built event through the real handler."""
    await runner._handle_message_with_agent(event, event.source, KEY, 1)


def _effective_ephemeral(runner, kw) -> str:
    """The real TurnRunner combiner applied to what reached ``_run_agent``."""
    ctx = TurnContext(source=kw["source"], context_prompt=kw["context_prompt"], channel_prompt=kw.get("channel_prompt"))
    return TurnRunner(runner, ctx)._combined_ephemeral_prompt()


def _ephemeral_prompts(runner, calls: list) -> list[str]:
    return [_effective_ephemeral(runner, kw) for kw in calls]


def _queued_events(runner) -> list:
    return runner._peek_session_state(KEY).conversation.queued_events


class _Adapter(BasePlatformAdapter):
    """Minimal real adapter for the merge-boundary assertions (no transport)."""

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, text, **kwargs):
        pass

    async def get_chat_info(self, chat_id):
        return {}


def _merge_adapter(runner) -> _Adapter:
    """An adapter wired to *runner* as its own delivery adapter — the merge boundary."""
    adapter = _Adapter(PlatformConfig(enabled=True), Platform.DISCORD)
    adapter.gateway_runner = runner
    adapter._busy_text_mode = "queue"
    runner._delivery_adapter_for = lambda src: adapter
    return adapter


def _merge_boundary(monkeypatch, *, busy_text_mode: str = "queue") -> tuple:
    """A pinned-config runner and its own delivery adapter: the merge boundary under test."""
    runner = _make_runner(monkeypatch, _pinned_config())
    adapter = _merge_adapter(runner)
    adapter._busy_text_mode = busy_text_mode
    return runner, adapter


def _pinning_boundary(monkeypatch) -> tuple:
    """A pinned-config runner plus a call sink, with the turn-tail hooks stubbed so a follow-up
    can be driven without the real delivery tail."""
    runner = _make_runner(monkeypatch, _pinned_config())
    calls: list[dict] = []
    _capture(runner, calls)
    runner._run_agent_deliver_first_response = AsyncMock()
    runner._refresh_agent_cache_message_count = AsyncMock()
    return runner, calls


def _buffer_preserving_head(runner, adapter: _Adapter) -> MessageEvent:
    """Park a preserving head in the debounce buffer behind a foreign slot occupant.

    The two differ in prompt identity, which is what makes the merge boundary the one under
    test, and the buffer is the only durable home a refused turn has.
    """
    head = runner._synthetic_prompt_event(_human_source(), "[goal] buffered head")
    adapter._pending_messages[KEY] = MessageEvent(
        text="other sender occupant", source=_other_sender_source(), message_id="O1")
    adapter._text_debounce_store()[KEY] = TextDebounceState(
        event=head, task=None, first_ts=0.0, last_ts=0.0)
    return head


def _cap_turn_ctx(source, *, depth: int) -> TurnContext:
    """TurnContext shaped for the real recursion-cap branch of _run_agent_queued_followup."""
    ctx = TurnContext(source=source, context_prompt="ctx", channel_prompt=None,
                      session_key=KEY, session_id="sess-cap", run_generation=1)
    ctx._interrupt_depth = depth
    ctx._status_thread_metadata = None
    ctx.result_holder = [None]
    ctx.history = []
    return ctx


async def _drive_recursion_cap(runner, adapter, source, *, media):
    """Drain-ordering where the human turn is dequeued and a synthetic is staged in the slot.

    ``pending_event`` is ALREADY ACCEPTED work at this point (the drain popped it out of a
    queued slot), so the cap branch must give it a slot of its own rather than merge it.
    """
    human = MessageEvent(
        text="human photo caption" if media else "human text turn",
        source=source, message_id="H-cap",
        message_type=MessageType.PHOTO if media else MessageType.TEXT,
        media_urls=["/tmp/cap.png"] if media else [],
        media_types=["image/png"] if media else [],
    )
    human._gateway_accepted = True
    staged = runner._synthetic_prompt_event(source, "[goal] resume continuation")
    adapter._pending_messages[KEY] = staged
    await runner._run_agent_queued_followup(
        _cap_turn_ctx(source, depth=runner._MAX_INTERRUPT_DEPTH), adapter, human.text, human,
        "prev-response", {"interrupted": True, "messages": []}, None)
    return human, staged


# ═══════════════════════════════════════════════════════════════════════════
# Pre-existing call-site cases
# ═══════════════════════════════════════════════════════════════════════════

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


@pytest.mark.asyncio
async def test_internal_event_keeps_channel_prompt_and_parent_override(monkeypatch):
    runner = _make_runner(monkeypatch, _pinned_config())
    calls: list[dict] = []
    _capture(runner, calls)

    await _drive(
        runner,
        ((False, _human_thread_source()), (True, _wake_thread_source()), (False, _human_thread_source())),
        channel_prompt="Channel hint.",
    )

    assert len(calls) == 3
    # The other ephemeral prompt components must not toggle across the sequence either.
    eph = _ephemeral_prompts(runner, calls)
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
    runner, calls = _pinning_boundary(monkeypatch)

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
    ephemeral = _ephemeral_prompts(runner, calls)
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
    config = _pinned_config()
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
    runner = _make_runner(monkeypatch, _pinned_config(), durable_prompt_pin=durable)
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


# ═══════════════════════════════════════════════════════════════════════════
# INVARIANT 1 — prompt identity + turn ownership at every entrypoint
# ═══════════════════════════════════════════════════════════════════════════

async def _owns_builder_contract(monkeypatch, caplog) -> None:
    """The REAL builder marks the event preserving (internal stays False), clears the reply
    anchor, and copies routing provenance through session_identity.replace_source — a plain
    dataclasses.replace drops _identity / adapter ref / authorization home."""
    runner = _make_runner(monkeypatch, _pinned_config())
    origin = _provenance_source()
    synthetic = runner._synthetic_prompt_event(origin, "[goal] keep going")

    assert synthetic.internal is False
    assert synthetic.preserve_prompt_pins is True
    assert synthetic.message_id is None and synthetic.source.message_id is None
    for attr in ("_identity", "_transport_adapter_ref", "_authorization_profile_home"):
        assert getattr(synthetic.source, attr) is getattr(origin, attr), \
            "builder dropped routing provenance"


async def _owns_handler_prompt_bytes(monkeypatch, caplog) -> None:
    """human -> non-internal synthetic -> human through the REAL handler on ONE session: the
    effective ephemeral bytes must not flip (the synthetic reuses the human turn's pins)."""
    runner = _make_runner(monkeypatch, _pinned_config())
    calls: list[dict] = []
    _capture(runner, calls)
    human = _human_thread_source()

    def human_turn() -> MessageEvent:
        return MessageEvent(text="hi", source=human, message_id=human.message_id,
                            channel_prompt="Channel hint.")

    await _send(runner, human_turn())
    await _send(runner, runner._synthetic_prompt_event(_wake_thread_source(), "[goal] keep going"))
    await _send(runner, human_turn())

    assert len(calls) == 3, f"_run_agent reached {len(calls)}/3 turns"
    ephemeral = _ephemeral_prompts(runner, calls)
    assert ephemeral[0] == ephemeral[1] == ephemeral[2], \
        "non-internal synthetic turn re-keyed the pinned prompt"
    assert all(call["channel_prompt"] == "Channel hint." for call in calls)


async def _owns_overwrite_negative_control(monkeypatch, caplog) -> None:
    """Negative control (#124179): a TRUE human event with channel_prompt=None MUST still
    overwrite the pin — the flag must not make every non-internal turn preserving."""
    runner = _make_runner(monkeypatch, _pinned_config())
    calls: list[dict] = []
    _capture(runner, calls)

    await _drive(runner, ((False, _human_thread_source()),), channel_prompt="Channel hint.")
    await _drive(runner, ((False, _human_thread_source()),), channel_prompt=None)

    assert calls[0]["channel_prompt"] == "Channel hint."
    assert calls[1]["channel_prompt"] is None, "human turn inherited a stale channel pin"
    assert runner._peek_session_state(KEY).conversation.channel_pin == (None, PARENT_ID)


async def _owns_refused_merge_runs_its_own_turn(monkeypatch, caplog, *, media: bool) -> None:
    """Same-sender human input must NOT coalesce into a preserving synthetic head, and must not
    be lost: it runs as its own queued turn. A PHOTO pair additionally carries media that must
    not be absorbed into the synthetic's pinned identity."""
    runner, adapter = _merge_boundary(monkeypatch)
    source = _human_source()
    head = runner._synthetic_prompt_event(source, "[goal] continue")
    adapter._pending_messages[KEY] = head

    if media:
        human = MessageEvent(text="caption", source=source, message_id="m-photo",
                             message_type=MessageType.PHOTO, media_urls=["/tmp/p.png"],
                             media_types=["image/png"])
        runner._queue_or_replace_pending_event(KEY, human)
        assert head.media_urls == [], "human photo absorbed into the preserving synthetic head"
    else:
        human = MessageEvent(text="a real question", source=source, message_id="m-human")
        assert adapter._is_queue_text_debounce_candidate(human) is True
        await adapter._queue_text_debounce(KEY, human)
        await adapter._flush_text_debounce_now(KEY)
        assert head.text == "[goal] continue", "human text coalesced into the preserving synthetic head"

    assert adapter._pending_messages[KEY] is head
    assert human in _queued_events(runner), "refused merge dropped the human follow-up"
    assert human._gateway_accepted is True


async def _owns_recursion_cap_restage(monkeypatch, caplog, *, media: bool) -> None:
    """The recursion cap must not merge already-accepted drained work into the slot occupant.

    Before the guard, a TEXT pair silently destroyed the staged occupant and a PHOTO pair
    absorbed the human turn's media into the synthetic's pinned identity with a ``None`` reply
    anchor. The occupant keeps the slot and the accepted turn gets its own FIFO slot, so the
    next promotion runs both.
    """
    runner, adapter = _merge_boundary(monkeypatch)

    human, staged = await _drive_recursion_cap(runner, adapter, _human_source(), media=media)

    assert adapter._pending_messages[KEY] is staged, "cap merge destroyed the staged occupant"
    assert staged.text == "[goal] resume continuation"
    assert staged.media_urls == [], "human media absorbed into the preserving synthetic head"
    assert human in _queued_events(runner), "already-accepted drained turn owns no queued slot"
    assert human._gateway_accepted is True
    # The accepted turn keeps its own prompt identity and reply anchor.
    assert human.preserve_prompt_pins is False
    assert human.message_id == "H-cap"
    assert human.media_urls == (["/tmp/cap.png"] if media else [])


async def _owns_recursion_cap_same_identity_control(monkeypatch, caplog) -> None:
    """Control: with no identity difference the cap keeps its original merge behaviour."""
    runner, adapter = _merge_boundary(monkeypatch)
    source = _human_source()

    normal = MessageEvent(text="normal follow-up", source=source, message_id="N1")
    normal._gateway_accepted = True
    adapter._pending_messages[KEY] = normal

    drained = MessageEvent(text="drained human turn", source=source, message_id="H2")
    drained._gateway_accepted = True
    await runner._run_agent_queued_followup(
        _cap_turn_ctx(source, depth=runner._MAX_INTERRUPT_DEPTH), adapter, drained.text, drained,
        "prev-response", {"interrupted": True, "messages": []}, None)

    assert adapter._pending_messages[KEY] is drained, "same-identity cap merge behaviour changed"


async def _owns_refused_debounce_arrival_is_reported(monkeypatch, caplog) -> None:
    """A refused differing-identity debounce arrival leaves the BUFFER intact and says so.

    The arrival is a fresh input that owns no slot yet, so the fresh-input cap may legitimately
    leave it unaccepted; what must hold is that the already-buffered turn keeps its place and
    the refusal is visible rather than a silent debug line.
    """
    runner, adapter = _merge_boundary(monkeypatch)
    head = _buffer_preserving_head(runner, adapter)
    # Real refusal receipt: the queue is at cap, so admission returns False.
    runner._queue_or_replace_pending_event = lambda key, ev: False

    arrival = MessageEvent(text="my real question", source=_human_source(), message_id="I1")
    assert adapter._is_queue_text_debounce_candidate(arrival) is True
    await adapter._queue_text_debounce(KEY, arrival)

    store = adapter._text_debounce_store()
    assert store.get(KEY) is not None and store[KEY].event is head, \
        "refusal must retain the already-buffered head for a later flush"
    assert head.text == "[goal] buffered head"
    assert adapter._pending_messages[KEY].text == "other sender occupant"
    assert arrival._gateway_accepted is False, "a refused arrival is unaccepted, not merged"
    assert "differing-identity" in caplog.text and "Dropped" in caplog.text, \
        "a dropped human follow-up must be reported above debug"


async def _owns_adapter_fallback_refusal_is_reported(monkeypatch, caplog) -> None:
    """A follow-up that arrives while the session is busy: the adapter-only merge is the only
    owner, so its refusal is a drop nothing retries and must be reported."""
    runner, adapter = _merge_boundary(monkeypatch, busy_text_mode="interrupt")  # not a debounce candidate
    source = _human_source()
    head = runner._synthetic_prompt_event(source, "[goal] continue")
    adapter._pending_messages[KEY] = head

    await adapter._handle_message_while_active(
        MessageEvent(text="a real question", source=source, message_id="m1"), KEY)

    assert adapter._pending_messages[KEY] is head
    assert head.text == "[goal] continue", "human text coalesced into the preserving synthetic head"
    assert "differing-identity" in caplog.text and "Dropped" in caplog.text, \
        "a drop nothing retries must be reported above debug"


# One row per entrypoint under the invariant; a new ingress path is a new row. The ``-text`` /
# ``-photo`` pair is the same contract over a different payload shape, which is where the old
# merge guard lost the human turn's media. ``None`` means the case takes no ``media`` flag.
_IDENTITY_OWNERSHIP_CASES = {
    "builder-contract": (_owns_builder_contract, None),
    "handler-prompt-bytes": (_owns_handler_prompt_bytes, None),
    "overwrite-negative-control": (_owns_overwrite_negative_control, None),
    "refused-merge-text": (_owns_refused_merge_runs_its_own_turn, False),
    "refused-merge-photo": (_owns_refused_merge_runs_its_own_turn, True),
    "recursion-cap-restage-text": (_owns_recursion_cap_restage, False),
    "recursion-cap-restage-photo": (_owns_recursion_cap_restage, True),
    "recursion-cap-same-identity": (_owns_recursion_cap_same_identity_control, None),
    "refused-debounce-arrival": (_owns_refused_debounce_arrival_is_reported, None),
    "adapter-fallback-refusal": (_owns_adapter_fallback_refusal_is_reported, None),
}
_IDENTITY_OWNERSHIP_CASE_IDS = (
    "builder-contract",
    "handler-prompt-bytes",
    "overwrite-negative-control",
    "refused-merge-text",
    "refused-merge-photo",
    "recursion-cap-restage-text",
    "recursion-cap-restage-photo",
    "recursion-cap-same-identity",
    "refused-debounce-arrival",
    "adapter-fallback-refusal",
)
assert tuple(_IDENTITY_OWNERSHIP_CASES) == _IDENTITY_OWNERSHIP_CASE_IDS, "case table drifted from its ids"


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _IDENTITY_OWNERSHIP_CASE_IDS)
async def test_non_internal_synthetic_event_preserves_all_prompt_pins(monkeypatch, caplog, case):
    """INVARIANT 1 — a turn with no fresh prompt identity reuses the session's pins through
    every entrypoint, and a turn that owns one is never coalesced into — or lost with — a
    foreign head. Table-driven: one row per entrypoint, every row on a real handler, merge
    boundary, or admission receipt."""
    case_fn, media = _IDENTITY_OWNERSHIP_CASES[case]
    with caplog.at_level(logging.WARNING, logger=_BASE_LOGGER):
        if media is None:
            await case_fn(monkeypatch, caplog)
        else:
            await case_fn(monkeypatch, caplog, media=media)


# ═══════════════════════════════════════════════════════════════════════════
# INVARIANT 2 — restart + teardown recovery
# ═══════════════════════════════════════════════════════════════════════════

async def _rehydrates_cold_preserving_turn(monkeypatch, caplog, tmp_path) -> None:
    """A restart drops in-memory pins; the first event carries no fresh prompt identity, so it
    must rehydrate the durable snapshot rather than re-render (or publish) its own."""
    config = _pinned_config()
    durable: dict = {}

    before = _make_runner(monkeypatch, config, durable_prompt_pin=durable)
    calls_before: list[dict] = []
    _capture(before, calls_before)
    await _drive(before, ((False, _human_thread_source()),), channel_prompt="Channel hint.")

    snapshot = durable.get("value")
    assert isinstance(snapshot, dict)
    assert snapshot["channel_prompt"] == "Channel hint." and snapshot["parent_chat_id"] == PARENT_ID

    after = _make_runner(monkeypatch, config, durable_prompt_pin=durable)
    calls_after: list[dict] = []
    _capture(after, calls_after)
    synth = after._synthetic_prompt_event(_wake_thread_source(), "[goal] keep going")
    assert synth.internal is False and synth.preserve_prompt_pins is True
    await _send(after, synth)

    # The preserving turn must NOT publish synthetic (None, parent) inputs: inspect the
    # durable snapshot BEFORE any human repair turn.
    assert durable["value"] == snapshot, "cold preserving turn poisoned the durable prompt pin"
    assert after._peek_session_state(KEY).conversation.ephemeral_pin is not None
    assert after._peek_session_state(KEY).conversation.channel_pin is not None

    await _drive(after, ((False, _human_thread_source()),), channel_prompt="Channel hint.")

    human, first_synthetic, next_human = calls_before[0], calls_after[0], calls_after[1]
    assert first_synthetic["context_prompt"] == human["context_prompt"] == next_human["context_prompt"]
    assert first_synthetic["channel_prompt"] == human["channel_prompt"] == next_human["channel_prompt"]
    assert first_synthetic["source"].parent_chat_id == human["source"].parent_chat_id == PARENT_ID
    assert _effective_ephemeral(after, first_synthetic) == _effective_ephemeral(before, human)


async def _spools_retained_turn_on_teardown(monkeypatch, caplog, tmp_path) -> None:
    """A refused merge leaves its turn buffered; teardown must spool it, not clear it away."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)

    runner, adapter = _merge_boundary(monkeypatch)
    head = _buffer_preserving_head(runner, adapter)

    await adapter.cancel_background_tasks()

    spool = sorted((tmp_path / "pending_messages").glob("*.json"))
    assert spool, "adapter teardown dropped the retained debounce buffer with nothing durable"
    texts = [json.loads(p.read_text(encoding="utf-8-sig"))["data"]["text"] for p in spool]
    assert head.text in texts, "the retained turn is not recoverable from the shutdown spool"
    assert not adapter._text_debounce_store()


async def _reports_failed_teardown_spool(monkeypatch, caplog, tmp_path) -> None:
    """A spool that fails during teardown is best-effort, not silent: the loss is reported with
    the safe failure class, teardown still completes, and no secret or path is logged."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import gateway.shutdown_flush as sf

    secret = "sk-live-DO-NOT-LOG-1234"
    monkeypatch.setattr(sf, "flush_pending_to_file", MagicMock(
        side_effect=OSError(f"{secret} at {tmp_path / 'pending_messages'}")))

    runner, adapter = _merge_boundary(monkeypatch)
    head = _buffer_preserving_head(runner, adapter)

    with caplog.at_level(logging.WARNING, logger=_BASE_LOGGER):
        await adapter.cancel_background_tasks()

    assert "Shutdown spool" in caplog.text, "a failed teardown spool must be reported"
    assert "OSError" in caplog.text, "the safe failure class must be named for diagnosis"
    assert "1 debounced turn(s) failed" in caplog.text, "the loss must state its own scope"
    assert secret not in caplog.text, "raw error payload must not be logged"
    assert str(tmp_path) not in caplog.text, "filesystem path must not be logged"
    assert head.text not in caplog.text, "a turn's text is not a safe log field"
    # Bounded cleanup: the failure is reported, not raised, so the buckets are still cleared.
    assert not adapter._text_debounce_store()
    assert not adapter._pending_messages


_RECOVERY_CASES = {
    "cold-preserving-restart": _rehydrates_cold_preserving_turn,
    "retained-turn-teardown-spool": _spools_retained_turn_on_teardown,
    "failed-teardown-spool-reported": _reports_failed_teardown_spool,
}
_RECOVERY_CASE_IDS = (
    "cold-preserving-restart",
    "retained-turn-teardown-spool",
    "failed-teardown-spool-reported",
)
assert tuple(_RECOVERY_CASES) == _RECOVERY_CASE_IDS, "recovery table drifted from its ids"


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _RECOVERY_CASE_IDS)
async def test_first_non_internal_synthetic_after_restart_rehydrates_prompt_pins(
    monkeypatch, caplog, tmp_path, case,
):
    """INVARIANT 2 — one durability contract on both sides of a process boundary: a restart
    rehydrates prompt identity from the durable snapshot, and a turn retained in the debounce
    buffer survives teardown durably, with any spool failure observable."""
    case_fn = _RECOVERY_CASES[case]
    with caplog.at_level(logging.WARNING, logger=_BASE_LOGGER):
        await case_fn(monkeypatch, caplog, tmp_path)


# ---------------------------------------------------------------------------
# #131294 — a channel_overrides model is the session's CONFIGURED model, not a
# fallback.  The post-turn ``_run_agent_evict_on_fallback`` check baselined
# against the global model only, so every successful turn in an overridden chat
# evicted the cached agent (and cleared its ephemeral pin).  These drive the
# REAL ``_run_agent`` / ``TurnRunner.run_sync`` path (only the AIAgent class and
# config reads are substituted), not the ``_capture`` stub above.
# ---------------------------------------------------------------------------

_WELCOME = "https://welcome-api.nousresearch.com/v1"


class _TurnAgent:
    """AIAgent stand-in: keeps the resolved route, answers without an API call."""

    instances = 0

    def __init__(self, *args, **kwargs):
        type(self).instances += 1
        self.model = kwargs.get("model")
        self.provider = kwargs.get("provider")
        self.base_url = kwargs.get("base_url")
        self.session_id = kwargs.get("session_id")
        self.tools = []
        self.request_overrides = dict(kwargs.get("request_overrides") or {})
        # Mirror AIAgent.__init__: the welcome host pins its one model, and the primary
        # route is snapshotted for the post-turn fallback classifier.
        from hermes_cli.anon_auth import pin_model_for_route
        self.model = pin_model_for_route(self.provider, self.base_url, self.model)
        self._primary_runtime = {"model": self.model, "provider": self.provider, "base_url": self.base_url}

    def run_conversation(self, user_message, conversation_history=None, task_id=None, **kwargs):
        return {
            "final_response": "ok",
            "messages": (conversation_history or []) + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": "ok"},
            ],
            "api_calls": 1,
        }


@pytest.mark.parametrize(
    "runtime, global_model, expected_model",
    [
        ({"api_key": "fake", "provider": "openrouter"}, "default/model", "chan/model"),
        ({"api_key": "fake", "provider": "nous", "base_url": _WELCOME}, "nous/welcome", "nous/welcome"),
    ],
    ids=["named-channel-model", "welcome-host-pins-over-channel-model"],
)
@pytest.mark.asyncio
async def test_channel_override_turns_keep_one_cached_agent_and_the_pin(
    monkeypatch, runtime, global_model, expected_model,
):
    import gateway.run as gr

    config = GatewayConfig()
    config.platforms[Platform.DISCORD] = PlatformConfig(
        enabled=True, channel_overrides={_ORIGIN["chat_id"]: ChannelOverride(model="chan/model")},
    )
    monkeypatch.setattr(gr, "_load_gateway_config", lambda: {"model": {"default": global_model}})
    monkeypatch.setattr(gr, "_resolve_gateway_model", lambda config=None: global_model)
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _TurnAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    _TurnAgent.instances = 0

    runner = _make_runner(monkeypatch, config)
    monkeypatch.setattr(gr, "_resolve_runtime_agent_kwargs", lambda: dict(runtime))
    runner.session_store._entries = {}

    await _drive(runner, ((False, _human_source()), (True, _wake_source()), (False, _human_source())))

    entry = runner._agent_cache.get(KEY)
    assert entry is not None, "channel-override turn evicted the session's cached agent"
    assert entry[0].model == expected_model
    assert _TurnAgent.instances == 1, f"agent rebuilt {_TurnAgent.instances - 1}x across 3 turns"
    assert runner._peek_session_state(KEY).conversation.ephemeral_pin is not None, "context pin cleared"


@pytest.mark.asyncio
async def test_draining_turn_leaves_accepted_followups_for_shutdown_spool(monkeypatch):
    """#126167 review F4 — a finishing agent turn must not consume accepted work while a
    restart/shutdown waits: ``_draining`` teardown snapshots the adapter's pending and
    overflow buckets for durable recovery, so dequeueing them here (and then discarding
    under the drain guard) loses events the spool never saw. Admitted A and B must survive
    the drain intact, in place, in order, with their prompt identity."""
    from gateway.run import _build_media_placeholder  # noqa: F401  (import shape check)

    runner, adapter = _merge_boundary(monkeypatch)
    source = _human_source()
    a = MessageEvent(text="accepted-first", source=source, message_id="A1")
    b = MessageEvent(text="accepted-second", source=_other_sender_source(), message_id="B1")
    adapter._pending_messages[KEY] = a
    runner._session_state(KEY).conversation.queued_events.append(b)

    runner._draining = True
    result = {"final_response": "done", "messages": []}
    pending_event, pending = await runner._run_agent_drain_pending(result, adapter, source, KEY)

    assert pending_event is None and pending is None
    # Both accepted events survive, in place, in FIFO order, for the shutdown spool.
    assert adapter._pending_messages[KEY] is a
    assert _queued_events(runner) == [b]

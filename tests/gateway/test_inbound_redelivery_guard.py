"""A redelivered platform update must not start a second turn (#inbound-dedupe).

Live symptom (2026-09-18, Telegram DM): the same message was ingested twice —
18:17:47 and 18:24:36 — and each ingest started its own turn on the same
session. The first ran 374s and committed work; the second re-did the same
investigation. The gateway logged the fallout but not the cause::

    gateway.run: inbound message: ... msg='Step 4 is done'      (18:17:47)
    gateway.run: inbound message: ... msg='Step 4 is done'      (18:24:36)
    gateway.run: Agent cache invalidated for session ...:
        message_count changed (234 -> 169), possible cross-process write
    agent.relay_runtime: Skipping Relay instrumentation for concurrent
        Hermes turn ... in session ...

Nothing at the turn boundary asked "have I already consumed this update?".
The one existing guard (``run_turn.py`` transient-failure dedupe) is (a) keyed
on ``message_id`` and (b) consulted only when the previous turn FAILED EARLY —
so a successful turn left no marker, and a redelivery sailed through.

These tests pin the contract at the ingress gate: one update id, one turn.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _config() -> GatewayConfig:
    return GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="x")}
    )


def _source(chat_id: str = "6513506906") -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="dm",
        user_id=chat_id,
    )


def _event(text: str, *, update_id, message_id: int, source: SessionSource) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id=str(message_id),
        platform_update_id=update_id,
    )


def _gate_runner(store) -> GatewayRunner:
    """A runner with just the ingress gate stack wired, using the real methods.

    Follows the ``object.__new__`` + ``__get__`` idiom the gateway tests use:
    the bare runner has no ``__init__``, so every attribute the gates touch is
    supplied here rather than mocked away — the point is to exercise the real
    ``_hm_admit_event`` ordering.
    """
    runner = object.__new__(GatewayRunner)
    runner.config = _config()
    runner.session_store = store
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner._background_tasks = set()
    # Real gate implementation under test.
    runner._hm_admit_event = GatewayRunner._hm_admit_event.__get__(runner)
    # Deterministic collaborators; each is out of scope for this contract.
    runner._session_key_for_source = lambda source: f"k:{source.chat_id}"
    runner._scale_to_zero_note_real_inbound = MagicMock()
    runner._hm_pre_gateway_dispatch_hook = lambda event, source: event
    runner._is_user_authorized_for_source = lambda source, **kw: True
    runner._admit_bot_message_for_source = lambda source: True
    return runner


def _store(tmp_path):
    return MagicMock()  # the ingress gate does not touch the store


@pytest.mark.asyncio
async def test_redelivered_update_id_is_dropped(tmp_path):
    """The live symptom: the same update id twice -> admitted once.

    A redelivery is a NEW ``MessageEvent`` built from the re-sent ``Update`` (the
    adapter always constructs one per delivery), whereas a drained follow-up re-admits
    the SAME object. That identity difference is the discriminator the guard relies on,
    so this test must build two distinct objects carrying one update id — reusing the
    object would silently test the drain path instead.
    """
    runner = _gate_runner(_store(tmp_path))
    src = _source()

    first = await runner._hm_admit_event(
        _event("Step 4 is done", update_id=1001, message_id=555, source=src)
    )
    assert first is not None, "the first delivery must be admitted"

    # The re-sent Update arrives as a second event object with the same update id.
    redelivered = await runner._hm_admit_event(
        _event("Step 4 is done", update_id=1001, message_id=555, source=src)
    )
    assert redelivered is None, (
        "a redelivered update id must be dropped at ingress — admitting it starts "
        "a second concurrent turn on the same session"
    )


@pytest.mark.asyncio
async def test_distinct_update_ids_are_both_admitted(tmp_path):
    """Control: ordinary consecutive messages are unaffected."""
    runner = _gate_runner(_store(tmp_path))
    src = _source()

    first = await runner._hm_admit_event(
        _event("first message", update_id=2001, message_id=601, source=src)
    )
    second = await runner._hm_admit_event(
        _event("second message", update_id=2002, message_id=602, source=src)
    )

    assert first is not None
    assert second is not None, "dedupe must key on update_id, not on session"


@pytest.mark.asyncio
async def test_edited_message_is_admitted(tmp_path):
    """The trap: an EDIT reuses message_id but carries a NEW update_id.

    Keying the guard on message_id (as the existing transient-failure guard
    does) would silently swallow every Telegram edit. This test fails if the
    key is ever changed to message_id.
    """
    runner = _gate_runner(_store(tmp_path))
    src = _source()

    original = await runner._hm_admit_event(
        _event("initial text", update_id=3001, message_id=700, source=src)
    )
    edited = await runner._hm_admit_event(
        _event("initial text, corrected", update_id=3002, message_id=700, source=src)
    )

    assert original is not None
    assert edited is not None, (
        "an edited message reuses message_id with a new update_id and MUST be "
        "admitted — the guard keys on update_id for exactly this reason"
    )


@pytest.mark.asyncio
async def test_absent_update_id_is_admitted(tmp_path):
    """Non-Telegram platforms populate no update_id; behaviour must not change."""
    runner = _gate_runner(_store(tmp_path))
    src = _source()

    event = _event("no platform update id", update_id=None, message_id=800, source=src)

    assert await runner._hm_admit_event(event) is not None
    assert await runner._hm_admit_event(event) is not None, (
        "with no platform_update_id the guard must be inert, not drop everything"
    )


@pytest.mark.asyncio
async def test_dedupe_state_is_per_session(tmp_path):
    """Two chats may legitimately carry unrelated update ids."""
    runner = _gate_runner(_store(tmp_path))

    a = await runner._hm_admit_event(
        _event("chat a", update_id=4001, message_id=901, source=_source("111"))
    )
    b = await runner._hm_admit_event(
        _event("chat b", update_id=4001, message_id=902, source=_source("222"))
    )

    assert a is not None
    assert b is not None, "update ids are only unique per bot, not per chat key"


@pytest.mark.asyncio
async def test_queued_followup_drain_is_admitted(tmp_path):
    """The feature this guard must NOT break.

    When a message arrives while a turn is running, the base adapter queues it and
    later re-dispatches the SAME event object (``platforms/base.py`` →
    ``_process_message_background``), which is a second trip through this gate for one
    update id. Keying the guard on the update id alone would silently swallow every
    queued follow-up and interrupt — the failure mode this test exists to catch.
    """
    runner = _gate_runner(_store(tmp_path))
    src = _source()
    event = _event("follow-up while running", update_id=5001, message_id=1001, source=src)

    on_arrival = await runner._hm_admit_event(event)
    assert on_arrival is not None, "the message is admitted (and charged) on arrival"

    drained = await runner._hm_admit_event(event)
    assert drained is not None, (
        "a drained follow-up re-admits the SAME event object and must still be "
        "admitted — otherwise this guard destroys the feature it protects"
    )


def test_ledger_forgets_old_ids_within_its_bound():
    """Bounded, not unbounded: the oldest id is evicted, so the map cannot grow forever."""
    from gateway.run_inbound import PlatformRedeliveryLedger

    ledger = PlatformRedeliveryLedger(ttl=10_000.0, per_session=2, max_sessions=1)

    assert ledger.is_redelivery("k", 1, now=0.0) is False
    assert ledger.is_redelivery("k", 2, now=0.0) is False
    assert ledger.is_redelivery("k", 3, now=0.0) is False

    # Within the bound the recent ids still dedupe...
    assert ledger.is_redelivery("k", 3, now=0.0) is True
    # ...and the evicted one is forgotten rather than accumulated.
    assert ledger.is_redelivery("k", 1, now=0.0) is False


def test_ledger_expires_ids_after_the_ttl():
    """Time-based expiry too: a restarted bot reusing an id is not blocked forever."""
    from gateway.run_inbound import PlatformRedeliveryLedger

    ledger = PlatformRedeliveryLedger(ttl=100.0, per_session=8, max_sessions=4)

    assert ledger.is_redelivery("k", 7, now=0.0) is False
    assert ledger.is_redelivery("k", 7, now=50.0) is True
    assert ledger.is_redelivery("k", 7, now=500.0) is False, "expired: treated as new"


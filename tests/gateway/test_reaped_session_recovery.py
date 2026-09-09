"""Regression tests for the durable-reaped session guard in _handle_message (#99106).

A session whose durable row was ended in state.db (``ws_orphan_reap`` /
``agent_close``) while the gateway process stayed alive keeps its in-memory
turn slot (``_is_session_running`` stays True). Before the guard, the next
inbound message took the PRIORITY fast-path and was interrupt()-delivered into
the dead runtime — silently dropped, never reaching the
``get_or_create_session`` routing self-heal (#54878). The guard evicts the
stale slot at routing time so the message falls through to the cold path.

Salvaged from PR #99183 (@Finn763).
"""

import asyncio
import time
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import (
    GatewayConfig,
    Platform,
    PlatformConfig,
)
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import _AGENT_PENDING_SENTINEL, _INTERRUPT_REASON_EVICTED, GatewayRunner
from gateway.session import SessionSource, SessionStore
from gateway.turn_lease import SessionTurnLeaseRegistry


class _FakeAdapter:
    def __init__(self):
        self._pending_messages = {}
        self._active_sessions = {}

    async def send(self, *args, **kwargs):
        pass


class _DeadReapedAgent:
    """Runtime whose turn was reaped: interrupt() lands nowhere."""

    def __init__(self):
        self.interrupts = []

    def interrupt(self, text):
        self.interrupts.append(text)

    def get_activity_summary(self):
        # Recently active — the pre-existing idle-staleness eviction must NOT
        # fire, so only the durable-reaped guard can heal this shape.
        return {
            "seconds_since_activity": 0,
            "last_activity_desc": "tool",
            "api_call_count": 1,
            "max_iterations": 50,
        }


def _make_runner(store) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="x")}
    )
    runner.adapters = {Platform.TELEGRAM: _FakeAdapter()}
    runner._pending_messages = {}
    runner._voice_mode = {}
    runner._background_tasks = set()
    runner._draining = False
    runner._restart_requested = False
    runner._restart_task_started = False
    runner._restart_detached = False
    runner._restart_via_service = False
    runner._restart_drain_timeout = 0.0
    runner._stop_task = None
    runner._exit_code = None
    runner._update_runtime_status = MagicMock()
    runner._is_user_authorized = lambda _source: True
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.session_store = store
    runner.delivery_router = MagicMock()
    return runner


def _source(chat_id="555001") -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="dm",
        user_id=chat_id,
    )


def _store(tmp_path) -> SessionStore:
    config = GatewayConfig(

    )
    return SessionStore(sessions_dir=tmp_path, config=config)


def _occupy_turn_slot(runner, key, agent):
    state = runner._session_state(key)
    state.turn.agent = agent
    # Turn started 2 minutes ago — outside the Telegram follow-up grace
    # window, inside the idle-staleness threshold.
    state.turn.started_ts = time.time() - 120


@pytest.mark.asyncio
async def test_reaped_session_message_reaches_cold_path(tmp_path):
    """DB-ended session + live turn slot: next message must heal, not drop."""
    store = _store(tmp_path)
    src = _source()
    entry = store.get_or_create_session(src)
    key = store._generate_session_key(src)

    runner = _make_runner(store)
    agent = _DeadReapedAgent()
    _occupy_turn_slot(runner, key, agent)

    # The durable row is ended out from under the live slot (the reap).
    store._db.end_session(entry.session_id, "ws_orphan_reap")
    assert store._is_session_ended_in_db(entry.session_id) is True

    event = MessageEvent(
        text="hello, are you there?",
        message_type=MessageType.TEXT,
        source=src,
    )

    cold_path = AsyncMock(return_value="COLD_PATH_REPLY")
    with (
        patch.object(GatewayRunner, "_handle_message_with_agent", cold_path),
        patch.object(GatewayRunner, "_run_post_turn_hooks", AsyncMock()),
        patch.object(GatewayRunner, "_clear_durable_active_turn", AsyncMock()),
        patch.object(GatewayRunner, "_persist_active_agents", lambda self: None),
    ):
        result = await runner._handle_message(event)

    assert result == "COLD_PATH_REPLY"
    assert cold_path.await_count == 1
    # The user's message was never interrupt()-delivered into the dead runtime. The only thing
    # that reached it is the gateway's own stop request, so the orphaned run stops instead of
    # finishing invisibly after its result was discarded (#106963).
    assert agent.interrupts == [_INTERRUPT_REASON_EVICTED]


@pytest.mark.asyncio
async def test_alive_session_keeps_priority_interrupt_path(tmp_path):
    """Control: with the durable row still open, the PRIORITY interrupt
    fast-path must behave exactly as before (no eviction)."""
    store = _store(tmp_path)
    src = _source(chat_id="555002")
    store.get_or_create_session(src)
    key = store._generate_session_key(src)

    runner = _make_runner(store)
    agent = _DeadReapedAgent()
    _occupy_turn_slot(runner, key, agent)

    event = MessageEvent(
        text="follow-up while running",
        message_type=MessageType.TEXT,
        source=src,
    )

    cold_path = AsyncMock(return_value="COLD_PATH_REPLY")
    with (
        patch.object(GatewayRunner, "_handle_message_with_agent", cold_path),
        patch.object(GatewayRunner, "_agent_has_active_subagents", lambda self, a: False),
        patch.object(
            GatewayRunner,
            "_session_has_compression_in_flight",
            AsyncMock(return_value=False),
        ),
    ):
        result = await runner._handle_message(event)

    assert result is None
    assert cold_path.await_count == 0
    assert agent.interrupts == ["follow-up while running"]


@pytest.mark.asyncio
async def test_guard_is_inert_with_stubbed_session_store(tmp_path):
    """Bare test runners stub session_store with MagicMock; the guard must
    not evict (peek_session_id returns a Mock, not a str) and must not raise."""
    store = MagicMock()
    src = _source(chat_id="555003")

    runner = _make_runner(store)
    key = runner._session_key_for_source(src)
    agent = _DeadReapedAgent()
    _occupy_turn_slot(runner, key, agent)

    event = MessageEvent(
        text="stub store follow-up",
        message_type=MessageType.TEXT,
        source=src,
    )

    with (
        patch.object(GatewayRunner, "_agent_has_active_subagents", lambda self, a: False),
        patch.object(
            GatewayRunner,
            "_session_has_compression_in_flight",
            AsyncMock(return_value=False),
        ),
    ):
        result = await runner._handle_message(event)

    # Slot untouched; the message took the normal busy path.
    assert runner._is_session_running(key) is True
    assert result is None
    assert agent.interrupts == ["stub store follow-up"]


# ---------------------------------------------------------------------------
# Eviction vs. the evicted turn's own finalizer (#106966 review interleaving)
# ---------------------------------------------------------------------------


class _EvictableAgent(_DeadReapedAgent):
    """Reaped runtime that honours the hard interrupt: it records the reason and wakes the
    turn blocked inside it, the way an interrupted agent loop unwinds in production."""

    def __init__(self, interrupted: asyncio.Event):
        super().__init__()
        self.hard_interrupts = []
        self._interrupted = interrupted

    def hard_interrupt(self, message=None, **kwargs):
        self.hard_interrupts.append(message)
        self._interrupted.set()


def _event(text: str, src: SessionSource) -> MessageEvent:
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=src)


async def _run_eviction_interleaving(tmp_path, *, end_reason: str, moa: bool = False) -> SimpleNamespace:
    """Drive the review interleaving through the real ``_handle_message``:

    1. turn A runs on the session (real agent in the slot, real turn lease on its session id);
    2. the durable row is ended with ``end_reason``; message 2 evicts A (interrupt ->
       invalidate -> release) and claims the slot for a replacement turn, whose cold path
       self-heals the routing (reopening the row, or creating a fresh session id) and takes
       its own turn lease;
    3. the interrupted A unwinds and runs its ``_handle_message`` finalizer while the
       replacement is still running;
    4. a third message arrives while the replacement is still running.

    With ``moa`` both messages are ``/moa <prompt>`` one-shots (the real idle-path producer
    runs before the claim) and the session starts with a plain override to put back.

    Only the agent-turn body is stubbed: it installs the agent and, once interrupted, keeps
    unwinding only after the replacement has claimed the slot. That is the production timing:
    the interrupt is cooperative, the evicted run notices it when its model call or tool
    returns (seconds to minutes), while message 2 claims the slot within milliseconds.
    Admission, eviction, claim, routing heal, turn leases and the finalizer are production code."""
    store = _store(tmp_path)
    src = _source(chat_id="555004")
    entry = store.get_or_create_session(src)
    key = store._generate_session_key(src)
    runner = _make_runner(store)
    runner._turn_leases = SessionTurnLeaseRegistry()
    runner._session_model_overrides = {}
    if moa:
        runner._session_model_overrides[key] = {"provider": "openrouter", "model": "gpt-4"}

    interrupted, a_running, replacement_claimed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    agent_a = _EvictableAgent(interrupted)
    leases, turns, seen = [], [], SimpleNamespace()
    timeline = []  # ("override", phase, value) and ("cache_evict", key) in order
    runner._evict_cached_agent = lambda k: timeline.append(("cache_evict", k))

    def _note(phase):
        timeline.append(("override", phase, dict(runner._session_model_overrides.get(key) or {})))

    def acquire_slot(**kwargs):
        lease = MagicMock(name=f"lease-{len(leases)}")
        leases.append(lease)
        return lease, None

    async def agent_turn(self, event, source, quick_key, run_generation):
        turns.append(event.text)
        turn = self._session_state(quick_key).turn
        if event.text == "turn A":
            _note("A running")
            await self._hmwa_acquire_turn_lease(quick_key, run_generation, entry, None)
            seen.token_a = turn.lease_token
            turn.agent = agent_a
            a_running.set()
            await interrupted.wait()  # in the model/tool loop until the eviction interrupts it
            await replacement_claimed.wait()  # ...and still unwinding when the slot is re-claimed
            return {"final_response": "", "interrupted": True}
        if event.text != "message 2":
            return "UNEXPECTED_TURN"
        seen.b_model_restore = turn.model_restore  # what B will put back when IT ends
        _note("B running")
        healed = store.get_or_create_session(source)  # cold-path routing self-heal (#54878)
        seen.healed_session_id = healed.session_id
        if healed.session_id == entry.session_id:
            # Reopened row: this turn's lease waits for A's, so A must be allowed to unwind (and
            # run its finalizer against the already re-claimed slot) before the acquire returns.
            replacement_claimed.set()
            await self._hmwa_acquire_turn_lease(quick_key, run_generation, healed, None)
        else:
            # Fresh id: take this turn's lease first, overwriting the slot's token while A still
            # holds its own — the overwrite the finalizer's identity-based release exists for.
            await self._hmwa_acquire_turn_lease(quick_key, run_generation, healed, None)
            replacement_claimed.set()
        seen.token_b = turn.lease_token
        await task_a  # A has unwound completely: its finalizer ran, mid-replacement
        _note("after A finalizer")
        seen.agent_after_a_finalizer = turn.agent
        seen.lease_after_a_finalizer = turn.lease
        seen.lease_b_releases_after_a_finalizer = leases[1].release.call_count
        seen.running_after_a_finalizer = self._is_session_running(quick_key)
        seen.token_b_still_held = (
            runner._turn_leases._leases[healed.session_id].holder is seen.token_b
        )
        seen.old_session_holder = runner._turn_leases._leases[entry.session_id].holder
        seen.third_reply = await self._handle_message(_event("third", src))
        return "REPLACEMENT_REPLY"

    with (
        patch.object(GatewayRunner, "_handle_message_with_agent", agent_turn),
        patch.object(GatewayRunner, "_run_post_turn_hooks", AsyncMock()),
        patch.object(GatewayRunner, "_clear_durable_active_turn", AsyncMock()),
        patch.object(GatewayRunner, "_persist_active_agents", lambda self: None),
        patch("hermes_cli.active_sessions.try_acquire_active_session", side_effect=acquire_slot),
    ):
        prefix = "/moa " if moa else ""
        task_a = asyncio.create_task(runner._handle_message(_event(prefix + "turn A", src)))
        await asyncio.wait_for(a_running.wait(), timeout=5)
        store._db.end_session(entry.session_id, end_reason)
        seen.reply_2 = await asyncio.wait_for(
            runner._handle_message(_event(prefix + "message 2", src)), timeout=5
        )
        await task_a
    _note("after B finalizer")
    seen.timeline = timeline
    seen.original_session_id = entry.session_id
    seen.leases, seen.turns, seen.agent_a, seen.runner, seen.key = leases, turns, agent_a, runner, key
    return seen


def _assert_replacement_turn_survived_the_old_finalizer(seen: SimpleNamespace) -> None:
    assert seen.reply_2 == "REPLACEMENT_REPLY"
    assert seen.agent_a.hard_interrupts == [_INTERRUPT_REASON_EVICTED]
    assert seen.agent_a.interrupts == []  # message 2 never went into the dead runtime (#99106)

    assert seen.agent_after_a_finalizer is _AGENT_PENDING_SENTINEL, (
        "the evicted turn's finalizer erased the replacement turn's slot"
    )
    assert seen.lease_after_a_finalizer is seen.leases[1]
    assert seen.lease_b_releases_after_a_finalizer == 0
    assert seen.token_b_still_held is True
    assert seen.running_after_a_finalizer is True
    assert seen.third_reply is None
    assert seen.turns == ["turn A", "message 2"], "a concurrent turn was admitted"

    # The evicted turn released what was its own: its slot lease once (by the eviction) and
    # its turn lease; the replacement's own finalizer then cleaned up after it (no zombie).
    assert seen.leases[0].release.call_count == 1
    assert seen.token_a.released is True
    assert seen.runner._is_session_running(seen.key) is False
    assert seen.leases[1].release.call_count == 1


@pytest.mark.asyncio
async def test_old_turn_finalizer_does_not_erase_the_replacement_turn(tmp_path):
    """``ws_orphan_reap``: recovery reopens the same session id, so the replacement's turn lease
    waits for the evicted turn's finalizer — which must release only the slot IT claimed. The
    replacement's sentinel, active-session lease and turn lease survive it, and a third message
    is still refused as a concurrent turn."""
    seen = await _run_eviction_interleaving(tmp_path, end_reason="ws_orphan_reap")

    assert seen.healed_session_id == seen.original_session_id
    _assert_replacement_turn_survived_the_old_finalizer(seen)


@pytest.mark.asyncio
async def test_evicted_turn_releases_its_turn_lease_when_the_replacement_moved_to_a_fresh_session(tmp_path):
    """A row ended for a reason recovery does not reopen: the cold path creates a fresh session
    id, the replacement's turn lease is acquired at once and overwrites the slot's token while
    the evicted turn still holds its own. The evicted turn's finalizer must still release the
    lease on the old session id (or it stays held forever and a later /resume of it times out
    on every turn), without touching the replacement's."""
    seen = await _run_eviction_interleaving(tmp_path, end_reason="idle")

    assert seen.healed_session_id != seen.original_session_id
    _assert_replacement_turn_survived_the_old_finalizer(seen)
    assert seen.old_session_holder is None, "the reaped session id stayed leased by the evicted turn"


@pytest.mark.asyncio
async def test_evicted_moa_one_shot_restores_before_the_replacement_and_never_clobbers_it(tmp_path):
    """Review of #106966 (ehz0ah): the one-shot restore used to run from A's late ``finally``
    against shared conversation state, after replacement B had installed its own override —
    B then resolved A's prior model and lost its cached agent. The restore is turn-owned now:
    it rides A's slot release (the eviction), before B claims, and A's finalizer touches
    nothing of B's."""
    seen = await _run_eviction_interleaving(tmp_path, end_reason="ws_orphan_reap", moa=True)
    _assert_replacement_turn_survived_the_old_finalizer(seen)

    overrides = {phase: value for kind, phase, value in
                 (e for e in seen.timeline if e[0] == "override")}
    assert overrides["A running"]["provider"] == "moa"
    # A's one-shot ended with its slot: B snapshots the prior override, not A's MoA.
    assert seen.b_model_restore == {"had_override": True, "override": {"provider": "openrouter", "model": "gpt-4"}}
    assert overrides["B running"]["provider"] == "moa"
    assert overrides["after A finalizer"]["provider"] == "moa", "A's stale snapshot overwrote B's override"
    b_running = seen.timeline.index(("override", "B running", overrides["B running"]))
    a_done = seen.timeline.index(("override", "after A finalizer", overrides["after A finalizer"]))
    assert [e for e in seen.timeline[b_running:a_done] if e[0] == "cache_evict"] == [], (
        "A's finalizer evicted B's cached agent mid-turn"
    )
    assert overrides["after B finalizer"] == {"provider": "openrouter", "model": "gpt-4"}

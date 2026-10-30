"""sys-8xces5 — /loop wakeup injection is idempotent per (session, tick index, payload digest).

Regression for the lane-b73cc5 incident: the gateway watcher's fire path wrapped
admit_route_wake/handle_message in try/except and called ``abandon_tick()`` on ANY exception.
When the raise came AFTER the prompt had persisted as a role=user turn, the rollback re-fired
the same '#N' prompt every pass — five byte-identical wakeup turns in one session.
"""

import time
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli import goals, loops

SID = "sid-coalesce-loop"
SESSION_KEY = "agent:main:discord:channel:coalesce"


@pytest.fixture
def loop_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()
    goals._get_session_db()  # pre-warm; see test_loop_command.py for why
    # No session authority -> admit_route_wake returns False -> the adapter path is exercised.
    import gateway.session_authorities

    monkeypatch.setattr(gateway.session_authorities, "active_authority", lambda runner: None)
    yield home
    goals._DB_CACHE.clear()


class _FakeStore:
    """Transcript-backed dedupe ledger: ``turns`` is what the landed-check reads."""

    def __init__(self):
        self.turns = []
        self.entry = SimpleNamespace(session_id=SID)

    def lookup_by_session_key(self, key):
        return self.entry

    def load_transcript(self, session_id):
        return list(self.turns)


class _Adapter:
    """handle_message persists the prompt then raises (the incident shape) when ``fail`` is set."""

    def __init__(self, store, fail="after"):
        self.store = store
        self.fail = fail

    async def handle_message(self, event):
        if self.fail == "before":
            raise RuntimeError("injection boom (pre-persist)")
        self.store.turns.append({"role": "user", "content": event.text})
        if self.fail == "after":
            raise RuntimeError("injection boom (post-persist)")


def _make_runner(store, adapter):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="t")})
    runner.session_store = store
    runner.adapters = {Platform.DISCORD: adapter}
    runner._running_agents = {}
    runner._adapters_for_profile = lambda profile: {Platform.DISCORD: adapter}
    runner._build_process_event_source = lambda evt: SimpleNamespace(
        platform=Platform.DISCORD, chat_id=evt["chat_id"], chat_type=evt["chat_type"],
        thread_id=evt["thread_id"] or None, user_id=evt["user_id"], user_name=evt["user_name"],
    )
    runner._session_key_for_source = lambda source: SESSION_KEY
    runner._publish_local_control = lambda source: None

    async def _exec(fn, *args):
        return fn(*args)

    runner._run_in_executor_with_context = _exec
    return runner


def _set_loop(prompt="poll CI", **overrides):
    state = loops.LoopState(prompt=prompt, interval_seconds=60,
                            route={"platform": "discord", "chat_id": "chat-coalesce"})
    state.next_due_at = time.time() - 1
    for k, v in overrides.items():
        setattr(state, k, v)
    loops.save_loop(SID, state)


async def _watcher_pass(runner):
    await GatewayRunner._loop_wakeup_fire_one(
        runner, SID, loops.load_loop(SID), time.time(), set())


def _wakeup_turns(store, tick):
    return [t for t in store.turns if f"[/loop wakeup #{tick}" in t["content"]]


# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_post_persist_raise_lands_once_across_five_watcher_passes(loop_env):
    """The incident: persist-then-raise on every pass used to duplicate '#1' five times."""
    store = _FakeStore()
    runner = _make_runner(store, _Adapter(store, fail="after"))
    _set_loop()

    for _ in range(5):
        await _watcher_pass(runner)

    assert len(_wakeup_turns(store, 1)) == 1
    state = loops.load_loop(SID)
    assert state.ticks_fired == 1
    assert state.awaiting_response is True  # landed: the turn owns the tick until it is answered
    assert state.landed_tick == 1  # the coalesce marker is stamped for cross-surface re-fires


@pytest.mark.asyncio
async def test_viewer_rollback_of_landed_tick_coalesces_the_refire(loop_env):
    """After a landed post-persist raise, a bare abandon (viewer/RPC surface) rolls the tick
    back; the next fire re-claims index #1 WITHOUT injecting a second role=user turn."""
    store = _FakeStore()
    runner = _make_runner(store, _Adapter(store, fail="after"))
    _set_loop()
    await _watcher_pass(runner)
    assert len(_wakeup_turns(store, 1)) == 1

    loops.LoopManager(session_id=SID).abandon_tick()  # viewer "roll it back" (no payload)
    state = loops.load_loop(SID)
    assert state.ticks_fired == 0 and state.awaiting_response is False
    state.next_due_at = time.time() - 1
    loops.save_loop(SID, state)

    for _ in range(3):
        await _watcher_pass(runner)

    assert len(_wakeup_turns(store, 1)) == 1  # coalesced: no re-injection
    state = loops.load_loop(SID)
    assert state.ticks_fired == 1 and state.awaiting_response is True
    assert state.landed_tick == 0  # marker consumed by the coalesce


@pytest.mark.asyncio
async def test_pre_persist_failure_still_rolls_back_and_recovers_the_tick(loop_env):
    """Genuine pre-persist failures keep the old rollback (the tick is not burned); once the
    injection path recovers, the SAME index fires exactly once."""
    store = _FakeStore()
    adapter = _Adapter(store, fail="before")
    runner = _make_runner(store, adapter)
    _set_loop()

    for _ in range(2):
        await _watcher_pass(runner)

    assert store.turns == []
    state = loops.load_loop(SID)
    assert state.ticks_fired == 0 and state.awaiting_response is False

    state.next_due_at = time.time() - 1  # interval elapsed
    loops.save_loop(SID, state)
    adapter.fail = None  # injection path recovers
    await _watcher_pass(runner)

    assert len(_wakeup_turns(store, 1)) == 1
    state = loops.load_loop(SID)
    assert state.ticks_fired == 1 and state.awaiting_response is True


@pytest.mark.asyncio
async def test_re_set_loop_pre_persist_failure_rolls_back_despite_stale_identical_turn(loop_env):
    """Review fix (sys-8xces5): the landed probe is attempt-scoped. A re-set loop restarts at
    '#1', so the OLD loop's byte-identical '#1' turn sits in the transcript tail; a pre-persist
    failure must roll the tick back, NOT false-land on that stale turn and wedge
    awaiting_response (persisted, so the wedge would survive restarts)."""
    store = _FakeStore()
    adapter = _Adapter(store, fail="after")
    runner = _make_runner(store, adapter)
    _set_loop()
    await _watcher_pass(runner)  # old loop: '#1' persists, then raise -> landed
    assert len(_wakeup_turns(store, 1)) == 1

    loops.LoopManager(session_id=SID).clear()
    _set_loop()  # re-set the same prompt: fresh state, ticks_fired back to 0
    adapter.fail = "before"  # genuine pre-persist failure this time
    await _watcher_pass(runner)

    state = loops.load_loop(SID)
    assert state.ticks_fired == 0 and state.awaiting_response is False  # rolled back, not burned
    assert state.landed_tick == 0  # and NOT false-landed on the stale '#1'
    assert len(_wakeup_turns(store, 1)) == 1  # the stale turn is still the only one


@pytest.mark.asyncio
async def test_slash_wakeup_pre_persist_failure_skips_the_landed_probe(loop_env):
    """Review fix (sys-8xces5): a slash wakeup is the SAME raw command every tick, so the digest
    cannot distinguish attempts at all — the probe is skipped for slash wakeups and any raise
    rolls the tick back, even with a byte-identical '/recap' user turn in the tail."""
    store = _FakeStore()
    store.turns.append({"role": "user", "content": "/recap"})  # a past /recap command turn
    runner = _make_runner(store, _Adapter(store, fail="before"))
    _set_loop(prompt="/recap")

    await _watcher_pass(runner)

    state = loops.load_loop(SID)
    assert state.ticks_fired == 0 and state.awaiting_response is False
    assert state.landed_tick == 0
    assert store.turns == [{"role": "user", "content": "/recap"}]


class TestLoopManagerCoalesce:
    """Shared-manager surface (covers the CLI/TUI firers and the viewer RPC too)."""

    def test_landed_abandon_then_refire_reclaims_without_reinjecting(self, loop_env):
        _set_loop()
        mgr = loops.LoopManager(session_id=SID)
        msg1 = mgr.fire_tick()
        assert msg1 and "[/loop wakeup #1" in msg1

        mgr.abandon_tick(msg1)  # driver: raised after persist
        state = loops.load_loop(SID)
        assert state.ticks_fired == 1 and state.awaiting_response is True
        assert state.landed_tick == 1 and state.landed_digest == loops.wakeup_payload_digest(msg1)

        mgr.abandon_tick()  # rollback of the landed tick (same index re-renders)
        mgr.state.next_due_at = time.time() - 1  # interval elapsed
        loops.save_loop(SID, mgr.state)

        assert mgr.fire_tick() is None  # coalesced — no second '#1' prompt
        state = loops.load_loop(SID)
        assert state.ticks_fired == 1 and state.awaiting_response is True
        assert state.landed_tick == 0

    def test_plain_abandon_reuses_index_when_nothing_persisted(self, loop_env):
        _set_loop()
        mgr = loops.LoopManager(session_id=SID)
        msg1 = mgr.fire_tick()
        mgr.abandon_tick()  # pre-persist failure: rollback unchanged
        state = loops.load_loop(SID)
        assert state.ticks_fired == 0 and state.awaiting_response is False
        mgr.state.next_due_at = time.time() - 1  # interval elapsed
        loops.save_loop(SID, mgr.state)
        assert mgr.fire_tick() == msg1  # same index, not blocked — nothing ever landed

    def test_old_state_rows_without_landed_fields_round_trip(self, loop_env):
        """Missing landed keys in persisted JSON cast to the zero defaults (old rows, new code)."""
        import json

        data = json.loads(loops.LoopState(prompt="p").to_json())
        data.pop("landed_tick")
        data.pop("landed_digest")
        state = loops.LoopState.from_json(json.dumps(data))
        assert state.landed_tick == 0 and state.landed_digest == ""

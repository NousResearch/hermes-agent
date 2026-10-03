"""#120334 regression: a queued background-process heartbeat for a process that
has already exited must NOT start a fresh agent turn.

The fix lives at two levels:
  * ``tools.process_registry`` — ``_heartbeat_loop`` and ``_emit_heartbeat``
    re-validate liveness atomically under the registry lock right before
    enqueueing. A heartbeat emitted while the process was running can be
    stale by the time the gateway consumes it, so the gateway revalidates
    again at admission time.
  * ``gateway.run_heartbeat_acceptance.process_heartbeat_still_alive`` — the
    synthetic ``MessageEvent`` carries ``_process_heartbeat_session_id`` +
    ``_process_heartbeat_started_at`` stamped at ``_inject_watch_notification``
    time; the liveness helper looks the session up in the live process
    registry and rejects events whose process has exited or whose id has been
    recycled (started_at mismatch).

These tests assert the helper directly (unit) and the injection path +
helper combined (integration). Pre-fix: the synthetic event carries no
identity, the helper has no logic, and the stale event is admitted. Post-fix:
the event is dropped, no turn starts, and the helper returns False.
"""
import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import patch

from gateway.platforms.event import MessageType


# ── unit: process_heartbeat_still_alive ───────────────────────────────────


def _stub_session(*, exited, started_at=100.0, session_id="proc_x"):
    """Build a ProcessSession-like object compatible with the helper's reads."""
    return SimpleNamespace(id=session_id, started_at=started_at, exited=exited)


def test_process_heartbeat_still_alive_returns_true_for_non_heartbeat_event():
    """A normal user turn or scheduled /heartbeat has no process provenance;
    the helper must NOT block it. The scheduled /heartbeat path keeps its own
    provenance check (heartbeat_owner_is_current)."""
    from gateway.run_heartbeat_acceptance import process_heartbeat_still_alive

    event = SimpleNamespace()  # no _process_heartbeat_session_id
    assert process_heartbeat_still_alive(event) is True


def test_process_heartbeat_still_alive_returns_true_for_running_session(monkeypatch):
    """A still-running session in the live process registry is alive — the
    helper must return True (admit the heartbeat)."""
    from gateway.run_heartbeat_acceptance import process_heartbeat_still_alive
    from tools import process_registry as pr_mod

    live = _stub_session(exited=False, started_at=100.0)
    monkeypatch.setattr(pr_mod.process_registry, "get", lambda sid: live)

    event = SimpleNamespace(
        _process_heartbeat_session_id="proc_x",
        _process_heartbeat_started_at=100.0,
    )
    assert process_heartbeat_still_alive(event) is True


def test_process_heartbeat_still_alive_returns_false_for_exited_session(monkeypatch, caplog):
    """The session exited between emission and delivery — pre-fix this would
    start a turn with a 'still running' message after the user saw the
    completion. Post-fix the helper returns False and the gateway drops it."""
    from gateway.run_heartbeat_acceptance import process_heartbeat_still_alive
    from tools import process_registry as pr_mod

    dead = _stub_session(exited=True, started_at=100.0)
    monkeypatch.setattr(pr_mod.process_registry, "get", lambda sid: dead)

    event = SimpleNamespace(
        _process_heartbeat_session_id="proc_x",
        _process_heartbeat_started_at=100.0,
    )
    with caplog.at_level(logging.DEBUG, logger="gateway.run"):
        assert process_heartbeat_still_alive(event) is False


def test_process_heartbeat_still_alive_returns_false_for_missing_session(monkeypatch):
    """The session is no longer in the registry at all (reaped, evicted, or
    from a different gateway lifetime). Helper must return False."""
    from gateway.run_heartbeat_acceptance import process_heartbeat_still_alive
    from tools import process_registry as pr_mod

    monkeypatch.setattr(pr_mod.process_registry, "get", lambda sid: None)

    event = SimpleNamespace(
        _process_heartbeat_session_id="proc_missing",
        _process_heartbeat_started_at=100.0,
    )
    assert process_heartbeat_still_alive(event) is False


def test_process_heartbeat_still_alive_returns_false_for_started_at_mismatch(monkeypatch):
    """The session id was recycled by a fresh spawn (different started_at).
    The old heartbeat must be rejected — the original incarnation finished."""
    from gateway.run_heartbeat_acceptance import process_heartbeat_still_alive
    from tools import process_registry as pr_mod

    fresh = _stub_session(exited=False, started_at=200.0)  # NEW incarnation
    monkeypatch.setattr(pr_mod.process_registry, "get", lambda sid: fresh)

    event = SimpleNamespace(
        _process_heartbeat_session_id="proc_x",
        _process_heartbeat_started_at=100.0,  # OLD incarnation's epoch
    )
    assert process_heartbeat_still_alive(event) is False


# ── integration: _inject_watch_notification stamps the provenance ──────────


def test_inject_watch_notification_stamps_process_heartbeat_identity(monkeypatch):
    """``_inject_watch_notification`` must stamp the synthetic event with the
    heartbeat session identity so the gateway can revalidate at admission.
    Pre-fix: no attributes set, the staleness check has nothing to look at."""
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    from gateway.config import Platform

    # Stand up a minimal GatewayRunner with the run_notifications mixin.
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)

    # Minimal adapter that records the synth event and accepts it.
    captured = {}

    class _Adapter:
        name = "fake"

        def fronts_platform(self, platform):
            return True

        def prime_routing_cache(self, event):
            pass

        async def handle_message(self, event):
            captured["event"] = event
            event._gateway_accepted = True

    runner.adapters = {Platform.TELEGRAM: _Adapter()}
    runner.config = SimpleNamespace(platforms={})

    # Pre-build the event source.
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="42", user_id="42", chat_type="dm",
    )
    runner._build_process_event_source = lambda evt: source

    def _resolve(name, source=None):
        return runner.adapters[Platform.TELEGRAM]
    runner._resolve_injection_adapter = _resolve

    evt = {
        "type": "heartbeat",
        "session_id": "proc_stale_inject",
        "started_at": 999.0,
        "session_key": "agent:telegram:dm:42",
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "42",
        "command": "sleep 60",
    }

    import asyncio
    asyncio.run(runner._inject_watch_notification("still working", evt))

    injected = captured.get("event")
    assert injected is not None, "adapter never received the synthetic event"
    assert isinstance(injected, MessageEvent)
    assert getattr(injected, "_process_heartbeat_session_id", None) == "proc_stale_inject"
    assert getattr(injected, "_process_heartbeat_started_at", None) == 999.0


def test_inject_watch_notification_does_not_stamp_non_heartbeat_events(monkeypatch):
    """A watch_match / completion / async_delegation event must NOT receive
    heartbeat provenance attributes — the helper would (correctly) treat it
    as a heartbeat and look up a process session that doesn't exist."""
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    from gateway.config import Platform

    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)

    captured = {}

    class _Adapter:
        name = "fake"

        def fronts_platform(self, platform):
            return True

        def prime_routing_cache(self, event):
            pass

        async def handle_message(self, event):
            captured["event"] = event
            event._gateway_accepted = True

    runner.adapters = {Platform.TELEGRAM: _Adapter()}
    runner.config = SimpleNamespace(platforms={})
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="42", user_id="42", chat_type="dm",
    )
    runner._build_process_event_source = lambda evt: source

    def _resolve(name, source=None):
        return runner.adapters[Platform.TELEGRAM]
    runner._resolve_injection_adapter = _resolve

    evt = {
        "type": "watch_match",
        "session_id": "proc_w",
        "session_key": "agent:telegram:dm:42",
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "42",
        "command": "echo done",
        "pattern": "DONE",
        "output": "DONE",
    }

    import asyncio
    asyncio.run(runner._inject_watch_notification("matched", evt))

    injected = captured["event"]
    assert not hasattr(injected, "_process_heartbeat_session_id") \
        or getattr(injected, "_process_heartbeat_session_id", None) is None


# ── integration: end-to-end stale heartbeat does NOT start a turn ──────────


def test_stale_queued_heartbeat_does_not_start_turn(monkeypatch, caplog):
    """End-to-end #120334. Build a synthetic MessageEvent that looks like
    a queued background-process heartbeat for a session that has already
    exited, then call the liveness helper the gateway uses. Pre-fix: the
    helper does not exist / returns True / a turn starts. Post-fix: the
    helper returns False, the caller drops the event, no turn starts."""
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    from gateway.config import Platform
    from gateway.run_heartbeat_acceptance import process_heartbeat_still_alive
    from tools import process_registry as pr_mod

    # The session is GONE (process exited long ago).
    monkeypatch.setattr(pr_mod.process_registry, "get", lambda sid: None)

    event = MessageEvent(
        text="heartbeat #1 — still running after 1m",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="42", user_id="42", chat_type="dm"),
    )
    event._process_heartbeat_session_id = "proc_120334"
    event._process_heartbeat_started_at = 1700000000.0

    with caplog.at_level(logging.DEBUG, logger="gateway.run"):
        result = process_heartbeat_still_alive(event)
    assert result is False, "Stale heartbeat must be rejected (#120334)"


def test_fresh_queued_heartbeat_admitted(monkeypatch):
    """Sanity: a still-running process admits its heartbeat. Regression guard
    against an over-eager helper that would silence valid heartbeats."""
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    from gateway.config import Platform
    from gateway.run_heartbeat_acceptance import process_heartbeat_still_alive
    from tools import process_registry as pr_mod

    monkeypatch.setattr(
        pr_mod.process_registry, "get",
        lambda sid: SimpleNamespace(id=sid, started_at=1700000000.0, exited=False),
    )

    event = MessageEvent(
        text="heartbeat #1",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="42", user_id="42", chat_type="dm"),
    )
    event._process_heartbeat_session_id = "proc_alive"
    event._process_heartbeat_started_at = 1700000000.0

    assert process_heartbeat_still_alive(event) is True


def _ret(value):
    """An async callable that ignores its arguments and returns ``value``."""
    async def _coro(*_a, **_kw):
        return value
    return _coro


def _make_prepared(message_text: str = "heartbeat #1"):
    """A ``GatewayRunner._PreparedTurn`` carrying the minimum the guard path needs."""
    from gateway.run import GatewayRunner

    return GatewayRunner._PreparedTurn(
        history=[],
        context_prompt="ctx",
        message_text=message_text,
        persist_user_message=None,
        persist_user_timestamp=None,
        persist_user_display_kind=None,
    )


# ── entry-point 2: the pre-``_run_agent`` re-check ───────────────────────────
# The comment on these three entry points claims all of them are covered.
# Muting each guard branch in turn showed they were NOT: mutating
# ``_hmwa_resolve_session`` turns ``test_turn_resolve_session_drops_a_stale_heartbeat``
# red and mutating the busy-drain guard turns the two busy-drain tests red, but
# mutating this one left the whole file green — it had no driver at all. This is
# that driver.


def test_turn_start_recheck_drops_a_heartbeat_that_died_while_waiting(monkeypatch):
    """Guard site 2 (the pre-``_run_agent`` re-check) must reject a heartbeat
    whose process exited *after* the turn was admitted.

    The site exists for the gap between admission and execution: on a busy
    session the ``_hmwa_resolve_session`` check can be minutes stale, so this
    re-check runs immediately before the runner starts. A heartbeat that was
    live at admission but whose process died while the turn waited must not buy
    a model call to deliver a \"still running\" line the user already saw finish.

    Drives the real ``_handle_message_with_agent`` with ``_hmwa_prepare_turn``
    stubbed, so deleting the guard branch turns this red.
    """
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from gateway.config import Platform
    from gateway.run_heartbeat_acceptance import process_heartbeat_still_alive
    from tools import process_registry as pr_mod

    # Live at admission, dead by turn-start — the case this site exists for.
    monkeypatch.setattr(
        pr_mod.process_registry, "get",
        lambda sid: SimpleNamespace(id=sid, started_at=1700000000.0, exited=True),
    )

    event = MessageEvent(
        text="heartbeat #1 — still running after 1m",
        message_id="hb-recheck",
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="42", user_id="42", chat_type="dm"),
    )
    event._process_heartbeat_session_id = "proc_died_while_waiting"
    event._process_heartbeat_started_at = 1700000000.0

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._session_sources = {}
    ran_agent = []

    async def _get_or_create(source, **_kw):
        return SimpleNamespace(session_id="sid-1", session_key="sess", active_stream_id=None)

    async def _lookup_by_key(key, **_kw):
        return None

    async_session = SimpleNamespace(
        get_or_create_session=_get_or_create,
        lookup_by_session_key=_lookup_by_key,
    )
    monkeypatch.setattr(
        type(runner), "async_session_store", property(lambda _s: async_session), raising=True
    )

    # Admission must succeed (the process looked alive when the turn resolved),
    # then the turn must be dropped at the re-check without running the agent.
    monkeypatch.setattr(
        runner, "_hmwa_resolve_session",
        _ret((event.source, SimpleNamespace(session_id="sid-1"), "sess")),
    )
    monkeypatch.setattr(
        runner, "_hmwa_prepare_turn",
        _ret((_make_prepared("heartbeat #1"), [])),
    )

    async def _run_agent(**kwargs):
        ran_agent.append(kwargs)
        return {}

    monkeypatch.setattr(runner, "_run_agent", _run_agent)
    # hooks.emit must not blow up before the re-check.
    monkeypatch.setattr(
        runner, "hooks", SimpleNamespace(emit=_ret(None)), raising=False
    )

    asyncio.run(runner._handle_message_with_agent(event, event.source, "quick", 1))

    assert not ran_agent, (
        "a heartbeat whose process exited while the turn waited must not reach "
        "the agent runner — that is exactly the wasted model call the re-check "
        "exists to prevent (#120395)"
    )
    assert process_heartbeat_still_alive(event) is False


def test_turn_start_recheck_admits_a_live_heartbeat(monkeypatch):
    """The sibling guard: a heartbeat whose process is still alive at turn-start
    must still run. Keeps the re-check from over-collecting (#120395)."""
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from gateway.config import Platform
    from tools import process_registry as pr_mod

    monkeypatch.setattr(
        pr_mod.process_registry, "get",
        lambda sid: SimpleNamespace(id=sid, started_at=1700000000.0, exited=False),
    )

    event = MessageEvent(
        text="heartbeat #2",
        message_id="hb-recheck-live",
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="42", user_id="42", chat_type="dm"),
    )
    event._process_heartbeat_session_id = "proc_alive_at_start"
    event._process_heartbeat_started_at = 1700000000.0

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._session_sources = {}
    ran_agent = []

    async def _get_or_create(source, **_kw):
        return SimpleNamespace(session_id="sid-1", session_key="sess", active_stream_id=None)

    async_session = SimpleNamespace(
        get_or_create_session=_get_or_create, lookup_by_session_key=_ret(None),
    )
    monkeypatch.setattr(
        type(runner), "async_session_store", property(lambda _s: async_session), raising=True
    )
    monkeypatch.setattr(
        runner, "_hmwa_resolve_session",
        _ret((event.source, SimpleNamespace(session_id="sid-1"), "sess")),
    )
    monkeypatch.setattr(
        runner, "_hmwa_prepare_turn",
        _ret((_make_prepared("heartbeat #2"), [])),
    )

    async def _run_agent(**kwargs):
        ran_agent.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr(runner, "_run_agent", _run_agent)
    monkeypatch.setattr(
        runner, "hooks", SimpleNamespace(emit=_ret(None)), raising=False
    )

    asyncio.run(runner._handle_message_with_agent(event, event.source, "quick", 1))

    assert ran_agent, "a live heartbeat must still reach the agent runner"


# ── entry-point 3: the busy drain path ──────────────────────────────────────
# The two guards above (``_hmwa_resolve_session`` and the pre-``_run_agent``
# check) only run on the turn that ADMITS the heartbeat. A heartbeat queued
# behind a busy session is consumed later by ``_run_agent_drain_pending`` ->
# ``_run_agent_queued_followup`` -> ``_run_agent`` — none of which pass through
# those sites, so before this was fixed the guard ran ZERO times on that path
# and a dead process's "still running" text reached the model verbatim.
# These tests drive the real drain entry point, not the helper.


def _heartbeat_event(*, exited, started_at=100.0, proc_id="proc_dead") -> SimpleNamespace:
    """A synthetic process-heartbeat MessageEvent as injected on the completion queue."""
    return SimpleNamespace(
        text="heartbeat #1 - proc_dead still running after 1m",
        message_id=proc_id,
        _process_heartbeat_session_id=proc_id,
        _process_heartbeat_started_at=started_at,
        reply_expected=False,
        source=None,
        _gateway_accepted=True,
        exited=exited,  # only read by these tests' registry stub
    )


def _real_runner():
    """A real GatewayRunner instance (no __new__ trickery) for the drain call."""
    from gateway.run import GatewayRunner

    return GatewayRunner.__new__(GatewayRunner)


class _Adapter:
    """Minimal adapter with the pending-slot store the drain reads."""

    def __init__(self, event):
        self._pending_messages = {"sess": event}
        self._active_sessions = {}


def _registry_stub(session):
    """A process_registry-like object whose get() returns *session*."""
    from types import SimpleNamespace as NS

    return NS(get=lambda _sid: session)


def test_busy_drain_drops_a_stale_process_heartbeat(monkeypatch):
    """The reviewer's repro: dead process, drained behind a busy session.

    Drives ``_run_agent_drain_pending`` for real (the entry point the guard was
    missing from). Pre-fix the heartbeat text came back out as the next turn's
    prompt; post-fix the event is dropped and no pending prompt is produced.
    """
    import gateway.run_turn as run_turn

    runner = _real_runner()
    event = _heartbeat_event(exited=True)
    adapter = _Adapter(event)

    import gateway.run as gateway_run

    monkeypatch.setattr(
        gateway_run, "_dequeue_pending_event", lambda _ad, _key: event, raising=True
    )
    monkeypatch.setattr(
        "tools.process_registry.process_registry", _registry_stub(None), raising=True
    )
    # Prove the drain's own guard ran — the reviewer's point was that it never did.
    calls: list[str] = []
    from gateway.run import GatewayRunner

    def _spy(evt):
        calls.append(getattr(evt, "_process_heartbeat_session_id", None))
        # Delegate to the real staticmethod so the drop under test is the
        # production predicate, not a stub that always agrees.
        return GatewayRunner._queued_process_heartbeat_is_live(evt)

    monkeypatch.setattr(runner, "_queued_process_heartbeat_is_live", _spy, raising=True)

    pending_event, pending = asyncio.run(
        runner._run_agent_drain_pending({"interrupted": False}, adapter, None, "sess")
    )

    assert calls == ["proc_dead"], "the busy-path guard must actually be called"
    assert pending_event is None, "a stale heartbeat must not be promoted to the next turn"
    assert pending is None, "and must not become the next user prompt"


def test_busy_drain_admits_a_live_process_heartbeat(monkeypatch):
    """The control: the same path must still pass a heartbeat that is alive."""
    import gateway.run_turn as run_turn
    from types import SimpleNamespace as NS

    runner = _real_runner()
    event = _heartbeat_event(exited=False)
    adapter = _Adapter(event)

    import gateway.run as gateway_run

    monkeypatch.setattr(
        gateway_run, "_dequeue_pending_event", lambda _ad, _key: event, raising=True
    )
    monkeypatch.setattr(
        "tools.process_registry.process_registry",
        _registry_stub(NS(id="proc_dead", started_at=100.0, exited=False)),
        raising=True,
    )

    pending_event, pending = asyncio.run(
        runner._run_agent_drain_pending({"interrupted": False}, adapter, None, "sess")
    )

    assert pending_event is event, "a live heartbeat must still be delivered"
    assert pending is not None and "still running" in pending


def test_busy_drain_promotes_the_next_event_after_dropping_one(monkeypatch):
    """FIFO must not stall behind a dropped heartbeat.

    A live /queue item sitting in overflow behind a dead heartbeat has to be
    promoted into the vacated slot, not left to rot.
    """
    import gateway.run_turn as run_turn

    runner = _real_runner()
    dead = _heartbeat_event(exited=True, proc_id="proc_dead")
    next_up = SimpleNamespace(text="follow-up the queued message", message_id="live-1", _gateway_accepted=True)
    adapter = _Adapter(dead)

    import gateway.run as gateway_run

    monkeypatch.setattr(
        gateway_run, "_dequeue_pending_event", lambda _ad, _key: dead, raising=True
    )
    monkeypatch.setattr(
        "tools.process_registry.process_registry", _registry_stub(None), raising=True
    )
    # The first promotion yields the real follow-up (the dead heartbeat is the
    # slot occupant, so the head of overflow is promoted over it).
    promotions = [dead, next_up, next_up]
    monkeypatch.setattr(
        runner, "_promote_queued_event", lambda _k, _ad, cur: promotions.pop(0) if promotions else cur,
        raising=True,
    )

    pending_event, pending = asyncio.run(
        runner._run_agent_drain_pending({"interrupted": False}, adapter, None, "sess")
    )

    assert pending_event is next_up, "the event behind the dropped heartbeat must run"
    assert pending is not None and "follow-up" in pending


def test_guard_removal_from_the_busy_path_turns_these_red(monkeypatch):
    """Mutation check: deleting the drain guard must break the stale test.

    Mirrors deleting the branch the way the other two guard sites were mutated
    in the review — this is what proves the test above has teeth.
    """
    import gateway.run_turn as run_turn

    runner = _real_runner()
    event = _heartbeat_event(exited=True)
    adapter = _Adapter(event)

    import gateway.run as gateway_run

    monkeypatch.setattr(
        gateway_run, "_dequeue_pending_event", lambda _ad, _key: event, raising=True
    )
    monkeypatch.setattr(
        "tools.process_registry.process_registry", _registry_stub(None), raising=True
    )
    # Simulate the guard having never been called (pre-fix behaviour).
    monkeypatch.setattr(
        runner, "_queued_process_heartbeat_is_live", lambda _e: True, raising=True
    )

    pending_event, pending = asyncio.run(
        runner._run_agent_drain_pending({"interrupted": False}, adapter, None, "sess")
    )

    assert pending is not None and "still running" in pending, (
        "with the guard bypassed the stale heartbeat reaches the model — "
        "which is exactly the bug these tests must catch"
    )


# ── entry-point 1: _hmwa_resolve_session ────────────────────────────────────
# The review showed the two turn-level guards have zero coverage: every test
# here calls the helper directly. This drives the real entry point, so
# deleting the guard branch at that site turns it red.


def test_turn_resolve_session_drops_a_stale_heartbeat(monkeypatch):
    """Guard site 1 (``_hmwa_resolve_session``) must actually reject stale ones."""
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from gateway.config import Platform, PlatformConfig
    from tools import process_registry as pr_mod

    monkeypatch.setattr(pr_mod.process_registry, "get", lambda sid: None)

    event = MessageEvent(
        text="heartbeat #1 — still running after 1m",
        message_id="hb-1",
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="42", user_id="42", chat_type="dm"),
    )
    event._process_heartbeat_session_id = "proc_dead"
    event._process_heartbeat_started_at = 1700000000.0

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._session_sources = {}

    async def _get_or_create(source, **_kw):
        return SimpleNamespace(session_id="sid-1", session_key="sess", active_stream_id=None)

    async def _lookup_by_key(key, **_kw):
        return None

    async_session = SimpleNamespace(
        get_or_create_session=_get_or_create,
        lookup_by_session_key=_lookup_by_key,
    )
    monkeypatch.setattr(
        type(runner), "async_session_store", property(lambda _s: async_session), raising=True
    )

    resolved = asyncio.run(
        runner._hmwa_resolve_session(event, event.source)
    )

    assert resolved is None, (
        "a heartbeat whose process has exited must not resolve into a turn"
    )

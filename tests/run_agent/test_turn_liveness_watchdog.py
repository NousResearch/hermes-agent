"""Turn liveness watchdog (#95548): force-abort turns that stall silently.

Regression coverage for the gateway turn hang reported in #95548: a turn can
stall in the middle (observed between "model returned tool_calls" and tool
execution) with no error logged, no further progress, and the durable turn
lease kept renewing — so nothing ever force-aborts it and the session is
stuck until the gateway process is killed.

The fix adds a turn liveness watchdog next to the durable lease refresher in
``AIAgent.run_conversation`` (policy lives in ``agent/turn_liveness.py``).
It keys off the agent's activity clock (``_last_activity_ts`` — the #72039
single progress source, which lease renewal never touches). When a turn
shows no observable progress for the configured bound
(``agent.turn_liveness.timeout_s`` in config.yaml), it:

1. logs the stall loudly (surface instead of silent blocking),
2. force-interrupts the turn so it unwinds as an interrupted turn,
3. stops lease renewal so the durable lease lapses and stale-turn cleanup
   can reclaim the session even if the hard interrupt cannot unwind a
   truly wedged loop.

The abort is bound to the sampled ``(activity_generation, timestamp)`` pair
and revalidated at the commit point under the lock shared with
``_touch_activity`` (#95663 review): a turn that resumes while the stall is
being surfaced continues running and its lease keeps renewing.
"""

from __future__ import annotations

import logging
import threading
import time

import pytest

from run_agent import AIAgent


class _DB:
    def __init__(self, session_exists=True, acquire_result=True):
        self.events = []
        self.refresh_times = []
        self.session_exists = session_exists
        self.acquire_result = acquire_result

    def get_session(self, session_id):
        return {"id": session_id} if self.session_exists else None

    def acquire_session_turn_lease(self, session_id, holder, **kwargs):
        self.events.append(("acquire", session_id, holder))
        on_wait = kwargs.get("on_wait")
        if on_wait is not None and self.acquire_result is False:
            on_wait(0.0)
        return self.acquire_result

    def resolve_resume_session_id(self, session_id):
        self.events.append(("resolve", session_id))
        return session_id

    def get_messages_as_conversation(self, session_id, **kwargs):
        self.events.append(("reload", session_id, kwargs))
        return [{"role": "user", "content": "durable latest"}]

    def refresh_session_turn_lease(self, session_id, holder, **kwargs):
        self.events.append(("refresh", session_id, holder))
        self.refresh_times.append(time.time())
        return True

    def release_session_turn_lease(self, session_id, holder):
        self.events.append(("release", session_id, holder))


class _BlockingCommitFence:
    """Controllable compression commit fence for the stall-abort witness.

    ``cancel_before_commit`` parks the interrupt thread AFTER it has passed
    the internal ``require_generation`` comparison, simulating the unbounded
    compression-commit wait. The test resumes the turn (``_touch_activity``)
    while the hammer is parked and then releases the fence; a correct
    interrupt must abandon itself without publishing anything.
    """

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def cancel_before_commit(self, cancel_event=None):
        # `cancel_event` is accepted (and ignored) to mirror the production
        # fence signature; publication happens at the final claim edge, so
        # nothing observable is set here.
        self.calls += 1
        self.entered.set()
        assert self.release.wait(10.0), "fence was never released"
        return True


class _ParkingReleaseLock:
    """Activity-lock wrapper that parks the releasing thread at one exact
    release boundary, so a test can deterministically land real activity
    on a competing thread in the precise window under test (#95663
    round-6 review: the consume→publication boundary).

    ``park_on_release=N`` parks the thread that performs the Nth release
    AFTER the inner lock has actually been released — so the competing
    thread can immediately acquire and stamp the clock while the parked
    thread has not yet executed its next statement.
    """

    def __init__(self, inner):
        self._inner = inner
        self.park_on_release = None
        self.release_count = 0
        self.parked = threading.Event()
        self.release_park = threading.Event()

    def __enter__(self):
        return self._inner.__enter__()

    def __exit__(self, *exc_info):
        result = self._inner.__exit__(*exc_info)
        self.release_count += 1
        if (
            self.park_on_release is not None
            and self.release_count == self.park_on_release
        ):
            self.parked.set()
            assert self.release_park.wait(10.0), (
                "parked activity-lock release was never released"
            )
        return result


def _agent_with_db(db, *, session_id="stalled-session", platform="desktop"):
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = session_id
    agent.platform = platform
    agent.model = "test-model"
    agent._session_db = db
    agent._session_db_created = True
    agent._persist_disabled = False
    agent._parent_session_id = None
    agent._relay_pending_turn_id = None
    agent._reset_activity_labels_after_turn = lambda: None
    agent._conversation_root_id = lambda: session_id
    agent.log_prefix = ""
    agent._vprint = lambda *a, **k: None
    agent.status_callback = None
    agent._interrupt_requested = False
    agent._interrupt_message = None
    agent._pending_redirect = None
    agent._execution_thread_id = None
    agent._interrupt_thread_signal_pending = False
    agent._hard_interrupt_requested = threading.Event()
    agent._active_children_lock = threading.Lock()
    agent._active_children = set()
    agent.quiet_mode = True
    # A real cached agent entering a new turn holds the activity clock
    # from its PREVIOUS turn: `_reset_activity_labels_after_turn` keeps
    # `_last_activity_ts` across turns by design, so an agent that sat
    # idle longer than the watchdog bound (user walked away, came back,
    # sent a message) enters with a STALE clock. `AIAgent.run_conversation`
    # stamps the clock at turn entry (#95663 review), so the watchdog
    # measures idle from THIS turn's start — mirror that reality: stale
    # entry clock, fresh measurement after the wrapper's turn-entry stamp.
    agent._last_activity_ts = time.time() - 1000.0
    agent._last_activity_desc = "previous turn (idle)"
    agent._session_turn_lease_refresh_interval = 60.0
    return agent


@pytest.fixture
def watchdog_config(monkeypatch):
    """Arm the watchdog fast through config.yaml — the only supported surface.

    `agent.turn_liveness` is the config authority the watchdog resolves
    (AGENTS.md rejects new non-secret HERMES_* env knobs); the resolver in
    agent/turn_liveness.py validates the values and the env is never read.
    """
    import hermes_cli.config as config_module

    monkeypatch.setattr(
        config_module,
        "load_config_readonly",
        lambda: {
            "agent": {
                "turn_liveness": {"timeout_s": 0.3, "poll_s": 0.05},
            }
        },
    )
    return monkeypatch


def _run_turn(agent, inner_loop, monkeypatch):
    """Drive AIAgent.run_conversation with a fake inner conversation loop."""
    from agent import conversation_loop as loop_module

    monkeypatch.setattr(loop_module, "run_conversation", inner_loop)
    return AIAgent.run_conversation(
        agent,
        "new message",
        conversation_history=[{"role": "user", "content": "stale"}],
    )


def test_watchdog_force_aborts_silently_stalled_turn(watchdog_config, monkeypatch, caplog):
    """A turn with zero observable progress past the bound is surfaced and
    force-aborted as an interrupted turn instead of hanging forever."""
    db = _DB()
    agent = _agent_with_db(db)

    interrupt_seen = {}
    t_start = time.time()

    def stalled_loop(_agent, _message, _system, history, *_args, **_kwargs):
        # Simulate the #95548 zombie: the loop makes no progress and never
        # touches the activity clock. It only notices the watchdog's
        # hard interrupt (real wedges may not even do that — see the lease
        # test below).
        while not _agent._interrupt_requested:
            if time.time() - t_start > 10:
                break
            time.sleep(0.005)
        interrupt_seen["at"] = time.time()
        interrupt_seen["message"] = _agent._interrupt_message
        return {
            "final_response": "aborted",
            "messages": history,
            "api_calls": 0,
            "completed": False,
            "interrupted": True,
        }

    with caplog.at_level(logging.ERROR, logger="agent.turn_liveness"):
        result = _run_turn(agent, stalled_loop, monkeypatch)

    elapsed = time.time() - t_start

    # The turn was surfaced as interrupted, not hung.
    assert result["interrupted"] is True
    assert result["final_response"] == "aborted"
    # The watchdog fired before our 10s outer bound, and after the 0.3s idle
    # bound (poll interval makes the exact fire instant approximate).
    assert 0.2 <= elapsed < 10.0
    # The stall was logged loudly with the session named.
    assert any(
        "Turn liveness watchdog fired" in record.getMessage()
        and "stalled-session" in record.getMessage()
        for record in caplog.records
    )
    # The interrupt message tells the UI why the turn ended.
    assert agent._interrupt_message is None  # cleared by the wrapper's finally
    assert "no progress" in (interrupt_seen.get("message") or "")  # watchdog fired
    # The durable lease was released on the interrupted exit path.
    assert db.events[-1][0] == "release"
    assert db.events[-1][1] == "stalled-session"


def test_watchdog_does_not_fire_while_turn_still_making_progress(
    watchdog_config, monkeypatch, caplog
):
    """A turn that keeps touching the activity clock (API waits, stream
    tokens, tool heartbeats, tool completions) is never force-aborted, and
    the lease keeps renewing normally."""
    db = _DB()
    agent = _agent_with_db(db)
    # Fast lease refresh so the test can prove renewal stayed alive.
    agent._session_turn_lease_refresh_interval = 0.05
    t_start = time.time()

    def busy_loop(_agent, _message, _system, history, *_args, **_kwargs):
        # Keep making progress well past the 0.3s idle bound.
        while time.time() - t_start < 0.6:
            _agent._touch_activity("test tick")
            time.sleep(0.02)
        return {
            "final_response": "done",
            "messages": history,
            "api_calls": 1,
            "completed": True,
        }

    with caplog.at_level(logging.ERROR, logger="agent.turn_liveness"):
        result = _run_turn(agent, busy_loop, monkeypatch)

    assert result["completed"] is True
    assert result.get("interrupted") is not True
    assert agent._interrupt_requested is False
    assert not any(
        "Turn liveness watchdog fired" in record.getMessage()
        for record in caplog.records
    )
    # The lease refresher ran during the turn — renewal is orthogonal to the
    # watchdog and continued while the turn was alive.
    assert len(db.refresh_times) >= 1


def test_watchdog_stops_lease_renewal_when_interrupt_cannot_unwind_wedge(
    watchdog_config, monkeypatch
):
    """The issue's 'lease keeps renewing' masking: even when the hard
    interrupt cannot immediately unwind the loop (a truly wedged frame), the
    watchdog stops renewing the durable lease so TTL expiry lets stale-turn
    cleanup reclaim the session."""
    db = _DB()
    agent = _agent_with_db(db)
    agent._session_turn_lease_refresh_interval = 0.05
    t_start = time.time()
    fire_ts = {}

    def wedged_loop(_agent, _message, _system, history, *_args, **_kwargs):
        # Notice the interrupt but keep "wedging" (no activity) for another
        # half second — simulating a blocked frame the interrupt cannot free
        # immediately.
        while not _agent._interrupt_requested:
            if time.time() - t_start > 10:
                break
            time.sleep(0.005)
        fire_ts["at"] = time.time()
        while time.time() - fire_ts["at"] < 0.5:
            time.sleep(0.005)
        return {
            "final_response": "recovered",
            "messages": history,
            "api_calls": 0,
            "completed": True,
        }

    result = _run_turn(agent, wedged_loop, monkeypatch)
    assert result["completed"] is True
    assert time.time() - t_start < 10.0

    # The watchdog fired during the wedge.
    assert "at" in fire_ts
    # Once the watchdog fired, lease renewal stopped: refreshes cadence at
    # 0.05s would have produced ~8 more events in the 0.5s post-fire wedge if
    # renewal had continued. Tolerate only in-flight refreshes racing the
    # stop (<= 0.15s window).
    late_refreshes = [
        t for t in db.refresh_times if t > fire_ts["at"] + 0.15
    ]
    assert late_refreshes == [], (
        f"lease renewal continued after watchdog fired: {len(late_refreshes)} "
        "post-fire refreshes"
    )
    # The lease row was still released when the turn finally unwound.
    assert [e[0] for e in db.events][-1] == "release"


def test_watchdog_declines_abort_when_activity_resumes_during_warning(
    watchdog_config, monkeypatch, caplog
):
    """#95663 review race: the watchdog sampled a stale activity clock, then
    the turn resumed while the stall was being logged/emitted — before the
    abort commits. The commit point revalidates the observed
    (generation, timestamp) pair under the lock shared with
    `_touch_activity`, so the turn continues and its lease keeps renewing
    instead of being hard-cancelled mid-recovery."""
    db = _DB()
    agent = _agent_with_db(db)
    agent._session_turn_lease_refresh_interval = 0.05
    resume_event = threading.Event()
    touched_event = threading.Event()
    t_start = time.time()
    interrupt_seen = {}

    def stalled_then_resumed_loop(_agent, _message, _system, history, *_a, **_k):
        # Stall (zero activity) until the watchdog surfaces the stall. The
        # warning delivery itself is what unblocks the wedge: the turn
        # resumes DURING the warning window, before the commit point.
        assert resume_event.wait(10.0)
        _agent._touch_activity("turn resumed")
        touched_event.set()
        # Keep the turn alive a while so lease renewal is observable.
        while time.time() - t_start < 0.8:
            _agent._touch_activity("still alive")
            time.sleep(0.02)
        # Capture DURING the turn — the wrapper's finally clears any
        # interrupt after the loop returns, so post-run assertions on the
        # agent's interrupt state cannot prove the abort never happened.
        interrupt_seen["requested"] = _agent._interrupt_requested
        interrupt_seen["message"] = _agent._interrupt_message
        return {
            "final_response": "recovered",
            "messages": history,
            "api_calls": 1,
            "completed": True,
        }

    def blocking_warning(_message):
        # Mirrors _emit_warning: the stall warning is being delivered when
        # the turn resumes. Hold the window open until the resumed activity
        # has definitively landed on the clock, so the commit revalidation
        # runs against the new generation.
        resume_event.set()
        assert touched_event.wait(10.0)

    agent._emit_warning = blocking_warning

    with caplog.at_level(logging.ERROR, logger="agent.turn_liveness"):
        result = _run_turn(agent, stalled_then_resumed_loop, monkeypatch)

    assert time.time() - t_start < 10.0
    # The turn resumed and completed normally — it was NOT hard-cancelled
    # mid-flight (the commit point declined the stale observation).
    assert result["completed"] is True
    assert result.get("interrupted") is not True
    assert interrupt_seen.get("requested") is False, (
        f"turn was hard-interrupted while resumed: {interrupt_seen!r}"
    )
    # The stall was still surfaced loudly (so the regression isn't a
    # vacuous pass from the watchdog never sampling)…
    assert any(
        "Turn liveness watchdog fired" in record.getMessage()
        and "stalled-session" in record.getMessage()
        for record in caplog.records
    )
    # …but the surface was OBSERVATIONAL only: the declined abort must
    # never publish a committed-abort or lease-stop settlement
    # (#95663 review — false settlement before commit veto). On the
    # pre-fix tree the pre-commit surface itself logged the definitive
    # "Force-aborting … stopping lease renewal" outcome — this assertion
    # is what made that witness red.
    assert not any(
        "watchdog aborted turn" in record.getMessage()
        for record in caplog.records
    ), "declined abort published a committed-abort settlement"
    assert not any(
        "Force-aborting" in record.getMessage()
        for record in caplog.records
    ), "pre-commit surface published the definitive abort outcome"
    # …and the lease kept renewing through the resumed turn.
    assert len(db.refresh_times) >= 1
    assert db.events[-1][0] == "release"


def test_watchdog_publishes_definitive_settlement_only_after_commit(
    watchdog_config, monkeypatch, caplog
):
    """#95663 review: the definitive aborted/lease-stopped settlement is
    published only AFTER the abort has authority (commit succeeded and
    the turn lease was deactivated). The committed path must show the
    settlement; the pre-commit surface must not claim it."""
    db = _DB()
    agent = _agent_with_db(db)
    warnings = []

    def stalled_loop(_agent, _message, _system, history, *_args, **_kwargs):
        while not _agent._interrupt_requested:
            time.sleep(0.005)
        return {
            "final_response": "aborted",
            "messages": history,
            "api_calls": 0,
            "completed": False,
            "interrupted": True,
        }

    agent._emit_warning = lambda msg: warnings.append(msg)

    with caplog.at_level(logging.ERROR, logger="agent.turn_liveness"):
        result = _run_turn(agent, stalled_loop, monkeypatch)

    assert result["interrupted"] is True
    # Pre-commit surface: observational, recovery-attempt language.
    assert any(
        "Turn liveness watchdog fired" in record.getMessage()
        and "Attempting recovery" in record.getMessage()
        for record in caplog.records
    )
    # Definitive settlement: only present because the abort committed.
    assert any(
        "watchdog aborted turn" in record.getMessage()
        and "lease renewal stopped" in record.getMessage()
        for record in caplog.records
    ), "committed abort did not publish the definitive settlement"
    # User-visible warnings follow the same split: first observational,
    # then (and only then) the committed outcome.
    assert any("attempting recovery" in w for w in warnings)
    assert any("Turn aborted by the liveness watchdog" in w for w in warnings)
    # Ordering: the committed-abort warning came after the recovery one.
    recovery_idx = next(i for i, w in enumerate(warnings) if "attempting recovery" in w)
    aborted_idx = next(
        i for i, w in enumerate(warnings) if "Turn aborted by the liveness watchdog" in w
    )
    assert aborted_idx > recovery_idx
    assert db.events[-1][0] == "release"


def test_watchdog_declines_abort_when_activity_resumes_after_revalidation(
    watchdog_config, monkeypatch, caplog
):
    """#95663 round-3 review race: the commit point revalidates the observed
    (generation, timestamp) pair under the activity lock and then releases
    it before the hard interrupt. Real progress landing in that
    post-revalidation window published a new generation yet was still
    hard-cancelled by the already-authorized abort. The abort now carries
    its revalidated generation into the interrupt path, which re-compares
    it against the live clock at the last instant before the hammer and
    abandons the stale claim — the turn continues and its lease keeps
    renewing."""
    db = _DB()
    agent = _agent_with_db(db)
    agent._session_turn_lease_refresh_interval = 0.05
    t_start = time.time()
    interrupt_seen = {}
    injected = {"count": 0}

    real_interrupt = agent.interrupt

    def intercepting_interrupt(
        message=None,
        *,
        hard_cancel=False,
        tool_reason=None,
        require_generation=None,
    ):
        # Deterministically land real progress in the exact
        # post-revalidation / pre-interrupt window: every abort hammer
        # attempt is preceded by a fresh activity stamp on the clock.
        injected["count"] += 1
        agent._touch_activity("resumed after revalidation")
        if require_generation is not None:
            return real_interrupt(
                message,
                hard_cancel=hard_cancel,
                tool_reason=tool_reason,
                require_generation=require_generation,
            )
        return real_interrupt(
            message, hard_cancel=hard_cancel, tool_reason=tool_reason
        )

    agent.interrupt = intercepting_interrupt

    def stalled_loop(_agent, _message, _system, history, *_args, **_kwargs):
        # Stall without touching the clock: the watchdog fires and tries to
        # abort; the interceptor above injects the resumed activity at the
        # commit's hammer point. Keep the turn alive long enough for the
        # abort decision to play out and record whether the interrupt was
        # ever published.
        while time.time() - t_start < 1.0:
            if _agent._interrupt_requested:
                interrupt_seen["requested"] = True
                break
            time.sleep(0.005)
        # Capture DURING the turn — the wrapper's finally clears interrupt
        # state after the loop returns.
        if "requested" not in interrupt_seen:
            interrupt_seen["requested"] = _agent._interrupt_requested
        interrupt_seen["message"] = _agent._interrupt_message
        return {
            "final_response": "recovered",
            "messages": history,
            "api_calls": 1,
            "completed": True,
        }

    with caplog.at_level(logging.ERROR, logger="agent.turn_liveness"):
        result = _run_turn(agent, stalled_loop, monkeypatch)

    assert time.time() - t_start < 10.0
    # The window was really hit: the interceptor's injected activity ran.
    assert injected["count"] >= 1
    # The turn resumed and completed normally — the stale abort claim was
    # abandoned instead of hard-cancelling the resumed turn.
    assert result["completed"] is True
    assert result.get("interrupted") is not True
    assert interrupt_seen.get("requested") is False, (
        f"turn was hard-interrupted although activity resumed after "
        f"revalidation: {interrupt_seen!r}"
    )
    # The stall was still surfaced loudly…
    assert any(
        "Turn liveness watchdog fired" in record.getMessage()
        and "stalled-session" in record.getMessage()
        for record in caplog.records
    )
    # …and the lease kept renewing through the resumed turn.
    assert len(db.refresh_times) >= 1
    assert db.events[-1][0] == "release"


def test_watchdog_declines_abort_when_activity_resumes_inside_interrupt_publication(
    watchdog_config, monkeypatch, caplog
):
    """#95663 round-4 review race: the generation claim must survive every
    blocking boundary inside ``interrupt()`` — including the compression
    commit fence — and be consumed at the final mutation edge, immediately
    before the first observable publication.

    The witness parks ``interrupt()`` inside a controllable
    ``cancel_before_commit`` AFTER its internal generation comparison
    passed, resumes the turn with real progress (``_touch_activity``
    publishes generation G+1) while the hammer is parked, then releases the
    fence. The abort must abandon itself: no ``_interrupt_requested`` flag,
    no hard-cancel event, no tool signal, and the turn's lease keeps
    renewing.
    """
    db = _DB()
    agent = _agent_with_db(db)
    agent._session_turn_lease_refresh_interval = 0.05
    fence = _BlockingCommitFence()
    agent._active_compression_commit_fence = fence
    t_start = time.time()
    published = {"requested": False, "hard_event": False, "tool_signal": False}

    def stalled_then_resumed_loop(_agent, _message, _system, history, *_a, **_k):
        # Stall (zero activity) until the watchdog's abort attempt is parked
        # inside the compression fence — i.e. AFTER interrupt() passed its
        # internal generation comparison. Then resume with real progress
        # while the hammer is mid-flight.
        assert fence.entered.wait(10.0), "interrupt never reached the fence"
        _agent._touch_activity("resumed inside interrupt publication window")
        fence.release.set()
        # Keep the turn making REAL progress while observing whether the
        # parked interrupt published anything. Continuous activity also
        # guarantees no NEW legitimate watchdog decision can fire, so any
        # publication observed here is attributable to the stale attempt.
        deadline = time.time() + 1.0
        while time.time() < deadline:
            _agent._touch_activity("still alive")
            if _agent._interrupt_requested or _agent._hard_interrupt_requested.is_set():
                break
            time.sleep(0.02)
        published["requested"] = _agent._interrupt_requested
        published["hard_event"] = _agent._hard_interrupt_requested.is_set()
        published["tool_signal"] = _agent._interrupt_thread_signal_pending
        published["message"] = _agent._interrupt_message
        return {
            "final_response": "recovered",
            "messages": history,
            "api_calls": 1,
            "completed": True,
        }

    with caplog.at_level(logging.ERROR, logger="agent.turn_liveness"):
        result = _run_turn(agent, stalled_then_resumed_loop, monkeypatch)

    assert time.time() - t_start < 10.0
    # The window was really hit: the interrupt entered the fence exactly
    # once — the stale attempt declined and no retry ever fired.
    assert fence.calls == 1, f"unexpected fence admissions: {fence.calls}"
    # The resumed turn completed normally — the stale claim was abandoned
    # at the final mutation edge without publishing any interrupt state.
    assert result["completed"] is True
    assert result.get("interrupted") is not True
    assert published["requested"] is False, (
        f"_interrupt_requested published against stale generation: {published!r}"
    )
    assert published["hard_event"] is False, (
        f"hard-cancel event published against stale generation: {published!r}"
    )
    assert published["tool_signal"] is False, (
        f"tool interrupt signal published against stale generation: {published!r}"
    )
    assert published["message"] is None
    # The stall was still surfaced loudly…
    assert any(
        "Turn liveness watchdog fired" in record.getMessage()
        and "stalled-session" in record.getMessage()
        for record in caplog.records
    )
    # …and the lease kept renewing through the resumed turn.
    assert len(db.refresh_times) >= 1
    assert db.events[-1][0] == "release"


def test_watchdog_declines_abort_when_interrupt_publish_raises(
    watchdog_config, monkeypatch, caplog
):
    """#95663 round-4 review: the exceptional interrupt path must not
    convert an unvalidated generation claim into direct interrupt-flag
    mutation. When ``interrupt()`` raises, the abort declines fail-closed —
    no ``_interrupt_requested`` / ``_interrupt_message`` is set, the
    watchdog keeps sampling, and the turn's lease keeps renewing.
    """
    db = _DB()
    agent = _agent_with_db(db)
    agent._session_turn_lease_refresh_interval = 0.05
    t_start = time.time()
    raised = {"count": 0}
    seen = {"requested": False, "message": None}

    def raising_interrupt(
        message=None,
        *,
        hard_cancel=False,
        tool_reason=None,
        require_generation=None,
    ):
        raised["count"] += 1
        raise RuntimeError("synthetic interrupt publication failure")

    agent.interrupt = raising_interrupt

    def stalled_loop(_agent, _message, _system, history, *_args, **_kwargs):
        # Stall so the watchdog fires; every abort attempt raises inside
        # interrupt(). Run a fixed window while recording whether the
        # exception fallback published interrupt state anyway.
        while time.time() - t_start < 1.0:
            seen["requested"] = _agent._interrupt_requested
            seen["message"] = _agent._interrupt_message
            if _agent._interrupt_requested:
                break
            time.sleep(0.005)
        seen["requested"] = _agent._interrupt_requested
        seen["message"] = _agent._interrupt_message
        return {
            "final_response": "recovered",
            "messages": history,
            "api_calls": 1,
            "completed": True,
        }

    with caplog.at_level(logging.ERROR, logger="agent.turn_liveness"):
        result = _run_turn(agent, stalled_loop, monkeypatch)

    assert time.time() - t_start < 10.0
    # The exceptional abort attempt really happened inside the window.
    assert raised["count"] >= 1
    # The turn completed without being interrupted…
    assert result["completed"] is True
    assert result.get("interrupted") is not True
    # …and the exception fallback published NOTHING: fail-closed, no flag,
    # no message.
    assert seen["requested"] is False, (
        f"exception fallback mutated interrupt state: {seen!r}"
    )
    assert seen["message"] is None
    # The stall was surfaced loudly (the watchdog really fired)…
    assert any(
        "Turn liveness watchdog fired" in record.getMessage()
        and "stalled-session" in record.getMessage()
        for record in caplog.records
    )
    # …and the lease kept renewing because the abort was declined.
    assert len(db.refresh_times) >= 1
    assert db.events[-1][0] == "release"


def test_interrupt_consumes_claim_and_publishes_first_state_atomically():
    """#95663 round-6 review race: claim consumption and the first
    interrupt publication must be ONE activity-lock critical section.

    The round-4 tree consumed the generation claim under the activity
    lock, released it, and only then published ``_interrupt_requested``
    / ``_interrupt_message`` / ``_tool_interrupt_reason`` — so a turn
    that resumed in that window (real progress, G+1) was still
    hard-cancelled by the already-consumed claim. This witness parks
    the interrupt thread exactly at the claim-publication boundary (the
    Nth activity-lock release), publishes real activity on a competing
    thread while the hammer is parked, and asserts the total order: any
    interrupt publication must have committed under the lock BEFORE the
    competing activity stamp. Publishing AFTER it is the round-6 defect
    and makes this test red on that tree.
    """
    agent = AIAgent.__new__(AIAgent)
    agent._turn_liveness_activity_generation = 5
    agent._turn_liveness_abort_claim = None
    agent._interrupt_requested = False
    agent._interrupt_message = None
    agent._tool_interrupt_reason = None
    agent._hard_interrupt_requested = threading.Event()
    agent._execution_thread_id = None
    agent._active_children_lock = threading.Lock()
    agent._active_children = set()
    agent.quiet_mode = True

    parking_lock = _ParkingReleaseLock(threading.Lock())
    # Release #1 is the claim RESERVATION at interrupt() entry; release
    # #2 is the CONSUME on the round-4 tree / the atomic
    # consume+publication critical section on the repaired tree. Parking
    # on release #2 puts the competing activity exactly at the
    # claim-publication boundary under test.
    parking_lock.park_on_release = 2
    agent._turn_liveness_activity_lock = parking_lock

    result = {}

    def interrupt_fn():
        result["ret"] = AIAgent.interrupt(
            agent,
            "watchdog: no progress",
            hard_cancel=True,
            tool_reason="turn liveness watchdog fired",
            require_generation=5,
        )

    interrupt_thread = threading.Thread(target=interrupt_fn)
    interrupt_thread.start()
    assert parking_lock.parked.wait(10.0), (
        "interrupt never reached the claim-publication boundary"
    )

    # Real progress lands on a competing thread while the hammer is
    # parked: the activity clock advances to generation G+1 (6).
    touched = threading.Event()

    def turn_fn():
        agent._touch_activity("turn resumed")
        touched.set()

    turn_thread = threading.Thread(target=turn_fn)
    turn_thread.start()
    assert touched.wait(10.0), "competing activity never landed"

    def _published():
        return (
            agent._interrupt_requested
            or agent._interrupt_message is not None
            or agent._tool_interrupt_reason is not None
            or agent._hard_interrupt_requested.is_set()
        )

    # The window was really hit: the claim was consumed and the
    # competing activity advanced the generation.
    assert agent._turn_liveness_abort_claim is None
    assert agent._turn_liveness_activity_generation == 6

    published_before_activity = _published()
    parking_lock.release_park.set()
    interrupt_thread.join(10.0)
    turn_thread.join(10.0)
    published_after = _published()

    assert result.get("ret") is True
    assert not (published_after and not published_before_activity), (
        "interrupt state published after competing activity landed: "
        f"before={published_before_activity}, after={published_after}"
    )

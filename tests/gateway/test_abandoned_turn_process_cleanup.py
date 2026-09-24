"""Regression coverage for abandoned gateway-turn subprocess cleanup (#76115)."""

import json
import threading
from contextvars import copy_context

import pytest

from gateway.run import (
    GatewayRunner,
    _abandon_timed_out_gateway_turn,
    _reap_gateway_turn_processes,
    _watch_gateway_turn_inactivity,
)
from gateway.session_context import set_session_vars
from gateway.session_state import SessionState
from model_tools import handle_function_call
from tools.approval_context import set_current_session_key
from tools.process_registry import process_registry


class _IdleAgent:
    def __init__(self, idle_seconds=60.0):
        self.idle_seconds = idle_seconds
        self.interrupts = []

    def get_activity_summary(self):
        return {"seconds_since_activity": self.idle_seconds}

    def interrupt(self, reason):
        self.interrupts.append(reason)


class _RaisingActivityAgent:
    """Agent whose activity snapshot read raises (fail-safe case).

    A real AIAgent always carries ``_last_activity_ts``; the snapshot is only
    missing when the diagnostic read itself fails.  The watchdog must still bound
    the turn instead of skipping every poll.
    """

    def __init__(self):
        self.interrupts = []

    def get_activity_summary(self):
        raise RuntimeError("activity snapshot unavailable")

    def interrupt(self, reason):
        self.interrupts.append(reason)


def _state():
    return threading.Event(), threading.Event(), threading.Lock()


def _run_watchdog(agent_holder, task_id, *, worker_done, timeout_fired, cleanup_lock):
    watchdog = threading.Thread(
        target=_watch_gateway_turn_inactivity,
        kwargs={
            "agent_holder": agent_holder,
            "task_id": task_id,
            "process_baseline": frozenset(),
            "timeout": 0.03,
            "worker_done": worker_done,
            "timeout_fired": timeout_fired,
            "cleanup_lock": cleanup_lock,
            "poll_interval": 0.01,
        },
    )
    watchdog.start()
    watchdog.join(timeout=2)
    if watchdog.is_alive():
        worker_done.set()
        watchdog.join(timeout=2)
    assert not watchdog.is_alive()
    return watchdog


def test_thread_watchdog_times_out_before_agent_exists(monkeypatch):
    """``agent_holder`` is still ``[None]`` while the turn is being set up
    (``run_turn_runner`` fills it later).  A turn wedged in that window has no
    activity snapshot, so the watchdog must fall back to elapsed wall-clock time
    and still reap the turn rather than skip every poll forever."""
    worker_done, timeout_fired, cleanup_lock = _state()
    calls = []
    monkeypatch.setattr(
        process_registry,
        "kill_started_since",
        lambda task_id, baseline, *, source: calls.append((task_id, baseline, source)) or 0,
    )

    _run_watchdog(
        [None], "session-before-agent",
        worker_done=worker_done, timeout_fired=timeout_fired, cleanup_lock=cleanup_lock,
    )

    assert timeout_fired.is_set()
    assert calls == [("session-before-agent", frozenset(), "gateway_turn_timeout")]


def test_thread_watchdog_times_out_when_activity_snapshot_raises(monkeypatch):
    """Fail-safe: a snapshot read that raises must not disable the watchdog; the
    wall-clock fallback bounds the turn."""
    agent = _RaisingActivityAgent()
    worker_done, timeout_fired, cleanup_lock = _state()
    calls = []
    monkeypatch.setattr(
        process_registry,
        "kill_started_since",
        lambda task_id, baseline, *, source: calls.append((task_id, baseline, source)) or 0,
    )

    _run_watchdog(
        [agent], "session-snapshot-raises",
        worker_done=worker_done, timeout_fired=timeout_fired, cleanup_lock=cleanup_lock,
    )

    assert timeout_fired.is_set()
    assert agent.interrupts == ["Execution timed out (inactivity)"]
    assert calls == [("session-snapshot-raises", frozenset(), "gateway_turn_timeout")]


def test_thread_watchdog_reaps_only_processes_created_by_timed_out_turn(monkeypatch):
    agent = _IdleAgent()
    worker_done, timeout_fired, cleanup_lock = _state()
    calls = []
    monkeypatch.setattr(
        process_registry,
        "kill_started_since",
        lambda task_id, baseline, *, source: calls.append(
            (task_id, baseline, source)
        )
        or 1,
    )

    watchdog = threading.Thread(
        target=_watch_gateway_turn_inactivity,
        kwargs={
            "agent_holder": [agent],
            "task_id": "session-a",
            "process_baseline": frozenset({"proc_existing"}),
            "timeout": 30.0,
            "worker_done": worker_done,
            "timeout_fired": timeout_fired,
            "cleanup_lock": cleanup_lock,
            "poll_interval": 0.01,
        },
    )
    watchdog.start()
    watchdog.join(timeout=1)

    assert not watchdog.is_alive()
    assert timeout_fired.is_set()
    assert agent.interrupts == ["Execution timed out (inactivity)"]
    assert calls == [
        (
            "session-a",
            frozenset({"proc_existing"}),
            "gateway_turn_timeout",
        )
    ]


def test_completed_worker_wins_race_and_preserves_background_process(monkeypatch):
    agent = _IdleAgent()
    worker_done, timeout_fired, cleanup_lock = _state()
    worker_done.set()
    monkeypatch.setattr(
        process_registry,
        "kill_started_since",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("completed turn must not reap background work")
        ),
    )

    assert not _abandon_timed_out_gateway_turn(
        agent_holder=[agent],
        task_id="session-a",
        process_baseline=frozenset(),
        worker_done=worker_done,
        timeout_fired=timeout_fired,
        cleanup_lock=cleanup_lock,
    )
    assert not timeout_fired.is_set()
    assert agent.interrupts == []


def test_timeout_cleanup_is_idempotent(monkeypatch):
    agent = _IdleAgent()
    worker_done, timeout_fired, cleanup_lock = _state()
    calls = []
    monkeypatch.setattr(
        process_registry,
        "kill_started_since",
        lambda *_args, **_kwargs: calls.append(True) or 0,
    )
    kwargs = {
        "agent_holder": [agent],
        "task_id": "session-a",
        "process_baseline": frozenset(),
        "worker_done": worker_done,
        "timeout_fired": timeout_fired,
        "cleanup_lock": cleanup_lock,
    }

    assert _abandon_timed_out_gateway_turn(**kwargs)
    assert not _abandon_timed_out_gateway_turn(**kwargs)
    assert len(calls) == 1
    assert len(agent.interrupts) == 1


# ---------------------------------------------------------------------------
# Cross-turn race guard (#76188 review): task_id is session-scoped, not
# turn-scoped, so a replacement turn on the same session could otherwise
# have its freshly-spawned process killed by a stale reaper. Gated on
# run_generation via an injected `is_still_current` check.
# ---------------------------------------------------------------------------


def test_reap_skips_when_a_newer_turn_has_claimed_the_session(monkeypatch):
    calls = []
    monkeypatch.setattr(
        process_registry,
        "kill_started_since",
        lambda *_a, **_k: calls.append(True) or 1,
    )

    killed = _reap_gateway_turn_processes(
        "session-a",
        frozenset({"proc_old"}),
        source="gateway_turn_timeout",
        is_still_current=lambda: False,
    )

    assert killed == 0
    assert calls == []


def test_reap_proceeds_when_this_turn_is_still_current(monkeypatch):
    calls = []
    monkeypatch.setattr(
        process_registry,
        "kill_started_since",
        lambda task_id, baseline, *, source: calls.append(
            (task_id, baseline, source)
        )
        or 1,
    )

    killed = _reap_gateway_turn_processes(
        "session-a",
        frozenset({"proc_old"}),
        source="gateway_turn_timeout",
        is_still_current=lambda: True,
    )

    assert killed == 1
    assert calls == [("session-a", frozenset({"proc_old"}), "gateway_turn_timeout")]


def test_reap_fails_open_when_is_still_current_raises(monkeypatch):
    """A bug in the generation-check closure must not silently disable the
    underlying leak fix — it should log and fall through to reaping."""
    calls = []
    monkeypatch.setattr(
        process_registry,
        "kill_started_since",
        lambda *_a, **_k: calls.append(True) or 1,
    )

    def _boom():
        raise RuntimeError("session state lookup failed")

    killed = _reap_gateway_turn_processes(
        "session-a",
        frozenset(),
        source="gateway_turn_timeout",
        is_still_current=_boom,
    )

    assert killed == 1
    assert calls == [True]


def test_reap_skips_empty_task_id(monkeypatch):
    """ProcessSession.task_id defaults to "" — a blank turn id must never
    fan out into killing unrelated sessionless processes (#76188 review)."""
    calls = []
    monkeypatch.setattr(
        process_registry,
        "kill_started_since",
        lambda *_a, **_k: calls.append(True) or 1,
    )

    killed = _reap_gateway_turn_processes(
        "",
        frozenset(),
        source="gateway_turn_timeout",
    )

    assert killed == 0
    assert calls == []


def test_timeout_abandon_propagates_is_still_current_to_the_reap(monkeypatch):
    agent = _IdleAgent()
    worker_done, timeout_fired, cleanup_lock = _state()
    calls = []
    monkeypatch.setattr(
        process_registry,
        "kill_started_since",
        lambda *_a, **_k: calls.append(True) or 1,
    )

    assert _abandon_timed_out_gateway_turn(
        agent_holder=[agent],
        task_id="session-a",
        process_baseline=frozenset(),
        worker_done=worker_done,
        timeout_fired=timeout_fired,
        cleanup_lock=cleanup_lock,
        is_still_current=lambda: False,
    )

    # The turn was still marked abandoned (interrupt fired), but the actual
    # reap was skipped because a newer turn already claimed the session.
    assert agent.interrupts == ["Execution timed out (inactivity)"]
    assert calls == []


# ---------------------------------------------------------------------------
# Wedged-turn stack dump at reap time (Aug 2026 zombie-turn incident):
# the reaper's interrupt frees the blocked frame, so the dump must run
# BEFORE the interrupt and must capture the actual wedged stack.
# ---------------------------------------------------------------------------


def _run_wedged_worker(release: threading.Event, entered: threading.Event):
    """Worker blocked inside a frame named like turn machinery."""

    def run_sync():  # marker frame the dump filter matches on
        entered.set()
        release.wait(timeout=30.0)

    run_sync()


def test_reaper_dumps_wedged_worker_stack_before_interrupt(monkeypatch, caplog):
    import logging

    from gateway.run import _dump_wedged_turn_stacks

    release = threading.Event()
    entered = threading.Event()
    worker = threading.Thread(
        target=_run_wedged_worker,
        args=(release, entered),
        name="wedged-test-worker",
        daemon=True,
    )
    worker.start()
    try:
        assert entered.wait(timeout=5.0)
        with caplog.at_level(logging.ERROR, logger="gateway.run"):
            _dump_wedged_turn_stacks("task-wedge-test")
        dumps = [
            r for r in caplog.records if "Wedged-turn stack dump" in r.getMessage()
        ]
        assert dumps, "no stack dump was logged"
        joined = "\n".join(r.getMessage() for r in dumps)
        assert "wedged-test-worker" in joined
        assert "run_sync" in joined
        assert "release.wait" in joined  # the actual blocked line is named
    finally:
        release.set()
        worker.join(timeout=5.0)


def test_abandon_timed_out_turn_dumps_stacks_before_interrupt(monkeypatch):
    """The dump hook runs inside the reaper, before the agent interrupt."""
    import gateway.run as gateway_run

    order = []
    monkeypatch.setattr(
        gateway_run,
        "_dump_wedged_turn_stacks",
        lambda task_id: order.append(("dump", task_id)),
    )
    monkeypatch.setattr(
        gateway_run,
        "_reap_gateway_turn_processes",
        lambda *a, **k: order.append(("reap",)),
    )

    class _Agent:
        def interrupt(self, reason):
            order.append(("interrupt", reason))

    worker_done, timeout_fired, cleanup_lock = _state()
    assert _abandon_timed_out_gateway_turn(
        agent_holder=[_Agent()],
        task_id="t-dump-order",
        process_baseline=frozenset(),
        worker_done=worker_done,
        timeout_fired=timeout_fired,
        cleanup_lock=cleanup_lock,
    )
    assert order[0] == ("dump", "t-dump-order")
    assert ("interrupt", order[1][1]) == order[1]
    assert order[-1] == ("reap",)


def test_dump_wedged_turn_stacks_never_raises(monkeypatch):
    import gateway.run as gateway_run

    monkeypatch.setattr(
        gateway_run.sys,
        "_current_frames",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    from gateway.run import _dump_wedged_turn_stacks

    _dump_wedged_turn_stacks("t-no-raise")  # must not raise


# ---------------------------------------------------------------------------
# Real spawns: the terminal tool stores the CONTAINER key in ``task_id``
# (``session:<key>``, ``default``) and the turn's own id in ``owner_task_id``;
# the reap is keyed by the turn's id, so it must match on the owner.
# ---------------------------------------------------------------------------


@pytest.fixture
def _plain_spawn(monkeypatch):
    """The systemd-run --user --scope wrapper is irrelevant here and stalls under pytest."""
    import tools.process_registry as _pr
    monkeypatch.setattr(_pr, "_SYSTEMD_SCOPE_AVAILABLE", False)


def _bind_session(session_key, session_id, platform="telegram"):
    set_session_vars(platform=platform, chat_id=session_id, user_id=session_id,
                     session_key=session_key, session_id=session_id)
    set_current_session_key(session_key)


def _spawn_background(turn_task_id, spawned):
    out = json.loads(handle_function_call(
        "terminal", {"command": "sleep 300", "background": True}, task_id=turn_task_id))
    spawned.append(process_registry.get(out["session_id"]))
    return spawned[-1]


def _kill_own(spawned):
    for session in spawned:
        if session is not None:
            process_registry.kill_process(session.id)


def test_stop_reaps_the_background_job_the_stopped_turn_started(_plain_spawn):
    """/stop (shared with /new and eviction) reaps the job the stopped turn started and keeps
    the session's older job, on the local backend's ``session:<key>`` container key."""
    session_key, session_id = "agent:main:telegram:dm:76115", "20260924_stop_76115"
    spawned = []

    def scenario():
        _bind_session(session_key, session_id)
        older = _spawn_background(session_id, spawned)
        # Turn start, as the gateway turn does it: baseline keyed by the turn's session id.
        agent = _IdleAgent()
        agent._gateway_turn_process_task_id = session_id
        agent._gateway_turn_process_baseline = process_registry.snapshot_running_ids(session_id)
        job = _spawn_background(session_id, spawned)
        assert job.task_id != job.owner_task_id == session_id

        runner = object.__new__(GatewayRunner)
        runner._sessions = {session_key: SessionState()}
        runner._sessions[session_key].turn.agent = agent
        runner._interrupt_running_turn(
            session_key, interrupt_reason="Stop requested", invalidation_reason="stop_command_handler")

        assert agent.interrupts == ["Stop requested"]
        assert process_registry.wait(job.id, timeout=30)["status"] == "exited"
        assert process_registry.poll(older.id)["status"] == "running"

    try:
        copy_context().run(scenario)
    finally:
        _kill_own(spawned)


def test_timed_out_turn_reaps_only_its_own_job_on_a_shared_container_key(_plain_spawn):
    """Two sessions on ONE container key (``default``: keyless API sessions here, every
    default-profile session under persistent Docker). The timed-out turn lists and loses only
    the job it started; the sibling session's job, started after the baseline, survives."""
    turn_a, turn_b = "api-session-a-76115", "api-session-b-76115"
    spawned = []

    def scenario():
        _bind_session("", turn_a, platform="api_server")
        baseline = process_registry.snapshot_running_ids(turn_a)
        mine = _spawn_background(turn_a, spawned)
        _bind_session("", turn_b, platform="api_server")
        sibling = _spawn_background(turn_b, spawned)
        assert mine.task_id == sibling.task_id and mine.owner_task_id != sibling.owner_task_id

        _bind_session("", turn_a, platform="api_server")
        listed = json.loads(handle_function_call("process_manage", {"action": "list"}, task_id=turn_a))
        assert [p["session_id"] for p in listed["processes"]] == [mine.id]

        agent, worker_done, timeout_fired, cleanup_lock = _IdleAgent(), *_state()
        _watch_gateway_turn_inactivity(
            agent_holder=[agent], task_id=turn_a, process_baseline=baseline, timeout=0.05,
            worker_done=worker_done, timeout_fired=timeout_fired, cleanup_lock=cleanup_lock,
            poll_interval=0.01)

        assert timeout_fired.is_set()
        assert process_registry.wait(mine.id, timeout=30)["status"] == "exited"
        assert process_registry.poll(sibling.id)["status"] == "running"

    try:
        copy_context().run(scenario)
    finally:
        _kill_own(spawned)

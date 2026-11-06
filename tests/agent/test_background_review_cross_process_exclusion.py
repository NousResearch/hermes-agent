"""Cross-process exclusion of the background review: an explicit /refine holds the durable session
turn lease too, and a foreground waiter wins it. Split from
``test_background_review_compute_overlap.py`` (same agent double and admission-registry reset)."""

from __future__ import annotations

import threading
import types

import pytest

from agent import background_review as background_review_module
from agent import review_admission
from tests.agent.test_background_review_compute_overlap import (
    _bare_agent,
    _reset_review_admission_state,
)


@pytest.fixture(autouse=True)
def clear_review_admission_state():
    _reset_review_admission_state()
    yield
    _reset_review_admission_state()


def _durable_fork_stub(run_conversation):
    """Minimal fork double for ``_run_review_fork`` with a real durable lease around it."""
    return types.SimpleNamespace(
        _memory_enabled=True,
        _user_profile_enabled=False,
        _session_messages=[],
        _touch_activity=lambda *_a, **_k: None,
        run_conversation=run_conversation,
        release_clients=lambda: None,
        hard_interrupt=lambda *_a, **_k: None,
    )


def _no_liveness_watchdog(monkeypatch):
    import agent.turn_liveness as turn_liveness_module

    monkeypatch.setattr(
        turn_liveness_module,
        "resolve_turn_liveness_settings",
        lambda *_a, **_k: (None, 15.0),
    )


def test_explicit_refine_takes_the_durable_turn_lease(tmp_path, monkeypatch):
    """/refine is exempt from the replay/budget policy, never from exclusion: while its fork
    replays, a foreground turn in ANOTHER process (CLI resume, Desktop) must find the durable
    session row owned."""
    import os

    from hermes_state import SessionDB

    _no_liveness_watchdog(monkeypatch)
    path = tmp_path / "state.db"
    review_db = SessionDB(path)
    foreground_db = SessionDB(path)
    review_db.create_session("shared-session", source="test")
    parent = types.SimpleNamespace(_session_db=review_db)
    entered = threading.Event()
    release = threading.Event()

    def blocking_run(**_kwargs):
        entered.set()
        assert release.wait(timeout=10.0)

    fork = _durable_fork_stub(blocking_run)
    monkeypatch.setattr(
        background_review_module,
        "build_cache_parity_fork",
        lambda *_a, **_k: (fork, {"routed": False}, False),
    )
    run = background_review_module._BackgroundReviewRun()
    st = background_review_module._ReviewForkState()
    worker = threading.Thread(
        target=background_review_module._run_review_fork,
        args=(parent, [{"role": "user", "content": "hi"}], "prompt", None, run, st),
        kwargs={"explicit": True, "review_session_id": "shared-session"},
        daemon=True,
    )
    worker.start()
    foreground_holder = f"pid={os.getpid()}:turn=foreground"
    try:
        assert entered.wait(timeout=10.0)
        assert not foreground_db.try_acquire_session_turn_lease(
            "shared-session", foreground_holder, ttl_seconds=5
        )
    finally:
        release.set()
        worker.join(timeout=10.0)

    assert not worker.is_alive()
    assert foreground_db.try_acquire_session_turn_lease(
        "shared-session", foreground_holder, ttl_seconds=5
    )
    foreground_db.release_session_turn_lease("shared-session", foreground_holder)


def test_explicit_refine_reports_a_session_busy_in_another_process(
    tmp_path, monkeypatch, caplog
):
    """The handler has already said 'reviewing in the background' by the time the fork claims
    the row, so a durable refusal must reach the user through the review's own report channel
    and log the body-free reason with the owner tag."""
    import os

    from hermes_state import SessionDB

    _no_liveness_watchdog(monkeypatch)
    path = tmp_path / "state.db"
    review_db = SessionDB(path)
    foreground_db = SessionDB(path)
    review_db.create_session("shared-session", source="test")
    other_process = f"pid={os.getpid()}:turn=cli-resume:platform=cli"
    assert foreground_db.try_acquire_session_turn_lease(
        "shared-session", other_process, ttl_seconds=60
    )
    agent = _bare_agent("shared-session")
    agent._session_db = review_db
    notices: list[str] = []
    agent.background_review_callback = notices.append
    replayed = []
    fork = _durable_fork_stub(lambda **kwargs: replayed.append(kwargs))
    monkeypatch.setattr(
        background_review_module,
        "build_cache_parity_fork",
        lambda *_a, **_k: (fork, {"routed": False}, False),
    )
    run = background_review_module.prepare_background_review_run(
        agent, followup_cancellable=False, session_id="shared-session"
    )
    assert run is not None

    with caplog.at_level("INFO"):
        background_review_module._run_review_in_thread(
            agent,
            [{"role": "user", "content": "hi"}],
            "prompt",
            review_run=run,
            review_memory=True,
            explicit=True,
            review_session_id="shared-session",
        )
    foreground_db.release_session_turn_lease("shared-session", other_process)

    assert replayed == []
    assert run.request_done.is_set()
    assert len(notices) == 1 and "another Hermes process" in notices[0]
    lines = [
        r.getMessage()
        for r in caplog.records
        if review_admission.REASON_DURABLE_BUSY in r.getMessage()
    ]
    assert len(lines) == 1
    assert _owner_tag_for(run, "shared-session") in lines[0]
    assert "shared-session" not in lines[0]


def _owner_tag_for(run, session_id):
    key = getattr(run, "_review_owner_key", None)
    return (
        review_admission.owner_tag(*key)
        if key
        else review_admission.owner_tag("", session_id)
    )


def test_foreground_waiter_preempts_a_background_review_lease(tmp_path, caplog):
    """Foreground outranks automatic review ACROSS processes: a waiting foreground asks the
    review holder to yield; the review's next renewal tick hard-interrupts its fork, fences its
    run and stops renewing; the waiter acquires once the fork's exit releases the row — never
    while the row is still held (no overlap), and a wedged fork is escalated from the tick."""
    import os

    from hermes_state import SessionDB

    path = tmp_path / "state.db"
    review_db = SessionDB(path)
    foreground_db = SessionDB(path)
    review_db.create_session("shared-session", source="test")
    parent = types.SimpleNamespace(_session_db=review_db)
    interrupted = threading.Event()
    released_clients = []
    fork = types.SimpleNamespace(
        hard_interrupt=lambda *_a, **_k: interrupted.set(),
        release_clients=lambda: released_clients.append(True),
    )
    run = background_review_module._BackgroundReviewRun()
    assert run.begin_request(fork) is True
    lease, reason = background_review_module._try_acquire_durable_review_lease(
        parent, fork, "shared-session", run
    )
    assert reason is None and lease is not None
    clock = [0.0]
    lease._now = lambda: clock[0]
    acquired: list[bool] = []
    foreground_holder = f"pid={os.getpid()}:turn=foreground:platform=cli"
    waiter = threading.Thread(
        target=lambda: acquired.append(
            foreground_db.acquire_session_turn_lease(
                "shared-session",
                foreground_holder,
                ttl_seconds=5,
                wait_seconds=30,
                poll_interval_seconds=0.05,
            )
        ),
        daemon=True,
    )
    with caplog.at_level("INFO"):
        waiter.start()
        try:
            # Bounded: the waiter's next poll writes the yield marker.
            for _ in range(400):
                lease.refresh_tick()
                if interrupted.is_set():
                    break
                interrupted.wait(0.05)
            assert interrupted.is_set()
            assert run.cancel_requested.is_set()
            # Still held: a waiter never reclaims a lease its holder has not released.
            assert acquired == []
            assert not foreground_db.try_acquire_session_turn_lease(
                "shared-session", foreground_holder, ttl_seconds=5
            )
            # A fork that does not unwind is escalated from the tick, not left to the TTL.
            clock[0] += background_review_module._CANCEL_ACK_ESCALATION_SECONDS
            assert lease.refresh_tick() is None
            assert released_clients == [True]
        finally:
            lease.stop_refresher()
            lease.release()  # the fork's finally, once the interrupt unwound it
            waiter.join(timeout=10.0)

    assert acquired == [True]
    assert lease.refresh_tick() is False
    foreground_db.release_session_turn_lease("shared-session", foreground_holder)
    lines = [
        r.getMessage()
        for r in caplog.records
        if review_admission.REASON_PREEMPTED_CROSS_PROCESS in r.getMessage()
    ]
    assert len(lines) == 1
    assert "shared-session" not in lines[0]


def _expire_lease_row(path, holder: str) -> None:
    """Rewind a row's ``expires_at`` on a raw connection: the TTL elapses without a clock or a sleep."""
    import sqlite3
    import time

    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "UPDATE session_turn_leases SET expires_at = ? WHERE holder = ?",
            (time.time() - 1.0, holder),
        )
        conn.commit()
    finally:
        conn.close()


def _shared_review_lease(tmp_path):
    """An admitted review lease on a store shared with a second (foreground) handle."""
    import os

    from hermes_state import SessionDB

    path = tmp_path / "state.db"
    review_db = SessionDB(path)
    foreground_db = SessionDB(path)
    review_db.create_session("shared-session", source="test")
    interrupted = threading.Event()
    fork = types.SimpleNamespace(
        hard_interrupt=lambda *_a, **_k: interrupted.set(),
        release_clients=lambda: None,
    )
    run = background_review_module._BackgroundReviewRun()
    assert run.begin_request(fork) is True
    lease, reason = background_review_module._try_acquire_durable_review_lease(
        types.SimpleNamespace(_session_db=review_db), fork, "shared-session", run
    )
    assert reason is None and lease is not None
    foreground_holder = f"pid={os.getpid()}:turn=foreground:platform=cli"
    return path, foreground_db, lease, run, interrupted, foreground_holder


def test_yielded_review_lease_keeps_renewing_until_the_fork_exits(tmp_path):
    """After a cross-process yield the review keeps renewing its row: the row is released only
    by the fork's own exit, so a waiter's reclaim-on-expiry never lands beside a live fork —
    the TTL is a bound on a DEAD fork, not a fail-open window on a yielded one."""
    path, foreground_db, lease, run, interrupted, fg = _shared_review_lease(tmp_path)
    # The waiting foreground's failed attempt stamps the yield; the next tick honours it.
    assert not foreground_db.try_acquire_session_turn_lease(
        "shared-session", fg, ttl_seconds=5, preempt_background_review=True
    )
    assert lease.refresh_tick() is None
    assert interrupted.is_set()
    assert run.cancel_requested.is_set()

    # The TTL elapses with the fork still alive: the yielded tick must renew.
    _expire_lease_row(path, lease.holder)
    assert lease.refresh_tick() is None
    assert not foreground_db.try_acquire_session_turn_lease(
        "shared-session", fg, ttl_seconds=5, preempt_background_review=True
    )

    lease.stop_refresher()
    lease.release()  # the fork's finally
    assert foreground_db.try_acquire_session_turn_lease(
        "shared-session", fg, ttl_seconds=5
    )
    foreground_db.release_session_turn_lease("shared-session", fg)


def test_transcript_edit_yield_is_logged_with_its_own_cause_and_recorded_on_the_run(
    tmp_path, caplog
):
    """The fork's tick names the cause that stamped its row: a user edit reads as
    ``review_preempted_by_transcript_edit`` — never as a cross-process preemption, so grepping
    the cross-process slug counts only foreground turns from other processes — and the run
    remembers the yield for the requeue policy. A detached delegation delivery is an append
    after the snapshot: it neither stamps the row nor stops the fork."""
    _path, foreground_db, lease, run, interrupted, _fg = _shared_review_lease(tmp_path)
    user_row_id = foreground_db.append_message("shared-session", "user", "ask")
    foreground_db.append_message("shared-session", "assistant", "reply")

    foreground_db.append_delegation_delivery(
        "shared-session", "done", {"delegation_id": "d1"}
    )
    assert lease.refresh_tick() is None
    assert not run.cancel_requested.is_set()
    assert not interrupted.is_set()

    foreground_db.rewind_to_message("shared-session", user_row_id)
    with caplog.at_level("INFO"):
        assert lease.refresh_tick() is None
    assert interrupted.is_set()
    assert run.cancel_requested.is_set()
    assert review_admission.REASON_PREEMPTED_CROSS_PROCESS not in caplog.text
    fork_lines = [
        r.getMessage()
        for r in caplog.records
        if r.name == background_review_module.logger.name
        and review_admission.REASON_PREEMPTED_BY_TRANSCRIPT_EDIT in r.getMessage()
    ]
    assert len(fork_lines) == 1, caplog.text
    assert lease._owner in fork_lines[0]
    assert "shared-session" not in fork_lines[0]
    assert (
        run.lease_yield_reason == review_admission.REASON_PREEMPTED_BY_TRANSCRIPT_EDIT
    )
    lease.stop_refresher()
    lease.release()


@pytest.mark.parametrize("yielded", [False, True], ids=["before_yield", "after_yield"])
def test_review_lease_loss_is_logged_once_and_stops_the_fork(tmp_path, caplog, yielded):
    """A holder-fenced renewal miss means the row was reclaimed under the fork (a dead-PID
    sweep, or a TTL that elapsed with no renewal) and another process may now own the session:
    the fork is fenced and hard-interrupted exactly like a yield, the loss is logged ONCE with
    the hashed owner, and the tick keeps running as the escalation clock."""
    _path, foreground_db, lease, run, interrupted, fg = _shared_review_lease(tmp_path)
    if yielded:
        assert not foreground_db.try_acquire_session_turn_lease(
            "shared-session", fg, ttl_seconds=5, preempt_background_review=True
        )
        assert lease.refresh_tick() is None
        assert interrupted.is_set()

    # The row vanishes under the fork.
    foreground_db.release_session_turn_lease("shared-session", lease.holder)
    with caplog.at_level("INFO"):
        assert lease.refresh_tick() is None
        assert lease.refresh_tick() is None

    assert interrupted.is_set()
    assert run.cancel_requested.is_set()
    lines = [
        r.getMessage()
        for r in caplog.records
        if review_admission.REASON_LEASE_LOST in r.getMessage()
    ]
    assert len(lines) == 1, caplog.text
    assert lease._owner in lines[0]
    assert "shared-session" not in lines[0]
    lease.stop_refresher()
    assert lease.refresh_tick() is False

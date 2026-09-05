"""Regression suite: a stale ``gave_up`` circuit-breaker block must
auto-recover instead of silently waiting for a human forever.

Observed in production: a task that timed out ``failure_limit`` times parked in
``status='blocked'`` at ``consecutive_failures == limit``.  The
``recompute_ready`` breaker guard (``failures >= effective_limit:
continue``) then skipped it forever — a blocked task is not dispatchable,
so it can never re-fail to change the guard's inputs.  Result: ~16h of
silent waiting until a human ran ``hermes kanban unblock`` manually.

Contract pinned here:

* A stale gave_up block — breaker tripped, failure counter not climbing,
  and ``blocked`` longer than ``kanban.gave_up_stale_hours`` (default 1h)
  — is auto-unblocked by ``recompute_ready``: failures reset, the
  ``unblocked`` event carries ``auto: true``, ``stale_hours`` and
  ``by: dispatcher:auto-gave-up-recovery``.
* A *fresh* gave_up (younger than the stale delay) stays parked — the
  breaker's retry-storm protection is unchanged.
* Worker-initiated sticky blocks never age out (the ``_has_sticky_block``
  guard still wins over stale recovery).
* Per-task ``max_retries`` and the dispatcher ``failure_limit`` both
  continue to feed the guard's effective limit (#35072 semantics intact).
* ``unblock_task`` always writes an ``unblocked`` payload (ready->ready
  shape included) with ``status`` / ``resume_status`` / ``by`` — the
  audit gap that made the 16h reconstruction rely on journalctl.

End-to-end: claim -> max_runtime timeout x2 -> gave_up -> stale ->
recompute_ready auto-recovery, driven through the real
``enforce_max_runtime`` (stubbed signals) instead of hand-crafted rows.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


NOW = int(time.time())
TEST_DELAY_HOURS = 0.5


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB and a shortened
    (0.5h) stale-recovery delay; the override is cleared after each test
    so sibling fixtures that patch ``time.time`` don't leak state."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kbc.init_db()
    kb.configure_gave_up_stale_recovery(TEST_DELAY_HOURS)
    try:
        yield home
    finally:
        kb.configure_gave_up_stale_recovery(None)


def _events(conn, task_id, kind):
    out = []
    for row in conn.execute(
        "SELECT payload, created_at FROM task_events "
        "WHERE task_id = ? AND kind = ? ORDER BY id",
        (task_id, kind),
    ):
        try:
            payload = json.loads(row["payload"]) if row["payload"] else {}
        except (json.JSONDecodeError, TypeError):
            payload = {}
        out.append((payload, int(row["created_at"])))
    return out


def _latest_unblocked(conn, task_id):
    """The parsed payload dict of the most recent ``unblocked`` event.

    ``_events`` already returns payloads parsed (dict or {}), so this is a
    plain index — double-``json.loads``-ing an already-parsed dict raises
    ``TypeError`` and collapses to ``None`` (the very bug the unblock audit is
    meant to close, hiding in the test helper).
    """
    events = _events(conn, task_id, "unblocked")
    if not events or not events[-1][0]:
        return None
    return events[-1][0]


def _mark_gave_up(conn, tid, failures: int, *, extra: dict | None = None) -> None:
    """Faithfully park a task the way the real breaker does: status
    ``blocked`` + counter + a ``gave_up`` event (the row ``_record_task
    _failure`` writes when the breaker trips)."""
    payload = {
        "failures": failures,
        "effective_limit": max(1, failures),
        "limit_source": "dispatcher",
        "trigger_outcome": "timed_out",
    }
    if extra:
        payload.update(extra)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'blocked', "
            "consecutive_failures = ?, last_failure_error = 'timed out' "
            "WHERE id = ?",
            (failures, tid),
        )
        kb._append_event(conn, tid, "gave_up", payload)


def _stale_it(conn, task_id, now: int, back_seconds: int) -> None:
    """Backdate the most recent block-cause event (and any events after
    it) so that at time ``now`` the task appears to have been sitting in
    ``blocked`` for ``back_seconds``.  ``now`` must be the same timestamp
    the code under test will read (e.g. a monkeypatched ``time.time``);
    event ids keep ascending order — the log stays append-only."""
    row = conn.execute(
        "SELECT id, created_at FROM task_events "
        "WHERE task_id = ? AND kind IN ('blocked', 'gave_up') "
        "ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    assert row is not None, "expected a block-cause event to backdate"
    # Anchor the backdate to the row's own (unmodified) timestamp of the
    # given anchor call: repeated calls must not re-backdate a row that
    # this helper has already shifted — take the min of row ts and now so
    # a later _stale_it(...) never pushes the row into the future.
    anchor = min(int(row["created_at"]), now)
    target = anchor - back_seconds
    shift = anchor - target  # == back_seconds
    if shift <= 0:
        conn.execute("SELECT 1")
    else:
        conn.execute(
            "UPDATE task_events SET created_at = MAX(created_at - ?, 1) "
            "WHERE id >= ?",
            (shift, row["id"]),
        )
    conn.commit()


def _trip_breaker_via_timeouts(conn, tid, *, limit: int = 2, signal_fn) -> None:
    """Drive ``enforce_max_runtime`` to the point of a ``gave_up`` by
    backdating the active run start on each attempt — the exact path that
    RCA'd in the 16h incident (worker alive, dispatcher kills at limit).
    ``limit`` = effective failure limit (default 2)."""
    original_alive = kbd._pid_alive
    kbd._pid_alive = lambda pid: False
    try:
        for attempt in range(limit):
            task = kb.get_task(conn, tid)
            assert task.status == "ready", (
                f"attempt {attempt + 1}: expected ready, got {task.status}"
            )
            kb.claim_task(conn, tid)
            kbd._set_worker_pid(conn, tid, os.getpid())
            old_started = int(time.time()) - 120
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET started_at = ? WHERE id = ?",
                    (old_started, tid),
                )
                conn.execute(
                    "UPDATE task_runs SET started_at = ? "
                    "WHERE id = (SELECT current_run_id FROM tasks "
                    "WHERE id = ?)",
                    (old_started, tid),
                )
            assert tid in kbd.enforce_max_runtime(conn, signal_fn=signal_fn)
    finally:
        kbd._pid_alive = original_alive


# ---------------------------------------------------------------------------
# Stale gave_up auto-recovery (proposal A)
# ---------------------------------------------------------------------------


def test_stale_gave_up_auto_reclaim(kanban_home: Path) -> None:
    """The incident shape: timeout-gave_up, parked, healthy dispatcher.
    After the stale delay the same ``recompute_ready`` call that used to
    skip the task unblocks it — and the event chain says exactly who did
    it and why."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="stale gave_up", assignee="worker-a")
        _mark_gave_up(conn, tid, 2)
        # Parked for 20h; test delay is 0.5h — long stale.
        _stale_it(conn, tid, NOW + 20 * 3600, back_seconds=20 * 3600)

        assert kb.recompute_ready(conn) == 1
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.consecutive_failures == 0
        assert task.last_failure_error is None

        payload = _latest_unblocked(conn, tid)
        assert payload is not None
        assert payload["auto"] is True
        assert payload["stale_hours"] == TEST_DELAY_HOURS
        assert payload["by"] == "dispatcher:auto-gave-up-recovery"
        assert payload["status"] == "ready"
        assert payload["resume_status"] == "ready"
        # The original breaker trip is preserved behind the recovery.
        assert _events(conn, tid, "gave_up")[0][0]["failures"] == 2

        # A second tick is a no-op (no double-promote, no re-block).
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, tid).status == "ready"


def test_fresh_gave_up_stays_parked(kanban_home: Path) -> None:
    """Inside the stale delay the breaker keeps the task parked — the
    retry-storm protection the guard exists for does not regress."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="fresh gave_up", assignee="worker-a")
        _mark_gave_up(conn, tid, 2)
        _stale_it(conn, tid, NOW, back_seconds=10 * 60)  # 10 min < 0.5h

        for _ in range(3):
            assert kb.recompute_ready(conn) == 0
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.consecutive_failures == 2  # counter preserved
        assert _latest_unblocked(conn, tid) is None

        # Crossing the delay boundary recovers it (counter reset).
        _stale_it(conn, tid, NOW + 20 * 3600, back_seconds=20 * 3600)
        assert kb.recompute_ready(conn) == 1
        assert kb.get_task(conn, tid).consecutive_failures == 0


def test_sticky_block_never_ages_out(kanban_home: Path) -> None:
    """A genuine worker ``kanban_block`` (``'blocked'`` event) is sticky:
    even when it has been parked far past the stale delay, the
    ``_has_sticky_block`` guard wins and no ``unblocked`` event fires.
    This is the #28712 contract — a deliberate human handoff must not be
    auto-recovered out from under it."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="sticky wins", assignee="worker-a")
        # A genuine worker block: claim (running) then block -> writes the
        # 'blocked' event that makes it sticky.
        kb.claim_task(conn, tid)
        assert kb.block_task(
            conn, tid, reason="needs-input: confirm venue filter",
        )
        assert kb.get_task(conn, tid).status == "blocked"
        assert _events(conn, tid, "blocked"), "expected a worker 'blocked' event"
        # Age it way past the delay.
        _stale_it(conn, tid, NOW + 100 * 3600, back_seconds=100 * 3600)

        for _ in range(3):
            assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, tid).status == "blocked"
        assert _latest_unblocked(conn, tid) is None


def test_breaker_guard_honours_per_task_max_retries(kanban_home: Path) -> None:
    """#35072 semantics inside the stale-recovery guard: a recorded
    ``gave_up`` trip binds even when the live per-task ``max_retries``
    is looser than the limit that tripped the breaker — the durable
    decision wins over today's config (the numeric recomputation must
    not erase it)."""
    with kbc.connect() as conn:
        # max_retries=1, failures=1 (>= limit): waits for the stale delay.
        t1 = kb.create_task(conn, title="strict", assignee="worker-a")
        _mark_gave_up(conn, t1, 1)
        conn.execute(
            "UPDATE tasks SET max_retries = 1 WHERE id = ?", (t1,),
        )
        conn.commit()
        _stale_it(conn, t1, NOW, back_seconds=10 * 60)  # fresh
        assert kb.recompute_ready(conn) == 0
        task = kb.get_task(conn, t1)
        assert task.status == "blocked"
        assert task.consecutive_failures == 1  # preserved while parked
        # Stale now -> auto-recover with the counter reset.
        _stale_it(conn, t1, NOW + 20 * 3600, back_seconds=20 * 3600)
        assert kb.recompute_ready(conn) == 1
        assert kb.get_task(conn, t1).status == "ready"
        assert kb.get_task(conn, t1).consecutive_failures == 0
        assert _latest_unblocked(conn, t1)["auto"] is True

        # max_retries=3 but the durable gave_up tripped at effective_limit=1
        # (systemic breaker shape): the looser live limit must NOT
        # same-tick re-promote over the recorded trip — parked while
        # fresh, then the stale path recovers (counter reset).
        t2 = kb.create_task(conn, title="lenient", assignee="worker-a")
        _mark_gave_up(conn, t2, 1)
        conn.execute(
            "UPDATE tasks SET max_retries = 3 WHERE id = ?", (t2,),
        )
        conn.commit()
        _stale_it(conn, t2, NOW, back_seconds=10 * 60)  # still fresh
        assert kb.recompute_ready(conn) == 0
        task = kb.get_task(conn, t2)
        assert task.status == "blocked"
        assert task.consecutive_failures == 1
        assert _latest_unblocked(conn, t2) is None
        # Stale now -> auto-recover.
        _stale_it(conn, t2, NOW + 20 * 3600, back_seconds=20 * 3600)
        assert kb.recompute_ready(conn) == 1
        task = kb.get_task(conn, t2)
        assert task.status == "ready"
        assert task.consecutive_failures == 0
        assert _latest_unblocked(conn, t2)["auto"] is True


def test_protocol_breaker_trip_parks_despite_low_unified_counter(kanban_home: Path) -> None:
    """The protocol-violation breaker trips on its own streak budget
    (default 3) and deliberately never feeds ``consecutive_failures`` —
    so the unified counter can sit BELOW the recorded trip limit.  The
    park decision must bind to the recorded trip, not the numeric
    comparison, or a protocol gave_up is erased on the same tick."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="protocol gave_up", assignee="worker-a")
        _mark_gave_up(
            conn, tid, 1,
            extra={
                "effective_limit": 3,
                "limit_source": "protocol",
                "trigger_outcome": "crashed",
            },
        )
        _stale_it(conn, tid, NOW, back_seconds=10 * 60)  # fresh
        for _ in range(3):
            assert kb.recompute_ready(conn) == 0
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.consecutive_failures == 1
        assert _latest_unblocked(conn, tid) is None

        # Past the stale delay it cools down like any other breaker trip.
        _stale_it(conn, tid, NOW + 20 * 3600, back_seconds=20 * 3600)
        assert kb.recompute_ready(conn) == 1
        assert kb.get_task(conn, tid).status == "ready"
        assert _latest_unblocked(conn, tid)["auto"] is True


def test_legacy_no_event_numeric_fallback(kanban_home: Path) -> None:
    """Rows blocked WITHOUT any ``gave_up`` event (pre-event-log
    databases, direct DB manipulation) keep the pre-existing numeric
    semantics: below the effective limit, recompute promotes
    immediately with the counter preserved; at the limit, the task
    stays parked (no event to anchor a stale recovery to)."""
    with kbc.connect() as conn:
        t1 = kb.create_task(conn, title="legacy below limit", assignee="worker-a")
        conn.execute(
            "UPDATE tasks SET status = 'blocked', consecutive_failures = 1, "
            "max_retries = 3 WHERE id = ?",
            (t1,),
        )
        conn.commit()
        assert kb.recompute_ready(conn) == 1
        task = kb.get_task(conn, t1)
        assert task.status == "ready"
        assert task.consecutive_failures == 1  # classic path preserves

        t2 = kb.create_task(conn, title="legacy at limit", assignee="worker-a")
        conn.execute(
            "UPDATE tasks SET status = 'blocked', consecutive_failures = 3, "
            "max_retries = 3 WHERE id = ?",
            (t2,),
        )
        conn.commit()
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, t2).status == "blocked"


def test_reassign_clears_unrecovered_gave_up(kanban_home: Path) -> None:
    """Reassigning a breaker-parked task to a DIFFERENT profile is an
    explicit human recovery: the durable trip is cleared immediately
    (same unblock semantics as ``kanban_unblock``) instead of the task
    sitting out the stale cooldown after the human already took over.
    A same-profile reassign clears nothing, and a sticky worker/operator
    block is never cleared by reassignment (#28712 holds)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="reassign me", assignee="worker-a")
        _mark_gave_up(conn, tid, 2)
        _stale_it(conn, tid, NOW, back_seconds=10 * 60)  # fresh trip

        # Same-profile reassign: not a recovery — the trip stands.
        kb.assign_task(conn, tid, "worker-a")
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, tid).status == "blocked"
        assert _latest_unblocked(conn, tid) is None

        # Different profile: explicit recovery — trip cleared at once,
        # no cooldown wait, fully audited.
        kb.assign_task(conn, tid, "beta-runner")
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.assignee == "beta-runner"
        assert task.consecutive_failures == 0
        payload = _latest_unblocked(conn, tid)
        assert payload["by"] == "reassign"
        assert payload["status"] == "ready"

        # gave_up trip followed by a sticky human block: the human
        # handoff wins — reassignment must NOT clear it.
        t2 = kb.create_task(conn, title="sticky reassign", assignee="worker-a")
        _mark_gave_up(conn, t2, 2)
        kb._append_event(conn, t2, "blocked", {"reason": "needs-input: human call"})
        kb.assign_task(conn, t2, "beta-runner")
        assert kb.get_task(conn, t2).status == "blocked"
        assert _latest_unblocked(conn, t2) is None


def test_stale_review_phase_lands_in_review(kanban_home: Path) -> None:
    """A breaker trip from the review lane carrying a ``retry_status:
    review`` marker recovers to review (the dispatcher re-dispatches the
    reviewer), not ready."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="review gave_up", assignee="reviewer")
        _mark_gave_up(conn, tid, 2, extra={"retry_status": "review"})
        _stale_it(conn, tid, NOW + 20 * 3600, back_seconds=20 * 3600)
        assert kb.recompute_ready(conn) == 1
        assert kb.get_task(conn, tid).status == "review"
        assert (
            _latest_unblocked(conn, tid) or {}
        )["resume_status"] == "review"


# ---------------------------------------------------------------------------
# Audit completeness (proposal B)
# ---------------------------------------------------------------------------


def test_unblock_always_writes_audit_payload(kanban_home: Path) -> None:
    """The ready->ready unblock — the shape that used to write a NULL
    payload — now always records status / resume_status / by."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="unblock audit", assignee="worker-a")
        _mark_gave_up(conn, tid, 2)

        assert kb.unblock_task(conn, tid) is True
        payload = _latest_unblocked(conn, tid)
        assert payload == {
            "status": "ready",
            "resume_status": "ready",
            "by": "unknown",
        }
        assert kb.get_task(conn, tid).consecutive_failures == 0

        # Blocked + re-tripped, unblock with an explicit actor.
        _mark_gave_up(conn, tid, 2)
        assert kb.unblock_task(conn, tid, actor="cli:operator") is True
        payload = _latest_unblocked(conn, tid)
        assert payload["by"] == "cli:operator"
        assert payload["status"] == "ready"

        # A review-resume unblock also carries the full payload.
        _mark_gave_up(conn, tid, 2, extra={"retry_status": "review"})
        assert kb.unblock_task(conn, tid, actor="dashboard") is True
        payload = _latest_unblocked(conn, tid)
        assert payload["status"] == "review"
        assert payload["resume_status"] == "review"
        assert payload["by"] == "dashboard"


# ---------------------------------------------------------------------------
# End-to-end: timeout -> gave_up -> stale -> auto-recover (acceptance)
# ---------------------------------------------------------------------------


def test_e2e_timeout_gave_up_auto_reclaims(kanban_home: Path, monkeypatch) -> None:
    """Acceptance from the card: a *must-timeout* task (max_runtime
    backdated beyond its limit on every attempt) reaches gave_up by the
    real dispatcher timeout path, then — with no human in the loop —
    the same ``recompute_ready`` the dispatcher runs every tick unblocks
    it after the stale delay, with a fully auditable event chain.

    The 16h incident shape: two consecutive timeouts, then silence.
    """
    killed = []

    def _signal_fn(pid, sig):
        killed.append((pid, sig))

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="always times out",
            assignee="worker-a", max_runtime_seconds=60,
        )
        _trip_breaker_via_timeouts(conn, tid, limit=2, signal_fn=_signal_fn)

        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.consecutive_failures == 2
        assert any(e.kind == "gave_up" for e in kb.list_events(conn, tid))
        assert killed, "two attempts, two kills expected"

        # Dispatcher keeps ticking — before the elapse the task stays parked.
        for _ in range(3):
            assert kb.recompute_ready(conn) == 0
            assert kb.get_task(conn, tid).status == "blocked"

        # 20h later (test delay = 0.5h): the next tick recovers it, and
        # the next attempt can start from a fresh retry budget.
        _stale_it(conn, tid, NOW + 20 * 3600, back_seconds=20 * 3600)
        assert kb.recompute_ready(conn) == 1
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.consecutive_failures == 0

        events = [e.kind for e in kb.list_events(conn, tid)]
        assert "timed_out" in events
        assert "gave_up" in events
        assert "unblocked" in events
        payload = _latest_unblocked(conn, tid)
        assert payload["auto"] is True
        assert payload["by"] == "dispatcher:auto-gave-up-recovery"
        assert payload["stale_hours"] == TEST_DELAY_HOURS


def test_stale_recovery_with_config_file(kanban_home: Path, tmp_path) -> None:
    """The config key ``kanban.gave_up_stale_hours`` is read and honoured
    for the delay (0.25h in the file, not the 1h default nor any
    override)."""
    home = tmp_path / ".hermes"
    (home / "config.yaml").write_text("kanban:\n  gave_up_stale_hours: 0.25\n")
    kb.configure_gave_up_stale_recovery(None)  # fall through to the file
    assert kb.configured_gave_up_stale_hours() == 0.25
    try:
        with kbc.connect() as conn:
            tid = kb.create_task(conn, title="cfg", assignee="worker-a")
            _mark_gave_up(conn, tid, 2)
            # 30 min in blocked: stale at 0.25h (15 min), fresh at 1h.
            _stale_it(conn, tid, NOW, back_seconds=30 * 60)
            assert kb.recompute_ready(conn) == 1
            assert kb.get_task(conn, tid).status == "ready"
            assert _latest_unblocked(conn, tid)["auto"] is True
    finally:
        kb.configure_gave_up_stale_recovery(None)
# ---------------------------------------------------------------------------
# Config robustness + observer contract (review follow-ups)
# ---------------------------------------------------------------------------


def test_non_finite_config_value_rejected(kanban_home: Path) -> None:
    """A YAML ``.inf`` (or nan) for ``kanban.gave_up_stale_hours`` must be
    treated as invalid (fall back to the default), never reach
    ``int(stale_hours * 3600)`` — an OverflowError inside
    ``recompute_ready``'s write txn would crash every dispatch tick."""
    import math

    kb.configure_gave_up_stale_recovery(float("inf"))
    try:
        # Non-finite override is rejected -> default delay applies.
        assert kb._gave_up_stale_hours() == pytest.approx(
            kb.DEFAULT_GAVE_UP_STALE_HOURS
        )
        with kbc.connect() as conn:
            tid = kb.create_task(conn, title="inf", assignee="worker-a")
            _mark_gave_up(conn, tid, 2)
            _stale_it(conn, tid, NOW + 20 * 3600, back_seconds=20 * 3600)
            # Must not raise, and recovers on the default delay.
            assert kb.recompute_ready(conn) == 1
            assert kb.get_task(conn, tid).status == "ready"
    finally:
        kb.configure_gave_up_stale_recovery(None)
    assert math.isfinite(kb.DEFAULT_GAVE_UP_STALE_HOURS)


def test_negative_config_disables_auto_recovery(kanban_home: Path) -> None:
    """``gave_up_stale_hours: -1`` is the explicit opt-out: gave_up blocks
    park until a human unblocks them (the pre-fix behaviour)."""
    kb.configure_gave_up_stale_recovery(-1.0)
    try:
        assert kb._gave_up_stale_hours() is None
        with kbc.connect() as conn:
            tid = kb.create_task(conn, title="disabled", assignee="worker-a")
            _mark_gave_up(conn, tid, 2)
            _stale_it(conn, tid, NOW + 100 * 3600, back_seconds=100 * 3600)
            for _ in range(3):
                assert kb.recompute_ready(conn) == 0
            assert kb.get_task(conn, tid).status == "blocked"
            assert _latest_unblocked(conn, tid) is None
    finally:
        kb.configure_gave_up_stale_recovery(None)


def test_future_event_timestamp_clamps_to_now(kanban_home: Path) -> None:
    """A gave_up event stamped in the future (clock skew, restored backup,
    hand-edited DB) clamps to now — the task parks one full delay and then
    recovers, instead of parking forever with no diagnostic."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="future ts", assignee="worker-a")
        _mark_gave_up(conn, tid, 2)
        # Stamp the gave_up 24h into the future (direct write — the
        # backdating helper deliberately never pushes rows forward).
        conn.execute(
            "UPDATE task_events SET created_at = ? "
            "WHERE task_id = ? AND kind = 'gave_up'",
            (NOW + 24 * 3600, tid),
        )
        conn.commit()
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, tid).status == "blocked"
        # One full delay later it recovers normally.
        _stale_it(conn, tid, NOW + 40 * 3600, back_seconds=40 * 3600)
        assert kb.recompute_ready(conn) == 1
        assert kb.get_task(conn, tid).status == "ready"


def test_reassign_trip_clear_reports_status_field(kanban_home: Path, monkeypatch) -> None:
    """The reassign trip-clearing path flips ``status`` as a side effect —
    the observer notification must carry ``status`` alongside
    ``assignee`` so incremental consumers don't miss the transition."""
    calls = []
    monkeypatch.setattr(
        kb, "notify_task_updated",
        lambda conn, task_id, fields: calls.append((task_id, fields)),
    )
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="notify fields", assignee="worker-a")
        _mark_gave_up(conn, tid, 2)

        kb.assign_task(conn, tid, "worker-a")  # same profile: no clear
        assert calls[-1] == (tid, ("assignee",))

        kb.assign_task(conn, tid, "beta-runner")  # clears the trip
        assert calls[-1] == (tid, ("assignee", "status"))
        assert kb.get_task(conn, tid).status == "ready"

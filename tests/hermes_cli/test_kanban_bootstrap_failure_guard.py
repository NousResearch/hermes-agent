"""Bootstrap-failure guard: a worker that dies at CLI init on an
unresolvable ``--skills``/profile name must not burn ``consecutive_failures``
toward the ``gave_up`` circuit breaker.

Regression coverage for #kanban-bootstrap-failure-gaveup (2026-08-19,
t_eee837c7, diagnosed against t_ea832a93/t_83812b77/t_ea89769e/t_966f138b/
t_f7f32991): the dispatcher force-injects a skill (e.g. ``sdlc-review``) into
every review-lane worker's ``--skills`` list. When that skill isn't linked on
the target profile, the worker process dies during CLI init with
``Error: Unknown skill(s): ...`` (or ``Profile '...' does not exist``) in
well under a minute, before it can call ANY kanban lifecycle tool
(``kanban_block`` / ``kanban_complete``). Before this fix,
``detect_crashed_workers`` classified that exit as an ordinary
``clean_exit``/``nonzero_exit`` protocol violation and counted it toward
``consecutive_failures`` — but retrying is pointless: the same unresolvable
name crashes identically every time, so the counter marches straight to
``gave_up`` on a card whose actual task work may already be complete.
Measured case: 19 crashes across 5 review-lane cards in one 24h window, all
sharing this exact log signature.

Fix: ``_worker_log_bootstrap_failure`` tails the worker's log for the
deterministic signature BEFORE the ordinary clean_exit/nonzero_exit
classification runs. When found, ``detect_crashed_workers`` requeues the
task to its source phase (like a rate-limit release) WITHOUT incrementing
``consecutive_failures``, records the outcome as ``bootstrap_failed`` (not
``crashed``), and emits a one-time event + card comment via
``_emit_bootstrap_failure_alert`` so a human can fix the underlying
profile/skill config — unlike a quota wall, this never clears on its own.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Existing crash-detection tests (and this one) pre-date the reap grace
    # window; pin to 0 for immediate-reclaim semantics — see the identical
    # fixture note in test_kanban_core_functionality.py.
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _plant_worker_log(task_id: str, text: str, board=None) -> Path:
    log_dir = kb.worker_logs_dir(board=board)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task_id}.log"
    log_path.write_text(text)
    return log_path


def test_unknown_skill_bootstrap_failure_not_counted_as_failure(
    kanban_home, monkeypatch,
):
    """The deterministic 'Unknown skill(s): ...' signature must release the
    task WITHOUT touching consecutive_failures, and classify as
    bootstrap_failed rather than an ordinary crash."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="review this", assignee="worker")
        kb.claim_task(conn, tid)
        kb._set_worker_pid(conn, tid, 55501)

        _plant_worker_log(
            tid,
            "some startup banner\n"
            "Error: Unknown skill(s): sdlc-review\n"
            "more trailing noise\n",
        )

        monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
        monkeypatch.setattr(
            kb, "_classify_worker_exit", lambda pid: ("clean_exit", 0),
        )

        before = kb.get_task(conn, tid)
        assert before.consecutive_failures == 0

        crashed = kb.detect_crashed_workers(conn)

        # Not reported via the ordinary crashed list...
        assert tid not in crashed
        # ...but IS surfaced via the bootstrap-failed side channel.
        bootstrap_failed = getattr(
            kb.detect_crashed_workers, "_last_bootstrap_failed", [],
        )
        assert tid in bootstrap_failed

        after = kb.get_task(conn, tid)
        assert after.consecutive_failures == 0, (
            "a deterministic bootstrap failure must NOT increment "
            "consecutive_failures — retrying can't fix an unresolvable name"
        )
        assert after.status in ("ready", "todo")

        runs = kb.list_runs(conn, tid)
        assert runs[-1].outcome == "bootstrap_failed"

        events = kb.list_events(conn, tid)
        alert_events = [e for e in events if e.kind == "bootstrap_failure_alerted"]
        assert len(alert_events) == 1
        assert alert_events[0].payload["assignee"] == "worker"

        comments = kb.list_comments(conn, tid)
        assert len(comments) == 1
        assert "Unknown skill" in comments[0].body or "unresolvable" in comments[0].body


def test_unknown_profile_bootstrap_failure_also_detected(kanban_home, monkeypatch):
    """The second deterministic signature — 'Profile '...' does not exist'
    — must be recognized too (both come from the same CLI-init failure
    class, just different missing-name flavors)."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="review this too", assignee="worker")
        kb.claim_task(conn, tid)
        kb._set_worker_pid(conn, tid, 55502)

        _plant_worker_log(
            tid, "Profile 'retired-role' does not exist\n",
        )

        monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
        monkeypatch.setattr(
            kb, "_classify_worker_exit", lambda pid: ("nonzero_exit", 1),
        )

        kb.detect_crashed_workers(conn)
        bootstrap_failed = getattr(
            kb.detect_crashed_workers, "_last_bootstrap_failed", [],
        )
        assert tid in bootstrap_failed
        assert kb.get_task(conn, tid).consecutive_failures == 0


def test_bootstrap_failure_alert_does_not_repeat_across_ticks(
    kanban_home, monkeypatch,
):
    """Re-crashing on the identical bootstrap failure (e.g. the operator
    hasn't fixed the config yet and the card gets re-dispatched) must only
    alert once per (task, assignee) — not spam a comment every reclaim."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="loops on bad skill", assignee="worker")

        for pid in (66601, 66602):
            kb.claim_task(conn, tid)
            kb._set_worker_pid(conn, tid, pid)
            _plant_worker_log(
                tid, "Error: Unknown skill(s): sdlc-review\n",
            )
            monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
            monkeypatch.setattr(
                kb, "_classify_worker_exit", lambda pid: ("clean_exit", 0),
            )
            kb.detect_crashed_workers(conn)

        events = kb.list_events(conn, tid)
        alert_events = [e for e in events if e.kind == "bootstrap_failure_alerted"]
        assert len(alert_events) == 1


def test_ordinary_crash_without_bootstrap_signature_still_counts_failure(
    kanban_home, monkeypatch,
):
    """Sanity check: a real crash with NO deterministic signature in the
    log is unaffected by this guard and still increments
    consecutive_failures via the pre-existing protocol-violation path."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="genuinely crashes", assignee="worker")
        kb.claim_task(conn, tid)
        kb._set_worker_pid(conn, tid, 77701)

        _plant_worker_log(tid, "some ordinary traceback, nothing deterministic\n")

        monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
        monkeypatch.setattr(
            kb, "_classify_worker_exit", lambda pid: ("clean_exit", 0),
        )

        crashed = kb.detect_crashed_workers(conn)
        assert tid in crashed
        bootstrap_failed = getattr(
            kb.detect_crashed_workers, "_last_bootstrap_failed", [],
        )
        assert tid not in bootstrap_failed

"""Tests for the kanban respawn guard (``hermes_cli.kanban_db_respawn_guard``,
reached as ``kanban_db_dispatch.check_respawn_guard``): the per-card decision
whether a ``ready``/``review`` task may re-spawn on this tick.

Covers the cooldown order (infrastructure spawn refusal, rate-limit, then the
quota/auth blocker), the ``crashed`` exclusion, and the
stale-``last_failure_error`` wedge the guard caused when it judged the
task-level stamp instead of the latest run's own ``error`` (#119070; the live
incident write-up is shimpeiws/hermes-setup#545). ``recent_success`` /
``active_pr`` interplay through real ``dispatch_once`` passes lives in
``test_kanban_review_lifecycle.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_respawn_guard_defers_rate_limited_within_cooldown(
    kanban_home, monkeypatch,
):
    """Within the cooldown after a rate-limit requeue, the guard defers the
    respawn; after the cooldown it allows a probe — and crucially does NOT
    fall into ``blocker_auth`` (which would defer forever)."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    now = 5_000_000

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="rl-guard", assignee="a")
        # Seed a rate_limited run that just ended + the stamped error.
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id
        conn.execute(
            "UPDATE task_runs SET outcome='rate_limited', status='rate_limited', "
            "ended_at=?, error=? WHERE id=?",
            (now, "pid 1 exited rate-limited (quota wall) — requeued", run_id),
        )
        conn.execute(
            "UPDATE tasks SET status='ready', current_run_id=NULL, "
            "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL, "
            "last_failure_error=? WHERE id=?",
            ("pid 1 exited rate-limited (quota wall) — requeued", tid),
        )
        conn.commit()

        # Inside cooldown → defer with the rate-limit-specific reason.
        monkeypatch.setattr(_kb.time, "time", lambda: now + 100)
        assert kbd.check_respawn_guard(conn, tid) == "rate_limit_cooldown"

        # Past cooldown → allowed (None), NOT trapped by blocker_auth even
        # though last_failure_error contains "rate-limited".
        monkeypatch.setattr(_kb.time, "time", lambda: now + 400)
        assert kbd.check_respawn_guard(conn, tid) is None


@pytest.mark.parametrize(
    "error_text, expected",
    [
        # Worker progress prose talking about *writing*, not an auth failure
        # (#117009): must NOT trip the guard.
        ("Workstream C items C-3 and C-4: author t  (90.59s)", None),
        ("docs authored by the previous cycle", None),
        ("relying on an authoritative source", None),
        # Genuine auth failures must still trip the guard, one row per
        # curated stem family (bare, -ate, -ize, -ise).
        ("401 auth failed", "blocker_auth"),
        ("authentication error from provider", "blocker_auth"),
        ("still authorizing the request", "blocker_auth"),
        ("still authorising the request", "blocker_auth"),
    ],
)
def test_respawn_guard_blocker_auth_curated_not_open_stem(
    kanban_home, monkeypatch, error_text, expected,
):
    """``_RESPAWN_BLOCKER_RE`` used to use an open ``auth\\w*`` stem that matched
    ordinary English words like "author"/"authored"/"authoring"/"authoritative"
    in worker progress prose, parking a healthy ``ready`` card forever (#117009).
    The auth family must be a curated set of real auth-failure tokens. The text
    judged is the LATEST run's own ``error``; ``last_failure_error`` only acts
    as the not-cleared latch, so a non-NULL placeholder stands in for it."""
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "0")

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="prose", assignee="a")
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id
        conn.execute(
            "UPDATE task_runs SET outcome='gave_up', status='failed', "
            "ended_at=?, error=? WHERE id=?",
            (5_000_000, error_text, run_id),
        )
        conn.execute(
            "UPDATE tasks SET status='ready', current_run_id=NULL, "
            "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL, "
            "last_failure_error='failure recorded' WHERE id=?",
            (tid,),
        )
        conn.commit()
        assert kbd.check_respawn_guard(conn, tid) == expected


def test_respawn_guard_ignores_auth_words_in_crashed_worker_output(kanban_home):
    """A plain crash's captured stdout is context, not a diagnosis.

    ``_classify_dead_worker`` appends the worker's last output to the persisted
    failure text.  A benign command such as ``claude auth status`` must not turn
    an unrelated crash into a permanent auth guard on the next dispatch.
    """
    with kbc.connect() as conn:
        crashed_id = kb.create_task(conn, title="crashed", assignee="a")
        kb.claim_task(conn, crashed_id)
        crashed_run_id = kb.get_task(conn, crashed_id).current_run_id
        conn.execute(
            "UPDATE task_runs SET outcome='crashed', status='failed', ended_at=?, "
            "error=? WHERE id=?",
            (
                5_000_000,
                "pid 1 killed by signal 9. Worker's last output: "
                "'env -u ANTHROPIC_API_KEY claude auth status --text'",
                crashed_run_id,
            ),
        )
        conn.execute(
            "UPDATE tasks SET status='ready', current_run_id=NULL, "
            "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL, "
            "last_failure_error=? WHERE id=?",
            (
                "pid 1 killed by signal 9. Worker's last output: "
                "'env -u ANTHROPIC_API_KEY claude auth status --text'",
                crashed_id,
            ),
        )

        spawn_failed_id = kb.create_task(conn, title="spawn failed", assignee="a")
        kb.claim_task(conn, spawn_failed_id)
        spawn_run_id = kb.get_task(conn, spawn_failed_id).current_run_id
        conn.execute(
            "UPDATE task_runs SET outcome='spawn_failed', status='failed', ended_at=?, "
            "error=? WHERE id=?",
            (5_000_000, "provider authentication failed", spawn_run_id),
        )
        conn.execute(
            "UPDATE tasks SET status='ready', current_run_id=NULL, "
            "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL, "
            "last_failure_error=? WHERE id=?",
            ("provider authentication failed", spawn_failed_id),
        )
        conn.commit()

        assert kbd.check_respawn_guard(conn, crashed_id) is None
        assert kbd.check_respawn_guard(conn, spawn_failed_id) == "blocker_auth"


def _seed_ended_run(conn, tid, *, outcome, error=None, ended_at=5_000_000):
    """Claim ``tid`` then close its run with ``outcome``/``error``; the task
    lands back on ``ready`` with a caller-supplied ``last_failure_error``."""
    kb.claim_task(conn, tid)
    run_id = kb.get_task(conn, tid).current_run_id
    conn.execute(
        "UPDATE task_runs SET outcome=?, status='failed', ended_at=?, error=? WHERE id=?",
        (outcome, ended_at, error, run_id),
    )


def test_respawn_guard_ignores_stale_stamp_after_later_run(kanban_home):
    """A quota/auth ``last_failure_error`` left by an OLDER failure must not
    park the card once a later run ended without that diagnosis.

    The stamp is written only on failures and cleared only on success or a
    deliberate unblock, so a run that ends ``blocked`` (observation timeout),
    ``review_requested`` or ``reclaimed`` leaves the stale quota string behind.
    Judging the stamp instead of the latest run's own ``error`` parked ready
    cards forever — observed live in shimpeiws/hermes-setup#545 and the same
    wedge as upstream #119070 in the review lane.
    """
    with kbc.connect() as conn:
        stale_stamp = "pid 1 exited rate-limited (quota wall) — requeued"

        # Latest run ended ``blocked`` with no error text: the quota stamp is
        # stale and the card must be allowed to respawn.
        timed_out = kb.create_task(conn, title="timed out", assignee="a")
        _seed_ended_run(conn, timed_out, outcome="blocked")
        conn.execute(
            "UPDATE tasks SET status='ready', current_run_id=NULL, "
            "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL, "
            "last_failure_error=? WHERE id=?",
            (stale_stamp, timed_out),
        )

        # Same stale stamp on the review lane after a ``review_requested``
        # run — the #119070 wedge.
        handed_off = kb.create_task(conn, title="handed off", assignee="a")
        _seed_ended_run(conn, handed_off, outcome="review_requested")
        conn.execute(
            "UPDATE tasks SET status='review', current_run_id=NULL, "
            "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL, "
            "last_failure_error=? WHERE id=?",
            (stale_stamp, handed_off),
        )

        # No runs at all: a stamped card that never executed must not park
        # either (there is no evidence to judge, so fail open).
        never_ran = kb.create_task(conn, title="never ran", assignee="a")
        conn.execute(
            "UPDATE tasks SET last_failure_error=? WHERE id=?",
            (stale_stamp, never_ran),
        )
        conn.commit()

        assert kbd.check_respawn_guard(conn, timed_out) is None
        assert kbd.check_respawn_guard(conn, handed_off, lane="review") is None
        assert kbd.check_respawn_guard(conn, never_ran) is None


def test_respawn_guard_cleared_stamp_lets_operator_retry(kanban_home):
    """``unblock_task`` NULLs ``last_failure_error`` — the operator's "retry".

    A card blocked by a quota/auth ``gave_up`` run must respawn after a real
    unblock even though the run's recorded ``error`` still matches the blocker
    pattern: otherwise the cleared stamp buys nothing and the card re-parks on
    the next tick with no CLI escape left.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="auth gave up", assignee="a")
        _seed_ended_run(
            conn, tid, outcome="gave_up", error="provider authentication failed",
        )
        conn.execute(
            "UPDATE tasks SET status='blocked', current_run_id=NULL, "
            "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL, "
            "consecutive_failures=2, last_failure_error=? WHERE id=?",
            ("provider authentication failed", tid),
        )
        conn.commit()

        # Guarded while still blocked-fresh: fresh auth evidence, stamp set.
        assert kbd.check_respawn_guard(conn, tid) == "blocker_auth"

        assert kb.unblock_task(conn, tid)
        assert kb.get_task(conn, tid).status == "ready"
        assert kbd.check_respawn_guard(conn, tid) is None


def test_infrastructure_spawn_refusal_never_charges_the_card(
    kanban_home, monkeypatch, all_assignees_spawnable,
):
    """The host refusing to place a worker (managed gateway, user bus gone —
    #114720) is not a card failure: through the REAL spawn boundary and the
    real dispatcher accounting, ``consecutive_failures`` stays put, the breaker
    never parks the card as a bare ``blocked``, the run is tagged
    ``infrastructure`` and the guard spaces the retries. A control spawn
    failure on the same card still counts."""
    from tools import process_registry

    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: True)
    monkeypatch.setenv("INVOCATION_ID", "managed-gateway")
    monkeypatch.setattr(process_registry, "_systemd_run_user_scope_available", lambda: False)
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "0")

    def spawn_via_real_boundary(task, workspace, board=None):
        kbd._restart_safe_worker_argv(task, ["hermes", "chat"])  # raises: real probe verdict, real _degrade()
        raise AssertionError("unreachable")

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="bus is down", assignee="a")
        for _ in range(3):
            res = kbd.dispatch_once(conn, spawn_fn=spawn_via_real_boundary, failure_limit=2)
            assert res.auto_blocked == []
        row = conn.execute(
            "SELECT status, block_kind, consecutive_failures, last_failure_error FROM tasks WHERE id = ?", (tid,),
        ).fetchone()
        assert (row["status"], row["block_kind"], row["consecutive_failures"]) == ("ready", None, 0)
        assert "enable-linger" in row["last_failure_error"]
        runs = conn.execute(
            "SELECT outcome, metadata FROM task_runs WHERE task_id = ? ORDER BY id", (tid,),
        ).fetchall()
        assert [r["outcome"] for r in runs] == ["spawn_failed"] * 3
        assert all(json.loads(r["metadata"])["infrastructure"] is True for r in runs)

        monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
        assert kbd.check_respawn_guard(conn, tid) == "infrastructure_cooldown"

        # Control: an ordinary spawn failure on the same card still spends budget.
        monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "0")

        def spawn_broken(task, workspace, board=None):
            raise RuntimeError("profile launcher exploded")

        kbd.dispatch_once(conn, spawn_fn=spawn_broken, failure_limit=2)
        assert conn.execute(
            "SELECT consecutive_failures FROM tasks WHERE id = ?", (tid,),
        ).fetchone()[0] == 1

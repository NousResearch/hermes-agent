"""Regression tests for #111910: active_pr must not block authorized Closer
recovery, and suppression reasons must be visible to operators.

``check_respawn_guard`` rule 4 (``active_pr``) exists to stop a *duplicate
implementation* worker from opening a second PR when one is already open.
That rationale does not apply to an authorized recovery role (e.g. Closer)
whose whole job is to act on the existing PR — but the guard fired for every
assignee/lane alike, so a Closer task assigned in the ready lane could never
be dispatched while its PR stayed open (#111910 evidence item 3).

These tests pin:
1. active PR + Dev (or any non-recovery assignee) in the ready lane => still
   guarded (duplicate-work protection is NOT weakened).
2. active PR + an authorized recovery assignee (default: "closer") in the
   ready lane => spawnable.
3. The recovery-assignee set is configurable via
   ``kanban.active_pr_recovery_assignees`` (not a hardcoded name-only carve-out).
4. ``hermes kanban dispatch --json`` exposes ``respawn_guarded`` (including
   ``active_pr``), ``rate_limited``, ``skipped_locked``, ``memory_pressure``.
5. The dispatcher "stuck" health warning reports the respawn-guard reason(s)
   that suppressed the ready queue, not just a bare "0 spawned".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_ops


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _seed_active_pr_task(conn, *, assignee: str) -> str:
    tid = kb.create_task(conn, title="handle the PR", assignee=assignee)
    kb.add_comment(
        conn, tid, author="worker",
        body="Opened https://github.com/example/repo/pull/456 for review.",
    )
    return tid


# ---------------------------------------------------------------------------
# 1 & 2: Dev stays guarded, authorized Closer recovery is spawnable
# ---------------------------------------------------------------------------


def test_active_pr_guard_still_blocks_dev_in_ready_lane(kanban_home, monkeypatch):
    """Duplicate-work protection is preserved: a non-recovery assignee (Dev)
    with an active PR comment stays guarded in the ready lane."""
    with kbc.connect() as conn:
        tid = _seed_active_pr_task(conn, assignee="dev")
        assert kbd.check_respawn_guard(conn, tid, lane="ready") == "active_pr"


def test_active_pr_guard_exempts_authorized_closer_recovery_in_ready_lane(
    kanban_home, monkeypatch,
):
    """The default recovery assignee (Closer) is not blocked by an active PR
    in the ready lane — its job IS to act on that PR, not duplicate it."""
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)
    monkeypatch.setattr(cfgmod, "load_config", lambda *a, **k: {})

    with kbc.connect() as conn:
        tid = _seed_active_pr_task(conn, assignee="closer")
        assert kbd.check_respawn_guard(conn, tid, lane="ready") is None

        res = kbd.dispatch_once(conn, dry_run=True)
        spawned_ids = [s[0] for s in res.spawned]
        assert tid in spawned_ids
        assert tid not in dict(res.respawn_guarded)


def test_active_pr_recovery_assignees_configurable(kanban_home, monkeypatch):
    """Operators can name additional (or different) recovery assignees via
    config — this is not a hardcoded 'closer' string special-case."""
    import hermes_cli.config as cfgmod

    with kbc.connect() as conn:
        tid = _seed_active_pr_task(conn, assignee="hotfixer")

        # Not in the default recovery set -> still guarded.
        monkeypatch.setattr(cfgmod, "load_config", lambda *a, **k: {})
        assert kbd.check_respawn_guard(conn, tid, lane="ready") == "active_pr"

        # Configured in -> exempt.
        monkeypatch.setattr(
            cfgmod, "load_config",
            lambda *a, **k: {"kanban": {"active_pr_recovery_assignees": ["hotfixer"]}},
        )
        assert kbd.check_respawn_guard(conn, tid, lane="ready") is None

        # The default "closer" name is also still exempt when the operator
        # narrows the config list, unless they explicitly remove it too — the
        # override REPLACES the default set (explicit config wins).
        closer_tid = _seed_active_pr_task(conn, assignee="closer")
        assert kbd.check_respawn_guard(conn, closer_tid, lane="ready") == "active_pr"


# ---------------------------------------------------------------------------
# 3: recent_success still applies to recovery assignees (only rule 4 exempt)
# ---------------------------------------------------------------------------


def test_recovery_assignee_still_guarded_by_recent_success(kanban_home, monkeypatch):
    """The active_pr carve-out is narrow: a Closer that just completed a run
    is still deferred by recent_success — recovery status doesn't blanket-
    exempt every duplicate-work guard, only the active-PR one."""
    import hermes_cli.config as cfgmod

    monkeypatch.setattr(cfgmod, "load_config", lambda *a, **k: {})
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="closer redo", assignee="closer")
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id
        import time as _time
        now = int(_time.time())
        conn.execute(
            "UPDATE task_runs SET outcome='completed', status='completed', "
            "ended_at=? WHERE id=?", (now, run_id),
        )
        conn.execute(
            "UPDATE tasks SET status='ready', current_run_id=NULL, "
            "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL WHERE id=?",
            (tid,),
        )
        conn.commit()
        assert kbd.check_respawn_guard(conn, tid, lane="ready") == "recent_success"


# ---------------------------------------------------------------------------
# 4: CLI JSON exposes active_pr / suppression fields
# ---------------------------------------------------------------------------


def test_dispatch_json_exposes_active_pr_and_suppression_fields(
    kanban_home, monkeypatch, capsys,
):
    monkeypatch.setattr(
        kanban_ops.kbd, "dispatch_once",
        lambda *a, **k: kb.DispatchResult(
            respawn_guarded=[("t_stuck", "active_pr")],
            rate_limited=["t_quota"],
            skipped_locked=True,
            memory_pressure="critical",
        ),
    )
    args = argparse.Namespace(dry_run=True, max=None, failure_limit=2, json=True)
    assert kanban_ops._cmd_dispatch(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["respawn_guarded"] == [{"task_id": "t_stuck", "reason": "active_pr"}]
    assert payload["rate_limited"] == ["t_quota"]
    assert payload["skipped_locked"] is True
    assert payload["memory_pressure"] == "critical"


# ---------------------------------------------------------------------------
# 5: stuck-health warning reports the guard reason
# ---------------------------------------------------------------------------


def test_summarize_respawn_guard_reasons_counts_by_reason():
    guarded = [("t_a", "active_pr"), ("t_b", "active_pr"), ("t_c", "recent_success")]
    assert kbd.summarize_respawn_guard_reasons(guarded) == {
        "active_pr": 2, "recent_success": 1,
    }
    assert kbd.summarize_respawn_guard_reasons([]) == {}


def test_daemon_stuck_warning_includes_guard_reason(kanban_home, monkeypatch, capsys):
    """The real ``_cmd_daemon`` 'dispatcher stuck' warning names the
    respawn-guard reason(s) instead of only reporting a bare zero-spawn count
    (#111910 evidence item 4). Drives the actual handler end-to-end through a
    stubbed ``run_daemon`` that calls ``on_tick`` synchronously HEALTH_WINDOW
    times, exactly as the real loop would across consecutive bad ticks.
    """
    monkeypatch.setattr(kanban_ops.kbd, "has_spawnable_ready", lambda conn: True)

    def _fake_run_daemon(*, interval, max_spawn, failure_limit, on_tick):
        for _ in range(6):  # matches _cmd_daemon's HEALTH_WINDOW
            on_tick(kb.DispatchResult(respawn_guarded=[("t_signal", "active_pr")]))

    monkeypatch.setattr(kanban_ops.kbd, "run_daemon", _fake_run_daemon)

    args = argparse.Namespace(
        interval=0.0, max=None, failure_limit=2, force=True,
        pidfile=None, verbose=False,
    )
    assert kanban_ops._cmd_daemon(args) == 0

    stderr = capsys.readouterr().err
    assert "dispatcher stuck" in stderr
    assert "active_pr" in stderr

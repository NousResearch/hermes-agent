"""Workspace-resolve spawn failures must not be trapped forever as blocker_auth.

The dispatch path stamps ``workspace: [Errno 13] Permission denied: ...`` with
outcome ``spawn_failed`` when workspace resolution fails. That text is in the
auth family, so the respawn guard's ``blocker_auth`` regex parked the card
forever even after the operator fixed the path (moved the workspace dir,
permissions, ownership) — nothing in the persisted state can show the fix.
The guard now re-probes resolution for the ``workspace:`` prefix only;
provider-auth spawn failures (account state) stay permanent blockers.
"""

from __future__ import annotations

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb.init_db()
    return home


_WS_ERR = "workspace: [Errno 13] Permission denied: '/workspace'"


def _stamp(conn, tid, outcome="spawn_failed", error=_WS_ERR):
    kb.claim_task(conn, tid)
    run_id = kb.get_task(conn, tid).current_run_id
    conn.execute(
        "UPDATE task_runs SET outcome=?, status='failed', ended_at=5000000 WHERE id=?",
        (outcome, run_id),
    )
    conn.execute(
        "UPDATE tasks SET status='ready', current_run_id=NULL, claim_lock=NULL, "
        "claim_expires=NULL, worker_pid=NULL, last_failure_error=? WHERE id=?",
        (error, tid),
    )
    conn.commit()


def test_workspace_failure_with_fixable_path_retries(kanban_home, tmp_path):
    good = tmp_path / "lane"
    good.mkdir()
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn,
            title="lane",
            assignee="a",
            workspace_kind="dir",
            workspace_path=str(good),
        )
        _stamp(conn, tid)
        assert kbd.check_respawn_guard(conn, tid) is None


def test_workspace_failure_still_unresolvable_stays_guarded(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn,
            title="lane",
            assignee="a",
            workspace_kind="dir",
            workspace_path="/proc/kanban-probe",
        )
        _stamp(conn, tid)
        assert kbd.check_respawn_guard(conn, tid) == "workspace_unresolvable"


def test_provider_auth_spawn_failure_still_blocks(kanban_home):
    # Contract of test_kanban_db.py:551 — account-state auth failures stay
    # permanent blockers; the probe is scoped to the `workspace:` prefix.
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn,
            title="lane",
            assignee="a",
            workspace_kind="dir",
            workspace_path=str(kanban_home),
        )
        _stamp(conn, tid, error="provider authentication failed")
        assert kbd.check_respawn_guard(conn, tid) == "blocker_auth"

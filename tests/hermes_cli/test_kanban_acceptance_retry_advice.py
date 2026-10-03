"""Acceptance retry advice must never become a worker auth diagnosis (#132473)."""
import json
import subprocess

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch
from hermes_cli import kanban_pr_acceptance as acceptance
from hermes_cli import kanban_pr_acceptance_store as store


@pytest.mark.parametrize("lane", ["ready", "review"])
@pytest.mark.parametrize("previous", [None, "401 auth failed"])
@pytest.mark.parametrize("failure", ["timeout", "invalid_json"])
def test_retry_advice_stays_in_receipt_not_worker_guard(tmp_path, monkeypatch, lane, previous, failure):
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "board"))
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
    for key in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_TASK"):
        monkeypatch.delenv(key, raising=False)

    calls = []
    def unavailable(command, **kwargs):
        calls.append(command)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 30)
        return subprocess.CompletedProcess(command, 0, stdout="not JSON", stderr="")

    # Substitute only the external transport. Collection, receipt classification,
    # SQLite completion, and dispatch eligibility all execute their real paths.
    monkeypatch.setattr(acceptance.subprocess, "run", unavailable)
    kbc.init_db()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="Retry acceptance", workspace_kind="scratch",
                             completion_contract="https://github.com/example/widgets/pull/1")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status=?, last_failure_error=? WHERE id=?",
                         (lane, previous, tid))
        assert not kb.complete_task(conn, tid, result="retry")
        assert len(calls) == 1
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance' ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()[0])
        assert receipt["classification"] == "infra"
        assert "authentication" in receipt["detail"]
        assert kb.get_task(conn, tid).last_failure_error == previous
        assert kb.get_task(conn, tid).status == lane
        assert dispatch.check_respawn_guard(conn, tid, lane=lane) == "acceptance_rejected"
        assert store.clear_acceptance_hold(conn, tid)
        expected = "blocker_auth" if previous else None
        assert dispatch.check_respawn_guard(conn, tid, lane=lane) == expected
        assert kb.get_task(conn, tid).last_failure_error == previous
        assert kb.get_task(conn, tid).status == lane

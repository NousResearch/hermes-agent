"""Opt-in human waits reuse Kanban's blocked state and durable event log."""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban import run_slash


@pytest.fixture
def board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()


def _wait(conn, expires_at: int) -> tuple[str, int]:
    tid = kb.create_task(conn, title="publish exact candidate", assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.block_task(
        conn, tid, kind="human_gate", reason="approve this exact operation",
        human_gate={"operation": "merge", "target": "a" * 40, "expires_at": expires_at},
    )
    gate = kb.latest_human_gate(conn, tid)
    assert gate is not None
    return tid, gate[0]


def test_exact_decision_survives_restart_and_resumes_once(board):
    with kbc.connect_closing() as conn:
        tid, event_id = _wait(conn, int(time.time()) + 3600)

    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, tid).status == "blocked"
        assert not kb.unblock_task(conn, tid)
        assert not kb.schedule_task(conn, tid, reason="time delay cannot replace consent")
        assert kb.get_task(conn, tid).status == "blocked"
        assert not kb.unblock_task(conn, tid, gate_event_id=event_id + 1,
                                   expected_target="a" * 40, decision="approve", actor="telegram:42")
        assert not kb.unblock_task(conn, tid, gate_event_id=event_id,
                                   expected_target="b" * 40, decision="approve", actor="telegram:42")

    def approve_once() -> bool:
        with kbc.connect_closing() as conn:
            return kb.unblock_task(conn, tid, gate_event_id=event_id,
                                   expected_target="a" * 40, decision="approve", actor="telegram:42")

    with ThreadPoolExecutor(max_workers=2) as workers:
        assert sorted(workers.map(lambda _: approve_once(), range(2))) == [False, True]

    with kbc.connect_closing() as conn:
        decisions = [e for e in kb.list_events(conn, tid) if e.kind == "human_decision"]
        assert len(decisions) == 1
        assert decisions[0].payload["actor"] == "telegram:42"
        assert kb.latest_human_gate(conn, tid) is None

    with kbc.connect_closing() as conn:
        assert kb.claim_task(conn, tid, claimer="worker") is not None

    with kbc.connect_closing() as conn:
        cli_task = kb.create_task(conn, title="CLI decision", assignee="worker")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (cli_task,))
    deadline = int(time.time()) + 3600
    output = run_slash(
        f"block {cli_task} --kind human_gate --gate-operation merge "
        f"--gate-target {'b' * 40} --gate-expires-at {deadline}"
    )
    assert "for human decision" in output
    with kbc.connect_closing() as conn:
        cli_event = kb.latest_human_gate(conn, cli_task)[0]
    command = (
        f"decide {cli_task} --gate-event-id {cli_event} --gate-target {'b' * 40} --approve"
    )
    assert "configured admin identity" in run_slash(command, gateway_call=True)
    assert "configured admin identity" in run_slash(
        "--bo=default " + command, gateway_call=True
    )
    output = run_slash(command, operator_actor="telegram:42", gateway_call=True)
    assert f"Approved human gate {cli_event}" in output
    with kbc.connect_closing() as conn:
        decision = [e for e in kb.list_events(conn, cli_task) if e.kind == "human_decision"][-1]
        assert decision.payload["actor"] == "telegram:42"


def test_denial_and_expiry_fail_closed_even_after_status_edit(board, monkeypatch):
    now = int(time.time())
    with kbc.connect_closing() as conn:
        denied, denied_event = _wait(conn, now + 3600)
        assert not kb.unblock_task(conn, denied, gate_event_id=denied_event,
                                   expected_target="a" * 40, decision="deny", actor="telegram:42")
        assert kb.latest_human_gate(conn, denied)[2] == "deny"
        assert not kb.unblock_task(conn, denied, gate_event_id=denied_event,
                                   expected_target="a" * 40, decision="approve", actor="telegram:42")
        expired, expired_event = _wait(conn, now + 3600)
        monkeypatch.setattr(kb, "time", SimpleNamespace(time=lambda: now + 3601))
        assert not kb.unblock_task(conn, expired, gate_event_id=expired_event,
                                   expected_target="a" * 40, decision="approve", actor="telegram:42")
        # A noncanonical status edit does not give the dispatcher permission to run.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (denied,))
        assert kb.claim_task(conn, denied, claimer="worker") is None
        assert kb.get_task(conn, denied).status == "ready"

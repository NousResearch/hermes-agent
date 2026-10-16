"""``proof:<command>`` completion contracts: ``done`` only on exit 0 in the workspace."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_pr_acceptance import validate_contract
from hermes_cli.kanban_proof import collect_proof

pytestmark = pytest.mark.platforms("posix")

PY = shlex.quote(sys.executable)


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as c:
        yield c


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


def _proof_task(conn, workspace: Path, command: str, **kwargs) -> str:
    return kb.create_task(
        conn, title="proof me", assignee="worker", workspace_kind="dir",
        workspace_path=str(workspace), completion_contract=f"proof:{command}", **kwargs,
    )


def _receipts(conn, tid: str) -> list[dict]:
    return [json.loads(r[0]) for r in conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind='proof_acceptance' ORDER BY id", (tid,))]


def _status(conn, tid: str) -> str:
    return conn.execute("SELECT status FROM tasks WHERE id=?", (tid,)).fetchone()["status"]


# --- contract validation ---------------------------------------------------

def test_validate_contract_normalises_proof_commands():
    assert validate_contract("proof:  pytest -q  ") == "proof:pytest -q"
    assert validate_contract("local-only") == "local-only"
    assert validate_contract("acme/repo") == "acme/repo"


@pytest.mark.parametrize("bad", ["proof:", "proof:   ", "proof:" + "x" * 2001, "proof:echo\x00hi"])
def test_validate_contract_rejects_empty_or_oversized_proof(bad):
    with pytest.raises(ValueError, match="proof:"):
        validate_contract(bad)


def test_create_task_rejects_bad_proof(conn, workspace):
    with pytest.raises(ValueError):
        _proof_task(conn, workspace, "   ")


# --- the gate ------------------------------------------------------------------

def test_completion_refused_until_proof_exits_zero(conn, workspace):
    tid = _proof_task(conn, workspace, "test -f report.md")

    assert kb.complete_task(conn, tid, summary="wrote the report (it did not)") is False
    assert _status(conn, tid) != "done"
    receipt = _receipts(conn, tid)[-1]
    assert receipt["ok"] is False
    assert receipt["classification"] == "failure"
    assert receipt["exit_code"] == 1
    assert receipt["command"] == "test -f report.md"
    assert receipt["cwd"] == str(workspace)
    error = kb.get_task(conn, tid).last_failure_error
    assert error.startswith("Proof failure")
    assert "exited 1" in error
    assert "retry completion" in error

    (workspace / "report.md").write_text("# done\n")
    assert kb.complete_task(conn, tid, summary="wrote the report") is True
    assert _status(conn, tid) == "done"
    receipt = _receipts(conn, tid)[-1]
    assert receipt["ok"] is True
    assert receipt["classification"] == "success"
    assert receipt["exit_code"] == 0


def test_proof_runs_in_workspace_with_task_env_and_no_credentials(conn, workspace, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot-should-not-leak")
    probe = (
        "import json, os; json.dump({'cwd': os.getcwd(), 'task': os.environ.get('HERMES_KANBAN_TASK'), "
        "'ws': os.environ.get('HERMES_KANBAN_WORKSPACE'), "
        "'leaked': [k for k in ('OPENROUTER_API_KEY', 'TELEGRAM_BOT_TOKEN') if k in os.environ]}, "
        "open('probe.json', 'w'))"
    )
    tid = _proof_task(conn, workspace, f"{PY} -c {shlex.quote(probe)}")

    assert kb.complete_task(conn, tid, summary="probe") is True
    seen = json.loads((workspace / "probe.json").read_text())
    assert Path(seen["cwd"]).resolve() == workspace.resolve()
    assert seen["task"] == tid
    assert Path(seen["ws"]).resolve() == workspace.resolve()
    assert seen["leaked"] == []


def test_proof_output_tail_is_captured_and_shown_in_failure(conn, workspace):
    tid = _proof_task(conn, workspace, "echo missing-section >&2; echo partial; exit 3")

    assert kb.complete_task(conn, tid, summary="claimed done") is False
    receipt = _receipts(conn, tid)[-1]
    assert receipt["exit_code"] == 3
    assert "missing-section" in receipt["stderr_tail"]
    assert "partial" in receipt["stdout_tail"]
    assert "missing-section" in kb.get_task(conn, tid).last_failure_error


def test_proof_timeout_refuses_and_kills_the_process_group(conn, workspace, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_PROOF_TIMEOUT", "1")
    tid = _proof_task(conn, workspace, "sleep 30")

    assert kb.complete_task(conn, tid, summary="still running") is False
    assert _status(conn, tid) != "done"
    receipt = _receipts(conn, tid)[-1]
    assert receipt["classification"] == "timeout"
    assert receipt["exit_code"] is None
    assert receipt["duration_ms"] < 15_000
    assert "timeout" in kb.get_task(conn, tid).last_failure_error


def test_proof_without_workspace_directory_is_refused(conn, tmp_path):
    tid = kb.create_task(conn, title="no workspace", assignee="worker",
                         completion_contract="proof:true")
    assert kb.get_task(conn, tid).workspace_path is None

    assert kb.complete_task(conn, tid, summary="done") is False
    receipt = _receipts(conn, tid)[-1]
    assert receipt["classification"] == "workspace_missing"
    assert receipt["cwd"] is None
    assert "workspace" in kb.get_task(conn, tid).last_failure_error


def test_relative_workspace_path_never_runs_the_proof(tmp_path):
    receipt = collect_proof("proof:true", task_id="t_x", workspace_kind="dir", workspace_path="relative/dir")
    assert receipt["classification"] == "workspace_missing"


def test_review_approval_still_runs_the_proof(conn, workspace):
    tid = _proof_task(conn, workspace, "test -f approved.md")
    assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
    assert kb.request_review(conn, tid, summary="please check") is True
    assert _status(conn, tid) == "review"

    assert kb.complete_task(conn, tid) is False
    assert _status(conn, tid) == "review"
    assert _receipts(conn, tid)[-1]["classification"] == "failure"

    (workspace / "approved.md").write_text("ok\n")
    assert kb.complete_task(conn, tid) is True
    assert _status(conn, tid) == "done"


def test_claimed_worker_completion_runs_proof_with_run_ownership(conn, workspace):
    tid = _proof_task(conn, workspace, "test -f out.txt")
    owner = kb.claim_task(conn, tid, claimer=kb._claimer_id())
    run_id = owner.current_run_id

    assert kb.complete_task(conn, tid, summary="x", expected_run_id=run_id) is False
    assert _status(conn, tid) == "running"
    assert _receipts(conn, tid)[-1]["ok"] is False

    (workspace / "out.txt").write_text("x")
    assert kb.complete_task(conn, tid, summary="x", expected_run_id=run_id) is True
    assert _status(conn, tid) == "done"


def test_local_only_cards_never_run_a_proof(conn, workspace):
    tid = kb.create_task(conn, title="plain", assignee="worker", workspace_kind="dir",
                         workspace_path=str(workspace))
    assert kb.complete_task(conn, tid, summary="done") is True
    assert _receipts(conn, tid) == []

"""Typed completion contracts: refuse done without the declared artifact, persist the report."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.discard(str(kb.kanban_db_path(board="default").resolve()))
    kb.init_db()
    with kbc.connect() as connection:
        yield connection


ARTIFACTS = {
    "pr-merged": {"type": "pr-merged", "url": "https://github.com/acme/repo/pull/1", "merged": True, "merge_commit": "deadbeef"},
    "deployed": {"type": "deployed", "url": "https://dev.example.test", "smoke_test": {"command": "curl /health", "exit_code": 0}},
    "document-committed": {"type": "document-committed", "path": "docs/decision.md", "commit": "deadbeef"},
    "decision-recorded": {"type": "decision-recorded", "path": "docs/adr/001.md", "decision": "accepted"},
}


def _running_task(conn, completion_contract="local-only") -> str:
    tid = kb.create_task(
        conn, title="gated completion", assignee="coder",
        completion_contract=completion_contract,
    )
    assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
    return tid


def _last_run_metadata(conn, tid: str) -> str:
    row = conn.execute(
        "SELECT metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1", (tid,)
    ).fetchone()
    assert row is not None
    return row["metadata"]


@pytest.mark.parametrize("contract", sorted(ARTIFACTS))
def test_typed_card_contract_requires_and_persists_its_artifact(conn, contract):
    tid = _running_task(conn, contract)

    assert kb.complete_task(
        conn, tid, result="verified", metadata={"verification": {"artifact": ARTIFACTS[contract]}},
    ) is True

    assert kb.get_task(conn, tid).status == "done"
    assert '"required_artifact": true' in _last_run_metadata(conn, tid)
    event = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'completed' ORDER BY id DESC LIMIT 1",
        (tid,),
    ).fetchone()
    assert event is not None
    assert '"verification_report"' in event["payload"]


@pytest.mark.parametrize("contract", sorted(ARTIFACTS))
def test_typed_card_contract_refuses_missing_artifact(conn, contract):
    tid = _running_task(conn, contract)

    with pytest.raises(kb.CompletionVerificationError) as error:
        kb.complete_task(conn, tid, result="claimed without proof")

    assert [item["code"] for item in error.value.report["failures"]] == ["missing_artifact"]
    assert kb.get_task(conn, tid).status == "running"
    events = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id", (tid,)
    ).fetchall()
    assert events[-1]["kind"] == "completion_blocked_verification"
    assert "missing_artifact" in events[-1]["payload"]


def test_typed_card_contract_refuses_artifact_of_another_type(conn):
    tid = _running_task(conn, "deployed")

    with pytest.raises(kb.CompletionVerificationError) as error:
        kb.complete_task(
            conn, tid, result="wrong proof", metadata={"verification": {"artifact": ARTIFACTS["pr-merged"]}},
        )

    assert error.value.report["failures"][0]["code"] == "missing_artifact"
    assert kb.get_task(conn, tid).status == "running"


def test_deployed_contract_refuses_failed_smoke_test(conn):
    tid = _running_task(conn, "deployed")
    artifact = dict(ARTIFACTS["deployed"], smoke_test={"command": "curl /health", "exit_code": 1})

    with pytest.raises(kb.CompletionVerificationError) as error:
        kb.complete_task(conn, tid, result="smoke failed", metadata={"verification": {"artifact": artifact}})

    assert error.value.report["failures"][0]["code"] == "incomplete_artifact"
    assert "smoke_test.exit_code=0" in error.value.report["failures"][0]["detail"]
    assert kb.get_task(conn, tid).status == "running"


def test_pr_merged_contract_refuses_incomplete_artifact(conn):
    tid = _running_task(conn, "pr-merged")
    artifact = {"type": "pr-merged", "url": "https://github.com/acme/repo/pull/1", "merged": True}

    with pytest.raises(kb.CompletionVerificationError) as error:
        kb.complete_task(conn, tid, result="no merge sha", metadata={"verification": {"artifact": artifact}})

    assert error.value.report["failures"] == [
        {"code": "incomplete_artifact", "detail": "artifact is missing: merge_commit"},
    ]


def test_local_only_completes_without_verification_payload(conn):
    tid = _running_task(conn)

    assert kb.complete_task(conn, tid, result="done locally") is True

    assert kb.get_task(conn, tid).status == "done"
    assert "verification_report" not in (_last_run_metadata(conn, tid) or "")

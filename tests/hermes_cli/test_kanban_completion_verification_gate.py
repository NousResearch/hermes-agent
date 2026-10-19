"""Completion gate integration: reject before done and persist accepted evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def home(tmp_path):
    path = tmp_path / ".hermes"
    path.mkdir()
    return path


@pytest.fixture
def conn(home, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.discard(str(kb.kanban_db_path(board="default").resolve()))
    kb.init_db()
    with kbc.connect() as connection:
        yield connection


def _enable_completion_gate(home: Path) -> None:
    (home / "config.yaml").write_text("kanban:\n  completion_gate: true\n", encoding="utf-8")


def _valid_verification() -> dict:
    return {
        "task": {
            "status": "done",
            "summary": "implemented",
            "evidence": [{"command": "pytest", "exit_code": 0, "output": "passed"}],
        },
        "repo": {
            "dirty": False,
            "commit": "abc1234",
            "pushed": True,
            "pr": {"number": 1, "state": "open"},
            "ci": {"status": "success"},
            "review": {"approved": True},
        },
        "preflight": {
            "requested_skills": [],
            "available_skills": [],
            "review_checks": [],
            "supported_flags": [],
            "dependency_copies": 0,
            "max_dependency_copies": 0,
        },
        "scope": {"changed_paths": [], "allowed_roots": [], "secret_findings": []},
    }


def _running_task(conn, completion_contract="local-only") -> str:
    tid = kb.create_task(
        conn, title="gated completion", assignee="coder",
        completion_contract=completion_contract,
    )
    assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
    return tid


def test_rejected_completion_remains_running_and_records_failure(conn, home):
    _enable_completion_gate(home)
    tid = _running_task(conn)
    payload = _valid_verification()
    payload["repo"]["dirty"] = True

    with pytest.raises(kb.CompletionVerificationError) as error:
        kb.complete_task(conn, tid, result="blocked", metadata={"verification": payload})

    assert error.value.report["ok"] is False
    assert kb.get_task(conn, tid).status == "running"
    events = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id", (tid,)
    ).fetchall()
    assert events[-1]["kind"] == "completion_blocked_verification"
    assert "unsaved_work" in events[-1]["payload"]


def test_accepted_completion_preserves_stable_verification_report(conn, home):
    _enable_completion_gate(home)
    tid = _running_task(conn)
    payload = _valid_verification()

    assert kb.complete_task(conn, tid, result="shipped", metadata={"verification": payload}) is True

    assert kb.get_task(conn, tid).status == "done"
    run = conn.execute(
        "SELECT metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1", (tid,)
    ).fetchone()
    assert run is not None
    assert '"ok": true' in run["metadata"]
    event = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'completed' ORDER BY id DESC LIMIT 1",
        (tid,),
    ).fetchone()
    assert event is not None
    assert '"verification_report"' in event["payload"]


@pytest.mark.parametrize(
    ("contract", "artifact"),
    [
        ("pr-merged", {"type": "pr-merged", "url": "https://github.com/acme/repo/pull/1", "merged": True, "merge_commit": "deadbeef"}),
        ("deployed", {"type": "deployed", "url": "https://dev.example.test", "smoke_test": {"command": "curl /health", "exit_code": 0}}),
        ("document-committed", {"type": "document-committed", "path": "docs/decision.md", "commit": "deadbeef"}),
        ("decision-recorded", {"type": "decision-recorded", "path": "docs/adr/001.md", "decision": "accepted"}),
    ],
)
def test_typed_card_contract_requires_and_persists_its_artifact(conn, home, contract, artifact):
    _enable_completion_gate(home)
    tid = _running_task(conn, contract)
    payload = _valid_verification()
    payload["artifact"] = artifact

    assert kb.complete_task(conn, tid, result="verified", metadata={"verification": payload}) is True
    run = conn.execute(
        "SELECT metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1", (tid,)
    ).fetchone()
    assert run is not None
    assert '"required_artifact": true' in run["metadata"]


def test_typed_card_contract_refuses_missing_artifact(conn, home):
    _enable_completion_gate(home)
    tid = _running_task(conn, "deployed")

    with pytest.raises(kb.CompletionVerificationError) as error:
        kb.complete_task(conn, tid, result="not deployed", metadata={"verification": _valid_verification()})

    assert error.value.report["ok"] is False
    assert any(item["code"] == "missing_artifact" for item in error.value.report["failures"])
    assert kb.get_task(conn, tid).status == "running"


_DEPLOYED_ARTIFACT = {
    "type": "deployed", "url": "https://dev.example.test",
    "smoke_test": {"command": "curl /health", "exit_code": 0},
}


def test_typed_card_contract_is_enforced_without_completion_gate(conn):
    tid = _running_task(conn, "deployed")

    with pytest.raises(kb.CompletionVerificationError) as error:
        kb.complete_task(conn, tid, result="not deployed")

    assert [item["code"] for item in error.value.report["failures"]] == ["missing_artifact"]
    assert kb.get_task(conn, tid).status == "running"


def test_typed_card_contract_needs_only_its_artifact_without_completion_gate(conn):
    tid = _running_task(conn, "deployed")

    assert kb.complete_task(
        conn, tid, result="deployed", metadata={"verification": {"artifact": _DEPLOYED_ARTIFACT}},
    ) is True

    assert kb.get_task(conn, tid).status == "done"
    run = conn.execute(
        "SELECT metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1", (tid,)
    ).fetchone()
    assert '"required_artifact": true' in run["metadata"]


def test_untyped_contract_is_unchanged_without_completion_gate(conn):
    tid = _running_task(conn)

    assert kb.complete_task(conn, tid, result="done locally") is True
    assert kb.get_task(conn, tid).status == "done"


def test_local_only_completes_without_repository_fields(conn, home):
    _enable_completion_gate(home)
    tid = _running_task(conn)
    payload = _valid_verification()
    del payload["repo"]

    assert kb.complete_task(conn, tid, result="local", metadata={"verification": payload}) is True
    assert kb.get_task(conn, tid).status == "done"


@pytest.mark.parametrize("contract", ["deployed", "document-committed", "decision-recorded"])
def test_non_pr_typed_contract_skips_repository_predicates(conn, home, contract):
    _enable_completion_gate(home)
    tid = _running_task(conn, contract)
    artifact = {
        "deployed": _DEPLOYED_ARTIFACT,
        "document-committed": {"type": "document-committed", "path": "docs/d.md", "commit": "deadbeef"},
        "decision-recorded": {"type": "decision-recorded", "path": "docs/adr/001.md", "decision": "accepted"},
    }[contract]
    payload = _valid_verification()
    del payload["repo"]
    payload["artifact"] = artifact

    assert kb.complete_task(conn, tid, result="verified", metadata={"verification": payload}) is True


@pytest.mark.parametrize("contract", ["pr-merged", "acme/repo"])
def test_pr_contract_still_requires_repository_predicates(conn, home, contract):
    _enable_completion_gate(home)
    tid = _running_task(conn, contract)
    payload = _valid_verification()
    del payload["repo"]
    payload["artifact"] = {
        "type": "pr-merged", "url": "https://github.com/acme/repo/pull/1", "merged": True, "merge_commit": "deadbeef",
    }

    with pytest.raises(kb.CompletionVerificationError) as error:
        kb.complete_task(conn, tid, result="no pr", metadata={"verification": payload})

    codes = {item["code"] for item in error.value.report["failures"]}
    assert {"missing_commit", "missing_push", "missing_pr", "ci_not_green", "missing_review"} <= codes
    assert kb.get_task(conn, tid).status == "running"

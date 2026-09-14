"""Completion-evidence contracts are typed, durable, and workspace-aware."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_completion_evidence import CompletionEvidenceError


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_workspace_relative_proof_is_resolved_and_persisted(kanban_home: Path, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    artifact = workspace / "reports" / "final.txt"
    artifact.parent.mkdir()
    artifact.write_text("verified", encoding="utf-8")

    with kbc.connect_closing() as conn:
        tid = kb.create_task(
            conn, title="evidenced", assignee="worker", workspace_kind="dir",
            workspace_path=str(workspace), completion_contract="evidence",
        )
        assert kb.complete_task(conn, tid, summary="done", proof=["path:reports/final.txt"])

        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.completion_proof == [{
            "type": "path",
            "value": "reports/final.txt",
            "resolved": str(artifact.resolve()),
            "size": len(b"verified"),
            "sha256": hashlib.sha256(b"verified").hexdigest(),
        }]
        evidence_events = [e for e in kb.list_events(conn, tid) if e.kind == "completion_evidence_recorded"]
        assert evidence_events[-1].payload == {"proof": task.completion_proof}


def test_evidence_contract_requires_proof_or_audited_override(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="gated", assignee="worker", completion_contract="evidence")
        with pytest.raises(CompletionEvidenceError, match="completion requires proof"):
            kb.complete_task(conn, tid, summary="claimed done")
        assert kb.get_task(conn, tid).status != "done"

        assert kb.complete_task(conn, tid, summary="human override", accept_unproven=True)
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "done"
        assert task.completion_proof is None
        override_events = [e for e in kb.list_events(conn, tid) if e.kind == "card_closed_without_proof"]
        assert override_events[-1].payload == {
            "completion_contract": "evidence",
            "explicit_override": True,
        }


def test_path_proof_rejects_workspace_container(kanban_home: Path, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with kbc.connect_closing() as conn:
        tid = kb.create_task(
            conn, title="gated", workspace_kind="dir", workspace_path=str(workspace),
            completion_contract="evidence",
        )
        with pytest.raises(CompletionEvidenceError, match="must identify a file"):
            kb.complete_task(conn, tid, proof=["path:."])
        assert kb.get_task(conn, tid).status != "done"


def test_url_proof_rejects_reusable_credentials_before_persistence(kanban_home: Path) -> None:
    secret = "super-secret-value"
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="gated", completion_contract="evidence")
        with pytest.raises(CompletionEvidenceError, match="must not contain reusable credentials"):
            kb.complete_task(conn, tid, proof=[f"url:https://example.com/run?token={secret}"])
        task = kb.get_task(conn, tid)
        assert task.status != "done"
        assert task.completion_proof is None
        assert secret not in str([event.payload for event in kb.list_events(conn, tid)])


def test_task_proof_requires_distinct_completed_linked_task(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        unrelated = kb.create_task(conn, title="unrelated")
        assert kb.complete_task(conn, unrelated)
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child", parents=[parent], completion_contract="evidence")

        for invalid in (child, unrelated, parent):
            with pytest.raises(CompletionEvidenceError, match="different, completed task directly linked"):
                kb.complete_task(conn, child, proof=[f"task:{invalid}"])

        assert kb.complete_task(conn, parent)
        assert kb.complete_task(conn, child, proof=[f"task:{parent}"])


def test_attachment_is_rechecked_after_prepare_before_terminal_write(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import kanban_completion_evidence as evidence

    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="gated", completion_contract="evidence")
        attachment_id = kb.add_attachment(
            conn, tid, filename="proof.txt", stored_path="proof.txt", size=5,
        )
        original_prepare = evidence.prepare_completion_evidence

        def prepare_then_remove(*args, **kwargs):
            prepared = original_prepare(*args, **kwargs)
            with kbc.connect_closing() as rival:
                removed = kb.delete_attachment(rival, attachment_id)
                assert removed is not None
            return prepared

        monkeypatch.setattr(evidence, "prepare_completion_evidence", prepare_then_remove)
        with pytest.raises(CompletionEvidenceError, match="does not belong"):
            kb.complete_task(conn, tid, proof=[f"attachment:{attachment_id}"])
        assert kb.get_task(conn, tid).status != "done"
        assert kb.get_task(conn, tid).completion_proof is None

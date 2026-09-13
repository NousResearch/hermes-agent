"""Completion-evidence contracts are typed, durable, and workspace-aware."""
from __future__ import annotations

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

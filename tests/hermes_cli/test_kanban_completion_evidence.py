"""Regression coverage for coding-task completion evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _running_coding_task(conn):
    task_id = kb.create_task(
        conn,
        title="fix API persistence bug",
        body="Change repository code, run tests, and open a pull request.",
        assignee="programmer",
        workspace_kind="worktree",
        workspace_path="/tmp/hermes-test-worktree",
    )
    assert kb.claim_task(conn, task_id) is not None
    return task_id


def test_coding_task_completion_without_evidence_is_rejected_and_audited(kanban_home):
    with kb.connect() as conn:
        task_id = _running_coding_task(conn)

        with pytest.raises(kb.CompletionEvidenceError) as exc_info:
            kb.complete_task(
                conn,
                task_id,
                summary="Code changed locally; tests pass; PR can be opened later.",
                metadata={
                    "changed_files": ["hermes_cli/kanban_db.py"],
                    "tests_run": ["python -m pytest tests/hermes_cli/test_kanban_completion_evidence.py -q"],
                },
            )

        task = kb.get_task(conn, task_id)
        events = kb.list_events(conn, task_id)

    assert "GitHub PR URL plus passing CI/checks" in exc_info.value.reason
    assert task is not None
    assert task.status == "running"
    assert "completed" not in [event.kind for event in events]
    evidence_events = [event for event in events if event.kind == "completion_blocked_evidence"]
    assert len(evidence_events) == 1
    evidence_payload = evidence_events[0].payload or {}
    assert "changed_files" in evidence_payload["metadata_keys"]


def test_coding_task_completion_accepts_pr_url_with_passing_checks(kanban_home):
    with kb.connect() as conn:
        task_id = _running_coding_task(conn)

        assert kb.complete_task(
            conn,
            task_id,
            summary=(
                "PR https://github.com/NousResearch/hermes-agent/pull/123 "
                "has passing checks."
            ),
            metadata={
                "pr_url": "https://github.com/NousResearch/hermes-agent/pull/123",
                "ci_status": "passed",
            },
        )
        task = kb.get_task(conn, task_id)

    assert task is not None
    assert task.status == "done"


def test_coding_task_completion_rejects_incomplete_evidence_shapes(kanban_home):
    with kb.connect() as conn:
        pr_only = _running_coding_task(conn)
        with pytest.raises(kb.CompletionEvidenceError):
            kb.complete_task(
                conn,
                pr_only,
                summary="PR https://github.com/NousResearch/hermes-agent/pull/123 is open.",
                metadata={"pr_url": "https://github.com/NousResearch/hermes-agent/pull/123"},
            )
        pr_task = kb.get_task(conn, pr_only)
        assert pr_task is not None
        assert pr_task.status == "running"

        bare_waiver = _running_coding_task(conn)
        with pytest.raises(kb.CompletionEvidenceError):
            kb.complete_task(
                conn,
                bare_waiver,
                summary="Done under waiver.",
                metadata={"human_waiver": True},
            )
        waiver_task = kb.get_task(conn, bare_waiver)
        assert waiver_task is not None
        assert waiver_task.status == "running"


def test_coding_task_completion_accepts_auditable_human_waiver(kanban_home):
    with kb.connect() as conn:
        task_id = _running_coding_task(conn)

        assert kb.complete_task(
            conn,
            task_id,
            summary="Done under explicit human waiver.",
            metadata={
                "human_waiver": {
                    "approved_by": "maintainer",
                    "reason": "Maintainer accepted out-of-band completion proof.",
                },
            },
        )
        task = kb.get_task(conn, task_id)
        run = kb.latest_run(conn, task_id)

    assert task is not None
    assert run is not None
    assert task.status == "done"
    assert run.metadata["human_waiver"]["approved_by"] == "maintainer"


def test_non_coding_task_completion_does_not_require_pr_evidence(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="summarize source notes",
            body="Produce a short read-only research summary.",
            assignee="researcher",
        )
        assert kb.claim_task(conn, task_id) is not None

        assert kb.complete_task(conn, task_id, summary="Summary written.")
        task = kb.get_task(conn, task_id)

    assert task is not None
    assert task.status == "done"

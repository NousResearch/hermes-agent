import os
from pathlib import Path

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_validation as kv


class PendingRun:
    def __init__(self, summary=None, metadata=None):
        self.summary = summary
        self.metadata = metadata


def _isolated_board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    kb.init_db()


def test_validate_flags_promissory_known_failure_without_issue(tmp_path, monkeypatch):
    _isolated_board(tmp_path, monkeypatch)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="Review PR", assignee="reviewer")
        task = kb.get_task(conn, tid)
    findings = kv.validate_tasks(
        [task],
        {
            tid: [
                PendingRun(
                    summary="Known pre-existing failure: bootstrap still fails; issue will be created.",
                    metadata={"commands_run": [{"command": "pytest", "exit_code": 1}]},
                )
            ]
        },
    )
    assert any(f.code == "known_failure_not_verified_tracked" for f in findings)


def test_validate_accepts_clean_repo_completion_with_command_exit_codes(tmp_path, monkeypatch):
    _isolated_board(tmp_path, monkeypatch)
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="Implement UI page",
            body="validation_assertions_to_satisfy: VC-001",
            assignee="fullstack-eng",
            workspace_kind="worktree",
            workspace_path=str(tmp_path),
        )
        task = kb.get_task(conn, tid)
    findings = kv.validate_tasks(
        [task],
        {
            tid: [
                PendingRun(
                    summary="done",
                    metadata={
                        "commands_run": [
                            {
                                "command": "pytest",
                                "exit_code": 0,
                                "purpose": "tests",
                                "result_summary": "passed",
                            }
                        ],
                        "commit": "abc1234",
                        "git_status": "clean",
                        "changed_files": ["app/page.tsx"],
                        "validation_assertions_satisfied": ["VC-001"],
                        "validation_assertions_failed": [],
                    },
                )
            ]
        },
    )
    assert [f.to_dict() for f in findings] == []


def test_validate_flags_missing_live_qa_evidence_for_user_visible_card(tmp_path, monkeypatch):
    _isolated_board(tmp_path, monkeypatch)
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="QA browser page flow VC-002",
            body="validation_assertions_to_verify: VC-002",
            assignee="qa-eng",
        )
        task = kb.get_task(conn, tid)
    findings = kv.validate_tasks(
        [task],
        {
            tid: [
                PendingRun(
                    summary="qa done",
                    metadata={
                        "commands_run": [
                            {"command": "npm run test:e2e", "exit_code": 0}
                        ],
                    },
                )
            ]
        },
    )
    assert any(f.code == "user_visible_qa_missing_live_evidence" for f in findings)

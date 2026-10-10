"""Regression test for t_e2cf891c: review/verifier-tagged kanban tasks must
dispatch their worker subprocess with --ignore-rules so it does not load the
profile's own MEMORY.md/USER.md/preloaded skills — the same isolation
delegate_task hardcodes for its children (tools/delegate_tool.py). This is
isolation only; it does not remove the task's own explicitly-forced --skills.
"""
from __future__ import annotations

from unittest.mock import patch

from hermes_cli.kanban_db import Task
from hermes_cli.kanban_db_dispatch import _is_review_tagged, _worker_argv


def _make_task(skills):
    return Task(
        id="t_test0001",
        title="test task",
        body=None,
        assignee="default",
        status="running",
        priority=0,
        created_by=None,
        created_at=0,
        started_at=None,
        completed_at=None,
        workspace_kind="scratch",
        workspace_path=None,
        claim_lock=None,
        claim_expires=None,
        tenant=None,
        skills=skills,
    )


def _build_argv(task):
    with patch(
        "hermes_cli.kanban_db_dispatch._resolve_hermes_argv",
        return_value=["hermes"],
    ), patch(
        "hermes_cli.kanban_db_dispatch._resolve_worker_cli_toolsets",
        return_value=None,
    ):
        return _worker_argv(task, "default", None)


def test_review_tagged_task_gets_ignore_rules():
    task = _make_task(["kanban-independent-verification"])
    assert _is_review_tagged(task) is True

    argv = _build_argv(task)
    assert "--ignore-rules" in argv
    # --ignore-rules must not drop the task's own forced --skills.
    assert "--skills" in argv
    assert "kanban-independent-verification" in argv


def test_requesting_code_review_tag_also_gets_ignore_rules():
    task = _make_task(["requesting-code-review"])
    assert _is_review_tagged(task) is True
    assert "--ignore-rules" in _build_argv(task)


def test_untagged_task_does_not_get_ignore_rules():
    task = _make_task(["some-other-skill"])
    assert _is_review_tagged(task) is False

    argv = _build_argv(task)
    assert "--ignore-rules" not in argv
    assert "--skills" in argv


def test_task_with_no_skills_does_not_get_ignore_rules():
    task = _make_task(None)
    assert _is_review_tagged(task) is False

    argv = _build_argv(task)
    assert "--ignore-rules" not in argv
    assert "--skills" not in argv

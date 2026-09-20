"""Regression coverage for profile-scoped defaults on newly created tasks."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        yield conn
    finally:
        conn.close()


def test_dispatch_defaults_match_worker_budget_contract():
    resolved = kb.load_dispatch_config({})

    assert resolved.default_max_runtime_seconds == 5400
    # This tree's shipped breaker default (config_defaults pins it explicitly;
    # the 010804 stash era used 3 before it was tuned down).
    assert resolved.failure_limit == 2
    assert kb.load_dispatch_config(
        {"kanban": {"default_max_runtime_seconds": None}}
    ).default_max_runtime_seconds is None


def test_create_task_uses_configured_runtime_only_when_omitted(board):
    default_id = kb.create_task(board, title="default runtime")
    override_id = kb.create_task(
        board,
        title="explicit runtime",
        max_runtime_seconds=123,
    )
    unbounded_id = kb.create_task(
        board,
        title="explicitly unbounded",
        max_runtime_seconds=None,
    )

    assert kb.get_task(board, default_id).max_runtime_seconds == 5400
    assert kb.get_task(board, override_id).max_runtime_seconds == 123
    assert kb.get_task(board, unbounded_id).max_runtime_seconds is None


def test_create_task_persists_explicit_retry_override(board):
    default_id = kb.create_task(board, title="dispatcher default retries")
    override_id = kb.create_task(
        board,
        title="task retry override",
        max_retries=7,
    )

    assert kb.get_task(board, default_id).max_retries is None
    assert kb.get_task(board, override_id).max_retries == 7
    assert kb.load_dispatch_config({}).failure_limit == 2


def test_cli_task_json_exposes_effective_runtime_and_retry_override(board):
    from hermes_cli.kanban import _task_to_dict

    task_id = kb.create_task(
        board,
        title="serialized task",
        max_retries=4,
    )
    task = kb.get_task(board, task_id)

    payload = _task_to_dict(task)

    assert payload["max_runtime_seconds"] == 5400
    assert payload["max_retries"] == 4


def test_runtime_default_is_isolated_per_active_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    profile_a.mkdir()
    profile_b.mkdir()
    (profile_a / "config.yaml").write_text(
        "kanban:\n  default_max_runtime_seconds: 111\n  failure_limit: 5\n",
        encoding="utf-8",
    )
    (profile_b / "config.yaml").write_text(
        "kanban:\n  default_max_runtime_seconds: 222\n  failure_limit: 7\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(profile_a))
    conn_a = kb.connect(tmp_path / "profile-a.db")
    try:
        task_a = kb.create_task(conn_a, title="profile a")
        assert kb.get_task(conn_a, task_a).max_runtime_seconds == 111
        assert kb.load_dispatch_config().failure_limit == 5
    finally:
        conn_a.close()

    monkeypatch.setenv("HERMES_HOME", str(profile_b))
    conn_b = kb.connect(tmp_path / "profile-b.db")
    try:
        task_b = kb.create_task(conn_b, title="profile b")
        assert kb.get_task(conn_b, task_b).max_runtime_seconds == 222
        assert kb.load_dispatch_config().failure_limit == 7
    finally:
        conn_b.close()


def test_existing_unbounded_task_is_not_backfilled(board):
    existing_id = kb.create_task(
        board,
        title="legacy unbounded task",
        max_runtime_seconds=None,
    )
    assert kb.get_task(board, existing_id).max_runtime_seconds is None
    new_id = kb.create_task(board, title="new task")
    assert kb.get_task(board, new_id).max_runtime_seconds == 5400


def test_idempotent_existing_task_keeps_its_original_runtime(board):
    existing_id = kb.create_task(
        board,
        title="idempotent legacy task",
        idempotency_key="legacy-runtime",
        max_runtime_seconds=None,
    )

    returned_id = kb.create_task(
        board,
        title="should not replace legacy task",
        idempotency_key="legacy-runtime",
    )

    assert returned_id == existing_id
    assert kb.get_task(board, existing_id).max_runtime_seconds is None


def test_decomposition_applies_defaults_and_child_overrides(board):
    root_id = kb.create_task(board, title="triage root", triage=True)

    child_ids = kb.decompose_triage_task(
        board,
        root_id,
        root_assignee=None,
        children=[
            {"title": "default child", "assignee": "worker"},
            {
                "title": "overridden child",
                "assignee": "worker",
                "max_runtime_seconds": 45,
                "max_retries": 5,
            },
        ],
    )

    assert child_ids is not None
    default_child = kb.get_task(board, child_ids[0])
    overridden_child = kb.get_task(board, child_ids[1])
    assert default_child.max_runtime_seconds == 5400
    assert overridden_child.max_runtime_seconds == 45
    assert overridden_child.max_retries == 5

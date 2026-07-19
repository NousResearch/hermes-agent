"""Behavioral contract tests for fixed per-task Kanban tool authority."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def authority_board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_contract_is_advertised_only_with_native_task_field(authority_board):
    assert kb.TASK_AUTHORITY_CONTRACT == "task_toolsets.v1"
    with kb.connect() as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
    assert "toolsets" in columns


def test_none_and_empty_allowlists_round_trip_differently(authority_board):
    with kb.connect() as conn:
        legacy_id = kb.create_task(conn, title="legacy profile authority")
        frozen_id = kb.create_task(conn, title="frozen authority", toolsets=[])

        legacy = kb.get_task(conn, legacy_id)
        frozen = kb.get_task(conn, frozen_id)
        raw = conn.execute(
            "SELECT id, toolsets FROM tasks WHERE id IN (?, ?)",
            (legacy_id, frozen_id),
        ).fetchall()

    assert legacy is not None and legacy.toolsets is None
    assert frozen is not None and frozen.toolsets == []
    persisted = {row["id"]: row["toolsets"] for row in raw}
    assert persisted[legacy_id] is None
    assert persisted[frozen_id] == "[]"


def test_explicit_allowlist_is_normalized_and_persisted(authority_board):
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="bounded tools",
            toolsets=["file", "file", "web"],
        )
        task = kb.get_task(conn, task_id)

    assert task is not None
    assert task.toolsets == ["file", "web"]


@pytest.mark.parametrize(
    "toolsets",
    [
        ["not-a-real-toolset"],
        ["all"],
        ["*"],
        ["kanban"],
        ["kanban_worker"],
        ["file,web"],
        [""],
    ],
)
def test_invalid_or_broad_task_authority_fails_closed(authority_board, toolsets):
    with kb.connect() as conn, pytest.raises(ValueError):
        kb.create_task(conn, title="must fail", toolsets=toolsets)


def test_restricted_worker_bundle_exposes_only_own_task_lifecycle(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_authority")

    from model_tools import get_tool_definitions

    tools = get_tool_definitions(
        ["kanban_worker"],
        quiet_mode=True,
        skip_tool_search_assembly=True,
    )
    names = {tool["function"]["name"] for tool in tools}

    assert {
        "kanban_show",
        "kanban_complete",
        "kanban_block",
        "kanban_heartbeat",
        "kanban_comment",
        "kanban_attach",
        "kanban_attach_url",
        "kanban_attachments",
    } <= names
    assert not {
        "kanban_list",
        "kanban_create",
        "kanban_link",
        "kanban_unblock",
        "delegate_task",
        "terminal",
        "memory",
    } & names

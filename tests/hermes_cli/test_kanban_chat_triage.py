from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_chat_triage import ChatTriageError, route_chat_triage_task


@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD",
    ):
        monkeypatch.delenv(var, raising=False)
    try:
        import hermes_constants
        hermes_constants._cached_default_hermes_root = None  # type: ignore[attr-defined]
    except Exception:
        pass
    kb._INITIALIZED_PATHS.clear()
    return home


def _config(**overrides):
    config = {
        "triage_assignee": "paul",
        "fallback_board": "inbox",
        "create_missing_boards": True,
        "board_create_policy": "explicit_project_only",
        "ack_template": "Queued for triage: board={board_slug} task={task_id}",
        "boards": {
            "hermes-agent": {"aliases": ["hermes", "hermes agent"], "default_category": "engineering"},
            "inbox": {"aliases": ["inbox", "general"], "default_category": "general"},
        },
        "categories": {
            "engineering": {"board": "hermes-agent", "assignee": "paul"},
            "general": {"board": "inbox", "assignee": "paul"},
        },
    }
    config.update(overrides)
    return config


def _source(message_id="m1"):
    return {
        "platform": "telegram",
        "chat_id": "-100123",
        "thread_id": "17585",
        "message_id": message_id,
        "user_id": "12345",
        "user_display": "Sean",
        "received_at": "2026-05-23T17:54:00Z",
    }


def _routing_event(board_slug, task_id):
    with kb.connect(board=board_slug) as conn:
        events = kb.list_events(conn, task_id)
    event = next(e for e in events if e.kind == "chat_routing")
    return event.payload


def test_reuses_existing_board_by_alias_and_creates_triage_task(fresh_home):
    kb.create_board("hermes-agent")
    kb.create_board("inbox")

    result = route_chat_triage_task(
        message_text="Add this to kanban for Hermes Agent: fix gateway retries",
        classification={
            "intent": "task_request",
            "category": "engineering",
            "project_hint": "Hermes Agent",
            "confidence": 0.91,
            "title": "Fix gateway retries",
        },
        source=_source(),
        config=_config(),
    )

    assert result["board_id"] == "hermes-agent"
    assert result["created_board"] is False
    assert result["ack"] == f"Queued for triage: board=hermes-agent task={result['task_id']}"
    with kb.connect(board="hermes-agent") as conn:
        task = kb.get_task(conn, result["task_id"])
    assert task is not None
    assert task.status == "triage"
    assert task.assignee == "paul"
    assert task.title == "Fix gateway retries"
    assert "Original message" in (task.body or "")
    payload = _routing_event("hermes-agent", result["task_id"])
    assert payload["chat_routing"]["classification"]["category"] == "engineering"
    assert payload["chat_routing"]["board_decision"]["source"] == "alias_match"


def test_missing_explicit_project_board_is_created(fresh_home):
    kb.create_board("inbox")

    result = route_chat_triage_task(
        message_text="project: New Launch Site — build landing page backlog",
        classification={
            "intent": "website_request",
            "category": "website",
            "project_hint": "New Launch Site",
            "confidence": 0.88,
            "title": "Build landing page backlog",
        },
        source=_source(),
        config=_config(categories={"website": {"board": "barista-labs-website", "assignee": "paul"}}),
    )

    assert result["board_id"] == "new-launch-site"
    assert result["created_board"] is True
    assert kb.board_exists("new-launch-site")
    with kb.connect(board="new-launch-site") as conn:
        task = kb.get_task(conn, result["task_id"])
    assert task is not None
    assert task.status == "triage"


def test_uncertain_missing_board_falls_back_without_creating_generic_board(fresh_home):
    kb.create_board("inbox")

    result = route_chat_triage_task(
        message_text="We should fix the task thing someday",
        classification={
            "intent": "task_request",
            "category": "general",
            "project_hint": "task",
            "confidence": 0.62,
        },
        source=_source(),
        config=_config(),
    )

    assert result["board_id"] == "inbox"
    assert result["fallback_used"] is True
    assert not kb.board_exists("task")
    with kb.connect(board="inbox") as conn:
        task = kb.get_task(conn, result["task_id"])
    assert task is not None
    assert task.status == "triage"


def test_duplicate_source_message_returns_original_task(fresh_home):
    kb.create_board("hermes-agent")
    kb.create_board("inbox")
    kwargs = {
        "message_text": "board:hermes-agent Add retry-safe gateway triage routing",
        "classification": {
            "intent": "task_request",
            "category": "engineering",
            "project_hint": "hermes-agent",
            "confidence": 0.95,
            "title": "Add retry-safe gateway triage routing",
        },
        "source": _source("dup1"),
        "config": _config(),
    }

    first = route_chat_triage_task(**kwargs)
    second = route_chat_triage_task(**kwargs)

    assert second["task_id"] == first["task_id"]
    with kb.connect(board="hermes-agent") as conn:
        rows = conn.execute("SELECT id FROM tasks WHERE idempotency_key IS NOT NULL").fetchall()
        routing_events = [e for e in kb.list_events(conn, first["task_id"]) if e.kind == "chat_routing"]
    assert [row["id"] for row in rows] == [first["task_id"]]
    assert len(routing_events) == 1


def test_missing_fallback_board_without_create_permission_is_actionable(fresh_home):
    with pytest.raises(ChatTriageError) as exc:
        route_chat_triage_task(
            message_text="track this",
            classification={"intent": "task_request", "category": "general", "confidence": 0.9},
            source=_source(),
            config=_config(create_missing_boards=False),
        )

    assert "fallback board 'inbox' does not exist" in str(exc.value)
    assert "create_missing_boards" in str(exc.value)

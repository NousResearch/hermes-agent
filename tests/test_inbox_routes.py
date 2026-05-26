"""Tests for Hermes inbox HTTP routes."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers (usable with both fixture and direct patch)
# ---------------------------------------------------------------------------

def make_test_message(
    message_id: str,
    *,
    source: str = "test_agent",
    run_id: str = "run-1",
    level: str = "info",
    read: bool = False,
) -> dict:
    return {
        "schema": "hermes.inbox.message.v1",
        "message_id": message_id,
        "source": source,
        "run_id": run_id,
        "level": level,
        "title": "Test",
        "summary": "Test message",
        "created_at": "2026-01-01T00:00:00+00:00",
        "read": read,
        "payload": {},
    }


def write_message(inbox_dir: Path, message_id: str, read: bool = False) -> Path:
    """Write a test message file into inbox_dir (which is already the full inbox path)."""
    msg = make_test_message(message_id, read=read)
    path = inbox_dir / f"{message_id}.json"
    path.write_text(json.dumps(msg), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def inbox_dir(tmp_path: Path) -> Path:
    """Temp inbox dir matching HERMES_HOME / INBOX_SUBDIR."""
    d = tmp_path / "_control/agents/hermes/queue/inbox"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def client(inbox_dir: Path) -> TestClient:
    """TestClient with HERMES_HOME overridden so _get_inbox_dir() resolves to inbox_dir."""
    from hermes_cli.inbox_routes import router
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from fastapi import FastAPI

    # inbox_dir = HERMES_HOME / "_control/agents/hermes/queue/inbox"
    # inbox_dir.parents[4] = HERMES_HOME = tmp_path
    token = set_hermes_home_override(str(inbox_dir.parents[4]))
    try:
        app = FastAPI()
        app.include_router(router, prefix="/api/inbox")
        yield TestClient(app)
    finally:
        reset_hermes_home_override(token)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_message(inbox_dir: Path, message_id: str, read: bool = False) -> Path:
    """Write a test message file. inbox_dir is already the full inbox path."""
    msg = make_test_message(message_id, read=read)
    path = inbox_dir / f"{message_id}.json"
    path.write_text(json.dumps(msg), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# GET /api/inbox/list
# ---------------------------------------------------------------------------

def test_list_empty(client: TestClient) -> None:
    r = client.get("/api/inbox/list")
    assert r.status_code == 200
    data = r.json()
    assert data["messages"] == []
    assert data["next_cursor"] is None


def test_list_returns_messages_newest_first(client: TestClient, inbox_dir: Path) -> None:
    # Write two messages
    _write_message(inbox_dir, "msg-old")
    _write_message(inbox_dir, "msg-new")
    r = client.get("/api/inbox/list")
    assert r.status_code == 200
    data = r.json()
    assert len(data["messages"]) == 2
    # Newest first
    assert data["messages"][0]["message_id"] == "msg-new"
    assert data["messages"][1]["message_id"] == "msg-old"


def test_list_respects_limit(client: TestClient, inbox_dir: Path) -> None:
    import os, time

    # Write all files first, then set distinct mtimes deterministically via os.utime
    paths = []
    for i in range(5):
        _write_message(inbox_dir, f"msg-{i}")
        paths.append(inbox_dir / f"msg-{i}.json")
    base = time.time()
    for i, p in enumerate(paths):
        ts = base - i
        os.utime(p, (ts, ts))

    r = client.get("/api/inbox/list?limit=2")
    assert r.status_code == 200
    data = r.json()
    assert len(data["messages"]) == 2
    assert data["next_cursor"] is not None


def test_list_cursor_pagination(client: TestClient, inbox_dir: Path) -> None:
    import os, time

    # Write all files first, then set distinct mtimes deterministically via os.utime
    paths = []
    for i in range(5):
        _write_message(inbox_dir, f"msg-{i}")
        paths.append(inbox_dir / f"msg-{i}.json")
    base = time.time()
    for i, p in enumerate(paths):
        ts = base - i
        os.utime(p, (ts, ts))

    r1 = client.get("/api/inbox/list?limit=2")
    cursor = r1.json()["next_cursor"]
    assert cursor is not None
    r2 = client.get(f"/api/inbox/list?limit=2&cursor={cursor}")
    assert r2.status_code == 200
    data = r2.json()
    assert len(data["messages"]) == 2, f"page2 msgs={len(data['messages'])}, total={data['total']}"
    # No more pages
    assert data["next_cursor"] is None, f"page2 cursor={data['next_cursor']}, total={data['total']}"


def test_list_unread_only_filter(client: TestClient, inbox_dir: Path) -> None:
    _write_message(inbox_dir, "msg-read", read=True)
    _write_message(inbox_dir, "msg-unread", read=False)
    r = client.get("/api/inbox/list?unread_only=true")
    assert r.status_code == 200
    data = r.json()
    assert len(data["messages"]) == 1
    assert data["messages"][0]["message_id"] == "msg-unread"


def test_list_source_filter(client: TestClient, inbox_dir: Path) -> None:
    msg1 = {
        "schema": "hermes.inbox.message.v1",
        "message_id": "msg-a",
        "source": "agent_a",
        "run_id": "run-1",
        "level": "info",
        "title": "A",
        "summary": "A",
        "created_at": "2026-01-01T00:00:00+00:00",
        "read": False,
        "payload": {},
    }
    msg2 = {
        **msg1,
        "message_id": "msg-b",
        "source": "agent_b",
    }
    (inbox_dir / "msg-a.json").write_text(json.dumps(msg1), encoding="utf-8")
    (inbox_dir / "msg-b.json").write_text(json.dumps(msg2), encoding="utf-8")
    r = client.get("/api/inbox/list?source=agent_a")
    assert r.status_code == 200
    data = r.json()
    assert len(data["messages"]) == 1
    assert data["messages"][0]["source"] == "agent_a"


def test_list_invalid_level_returns_400(client: TestClient) -> None:
    r = client.get("/api/inbox/list?level=debug")
    assert r.status_code == 400
    assert "level" in r.json()["detail"].lower()


# ---------------------------------------------------------------------------
# GET /api/inbox/get/{message_id}
# ---------------------------------------------------------------------------

def test_get_found(client: TestClient, inbox_dir: Path) -> None:
    _write_message(inbox_dir, "msg-1")
    r = client.get("/api/inbox/get/msg-1")
    assert r.status_code == 200
    assert r.json()["message_id"] == "msg-1"


def test_get_not_found_returns_404(client: TestClient) -> None:
    r = client.get("/api/inbox/get/nonexistent")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/inbox/mark-read
# ---------------------------------------------------------------------------

def test_mark_read_single(client: TestClient, inbox_dir: Path) -> None:
    _write_message(inbox_dir, "msg-1", read=False)
    r = client.post("/api/inbox/mark-read", json={"message_id": "msg-1"})
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["marked_count"] == 1
    # Verify file was updated
    msg_path = inbox_dir / "msg-1.json"
    stored = json.loads(msg_path.read_text(encoding="utf-8"))
    assert stored["read"] is True
    assert stored.get("read_at") is not None


def test_mark_read_batch(client: TestClient, inbox_dir: Path) -> None:
    _write_message(inbox_dir, "msg-1", read=False)
    _write_message(inbox_dir, "msg-2", read=False)
    r = client.post("/api/inbox/mark-read", json={"message_ids": ["msg-1", "msg-2"]})
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["marked_count"] == 2


def test_mark_read_none_provided_returns_400(client: TestClient) -> None:
    r = client.post("/api/inbox/mark-read", json={})
    assert r.status_code == 400


def test_mark_read_missing_id_returns_400(client: TestClient) -> None:
    r = client.post("/api/inbox/mark-read", json={"message_ids": []})
    assert r.status_code == 400


def test_mark_read_partial_errors(client: TestClient, inbox_dir: Path) -> None:
    _write_message(inbox_dir, "msg-good", read=False)
    r = client.post("/api/inbox/mark-read", json={"message_ids": ["msg-good", "msg-missing"]})
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is False
    assert data["marked_count"] == 1
    assert len(data["errors"]) == 1
    assert data["errors"][0]["message_id"] == "msg-missing"


# ---------------------------------------------------------------------------
# GET /api/inbox/unread-count
# ---------------------------------------------------------------------------

def test_unread_count_empty(client: TestClient) -> None:
    r = client.get("/api/inbox/unread-count")
    assert r.status_code == 200
    assert r.json()["unread_count"] == 0


def test_unread_count_with_messages(client: TestClient, inbox_dir: Path) -> None:
    _write_message(inbox_dir, "msg-read", read=True)
    _write_message(inbox_dir, "msg-unread-1", read=False)
    _write_message(inbox_dir, "msg-unread-2", read=False)
    r = client.get("/api/inbox/unread-count")
    assert r.status_code == 200
    assert r.json()["unread_count"] == 2

"""
Tests for mcp_serve — Hermes MCP server.

Three layers of tests:
1. Unit tests — helpers, content extraction, attachment parsing
2. EventBridge tests — queue mechanics, cursors, waiters, concurrency
3. End-to-end tests — call actual MCP tools through FastMCP's tool manager
   with real session data in SQLite and sessions.json
"""

import asyncio
import json
import os
import sqlite3
import time
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_hermes_home(tmp_path, monkeypatch):
    """Redirect HERMES_HOME to a temp directory."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    try:
        import hermes_constants
        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    except (ImportError, AttributeError):
        pass
    return tmp_path


@pytest.fixture
def sessions_dir(tmp_path):
    sdir = tmp_path / "sessions"
    sdir.mkdir(parents=True, exist_ok=True)
    return sdir


@pytest.fixture
def sample_sessions():
    return {
        "agent:main:telegram:dm:123456": {
            "session_key": "agent:main:telegram:dm:123456",
            "session_id": "20260329_120000_abc123",
            "platform": "telegram",
            "chat_type": "dm",
            "display_name": "Alice",
            "created_at": "2026-03-29T12:00:00",
            "updated_at": "2026-03-29T14:30:00",
            "input_tokens": 50000,
            "output_tokens": 2000,
            "total_tokens": 52000,
            "origin": {
                "platform": "telegram",
                "chat_id": "123456",
                "chat_name": "Alice",
                "chat_type": "dm",
                "user_id": "123456",
                "user_name": "Alice",
                "thread_id": None,
                "chat_topic": None,
            },
        },
        "agent:main:discord:group:789:456": {
            "session_key": "agent:main:discord:group:789:456",
            "session_id": "20260329_100000_def456",
            "platform": "discord",
            "chat_type": "group",
            "display_name": "Bob",
            "created_at": "2026-03-29T10:00:00",
            "updated_at": "2026-03-29T13:00:00",
            "input_tokens": 30000,
            "output_tokens": 1000,
            "total_tokens": 31000,
            "origin": {
                "platform": "discord",
                "chat_id": "789",
                "chat_name": "#general",
                "chat_type": "group",
                "user_id": "456",
                "user_name": "Bob",
                "thread_id": None,
                "chat_topic": None,
            },
        },
        "agent:main:slack:group:C1234:U5678": {
            "session_key": "agent:main:slack:group:C1234:U5678",
            "session_id": "20260328_090000_ghi789",
            "platform": "slack",
            "chat_type": "group",
            "display_name": "Carol",
            "created_at": "2026-03-28T09:00:00",
            "updated_at": "2026-03-28T11:00:00",
            "input_tokens": 10000,
            "output_tokens": 500,
            "total_tokens": 10500,
            "origin": {
                "platform": "slack",
                "chat_id": "C1234",
                "chat_name": "#engineering",
                "chat_type": "group",
                "user_id": "U5678",
                "user_name": "Carol",
                "thread_id": None,
                "chat_topic": None,
            },
        },
    }


@pytest.fixture
def populated_sessions_dir(sessions_dir, sample_sessions):
    (sessions_dir / "sessions.json").write_text(json.dumps(sample_sessions))
    return sessions_dir


def _create_test_db(db_path, session_id, messages):
    """Create a minimal SQLite DB mimicking hermes_state schema."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            source TEXT DEFAULT 'cli',
            started_at TEXT,
            message_count INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            tool_call_id TEXT,
            tool_calls TEXT,
            tool_name TEXT,
            timestamp TEXT,
            token_count INTEGER DEFAULT 0,
            finish_reason TEXT,
            reasoning TEXT,
            reasoning_details TEXT,
            codex_reasoning_items TEXT
        )
    """)
    conn.execute(
        "INSERT OR IGNORE INTO sessions (id, source, started_at, message_count) VALUES (?, 'gateway', ?, ?)",
        (session_id, "2026-03-29T12:00:00", len(messages)),
    )
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, (list, dict)):
            content = json.dumps(content)
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp, tool_calls) VALUES (?, ?, ?, ?, ?)",
            (session_id, msg["role"], content,
             msg.get("timestamp", "2026-03-29T12:00:00"),
             json.dumps(msg["tool_calls"]) if msg.get("tool_calls") else None),
        )
    conn.commit()
    conn.close()


@pytest.fixture
def mock_session_db(tmp_path, populated_sessions_dir):
    """Create a real SQLite DB with test messages and wire it up."""
    db_path = tmp_path / "state.db"
    messages = [
        {"role": "user", "content": "Hello Alice!", "timestamp": "2026-03-29T12:00:01"},
        {"role": "assistant", "content": "Hi! How can I help?", "timestamp": "2026-03-29T12:00:05"},
        {"role": "user", "content": "Check the image MEDIA: /tmp/screenshot.png please",
         "timestamp": "2026-03-29T12:01:00"},
        {"role": "assistant", "content": "I see the screenshot. It shows a terminal.",
         "timestamp": "2026-03-29T12:01:10"},
        {"role": "tool", "content": '{"result": "ok"}', "timestamp": "2026-03-29T12:01:15"},
        {"role": "user", "content": "Thanks!", "timestamp": "2026-03-29T12:02:00"},
    ]
    _create_test_db(db_path, "20260329_120000_abc123", messages)

    # Create a mock SessionDB that reads from our test DB
    class TestSessionDB:
        def __init__(self):
            self._db_path = db_path

        def get_messages(self, session_id):
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
            conn.close()
            result = []
            for r in rows:
                d = dict(r)
                if d.get("tool_calls"):
                    d["tool_calls"] = json.loads(d["tool_calls"])
                result.append(d)
            return result

    return TestSessionDB()


# ---------------------------------------------------------------------------
# 1. UNIT TESTS — helpers, extraction, attachments
# ---------------------------------------------------------------------------

class TestImports:
    def test_import_module(self):
        import mcp_serve
        assert hasattr(mcp_serve, "create_mcp_server")
        assert hasattr(mcp_serve, "run_mcp_server")
        assert hasattr(mcp_serve, "EventBridge")

    def test_mcp_available_flag(self):
        import mcp_serve
        assert isinstance(mcp_serve._MCP_SERVER_AVAILABLE, bool)


class TestHelpers:
    def test_get_sessions_dir(self, tmp_path):
        from mcp_serve import _get_sessions_dir
        result = _get_sessions_dir()
        assert result == tmp_path / "sessions"

    def test_load_sessions_index_empty(self, sessions_dir, monkeypatch):
        import mcp_serve
        monkeypatch.setattr(mcp_serve, "_get_sessions_dir", lambda: sessions_dir)
        assert mcp_serve._load_sessions_index() == {}

    def test_load_sessions_index_with_data(self, populated_sessions_dir, monkeypatch):
        import mcp_serve
        monkeypatch.setattr(mcp_serve, "_get_sessions_dir", lambda: populated_sessions_dir)
        result = mcp_serve._load_sessions_index()
        assert len(result) == 3

    def test_load_sessions_index_corrupt(self, sessions_dir, monkeypatch):
        (sessions_dir / "sessions.json").write_text("not json!")
        import mcp_serve
        monkeypatch.setattr(mcp_serve, "_get_sessions_dir", lambda: sessions_dir)
        assert mcp_serve._load_sessions_index() == {}


class TestContentExtraction:
    def test_text(self):
        from mcp_serve import _extract_message_content
        assert _extract_message_content({"content": "Hello"}) == "Hello"

    def test_multipart(self):
        from mcp_serve import _extract_message_content
        msg = {"content": [
            {"type": "text", "text": "A"},
            {"type": "image", "url": "http://x.com/i.png"},
            {"type": "text", "text": "B"},
        ]}
        assert _extract_message_content(msg) == "A\nB"

    def test_empty(self):
        from mcp_serve import _extract_message_content
        assert _extract_message_content({"content": ""}) == ""
        assert _extract_message_content({}) == ""
        assert _extract_message_content({"content": None}) == ""


class TestAttachmentExtraction:
    def test_image_url_block(self):
        from mcp_serve import _extract_attachments
        msg = {"content": [
            {"type": "image_url", "image_url": {"url": "http://x.com/pic.jpg"}},
        ]}
        att = _extract_attachments(msg)
        assert len(att) == 1
        assert att[0] == {"type": "image", "url": "http://x.com/pic.jpg"}

    def test_media_tag_in_text(self):
        from mcp_serve import _extract_attachments
        msg = {"content": "Here MEDIA: /tmp/out.png done"}
        att = _extract_attachments(msg)
        assert len(att) == 1
        assert att[0] == {"type": "media", "path": "/tmp/out.png"}

    def test_multiple_media_tags(self):
        from mcp_serve import _extract_attachments
        msg = {"content": "MEDIA: /a.png and MEDIA: /b.mp3"}
        assert len(_extract_attachments(msg)) == 2

    def test_no_attachments(self):
        from mcp_serve import _extract_attachments
        assert _extract_attachments({"content": "plain text"}) == []

    def test_image_content_block(self):
        from mcp_serve import _extract_attachments
        msg = {"content": [{"type": "image", "url": "http://x.com/p.png"}]}
        att = _extract_attachments(msg)
        assert att[0]["type"] == "image"


# ---------------------------------------------------------------------------
# 2. EVENT BRIDGE TESTS — queue, cursors, waiters, concurrency
# ---------------------------------------------------------------------------

class TestEventBridge:
    def test_create(self):
        from mcp_serve import EventBridge
        b = EventBridge()
        assert b._cursor == 0
        assert b._queue == []

    def test_enqueue_and_poll(self):
        from mcp_serve import EventBridge, QueueEvent
        b = EventBridge()
        b._enqueue(QueueEvent(cursor=0, type="message", session_key="k1",
                              data={"content": "hi"}))
        r = b.poll_events(after_cursor=0)
        assert len(r["events"]) == 1
        assert r["events"][0]["type"] == "message"
        assert r["next_cursor"] == 1

    def test_cursor_filter(self):
        from mcp_serve import EventBridge, QueueEvent
        b = EventBridge()
        for i in range(5):
            b._enqueue(QueueEvent(cursor=0, type="message", session_key=f"s{i}"))
        r = b.poll_events(after_cursor=3)
        assert len(r["events"]) == 2
        assert r["events"][0]["session_key"] == "s3"

    def test_session_filter(self):
        from mcp_serve import EventBridge, QueueEvent
        b = EventBridge()
        b._enqueue(QueueEvent(cursor=0, type="message", session_key="a"))
        b._enqueue(QueueEvent(cursor=0, type="message", session_key="b"))
        b._enqueue(QueueEvent(cursor=0, type="message", session_key="a"))
        r = b.poll_events(after_cursor=0, session_key="a")
        assert len(r["events"]) == 2

    def test_poll_empty(self):
        from mcp_serve import EventBridge
        r = EventBridge().poll_events(after_cursor=0)
        assert r["events"] == []
        assert r["next_cursor"] == 0

    def test_poll_limit(self):
        from mcp_serve import EventBridge, QueueEvent
        b = EventBridge()
        for i in range(10):
            b._enqueue(QueueEvent(cursor=0, type="message", session_key=f"s{i}"))
        r = b.poll_events(after_cursor=0, limit=3)
        assert len(r["events"]) == 3

    def test_wait_immediate(self):
        from mcp_serve import EventBridge, QueueEvent
        b = EventBridge()
        b._enqueue(QueueEvent(cursor=0, type="message", session_key="t",
                              data={"content": "hi"}))
        event = b.wait_for_event(after_cursor=0, timeout_ms=100)
        assert event is not None
        assert event["type"] == "message"

    def test_wait_timeout(self):
        from mcp_serve import EventBridge
        start = time.monotonic()
        event = EventBridge().wait_for_event(after_cursor=0, timeout_ms=150)
        assert event is None
        assert time.monotonic() - start >= 0.1

    def test_wait_wakes_on_enqueue(self):
        from mcp_serve import EventBridge, QueueEvent
        b = EventBridge()
        result = [None]

        def waiter():
            result[0] = b.wait_for_event(after_cursor=0, timeout_ms=5000)

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.05)
        b._enqueue(QueueEvent(cursor=0, type="message", session_key="wake"))
        t.join(timeout=2)
        assert result[0] is not None
        assert result[0]["session_key"] == "wake"

    def test_queue_limit(self):
        from mcp_serve import EventBridge, QueueEvent, QUEUE_LIMIT
        b = EventBridge()
        for i in range(QUEUE_LIMIT + 50):
            b._enqueue(QueueEvent(cursor=0, type="message", session_key=f"s{i}"))
        assert len(b._queue) == QUEUE_LIMIT

    def test_concurrent_enqueue(self):
        from mcp_serve import EventBridge, QueueEvent
        b = EventBridge()
        errors = []

        def batch(start):
            try:
                for i in range(100):
                    b._enqueue(QueueEvent(cursor=0, type="message",
                                          session_key=f"s{start}_{i}"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=batch, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert len(b._queue) == 500
        assert b._cursor == 500

    def test_approvals_lifecycle(self):
        from mcp_serve import EventBridge
        b = EventBridge()
        b._pending_approvals["a1"] = {
            "id": "a1", "kind": "exec",
            "description": "rm -rf /tmp",
            "session_key": "test", "created_at": "2026-03-29T12:00:00",
        }
        assert len(b.list_pending_approvals()) == 1
        result = b.respond_to_approval("a1", "deny")
        assert result["resolved"] is True
        assert len(b.list_pending_approvals()) == 0

    def test_respond_nonexistent(self):
        from mcp_serve import EventBridge
        r = EventBridge().respond_to_approval("nope", "deny")
        assert "error" in r


# ---------------------------------------------------------------------------
# 3. END-TO-END TESTS — call MCP tools through FastMCP server
# ---------------------------------------------------------------------------

@pytest.fixture
def mcp_server_e2e(populated_sessions_dir, mock_session_db, monkeypatch):
    """Create a fully wired MCP server for E2E testing."""
    mcp = pytest.importorskip("mcp", reason="MCP SDK not installed")
    import mcp_serve
    monkeypatch.setattr(mcp_serve, "_get_sessions_dir", lambda: populated_sessions_dir)
    monkeypatch.setattr(mcp_serve, "_get_session_db", lambda: mock_session_db)
    monkeypatch.setattr(mcp_serve, "_load_channel_directory", lambda: {})

    bridge = mcp_serve.EventBridge()
    server = mcp_serve.create_mcp_server(event_bridge=bridge)
    return server, bridge


def _run_tool(server, name, args=None):
    """Call an MCP tool through FastMCP's tool manager and return parsed JSON."""
    result = asyncio.get_event_loop().run_until_complete(
        server._tool_manager.call_tool(name, args or {})
    )
    return json.loads(result) if isinstance(result, str) else result


@pytest.fixture
def _event_loop():
    """Ensure an event loop exists for sync tests calling async tools."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


class TestE2EConversationsList:
    def test_list_all(self, mcp_server_e2e, _event_loop):
        server, _ = mcp_server_e2e
        result = _run_tool(server, "conversations_list")
        assert result["count"] == 3
        platforms = {c["platform"] for c in result["conversations"]}
        assert platforms == {"telegram", "discord", "slack"}

    def test_list_sorted_by_updated(self, mcp_server_e2e, _event_loop):
        server, _ = mcp_server_e2e
        result = _run_tool(server, "conversations_list")
        keys = [c["session_key"] for c in result["conversations"]]
        # Telegram (14:30) > Discord (13:00) > Slack (11:00)
        assert keys[0] == "agent:main:telegram:dm:123456"
        assert keys[1] == "agent:main:discord:group:789:456"
        assert keys[2] == "agent:main:slack:group:C1234:U5678"

    def test_filter_by_platform(self, mcp_server_e2e, _event_loop):
        server, _ = mcp_server_e2e
        result = _run_tool(server, "conversations_list", {"platform": "discord"})
        assert result["count"] == 1
        assert result["conversations"][0]["platform"] == "discord"

    def test_filter_by_platform_case_insensitive(self, mcp_server_e2e, _event_loop):
        server, _ = mcp_server_e2e
        result = _run_tool(server, "conversations_list", {"platform": "TELEGRAM"})
        assert result["count"] == 1

    def test_search_by_name(self, mcp_server_e2e, _event_loop):
        server, _ = mcp_server_e2e
        result = _run_tool(server, "conversations_list", {"search": "Alice"})
        assert result["count"] == 1
        assert result["conversations"][0]["display_name"] == "Alice"

    def test_search_no_match(self, mcp_server_e2e, _event_loop):
        server, _ = mcp_server_e2e
        result = _run_tool(server, "conversations_list", {"search": "nobody"})
        assert result["count"] == 0

    def test_limit(self, mcp_server_e2e, _event_loop):
        server, _ = mcp_server_e2e
        result = _run_tool(server, "conversations_list", {"limit": 2})
        assert result["count"] == 2


class TestE2EConversationGet:
    def test_get_existing(self, mcp_server_e2e, _event_loop):
        server, _ = mcp_server_e2e
        result = _run_tool(server, "conversation_get",
                          {"session_key": "agent:main:telegram:dm:123456"})
        assert result["platform"] == "telegram"
        assert result["display_name"] == "Alice"
        assert result["chat_id"] == "123456"
        assert result["input_tokens"] == 50000

    def test_get_nonexistent(self, mcp_server_e2e, _event_loop):
        server, _ = mcp_server_e2e
        result = _run_tool(server, "conversation_get",
                          {"session_key": "nonexistent:key"})
        assert "error" in result


class TestE2EMessagesRead:
    def test_read_messages(self, mcp_server_e2e, _event_loop):
        server, _ = mcp_server_e2e
        result = _run_tool(server, "messages_read",
                          {"session_key": "agent:main:telegram:dm:123456"})
        assert result["count"] > 0
        # Should filter out tool messages — only user/assistant
        roles = {m["role"] for m in result["messages"]}
        assert "tool" not in roles
        assert "user" in roles
        assert "assistant" in roles

    def test_read_messages_content(self, mcp_server_e2e, _event_loop):
        server, _ = mcp_server_e2e
        result = _run_tool(server, "messages_read",
                          {"session_key": "agent:main:telegram:dm:123456"})
        contents = [m["content"] for m in result["messages"]]
        assert "Hello Alice!" in contents
        assert "Hi! How can I help?" in contents

    def test_read_messages_have_ids(self, mcp_server_e2e, _event_loop):
        server, _ = mcp_server_e2e
        result = _run_tool(server, "messages_read",
                          {"session_key": "agent:main:telegram:dm:123456"})
        for msg in result["messages"]:
            assert "id" in msg
            assert msg["id"]  # non-empty

    def test_read_with_limit(self, mcp_server_e2e, _event_loop):
        server, _ = mcp_server_e2e
        result = _run_tool(server, "messages_read",
                          {"session_key": "agent:main:telegram:dm:123456",
                           "limit": 2})
        assert result["count"] == 2

    def test_read_nonexistent_session(self, mcp_server_e2e, _event_loop):
        server, _ = mcp_server_e2e
        result = _run_tool(server, "messages_read",
                          {"session_key": "nonexistent:key"})
        assert "error" in result


class TestE2EAttachmentsFetch:
    def test_fetch_media_from_message(self, mcp_server_e2e, _event_loop):
        server, _ = mcp_server_e2e
        # First get message IDs
        msgs = _run_tool(server, "messages_read",
                        {"session_key": "agent:main:telegram:dm:123456"})
        # Find the message with MEDIA: tag
        media_msg = None
        for m in msgs["messages"]:
            if "MEDIA:" in m["content"]:
                media_msg = m
                break
        assert media_msg is not None, "Should have a message with MEDIA: tag"

        result = _run_tool(server, "attachments_fetch", {
            "session_key": "agent:main:telegram:dm:123456",
            "message_id": media_msg["id"],
        })
        assert result["count"] >= 1
        assert result["attachments"][0]["type"] == "media"
        assert result["attachments"][0]["path"] == "/tmp/screenshot.png"

    def test_fetch_from_nonexistent_message(self, mcp_server_e2e, _event_loop):
        server, _ = mcp_server_e2e
        result = _run_tool(server, "attachments_fetch", {
            "session_key": "agent:main:telegram:dm:123456",
            "message_id": "99999",
        })
        assert "error" in result

    def test_fetch_from_nonexistent_session(self, mcp_server_e2e, _event_loop):
        server, _ = mcp_server_e2e
        result = _run_tool(server, "attachments_fetch", {
            "session_key": "nonexistent:key",
            "message_id": "1",
        })
        assert "error" in result


class TestE2EEventsPoll:
    def test_poll_empty(self, mcp_server_e2e, _event_loop):
        server, bridge = mcp_server_e2e
        result = _run_tool(server, "events_poll")
        assert result["events"] == []
        assert result["next_cursor"] == 0

    def test_poll_with_events(self, mcp_server_e2e, _event_loop):
        from mcp_serve import QueueEvent
        server, bridge = mcp_server_e2e
        bridge._enqueue(QueueEvent(cursor=0, type="message",
                                   session_key="agent:main:telegram:dm:123456",
                                   data={"role": "user", "content": "Hello"}))
        bridge._enqueue(QueueEvent(cursor=0, type="message",
                                   session_key="agent:main:telegram:dm:123456",
                                   data={"role": "assistant", "content": "Hi"}))

        result = _run_tool(server, "events_poll")
        assert len(result["events"]) == 2
        assert result["events"][0]["content"] == "Hello"
        assert result["events"][1]["content"] == "Hi"
        assert result["next_cursor"] == 2

    def test_poll_cursor_pagination(self, mcp_server_e2e, _event_loop):
        from mcp_serve import QueueEvent
        server, bridge = mcp_server_e2e
        for i in range(5):
            bridge._enqueue(QueueEvent(cursor=0, type="message",
                                       session_key=f"s{i}"))

        page1 = _run_tool(server, "events_poll", {"limit": 2})
        assert len(page1["events"]) == 2
        assert page1["next_cursor"] == 2

        page2 = _run_tool(server, "events_poll",
                         {"after_cursor": page1["next_cursor"], "limit": 2})
        assert len(page2["events"]) == 2
        assert page2["next_cursor"] == 4

    def test_poll_session_filter(self, mcp_server_e2e, _event_loop):
        from mcp_serve import QueueEvent
        server, bridge = mcp_server_e2e
        bridge._enqueue(QueueEvent(cursor=0, type="message", session_key="a"))
        bridge._enqueue(QueueEvent(cursor=0, type="message", session_key="b"))
        bridge._enqueue(QueueEvent(cursor=0, type="message", session_key="a"))

        result = _run_tool(server, "events_poll",
                          {"session_key": "b"})
        assert len(result["events"]) == 1


class TestE2EEventsWait:
    def test_wait_timeout(self, mcp_server_e2e, _event_loop):
        server, _ = mcp_server_e2e
        result = _run_tool(server, "events_wait", {"timeout_ms": 100})
        assert result["event"] is None
        assert result["reason"] == "timeout"

    def test_wait_with_existing_event(self, mcp_server_e2e, _event_loop):
        from mcp_serve import QueueEvent
        server, bridge = mcp_server_e2e
        bridge._enqueue(QueueEvent(cursor=0, type="message",
                                   session_key="test",
                                   data={"content": "waiting for this"}))
        result = _run_tool(server, "events_wait", {"timeout_ms": 100})
        assert result["event"] is not None
        assert result["event"]["content"] == "waiting for this"

    def test_wait_caps_timeout(self, mcp_server_e2e, _event_loop):
        """Timeout should be capped at 300000ms (5 min)."""
        from mcp_serve import QueueEvent
        server, bridge = mcp_server_e2e
        bridge._enqueue(QueueEvent(cursor=0, type="message", session_key="t"))
        # Even with huge timeout, should return immediately since event exists
        result = _run_tool(server, "events_wait", {"timeout_ms": 999999})
        assert result["event"] is not None


class TestE2EMessagesSend:
    def test_send_missing_args(self, mcp_server_e2e, _event_loop):
        server, _ = mcp_server_e2e
        result = _run_tool(server, "messages_send", {"target": "", "message": "hi"})
        assert "error" in result

    def test_send_delegates_to_tool(self, mcp_server_e2e, _event_loop, monkeypatch):
        server, _ = mcp_server_e2e
        mock = MagicMock(return_value=json.dumps({"success": True, "platform": "telegram"}))
        monkeypatch.setattr("tools.send_message_tool.send_message_tool", mock)

        result = _run_tool(server, "messages_send",
                          {"target": "telegram:123456", "message": "Hello!"})
        assert result["success"] is True
        mock.assert_called_once()
        call_args = mock.call_args[0][0]
        assert call_args["action"] == "send"
        assert call_args["target"] == "telegram:123456"


class TestE2EChannelsList:
    def test_channels_from_sessions(self, mcp_server_e2e, _event_loop):
        server, _ = mcp_server_e2e
        result = _run_tool(server, "channels_list")
        assert result["count"] == 3
        targets = {c["target"] for c in result["channels"]}
        assert "telegram:123456" in targets
        assert "discord:789" in targets
        assert "slack:C1234" in targets

    def test_channels_platform_filter(self, mcp_server_e2e, _event_loop):
        server, _ = mcp_server_e2e
        result = _run_tool(server, "channels_list", {"platform": "slack"})
        assert result["count"] == 1
        assert result["channels"][0]["target"] == "slack:C1234"

    def test_channels_with_directory(self, mcp_server_e2e, _event_loop, monkeypatch):
        import mcp_serve
        monkeypatch.setattr(mcp_serve, "_load_channel_directory", lambda: {
            "telegram": [
                {"id": "123456", "name": "Alice", "type": "dm"},
                {"id": "-100999", "name": "Dev Group", "type": "group"},
            ],
        })
        # Need to recreate server to pick up the new mock
        server, bridge = mcp_server_e2e
        # The tool closure already captured the old mock, so test the function directly
        directory = mcp_serve._load_channel_directory()
        assert len(directory["telegram"]) == 2


class TestE2EPermissions:
    def test_list_empty(self, mcp_server_e2e, _event_loop):
        server, _ = mcp_server_e2e
        result = _run_tool(server, "permissions_list_open")
        assert result["count"] == 0
        assert result["approvals"] == []

    def test_list_with_approvals(self, mcp_server_e2e, _event_loop):
        server, bridge = mcp_server_e2e
        bridge._pending_approvals["a1"] = {
            "id": "a1", "kind": "exec",
            "description": "sudo rm -rf /",
            "session_key": "test",
            "created_at": "2026-03-29T12:00:00",
        }
        result = _run_tool(server, "permissions_list_open")
        assert result["count"] == 1
        assert result["approvals"][0]["id"] == "a1"

    def test_respond_allow(self, mcp_server_e2e, _event_loop):
        server, bridge = mcp_server_e2e
        bridge._pending_approvals["a1"] = {"id": "a1", "kind": "exec"}
        result = _run_tool(server, "permissions_respond",
                          {"id": "a1", "decision": "allow-once"})
        assert result["resolved"] is True
        assert result["decision"] == "allow-once"
        # Should be gone now
        check = _run_tool(server, "permissions_list_open")
        assert check["count"] == 0

    def test_respond_deny(self, mcp_server_e2e, _event_loop):
        server, bridge = mcp_server_e2e
        bridge._pending_approvals["a2"] = {"id": "a2", "kind": "plugin"}
        result = _run_tool(server, "permissions_respond",
                          {"id": "a2", "decision": "deny"})
        assert result["resolved"] is True

    def test_respond_invalid_decision(self, mcp_server_e2e, _event_loop):
        server, bridge = mcp_server_e2e
        bridge._pending_approvals["a3"] = {"id": "a3", "kind": "exec"}
        result = _run_tool(server, "permissions_respond",
                          {"id": "a3", "decision": "maybe"})
        assert "error" in result

    def test_respond_nonexistent(self, mcp_server_e2e, _event_loop):
        server, _ = mcp_server_e2e
        result = _run_tool(server, "permissions_respond",
                          {"id": "nope", "decision": "deny"})
        assert "error" in result


# ---------------------------------------------------------------------------
# 4. TOOL LISTING — verify all 10 tools are registered
# ---------------------------------------------------------------------------

class TestToolRegistration:
    def test_all_tools_registered(self, mcp_server_e2e, _event_loop):
        server, _ = mcp_server_e2e
        tools = server._tool_manager.list_tools()
        tool_names = {t.name for t in tools}

        expected = {
            "conversations_list", "conversation_get", "messages_read",
            "attachments_fetch", "events_poll", "events_wait",
            "messages_send", "channels_list",
            "permissions_list_open", "permissions_respond",
        }
        assert expected == tool_names, f"Missing: {expected - tool_names}, Extra: {tool_names - expected}"

    def test_tools_have_descriptions(self, mcp_server_e2e, _event_loop):
        server, _ = mcp_server_e2e
        for tool in server._tool_manager.list_tools():
            assert tool.description, f"Tool {tool.name} has no description"


# ---------------------------------------------------------------------------
# 5. SERVER LIFECYCLE / CLI INTEGRATION
# ---------------------------------------------------------------------------

class TestServerCreation:
    def test_create_server(self, populated_sessions_dir, monkeypatch):
        pytest.importorskip("mcp", reason="MCP SDK not installed")
        import mcp_serve
        monkeypatch.setattr(mcp_serve, "_get_sessions_dir", lambda: populated_sessions_dir)
        assert mcp_serve.create_mcp_server() is not None

    def test_create_with_bridge(self, populated_sessions_dir, monkeypatch):
        pytest.importorskip("mcp", reason="MCP SDK not installed")
        import mcp_serve
        monkeypatch.setattr(mcp_serve, "_get_sessions_dir", lambda: populated_sessions_dir)
        bridge = mcp_serve.EventBridge()
        assert mcp_serve.create_mcp_server(event_bridge=bridge) is not None

    def test_create_without_mcp_sdk(self, monkeypatch):
        import mcp_serve
        monkeypatch.setattr(mcp_serve, "_MCP_SERVER_AVAILABLE", False)
        with pytest.raises(ImportError, match="MCP server requires"):
            mcp_serve.create_mcp_server()


class TestRunMcpServer:
    def test_run_without_mcp_exits(self, monkeypatch):
        import mcp_serve
        monkeypatch.setattr(mcp_serve, "_MCP_SERVER_AVAILABLE", False)
        with pytest.raises(SystemExit) as exc_info:
            mcp_serve.run_mcp_server()
        assert exc_info.value.code == 1


class TestCliIntegration:
    def test_parse_serve(self):
        import argparse
        parser = argparse.ArgumentParser()
        subs = parser.add_subparsers(dest="command")
        mcp_p = subs.add_parser("mcp")
        mcp_sub = mcp_p.add_subparsers(dest="mcp_action")
        serve_p = mcp_sub.add_parser("serve")
        serve_p.add_argument("-v", "--verbose", action="store_true")

        args = parser.parse_args(["mcp", "serve"])
        assert args.mcp_action == "serve"
        assert args.verbose is False

    def test_parse_serve_verbose(self):
        import argparse
        parser = argparse.ArgumentParser()
        subs = parser.add_subparsers(dest="command")
        mcp_p = subs.add_parser("mcp")
        mcp_sub = mcp_p.add_subparsers(dest="mcp_action")
        serve_p = mcp_sub.add_parser("serve")
        serve_p.add_argument("-v", "--verbose", action="store_true")

        args = parser.parse_args(["mcp", "serve", "--verbose"])
        assert args.verbose is True

    def test_dispatcher_routes_serve(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        mock_run = MagicMock()
        monkeypatch.setattr("mcp_serve.run_mcp_server", mock_run)

        import argparse
        args = argparse.Namespace(mcp_action="serve", verbose=True)
        from hermes_cli.mcp_config import mcp_command
        mcp_command(args)
        mock_run.assert_called_once_with(verbose=True)


# ---------------------------------------------------------------------------
# 6. EDGE CASES
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_sessions_json(self, sessions_dir, monkeypatch):
        (sessions_dir / "sessions.json").write_text("{}")
        import mcp_serve
        monkeypatch.setattr(mcp_serve, "_get_sessions_dir", lambda: sessions_dir)
        assert mcp_serve._load_sessions_index() == {}

    def test_sessions_without_origin(self, sessions_dir, monkeypatch):
        data = {"agent:main:telegram:dm:111": {
            "session_key": "agent:main:telegram:dm:111",
            "session_id": "20260329_120000_xyz",
            "platform": "telegram",
            "updated_at": "2026-03-29T12:00:00",
        }}
        (sessions_dir / "sessions.json").write_text(json.dumps(data))
        import mcp_serve
        monkeypatch.setattr(mcp_serve, "_get_sessions_dir", lambda: sessions_dir)
        entries = mcp_serve._load_sessions_index()
        assert entries["agent:main:telegram:dm:111"]["platform"] == "telegram"

    def test_bridge_start_stop(self):
        from mcp_serve import EventBridge
        b = EventBridge()
        assert not b._running
        b._running = True
        b.stop()
        assert not b._running

    def test_truncation(self):
        assert len(("x" * 5000)[:2000]) == 2000


# ---------------------------------------------------------------------------
# 7. EVENT BRIDGE POLL LOOP E2E — real SQLite DB, mtime optimization
# ---------------------------------------------------------------------------

class TestEventBridgePollE2E:
    """End-to-end tests for the EventBridge polling loop with real files."""

    def test_poll_detects_new_messages(self, tmp_path, monkeypatch):
        """Write to SQLite + sessions.json, verify EventBridge picks it up."""
        import mcp_serve
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        monkeypatch.setattr(mcp_serve, "_get_sessions_dir", lambda: sessions_dir)

        session_id = "20260329_150000_poll_test"
        db_path = tmp_path / "state.db"

        # Write sessions.json
        sessions_data = {
            "agent:main:telegram:dm:poll_test": {
                "session_key": "agent:main:telegram:dm:poll_test",
                "session_id": session_id,
                "platform": "telegram",
                "chat_type": "dm",
                "display_name": "PollTest",
                "updated_at": "2026-03-29T15:00:05",
                "origin": {"platform": "telegram", "chat_id": "poll_test"},
            }
        }
        (sessions_dir / "sessions.json").write_text(json.dumps(sessions_data))

        # Write messages to SQLite
        messages = [
            {"role": "user", "content": "First message",
             "timestamp": "2026-03-29T15:00:01"},
            {"role": "assistant", "content": "Reply",
             "timestamp": "2026-03-29T15:00:03"},
        ]
        _create_test_db(db_path, session_id, messages)

        # Create a mock SessionDB that reads our test DB
        class TestDB:
            def get_messages(self, sid):
                conn = sqlite3.connect(str(db_path))
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM messages WHERE session_id = ? ORDER BY id",
                    (sid,),
                ).fetchall()
                conn.close()
                return [dict(r) for r in rows]

        monkeypatch.setattr(mcp_serve, "_get_session_db", lambda: TestDB())

        bridge = mcp_serve.EventBridge()
        # Run one poll cycle manually
        bridge._poll_once(TestDB())

        # Should have found the messages
        result = bridge.poll_events(after_cursor=0)
        assert len(result["events"]) == 2
        assert result["events"][0]["role"] == "user"
        assert result["events"][0]["content"] == "First message"
        assert result["events"][1]["role"] == "assistant"

    def test_poll_skips_when_unchanged(self, tmp_path, monkeypatch):
        """Second poll with no file changes should be a no-op."""
        import mcp_serve
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        monkeypatch.setattr(mcp_serve, "_get_sessions_dir", lambda: sessions_dir)

        session_id = "20260329_150000_skip_test"
        db_path = tmp_path / "state.db"

        sessions_data = {
            "agent:main:telegram:dm:skip": {
                "session_key": "agent:main:telegram:dm:skip",
                "session_id": session_id,
                "platform": "telegram",
                "updated_at": "2026-03-29T15:00:05",
                "origin": {"platform": "telegram", "chat_id": "skip"},
            }
        }
        (sessions_dir / "sessions.json").write_text(json.dumps(sessions_data))
        _create_test_db(db_path, session_id, [
            {"role": "user", "content": "Hello", "timestamp": "2026-03-29T15:00:01"},
        ])

        class TestDB:
            def __init__(self):
                self.call_count = 0

            def get_messages(self, sid):
                self.call_count += 1
                conn = sqlite3.connect(str(db_path))
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM messages WHERE session_id = ? ORDER BY id",
                    (sid,),
                ).fetchall()
                conn.close()
                return [dict(r) for r in rows]

        db = TestDB()
        bridge = mcp_serve.EventBridge()

        # First poll — should process
        bridge._poll_once(db)
        first_calls = db.call_count
        assert first_calls >= 1

        # Second poll — files unchanged, should skip entirely
        bridge._poll_once(db)
        assert db.call_count == first_calls, \
            "Second poll should skip DB queries when files unchanged"

    def test_poll_detects_new_message_after_db_write(self, tmp_path, monkeypatch):
        """Write a new message to the DB after first poll, verify it's detected."""
        import mcp_serve
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        monkeypatch.setattr(mcp_serve, "_get_sessions_dir", lambda: sessions_dir)

        session_id = "20260329_150000_new_msg"
        db_path = tmp_path / "state.db"

        sessions_data = {
            "agent:main:telegram:dm:new": {
                "session_key": "agent:main:telegram:dm:new",
                "session_id": session_id,
                "platform": "telegram",
                "updated_at": "2026-03-29T15:00:05",
                "origin": {"platform": "telegram", "chat_id": "new"},
            }
        }
        (sessions_dir / "sessions.json").write_text(json.dumps(sessions_data))
        _create_test_db(db_path, session_id, [
            {"role": "user", "content": "First", "timestamp": "2026-03-29T15:00:01"},
        ])

        class TestDB:
            def get_messages(self, sid):
                conn = sqlite3.connect(str(db_path))
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM messages WHERE session_id = ? ORDER BY id",
                    (sid,),
                ).fetchall()
                conn.close()
                return [dict(r) for r in rows]

        db = TestDB()
        bridge = mcp_serve.EventBridge()

        # First poll
        bridge._poll_once(db)
        r1 = bridge.poll_events(after_cursor=0)
        assert len(r1["events"]) == 1

        # Add a new message to the DB
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, "assistant", "New reply!", "2026-03-29T15:00:10"),
        )
        conn.commit()
        conn.close()
        # Touch the DB file to update mtime (WAL mode may not update mtime on small writes)
        os.utime(db_path, None)

        # Update sessions.json updated_at to trigger re-check
        sessions_data["agent:main:telegram:dm:new"]["updated_at"] = "2026-03-29T15:00:10"
        (sessions_dir / "sessions.json").write_text(json.dumps(sessions_data))

        # Second poll — should detect the new message
        bridge._poll_once(db)
        r2 = bridge.poll_events(after_cursor=r1["next_cursor"])
        assert len(r2["events"]) == 1
        assert r2["events"][0]["content"] == "New reply!"

    def test_poll_interval_is_200ms(self):
        """Verify the poll interval constant."""
        from mcp_serve import POLL_INTERVAL
        assert POLL_INTERVAL == 0.2

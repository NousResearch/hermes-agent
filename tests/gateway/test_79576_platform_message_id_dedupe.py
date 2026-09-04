"""Regression tests for issue #79576 — Telegram session retry loop.

Three failure modes combine to bloat a session into infinite context loss:
transient provider failure persists the user message, the same inbound
event gets re-persisted (or raced by a sibling writer), and the session
grows without bound until the agent "forgets" and the user must retry.

The fix enforces a DB-level invariant — one platform_message_id per
session — atomically inside the write transaction:

1. ``SessionDB.append_message`` skips the insert (and the counter bump)
   when a row with the same ``(session_id, platform_message_id)`` already
   exists, returning the existing row id.
2. ``SessionDB.append_messages_batch`` drops rows whose
   platform_message_id already exists (or repeats within the batch) before
   inserting.
3. The gateway's crash-resilience persist (exception handler) dedupes by
   the inbound ``event.message_id`` first, falling back to the content
   check only for events without an id, and only treating a content match
   as a duplicate when the matching row carries no platform id (an
   agent-side early-persist row). A genuinely NEW message with identical
   text is still persisted.
"""

import sys
import types
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource
from hermes_state import SessionDB


def _make_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s1", "cli")
    return db


# ── DB-level: append_message ─────────────────────────────────────────


class TestAppendMessageDedupe:
    def test_same_platform_message_id_deduped(self, tmp_path):
        db = _make_db(tmp_path)
        first_id = db.append_message(
            session_id="s1",
            role="user",
            content="hello",
            platform_message_id="msg-123",
        )
        second_id = db.append_message(
            session_id="s1",
            role="user",
            content="hello",
            platform_message_id="msg-123",
        )
        assert second_id == first_id
        assert db.message_count("s1") == 1
        assert db.has_platform_message_id("s1", "msg-123")

    def test_same_platform_message_id_counter_bumped_once(self, tmp_path):
        db = _make_db(tmp_path)
        db.append_message(
            session_id="s1",
            role="user",
            content="hello",
            platform_message_id="msg-123",
        )
        db.append_message(
            session_id="s1",
            role="user",
            content="hello",
            platform_message_id="msg-123",
        )
        session = db.get_session("s1")
        assert session is not None
        assert session["message_count"] == 1

    def test_same_id_different_sessions_both_inserted(self, tmp_path):
        db = _make_db(tmp_path)
        db.create_session("s2", "cli")
        db.append_message(
            session_id="s1",
            role="user",
            content="hello",
            platform_message_id="msg-123",
        )
        db.append_message(
            session_id="s2",
            role="user",
            content="hello",
            platform_message_id="msg-123",
        )
        assert db.message_count("s1") == 1
        assert db.message_count("s2") == 1

    def test_null_platform_message_id_never_deduped(self, tmp_path):
        db = _make_db(tmp_path)
        db.append_message(session_id="s1", role="user", content="hello")
        db.append_message(session_id="s1", role="user", content="hello")
        assert db.message_count("s1") == 2

    def test_dedupe_returns_existing_row_and_keeps_first_content(self, tmp_path):
        db = _make_db(tmp_path)
        db.append_message(
            session_id="s1",
            role="user",
            content="first version",
            platform_message_id="msg-1",
        )
        db.append_message(
            session_id="s1",
            role="user",
            content="retry version",
            platform_message_id="msg-1",
        )
        rows = db.get_messages("s1")
        contents = [r["content"] for r in rows]
        assert contents == ["first version"]


# ── DB-level: append_messages_batch ──────────────────────────────────


class TestAppendMessagesBatchDedupe:
    def _rows(self, *ids_and_contents):
        return [
            {"role": "user", "content": content, "platform_message_id": pid}
            for pid, content in ids_and_contents
        ]

    def test_batch_dedupes_existing_and_within_batch(self, tmp_path):
        db = _make_db(tmp_path)
        db.append_message(
            session_id="s1",
            role="user",
            content="already there",
            platform_message_id="msg-a",
        )
        inserted = db.append_messages_batch(
            session_id="s1",
            messages=self._rows(
                ("msg-a", "already there (dup of existing)"),
                ("msg-b", "new one"),
                ("msg-b", "dup within batch"),
                ("msg-c", "another new one"),
            ),
        )
        # Only msg-b and msg-c are new.
        assert inserted == 2
        assert db.message_count("s1") == 3
        assert db.has_platform_message_id("s1", "msg-b")
        assert db.has_platform_message_id("s1", "msg-c")

    def test_batch_all_duplicates_returns_zero(self, tmp_path):
        db = _make_db(tmp_path)
        db.append_message(
            session_id="s1",
            role="user",
            content="hello",
            platform_message_id="msg-a",
        )
        inserted = db.append_messages_batch(
            session_id="s1",
            messages=self._rows(("msg-a", "hello again")),
        )
        assert inserted == 0
        assert db.message_count("s1") == 1

    def test_batch_without_ids_inserts_all(self, tmp_path):
        db = _make_db(tmp_path)
        inserted = db.append_messages_batch(
            session_id="s1",
            messages=[
                {"role": "user", "content": "one"},
                {"role": "user", "content": "two"},
            ],
        )
        assert inserted == 2
        assert db.message_count("s1") == 2

    def test_batch_accepts_message_id_alias(self, tmp_path):
        db = _make_db(tmp_path)
        inserted = db.append_messages_batch(
            session_id="s1",
            messages=[
                {"role": "user", "content": "x", "message_id": "alias-1"},
                {"role": "user", "content": "y", "message_id": "alias-1"},
            ],
        )
        assert inserted == 1
        assert db.message_count("s1") == 1
        assert db.has_platform_message_id("s1", "alias-1")


# ── Gateway-level: crash-resilience exception handler ────────────────


def _bootstrap(monkeypatch, tmp_path, real_db):
    """Minimal GatewayRunner setup backed by a real SessionDB for dedupe."""
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    config = GatewayConfig()
    runner = gateway_run.GatewayRunner(config)
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._handle_active_session_busy_message = AsyncMock(return_value=False)
    runner._session_db = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = lambda _key, _source: None
    runner._is_session_run_current = lambda _key, _gen: True
    runner._begin_session_run_generation = lambda _key: 1
    runner._reply_anchor_for_event = lambda _event: None
    runner._get_guild_id = lambda _event: None
    runner._should_send_voice_reply = lambda *_a, **_kw: False
    # Disable the turn-lease registry (#64934): leases are released in the
    # OUTER _handle_message finally, which these tests bypass by calling
    # _handle_message_with_agent directly. With the registry disabled the
    # per-session serialization is skipped entirely (the fail-open path).
    runner._turn_leases = None
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()

    real_db.create_session("sess-dedup", "telegram")

    def _append(session_id, message, skip_db=False):
        if skip_db:
            return
        real_db.append_message(
            session_id=session_id,
            role=message.get("role", "user"),
            content=message.get("content"),
            timestamp=message.get("timestamp"),
            platform_message_id=(
                message.get("platform_message_id") or message.get("message_id")
            ),
        )

    def _load(session_id):
        try:
            return real_db.get_messages_as_conversation(
                session_id, repair_alternation=True
            ) or []
        except Exception:
            return []

    store = MagicMock()
    store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:group:-1001:12345",
        session_id="sess-dedup",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="group",
    )
    store.has_platform_message_id = MagicMock(
        side_effect=lambda sid, pid: real_db.has_platform_message_id(sid, pid)
    )
    store.load_transcript = MagicMock(side_effect=_load)
    store.append_to_transcript = MagicMock(side_effect=_append)
    store.update_session = MagicMock()
    runner.session_store = store

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100_000,
    )
    return runner


def _event(message_id="msg-42", text="hello world"):
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1001",
            chat_type="group",
            user_id="12345",
        ),
        message_id=message_id,
    )


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        user_id="12345",
    )


def _failing_run_agent():
    async def _fail(*args, **kwargs):
        raise RuntimeError("provider init failed before run_conversation")

    return _fail


@pytest.mark.asyncio
async def test_exception_handler_same_event_persisted_once(monkeypatch, tmp_path):
    """Re-processing the SAME inbound event must not stack user turns."""
    db = SessionDB(tmp_path / "state.db")
    runner = _bootstrap(monkeypatch, tmp_path, db)
    runner._run_agent = _failing_run_agent()

    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )
    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    assert db.message_count("sess-dedup") == 1
    assert db.has_platform_message_id("sess-dedup", "msg-42")


@pytest.mark.asyncio
async def test_exception_handler_same_text_new_event_persisted(monkeypatch, tmp_path):
    """A NEW message with the same text is a new turn and must persist."""
    db = SessionDB(tmp_path / "state.db")
    runner = _bootstrap(monkeypatch, tmp_path, db)
    runner._run_agent = _failing_run_agent()

    await runner._handle_message_with_agent(
        _event(message_id="msg-42", text="retry this:"),
        _source(), "agent:main:telegram:group:-1001:12345", 1,
    )
    # Same text, NEW platform message id → distinct turn.
    await runner._handle_message_with_agent(
        _event(message_id="msg-43", text="retry this:"),
        _source(), "agent:main:telegram:group:-1001:12345", 1,
    )

    assert db.message_count("sess-dedup") == 2
    assert db.has_platform_message_id("sess-dedup", "msg-42")
    assert db.has_platform_message_id("sess-dedup", "msg-43")


@pytest.mark.asyncio
async def test_exception_handler_content_fallback_dedupes_agent_persisted_row(
    monkeypatch, tmp_path,
):
    """A content-matching row WITHOUT a platform id (agent early persist)
    is still treated as the same event and not duplicated."""
    db = SessionDB(tmp_path / "state.db")
    runner = _bootstrap(monkeypatch, tmp_path, db)
    runner._run_agent = _failing_run_agent()

    # Simulate the agent's early turn-start persistence: same text, no id.
    db.append_message(
        session_id="sess-dedup",
        role="user",
        content="hello world",
    )
    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    assert db.message_count("sess-dedup") == 1

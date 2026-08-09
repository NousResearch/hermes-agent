"""Regression tests for /retry replacement semantics."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionStore


@pytest.mark.asyncio
async def test_gateway_retry_replaces_last_user_turn_in_transcript(tmp_path, monkeypatch):
    # Pin DEFAULT_DB_PATH so SessionDB() doesn't write to the real ~/.hermes/state.db.
    # (Module-level constant snapshot, see test_load_transcript_db_only.)
    import hermes_state
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")

    config = GatewayConfig()
    store = SessionStore(sessions_dir=tmp_path, config=config)

    session_id = "retry_session"
    store._db.create_session(session_id=session_id, source="test")
    for msg in [
        {"role": "session_meta", "tools": []},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "retry me"},
        {"role": "assistant", "content": "old answer"},
    ]:
        store.append_to_transcript(session_id, msg)

    gw = GatewayRunner.__new__(GatewayRunner)
    gw.config = config
    gw.session_store = store

    session_entry = MagicMock(session_id=session_id)
    session_entry.last_prompt_tokens = 111
    gw.session_store.get_or_create_session = MagicMock(return_value=session_entry)

    async def fake_handle_message(event):
        assert event.text == "retry me"
        transcript_before = store.load_transcript(session_id)
        assert [m.get("content") for m in transcript_before if m.get("role") == "user"] == [
            "first question"
        ]
        store.append_to_transcript(session_id, {"role": "user", "content": event.text})
        store.append_to_transcript(session_id, {"role": "assistant", "content": "new answer"})
        return "new answer"

    gw._handle_message = AsyncMock(side_effect=fake_handle_message)

    result = await gw._handle_retry_command(
        MessageEvent(text="/retry", message_type=MessageType.TEXT, source=MagicMock())
    )

    assert result == "new answer"
    transcript_after = store.load_transcript(session_id)
    assert [m.get("content") for m in transcript_after if m.get("role") == "user"] == [
        "first question",
        "retry me",
    ]
    assert [m.get("content") for m in transcript_after if m.get("role") == "assistant"] == [
        "first answer",
        "new answer",
    ]


@pytest.mark.asyncio
async def test_gateway_retry_preserves_archived_compaction_rows_when_probe_fails(
    tmp_path, monkeypatch
):
    """/retry must not DELETE archives when an existence probe would fail.

    With compression.in_place (the default, #38763) archive_and_compact()
    keeps the pre-compaction transcript on disk as active=0/compacted=1 rows
    under the same session id. /retry used to persist its truncation via a
    bare rewrite_transcript(), whose replace_messages(active_only=False)
    DELETEs every row for the session and reinserts only the truncated live
    tail, wiping the archived history permanently (same class as #61145;
    #57803 named this call site as a residual gap). /retry never intends to
    purge archived history, so it must pass active_only=True unconditionally:
    a separate existence probe can fail open or race with the rewrite.
    """
    import hermes_state
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")

    config = GatewayConfig()
    store = SessionStore(sessions_dir=tmp_path, config=config)

    session_id = "retry_archived_session"
    store._db.create_session(session_id=session_id, source="test")
    store._db.append_message(session_id=session_id, role="user", content="old question")
    store._db.append_message(session_id=session_id, role="assistant", content="old answer")
    # In-place compaction: the two rows above are soft-archived and the
    # compacted transcript becomes the live set under the same id.
    store._db.archive_and_compact(
        session_id,
        [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "retry me"},
            {"role": "assistant", "content": "old answer"},
        ],
    )
    assert store._db.has_archived_messages(session_id) is True

    # A failed preflight lookup must not turn this data-preservation path back
    # into a destructive full-history rewrite. The write itself still works.
    archived_probe = MagicMock(side_effect=OSError("transient archive lookup failure"))
    monkeypatch.setattr(store._db, "has_archived_messages", archived_probe)

    gw = GatewayRunner.__new__(GatewayRunner)
    gw.config = config
    gw.session_store = store

    session_entry = MagicMock(session_id=session_id)
    session_entry.last_prompt_tokens = 111
    gw.session_store.get_or_create_session = MagicMock(return_value=session_entry)

    async def fake_handle_message(event):
        assert event.text == "retry me"
        store.append_to_transcript(session_id, {"role": "user", "content": event.text})
        store.append_to_transcript(session_id, {"role": "assistant", "content": "new answer"})
        return "new answer"

    gw._handle_message = AsyncMock(side_effect=fake_handle_message)

    result = await gw._handle_retry_command(
        MessageEvent(text="/retry", message_type=MessageType.TEXT, source=MagicMock())
    )

    assert result == "new answer"
    archived_probe.assert_not_called()
    # The archived pre-compaction rows survive the rewrite untouched.
    archived = [
        m for m in store._db.get_messages(session_id, include_inactive=True)
        if not m["active"]
    ]
    assert [(m["role"], m["content"]) for m in archived] == [
        ("user", "old question"),
        ("assistant", "old answer"),
    ]
    assert all(m["compacted"] == 1 for m in archived)
    # The live set reflects the truncation plus the retried exchange.
    transcript_after = store.load_transcript(session_id)
    assert [m.get("content") for m in transcript_after if m.get("role") == "user"] == [
        "first question",
        "retry me",
    ]

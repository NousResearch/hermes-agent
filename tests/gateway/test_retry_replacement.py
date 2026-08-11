"""Regression tests for /retry replacement semantics."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionStore


@pytest.mark.asyncio
async def test_gateway_retry_archives_superseded_turn_and_replaces_active_transcript(
    tmp_path, monkeypatch
):
    # Pin DEFAULT_DB_PATH so SessionDB() doesn't write to the real ~/.hermes/state.db.
    # (Module-level constant snapshot, see test_load_transcript_db_only.)
    import hermes_state
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")

    config = GatewayConfig()
    store = SessionStore(sessions_dir=tmp_path, config=config)
    db = store._db
    assert db is not None

    session_id = "retry_session"
    db.create_session(session_id=session_id, source="test")
    for msg in [
        {"role": "session_meta", "tools": []},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "retry me"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "old-call",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "content": "old tool result",
            "tool_name": "lookup",
            "tool_call_id": "old-call",
        },
        {"role": "assistant", "content": "old answer"},
    ]:
        store.append_to_transcript(session_id, msg)

    gw = GatewayRunner.__new__(GatewayRunner)
    gw.config = config
    gw.session_store = store

    session_entry = MagicMock(session_id=session_id)
    session_entry.last_prompt_tokens = 111
    gw.session_store.get_or_create_session = MagicMock(return_value=session_entry)

    retry_answers = iter(("new answer", "newer answer"))

    async def fake_handle_message(event):
        assert event.text == "retry me"
        transcript_before = store.load_transcript(session_id)
        assert [m.get("content") for m in transcript_before if m.get("role") == "user"] == [
            "first question"
        ]
        answer = next(retry_answers)
        store.append_to_transcript(session_id, {"role": "user", "content": event.text})
        store.append_to_transcript(session_id, {"role": "assistant", "content": answer})
        return answer

    gw._handle_message = AsyncMock(side_effect=fake_handle_message)

    retry_event = MessageEvent(
        text="/retry", message_type=MessageType.TEXT, source=MagicMock()
    )
    first_result = await gw._handle_retry_command(retry_event)
    result = await gw._handle_retry_command(retry_event)

    assert first_result == "new answer"
    assert result == "newer answer"
    transcript_after = store.load_transcript(session_id)
    assert [m.get("content") for m in transcript_after if m.get("role") == "user"] == [
        "first question",
        "retry me",
    ]
    assert [m.get("content") for m in transcript_after if m.get("role") == "assistant"] == [
        "first answer",
        "newer answer",
    ]

    # /retry's active transcript stays ordered exactly as before, while the
    # complete superseded user/assistant/tool trail remains recoverable as
    # rewind-style inactive rows. Repeating /retry archives only the newly
    # superseded delta and never copies the retained prefix into the archive.
    inactive = [
        m
        for m in db.get_messages(session_id, include_inactive=True)
        if not m["active"]
    ]
    assert [(m["role"], m["content"]) for m in inactive] == [
        ("user", "retry me"),
        ("assistant", None),
        ("tool", "old tool result"),
        ("assistant", "old answer"),
        ("user", "retry me"),
        ("assistant", "new answer"),
    ]
    assert inactive[1]["tool_calls"][0]["id"] == "old-call"
    assert inactive[2]["tool_call_id"] == "old-call"
    assert all(m["compacted"] == 0 for m in inactive)
    assert not any(
        m["content"] in {"first question", "first answer"} for m in inactive
    ), "retained prefix rows must never be copied into inactive history"


@pytest.mark.asyncio
async def test_gateway_retry_aborts_when_canonical_rewrite_fails(tmp_path, monkeypatch):
    """A failed durable rewind leaves memory, disk, and model state untouched."""
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    config = GatewayConfig()
    store = SessionStore(sessions_dir=tmp_path, config=config)
    db = store._db
    assert db is not None

    session_id = "retry_write_failure"
    db.create_session(session_id=session_id, source="test")
    db.append_messages_batch(
        session_id,
        [
            {"role": "user", "content": "retry me"},
            {"role": "assistant", "content": "keep this answer"},
        ],
    )
    before = db.get_messages(session_id, include_inactive=True)

    gw = GatewayRunner.__new__(GatewayRunner)
    gw.config = config
    gw.session_store = store
    session_entry = MagicMock(session_id=session_id)
    session_entry.last_prompt_tokens = 111
    gw.session_store.get_or_create_session = MagicMock(return_value=session_entry)
    gw._handle_message = AsyncMock(return_value="must not be sent")

    monkeypatch.setattr(
        db,
        "replace_messages",
        MagicMock(side_effect=OSError("simulated state.db write failure")),
    )

    result = await gw._handle_retry_command(
        MessageEvent(text="/retry", message_type=MessageType.TEXT, source=MagicMock())
    )

    assert "failed" in result.lower()
    assert session_entry.last_prompt_tokens == 111
    gw._handle_message.assert_not_awaited()
    assert db.get_messages(session_id, include_inactive=True) == before


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
    db = store._db
    assert db is not None

    session_id = "retry_archived_session"
    db.create_session(session_id=session_id, source="test")
    db.append_message(session_id=session_id, role="user", content="old question")
    db.append_message(session_id=session_id, role="assistant", content="old answer")
    # In-place compaction: the two rows above are soft-archived and the
    # compacted transcript becomes the live set under the same id.
    db.archive_and_compact(
        session_id,
        [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "retry me"},
            {"role": "assistant", "content": "old answer"},
        ],
    )
    assert db.has_archived_messages(session_id) is True

    # A failed preflight lookup must not turn this data-preservation path back
    # into a destructive full-history rewrite. The write itself still works.
    archived_probe = MagicMock(side_effect=OSError("transient archive lookup failure"))
    monkeypatch.setattr(db, "has_archived_messages", archived_probe)

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
    # The complete inactive set is stable: pre-compaction rows stay discoverable
    # with compacted=1, while only the superseded retry suffix uses compacted=0.
    inactive = [
        m
        for m in db.get_messages(session_id, include_inactive=True)
        if not m["active"]
    ]
    assert [(m["role"], m["content"], m["compacted"]) for m in inactive] == [
        ("user", "old question", 1),
        ("assistant", "old answer", 1),
        ("user", "retry me", 0),
        ("assistant", "old answer", 0),
    ]
    assert not any(
        m["content"] in {"first question", "first answer"} for m in inactive
    )
    # The live set reflects the truncation plus the retried exchange.
    transcript_after = store.load_transcript(session_id)
    assert [m.get("content") for m in transcript_after if m.get("role") == "user"] == [
        "first question",
        "retry me",
    ]

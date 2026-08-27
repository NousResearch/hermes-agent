"""Regression tests for /retry replacement and carrier-aware undo semantics."""

import os
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.context_compressor import (
    HISTORICAL_TASK_HEADING,
    SUMMARY_PREFIX,
    _SUMMARY_END_MARKER,
)
from gateway.config import GatewayConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionStore


def _composite_carrier(ask="REAL ASK"):
    return {
        "role": "user",
        "content": (
            f"{SUMMARY_PREFIX}\n{HISTORICAL_TASK_HEADING}\nold task\n\n"
            f"{_SUMMARY_END_MARKER}\n\n{ask}"
        ),
    }


def _seed_pending_recovery(store, session_id):
    pending = {"role": "assistant", "content": "pending recovery answer"}
    store._dirty_transcripts[session_id] = [dict(pending)]
    store._transcript_append_failures[session_id] = 3
    return pending


def test_rewrite_transcript_keeps_pending_recovery_state_when_lease_rejects(
    tmp_path, monkeypatch
):
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    session_id = "rewrite-pending-lease"
    store._db.create_session(session_id=session_id, source="test")
    store._db.append_message(session_id, "user", "old ask")
    pending = _seed_pending_recovery(store, session_id)
    before = store._db.get_messages(session_id, include_inactive=True)
    holder = f"pid={os.getpid()}:turn=foreign"
    assert store._db.try_acquire_session_turn_lease(
        session_id, holder, ttl_seconds=60
    )

    assert not store.rewrite_transcript(
        session_id,
        [{"role": "user", "content": "replacement ask"}],
        active_only=True,
        reject_active_turn_lease=True,
    )

    assert store._db.get_messages(session_id, include_inactive=True) == before
    assert store._dirty_transcripts[session_id] == [pending]
    assert store._transcript_append_failures[session_id] == 3

    store._db.release_session_turn_lease(session_id, holder)
    assert store.rewrite_transcript(
        session_id,
        [{"role": "user", "content": "replacement ask"}],
        active_only=True,
        reject_active_turn_lease=True,
    )
    assert session_id not in store._dirty_transcripts
    assert session_id not in store._transcript_append_failures


def test_rewind_session_keeps_pending_recovery_state_when_lease_rejects(
    tmp_path, monkeypatch
):
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    session_id = "rewind-pending-lease"
    store._db.create_session(session_id=session_id, source="test")
    store._db.append_message(session_id, "user", _composite_carrier()["content"])
    store._db.append_message(session_id, "assistant", "old answer")
    pending = _seed_pending_recovery(store, session_id)
    before = store._db.get_messages(session_id, include_inactive=True)
    holder = f"pid={os.getpid()}:turn=foreign"
    assert store._db.try_acquire_session_turn_lease(
        session_id, holder, ttl_seconds=60
    )

    assert (
        store.rewind_session(session_id, require_retryable_composite=True) is None
    )

    assert store._db.get_messages(session_id, include_inactive=True) == before
    assert store._dirty_transcripts[session_id] == [pending]
    assert store._transcript_append_failures[session_id] == 3

    store._db.release_session_turn_lease(session_id, holder)
    result = store.rewind_session(
        session_id, require_retryable_composite=True
    )
    assert result is not None
    assert result["target_text"] == "REAL ASK"
    assert session_id not in store._dirty_transcripts
    assert session_id not in store._transcript_append_failures


def test_rewind_session_surfaces_unretryable_media_before_mutation(
    tmp_path, monkeypatch
):
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    session_id = "rewind-composite-media"
    store._db.create_session(session_id=session_id, source="test")
    store._db.append_message(
        session_id,
        "user",
        [
            {"type": "text", "text": _composite_carrier()["content"]},
            {"type": "image_url", "image_url": {"url": "image"}},
        ],
    )
    store._db.append_message(session_id, "assistant", "old answer")
    before = store._db.get_messages(session_id, include_inactive=True)

    with pytest.raises(ValueError, match="media or unknown content"):
        store.rewind_session(session_id, require_retryable_composite=True)

    assert store._db.get_messages(session_id, include_inactive=True) == before


@pytest.mark.parametrize("operation", ["rewrite", "rewind"])
def test_transcript_mutation_serializes_pending_queue_drain(
    operation, tmp_path, monkeypatch
):
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    session_id = f"serialized-{operation}"
    store._db.create_session(session_id=session_id, source="test")
    store._db.append_message(session_id, "user", _composite_carrier()["content"])
    store._db.append_message(session_id, "assistant", "old answer")
    _seed_pending_recovery(store, session_id)

    mutation_entered = threading.Event()
    release_mutation = threading.Event()
    append_started = threading.Event()
    append_done = threading.Event()
    errors = []

    if operation == "rewrite":
        original_mutation = store._db.replace_messages

        def gated_mutation(*args, **kwargs):
            mutation_entered.set()
            assert release_mutation.wait(timeout=5)
            return original_mutation(*args, **kwargs)

        monkeypatch.setattr(store._db, "replace_messages", gated_mutation)

        def mutate():
            assert store.rewrite_transcript(
                session_id,
                [{"role": "user", "content": "replacement ask"}],
                active_only=True,
            )

    else:
        original_mutation = store._db.rewind_to_message

        def gated_mutation(*args, **kwargs):
            mutation_entered.set()
            assert release_mutation.wait(timeout=5)
            return original_mutation(*args, **kwargs)

        monkeypatch.setattr(store._db, "rewind_to_message", gated_mutation)

        def mutate():
            assert store.rewind_session(session_id) is not None

    def run_mutation():
        try:
            mutate()
        except BaseException as exc:  # surface worker failures in the test thread
            errors.append(exc)

    def append_after_mutation_starts():
        append_started.set()
        try:
            store.append_to_transcript(
                session_id,
                {"role": "assistant", "content": "concurrent answer"},
            )
        except BaseException as exc:  # surface worker failures in the test thread
            errors.append(exc)
        finally:
            append_done.set()

    mutation_thread = threading.Thread(target=run_mutation)
    mutation_thread.start()
    assert mutation_entered.wait(timeout=5)
    append_thread = threading.Thread(target=append_after_mutation_starts)
    append_thread.start()
    assert append_started.wait(timeout=5)
    assert not append_done.wait(timeout=0.1)

    release_mutation.set()
    mutation_thread.join(timeout=5)
    append_thread.join(timeout=5)
    assert not mutation_thread.is_alive()
    assert not append_thread.is_alive()
    assert errors == []
    assert store.load_transcript(session_id)[-1]["content"] == "concurrent answer"


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
async def test_gateway_retry_redispatches_live_carrier_text_and_keeps_scaffold(
    tmp_path, monkeypatch
):
    import hermes_state
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")

    config = GatewayConfig()
    store = SessionStore(sessions_dir=tmp_path, config=config)
    session_id = "retry-carrier-session"
    store._db.create_session(session_id=session_id, source="test")
    store._db.append_message(session_id, "user", "older ask")
    store._db.append_message(session_id, "assistant", "older answer")
    store._db.append_message(session_id, "user", _composite_carrier()["content"])
    store._db.append_message(session_id, "assistant", "failed answer")

    gw = GatewayRunner.__new__(GatewayRunner)
    gw.config = config
    gw.session_store = store
    session_entry = MagicMock(session_id=session_id, last_prompt_tokens=123)
    gw.session_store.get_or_create_session = MagicMock(return_value=session_entry)

    async def fake_handle_message(event):
        assert event.text == "REAL ASK"
        active = store.load_transcript(session_id)
        assert [m.get("content") for m in active[:2]] == ["older ask", "older answer"]
        scaffold = active[2]
        assert scaffold["display_kind"] == "hidden"
        assert "REAL ASK" not in scaffold["content"]
        return "new answer"

    gw._handle_message = AsyncMock(side_effect=fake_handle_message)

    result = await gw._handle_retry_command(
        MessageEvent(text="/retry", message_type=MessageType.TEXT, source=MagicMock())
    )

    assert result == "new answer"
    assert session_entry.last_prompt_tokens == 0
    gw._handle_message.assert_awaited_once()
    archived = [
        row
        for row in store._db.get_messages(session_id, include_inactive=True)
        if not row["active"]
    ]
    assert [row["content"] for row in archived] == [
        _composite_carrier()["content"],
        "failed answer",
    ]


@pytest.mark.asyncio
async def test_gateway_retry_does_not_rewind_a_newer_plain_turn(
    tmp_path, monkeypatch
):
    """The carrier selected for retry must still be latest at commit time."""
    import hermes_state
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")

    config = GatewayConfig()
    store = SessionStore(sessions_dir=tmp_path, config=config)
    session_id = "retry-carrier-race-session"
    store._db.create_session(session_id=session_id, source="test")
    store._db.append_message(session_id, "user", _composite_carrier()["content"])
    store._db.append_message(session_id, "assistant", "failed answer")

    gw = GatewayRunner.__new__(GatewayRunner)
    gw.config = config
    gw.session_store = store
    session_entry = MagicMock(session_id=session_id, last_prompt_tokens=123)
    gw.session_store.get_or_create_session = MagicMock(return_value=session_entry)
    original_rewind = store.rewind_session

    def append_newer_turn_then_rewind(*args, **kwargs):
        store._db.append_message(session_id, "user", "newer ask")
        store._db.append_message(session_id, "assistant", "newer answer")
        return original_rewind(*args, **kwargs)

    monkeypatch.setattr(store, "rewind_session", append_newer_turn_then_rewind)
    gw._handle_message = AsyncMock()

    result = await gw._handle_retry_command(
        MessageEvent(text="/retry", message_type=MessageType.TEXT, source=MagicMock())
    )

    assert result.startswith("Retry failed;")
    assert session_entry.last_prompt_tokens == 123
    gw._handle_message.assert_not_awaited()
    assert [
        message.get("content")
        for message in store.load_transcript(session_id)
        if message.get("role") == "user"
    ] == [_composite_carrier()["content"], "newer ask"]


@pytest.mark.asyncio
async def test_gateway_retry_rejects_media_before_redispatch_or_token_reset():
    gw = GatewayRunner.__new__(GatewayRunner)
    backing_store = MagicMock()
    gw.session_store = backing_store
    session_entry = SimpleNamespace(session_id="sid", last_prompt_tokens=123)
    facade = SimpleNamespace(
        _store=backing_store,
        get_or_create_session=AsyncMock(return_value=session_entry),
        load_transcript=AsyncMock(
            return_value=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look again"},
                        {"type": "image_url", "image_url": {"url": "image"}},
                    ],
                },
                {"role": "assistant", "content": "old answer"},
            ]
        ),
        rewrite_transcript=AsyncMock(return_value=True),
    )
    gw._async_session_store = facade
    gw._handle_message = AsyncMock()

    result = await gw._handle_retry_command(
        MessageEvent(text="/retry", message_type=MessageType.TEXT, source=MagicMock())
    )

    assert result.startswith("Cannot retry that message safely:")
    assert session_entry.last_prompt_tokens == 123
    gw._handle_message.assert_not_awaited()
    facade.rewrite_transcript.assert_not_awaited()


@pytest.mark.asyncio
async def test_gateway_retry_preserves_composite_media_diagnostic_from_store():
    gw = GatewayRunner.__new__(GatewayRunner)
    backing_store = MagicMock()
    gw.session_store = backing_store
    session_entry = SimpleNamespace(session_id="sid", last_prompt_tokens=123)
    facade = SimpleNamespace(
        _store=backing_store,
        get_or_create_session=AsyncMock(return_value=session_entry),
        load_transcript=AsyncMock(
            return_value=[
                _composite_carrier(),
                {"role": "assistant", "content": "old answer"},
            ]
        ),
        rewind_session=AsyncMock(
            side_effect=ValueError("retry does not support media content")
        ),
    )
    gw._async_session_store = facade
    gw._handle_message = AsyncMock()

    result = await gw._handle_retry_command(
        MessageEvent(text="/retry", message_type=MessageType.TEXT, source=MagicMock())
    )

    assert result == (
        "Cannot retry that message safely: retry does not support media content"
    )
    assert session_entry.last_prompt_tokens == 123
    gw._handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_gateway_retry_stops_when_transcript_rewrite_fails():
    gw = GatewayRunner.__new__(GatewayRunner)
    backing_store = MagicMock()
    gw.session_store = backing_store
    session_entry = SimpleNamespace(session_id="sid", last_prompt_tokens=123)
    facade = SimpleNamespace(
        _store=backing_store,
        get_or_create_session=AsyncMock(return_value=session_entry),
        load_transcript=AsyncMock(
            return_value=[
                {"role": "user", "content": "retry me"},
                {"role": "assistant", "content": "old answer"},
            ]
        ),
        rewrite_transcript=AsyncMock(return_value=False),
    )
    gw._async_session_store = facade
    gw._handle_message = AsyncMock()

    result = await gw._handle_retry_command(
        MessageEvent(text="/retry", message_type=MessageType.TEXT, source=MagicMock())
    )

    assert result.startswith("Retry failed;")
    assert session_entry.last_prompt_tokens == 123
    gw._handle_message.assert_not_awaited()
    facade.rewrite_transcript.assert_awaited_once()
    assert (
        facade.rewrite_transcript.await_args.kwargs["reject_active_turn_lease"]
        is True
    )


def test_gateway_undo_prefills_live_carrier_text_and_keeps_scaffold(
    tmp_path, monkeypatch
):
    import hermes_state
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")

    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    session_id = "undo-carrier-session"
    store._db.create_session(session_id=session_id, source="test")
    store._db.append_message(session_id, "user", _composite_carrier()["content"])
    store._db.append_message(session_id, "assistant", "failed answer")

    result = store.rewind_session(session_id)

    assert result["target_text"] == "REAL ASK"
    assert result["rewound_count"] == 2
    active = store._db.get_messages_as_conversation(
        session_id, include_row_ids=True
    )
    assert len(active) == 1
    assert active[0]["display_kind"] == "hidden"
    assert "REAL ASK" not in active[0]["content"]


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

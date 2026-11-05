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
from agent.i18n import t
from gateway.config import GatewayConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionStore
from persistence.conversation.messages import (
    ConversationRequest,
    NewMessage,
    ReplaceMode,
    TranscriptRead,
    TranscriptView,
)
from tests.fixtures.storage_read import conversation, transcript_rows
from tests.fixtures.storage_seed import seed_messages, seed_session

ALL_ROWS = TranscriptRead(view=TranscriptView.ALL)


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
    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    session_id = "rewrite-pending-lease"
    seed_session(store.persistence, session_id, "test")
    seed_messages(store.persistence, session_id, {"role": "user", "content": "old ask"})
    pending = _seed_pending_recovery(store, session_id)
    before = transcript_rows(store.persistence, session_id, ALL_ROWS)
    holder = f"pid={os.getpid()}:turn=foreign"
    assert store.persistence.coordination.try_acquire_turn_lease(
        session_id=session_id, holder=holder, ttl_seconds=60
    )

    assert not store.rewrite_transcript(
        session_id,
        [{"role": "user", "content": "replacement ask"}],
        active_only=True,
        reject_active_turn_lease=True,
    )

    assert transcript_rows(store.persistence, session_id, ALL_ROWS) == before
    assert store._dirty_transcripts[session_id] == [pending]
    assert store._transcript_append_failures[session_id] == 3

    store.persistence.coordination.release_turn_lease(session_id=session_id, holder=holder)
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
    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    session_id = "rewind-pending-lease"
    seed_session(store.persistence, session_id, "test")
    seed_messages(store.persistence, session_id, {"role": "user", "content": _composite_carrier()["content"]})
    seed_messages(store.persistence, session_id, {"role": "assistant", "content": "old answer"})
    pending = _seed_pending_recovery(store, session_id)
    before = transcript_rows(store.persistence, session_id, ALL_ROWS)
    holder = f"pid={os.getpid()}:turn=foreign"
    assert store.persistence.coordination.try_acquire_turn_lease(
        session_id=session_id, holder=holder, ttl_seconds=60
    )

    assert (
        store.rewind_session(session_id, require_retryable_composite=True) is None
    )

    assert transcript_rows(store.persistence, session_id, ALL_ROWS) == before
    assert store._dirty_transcripts[session_id] == [pending]
    assert store._transcript_append_failures[session_id] == 3

    store.persistence.coordination.release_turn_lease(session_id=session_id, holder=holder)
    result = store.rewind_session(
        session_id, require_retryable_composite=True
    )
    assert result is not None
    assert result["target_text"] == "REAL ASK"
    assert session_id not in store._dirty_transcripts
    assert session_id not in store._transcript_append_failures


def test_rewind_session_retry_text_is_the_stored_bytes_for_multipart_carriers(
    tmp_path, monkeypatch
):
    """/retry re-sends exactly what was stored: a carrier split across text parts comes back as the
    parts' concatenation (``"".join``), not the "\\n"-joined display flattening."""
    from agent.context_compressor import retryable_user_text

    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    session_id = "rewind-composite-multipart"
    seed_session(store.persistence, session_id, "test")
    parts = [
        {"type": "text", "text": _composite_carrier("REAL")["content"]},
        {"type": "text", "text": " ASK"},
        {"type": "text", "text": "\ncontinued"},
    ]
    seed_messages(store.persistence, session_id, {"role": "user", "content": parts})
    seed_messages(store.persistence, session_id, {"role": "assistant", "content": "old answer"})
    stored = conversation(store.persistence, session_id)[0]["content"]
    assert isinstance(stored, list) and len(stored) == 3

    result = store.rewind_session(session_id, require_retryable_composite=True)

    assert result is not None
    expected = "".join(p["text"] for p in stored).split(_SUMMARY_END_MARKER, 1)[1].lstrip("\n")
    assert result["target_text"] == "REAL ASK\ncontinued" == expected
    assert result["target_text"] == retryable_user_text(
        [{"type": "text", "text": "REAL"}, {"type": "text", "text": " ASK"}, {"type": "text", "text": "\ncontinued"}])


def test_rewind_session_surfaces_unretryable_media_before_mutation(
    tmp_path, monkeypatch
):
    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    session_id = "rewind-composite-media"
    seed_session(store.persistence, session_id, "test")
    seed_messages(
        store.persistence,
        session_id,
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _composite_carrier()["content"]},
                {"type": "image_url", "image_url": {"url": "image"}},
            ],
        },
    )
    seed_messages(store.persistence, session_id, {"role": "assistant", "content": "old answer"})
    before = transcript_rows(store.persistence, session_id, ALL_ROWS)

    with pytest.raises(ValueError, match="media or unknown content"):
        store.rewind_session(session_id, require_retryable_composite=True)

    assert transcript_rows(store.persistence, session_id, ALL_ROWS) == before


@pytest.mark.parametrize("operation", ["rewrite", "rewind"])
def test_transcript_mutation_serializes_pending_queue_drain(
    operation, tmp_path, monkeypatch
):
    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    session_id = f"serialized-{operation}"
    seed_session(store.persistence, session_id, "test")
    seed_messages(store.persistence, session_id, {"role": "user", "content": _composite_carrier()["content"]})
    seed_messages(store.persistence, session_id, {"role": "assistant", "content": "old answer"})
    _seed_pending_recovery(store, session_id)

    mutation_entered = threading.Event()
    release_mutation = threading.Event()
    append_started = threading.Event()
    append_done = threading.Event()
    errors = []

    if operation == "rewrite":
        messages = store.persistence.messages
        original_mutation = messages.replace_transcript

        def gated_mutation(*args, **kwargs):
            mutation_entered.set()
            assert release_mutation.wait(timeout=5)
            return original_mutation(*args, **kwargs)

        monkeypatch.setattr(messages, "replace_transcript", gated_mutation)

        def mutate():
            assert store.rewrite_transcript(
                session_id,
                [{"role": "user", "content": "replacement ask"}],
                active_only=True,
            )

    else:
        messages = store.persistence.messages
        original_mutation = messages.rewind_user_turn

        def gated_mutation(*args, **kwargs):
            mutation_entered.set()
            assert release_mutation.wait(timeout=5)
            return original_mutation(*args, **kwargs)

        monkeypatch.setattr(messages, "rewind_user_turn", gated_mutation)

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
    config = GatewayConfig()
    store = SessionStore(sessions_dir=tmp_path, config=config)

    session_id = "retry_session"
    seed_session(store.persistence, session_id, "test")
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
    config = GatewayConfig()
    store = SessionStore(sessions_dir=tmp_path, config=config)
    session_id = "retry-carrier-session"
    seed_session(store.persistence, session_id, "test")
    seed_messages(store.persistence, session_id, {"role": "user", "content": "older ask"})
    seed_messages(store.persistence, session_id, {"role": "assistant", "content": "older answer"})
    seed_messages(store.persistence, session_id, {"role": "user", "content": _composite_carrier()["content"]})
    seed_messages(store.persistence, session_id, {"role": "assistant", "content": "failed answer"})

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
        for row in transcript_rows(store.persistence, session_id, ALL_ROWS)
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
    config = GatewayConfig()
    store = SessionStore(sessions_dir=tmp_path, config=config)
    session_id = "retry-carrier-race-session"
    seed_session(store.persistence, session_id, "test")
    seed_messages(store.persistence, session_id, {"role": "user", "content": _composite_carrier()["content"]})
    seed_messages(store.persistence, session_id, {"role": "assistant", "content": "failed answer"})

    gw = GatewayRunner.__new__(GatewayRunner)
    gw.config = config
    gw.session_store = store
    session_entry = MagicMock(session_id=session_id, last_prompt_tokens=123)
    gw.session_store.get_or_create_session = MagicMock(return_value=session_entry)
    original_rewind = store.rewind_session

    def append_newer_turn_then_rewind(*args, **kwargs):
        seed_messages(store.persistence, session_id, {"role": "user", "content": "newer ask"})
        seed_messages(store.persistence, session_id, {"role": "assistant", "content": "newer answer"})
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
    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    session_id = "undo-carrier-session"
    seed_session(store.persistence, session_id, "test")
    seed_messages(store.persistence, session_id, {"role": "user", "content": _composite_carrier()["content"]})
    seed_messages(store.persistence, session_id, {"role": "assistant", "content": "failed answer"})

    result = store.rewind_session(session_id)

    assert result["target_text"] == "REAL ASK"
    assert result["rewound_count"] == 2
    active = conversation(
        store.persistence, session_id, ConversationRequest(include_row_ids=True)
    )
    assert len(active) == 1
    assert active[0]["display_kind"] == "hidden"
    assert "REAL ASK" not in active[0]["content"]


@pytest.mark.asyncio
async def test_gateway_retry_preserves_archived_compaction_rows_when_probe_fails(
    tmp_path, monkeypatch
):
    """/retry must not DELETE archives when an existence probe would fail.

    With compression.in_place (the default, #38763) the compaction
    keeps the pre-compaction transcript on disk as active=0/compacted=1 rows
    under the same session id. /retry used to persist its truncation via a
    bare rewrite_transcript(), whose destructive replace
    DELETEs every row for the session and reinserts only the truncated live
    tail, wiping the archived history permanently (same class as #61145;
    #57803 named this call site as a residual gap). /retry never intends to
    purge archived history, so it must pass active_only=True unconditionally:
    a separate existence probe can fail open or race with the rewrite.
    """
    config = GatewayConfig()
    store = SessionStore(sessions_dir=tmp_path, config=config)

    session_id = "retry_archived_session"
    seed_session(store.persistence, session_id, "test")
    seed_messages(store.persistence, session_id, {"role": "user", "content": "old question"},
                  {"role": "assistant", "content": "old answer"})
    # In-place compaction: the two rows above are soft-archived and the
    # compacted transcript becomes the live set under the same id.
    store.persistence.messages.compact(
        session_id,
        [
            NewMessage.from_live(message)
            for message in (
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "retry me"},
                {"role": "assistant", "content": "old answer"},
            )
        ],
    )
    assert any(not row["active"] for row in transcript_rows(store.persistence, session_id, ALL_ROWS))

    # The rewrite keeps the archive by its replace mode, never by an existence
    # probe that could fail open into a destructive full-history rewrite.
    messages = store.persistence.messages
    replace = messages.replace_transcript
    modes = []

    def recording_replace(session_id, rows, mode, **kwargs):
        modes.append(mode)
        return replace(session_id, rows, mode, **kwargs)

    monkeypatch.setattr(messages, "replace_transcript", recording_replace)

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
    assert modes == [ReplaceMode.PRESERVE_EXISTING_ARCHIVE]
    # The archived pre-compaction rows survive the rewrite untouched.
    archived = [
        m for m in transcript_rows(store.persistence, session_id, ALL_ROWS)
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


def _paused_runner_with_exchange(tmp_path, monkeypatch, session_id, rows):
    """Bare runner whose session holds one plain user/assistant exchange; returns
    ``(gw, store, session_entry)`` with the transcript snapshot taken before /retry."""
    config = GatewayConfig()
    store = SessionStore(sessions_dir=tmp_path, config=config)
    seed_session(store.persistence, session_id, "test")
    for role, content in rows:
        seed_messages(store.persistence, session_id, {"role": role, "content": content})

    gw = GatewayRunner.__new__(GatewayRunner)
    gw.config = config
    gw.session_store = store
    session_entry = MagicMock(session_id=session_id)
    session_entry.last_prompt_tokens = 111
    gw.session_store.get_or_create_session = MagicMock(return_value=session_entry)
    gw._handle_message = AsyncMock()
    return gw, store, session_entry


@pytest.mark.asyncio
async def test_gateway_retry_paused_refusal_leaves_transcript_unchanged(tmp_path, monkeypatch):
    """The estop pause refuses the re-sent plain-text turn (the /retry command itself passes the
    gate as a canonical command); /retry must answer it BEFORE the rewrite or the transcript
    ends up shorter and the user's turn is lost."""
    gw, store, session_entry = _paused_runner_with_exchange(
        tmp_path, monkeypatch, "retry-paused-session",
        [("user", "retry me"), ("assistant", "old answer")],
    )
    before = transcript_rows(store.persistence, "retry-paused-session", ALL_ROWS)
    paused_notice = "⏸️ Hermes is paused. New work is on hold; run `hermes resume`."
    monkeypatch.setattr("agent.estop.paused_reply", lambda: paused_notice)

    result = await gw._handle_retry_command(
        MessageEvent(text="/retry", message_type=MessageType.TEXT, source=MagicMock())
    )

    assert result == paused_notice
    assert transcript_rows(store.persistence, "retry-paused-session", ALL_ROWS) == before
    gw._handle_message.assert_not_awaited()
    assert session_entry.last_prompt_tokens == 111


@pytest.mark.asyncio
async def test_gateway_retry_paused_refusal_leaves_carrier_rewind_untouched(
    tmp_path, monkeypatch
):
    """Same pause pre-check on the composite-carrier path: the carrier-aware rewind must not run
    either, or the retained scaffold is swapped for the pure scaffold with the ask gone."""
    gw, store, session_entry = _paused_runner_with_exchange(
        tmp_path, monkeypatch, "retry-paused-carrier-session",
        [("user", _composite_carrier()["content"]), ("assistant", "failed answer")],
    )
    before = transcript_rows(store.persistence, "retry-paused-carrier-session", ALL_ROWS)
    paused_notice = "⏸️ Hermes is paused. New work is on hold; run `hermes resume`."
    monkeypatch.setattr("agent.estop.paused_reply", lambda: paused_notice)

    result = await gw._handle_retry_command(
        MessageEvent(text="/retry", message_type=MessageType.TEXT, source=MagicMock())
    )

    assert result == paused_notice
    assert transcript_rows(store.persistence, "retry-paused-carrier-session", ALL_ROWS) == before
    gw._handle_message.assert_not_awaited()
    assert session_entry.last_prompt_tokens == 111


@pytest.mark.asyncio
async def test_gateway_retry_draining_refusal_leaves_transcript_unchanged(tmp_path, monkeypatch):
    """An externally drained gateway refuses NEW turns after canonical command dispatch, so the
    /retry re-send is refused; answering that refusal before the rewrite keeps the exchange."""
    gw, store, session_entry = _paused_runner_with_exchange(
        tmp_path, monkeypatch, "retry-draining-session",
        [("user", "retry me"), ("assistant", "old answer")],
    )
    before = transcript_rows(store.persistence, "retry-draining-session", ALL_ROWS)
    gw._external_drain_active = True

    result = await gw._handle_retry_command(
        MessageEvent(text="/retry", message_type=MessageType.TEXT, source=MagicMock())
    )

    assert result == t("gateway.busy.draining_maintenance")
    assert transcript_rows(store.persistence, "retry-draining-session", ALL_ROWS) == before
    gw._handle_message.assert_not_awaited()
    assert session_entry.last_prompt_tokens == 111


@pytest.mark.asyncio
async def test_gateway_retry_session_cap_refusal_leaves_transcript_unchanged(
    tmp_path, monkeypatch
):
    """max_concurrent_sessions rejects the re-sent turn's slot claim; the same cap message must
    be answerable before the rewrite so the exchange survives for a later /retry."""
    gw, store, session_entry = _paused_runner_with_exchange(
        tmp_path, monkeypatch, "retry-cap-session",
        [("user", "retry me"), ("assistant", "old answer")],
    )
    before = transcript_rows(store.persistence, "retry-cap-session", ALL_ROWS)
    limit_message = "⏸️ Active session limit reached (4/4)."
    gw._active_session_limit_message = MagicMock(return_value=limit_message)

    result = await gw._handle_retry_command(
        MessageEvent(text="/retry", message_type=MessageType.TEXT, source=MagicMock())
    )

    assert result == limit_message
    assert transcript_rows(store.persistence, "retry-cap-session", ALL_ROWS) == before
    gw._handle_message.assert_not_awaited()
    assert session_entry.last_prompt_tokens == 111

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
async def test_gateway_retry_replays_original_text_not_retry_command(tmp_path):
    config = MagicMock()
    config.sessions_dir = tmp_path
    config.max_context_messages = 20
    gw = GatewayRunner.__new__(GatewayRunner)
    gw.config = config
    gw.session_store = MagicMock()

    session_entry = MagicMock(session_id="test-session")
    session_entry.last_prompt_tokens = 55
    gw.session_store.get_or_create_session.return_value = session_entry
    gw.session_store.load_transcript.return_value = [
        {"role": "user", "content": "real message"},
        {"role": "assistant", "content": "answer"},
    ]
    gw.session_store.rewrite_transcript = MagicMock()

    captured = {}

    async def fake_handle_message(event):
        captured["text"] = event.text
        return "ok"

    gw._handle_message = AsyncMock(side_effect=fake_handle_message)

    await gw._handle_retry_command(
        MessageEvent(text="/retry", message_type=MessageType.TEXT, source=MagicMock())
    )

    assert captured["text"] == "real message"


def _retry_gateway(tmp_path, transcript, remembered):
    """Build a GatewayRunner stub primed for a /retry replay test."""
    config = MagicMock()
    config.sessions_dir = tmp_path
    config.max_context_messages = 20
    gw = GatewayRunner.__new__(GatewayRunner)
    gw.config = config
    gw.session_store = MagicMock()

    session_entry = MagicMock(session_id="sid", session_key="skey")
    session_entry.last_prompt_tokens = 7
    gw.session_store.get_or_create_session.return_value = session_entry
    gw.session_store.load_transcript.return_value = transcript
    gw.session_store.rewrite_transcript = MagicMock()
    if remembered is not None:
        gw._last_user_event_by_session = {"skey": remembered}
    return gw


@pytest.mark.asyncio
async def test_gateway_retry_replays_reply_context(tmp_path):
    """/retry must replay the original reply pointer, not a text-only copy."""
    gw = _retry_gateway(
        tmp_path,
        transcript=[
            {"role": "user", "content": '[Replying to: "quoted original"]\n\nfollow up'},
            {"role": "assistant", "content": "bad answer"},
        ],
        remembered=MessageEvent(
            text="follow up",
            message_type=MessageType.TEXT,
            reply_to_message_id="42",
            reply_to_text="quoted original",
        ),
    )

    captured = {}

    async def fake_handle_message(event):
        captured["event"] = event
        return "ok"

    gw._handle_message = AsyncMock(side_effect=fake_handle_message)

    result = await gw._handle_retry_command(
        MessageEvent(text="/retry", message_type=MessageType.TEXT, source=MagicMock())
    )

    assert result == "ok"
    replay = captured["event"]
    assert replay.text == "follow up"
    assert replay.reply_to_message_id == "42"
    assert replay.reply_to_text == "quoted original"


@pytest.mark.asyncio
async def test_gateway_retry_replays_image_media(tmp_path):
    """/retry after an image turn must re-attach media_urls/types and PHOTO."""
    gw = _retry_gateway(
        tmp_path,
        transcript=[
            {"role": "user", "content": "look at this"},
            {"role": "assistant", "content": "bad answer"},
        ],
        remembered=MessageEvent(
            text="look at this",
            message_type=MessageType.PHOTO,
            media_urls=["/tmp/cat.png"],
            media_types=["image/png"],
        ),
    )

    captured = {}

    async def fake_handle_message(event):
        captured["event"] = event
        return "ok"

    gw._handle_message = AsyncMock(side_effect=fake_handle_message)

    await gw._handle_retry_command(
        MessageEvent(text="/retry", message_type=MessageType.TEXT, source=MagicMock())
    )

    replay = captured["event"]
    assert replay.message_type == MessageType.PHOTO
    assert replay.media_urls == ["/tmp/cat.png"]
    assert replay.media_types == ["image/png"]


@pytest.mark.asyncio
async def test_gateway_retry_falls_back_to_text_without_remembered_event(tmp_path):
    """When no structured event was captured (e.g. after a restart), /retry
    falls back to the historical text-only replay of the last user turn."""
    gw = _retry_gateway(
        tmp_path,
        transcript=[
            {"role": "user", "content": "plain old turn"},
            {"role": "assistant", "content": "bad answer"},
        ],
        remembered=None,
    )

    captured = {}

    async def fake_handle_message(event):
        captured["event"] = event
        return "ok"

    gw._handle_message = AsyncMock(side_effect=fake_handle_message)

    await gw._handle_retry_command(
        MessageEvent(text="/retry", message_type=MessageType.TEXT, source=MagicMock())
    )

    replay = captured["event"]
    assert replay.text == "plain old turn"
    assert replay.message_type == MessageType.TEXT
    assert replay.media_urls == []


@pytest.mark.asyncio
async def test_gateway_retry_rebuilds_native_image_attach_path(tmp_path):
    """End-to-end: the replayed image event re-establishes the per-session
    native image buffer that the run path consumes to attach pixels inline."""
    from gateway.config import Platform, PlatformConfig
    from gateway.session import SessionSource, build_session_key

    gw = _retry_gateway(
        tmp_path,
        transcript=[
            {"role": "user", "content": "see image"},
            {"role": "assistant", "content": "wrong"},
        ],
        remembered=MessageEvent(
            text="see image",
            message_type=MessageType.PHOTO,
            media_urls=["/tmp/a.png"],
            media_types=["image/png"],
        ),
    )
    # _prepare_inbound_message_text needs these to route images natively.
    gw.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")},
    )
    gw.adapters = {}
    gw._model = "openai/gpt-4.1-mini"
    gw._base_url = None
    gw._decide_image_input_mode = lambda: "native"

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="private",
        user_name="Alice",
    )

    captured = {}

    async def fake_handle_message(event):
        captured["event"] = event
        return "ok"

    gw._handle_message = AsyncMock(side_effect=fake_handle_message)

    await gw._handle_retry_command(
        MessageEvent(text="/retry", message_type=MessageType.TEXT, source=source)
    )

    replay = captured["event"]
    # Running the replayed event through the normal preprocessing pipeline
    # buffers the image for native (inline pixel) attachment — the path that
    # the old text-only /retry silently dropped.
    await gw._prepare_inbound_message_text(event=replay, source=source, history=[])
    assert gw._consume_pending_native_image_paths(build_session_key(source)) == [
        "/tmp/a.png"
    ]

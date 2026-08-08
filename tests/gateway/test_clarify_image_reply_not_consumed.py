"""Regression tests for #81117 — Feishu image replies during a pending clarify
must not be resolved with the adapter's ``[Image]`` placeholder and must not
silently drop the attachment.

The clarify intercept block in ``gateway/run.py`` is reached before the
normal media-routing block, so a Feishu ``post`` message that embeds an
image (rendered as ``[Image]`` / ``[Image: alt]`` in ``event.text``) used
to be coerced into the pending clarify as a literal answer. Downstream the
vision tool treated ``"[Image]"`` as a file path and errored with
``media file not found: '[Image]'`` while the real attachment in
``event.media_urls`` was never consumed.

These tests pin down the two deliverable behaviours:

* an image-only reply leaves the clarify pending and tells the user to
  reply with text,
* a reply that carries a real caption together with an image is stripped
  of the placeholder, the caption resolves the clarify, and the
  attachment survives on the event untouched.

A normal image message (no pending clarify) is asserted to keep flowing
past the intercept unchanged.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionEntry, SessionSource, build_session_key
from tools import clarify_gateway as cm


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="ou_user",
        chat_id="oc_chat",
        user_name="tester",
        chat_type="dm",
    )


def _make_image_event(
    text: str,
    *,
    media_paths: list[str],
    media_types: list[str],
    message_type: MessageType = MessageType.TEXT,
    message_id: str = "m-img",
) -> MessageEvent:
    """Construct an inbound event with the supplied image attachments."""
    return MessageEvent(
        text=text,
        message_type=message_type,
        source=_make_source(),
        message_id=message_id,
        media_urls=list(media_paths),
        media_types=list(media_types),
    )


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-img",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    return runner


def _clear_clarify_state() -> None:
    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()


def _register_pending(session_key: str, clarify_id: str) -> None:
    """Register an open-ended clarify so free-text replies can resolve it.

    Open-ended clarifies set ``awaiting_text=True`` at registration time,
    which mirrors the most common case where the user is asked an
    open question and is expected to type a free-form answer.
    """
    cm.register(clarify_id, session_key, "Pick the screenshot", None)


@pytest.mark.asyncio
async def test_image_only_post_reply_keeps_clarify_pending_and_asks_for_text(
    monkeypatch,
):
    """A Feishu ``post`` message that contains ONLY an image (rendered as
    ``[Image]`` in ``event.text``) must not resolve a pending clarify with
    the placeholder. The prompt stays pending and the user is informed that
    a text reply is required (#81117)."""
    import gateway.run as gateway_run

    _clear_clarify_state()
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    runner = _make_runner()
    session_key = build_session_key(_make_source())
    clarify_id = "cl-81117-image-only"
    _register_pending(session_key, clarify_id)

    event = _make_image_event(
        text="[Image]",
        media_paths=["/tmp/feishu-image.png"],
        media_types=["image/png"],
        message_type=MessageType.TEXT,
    )

    result = await runner._handle_message(event)

    # The user is told a text reply is needed; the placeholder never leaks
    # into a clarify answer and the attachment is not silently dropped.
    assert result is not None
    assert "text" in result.lower()
    assert "[Image]" not in (result or "")

    # The clarify entry is still pending — the gateway did NOT resolve it
    # with the placeholder.
    entry = cm._entries.get(clarify_id)
    assert entry is not None
    assert entry.event.is_set() is False
    assert entry.response is None

    # The attachment survives intact on the event so a follow-up turn could
    # route it through the normal media-routing block if needed.
    assert event.media_urls == ["/tmp/feishu-image.png"]
    assert event.media_types == ["image/png"]

    _clear_clarify_state()


@pytest.mark.asyncio
async def test_image_only_post_reply_with_alt_text_keeps_clarify_pending(
    monkeypatch,
):
    """A Feishu ``post`` message that renders the image as ``[Image: alt]``
    behaves the same way — the placeholder is not a valid answer."""
    import gateway.run as gateway_run

    _clear_clarify_state()
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    runner = _make_runner()
    session_key = build_session_key(_make_source())
    clarify_id = "cl-81117-image-alt"
    _register_pending(session_key, clarify_id)

    event = _make_image_event(
        text="[Image: screenshot]",
        media_paths=["/tmp/feishu-image.png"],
        media_types=["image/png"],
        message_type=MessageType.TEXT,
    )

    result = await runner._handle_message(event)

    assert result is not None
    assert "[Image" not in (result or "")

    entry = cm._entries.get(clarify_id)
    assert entry is not None
    assert entry.event.is_set() is False

    assert event.media_urls == ["/tmp/feishu-image.png"]
    _clear_clarify_state()


@pytest.mark.asyncio
async def test_image_with_caption_resolves_clarify_with_caption_and_preserves_attachment(
    monkeypatch,
):
    """A reply that carries both an image AND a real caption must resolve the
    clarify with the caption (placeholder stripped). The attachment remains
    on the event so it is not silently dropped (#81117)."""
    import gateway.run as gateway_run

    _clear_clarify_state()
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    runner = _make_runner()
    session_key = build_session_key(_make_source())
    clarify_id = "cl-81117-image-caption"
    _register_pending(session_key, clarify_id)

    event = _make_image_event(
        text="Use the blue variant [Image: screenshot]",
        media_paths=["/tmp/feishu-image.png"],
        media_types=["image/png"],
        message_type=MessageType.TEXT,
    )

    result = await runner._handle_message(event)

    # Clarify resolution returns an empty string (consumed by the intercept,
    # mirroring the voice precedent in #73518) so the adapter doesn't
    # double-post.
    assert result == ""

    # The caption — NOT the placeholder — is what resolved the clarify.
    entry = cm._entries.get(clarify_id)
    assert entry is not None
    assert entry.event.is_set() is True
    assert entry.response == "Use the blue variant"
    assert "[Image" not in (entry.response or "")

    # The attachment is preserved on the event, not consumed/mangled.
    assert event.media_urls == ["/tmp/feishu-image.png"]
    assert event.media_types == ["image/png"]
    _clear_clarify_state()


@pytest.mark.asyncio
async def test_placeholder_only_text_without_image_attachment_does_not_resolve_clarify(
    monkeypatch,
):
    """Defence-in-depth: a stray ``[Image]`` text with no real attachment
    (e.g. an image whose download failed) must NOT be treated as a valid
    clarify answer either."""
    import gateway.run as gateway_run

    _clear_clarify_state()
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    runner = _make_runner()
    session_key = build_session_key(_make_source())
    clarify_id = "cl-81117-image-nomedia"
    _register_pending(session_key, clarify_id)

    # No media_urls — the download failed, only the placeholder survived.
    event = _make_image_event(
        text="[Image: download failed]",
        media_paths=[],
        media_types=[],
        message_type=MessageType.TEXT,
    )

    result = await runner._handle_message(event)

    assert result is not None
    assert "[Image" not in (result or "")

    entry = cm._entries.get(clarify_id)
    assert entry is not None
    assert entry.event.is_set() is False
    assert entry.response is None
    _clear_clarify_state()


@pytest.mark.asyncio
async def test_normal_image_message_without_pending_clarify_unaffected(
    monkeypatch,
):
    """A normal Feishu image message (no pending clarify) must continue to
    reach the normal media-routing block unaffected by the clarify fix
    (#81117). We assert this by raising a tripwire just AFTER the
    slash-confirm block — if the clarify intercept fires, the tripwire
    runs the rest of the handler which would not raise."""
    import tools.slash_confirm as slash_confirm_mod

    class _TripwireFellThrough(Exception):
        """Sentinel: message flowed past slash-confirm — clarify intercept
        either did nothing (expected) or returned."""

    _clear_clarify_state()
    import gateway.run as gateway_run

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    def _tripwire(_key):
        raise _TripwireFellThrough()

    runner = _make_runner()
    monkeypatch.setattr(
        slash_confirm_mod, "get_pending", _tripwire
    )

    # Native Feishu image: text is empty, message_type PHOTO, real
    # attachment present.  Without a pending clarify this MUST fall through
    # to the slash-confirm block (which raises our sentinel).
    event = _make_image_event(
        text="",
        media_paths=["/tmp/feishu-image.png"],
        media_types=["image/png"],
        message_type=MessageType.PHOTO,
        message_id="m-img-no-clarify",
    )

    with pytest.raises(_TripwireFellThrough):
        await runner._handle_message(event)

    # And critically: no clarify entry was created or resolved by the fix.
    assert cm._entries == {}
    _clear_clarify_state()


@pytest.mark.asyncio
async def test_prepare_clarify_reply_text_strips_image_placeholders():
    """Direct unit coverage for the helper: image placeholders never
    survive into the text the gateway hands to clarify resolution."""
    from gateway.run import GatewayRunner, _strip_clarify_image_placeholders

    runner = object.__new__(GatewayRunner)
    runner._pending_event_audio_paths = lambda _event: []

    event = _make_image_event(
        text="[Image]",
        media_paths=["/tmp/feishu-image.png"],
        media_types=["image/png"],
    )

    cleaned = await runner._prepare_clarify_reply_text(event)
    assert cleaned == ""
    assert "[Image]" not in cleaned

    event_caption = _make_image_event(
        text="caption [Image: alt] tail",
        media_paths=["/tmp/feishu-image.png"],
        media_types=["image/png"],
    )
    cleaned_caption = await runner._prepare_clarify_reply_text(event_caption)
    assert cleaned_caption == "caption tail"
    assert "[Image" not in cleaned_caption


def test_strip_clarify_image_placeholders_unit():
    """Pin the regex behaviour so future edits to the placeholder format
    in the Feishu adapter don't silently desync from the gateway."""
    from gateway.run import _strip_clarify_image_placeholders

    assert _strip_clarify_image_placeholders("[Image]") == ""
    assert _strip_clarify_image_placeholders("[Image: alt]") == ""
    assert _strip_clarify_image_placeholders("Use [Image] now") == "Use now"
    assert (
        _strip_clarify_image_placeholders("caption [Image: diagram] here")
        == "caption here"
    )
    # Non-image text untouched.
    assert _strip_clarify_image_placeholders("plain text") == "plain text"
    # Empty input.
    assert _strip_clarify_image_placeholders("") == ""
    # Whitespace-only around the marker collapses cleanly.
    assert _strip_clarify_image_placeholders("  [Image]  ") == ""
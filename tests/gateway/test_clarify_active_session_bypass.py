"""Regression tests for clarify replies while a gateway session is busy."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
)
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key


class _ClarifyBypassAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="text")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "private"}


def _event(text="custom answer"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="private",
            user_id="user1",
        ),
        message_id="msg1",
    )


def _clear_clarify_state():
    from tools import clarify_gateway as cm

    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()


@pytest.mark.asyncio
async def test_active_session_routes_typed_choice_clarify_reply_to_runner_not_busy_queue():
    """Typed text must resolve a pending choice clarify even while the agent is busy.

    Telegram button clarifies keep the adapter session active while the agent
    thread blocks on ``wait_for_response``.  If the adapter only bypasses for
    entries already marked ``awaiting_text``, typed replies to the visible
    multi-choice prompt are handled as busy follow-ups and the clarify wait is
    never resolved.
    """
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    adapter = _ClarifyBypassAdapter()
    adapter._message_handler = AsyncMock(return_value="")
    adapter._busy_session_handler = AsyncMock(return_value=True)
    event = _event("None of those are valid options")
    session_key = build_session_key(
        event.source,
        group_sessions_per_user=adapter.config.extra.get("group_sessions_per_user", True),
        thread_sessions_per_user=adapter.config.extra.get("thread_sessions_per_user", False),
    )
    adapter._active_sessions[session_key] = asyncio.Event()
    cm.register("clarify-1", session_key, "Pick one", ["A", "B"])

    await adapter.handle_message(event)

    adapter._message_handler.assert_awaited_once_with(event)
    adapter._busy_session_handler.assert_not_awaited()
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_active_session_bypass_uses_profile_namespaced_key_under_multiplex():
    """Regression for issue #82975: under a named-profile multiplex, the
    adapter's clarify bypass lookup must use the SAME profile-namespaced
    session key that the runner registers pending clarifies under
    (SessionStore._generate_session_key() includes
    profile=self._resolve_profile_for_key(source)), not the legacy
    unnamespaced key. Otherwise the lookup misses, and a user's answer to
    a pending clarify is routed to the busy-session queue instead of
    resolving it -- the turn then hangs until the clarify's 3600s timeout."""
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    adapter = _ClarifyBypassAdapter()
    adapter._message_handler = AsyncMock(return_value="")
    adapter._busy_session_handler = AsyncMock(return_value=True)
    event = _event("None of those are valid options")

    # A session_store configured for profile multiplexing, matching what
    # the runner's SessionStore._generate_session_key() actually produces.
    session_store = MagicMock()
    session_store._resolve_profile_for_key.return_value = "ops"
    adapter._session_store = session_store

    profile_namespaced_key = build_session_key(
        event.source,
        group_sessions_per_user=adapter.config.extra.get("group_sessions_per_user", True),
        thread_sessions_per_user=adapter.config.extra.get("thread_sessions_per_user", False),
        profile="ops",
    )
    # Sanity: the profile-namespaced key really is different from the
    # legacy unnamespaced one -- otherwise this test wouldn't distinguish
    # the fixed behavior from the bug.
    legacy_key = build_session_key(
        event.source,
        group_sessions_per_user=adapter.config.extra.get("group_sessions_per_user", True),
        thread_sessions_per_user=adapter.config.extra.get("thread_sessions_per_user", False),
    )
    assert profile_namespaced_key != legacy_key

    adapter._active_sessions[profile_namespaced_key] = asyncio.Event()
    # The runner registers the pending clarify under its own
    # profile-namespaced key, exactly as it would in a real multiplexed
    # deployment.
    cm.register("clarify-1", profile_namespaced_key, "Pick one", ["A", "B"])

    await adapter.handle_message(event)

    adapter._message_handler.assert_awaited_once_with(event)
    adapter._busy_session_handler.assert_not_awaited()
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_gateway_clarify_document_reply_reaches_waiting_agent_with_file_context(
    tmp_path, monkeypatch
):
    """A PDF attached as an open-ended clarify answer must not collapse to its ID.

    Slack existing-file shares can carry a short ``F…`` text body plus a cached
    document path. The clarify intercept resolves the waiting tool directly and
    bypasses the normal inbound preprocessing path, so it must preserve the same
    document context note the agent would receive on an ordinary turn.
    """
    _clear_clarify_state()
    from gateway.run import GatewayRunner
    from tools import clarify_gateway as cm

    monkeypatch.setenv("TERMINAL_ENV", "local")
    pdf_path = tmp_path / "doc_abc_SIE_Regions_Offline_Reference.pdf"
    pdf_path.write_bytes(b"%PDF-1.7 fake")

    adapter = _ClarifyBypassAdapter()
    event = _event("F0BKY3BG6F4")
    event.message_type = MessageType.DOCUMENT
    event.media_urls = [str(pdf_path)]
    event.media_types = ["application/pdf"]

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._startup_restore_in_progress = False
    runner._scale_to_zero_note_real_inbound = lambda: None
    runner._is_user_authorized = lambda source: True
    runner._session_key_for_source = lambda source: "clarify-document-session"
    runner._adapter_for_source = lambda source: adapter
    runner._update_prompt_pending = {}

    cm.register(
        "clarify-document",
        "clarify-document-session",
        "Please attach the PDF",
        None,
    )

    with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
        result = await runner._handle_message(event)

    assert result == ""
    with cm._lock:
        entry = cm._entries.get("clarify-document")
    assert entry is not None
    assert entry.event.is_set()
    assert entry.response is not None
    assert "F0BKY3BG6F4" in entry.response
    assert "[The user sent a document:" in entry.response
    assert "SIE_Regions_Offline_Reference.pdf" in entry.response
    assert str(pdf_path) in entry.response


@pytest.mark.asyncio
async def test_gateway_clarify_photo_reply_carries_the_image_path_but_a_choice_pick_stays_bare(tmp_path):
    """A photo answering an open-ended clarify reaches the agent as a path it can look at; a photo
    captioned with a choice label on a choice prompt still answers with just that label."""
    _clear_clarify_state()
    from gateway.run import GatewayRunner
    from tools import clarify_gateway as cm

    png = tmp_path / "img_abc_bug.png"
    png.write_bytes(b"\x89PNG fake")
    adapter = _ClarifyBypassAdapter()
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._startup_restore_in_progress = False
    runner._scale_to_zero_note_real_inbound = lambda: None
    runner._is_user_authorized = lambda source: True
    runner._session_key_for_source = lambda source: "clarify-photo-session"
    runner._adapter_for_source = lambda source: adapter
    runner._update_prompt_pending = {}

    def photo(caption):
        event = _event(caption)
        event.message_type = MessageType.PHOTO
        event.media_urls, event.media_types = [str(png)], ["image/png"]
        return event

    for cid, choices, caption in (("open", None, ""), ("pick", ["staging", "prod"], "prod")):
        _clear_clarify_state()  # no waiting tool pops the settled entry here
        cm.register(cid, "clarify-photo-session", "Which one?", choices)
        with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
            assert await runner._handle_message(photo(caption)) == ""
        with cm._lock:
            entry = cm._entries.get(cid)
        assert entry is not None and entry.event.is_set()
        if choices:
            assert entry.response == "prod"
        else:
            assert "[The user sent an image." in entry.response and str(png) in entry.response

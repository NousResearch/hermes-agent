"""WhatsApp (Baileys) reaction behavior.

Verifies the adapter implements the unified reaction contract by POSTing to the
bridge ``/react`` endpoint with the right payload, and that ``remove_reaction``
clears by sending an empty emoji. The bridge resolves the original message from
its TTL'd message store and calls
``sock.sendMessage(jid, { react: { text, key } })`` — the same store and endpoint
the ``whatsapp_action`` tool reacts through.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from gateway.config import Platform


class _AsyncCM:
    """Minimal async context manager returning a fixed value."""

    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *exc):
        return False


def _make_adapter():
    """Create a connected WhatsAppAdapter with a mocked HTTP session."""
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter._bridge_port = 19876
    adapter._running = True
    adapter._bridge_process = None
    # _check_managed_bridge_exit() returns falsy when we don't manage a proc
    adapter._http_session = MagicMock()
    return adapter


def _mock_post(status=200, text="ok"):
    resp = MagicMock()
    resp.status = status
    resp.text = AsyncMock(return_value=text)
    post = MagicMock(return_value=_AsyncCM(resp))
    return post


def test_supports_reactions_flag():
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    assert WhatsAppAdapter.SUPPORTS_REACTIONS is True


def test_add_reaction_posts_emoji():
    adapter = _make_adapter()
    post = _mock_post()
    adapter._http_session.post = post

    res = asyncio.new_event_loop().run_until_complete(
        adapter.add_reaction("12345@s.whatsapp.net", "👍", "MSGID1")
    )

    assert res["success"] is True
    url = post.call_args[0][0]
    payload = post.call_args.kwargs["json"]
    assert url.endswith("/react")
    assert payload["messageId"] == "MSGID1"
    assert payload["emoji"] == "👍"
    assert payload["chatId"].endswith("@s.whatsapp.net")


def test_remove_reaction_sends_empty_emoji():
    adapter = _make_adapter()
    post = _mock_post()
    adapter._http_session.post = post

    res = asyncio.new_event_loop().run_until_complete(
        adapter.remove_reaction("12345@s.whatsapp.net", "MSGID1")
    )

    assert res["success"] is True
    assert post.call_args.kwargs["json"]["emoji"] == ""


def test_react_requires_message_id():
    adapter = _make_adapter()
    adapter._http_session.post = _mock_post()

    res = asyncio.new_event_loop().run_until_complete(
        adapter.add_reaction("12345@s.whatsapp.net", "👍", None)
    )

    assert res["success"] is False
    assert "message_id" in res["error"]


def test_react_propagates_bridge_error():
    adapter = _make_adapter()
    adapter._http_session.post = _mock_post(status=503, text="Not connected to WhatsApp")

    res = asyncio.new_event_loop().run_until_complete(
        adapter.add_reaction("12345@s.whatsapp.net", "👍", "MSGID1")
    )

    assert res["success"] is False
    assert "Not connected" in res["error"]


def test_react_propagates_uncached_message_404():
    # The bridge returns 404 when the target message isn't in its store
    # (resolveStoredMessage miss); the adapter surfaces that as a failure.
    adapter = _make_adapter()
    adapter._http_session.post = _mock_post(
        status=404, text="Referenced message not found in bridge cache"
    )

    res = asyncio.new_event_loop().run_until_complete(
        adapter.add_reaction("12345@s.whatsapp.net", "👍", "MSGID1")
    )

    assert res["success"] is False
    assert "not found" in res["error"]


def test_react_not_connected_short_circuits():
    adapter = _make_adapter()
    adapter._running = False
    adapter._http_session.post = _mock_post()

    res = asyncio.new_event_loop().run_until_complete(
        adapter.add_reaction("12345@s.whatsapp.net", "👍", "MSGID1")
    )

    assert res["success"] is False

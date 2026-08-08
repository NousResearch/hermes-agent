"""Seam-identity + aggressive tests for the R5-S1 standalone-sender extraction.

The 13-member C4 cluster (window 9405-9797) moved byte-verbatim from
``plugins/platforms/discord/adapter.py`` into
``plugins/platforms/discord/standalone_send.py``; the adapter re-exports every
name.  These tests pin the seam: adapter-namespace access must resolve to the
SAME objects the new module owns (including the mutable probe-cache dict), the
registry hook ``standalone_sender_fn`` must keep resolving through the
re-export, and the moved sender must behave identically (success, HTTP error,
forum-thread paths).
"""
import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import plugins.platforms.discord.adapter as adapter
import plugins.platforms.discord.standalone_send as standalone_send

MOVED_NAMES = [
    "_DISCORD_CHANNEL_TYPE_PROBE_CACHE",
    "_DISCORD_STANDALONE_JSON_BODY_LIMIT_BYTES",
    "_DISCORD_STANDALONE_ERROR_BODY_LIMIT_BYTES",
    "_remember_channel_is_forum",
    "_probe_is_forum_cached",
    "_derive_forum_thread_name",
    "_standalone_sanitize_error",
    "_standalone_close_response",
    "_standalone_read_response_bytes_limited",
    "_standalone_response_encoding",
    "_standalone_read_text_limited",
    "_standalone_read_json_limited",
    "_standalone_send",
]


@pytest.mark.parametrize("name", MOVED_NAMES)
def test_seam_identity_every_moved_name(name):
    """Adapter re-exports the exact objects the new module owns."""
    assert getattr(adapter, name) is getattr(standalone_send, name)


def test_probe_cache_shared_dict_across_namespaces():
    """The mutable probe cache is ONE dict object behind both namespaces."""
    assert adapter._DISCORD_CHANNEL_TYPE_PROBE_CACHE is standalone_send._DISCORD_CHANNEL_TYPE_PROBE_CACHE
    cache = adapter._DISCORD_CHANNEL_TYPE_PROBE_CACHE
    cache.clear()
    try:
        # Written through the adapter namespace, read through the module.
        adapter._remember_channel_is_forum("ch1", True)
        assert standalone_send._probe_is_forum_cached("ch1") is True
        # Written through the module namespace, read through the adapter.
        standalone_send._remember_channel_is_forum("ch2", False)
        assert adapter._probe_is_forum_cached("ch2") is False
    finally:
        cache.clear()


def test_register_wires_standalone_sender_fn_through_seam():
    """register()'s hook must still resolve to the moved object (loader contract)."""
    captured = {}

    class FakeCtx:
        @staticmethod
        def register_platform(**kwargs):
            captured.update(kwargs)

    adapter.register(FakeCtx())
    assert captured["standalone_sender_fn"] is standalone_send._standalone_send
    assert captured["standalone_sender_fn"] is adapter._standalone_send
    assert captured["name"] == "discord"


def _build_mock_chain(status, response_data=None, response_text="error body"):
    """Properly-structured aiohttp mock chain (session.post -> async CM -> resp)."""
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.json = AsyncMock(return_value=response_data or {"id": "msg123"})
    mock_resp.text = AsyncMock(return_value=response_text)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_session.post = MagicMock(return_value=mock_resp)
    return mock_session, mock_resp


def _no_channel_type(platform, chat_id):
    return None


def test_standalone_send_success_plain_message(monkeypatch):
    """Plain text send hits /channels/{id}/messages and returns success."""
    monkeypatch.setattr("gateway.channel_directory.lookup_channel_type", _no_channel_type)
    mock_session, mock_resp = _build_mock_chain(200, {"id": "msg42"})
    with patch("aiohttp.ClientSession", return_value=mock_session):
        result = asyncio.run(
            standalone_send._standalone_send(
                SimpleNamespace(token="tok"), "111222333", "hello from cron"
            )
        )

    assert result == {
        "success": True,
        "platform": "discord",
        "chat_id": "111222333",
        "message_id": "msg42",
    }
    call_url = mock_session.post.call_args.args[0]
    assert call_url == "https://discord.com/api/v10/channels/111222333/messages"
    sent_json = mock_session.post.call_args.kwargs["json"]
    assert sent_json == {"content": "hello from cron"}


def test_standalone_send_http_error_path(monkeypatch):
    """Non-2xx response returns a sanitized error dict, no exception raised."""
    monkeypatch.setattr("gateway.channel_directory.lookup_channel_type", _no_channel_type)
    mock_session, _ = _build_mock_chain(403, response_text="Forbidden: no perms")
    with patch("aiohttp.ClientSession", return_value=mock_session):
        result = asyncio.run(
            standalone_send._standalone_send(
                SimpleNamespace(token="tok"), "111222333", "hello"
            )
        )

    assert result == {"error": "Discord API error (403): Forbidden: no perms"}


def test_standalone_send_forum_thread_json_path(monkeypatch):
    """Forum channels (type 15) go to /channels/{id}/threads with a thread name."""
    monkeypatch.setattr(
        "gateway.channel_directory.lookup_channel_type", lambda platform, chat_id: "forum"
    )
    mock_session, _ = _build_mock_chain(200, {"id": "thread9", "message": {"id": "starter1"}})
    with patch("aiohttp.ClientSession", return_value=mock_session):
        result = asyncio.run(
            standalone_send._standalone_send(
                SimpleNamespace(token="tok"), "111222333", "New post title"
            )
        )

    assert result["success"] is True
    assert result["thread_id"] == "thread9"
    assert result["message_id"] == "starter1"
    call_url = mock_session.post.call_args.args[0]
    assert call_url == "https://discord.com/api/v10/channels/111222333/threads"
    sent_json = mock_session.post.call_args.kwargs["json"]
    assert sent_json["name"] == "New post title"
    assert sent_json["message"] == {"content": "New post title"}


def test_standalone_send_token_fallback_to_secret_scope(monkeypatch):
    """Empty pconfig.token falls back to agent.secret_scope.get_secret."""
    monkeypatch.setattr("gateway.channel_directory.lookup_channel_type", _no_channel_type)
    mock_session, _ = _build_mock_chain(200, {"id": "msg7"})
    with patch("aiohttp.ClientSession", return_value=mock_session):
        with patch("agent.secret_scope.get_secret", return_value="scoped-token") as get_secret:
            result = asyncio.run(
                standalone_send._standalone_send(
                    SimpleNamespace(token=""), "111222333", "hi"
                )
            )

    assert result["success"] is True
    get_secret.assert_called_once_with("DISCORD_BOT_TOKEN", "")
    auth_header = mock_session.post.call_args.kwargs["headers"]["Authorization"]
    assert auth_header == "Bot scoped-token"

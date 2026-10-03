"""Slack thread-root file recovery: images and documents, verified human senders only."""

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_slack_mock():
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return
    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock
    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock
    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        ("slack_bolt.adapter.socket_mode.async_handler", slack_bolt.adapter.socket_mode.async_handler),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)


_ensure_slack_mock()

import plugins.platforms.slack.adapter as _slack_mod  # noqa: E402

_slack_mod.SLACK_AVAILABLE = True

from plugins.platforms.slack.adapter import SlackAdapter, _ThreadContextCache  # noqa: E402

CH, TS, TEAM = "C1", "100.000", "T1"
PDF = {"id": "F1", "name": "quote.pdf", "mimetype": "application/pdf", "size": 1200,
       "url_private_download": "https://files.slack.com/quote.pdf"}
PNG = {"id": "F2", "name": "roof.png", "mimetype": "image/png",
       "url_private_download": "https://files.slack.com/roof.png"}


def _adapter(root, users_info):
    a = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test"))
    a._app = MagicMock()
    a._thread_context_cache[a._thread_cache_key(CH, TS, TEAM)] = _ThreadContextCache(
        content="", messages=[root])
    a._users_info_payload = AsyncMock(return_value=users_info)
    a._cache_slack_file = AsyncMock(side_effect=lambda kind, f, *_: (f"/cache/{f['name']}", f["mimetype"], ""))
    a._authorization_check = None  # no registered check: human verification decides
    return a


def _human(uid="U1"):
    return {"ok": True, "user": {"id": uid, "is_bot": False, "profile": {"display_name": "Nathan"}}}


@pytest.mark.asyncio
async def test_verified_human_root_delivers_images_and_documents():
    a = _adapter({"ts": TS, "user": "U1", "files": [PNG, PDF]}, _human())
    urls, types = await a._collect_thread_root_images(CH, TS, TEAM)
    assert urls == ["/cache/roof.png", "/cache/quote.pdf"]
    assert types == ["image/png", "application/pdf"]


@pytest.mark.asyncio
@pytest.mark.parametrize("root,info", [
    ({"ts": TS, "user": "U9", "bot_id": "B1", "files": [PDF]}, _human("U9")),
    ({"ts": TS, "subtype": "bot_message", "user": "U9", "files": [PDF]}, _human("U9")),
    ({"ts": TS, "files": [PDF]}, _human()),
    ({"ts": TS, "user": "U2", "files": [PDF]}, {"ok": True, "user": {"id": "U2", "is_bot": True}}),
    ({"ts": TS, "user": "U3", "files": [PDF]}, {}),
    ({"ts": TS, "user": "U4", "files": [PDF]}, {"ok": True, "user": {"id": "U4", "deleted": True}}),
])
async def test_unverified_root_sender_loads_no_documents(root, info):
    a = _adapter(root, info)
    assert await a._collect_thread_root_images(CH, TS, TEAM) == ([], [])
    a._cache_slack_file.assert_not_called()


@pytest.mark.asyncio
async def test_unauthorized_human_root_loads_no_documents():
    a = _adapter({"ts": TS, "user": "U7", "files": [PDF]}, _human("U7"))
    a._authorization_check = lambda *args, **kwargs: False  # e.g. Slack Connect guest
    assert await a._collect_thread_root_images(CH, TS, TEAM) == ([], [])


@pytest.mark.asyncio
async def test_bot_root_images_keep_upstream_behavior():
    a = _adapter({"ts": TS, "user": "U9", "bot_id": "B1", "files": [PNG, PDF]}, _human("U9"))
    assert await a._collect_thread_root_images(CH, TS, TEAM) == (["/cache/roof.png"], ["image/png"])


@pytest.mark.asyncio
async def test_users_info_failure_fails_closed():
    a = _adapter({"ts": TS, "user": "U1", "files": [PDF]}, _human())
    a._users_info_payload = AsyncMock(side_effect=RuntimeError("rate limited"))
    assert await a._collect_thread_root_images(CH, TS, TEAM) == ([], [])


@pytest.mark.asyncio
async def test_audio_and_video_roots_are_not_recovered():
    audio = {"id": "F3", "name": "a.mp3", "mimetype": "audio/mpeg", "url_private": "https://files.slack.com/a"}
    a = _adapter({"ts": TS, "user": "U1", "files": [audio]}, _human())
    assert await a._collect_thread_root_images(CH, TS, TEAM) == ([], [])

"""Matrix must gate (require_mention / allowed rooms) BEFORE downloading inbound media.

An unmentioned ``m.image`` in a gated group room used to be fetched from the homeserver
(``_download_and_cache_media``) and only then dropped by ``_resolve_message_context``.
The invariant: a dropped media event performs zero downloads; a mentioned one still does.
"""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def _make_adapter(monkeypatch):
    monkeypatch.setenv("MATRIX_REQUIRE_MENTION", "true")
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")
    from gateway.config import PlatformConfig
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter = MatrixAdapter(PlatformConfig(
        enabled=True, token="syt_test_token",
        extra={"homeserver": "https://matrix.example.org", "user_id": "@hermes:example.org"}))
    adapter._startup_ts = time.time() - 10
    adapter.handle_message = AsyncMock()
    adapter._client = None
    adapter._resolve_room_identity = AsyncMock(return_value=SimpleNamespace(
        display_name="Group Room", room_topic=None, server_name="example.org", chat_type="group",
        members_digest=None))
    adapter._is_dm_room = AsyncMock(return_value=False)
    adapter._download_and_cache_media = AsyncMock(return_value="/tmp/cached.png")
    return adapter


def _image_event(body):
    return SimpleNamespace(
        sender="@alice:example.org", event_id="$img1", room_id="!group:example.org",
        timestamp=int(time.time() * 1000),
        content={"body": body, "msgtype": "m.image", "url": "mxc://example.org/abc",
                 "info": {"mimetype": "image/png", "size": 1024}})


@pytest.mark.asyncio
@pytest.mark.parametrize("body, downloads, dispatched", [
    ("photo.png", 0, 0),           # unmentioned group media: never fetched
    ("@hermes:example.org look", 1, 1),  # mentioned: fetched and dispatched
])
async def test_unmentioned_group_media_is_not_downloaded(monkeypatch, body, downloads, dispatched):
    adapter = _make_adapter(monkeypatch)
    await adapter._on_room_message(_image_event(body))
    assert adapter._download_and_cache_media.await_count == downloads
    assert adapter.handle_message.await_count == dispatched


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario, msgtype, consumed, expected_type, expected_text, expected_bytes", [
    ("failure", "m.file", True, "text", "caption\n[matrix file attachment could not be downloaded: report.txt]", None),
    ("failure", "m.file", False, "text", "caption\n[matrix file attachment could not be downloaded: report.txt]", None),
    ("failure", "m.sticker", True, "text", "[sticker: caption]\n[matrix sticker attachment could not be downloaded: report.txt]", None),
    ("declared-cap", "m.file", True, "text", "caption\n[matrix file attachment too large: report.txt]", None),
    ("stream-cap", "m.file", True, "text", "caption\n[matrix file attachment too large: report.txt]", None),
    ("empty", "m.file", True, "document", "caption", b""),
    ("empty", "m.audio", True, "audio", "caption", b""),
    ("no-url", "m.file", True, "document", "caption", None),
    ("no-url", "m.audio", True, "audio", "caption", None),
])
async def test_media_failure_preserves_admission_and_retained_reply_context(
    monkeypatch, scenario, msgtype, consumed, expected_type, expected_text, expected_bytes,
):
    from pathlib import Path
    from unittest.mock import MagicMock

    from gateway.config import Platform
    from gateway.platforms.event import MessageType
    from gateway.session import SessionSource
    from plugins.platforms.matrix.adapter import MatrixAdapter
    from plugins.platforms.matrix.reply_context import MatrixEventContext, MatrixReplyContext
    from plugins.platforms.matrix.turn_context import MatrixTurnContext
    from tests.gateway.matrix_helpers import FakeMediaDownload

    adapter = _make_adapter(monkeypatch)
    del adapter._download_and_cache_media
    adapter._client = MagicMock()
    adapter._max_media_bytes = 4
    adapter._message_handler = AsyncMock() if consumed else None
    download = FakeMediaDownload(
        b"" if scenario == "empty" else b"12345", fail=scenario == "failure",
    ).install(adapter._client)
    source = SessionSource(
        platform=Platform.MATRIX, chat_id="!group:example.org", chat_type="group",
        user_id="@alice:example.org", user_name="Alice",
    )
    parent = MatrixEventContext("@bob:example.org", "earlier", event_id="$parent")
    adapter._resolve_message_context = AsyncMock(return_value=(
        "caption", False, "group", None, "Alice", True, source,
    ))
    adapter._extract_reply_context = AsyncMock(return_value=MatrixReplyContext(
        "caption", "$parent", "earlier", "@bob:example.org", "Bob", False, True,
    ))
    content = {
        "msgtype": msgtype, "body": "caption", "filename": "report.txt",
        "info": {"mimetype": "audio/ogg" if msgtype == "m.audio" else "text/plain"},
    }
    if scenario != "no-url":
        content["url"] = "mxc://example.org/file"
    if scenario == "declared-cap":
        content["info"]["size"] = 5
    result = await MatrixAdapter._handle_media_message(
        adapter, "!group:example.org", "@alice:example.org", "$attachment", 0.0,
        content, {"m.in_reply_to": {"event_id": "$parent"}}, msgtype,
        mention_claimed=True, reply_parent=parent,
    )
    event = adapter.handle_message.await_args.args[0]
    snapshots = event._inbound_context_dependencies
    files = [(Path(path).is_file(), Path(path).read_bytes()) for path in event.media_urls]
    observed = (
        result, event.text, event.message_type, event.source, event.reply_to_message_id,
        event.reply_to_text, event.reply_to_author_id, event.reply_to_author_name,
        event.reply_to_is_own_message, event.reply_to_author_authorized, event.metadata,
        event.media_types, files, tuple(snapshot.parent for snapshot in snapshots if isinstance(snapshot, MatrixTurnContext)),
        download.requested,
    )
    expected_media = ["audio/ogg" if msgtype == "m.audio" else "text/plain"] if expected_bytes is not None else []
    expected_downloads = [] if scenario in {"no-url", "declared-cap"} else ["mxc://example.org/file"]
    assert observed == (
        consumed, expected_text, MessageType(expected_type), source, "$parent", "earlier",
        "@bob:example.org", "Bob", False, True,
        {"matrix_mention_claimed": True, "matrix_requires_mention": True},
        expected_media, [(True, expected_bytes)] if expected_bytes is not None else [],
        (parent, parent) if msgtype == "m.sticker" else (parent,), expected_downloads,
    )

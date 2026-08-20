from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
from tests.gateway.test_whatsapp_formatting import _AsyncCM, _make_adapter


class TestWhatsAppNativeFormatting:

    def test_invisible_unicode_prefixes_are_sanitized(self):
        adapter = _make_adapter()

        assert adapter.format_message("\u2060\u202ftext") == " text"


@pytest.mark.asyncio
async def test_send_location_posts_to_bridge_location_endpoint():
    adapter = _make_adapter()
    resp = MagicMock(status=200)
    resp.json = AsyncMock(return_value={"success": True, "messageId": "loc-msg"})
    adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

    result = await adapter.send_location(
        "15551234567",
        41.015,
        28.979,
        name="HQ",
        address="Example Street",
    )

    assert result.success
    assert result.message_id == "loc-msg"
    call = adapter._http_session.post.call_args
    assert call.args[0] == "http://127.0.0.1:3000/send-location"
    assert call.kwargs["json"] == {
        "chatId": "15551234567@s.whatsapp.net",
        "latitude": 41.015,
        "longitude": 28.979,
        "name": "HQ",
        "address": "Example Street",
    }


@pytest.mark.asyncio
async def test_send_multiple_images_posts_one_native_album_request(tmp_path):
    adapter = _make_adapter()
    session = adapter._http_session
    assert session is not None
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    resp = MagicMock(status=200)
    resp.json = AsyncMock(return_value={
        "success": True,
        "parentMessageId": "album-parent",
        "childMessageIds": ["child-1", "child-2"],
    })
    session.post = MagicMock(return_value=_AsyncCM(resp))

    await adapter.send_multiple_images(
        "15551234567",
        [(first.as_uri(), ""), (second.as_uri(), "second caption")],
    )

    session.post.assert_called_once()
    call = session.post.call_args
    assert call.args[0] == "http://127.0.0.1:3000/send-album"
    assert call.kwargs["json"] == {
        "chatId": "15551234567@s.whatsapp.net",
        "items": [
            {"filePath": str(first), "mediaType": "image"},
            {"filePath": str(second), "mediaType": "image", "caption": "second caption"},
        ],
    }


@pytest.mark.asyncio
async def test_send_multiple_images_falls_back_when_album_preflight_was_not_attempted(tmp_path):
    adapter = _make_adapter()
    session = adapter._http_session
    assert session is not None
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    validation_resp = MagicMock(status=400)
    validation_resp.json = AsyncMock(return_value={
        "success": False,
        "attempted": False,
        "status": "validation_error",
        "error": "invalid album",
    })
    media_resp = MagicMock(status=200)
    media_resp.json = AsyncMock(return_value={"success": True, "messageId": "child"})
    session.post = MagicMock(side_effect=[
        _AsyncCM(validation_resp),
        _AsyncCM(media_resp),
        _AsyncCM(media_resp),
    ])

    await adapter.send_multiple_images(
        "15551234567",
        [(first.as_uri(), ""), (second.as_uri(), "")],
    )

    urls = [call.args[0] for call in session.post.call_args_list]
    assert urls == [
        "http://127.0.0.1:3000/send-album",
        "http://127.0.0.1:3000/send-media",
        "http://127.0.0.1:3000/send-media",
    ]


@pytest.mark.asyncio
async def test_native_album_does_not_hold_global_queue_for_human_delay(tmp_path):
    adapter = _make_adapter()
    session = adapter._http_session
    assert session is not None
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    resp = MagicMock(status=200)
    resp.json = AsyncMock(return_value={"success": True})
    session.post = MagicMock(return_value=_AsyncCM(resp))

    await adapter.send_multiple_images(
        "15551234567",
        [(first.as_uri(), ""), (second.as_uri(), "")],
        human_delay=2.5,
    )

    payload = session.post.call_args.kwargs["json"]
    assert "delayMs" not in payload


@pytest.mark.asyncio
async def test_native_album_falls_back_when_old_bridge_returns_non_json_404(tmp_path):
    adapter = _make_adapter()
    session = adapter._http_session
    assert session is not None
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    missing_route = MagicMock(status=404)
    missing_route.json = AsyncMock(side_effect=ValueError("not json"))
    media_resp = MagicMock(status=200)
    media_resp.json = AsyncMock(return_value={"success": True})
    session.post = MagicMock(side_effect=[
        _AsyncCM(missing_route),
        _AsyncCM(media_resp),
        _AsyncCM(media_resp),
    ])

    await adapter.send_multiple_images(
        "15551234567",
        [(first.as_uri(), ""), (second.as_uri(), "")],
    )

    assert [call.args[0] for call in session.post.call_args_list] == [
        "http://127.0.0.1:3000/send-album",
        "http://127.0.0.1:3000/send-media",
        "http://127.0.0.1:3000/send-media",
    ]


@pytest.mark.asyncio
async def test_native_album_preserves_windows_file_uri_paths():
    adapter = _make_adapter()
    session = adapter._http_session
    assert session is not None
    resp = MagicMock(status=200)
    resp.json = AsyncMock(return_value={"success": True})
    session.post = MagicMock(return_value=_AsyncCM(resp))

    with patch("plugins.platforms.whatsapp.adapter.os.path.exists", return_value=True):
        await adapter.send_multiple_images(
            "15551234567",
            [
                ("file://C%3A%5Cphotos%5Cone.jpg", ""),
                ("file://C%3A%5Cphotos%5Ctwo.jpg", ""),
            ],
        )

    items = session.post.call_args.kwargs["json"]["items"]
    assert [item["filePath"] for item in items] == [
        "C:\\photos\\one.jpg",
        "C:\\photos\\two.jpg",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "body"),
    [
        (207, {"success": False, "attempted": True, "status": "partial_failure"}),
        (502, {"success": False, "attempted": True, "status": "parent_failure"}),
    ],
)
async def test_attempted_album_failures_never_fall_back_to_individual_media(
    tmp_path, status, body,
):
    adapter = _make_adapter()
    session = adapter._http_session
    assert session is not None
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    resp = MagicMock(status=status)
    resp.json = AsyncMock(return_value=body)
    session.post = MagicMock(return_value=_AsyncCM(resp))

    await adapter.send_multiple_images(
        "15551234567",
        [(first.as_uri(), ""), (second.as_uri(), "")],
    )

    session.post.assert_called_once()
    assert session.post.call_args.args[0] == "http://127.0.0.1:3000/send-album"



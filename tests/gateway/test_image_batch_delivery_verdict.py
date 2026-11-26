"""Regression for #160210: a text link is not a delivered image."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from plugins.platforms.mattermost.adapter import MattermostAdapter
from plugins.platforms.slack.adapter import SlackAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", ["slack", "mattermost"])
@pytest.mark.parametrize("with_image", [False, True])
async def test_batch_success_requires_image_delivery(platform, with_image, monkeypatch, tmp_path):
    from tools import url_safety

    monkeypatch.setattr(url_safety, "is_safe_url", lambda _url: False)
    config = PlatformConfig(enabled=True, token="test-token")
    if platform == "slack":
        adapter = SlackAdapter(config)
        adapter._app = MagicMock()
        adapter._dm_target = AsyncMock(return_value="channel")
        upload = AsyncMock()
        client = MagicMock()
        client.files_upload_v2 = upload
        adapter._client_for = MagicMock(return_value=client)
        adapter._record_uploaded_file_thread = MagicMock()
    else:
        adapter = MattermostAdapter(config)
        upload = AsyncMock(return_value="image-id")
        adapter._upload_file = upload
        adapter._post_message = AsyncMock(return_value={"id": "image-post"})
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="link"))
    url = "http://169.254.169.254/latest/meta-data/"

    images = [(url, "metadata")]
    if with_image:
        image = tmp_path / "image.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n")
        images.append((image.as_uri(), "picture"))

    result = await adapter.send_multiple_images("channel", images)

    adapter.send.assert_awaited_once_with("channel", f"metadata\n{url}", metadata=None)
    if with_image:
        upload.assert_awaited_once()
    else:
        upload.assert_not_awaited()
    assert result.success is with_image
    assert result.error == (None if with_image else "all images failed to send")

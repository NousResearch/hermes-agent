from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.media_publisher import (
    PublishedMedia,
    external_media_config_for,
    format_published_media_message,
)
from plugins.platforms.discord.adapter import DiscordAdapter


def test_gateway_config_media_delivery_defaults_off():
    cfg = GatewayConfig.from_dict({})

    assert cfg.media_delivery == {}
    assert cfg.to_dict()["media_delivery"] == {}


def test_gateway_config_loads_media_delivery_block():
    raw = {
        "media_delivery": {
            "external_upload": {
                "enabled": True,
                "provider": "r2",
                "mode": "link_first",
                "platforms": {"discord": {"images": True}},
            }
        },
        "platforms": {"discord": {"enabled": True, "token": "x"}},
    }

    cfg = GatewayConfig.from_dict(raw)

    assert cfg.media_delivery["external_upload"]["enabled"] is True
    assert cfg.media_delivery["external_upload"]["mode"] == "link_first"
    assert Platform.DISCORD in cfg.platforms


def test_external_media_config_default_disabled_for_adapter():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    cfg = external_media_config_for(adapter, "images")

    assert cfg.enabled is False
    assert cfg.provider == "r2"


def test_external_media_config_merges_platform_override():
    adapter = DiscordAdapter(
        PlatformConfig(
            enabled=True,
            token="***",
            extra={
                "media_delivery": {
                    "external_upload": {
                        "enabled": True,
                        "provider": "r2",
                        "mode": "link_on_failure",
                        "images": False,
                        "public_base_url": "https://cdn.example.com",
                        "platforms": {
                            "discord": {
                                "mode": "link_first",
                                "images": True,
                            }
                        },
                    }
                }
            },
        )
    )

    cfg = external_media_config_for(adapter, "images")

    assert cfg.enabled is True
    assert cfg.mode == "link_first"
    assert cfg.images is True
    assert cfg.public_base_url == "https://cdn.example.com"


def test_format_published_media_message_multiple_items():
    msg = format_published_media_message(
        [
            PublishedMedia("/tmp/a.png", "https://cdn.example/a.png", "a.png", "image/png", 1),
            PublishedMedia("/tmp/b.png", "https://cdn.example/b.png", "b.png", "image/png", 2),
        ],
        heading="Images",
    )

    assert "Images (2 files):" in msg
    assert "a.png: https://cdn.example/a.png" in msg
    assert "b.png: https://cdn.example/b.png" in msg


@pytest.mark.asyncio
async def test_discord_send_image_file_link_first_uses_published_url(monkeypatch, tmp_path):
    image = tmp_path / "photo.png"
    image.write_bytes(b"fake-png")
    adapter = DiscordAdapter(
        PlatformConfig(
            enabled=True,
            token="***",
            extra={
                "media_delivery": {
                    "external_upload": {
                        "enabled": True,
                        "mode": "link_first",
                        "provider": "r2",
                    }
                }
            },
        )
    )
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="42"))

    async def fake_publish(_adapter, paths, media_kind=None):
        assert _adapter is adapter
        assert list(paths) == [str(image)]
        assert media_kind == "images"
        return [PublishedMedia(str(image), "https://cdn.example/photo.png", "photo.png", "image/png", 8)]

    monkeypatch.setattr("plugins.platforms.discord.adapter.publish_media_files", fake_publish)

    result = await adapter.send_image_file("123", str(image), caption="caption")

    assert result.success is True
    adapter.send.assert_awaited_once()
    awaited = adapter.send.await_args
    assert awaited is not None
    sent_content = awaited.args[1]
    assert "caption" in sent_content
    assert "https://cdn.example/photo.png" in sent_content


@pytest.mark.asyncio
async def test_discord_send_image_file_default_uses_native_attachment(monkeypatch, tmp_path):
    image = tmp_path / "photo.png"
    image.write_bytes(b"fake-png")
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    native = AsyncMock(return_value=SimpleNamespace(success=True, message_id="99"))
    monkeypatch.setattr(adapter, "_send_file_attachment", native)

    result = await adapter.send_image_file("123", str(image), caption="caption")

    assert result.success is True
    native.assert_awaited_once_with("123", str(image), "caption")

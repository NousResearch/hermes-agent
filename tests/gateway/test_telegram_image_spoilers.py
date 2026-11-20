"""Telegram hides a ``||![alt](url)||`` image: ``has_spoiler`` blurs the photo (or GIF) until
tapped, and one spoiler entity hides its caption. A spoiler photo no native send delivered gets
the "couldn't deliver" notice, never the text link, whose preview would show it unhidden."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from plugins.platforms.telegram.adapter import TelegramAdapter

URL = "https://cdn.example.com/boss.png"


@pytest.fixture
def adapter(monkeypatch):
    import telegram
    monkeypatch.setattr(telegram, "MessageEntity", lambda **kw: SimpleNamespace(**kw), raising=False)
    monkeypatch.setattr(telegram, "InputMediaPhoto", lambda media, caption=None, **kw: {"media": media, "caption": caption, **kw},
                        raising=False)
    monkeypatch.setattr("tools.url_safety.is_safe_url", lambda _url: True)
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="123456:test-token", extra={}))
    adapter._bot = MagicMock()
    adapter._bot.send_photo = AsyncMock(return_value=SimpleNamespace(message_id=7))
    adapter._bot.send_animation = AsyncMock(return_value=SimpleNamespace(message_id=8))
    adapter._bot.send_media_group = AsyncMock(return_value=[SimpleNamespace(message_id=9)])
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("spoiler, hidden", [
    # UTF-16 units: the emoji is 2
    (True, {"has_spoiler": True, "caption_entities": [SimpleNamespace(type="spoiler", offset=0, length=7)]}),
    (False, {}),
])
async def test_only_a_spoiler_photo_has_spoiler_and_a_hidden_caption(adapter, spoiler, hidden):
    result = await adapter.send_image("123", URL, caption="Boss 🐉", spoiler=spoiler)

    assert result.success
    kwargs = adapter._bot.send_photo.await_args.kwargs
    assert kwargs["photo"] == URL and kwargs["caption"] == "Boss 🐉"
    assert {k: v for k, v in kwargs.items() if k in ("has_spoiler", "caption_entities")} == hidden


@pytest.mark.asyncio
async def test_spoiler_gif_has_spoiler(adapter):
    await adapter.send_animation("123", "https://cdn.example.com/twist.gif", spoiler=True)

    assert adapter._bot.send_animation.await_args.kwargs["has_spoiler"] is True


@pytest.mark.asyncio
async def test_album_hides_only_the_marked_photo(adapter):
    images, _ = BasePlatformAdapter.extract_images(
        "||![The ending](https://e.com/end.png)|| ![World map](https://e.com/map.png)")

    result = await adapter.send_multiple_images("123", images)

    assert result.success
    ending, world_map = adapter._bot.send_media_group.await_args.kwargs["media"]
    assert ending["has_spoiler"] is True and ending["caption_entities"][0].length == len("The ending")
    assert "has_spoiler" not in world_map and "caption_entities" not in world_map


@pytest.mark.asyncio
async def test_undelivered_spoiler_photo_never_posts_its_link(adapter, monkeypatch):
    """URL send refused, download failed: an unmarked photo falls back to the text link; a spoiler
    must not (Telegram would preview the image), so the notice goes out with the caption hidden."""
    import httpx
    from telegram.error import BadRequest

    adapter._bot.send_photo = AsyncMock(side_effect=BadRequest("Wrong type of the web page content"))
    monkeypatch.setattr("tools.url_safety.create_ssrf_safe_async_client",
                        MagicMock(side_effect=httpx.ConnectError("All connection attempts failed")))
    text_link = AsyncMock(return_value=SendResult(success=True))
    monkeypatch.setattr(BasePlatformAdapter, "send_image", text_link)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="t"))

    await adapter.send_image("123", URL, caption="Final boss", spoiler=True)

    text_link.assert_not_awaited()
    sent = [call.args[1] if len(call.args) > 1 else call.kwargs["content"] for call in adapter.send.await_args_list]
    assert sent and not any(URL in text for text in sent)
    assert all(text.count("Final boss") == text.count("||Final boss||") for text in sent)  # only ever hidden

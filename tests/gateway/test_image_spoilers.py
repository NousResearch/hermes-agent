"""``||![alt](url)||`` marks an image a spoiler (``gateway.platforms.base_image_tags``).

The bars around the whole tag are the spoiler syntax of Discord and Telegram. The image is still
extracted as an attachment; its alt carries the mark to the adapter, which hides the image where
the platform can (Discord ``SPOILER_`` attachments, Telegram ``has_spoiler``). Unmarked images,
and adapters that cannot hide one, behave exactly as before.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.base_image_tags import is_spoiler
from gateway.runtime_footer import caption_with_footer

extract = BasePlatformAdapter.extract_images


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text, url, alt, spoiler, cleaned", [
    ("The boss room:\n||![Final boss](https://cdn.example.com/boss.png)||\nGood luck.",
     "https://cdn.example.com/boss.png", "Final boss", True, "The boss room:\n\nGood luck."),
    ("Map: ![World map](https://cdn.example.com/map.png) done",
     "https://cdn.example.com/map.png", "World map", False, "Map:  done"),
])
def test_only_a_marked_image_is_extracted_as_a_spoiler_and_its_bars_leave_the_text(text, url, alt, spoiler, cleaned):
    images, rest = extract(text)

    assert images == [(url, alt)] and is_spoiler(images[0][1]) is spoiler
    assert spoiler or type(images[0][1]) is str
    assert rest == cleaned


def test_mixed_reply_keeps_order_and_marks_only_the_spoiler():
    images, cleaned = extract(
        "Route: ![route](https://e.com/route.png)\n"
        "Ending: ||![ending](https://e.com/ending.jpg)||\n"
        "Gear: ![gear](https://e.com/gear.webp)")

    assert [url for url, _ in images] == ["https://e.com/route.png", "https://e.com/ending.jpg", "https://e.com/gear.webp"]
    assert [is_spoiler(alt) for _, alt in images] == [False, True, False]
    assert "|" not in cleaned and "e.com" not in cleaned


def test_spaces_inside_the_bars_and_query_strings_are_allowed():
    url = "https://cdn.example.com/maps/area.png?width=800&sig=a1b2"
    images, cleaned = extract(f"See || ![Secret area]({url}) || below")

    assert images == [(url, "Secret area")] and is_spoiler(images[0][1])
    assert cleaned == "See  below"


@pytest.mark.parametrize("text, cleaned", [
    ("Read ||![the twist](https://example.com/wiki/Twist)|| later",
     "Read ||[the twist](<https://example.com/wiki/Twist>)|| later"),
    ("|| ![](https://example.com/render?id=7) ||", "||<https://example.com/render?id=7>||"),
    ("Read ![the twist](https://example.com/wiki/Twist) later", "Read ![the twist](https://example.com/wiki/Twist) later"),
])
def test_marked_tag_without_an_image_url_becomes_a_hidden_link_without_a_preview(text, cleaned):
    """Not an attachment, so it stays text; as written, Discord would unfurl the URL (even with
    the closing ``)||`` glued on) into a preview of the image, so it goes as ``[alt](<url>)``
    (``<url>`` alone without alt). An unmarked tag stays as written."""
    assert extract(text) == ([], cleaned)


def test_hidden_link_and_extracted_spoiler_in_one_reply():
    images, cleaned = extract(
        "Map: ||![map](https://e.com/map.png)||\nWiki: ||![wiki](https://e.com/wiki/Map)||")

    assert images == [("https://e.com/map.png", "map")] and is_spoiler(images[0][1])
    assert cleaned == "Map: \nWiki: ||[wiki](<https://e.com/wiki/Map>)||"


def test_bars_that_do_not_wrap_the_tag_alone_are_not_a_mark():
    """A text spoiler that merely contains a tag is not a mark: the bars must hug the tag."""
    images, cleaned = extract("||He returns ![x](https://e.com/x.png) at the end||")

    assert images == [("https://e.com/x.png", "x")] and not is_spoiler(images[0][1])
    assert cleaned == "||He returns  at the end||"


def test_one_url_marked_once_and_shown_once_keeps_each_flag():
    images, _ = extract("![a](https://e.com/a.png) and ||![a](https://e.com/a.png)||")

    assert [is_spoiler(alt) for _, alt in images] == [False, True]


# ---------------------------------------------------------------------------
# The base per-image loop forwards the mark only to a send_image that declares it
# ---------------------------------------------------------------------------


class _Adapter(BasePlatformAdapter):
    name = "stub"

    def __init__(self):
        self.calls = []

    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, **kwargs):
        return SendResult(success=True)

    async def get_chat_info(self, chat_id):
        return {}


class _SpoilerAdapter(_Adapter):
    async def send_image(self, chat_id, image_url, caption=None, reply_to=None, metadata=None, spoiler=False):
        self.calls.append((image_url, caption, spoiler))
        return SendResult(success=True, message_id="1")

    async def send_animation(self, chat_id, animation_url, caption=None, reply_to=None, metadata=None, spoiler=False):
        self.calls.append((animation_url, caption, spoiler))
        return SendResult(success=True, message_id="2")


class _LegacyAdapter(_Adapter):
    async def send_image(self, chat_id, image_url, caption=None, **kwargs):
        self.calls.append((image_url, caption, kwargs))
        return SendResult(success=True, message_id="1")


def test_base_loop_passes_spoiler_to_an_adapter_that_can_hide_images():
    adapter = _SpoilerAdapter()
    images, _ = extract("![open](https://e.com/a.png) ||![hidden](https://e.com/b.png)|| ||![gif](https://e.com/c.gif)||")

    result = asyncio.run(adapter.send_multiple_images("chat", images))

    assert result.success
    assert adapter.calls == [("https://e.com/a.png", "open", False), ("https://e.com/b.png", "hidden", True),
                             ("https://e.com/c.gif", "gif", True)]


def test_base_loop_calls_an_adapter_without_spoiler_support_as_before():
    adapter = _LegacyAdapter()
    images, _ = extract("||![hidden](https://e.com/b.png)||")

    asyncio.run(adapter.send_multiple_images("chat", images))

    assert adapter.calls == [("https://e.com/b.png", "hidden", {"metadata": None})]


# ---------------------------------------------------------------------------
# Delivery paths keep the mark
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_stream_delivery_hands_the_mark_to_the_adapter():
    """A streamed reply's images go out after the stream; the marked one must stay marked."""
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    adapter = SimpleNamespace(
        name="test", extract_media=BasePlatformAdapter.extract_media, extract_images=extract,
        send_multiple_images=AsyncMock(return_value=SendResult(success=True, message_id="imgs")))
    runner = SimpleNamespace(_thread_metadata_for_source=lambda source, anchor=None: {},
                             _reply_anchor_for_event=lambda event: None)
    event = MessageEvent(text="hi", message_type=MessageType.TEXT, message_id="1",
                         source=SessionSource(platform=Platform.DISCORD, chat_id="C1", chat_type="group"))

    await GatewayRunner._deliver_media_from_response(
        runner, "Spoiler below.\n||![The ending](https://e.com/end.png)||", event, adapter)

    images = adapter.send_multiple_images.await_args.kwargs["images"]
    assert images == [("https://e.com/end.png", "The ending")] and is_spoiler(images[0][1])


def test_runtime_footer_never_rides_on_a_spoiler_caption():
    """The footer would be hidden with the image; it goes on an open photo, or on its own."""
    adapter = SimpleNamespace(media_caption_limit=1024, _is_animation_url=lambda url: url.endswith(".gif"))
    images, _ = extract("![open](https://e.com/a.png) ||![hidden](https://e.com/b.png)||")

    captioned, attached = caption_with_footer(adapter, images, "model · 12s")
    assert attached and captioned[0][1] == "open\n\nmodel · 12s" and captioned[1] == images[1]

    only_spoiler, _ = extract("||![hidden](https://e.com/b.png)||")
    assert caption_with_footer(adapter, only_spoiler, "model · 12s") == (only_spoiler, False)

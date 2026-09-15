"""Discord: a text send that precedes extracted images must not unfurl link previews.

When a response carries an image the gateway extracts, the text is sent first and the image is
uploaded as a native attachment after it. Any page URL left in the text (an xkcd strip's page,
a news article) still unfurls into a Discord preview card — often showing the very image that
is about to be attached — so the user sees the picture twice. The base delivery marks such text
sends with ``metadata["suppress_embeds"]`` and the Discord adapter maps that onto the embed-
suppression kwarg the running py-cord exposes.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig

import plugins.platforms.discord.adapter as discord_platform  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


class _Channel:
    """A text channel whose ``send`` has the real py-cord keyword surface."""

    def __init__(self, *, keyword: str = "suppress_embeds"):
        self.id = 100
        self.calls = []
        if keyword == "suppress_embeds":
            async def send(*, content=None, reference=None, suppress_embeds=False, **kw):
                self.calls.append({"content": content, "suppress_embeds": suppress_embeds})
                return SimpleNamespace(id=1)
        elif keyword == "suppress":
            async def send(*, content=None, reference=None, suppress=False, **kw):
                self.calls.append({"content": content, "suppress": suppress})
                return SimpleNamespace(id=1)
        else:
            async def send(*, content=None, reference=None):
                self.calls.append({"content": content})
                return SimpleNamespace(id=1)
        self.send = send


def _adapter(channel):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = MagicMock()
    adapter._client.user = SimpleNamespace(id=999, bot=True)
    adapter._resolve_channel = AsyncMock(return_value=channel)
    adapter._record_response_async = AsyncMock(side_effect=lambda reply_to, result, content, final: result)
    return adapter


@pytest.mark.parametrize("keyword", ["suppress_embeds", "suppress"])
def test_suppress_kwargs_follow_the_send_signature(keyword):
    channel = _Channel(keyword=keyword)
    assert discord_platform._discord_send_suppress_kwargs(channel.send, True) == {keyword: True}
    assert discord_platform._discord_send_suppress_kwargs(channel.send, False) == {}


def test_suppress_kwargs_are_omitted_when_the_library_has_no_such_parameter():
    channel = _Channel(keyword="none")
    assert discord_platform._discord_send_suppress_kwargs(channel.send, True) == {}


@pytest.mark.asyncio
async def test_send_suppresses_embeds_only_when_the_metadata_asks():
    channel = _Channel()
    adapter = _adapter(channel)

    plain = await adapter.send("100", "see https://xkcd.com/1234")
    marked = await adapter.send("100", "see https://xkcd.com/1234", metadata={"suppress_embeds": True})

    assert plain.success and marked.success
    assert [c["suppress_embeds"] for c in channel.calls] == [False, True]

"""Bot API recording and protocol assertions for Telegram voice echoes."""

from collections import deque
from html.parser import HTMLParser
import re
from types import SimpleNamespace

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


class RecordingTelegramBot:
    """Record transport effects while optional Bot API failures remain deterministic."""

    def __init__(self, failures=()):
        self.failures = deque(failures)
        self.attempts = []
        self.delivered = []
        self.rich_requests = []

    async def send_message(self, **kwargs):
        self.attempts.append(dict(kwargs))
        failure = self.failures.popleft() if self.failures else None
        if failure is not None:
            raise failure
        self.delivered.append(dict(kwargs))
        return SimpleNamespace(message_id=1000 + len(self.delivered))

    async def do_api_request(self, *args, **kwargs):
        self.rich_requests.append((args, kwargs))
        return {"message_id": 2000 + len(self.rich_requests)}


def telegram_adapter(bot=None, *, reply_to_mode="first"):
    """Use the real sender with rich capability available and typing hints disabled."""
    adapter = TelegramAdapter(PlatformConfig(
        enabled=True, token="test-token", typing_indicator=False,
        reply_to_mode=reply_to_mode,
        extra={"rich_messages": True, "disable_link_previews": True},
    ))
    adapter._bot = bot if bot is not None else RecordingTelegramBot()
    return adapter


class _EchoHTMLParser(HTMLParser):
    """Collect decoded quote contents and reject injected or incomplete markup."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_quote = False
        self.quote_count = 0
        self.body = []
        self.outside = []

    def handle_starttag(self, tag, attrs):
        assert tag == "blockquote"
        assert attrs == [("expandable", None)]
        assert not self.in_quote
        self.in_quote = True
        self.quote_count += 1

    def handle_endtag(self, tag):
        assert tag == "blockquote"
        assert self.in_quote
        self.in_quote = False

    def handle_data(self, data):
        (self.body if self.in_quote else self.outside).append(data)


def assert_stt_echo(messages, transcript, *, wire_limit=4096):
    """Assert lossless speech, independent expandable chunks and the Bot API budget."""
    assert messages
    bodies = []
    for index, message in enumerate(messages, 1):
        assert message["parse_mode"] == "HTML"
        wire = message["text"]
        assert len(wire.encode("utf-16-le")) // 2 <= wire_limit
        assert not re.search(r"&(?!amp;|lt;|gt;)", wire)
        parser = _EchoHTMLParser()
        parser.feed(wire)
        parser.close()
        assert parser.quote_count == 1
        assert not parser.in_quote
        bodies.append("".join(parser.body))
        outside = "".join(parser.outside)
        indicator = re.search(r" \((\d+)/(\d+)\)$", outside)
        if len(messages) > 1:
            assert indicator is not None
            assert tuple(map(int, indicator.groups())) == (index, len(messages))
        else:
            assert indicator is None
    assert "".join(bodies) == transcript

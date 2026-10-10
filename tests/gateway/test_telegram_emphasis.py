"""Legacy Telegram emphasis preserves structure without consuming literal stars."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.mark.parametrize(("source", "expected"), [
    ("***both***", "_*both*_"),
    ("**bold *italic* text**", "*bold _italic_ text*"),
    ("*italic **bold** text*", "_italic *bold* text_"),
    ("**bold\ntext**", "*bold\ntext*"),
    ("**outer **inner** end**", "*outer inner end*"),
    ("*outer *inner* end*", "_outer inner end_"),
])
@pytest.mark.asyncio
async def test_legacy_send_preserves_emphasis(source, expected):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._rich_messages_enabled = False
    adapter._bot = SimpleNamespace(send_message=AsyncMock(return_value=SimpleNamespace(message_id=1)))
    result = await adapter.send("12345", source)
    assert result.success
    adapter._bot.send_message.assert_awaited_once()
    sent = adapter._bot.send_message.call_args.kwargs
    assert sent["text"] == expected
    assert sent["parse_mode"] == "MarkdownV2"


@pytest.mark.parametrize(("source", "expected"), [
    ("**unclosed", r"\*\*unclosed"),
    ("**a *b*> c**", r"*a _b_\> c*"),
    ("line\\\n  next", "line\\\\\n  next"),
    ("> *first\n> second*", "> _first\n> second_"),
    ("Files: *.py\n# Heading\nFootnote*", "Files: \\*\\.py\n*Heading*\nFootnote\\*"),
    ("Files: *.py\n```\ncode\n```\nFootnote*", "Files: \\*\\.py\n```\ncode\n```\nFootnote\\*"),
    ("**bold ~~strike** tail~~", r"*bold \~\~strike* tail\~\~"),
    ("**bold ||secret** tail||", r"*bold \|\|secret* tail\|\|"),
    ("Files: *.py\n- Footnote*", "Files: \\*\\.py\n\\- Footnote\\*"),
    ("_**(bold)**_", r"\_*\(bold\)*\_"),
    ("Files: *.py\n\nFootnote*", "Files: \\*\\.py\n\nFootnote\\*"),
    (r"\_prefix **value\_ suffix**", r"\\\_prefix *value\\\_ suffix*"),
    ("trailing **", r"trailing \*\*"),
    (r"\*literal\*", r"\*literal\*"),
    ("2 * 3 * 4", r"2 \* 3 \* 4"),
    ("*.py and *.md", r"\*\.py and \*\.md"),
    ("* first\n* second", "\\* first\n\\* second"),
    ("`***code***`", "`***code***`"),
    ("```\n***code***\n```", "```\n***code***\n```"),
    ("[link](https://example.com)", "[link](https://example.com)"),
    ("~~strike~~ ||spoiler||", "~strike~ ||spoiler||"),
    ("> quote", "> quote"),
    ("**> quote||", "**> quote||"),
    ("snake_case and _literal_", r"snake\_case and \_literal\_"),
])
def test_legacy_emphasis_preserves_other_content(source, expected):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    assert adapter.format_message(source) == expected

"""MarkdownV2 / rich-markdown conversion contracts, judged by the Bot API stand-in's parser."""

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter
from tests.fakes.platforms.telegram_standin import mdv2_error


@pytest.fixture()
def adapter() -> TelegramAdapter:
    return TelegramAdapter(PlatformConfig(enabled=True, token="123:abc"))


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        # The quoted text opens with **bold**: the opener's ** must not pair with it.
        ("**> **Action Plan** 1. **Skill** - do it||", "**> *Action Plan* 1\\. *Skill* \\- do it||"),
        # Multi-line expandable quote: || stays raw on the last line only.
        ("**> First\n> **Second**\n> Last||", "**> First\n> *Second*\n> Last||"),
    ],
)
def test_expandable_quote_survives_bold_content(adapter, markdown, expected):
    out = adapter.format_message(markdown)
    assert out == expected
    assert mdv2_error(out) is None


def test_bold_without_quote_keeps_its_markers(adapter):
    """Only a line-start ``**>`` is a quote opener; ``**`` before other text is still bold."""
    out = adapter.format_message("**>5 items** and **a > b**")
    assert out == "*\\>5 items* and *a \\> b*"
    assert mdv2_error(out) is None


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        ("API $395k vs UI $483k **88k gap**", "API \\$395k vs UI \\$483k **88k gap**"),
        ("R$ 1.000 e R$ 2.000", "R\\$ 1.000 e R\\$ 2.000"),
        ("US$ 1,200 vs US$ 900", "US\\$ 1,200 vs US\\$ 900"),
        ("costs $5, area $x^2$, $$E=mc^2$$", "costs \\$5, area $x^2$, $$E=mc^2$$"),
        ("run `echo $1 $2` then \\$3", "run `echo $1 $2` then \\$3"),
        ("```\nprice=$5 $6\n```", "```\nprice=$5 $6\n```"),
    ],
)
def test_rich_payload_escapes_currency_but_keeps_math_and_code(adapter, markdown, expected):
    assert adapter._rich_message_payload(markdown)["markdown"] == expected

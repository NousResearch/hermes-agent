"""Regressions from #150997: emphasis rule of 3 (#151722) and the rich-message code-region scan
(#151725). Exact outputs are judged by the Bot API stand-in's MarkdownV2 parser too."""

import time

import pytest

from hermes_agent.gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter
from plugins.platforms.telegram.telegram_markdown import escape_rich_currency_dollars
from tests.fakes.platforms.telegram_standin import mdv2_error


@pytest.fixture()
def adapter() -> TelegramAdapter:
    return TelegramAdapter(PlatformConfig(enabled=True, token="123:abc"))


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        ("**Result: 2*3 = 6**", "*Result: 2\\*3 \\= 6*"),
        ("The answer is **6*7 = 42**.", "The answer is *6\\*7 \\= 42*\\."),
        ("**rm -rf *.tmp**", "*rm \\-rf \\*\\.tmp*"),
        ("**glob *.py files**", "*glob \\*\\.py files*"),
        ("*a**b**c*", "_a*b*c_"),
        ("**Use \\*args**", "*Use \\\\\\*args*"),
    ],
)
def test_a_lone_star_inside_bold_stays_a_visible_star(adapter, markdown, expected):
    """A ``*`` flanked by text inside ``**bold**`` is literal: bold stays bold and no character of
    the user's math or shell command disappears (#151722)."""
    out = adapter.format_message(markdown)
    assert out == expected
    assert mdv2_error(out) is None


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        # A fence still streaming (no closing fence yet) is code to the end of the text.
        ("```bash\necho $5\n", "```bash\necho $5\n"),
        ("~~~\necho $5\n~~~\nthen $6", "~~~\necho $5\n~~~\nthen \\$6"),
        ("````\n```\n$5\n```\n````\n$6", "````\n```\n$5\n```\n````\n\\$6"),
        ("text\n\n    echo $5\n\nthen $6", "text\n\n    echo $5\n\nthen \\$6"),
        ("<code>$5</code> and $6", "<code>$5</code> and \\$6"),
        # The closer pairs with the next run of the opener's width, not a shorter tail.
        ("x ``y` $5 z` w", "x ``y` $5 z` w"),
        ("$1\\text{ kg}$ and $x$", "$1\\text{ kg}$ and $x$"),
        # A fence opens at a line start: a mid-line run nothing closes is literal prose.
        ("Use ``` to open a fence. Cost is $5", "Use ``` to open a fence. Cost is \\$5"),
        ("Use ``` to open a fence. Cost is $5\n```\nx\n```", "Use ``` to open a fence. Cost is \\$5\n```\nx\n```"),
        ("Here is the code: ```bash\necho $5\n```\nthen $6", "Here is the code: ```bash\necho $5\n```\nthen \\$6"),
        # A prose $$ (the shell PID) never pairs across a code span with a later formula's $$.
        (
            "Use $$ for the PID.\n\nA `x` span. Use $$ again. Cost is $5 vs $10.\n\n$$\nE = mc^2\n$$",
            "Use $$ for the PID.\n\nA `x` span. Use $$ again. Cost is \\$5 vs \\$10.\n\n$$\nE = mc^2\n$$",
        ),
        ("Use $$ for the PID. A `x` span. Cost is $5. Use $$ again.", "Use $$ for the PID. A `x` span. Cost is \\$5. Use $$ again."),
    ],
)
def test_rich_dollar_escape_leaves_every_code_region_verbatim(markdown, expected):
    """CommonMark applies no backslash escape inside code, so an inserted ``\\$`` would show (#151725)."""
    assert escape_rich_currency_dollars(markdown) == expected


def test_rich_payload_is_linear_on_unmatched_backtick_runs(adapter):
    """#151725: every backtick run without a same-width closer rescanned the rest of the line, so
    this 32 KB reply blocked the gateway event loop for ~2 s on every rich send and draft frame."""
    text = "$5 " + "``x```" * 5400
    started = time.perf_counter()
    payload = adapter._rich_message_payload(text)
    assert time.perf_counter() - started < 1.0
    assert payload["markdown"].startswith("\\$5 ")


def test_escaped_star_check_does_not_copy_the_message_prefix(adapter):
    text = "\\* " * 350_000  # ~1 MB of literal stars; a prefix copy per star is quadratic
    started = time.perf_counter()
    out = adapter.format_message(text)
    assert time.perf_counter() - started < 8.0
    assert out.count("\\*") == 350_000

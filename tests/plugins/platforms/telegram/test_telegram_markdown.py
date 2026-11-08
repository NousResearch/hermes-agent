"""MarkdownV2 / rich-markdown conversion contracts, judged by the Bot API stand-in's parser."""

import time

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
        ("$2+2$ costs $5 vs $6", "$2+2$ costs \\$5 vs \\$6"),
        ("$3.5\\times10^3$ vs $4", "$3.5\\times10^3$ vs \\$4"),
        ("$$x^2=16$$ so $4$, not $-4$", "$$x^2=16$$ so $4$, not $-4$"),
        ("$2x$ but $5$6", "$2x$ but \\$5\\$6"),
        ("``price=$5`` costs $5", "``price=$5`` costs \\$5"),
        ("`` a`b $5 `` then $7", "`` a`b $5 `` then \\$7"),
    ],
)
def test_rich_payload_escapes_currency_but_keeps_math_and_code(adapter, markdown, expected):
    assert adapter._rich_message_payload(markdown)["markdown"] == expected


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        ("$$x=1$$ so $2 + 2$", "$$x=1$$ so $2 + 2$"),
        ("$$x=1$$ so $2\\pi r$", "$$x=1$$ so $2\\pi r$"),
        ("$$x=1$$ so $2x + 1$", "$$x=1$$ so $2x + 1$"),
        ("$$x=1$$ so $2\\,x$", "$$x=1$$ so $2\\,x$"),
        ("$5 - $3 = $2", "\\$5 - \\$3 = \\$2"),
        ("$1,000-$2,000", "\\$1,000-\\$2,000"),
        ("$2\\frac{1}{2}$ and $3", "$2\\frac{1}{2}$ and \\$3"),
    ],
)
def test_rich_payload_keeps_spaced_numeric_math_whole(adapter, markdown, expected):
    """Digit-led math with spaces or TeX commands is one formula; amounts around operators stay escaped."""
    assert adapter._rich_message_payload(markdown)["markdown"] == expected


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        ("***bold* rest**", "*_bold_ rest*"),
        ("***bold** rest*", "_*bold* rest_"),
        ("**a *b***", "*a _b_*"),
        ("*a **b***", "_a *b*_"),
    ],
)
def test_combined_emphasis_closing_at_different_places_parses(adapter, markdown, expected):
    """A delimiter run opening two styles that close separately emits every closing mark."""
    out = adapter.format_message(markdown)
    assert out == expected
    assert mdv2_error(out) is None


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        ("`a ``b-c`` d`", "`a \\`\\`b-c\\`\\` d`"),
        ("` ``x.y`` `", "` \\`\\`x.y\\`\\` `"),
        ("Use ``value`` now", "Use `value` now"),
    ],
)
def test_double_backticks_inside_code_stay_code_content(adapter, markdown, expected):
    """Backtick runs inside an open code span are its content, never a nested span closing it."""
    out = adapter.format_message(markdown)
    assert out == expected
    assert mdv2_error(out) is None


@pytest.mark.parametrize("markdown", ["> **a\nb**", "> **a\n> b**", "> [`\n`]()"])
def test_emphasis_and_links_never_cross_a_quote_line_edge(adapter, markdown):
    """Telegram ends a blockquote at a newline, so an entity left open across it is rejected."""
    assert mdv2_error(adapter.format_message(markdown)) is None


@pytest.mark.parametrize(
    "markdown",
    [
        # Unclosed spaced formula (a streaming draft frame): each \cmd could also be \c plus a one-letter variable.
        "so $2 \\cdot " + "\\alpha\\beta " * 12 + "and more",
        # Closed tight formula with a character the operator branch doesn't take (|).
        "$2" + "\\alpha" * 12 + "\\left|x\\right|$",
        # One long TeX command that never closes: the mandatory command must not retry every length.
        "$1\\" + "a" * 40000 + "!",
    ],
    ids=["unclosed-spaced", "tight-with-pipe", "long-command"],
)
def test_rich_currency_escape_does_not_backtrack_on_tex_commands(adapter, markdown):
    """A digit-led span that isn't math fails in linear time (each row takes 10 s or more under backtracking)."""
    started = time.perf_counter()
    adapter._rich_message_payload(markdown)
    assert time.perf_counter() - started < 2.0

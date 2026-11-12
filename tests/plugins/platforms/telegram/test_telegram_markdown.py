"""MarkdownV2 / rich-markdown conversion contracts, judged by the Bot API stand-in's parser."""

import time

import pytest

from hermes_agent.gateway.config import PlatformConfig
from hermes_agent.plugins.platforms.telegram import telegram_markdown
from hermes_agent.plugins.platforms.telegram.adapter import TelegramAdapter
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
        ("Price $5 (`$` or `$6`).", "Price \\$5 (`$` or `$6`)."),
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
        ("$$x=1$$ so $2[x+1]$", "$$x=1$$ so $2[x+1]$"),
        ("$$x=1$$ so $2\\left|x\\right|$", "$$x=1$$ so $2\\left|x\\right|$"),
        ("$$x=1$$ so $5! = 120$", "$$x=1$$ so $5! = 120$"),
        ("$$x=1$$ so $2 \\in [0, 1]$", "$$x=1$$ so $2 \\in [0, 1]$"),
        ("$$x=1$$ so $3'$ and $1:2$", "$$x=1$$ so $3'$ and $1:2$"),
        # A word of two or more letters is prose, in any script.
        ("$5 и 10$", "\\$5 и 10$"),
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
    ("text", "start", "end", "newline", "blank", "quote_edge"),
    [
        # A newline at the span's END is outside it; one at START is inside.
        ("a\nb", 0, 1, False, False, False),
        ("a\nb", 0, 2, True, False, False),
        # "\n\n" only counts when both characters are inside the span.
        ("a\n\nb", 0, 2, True, False, False),
        ("a\n\nb", 0, 3, True, True, False),
        # A quote line on either side of the newline makes the span cross the quote's edge.
        ("> a\nb", 0, 4, True, False, True),
        ("a\n> b", 0, 4, True, False, True),
        ("a\nb\n> c", 2, 4, True, False, True),
        ("a\nb\nc", 0, 5, True, False, False),
    ],
)
def test_span_edges_match_the_character_rule(text, start, end, newline, blank, quote_edge):
    """The bisect lookups must answer exactly what scanning the sliced span used to (#151723)."""
    edges = telegram_markdown._SpanEdges(text)
    assert edges.has_newline(start, end) is newline
    assert edges.has_blank_line(start, end) is blank
    assert edges.crosses_quote_edge(start, end) is quote_edge


def test_emphasis_pairing_does_not_rescan_a_distant_opener(adapter):
    """A distant unmatched `**` opener must not make every later closer rescan its span.

    #151723: each closer sliced text[o_end:start] and re-found its newlines (and recompiled the
    quote regex), so this 32 KB reply took ~1.4 s (200 KB ~106 s) synchronously on the gateway
    event loop before every send and streamed edit. The payload is 128 KB so the 2 s floor
    tests/AGENTS.md asks of wall-clock bounds still separates the linear pairing from the regression.
    """
    size = 131072
    text = "**a\n" + "plain line here\n" * (size // 32) + "> q\n" + "b** \n" * (size // 10)
    started = time.perf_counter()
    out = adapter.format_message(text)
    assert time.perf_counter() - started < 2.0
    assert mdv2_error(out) is None


def test_one_line_link_quote_edge_does_not_scan_the_message_prefix(adapter):
    """A short one-line link must not re-walk the text before it (#151728 review).

    The newline scan is bounded by the link span, but locating the link's own line ran an
    unconditional ``text.rfind("\\n", 0, start)``, so a long single line carrying many short
    links reintroduced quadratic work: measured 0.006 s / 0.017 s / 0.065 s / 0.235 s for 250 /
    500 / 1000 / 2000 links on one line. Answering "no newline, no crossing" first keeps it linear.
    """

    class PrefixGuard(str):
        def rfind(self, *_args, **_kwargs):
            raise AssertionError("a one-line span must not scan the message prefix")

    prefix = "a" * 32000
    message = PrefixGuard(prefix + "[docs](https://example.com)")
    assert telegram_markdown._crosses_quote_edge(message, len(prefix), len(message)) is False

    # The same shape through the public formatter still converts every link on the line.
    out = adapter.format_message("x" * 8 + " [d_i](https://e.co/p)" * 500)
    assert out.count("[d\\_i](https://e.co/p)") == 500
    assert mdv2_error(out) is None


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


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        ("Already underway in [Long-Term Fix](Long-Term Fix).", "Already underway in Long\\-Term Fix\\."),
        ("See [Run](@session:default/20260101_abc).", "See Run\\."),
        ("[Docs](https://example.com/a_b)", "[Docs](https://example.com/a_b)"),
        ("[me](tg://user?id=1) [w](example.com)", "[me](tg://user?id=1) [w](example.com)"),
    ],
)
def test_link_targets_telegram_cannot_open_degrade_to_label(adapter, markdown, expected):
    out = adapter.format_message(markdown)
    assert out == expected
    assert mdv2_error(out) is None


@pytest.mark.parametrize("span", ["``[k](v w)``", "``a ` [k](v w)``", "`[k](v w)`"])
def test_link_shaped_text_inside_a_code_span_stays_verbatim(span):
    from hermes_agent.plugins.platforms.telegram.telegram_markdown import degrade_unrenderable_links

    assert degrade_unrenderable_links(f"use {span} then [x](y)") == f"use {span} then x"

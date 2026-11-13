"""MarkdownV2 / rich-markdown conversion contracts, judged by the Bot API stand-in's parser."""

import re
import time

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram import telegram_markdown
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
    from plugins.platforms.telegram.telegram_markdown import degrade_unrenderable_links

    assert degrade_unrenderable_links(f"use {span} then [x](y)") == f"use {span} then x"


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:5173/?token=abc123",
    "https://web.archive.org/web/*/example.com/*",
    "http://h.example/p?t=a*b*c",
    "https://x.example/a~~b~~c",
    "http://h.example/x||y||z",
    "https://example.com/a*;b*c",
    "https://example.com/a**,b**c",
    "https://example.com/a*;b*c*",
    "https://x.example/a~~;b~~c~~",
])
def test_bare_url_reaches_the_chat_as_literal_text(adapter, url):
    """Every char of a bare URL is escaped, so * ~ || inside it never become italic/strike/spoiler."""
    import re
    out = adapter.format_message(f"Open {url} now")
    assert mdv2_error(out) is None
    assert re.sub(r"\\(.)", r"\1", out) == f"Open {url} now"
    assert re.search(r"(?<!\\)[*_~|]", out) is None
    italic = adapter.format_message(f"*Open {url} now*")
    assert mdv2_error(italic) is None and url in re.sub(r"\\(.)", r"\1", italic[1:-1])


@pytest.mark.parametrize("wrapped, rendered", [
    ("*https://example.com/*", "_https://example\\.com/_"),
    ("**https://example.com/**", "*https://example\\.com/*"),
    ("~~https://example.com/~~", "~https://example\\.com/~"),
    ("**bold https://example.com/**", "*bold https://example\\.com/*"),
    ("*see https://x.com/docs/*", "_see https://x\\.com/docs/_"),
    ("*https://a.example/* and *https://b.example/*", "_https://a\\.example/_ and _https://b\\.example/_"),
    ("**x https://a.example/** y **z**", "*x https://a\\.example/* y *z*"),
    ("*Open https://archive.example/web/* now* and https://b.example/*",
     "_Open https://archive\\.example/web/\\* now_ and https://b\\.example/\\*"),
    ("*See https://a.example/x and https://b.example/*", "_See https://a\\.example/x and https://b\\.example/_"),
    ('"*https://example.com/*"', '"_https://example\\.com/_"'),
    ('"**https://example.com/**"', '"*https://example\\.com/*"'),
    ('"~~https://example.com/~~"', '"~https://example\\.com/~"'),
])
def test_emphasis_wrapping_a_url_that_ends_in_slash_still_renders(adapter, wrapped, rendered):
    """A trailing mark after ``/`` is URL text (``/web/*``) unless it closes emphasis opened before the URL."""
    out = adapter.format_message(f"see {wrapped} ok")
    assert mdv2_error(out) is None
    assert out == f"see {rendered} ok"


@pytest.mark.parametrize("wrapped, rendered", [
    ("**See\nhttps://example.com/**", "*See\nhttps://example\\.com/*"),
    ("*outer **https://example.com/** end*", "_outer *https://example\\.com/* end_"),
    ("**https://example.com**—next", "*https://example\\.com*—next"),
    ("[**https://x.example/**]", "\\[*https://x\\.example/*\\]"),
    ("~~https://x.example/a~~,next", "~https://x\\.example/a~,next"),
    ("**https://x/a**,next", "*https://x/a*,next"),
    ("**https://example.com/a**,next**word**", "*https://example\\.com/a*,next*word*"),
    ("*Open https://example.com/a*;b*c* now*", "_Open https://example\\.com/a\\*;b\\*c\\* now_"),
    # A URL run closes part of a mixed opener run, as the renderer's own star pairing does.
    ("***bold* https://example.com/**", "*_bold_ https://example\\.com/*"),
    ("***a** https://x/y*", "_*a* https://x/y_"),
    ("**a *b https://x/c* d**", "*a _b https://x/c_ d*"),
    ("**rm *x https://x/y**", "*rm \\*x https://x/y*"),
])
def test_emphasis_around_a_url_pairs_as_around_any_word(adapter, wrapped, rendered):
    """Bold across a newline, nested emphasis and a closer with prose right after it pair as they
    would around plain text: the URL stash never turns a real delimiter into URL text."""
    out = adapter.format_message(wrapped)
    assert mdv2_error(out) is None
    assert out == rendered


def _url_survives(text: str, url: str, out: str) -> bool:
    """*url* reaches the chat whole (every mark escaped), or ends at a mark run that closes emphasis
    opened before it: its text up to that run shows contiguous and escaped, the run renders as the
    closer, and what follows the run is prose again."""
    esc = telegram_markdown._escape_mdv2
    if esc(url) in out:
        return True
    opened = text[:text.index(url)]
    return any(run.group()[0] in opened and re.search(re.escape(esc(url[:run.start()])) + r"[*_~|]", out)
               for run in re.finditer(r"[*~|]+", url))


_URL_SHAPES = [f"a{m}{p}b{m}c{t}" for m in ("*", "**", "~~", "||") for p in ".,;:!?)" for t in ("", m)]
_URL_WRAPPERS = ["{}", "Open {} now", "*{}*", "**{}**", "~~{}~~", "||{}||", '"{}"', "*Open {} now*",
                 "**see\n{}**", "*x* {} *y*", "{}, then **b**"]


@pytest.mark.parametrize("wrapper", _URL_WRAPPERS)
def test_no_bare_url_loses_a_character_to_formatting(wrapper):
    """Property over every internal-punctuation URL shape: a mark inside a URL is URL text unless it
    closes emphasis opened outside the URL, so no URL character is dropped or turned into formatting."""
    for shape in _URL_SHAPES:
        url = "https://example.com/" + shape
        text = wrapper.format(url)
        out = telegram_markdown.to_markdown_v2(text)
        assert mdv2_error(out) is None, text
        assert _url_survives(text, url, out), (text, out)


@pytest.mark.parametrize("unit", ["https://example.com/a ", "**see https://x.example/web/** ok "])
def test_many_bare_urls_format_in_linear_time(unit):
    """Placeholder restoration is one pass, not one whole-message scan per URL, and deciding
    ``/``-ended mark runs is budgeted: 10x the URLs costs about 10x the time (a per-URL scan ~100x)."""
    def best(n: int) -> float:
        text = unit * n
        runs = []
        for _ in range(3):
            start = time.perf_counter()
            telegram_markdown.to_markdown_v2(text)
            runs.append(time.perf_counter() - start)
        return min(runs)

    small, large = best(2_000), best(20_000)
    assert large < 40 * small + 0.5

"""Tests for MarkdownV2 inline-entity balance across truncate_message chunks.

The problem (issue #52093): ``truncate_message`` guards triple-backtick fences
and inline code spans when picking a split point, but nothing else.  When a
boundary lands inside ``*bold*``, ``_italic_``, ``~strike~`` or ``||spoiler||``,
each chunk carries an unpaired delimiter, Telegram rejects the message with
"can't parse entities", and the adapter falls back to plain text — silently
stripping formatting from the whole response.

Trigger conditions (both required, which is why this is rarely seen):

  1. No newline in the second half of the region, so the splitter falls back
     from ``region.rfind("\\n")`` to ``region.rfind(" ")``.  Every inline
     entity ``format_message`` emits is newline-free by construction (bold /
     strike / spoiler use ``.``, italic uses ``[^*\\n]+``), so a newline
     boundary is ALWAYS entity-safe.
  2. The chosen space falls *inside* an entity, i.e. the span contains
     internal spaces.  A span without them is skipped over harmlessly.

These tests drive the real production pipeline — ``TelegramAdapter.format_message``
followed by ``truncate_message(..., len_fn=utf16_len)`` — because the markers
that reach the splitter are the *converted* ones (``**bold**`` becomes
``*bold*``, ``*italic*`` becomes ``_italic_``, ``~~strike~~`` becomes
``~strike~``), not the markdown the model wrote.
"""

import re
import sys
from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, utf16_len

# The behavioural tests below deliberately do NOT depend on this helper, so
# they express the delivery contract and fail (rather than error) against an
# unguarded splitter.  Only the scanner unit tests need it.
try:
    from gateway.platforms.base import _open_inline_entities
except ImportError:  # pragma: no cover - pre-guard tree
    _open_inline_entities = None


def _ensure_telegram_mock():
    """Mock the telegram package if it's not installed."""
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    mod = MagicMock()
    mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    mod.constants.ChatType.GROUP = "group"
    mod.constants.ChatType.SUPERGROUP = "supergroup"
    mod.constants.ChatType.CHANNEL = "channel"
    mod.constants.ChatType.PRIVATE = "private"
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)


_ensure_telegram_mock()

from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402

MAX = 4096

# (markdown the model writes, delimiter format_message emits, label)
MARKERS = [
    ("**", "*", "bold"),
    ("*", "_", "italic"),
    ("~~", "~", "strike"),
    ("||", "||", "spoiler"),
]


@pytest.fixture()
def fmt():
    """The real Telegram markdown -> MarkdownV2 converter."""
    return TelegramAdapter(PlatformConfig(enabled=True, token="test-token")).format_message


# ── oracle ────────────────────────────────────────────────────────────────
# Deliberately implemented differently from _open_inline_entities (regex
# stripping + parity counting, rather than a stateful scan) so these tests do
# not merely assert the implementation against itself.

def _unclosed(text: str) -> list:
    s = re.sub(r"\\.", "", text, flags=re.S)      # escaped chars are literals
    s = re.sub(r"```.*?```", "", s, flags=re.S)   # fenced blocks
    s = re.sub(r"`[^`]*`", "", s)                 # inline code spans
    bad = []
    if s.count("||") % 2:
        bad.append("||")
    s = s.replace("||", "")
    for m in ("*", "_", "~"):
        if s.count(m) % 2:
            bad.append(m)
    return bad


def _forced_boundary(md_marker: str) -> str:
    """Source whose formatted form puts a space-containing span across 4096."""
    return ("x" * 3900 + md_marker + ("word inside span " * 60) + md_marker
            + " tail" + " z" * 1500)


def _split(fmt, raw, max_length=MAX):
    return BasePlatformAdapter.truncate_message(fmt(raw), max_length, len_fn=utf16_len)


def _content(chunks) -> str:
    """Whitespace- and delimiter-free view, for content-preservation checks."""
    joined = "".join(re.sub(r" \(\d+/\d+\)$", "", c) for c in chunks)
    joined = re.sub(r"\\(.)", r"\1", joined)
    for tok in ("```", "||", "*", "_", "~", "`"):
        joined = joined.replace(tok, "")
    return re.sub(r"\s+", "", joined)


# ── the scanner ───────────────────────────────────────────────────────────

@pytest.mark.skipif(_open_inline_entities is None,
                    reason="inline-entity scanner not present in this tree")
class TestOpenInlineEntities:
    def test_balanced_span_reports_nothing(self):
        assert _open_inline_entities("a *bold* b") == []

    @pytest.mark.parametrize("delim", ["*", "_", "~", "||"])
    def test_unclosed_delimiter_reported(self, delim):
        assert _open_inline_entities("text %sopen" % delim) == [delim]

    def test_escaped_delimiter_is_literal(self):
        assert _open_inline_entities(r"a \* literal star") == []

    def test_delimiters_inside_inline_code_ignored(self):
        assert _open_inline_entities("before `a * b` after") == []

    def test_delimiters_inside_fence_ignored(self):
        assert _open_inline_entities("```\na * b _ c\n```") == []

    def test_nesting_is_outermost_first(self):
        assert _open_inline_entities("||spoiler *bold") == ["||", "*"]

    def test_spoiler_matched_before_single_pipe(self):
        assert _open_inline_entities("||x||") == []

    def test_initial_state_is_carried(self):
        assert _open_inline_entities("still bold", initial=["*"]) == ["*"]


# ── the regression: real pipeline, every marker type ──────────────────────

class TestFormatThenTruncate:
    @pytest.mark.parametrize("md,delim,label", MARKERS, ids=[m[2] for m in MARKERS])
    def test_span_across_boundary_stays_balanced(self, fmt, md, delim, label):
        chunks = _split(fmt, _forced_boundary(md))
        assert len(chunks) > 1, "expected a multi-chunk split for this fixture"
        for i, chunk in enumerate(chunks, 1):
            assert _unclosed(chunk) == [], (
                "chunk %d/%d has unpaired %s delimiter(s) %s; Telegram would "
                "reject it and strip formatting"
                % (i, len(chunks), label, _unclosed(chunk))
            )

    @pytest.mark.parametrize("md,delim,label", MARKERS, ids=[m[2] for m in MARKERS])
    def test_chunks_respect_the_cap(self, fmt, md, delim, label):
        for chunk in _split(fmt, _forced_boundary(md)):
            assert utf16_len(chunk) <= MAX

    @pytest.mark.parametrize("md,delim,label", MARKERS, ids=[m[2] for m in MARKERS])
    def test_no_content_lost_or_duplicated(self, fmt, md, delim, label):
        raw = _forced_boundary(md)
        assert _content(_split(fmt, raw)) == _content([fmt(raw)])


# ── the defect that sank the previous attempt (#52096) ────────────────────

class TestOverlongEntity:
    """An entity longer than one whole chunk.

    Rewinding the split point cannot fix this: after moving the boundary
    before the opener, the next chunk starts *at* the opener with no preceding
    whitespace, so a later boundary splits it anyway.  Closing and reopening
    has no such limit.
    """

    def test_entity_spanning_several_chunks(self, fmt):
        raw = "y" * 200 + "**" + ("long bold body " * 900) + "**" + " end"
        chunks = _split(fmt, raw)
        assert len(chunks) >= 3, "fixture should span at least three chunks"
        for i, chunk in enumerate(chunks, 1):
            assert _unclosed(chunk) == [], "chunk %d/%d unbalanced" % (i, len(chunks))

    def test_overlong_entity_terminates(self, fmt):
        # A span with no spaces at all: the space fallback cannot help, so the
        # splitter hard-cuts.  Must still terminate and stay balanced.
        raw = "**" + ("b" * 9000) + "**"
        chunks = _split(fmt, raw)
        assert chunks
        for chunk in chunks:
            assert _unclosed(chunk) == []


# ── the streaming overflow branch ────────────────────────────────────────

class TestStreamingOverflowPreview:
    """The mid-stream preview delegates to the same splitter, so it inherits
    the guard rather than needing its own.

    ``_truncate_stream_overflow_preview`` returns ``truncate_message(...)[0]``
    (it deliberately takes only the head chunk, because splitting a preview
    would move the active message id and re-trigger the overflow cycle,
    #48648).  Before this change that head chunk could carry an unpaired
    delimiter exactly like any other chunk.
    """

    def test_preview_head_chunk_is_balanced(self, fmt):
        adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
        preview = adapter._truncate_stream_overflow_preview(fmt(_forced_boundary("**")))
        assert utf16_len(preview) <= MAX
        assert _unclosed(preview) == [], (
            "streaming preview carries an unpaired delimiter %s" % _unclosed(preview))

    def test_consumer_split_delegates_to_the_adapter(self):
        """Guard the delegation itself: if the consumer stopped calling the
        adapter's truncate_message, the fix would silently stop covering the
        streaming path."""
        import inspect
        from gateway.stream_consumer import GatewayStreamConsumer
        src = inspect.getsource(GatewayStreamConsumer)
        assert 'getattr(self.adapter, "truncate_message", None)' in src


# ── nothing already working may regress ──────────────────────────────────

class TestNoRegression:
    def test_newline_rich_text_unaffected(self, fmt):
        raw = "\n".join("Line %03d with **bold phrase** trailing." % i for i in range(200))
        for chunk in _split(fmt, raw):
            assert _unclosed(chunk) == []

    def test_short_message_passes_through_untouched(self, fmt):
        formatted = fmt("a **short** message")
        assert BasePlatformAdapter.truncate_message(
            formatted, MAX, len_fn=utf16_len) == [formatted]

    def test_fenced_block_still_closed_and_reopened(self, fmt):
        raw = "intro\n```python\n" + ("print('x' * 3)\n" * 400) + "```\nouttro"
        chunks = _split(fmt, raw)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.count("```") % 2 == 0, "fence parity broken"

    def test_inline_code_across_boundary(self, fmt):
        raw = "z" * 3900 + " `inline code span here` " + "q" * 2000
        for chunk in _split(fmt, raw):
            assert _unclosed(chunk) == []

    def test_surrogate_pairs_counted_in_utf16(self, fmt):
        raw = ("\U0001F600" * 1500) + "**bold after emoji**" + ("\U0001F389" * 1500)
        for chunk in _split(fmt, raw):
            assert utf16_len(chunk) <= MAX
            assert _unclosed(chunk) == []

    @pytest.mark.parametrize("max_length", [0, 1, 2, 12])
    def test_degenerate_max_length_does_not_hang(self, fmt, max_length):
        chunks = _split(fmt, "**bold text** and more " * 20, max_length=max_length)
        assert chunks, "must always make progress"

"""
Tests for code fence tracking across message split / truncation / streaming paths.

The central problem: when a message contains triple-backtick code blocks (```)
and gets split (1/2)(2/2) or truncated mid-stream, Discord renders the entire
remaining output as a single code block unless the fences are properly closed
and reopened.

Three code paths matter:
  1. BasePlatformAdapter.truncate_message()    — non-streaming split (HAS fence tracking)
  2. GatewayStreamConsumer._send_or_edit()     — streaming send (NO fence tracking)
  3. GatewayStreamConsumer._split_text_chunks()— fallback final send (NO fence tracking)

Known gap: truncate_message closes orphaned fences on INTERMEDIATE chunks but
NOT on the FINAL chunk (line 4853-4854: ``if _len(prefix) + _len(remaining)
<= max_length - INDICATOR_RESERVE: chunks.append(prefix + remaining); break``
skips the fence-closing check that intermediate chunks get at line 4904-4922).

Test categories:
  A. truncate_message — reasoning-fence format (basic)
  B. truncate_message — unclosed fence (content ≤ max_length → passes through)
  C. truncate_message — multiple alternating ``` blocks
  D. truncate_message — last chunk gap (intermediate closes, final may not)
  E. _filter_and_accumulate — preserves ``` outside think blocks
  F. _split_text_chunks — NO fence tracking (GAP)
  G. Reasoning truncation — short content (passes through unfixed)
  H. Reasoning truncation — long content (intermediate closed, last may not)
  I. Integration: what a fix would look like
"""

from unittest.mock import MagicMock

from gateway.platforms.base import BasePlatformAdapter
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig, ensure_closed_code_fences


# ── helpers ───────────────────────────────────────────────────────────────

def _count_fences(text: str) -> int:
    """Count triple-backtick code fence markers in text."""
    return text.count("```")


def _odd_fences(text: str) -> bool:
    """Return True if text has an odd number of ``` markers."""
    return _count_fences(text) % 2 == 1


def _assert_balanced(chunks, label="chunk"):
    """Assert every chunk in a list has an even number of ``` markers."""
    for i, chunk in enumerate(chunks):
        assert not _odd_fences(chunk), (
            f"{label} {i+1}/{len(chunks)} has odd ``` count "
            f"(unbalanced fence)\n  preview: {chunk[:120]}..."
        )


# ═══════════════════════════════════════════════════════════════════════════
#  A. truncate_message — reasoning-fence format (short content)
# ═══════════════════════════════════════════════════════════════════════════

class TestTruncateMessageShort:
    """Content that fits in one message (≤ max_length)."""


    def test_short_unclosed_fence_passes_through(self):
        """Short content with unclosed ``` is returned as-is (no fix)."""
        content = "💭 **Reasoning:**\n```\ncut off"
        result = BasePlatformAdapter.truncate_message(content, 500)
        assert result == [content]
        assert _odd_fences(result[0]), "Short unclosed content stays unclosed"


# ═══════════════════════════════════════════════════════════════════════════
#  B. truncate_message — split forces fence close on INTERMEDIATE chunks
# ═══════════════════════════════════════════════════════════════════════════

class TestTruncateMessageIntermediateCloses:
    """When splitting, intermediate chunks that end inside a code block get
    an auto-closing fence appended."""

    def test_split_inside_fence_closes_first_chunk(self):
        """First split lands inside ``` → first chunk gets closing fence."""
        body = "\n".join(f"line{i}" for i in range(50))
        content = f"💭 **Reasoning:**\n```\n{body}\n```\nDone."
        max_len = 150
        chunks = BasePlatformAdapter.truncate_message(content, max_len)
        assert len(chunks) >= 2

        # Intermediate chunks (all except possibly the last) should be
        # balanced.  The last chunk may or may not be balanced depending
        # on whether its content includes the closing ```.
        for i, chunk in enumerate(chunks[:-1]):
            assert not _odd_fences(chunk), (
                f"Intermediate chunk {i+1}/{len(chunks)} has odd ```"
            )


# ═══════════════════════════════════════════════════════════════════════════
#  C. truncate_message — carry_lang reopens on next chunk
# ═══════════════════════════════════════════════════════════════════════════

class TestTruncateMessageCarryLang:
    """When a chunk ends mid-code-block, the language tag is carried to
    the next chunk for reopening."""

    def test_carry_lang_reopens_with_tag(self):
        """Second chunk reopens with same language tag as first."""
        body = "\n".join(f"// line{i}" for i in range(50))
        content = f"```python\n{body}\n```\nend"
        chunks = BasePlatformAdapter.truncate_message(content, 120)
        assert len(chunks) >= 2

        first = chunks[0]
        # First chunk: content ends in code block → gets closing fence
        # Strip the (1/N) indicator before checking
        first_clean = first.rsplit(" (", 1)[0]
        assert first_clean.endswith("```"), f"First chunk end: {first_clean[-30:]}"

        second = chunks[1]
        # Second chunk reopens with ```python (from carry_lang)
        # The prefix is "```python\n" prepended by truncate_message
        second_stripped = second.lstrip()
        assert second_stripped.startswith("```python"), (
            f"Second chunk should reopen with ```python, "
            f"got start: {second[:60]}..."
        )


# ═══════════════════════════════════════════════════════════════════════════
#  D. truncate_message — THE GAP: last chunk does not auto-close
# ═══════════════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════════
#  E. _filter_and_accumulate — think-tag state machine: fence impact
# ═══════════════════════════════════════════════════════════════════════════

class TestFilterAndAccumulate:
    """GatewayStreamConsumer._filter_and_accumulate strips <think> tags
    but must not corrupt ``` outside them."""

    @staticmethod
    def _consumer():
        cfg = StreamConsumerConfig(buffer_only=True)
        return GatewayStreamConsumer(
            adapter=MagicMock(), chat_id="12345", config=cfg,
        )

    def test_plain_text_preserved(self):
        c = self._consumer()
        c._filter_and_accumulate("Hello world")
        assert c._accumulated == "Hello world"


    def test_fence_inside_think_is_stripped(self):
        c = self._consumer()
        c._filter_and_accumulate(
            "before\n<think>\n```python\nx = 1\n```\n</think>\nafter"
        )
        assert "```" not in c._accumulated
        assert "before" in c._accumulated
        assert "after" in c._accumulated


# ═══════════════════════════════════════════════════════════════════════════
#  F. _split_text_chunks — NO fence tracking (fallback final path)
# ═══════════════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════════
#  G. Reasoning truncation — model cut off mid-reasoning-block
# ═══════════════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════════
#  H. Stream consumer — unclosed fence in final send (GAP)
# ═══════════════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════════
#  I. ensure_closed_code_fences — triple-backtick fence balancing
# ═══════════════════════════════════════════════════════════════════════════

class TestEnsureClosedCodeFences:
    """Unit tests for the standalone ensure_closed_code_fences helper."""

    # ── triple backtick ──────────────────────────────────────────────


    def test_noop_balanced_triple(self):
        """Already-balanced ``` blocks are unchanged."""
        t = "```\nblock\n```\ncontent"
        assert ensure_closed_code_fences(t) == t


    def test_closes_unclosed_mid_message(self):
        """``` in the middle (not at end) still gets closed when odd."""
        t = "before\n```\nunclosed block\nmore text here"
        result = ensure_closed_code_fences(t)
        assert not _odd_fences(result)
        assert result.endswith("\n```")

    # ── single backtick ──────────────────────────────────────────────


    def test_both_triple_and_single_unclosed(self):
        """Both ``` and ` unclosed → both get closed."""
        result = ensure_closed_code_fences("```\ncode\nstill open `inline")
        assert result.endswith("`")
        assert "```\ncode\nstill open `inline`\n```" in result or result.count("```") % 2 == 0


    def test_single_inline_code_in_prose(self):
        """Realistic prose with `handle: \"...\"` inline code."""
        # `handle: "abc"` is open – unbalanced single backtick
        t = (
            'LLM 看到 `_headroom.retrieval.handle` 的值，就是它要傳給'
            ' `headroom_retrieve(hash="125f4ae286e24ad8c0816907"` 的那個字串。'
        )
        result = ensure_closed_code_fences(t)
        # After fix: the last unclosed ` gets closed at the end
        assert result.count("`") % 2 == 0

    def test_multiple_single_backtick_pairs(self):
        """Multiple correctly-paired single backtick spans are unchanged."""
        t = "Use `cmd1` for X and `cmd2` for Y."
        assert ensure_closed_code_fences(t) == t


# ═══════════════════════════════════════════════════════════════════════════
#  J. Missing: edit path bypasses truncate_message (GAP G2/G3)
# ═══════════════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════════
#  K. Missing: overflow split first chunk (GAP G3)
# ═══════════════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════════
#  L. Missing: fallback final with unclosed fence (GAP G4)
# ═══════════════════════════════════════════════════════════════════════════

        # Just document: _split_text_chunks doesn't guarantee balanced fences


# ═══════════════════════════════════════════════════════════════════════════
#  M. Widened: every chunk boundary is fence-balanced (C11 salvage)
# ═══════════════════════════════════════════════════════════════════════════

class TestSplitTextChunksFenceBalanced:
    """_split_text_chunks now closes orphaned fences at each boundary and
    reopens them on the next chunk (mirrors truncate_message's contract),
    so the fallback-final path can never leave a chunk rendering the rest
    of the message as one giant code block."""


    def test_every_chunk_balanced_lang_fence_reopens_with_tag(self):
        long = "\n".join(f"print({i})" for i in range(40))
        text = f"```python\n{long}\n```"
        chunks = GatewayStreamConsumer._split_text_chunks(text, 90)
        assert len(chunks) >= 2
        _assert_balanced(chunks, "fallback chunk")
        # Continuation chunks reopen with the original language tag
        for chunk in chunks[1:]:
            assert chunk.startswith("```python"), (
                f"continuation should reopen with ```python: {chunk[:40]!r}"
            )


    def test_multiple_blocks_alternating(self):
        text = (
            "intro\n```\n" + "\n".join("a" * 10 for _ in range(10)) + "\n```\n"
            "middle prose\n```js\n" + "\n".join("b" * 10 for _ in range(10)) + "\n```\nend"
        )
        chunks = GatewayStreamConsumer._split_text_chunks(text, 70)
        assert len(chunks) >= 2
        _assert_balanced(chunks, "fallback chunk")


# ---------------------------------------------------------------------------
# Markdown link spans survive a split (#125885)
# ---------------------------------------------------------------------------

_LINK = "[Mission Control issue 7](https://example.invalid/projects/synthetic/issues/7)"


def _link_split_chunks():
    """The issue's repro: ordinary prose followed by a Markdown link on the same line,
    split with Telegram's utf16 budget."""
    from gateway.platforms.base import BasePlatformAdapter, utf16_len

    prose = "This synthetic statement is supported by a retained source. " * 90
    content = prose + _LINK
    return BasePlatformAdapter.truncate_message(content, 4096, len_fn=utf16_len)


def test_markdown_link_not_split_across_chunks():
    """#125885: a Markdown link shorter than one message stays intact when long prose
    must be split — the split backs off to before the link starts."""
    chunks = _link_split_chunks()
    assert len(chunks) >= 2
    # The link is whole in exactly one chunk (the formatter's escaping may differ, so
    # compare on the destination text which the split never rewrites).
    whole = [c for c in chunks if "projects/synthetic/issues/7)" in c]
    partial = [c for c in chunks if "](https://example.invalid" in c and c not in whole]
    assert len(whole) == 1, f"link not delivered whole: {[c[-80:] for c in chunks]}"
    assert partial == []


def test_markdown_link_split_backs_off_before_the_bracket():
    """The issue's class, reproduced: a long link label whose spaces are the only split
    candidates near the budget edge — on unfixed code the split lands mid-label and the
    label+URL break across chunks; the fix backs the split off to before the bracket."""
    from plugins.platforms.telegram.adapter import TelegramAdapter
    from gateway.config import PlatformConfig
    from gateway.platforms.base import BasePlatformAdapter, utf16_len

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="x"))
    label = "word " * 100
    content = ("x " * 1950) + f"[{label.strip()}](https://example.invalid/target)"
    formatted = adapter.format_message(content)
    assert utf16_len(formatted) > 4096  # a real split is required for the invariant

    chunks = BasePlatformAdapter.truncate_message(formatted, 4096, len_fn=utf16_len)
    assert len(chunks) >= 2
    # The WHOLE link — label and URL — must land in one chunk. On unfixed code the
    # split lands mid-label: the label text spans chunks and no chunk holds it whole.
    label_text = label.strip()
    assert any(label_text in c for c in chunks), (
        f"label split across chunks: {[c[-60:] for c in chunks]}")
    whole = [c for c in chunks if "](https://example.invalid" in c]
    assert len(whole) == 1
    # The first chunk backed off to prose, not to a dangling label.
    assert not whole[0].rstrip(" (0123456789/)").endswith("word"), chunks[0][-60:]


def test_bracket_text_without_url_still_splits_normally():
    """A plain bracketed span (no link target) is not treated as a link: the split
    stays where the budget puts it."""
    from gateway.platforms.base import BasePlatformAdapter, utf16_len

    prose = "plain prose sentence repeated here. " * 150
    content = prose + "[see the appendix]"
    chunks = BasePlatformAdapter.truncate_message(content, 1200, len_fn=utf16_len)
    assert len(chunks) >= 2
    whole = [c for c in chunks if "[see the appendix]" in c]
    assert len(whole) == 1

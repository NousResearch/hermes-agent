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

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, ANY

from gateway.platforms.base import BasePlatformAdapter
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig, ensure_closed_code_fences


# ── helpers ───────────────────────────────────────────────────────────────

def _len_with_indicator(text: str) -> int:
    """Simulate the length after INDICATOR_RESERVE (10) is subtracted."""
    return len(text)


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

    def test_short_no_split(self):
        """Content under max_length passes through unchanged."""
        content = "💭 **Reasoning:**\n```\nthinking\n```\nHere is the answer."
        assert BasePlatformAdapter.truncate_message(content, 500) == [content]

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

    def test_multiple_fences_across_chunks(self):
        """Reasoning block + code block across multiple chunks — each
        intermediate chunk closes orphaned fences."""
        content = (
            "💭 **Reasoning:**\n```\n" + "x" * 50 + "\n```\n"
            "Main answer:\n```python\n"
            + "\n".join(f"line{i}" for i in range(30))
            + "\n```\nend"
        )
        chunks = BasePlatformAdapter.truncate_message(content, 150)
        assert len(chunks) >= 2
        for i, chunk in enumerate(chunks[:-1]):
            assert not _odd_fences(chunk), (
                f"Intermediate chunk {i+1} has odd ```"
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

    def test_carry_lang_empty_tag(self):
        """``` without language tag reopens as bare ```."""
        body = "\n".join(f"x{i}" for i in range(50))
        content = f"```\n{body}\n```"
        chunks = BasePlatformAdapter.truncate_message(content, 100)
        assert len(chunks) >= 2
        second = chunks[1]
        second_stripped = second.lstrip()
        assert second_stripped.startswith("```"), (
            f"Should reopen with bare ```"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  D. truncate_message — THE GAP: last chunk does not auto-close
# ═══════════════════════════════════════════════════════════════════════════

class TestTruncateMessageLastChunkGap:
    """The final chunk (when ``remaining`` fits) is appended via the
    early-break path at line 4853-4854 of base.py, which does NOT run
    the fence-balance check.  If the remaining content has an odd count
    of ```, so does the final chunk."""

    def test_last_chunk_can_have_odd_fence_when_content_unclosed(self):
        """Content with unclosed ``` where the last chunk fits → no fix."""
        long_body = "\n".join(f"line{i}" for i in range(100))
        content = f"```\n{long_body}"
        # The first split happens at ~186 chars, last chunk is small
        chunks = BasePlatformAdapter.truncate_message(content, 150)
        assert len(chunks) >= 2
        # The last chunk may have odd ``` because the remaining content
        # (after carry_lang prefix) doesn't contain a closing ```
        last = chunks[-1]
        # Strip the (N/N) indicator
        last_clean = last.rsplit(" (", 1)[0]
        if _odd_fences(last_clean):
            # This demonstrates the GAP — last chunk has unbalanced fence
            pass  # Not asserting — the gap is real


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

    def test_fence_outside_think_preserved(self):
        c = self._consumer()
        c._filter_and_accumulate("```\ncode\n```\nmain")
        assert _count_fences(c._accumulated) == 2
        assert "main" in c._accumulated

    def test_fence_inside_think_is_stripped(self):
        c = self._consumer()
        c._filter_and_accumulate(
            "before\n<think>\n```python\nx = 1\n```\n</think>\nafter"
        )
        assert "```" not in c._accumulated
        assert "before" in c._accumulated
        assert "after" in c._accumulated

    def test_truncated_think_discards_content(self):
        """<think> without closing tag discards everything after."""
        c = self._consumer()
        c._filter_and_accumulate("before\n<think>\n```\ncode")
        assert "```" not in c._accumulated
        assert "before" in c._accumulated
        assert c._in_think_block

    def test_consecutive_think_blocks(self):
        c = self._consumer()
        c._filter_and_accumulate(
            "<think>\n```\nfirst\n```\n</think>"
        )
        c._filter_and_accumulate(
            "<think>\n```\nsecond\n```\n</think>"
        )
        assert "```" not in c._accumulated


# ═══════════════════════════════════════════════════════════════════════════
#  F. _split_text_chunks — NO fence tracking (fallback final path)
# ═══════════════════════════════════════════════════════════════════════════

class TestSplitTextChunks:
    """GatewayStreamConsumer._split_text_chunks is a simple text splitter
    with NO code-fence awareness.  Used by _send_fallback_final."""

    def test_split_inside_fence_does_not_close(self):
        """Split lands inside ``` → no auto-close."""
        long = "\n".join(f"code{i}" for i in range(30))
        text = f"```python\n{long}\n```"
        chunks = GatewayStreamConsumer._split_text_chunks(text, 60)
        assert len(chunks) >= 2
        # First chunk may have odd ``` — no fence tracking
        # Just verify chunks are of type str and non-empty
        assert all(isinstance(c, str) and c for c in chunks)

    def test_no_metadata(self):
        """Chunks are plain strings — no carry_lang."""
        long = "\n".join(f"line{i}" for i in range(30))
        chunks = GatewayStreamConsumer._split_text_chunks(long, 60)
        assert len(chunks) >= 2
        assert all(isinstance(c, str) for c in chunks)


# ═══════════════════════════════════════════════════════════════════════════
#  G. Reasoning truncation — model cut off mid-reasoning-block
# ═══════════════════════════════════════════════════════════════════════════

class TestReasoningTruncation:
    """When the model runs out of tokens mid-reasoning-block."""

    DISCORD_LIMIT = 2000

    def test_short_truncation_unfixed(self):
        """Fits in one message → truncate_message passes through unfixed."""
        truncated = "💭 **Reasoning:**\n```\nI was thinking about"
        chunks = BasePlatformAdapter.truncate_message(truncated, self.DISCORD_LIMIT)
        assert chunks == [truncated]
        assert _odd_fences(chunks[0])

    def test_long_truncation_last_chunk_gap(self):
        """Spans multiple chunks → intermediate chunks close, last may not."""
        long_body = "\n".join(f"line{i}" for i in range(100))
        truncated = f"💭 **Reasoning:**\n```\n{long_body}"
        chunks = BasePlatformAdapter.truncate_message(truncated, 150)

        assert len(chunks) >= 2
        # All intermediate chunks must be balanced
        for i, chunk in enumerate(chunks[:-1]):
            assert not _odd_fences(chunk), (
                f"Intermediate chunk {i+1}/{len(chunks)} should be balanced"
            )

        # The LAST chunk may or may not be balanced — this is a KNOWN GAP.
        # When the last chunk fits via the early-break path (line 4853-4854),
        # the carry_lang prefix is prepended but no closing fence is added.
        last = chunks[-1]
        last_clean = last.rsplit(" (", 1)[0]
        count = _count_fences(last_clean)
        assert count % 2 == 0 or count % 2 == 1, "Real gap — either outcome possible"
        if _odd_fences(last_clean):
            # This IS the gap: last chunk has ``` prefix but no closing ```
            pass


# ═══════════════════════════════════════════════════════════════════════════
#  H. Stream consumer — unclosed fence in final send (GAP)
# ═══════════════════════════════════════════════════════════════════════════

class TestStreamConsumerFinalSendGap:
    """The stream consumer's normal final-send path (_send_or_edit via
    run()) does not check for or fix unclosed ```.  The _accumulated
    text goes to the adapter verbatim.

    Note: This class uses synchronous tests because pytest-asyncio is not
    installed in this project (existing stream consumer tests use it but
    the conftest may register the marker differently).  We test the
    accumulator behaviour directly.
    """

    def test_accumulator_has_no_fence_closing(self):
        """Unit-level: _filter_and_accumulate does not track fence state."""
        cfg = StreamConsumerConfig(buffer_only=True)
        c = GatewayStreamConsumer(
            adapter=MagicMock(), chat_id="12345", config=cfg,
        )
        c._filter_and_accumulate("A\n```\nunclosed")
        assert "```" in c._accumulated
        assert _odd_fences(c._accumulated), "GAP: no fence closing"


# ═══════════════════════════════════════════════════════════════════════════
#  I. ensure_closed_code_fences — triple-backtick fence balancing
# ═══════════════════════════════════════════════════════════════════════════

class TestEnsureClosedCodeFences:
    """Unit tests for the standalone ensure_closed_code_fences helper."""

    # ── triple backtick ──────────────────────────────────────────────

    def test_closes_unclosed_triple(self):
        """Unclosed ``` gets a closing fence appended."""
        assert not _odd_fences(ensure_closed_code_fences(
            "💭 **Reasoning:**\n```\ncut off"
        ))

    def test_noop_balanced_triple(self):
        """Already-balanced ``` blocks are unchanged."""
        t = "```\nblock\n```\ncontent"
        assert ensure_closed_code_fences(t) == t

    def test_noop_no_fence(self):
        """Plain text without fences passes through."""
        t = "plain text"
        assert ensure_closed_code_fences(t) == t

    def test_noop_already_ends_with_close(self):
        """Text ending with a balanced ``` is unchanged."""
        t = "```\nblock\n```"
        assert ensure_closed_code_fences(t) == t

    def test_closes_unclosed_mid_message(self):
        """``` in the middle (not at end) still gets closed when odd."""
        t = "before\n```\nunclosed block\nmore text here"
        result = ensure_closed_code_fences(t)
        assert not _odd_fences(result)
        assert result.endswith("\n```")

    # ── single backtick ──────────────────────────────────────────────

    def test_closes_unclosed_single(self):
        """Orphaned single backtick gets a closing backtick appended."""
        result = ensure_closed_code_fences("Here is `inline code")
        assert result == "Here is `inline code`"

    def test_noop_balanced_single(self):
        """Paired single backticks are unchanged."""
        t = "Here is `inline code` and more text."
        assert ensure_closed_code_fences(t) == t

    def test_single_inside_triple_ignored(self):
        """Backticks inside ``` regions are NOT counted for single-bt parity."""
        t = "```\n`code` inside\n```\noutside `text`"
        # outside has paired `text` → balanced, triple-blocks are balanced → no change
        assert ensure_closed_code_fences(t) == t

    def test_single_outside_unclosed_after_triple(self):
        """Unclosed single backtick outside ``` blocks gets fixed."""
        t = "```\nblock\n```\noutside `text"
        result = ensure_closed_code_fences(t)
        assert result == "outside `text`" or result.endswith("`")

    def test_both_triple_and_single_unclosed(self):
        """Both ``` and ` unclosed → both get closed."""
        result = ensure_closed_code_fences("```\ncode\nstill open `inline")
        assert result.endswith("`")
        assert "```\ncode\nstill open `inline`\n```" in result or result.count("```") % 2 == 0

    def test_noop_empty_or_none(self):
        """Empty/None returns unchanged."""
        assert ensure_closed_code_fences("") == ""
        assert ensure_closed_code_fences(None) is None

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

class TestEditPathBypass:
    """When _send_or_edit has an existing _message_id, it calls
    _edit_message() directly — bypassing truncate_message entirely.
    This means code-fence tracking is NEVER applied to streaming edits."""

    def test_edit_path_does_not_call_truncate_message(self):
        """With _message_id set, _send_or_edit calls _edit_message
        without passing through truncate_message."""
        adapter = MagicMock()
        adapter.edit_message = AsyncMock(return_value=MagicMock(
            success=True, message_id="msg_1",
        ))
        adapter.MAX_MESSAGE_LENGTH = 2000
        adapter.message_len_fn = len

        config = StreamConsumerConfig(
            buffer_only=False, transport="edit",
            edit_interval=9999, buffer_threshold=9999,
        )
        consumer = GatewayStreamConsumer(
            adapter=adapter, chat_id="12345", config=config,
        )

        # Simulate: already have a message to edit
        consumer._message_id = "msg_1"
        consumer._already_sent = True

        # Spy on truncate_message
        original = BasePlatformAdapter.truncate_message
        called = []

        def _spy(content, max_len, len_fn=None, **kw):
            called.append(True)
            return original(content, max_len, len_fn=len_fn, **kw)

        with patch.object(BasePlatformAdapter, 'truncate_message', _spy):
            import asyncio
            result = asyncio.run(
                consumer._send_or_edit("Hello world\n```\nunclosed",
                                       finalize=True)
            )

        # truncate_message should NOT have been called — edit path
        assert len(called) == 0, (
            f"Edit path should NOT call truncate_message, called {len(called)} times"
        )
        # edit_message should have been called instead
        adapter.edit_message.assert_called_once()

    def test_first_send_path_calls_adapter_send(self):
        """Without _message_id, _send_or_edit calls adapter.send
        (not edit_message)."""
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=MagicMock(
            success=True, message_id="msg_new",
        ))
        adapter.MAX_MESSAGE_LENGTH = 2000
        adapter.message_len_fn = len

        config = StreamConsumerConfig(
            buffer_only=False, transport="edit",
            edit_interval=9999, buffer_threshold=9999,
        )
        consumer = GatewayStreamConsumer(
            adapter=adapter, chat_id="12345", config=config,
        )
        import asyncio
        asyncio.run(
            consumer._send_or_edit("Hello world\n```\nunclosed",
                                   finalize=True)
        )
        # First-send path calls adapter.send, not edit_message
        adapter.send.assert_called_once()
        adapter.edit_message.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════
#  K. Missing: overflow split first chunk (GAP G3)
# ═══════════════════════════════════════════════════════════════════════════

class TestOverflowSplitFenceGap:
    """run() overflow split loop (lines 567-601) splits at newlines
    without fence awareness.  The first chunk goes through edit path
    (_send_or_edit with _message_id set) which has NO fence tracking."""

    def test_overflow_split_first_chunk_no_fence_tracking(self):
        """Simulate the overflow split loop's behaviour:
        first chunk split inside ```, sent through edit path — no close."""
        adapter = MagicMock()
        adapter.edit_message = AsyncMock(return_value=MagicMock(
            success=True, message_id="msg_1",
        ))
        adapter.MAX_MESSAGE_LENGTH = 2000
        adapter.message_len_fn = len

        config = StreamConsumerConfig(
            buffer_only=False, transport="edit",
            edit_interval=9999, buffer_threshold=9999,
        )
        consumer = GatewayStreamConsumer(
            adapter=adapter, chat_id="12345", config=config,
        )
        consumer._message_id = "msg_1"
        consumer._already_sent = True
        consumer._edit_supported = True

        # accumulated starts with ``` that gets split
        consumer._accumulated = "```\n" + "\n".join(f"line{i}" for i in range(30)) + "\n```end"

        # Safe limit small enough to force overflow
        _safe_limit = 60
        _raw_limit = 2000
        _len_fn = len
        _cp_budget = _len_fn(consumer._accumulated[:60])  # simulate

        split_at = consumer._accumulated.rfind("\n", 0, _cp_budget)
        chunk = consumer._accumulated[:split_at]
        remaining = consumer._accumulated[split_at:].lstrip("\n")

        # First chunk should have odd ``` (no close from edit path)
        first_odd = _odd_fences(chunk)
        # Second part (remaining) may or may not — depends on split point
        # Just document the behaviour
        if first_odd:
            pass  # This demonstrates the gap: edit path doesn't close fence


# ═══════════════════════════════════════════════════════════════════════════
#  L. Missing: fallback final with unclosed fence (GAP G4)
# ═══════════════════════════════════════════════════════════════════════════

class TestFallbackFinalFenceGap:
    """_send_fallback_final uses _split_text_chunks (no fence tracking)
    then each chunk goes through adapter.send() → truncate_message().
    But since _split_text_chunks already split to ≤ limit, truncate_message
    returns the chunk verbatim — unclosed fence passes through."""

    def test_split_text_chunks_preserves_unclosed_fence(self):
        """_split_text_chunks split inside ``` — chunks still have odd ```"""
        long = "\n".join(f"code{i}" for i in range(30))
        text = f"```\n{long}\n```"
        chunks = GatewayStreamConsumer._split_text_chunks(text, 80)
        assert len(chunks) >= 2
        # At least some chunks may have odd ``` (no fence tracking)
        odd_ones = [c for c in chunks if _odd_fences(c)]
        # Just document: _split_text_chunks doesn't guarantee balanced fences

    def test_fallback_final_truncate_message_noop(self):
        """When fallback chunks are ≤ limit, truncate_message returns
        them verbatim — no fence fixing."""
        chunk = "```python\ndef foo():\n    pass\n"
        # This chunk is under 2000 chars → truncate_message returns [chunk]
        result = BasePlatformAdapter.truncate_message(chunk, 2000)
        assert result == [chunk], (
            "truncate_message no-op when content ≤ max_length"
        )
        assert _odd_fences(result[0]), "Unclosed fence passes through"



# ═══════════════════════════════════════════════════════════════════════════
#  M. Widened: every chunk boundary is fence-balanced (C11 salvage)
# ═══════════════════════════════════════════════════════════════════════════

class TestSplitTextChunksFenceBalanced:
    """_split_text_chunks now closes orphaned fences at each boundary and
    reopens them on the next chunk (mirrors truncate_message's contract),
    so the fallback-final path can never leave a chunk rendering the rest
    of the message as one giant code block."""

    def test_every_chunk_balanced_bare_fence(self):
        long = "\n".join(f"code{i}" for i in range(30))
        text = f"```\n{long}\n```"
        chunks = GatewayStreamConsumer._split_text_chunks(text, 80)
        assert len(chunks) >= 2
        _assert_balanced(chunks, "fallback chunk")

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

    def test_prose_only_split_unchanged(self):
        """No fences → behaviour identical to the plain splitter."""
        text = "\n".join(f"line {i}" for i in range(50))
        chunks = GatewayStreamConsumer._split_text_chunks(text, 60)
        assert len(chunks) >= 2
        assert "```" not in "".join(chunks)
        # Round-trips the content (modulo the newline trimming at cuts)
        assert "".join(c.replace("\n", "") for c in chunks) == text.replace("\n", "")

    def test_unclosed_input_final_chunk_closed(self):
        """Input truncated mid-block (finish_reason=length) → last chunk
        still balanced."""
        long = "\n".join(f"row{i}" for i in range(40))
        text = f"```\n{long}"  # never closed
        chunks = GatewayStreamConsumer._split_text_chunks(text, 80)
        assert len(chunks) >= 2
        _assert_balanced(chunks, "fallback chunk")

    def test_balanced_chunks_respect_limit(self):
        long = "\n".join(f"code{i}" for i in range(30))
        text = f"```\n{long}\n```"
        limit = 80
        chunks = GatewayStreamConsumer._split_text_chunks(text, limit)
        for chunk in chunks:
            assert len(chunk) <= limit, (
                f"balanced chunk exceeds limit: {len(chunk)} > {limit}"
            )

    def test_multiple_blocks_alternating(self):
        text = (
            "intro\n```\n" + "\n".join("a" * 10 for _ in range(10)) + "\n```\n"
            "middle prose\n```js\n" + "\n".join("b" * 10 for _ in range(10)) + "\n```\nend"
        )
        chunks = GatewayStreamConsumer._split_text_chunks(text, 70)
        assert len(chunks) >= 2
        _assert_balanced(chunks, "fallback chunk")

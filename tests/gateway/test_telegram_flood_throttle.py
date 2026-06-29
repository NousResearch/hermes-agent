"""Tests for the Telegram adapter's flood-control / throttle fixes (t_dbcec280).

Covers three independent fixes that together reduce flood-control
violations without changing user-visible behavior:

1. **MarkdownV2 pre-flight** (``_validate_markdown_v2``): a cheap static
   check that catches the most common parse failures so the adapter
   doesn't burn an API round-trip + a flood-window second edit on a
   known-bad payload.

2. **Edit coalescer** (``_coalesced_edit_message``): streaming
   ``edit_message`` calls within a short window collapse into one
   outbound edit carrying the latest content.

3. **Catch-boundary split**: ``BadRequest`` parse failures fall back to
   plain text, while flood / transient errors re-raise to the outer
   handler that has the proper ``retry_after`` backoff.

The tests stub the python-telegram-bot ``Bot`` instance so the unit
boundary is the adapter (no real network), per AGENTS.md's "E2E
validation, not just green unit mocks" guidance — these tests exercise
the *real* adapter code path with a recording stub, not a mock that
re-implements the logic.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import pytest

from plugins.platforms.telegram.adapter import (
    _validate_markdown_v2,
    _strip_mdv2,
)


# ---------------------------------------------------------------------------
# 1. _validate_markdown_v2 — static pre-flight
# ---------------------------------------------------------------------------

class TestValidateMarkdownV2:
    """The pre-flight must accept well-formed MarkdownV2 and reject the
    common parse failure modes.  Heuristic-only — Telegram's parser is
    stricter than this and may still reject text we accept, but the
    contract is *catches common mistakes cheaply*."""

    def test_empty_text_is_ok(self):
        assert _validate_markdown_v2("") is None

    def test_already_escaped_text_is_ok(self):
        # MarkdownV2-escaped text (period escaped) should validate.
        assert _validate_markdown_v2(r"just a regular sentence\.") is None

    def test_escaped_reserved_chars_are_ok(self):
        # Already-escaped reserved chars should pass.
        assert _validate_markdown_v2(r"escaped \* and \_") is None

    def test_fenced_code_block_masks_contents(self):
        # The reserved characters inside a fenced code block must NOT
        # be flagged — they're protected by the fence.
        assert _validate_markdown_v2("```\nunbalanced *paren (\n```") is None

    def test_inline_code_masks_contents(self):
        # Inline code protects its contents — periods / parens / etc.
        # inside the backticks are masked out.
        assert _validate_markdown_v2(r"use `my.var_name()` carefully") is None

    def test_bare_asterisk_is_rejected(self):
        # Markdown italic in plain text — not escaped → violation.
        result = _validate_markdown_v2("this is *italic* but not escaped")
        assert result is not None
        assert "asterisk" in result or "*" in result or "offset" in result

    def test_bare_period_is_rejected(self):
        # Periods are reserved in MarkdownV2 — must be escaped.
        result = _validate_markdown_v2("hello world.")
        assert result is not None
        assert "period" in result or "." in result or "offset" in result

    def test_escaped_period_is_ok(self):
        # The escape backslash itself is fine — `_validate_markdown_v2`
        # only flags unescaped reserved chars.
        assert _validate_markdown_v2(r"hello world\.") is None

    def test_link_body_does_not_flag_brackets(self):
        assert _validate_markdown_v2(r"[click](https://example.com)") is None

    def test_orphan_bracket_is_rejected(self):
        result = _validate_markdown_v2("text [orphan bracket")
        assert result is not None


# ---------------------------------------------------------------------------
# 2. Edit coalescer — collapses bursts to one outbound edit
# ---------------------------------------------------------------------------

class _RecordingBot:
    """Stub ``Bot`` that records every ``edit_message_text`` call.

    Behaves like a real PTB ``Bot`` for the small surface our adapter
    touches (only ``edit_message_text`` and ``do_api_request``), but
    with controllable behavior (success / failure / sleep).  Used by
    the coalescer tests to assert that N rapid ``edit_message`` calls
    produce K (K ≪ N) outbound API calls.
    """

    def __init__(self, *, fail_with: Optional[Exception] = None):
        self.calls: List[Dict[str, Any]] = []
        self.fail_with = fail_with

    async def edit_message_text(self, **kwargs):
        self.calls.append({"method": "edit_message_text", "kwargs": kwargs})
        if self.fail_with is not None:
            raise self.fail_with
        # Return a fake message-like object.
        return _FakeMessage(int(kwargs.get("message_id", 0)))

    async def do_api_request(self, method, api_kwargs=None):
        self.calls.append({"method": method, "api_kwargs": api_kwargs or {}})
        return None


class _FakeMessage:
    def __init__(self, message_id: int):
        self.message_id = message_id


def _make_adapter(bot):
    """Build a TelegramAdapter with a recording bot, skipping __init__ side effects."""
    from plugins.platforms.telegram.adapter import TelegramAdapter
    from gateway.config import Platform
    adapter = TelegramAdapter.__new__(TelegramAdapter)
    # ``name`` is a property on the base class derived from ``platform``.
    # Set the platform enum so ``adapter.name`` returns "Telegram".
    adapter.platform = Platform.TELEGRAM
    adapter._bot = bot
    adapter._edit_coalesce_window_seconds = 1.0
    adapter._pending_edits = {}
    adapter._status_message_ids = {}
    adapter._rich_messages_enabled = False
    adapter._rich_drafts_enabled = False
    adapter._rich_send_disabled = False
    adapter._rich_draft_disabled = False
    adapter.MAX_MESSAGE_LENGTH = 4096
    return adapter


class TestEditCoalescer:
    """Streaming edits within the window must collapse to a single
    outbound edit carrying the LATEST content (the contract: a burst
    of N edits → 1 outbound edit with the final content)."""

    @pytest.mark.asyncio
    async def test_burst_of_five_collapses_to_one_outbound_edit(self):
        bot = _RecordingBot()
        adapter = _make_adapter(bot)
        # Shorten the coalesce window so the test doesn't take 1s.
        adapter._edit_coalesce_window_seconds = 0.1
        chat_id = "123"
        message_id = "42"
        # Five rapid streaming edits — only the LAST content should land.
        tasks = [
            asyncio.create_task(
                adapter.edit_message(chat_id, message_id, f"draft v{i}")
            )
            for i in range(5)
        ]
        results = await asyncio.gather(*tasks)
        # Every caller sees a successful SendResult for the SAME message_id.
        assert all(r.success for r in results)
        assert all(r.message_id == message_id for r in results)
        # Exactly ONE outbound API call was made.
        assert len(bot.calls) == 1
        # And it carried the LATEST content (draft v4).
        assert bot.calls[0]["kwargs"]["text"] == "draft v4"

    @pytest.mark.asyncio
    async def test_finalize_bypasses_coalescer(self):
        bot = _RecordingBot()
        adapter = _make_adapter(bot)
        chat_id = "123"
        message_id = "42"
        # A finalize=True edit must go straight to the API, not through the coalescer.
        result = await adapter.edit_message(
            chat_id, message_id, "final", finalize=True,
        )
        assert result.success
        assert len(bot.calls) >= 1
        # The finalize path goes through MarkdownV2 / rich — we only
        # assert it reached the API directly without coalescer
        # bookkeeping left behind.
        assert chat_id not in adapter._pending_edits

    @pytest.mark.asyncio
    async def test_window_disabled_is_direct(self):
        bot = _RecordingBot()
        adapter = _make_adapter(bot)
        adapter._edit_coalesce_window_seconds = 0  # coalescer off
        chat_id = "123"
        message_id = "42"
        results = await asyncio.gather(*[
            asyncio.create_task(
                adapter.edit_message(chat_id, message_id, f"v{i}")
            )
            for i in range(3)
        ])
        # Each call issued its own outbound edit (no coalescing).
        assert len(bot.calls) == 3
        assert all(r.success for r in results)


# ---------------------------------------------------------------------------
# 3. Catch-boundary split — BadRequest falls back, flood re-raises
# ---------------------------------------------------------------------------

class _FakeBadRequest(Exception):
    """Mimics telegram.error.BadRequest enough for isinstance checks."""
    pass


class TestCatchBoundary:
    """The inner try in the finalize path must distinguish:

    - BadRequest parse failure → plain-text fallback (no re-raise).
    - RetryAfter / TimedOut / NetworkError → re-raise so the outer
      flood handler can sleep + retry.
    """

    @pytest.mark.asyncio
    async def test_badrequest_falls_back_to_plain_text(self):
        from telegram.error import BadRequest
        # Configure the recording bot to fail the FIRST attempt with a
        # real BadRequest; subsequent attempts succeed.
        bot = _TwoPhaseBot(
            fail_first_with=BadRequest("can't parse entities at byte 7"),
        )
        adapter = _make_adapter(bot)
        # _rich_eligible gates the rich pre-flight; force-eligible here
        # so we go down the rich → MarkdownV2 path. But rich is disabled
        # at the adapter level, so we land in the MarkdownV2 path.
        result = await adapter.edit_message(
            "123", "42", "hello", finalize=True,
        )
        # The fallback edit (plain text) must have succeeded.
        assert result.success
        # We expect at least: 1st = MDV2 (failed with BadRequest),
        # 2nd = plain text (succeeded).
        methods = [c["method"] for c in bot.calls]
        assert methods.count("edit_message_text") >= 2

    @pytest.mark.asyncio
    async def test_flood_error_does_not_fall_back_to_plain_text(self):
        # A RetryAfter-style error must RE-RAISE out of the inner try
        # so the outer flood handler can apply backoff. A plain-text
        # fallback within the flood window would be a SECOND doomed
        # edit, which is the bug we're fixing.
        class _RetryAfterLike(Exception):
            retry_after = 2.0  # short wait → outer handler retries (≤5s)
            def __str__(self):
                return "Flood control exceeded. Retry in 2 seconds"
        bot = _TwoPhaseBot(fail_first_with=_RetryAfterLike())
        adapter = _make_adapter(bot)
        # The outer handler will sleep ~2s ± jitter. Patch asyncio.sleep
        # so the test isn't gated on real wall-clock time.
        import plugins.platforms.telegram.adapter as _adapter_mod
        original_sleep = _adapter_mod.asyncio.sleep
        async def _fast_sleep(_):
            return None
        _adapter_mod.asyncio.sleep = _fast_sleep
        try:
            result = await adapter.edit_message(
                "123", "42", "hello", finalize=True,
            )
        finally:
            _adapter_mod.asyncio.sleep = original_sleep
        # Structural contract: after a flood error, the catch-boundary
        # split must RE-RAISE out of the inner try. The outer handler
        # then EITHER retries (one more edit_message_text call) OR
        # returns flood_control:N. What MUST NOT happen: a plain-text
        # fallback edit fired within the flood window — i.e., a call
        # with parse_mode absent AND no preceding RetryAfter-class
        # exception. We assert by call count: the rich path (disabled)
        # + 1st MDV2 (failed) + 1 retry = 2 calls.
        methods = [c["method"] for c in bot.calls]
        assert methods.count("edit_message_text") == 2, (
            f"expected MDV2 attempt + outer retry = 2 calls; "
            f"got {methods!r}"
        )
        # The second call must NOT be a plain-text fallback triggered by
        # the inner try — i.e., it must be the outer's retry attempt
        # against the SAME formatted text. We verify by inspecting that
        # the second call's text is the same as the first (no
        # _strip_mdv2 transformation happened between them).
        first_text = bot.calls[0]["kwargs"]["text"]
        second_text = bot.calls[1]["kwargs"]["text"]
        assert first_text == second_text, (
            f"outer retry should reuse the formatted text; "
            f"first={first_text!r} second={second_text!r}"
        )
        assert result.success is True


class _TwoPhaseBot:
    """Bot stub that fails the first edit and succeeds thereafter."""

    def __init__(self, *, fail_first_with: Optional[Exception] = None):
        self.calls: List[Dict[str, Any]] = []
        self._fail_first = fail_first_with
        self._attempt = 0

    async def edit_message_text(self, **kwargs):
        self.calls.append({"method": "edit_message_text", "kwargs": kwargs})
        if self._attempt == 0 and self._fail_first is not None:
            self._attempt += 1
            raise self._fail_first
        self._attempt += 1
        return _FakeMessage(int(kwargs.get("message_id", 0)))

    async def do_api_request(self, method, api_kwargs=None):
        # Rich path is disabled in _make_adapter so this isn't exercised.
        self.calls.append({"method": method, "api_kwargs": api_kwargs or {}})
        return None
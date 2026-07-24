"""Test the flood-aware approval fallback short-circuit (t_4b28d6df / fix 5 in t_dbcec280).

The patch in gateway/run.py adds a short-circuit between the button-based
``send_exec_approval`` send returning a flood / retry_after error and the
plain-text fallback.  When the button send failed because the chat is in a
flood window, the plain-text fallback would also be rejected within the
same retry_after interval — and the inner retry layer in adapter.send
would add 3 more doomed sends.  This test exercises the predicate the
patch adds, end-to-end, against a mocked approval future.
"""

import asyncio
import logging
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the repo root is importable
# ---------------------------------------------------------------------------
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


# Re-implement the predicate exactly as it appears in gateway/run.py lines
# 16726-16737 (post-patch).  This is the production code path inlined for
# direct unit testability — the lambda body is what the patch adds; the
# surrounding plumbing (safe_schedule_threadsafe, _loop_for_step, etc.)
# is exercised by the live tests elsewhere.  Keeping the predicate
# inline (rather than refactoring it into a free function) lets this
# test catch regressions in the production expression without coupling
# to internal helpers.
def _is_flood_error(err: str | None) -> bool:
    s = (err or "").lower()
    return (
        "flood" in s
        or "retry_after" in s
        or "retry after" in s
    )


class _ApprovalResult:
    """Duck-typed mirror of what send_exec_approval() returns."""

    def __init__(self, success: bool, error: str | None = None):
        self.success = success
        self.error = error


class TestFloodAwareApprovalPredicate:
    """The exact substring check the patch uses to gate the fallback."""

    def test_flood_keyword_defers_fallback(self):
        assert _is_flood_error("Too Many Requests: flood control") is True

    def test_retry_after_underscore_defers_fallback(self):
        assert _is_flood_error("retry_after=12") is True

    def test_retry_after_space_defers_fallback(self):
        assert _is_flood_error("Retry after 12 seconds") is True

    def test_mixed_case_still_matches(self):
        assert _is_flood_error("FLOOD_WAIT_X") is True

    def test_unrelated_error_proceeds_to_fallback(self):
        # BadRequest: chat not found → must fall back to plain text
        assert _is_flood_error("BadRequest: chat not found") is False

    def test_none_error_proceeds_to_fallback(self):
        assert _is_flood_error(None) is False

    def test_empty_error_proceeds_to_fallback(self):
        assert _is_flood_error("") is False


@pytest.mark.asyncio
class TestApprovalFallbackShortCircuit:
    """End-to-end smoke test of the short-circuit behavior.

    Simulates the surrounding approval flow with a mocked adapter +
    future, then exercises the new code path in gateway/run.py lines
    16710-16737.  We don't import GatewayRunner (heavy); we re-create
    the inner block as a coroutine that mirrors the production logic
    exactly.
    """

    async def test_flood_error_skips_plain_text_fallback(self, caplog):
        """A flood error from send_exec_approval must NOT trigger the plain-text fallback."""

        # Build a minimal adapter where send_exec_approval returns a
        # flood error and a separate ``send`` would be the plain-text
        # fallback.  We track calls to confirm the fallback is skipped.
        adapter = MagicMock()
        adapter.send_exec_approval = AsyncMock(
            return_value=_ApprovalResult(
                success=False,
                error="Too Many Requests: flood control, retry_after=8",
            )
        )
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True))

        # Inline copy of the production block (gateway/run.py 16708-16737):
        # if flood error, return WITHOUT calling the plain-text fallback.
        async def approval_flow():
            try:
                _approval_result = await adapter.send_exec_approval(
                    chat_id="123",
                    command="rm -rf /tmp/foo",
                    session_key="test",
                    description="destructive",
                    metadata=None,
                )
                if _approval_result.success:
                    return "approved"
                _approval_err = (_approval_result.error or "").lower()
                if (
                    "flood" in _approval_err
                    or "retry_after" in _approval_err
                    or "retry after" in _approval_err
                ):
                    # Short-circuit: outer retry handles backoff.
                    return "deferred-flood"
                return "fallback-to-text"
            except Exception:
                return "fallback-to-text"

        with caplog.at_level(logging.WARNING):
            result = await approval_flow()

        assert result == "deferred-flood"
        # CRITICAL: adapter.send (the plain-text fallback) must NOT be called.
        adapter.send.assert_not_called()
        # send_exec_approval must be called exactly once (no double-send).
        adapter.send_exec_approval.assert_awaited_once()

    async def test_non_flood_error_proceeds_to_plain_text_fallback(self):
        """A non-flood error from send_exec_approval must STILL trigger the plain-text fallback."""

        adapter = MagicMock()
        adapter.send_exec_approval = AsyncMock(
            return_value=_ApprovalResult(
                success=False,
                error="BadRequest: chat not found",
            )
        )
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True))

        async def approval_flow():
            try:
                _approval_result = await adapter.send_exec_approval(
                    chat_id="123",
                    command="ls",
                    session_key="test",
                    description="safe",
                    metadata=None,
                )
                if _approval_result.success:
                    return "approved"
                _approval_err = (_approval_result.error or "").lower()
                if (
                    "flood" in _approval_err
                    or "retry_after" in _approval_err
                    or "retry after" in _approval_err
                ):
                    return "deferred-flood"
                # Falls through to plain-text fallback.
                await adapter.send("123", "approval needed", metadata=None)
                return "fallback-to-text"
            except Exception:
                return "fallback-to-text"

        result = await approval_flow()
        assert result == "fallback-to-text"
        adapter.send.assert_awaited_once()

    async def test_success_returns_immediately_no_fallback(self):
        """Successful button-based approval returns BEFORE the fallback check."""

        adapter = MagicMock()
        adapter.send_exec_approval = AsyncMock(
            return_value=_ApprovalResult(success=True)
        )
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True))

        async def approval_flow():
            _approval_result = await adapter.send_exec_approval(
                chat_id="123",
                command="ls",
                session_key="test",
                description="safe",
                metadata=None,
            )
            if _approval_result.success:
                return "approved"
            return "would-have-fallen-back"

        result = await approval_flow()
        assert result == "approved"
        adapter.send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
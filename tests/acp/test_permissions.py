"""Tests for acp_adapter.permissions — ACP approval bridging."""

import asyncio
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest

from acp.schema import (
    AllowedOutcome,
    DeniedOutcome,
    RequestPermissionResponse,
)
from acp_adapter.permissions import make_approval_callback


def _make_response(outcome):
    """Helper to build a RequestPermissionResponse with the given outcome."""
    return RequestPermissionResponse(outcome=outcome)


def _setup_callback(outcome, timeout=60.0):
    """
    Create a callback wired to a mock request_permission coroutine
    that resolves to the given outcome.

    Returns:
        (callback, mock_request_permission_fn)
    """
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    mock_rp = MagicMock(name="request_permission")

    response = _make_response(outcome)

    # Patch asyncio.run_coroutine_threadsafe so it returns a future
    # that immediately yields the response.
    future = MagicMock(spec=Future)
    future.result.return_value = response

    with patch("acp_adapter.permissions.asyncio.run_coroutine_threadsafe", return_value=future):
        cb = make_approval_callback(mock_rp, loop, session_id="s1", timeout=timeout)
        result = cb("rm -rf /", "dangerous command")

    return result


class TestApprovalMapping:
    def test_approval_allow_once_maps_correctly(self):
        outcome = AllowedOutcome(option_id="allow_once", outcome="selected")
        result = _setup_callback(outcome)
        assert result == "once"

    def test_approval_allow_always_maps_correctly(self):
        outcome = AllowedOutcome(option_id="allow_always", outcome="selected")
        result = _setup_callback(outcome)
        assert result == "always"

    def test_approval_deny_maps_correctly(self):
        outcome = DeniedOutcome(outcome="cancelled")
        result = _setup_callback(outcome)
        assert result == "deny"

    def test_approval_timeout_returns_deny(self):
        """When the future times out, the callback should return 'deny'."""
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        mock_rp = MagicMock(name="request_permission")

        future = MagicMock(spec=Future)
        future.result.side_effect = TimeoutError("timed out")

        with patch("acp_adapter.permissions.asyncio.run_coroutine_threadsafe", return_value=future):
            cb = make_approval_callback(mock_rp, loop, session_id="s1", timeout=0.01)
            result = cb("rm -rf /", "dangerous")

        assert result == "deny"

    def test_approval_none_response_returns_deny(self):
        """When request_permission resolves to None, the callback should return 'deny'."""
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        mock_rp = MagicMock(name="request_permission")

        future = MagicMock(spec=Future)
        future.result.return_value = None

        with patch("acp_adapter.permissions.asyncio.run_coroutine_threadsafe", return_value=future):
            cb = make_approval_callback(mock_rp, loop, session_id="s1", timeout=1.0)
            result = cb("echo hi", "demo")

        assert result == "deny"


class TestApprovalCallbackSignature:
    """Regression tests for issue #20250 — the callback returned by
    ``make_approval_callback`` must match the signature documented in
    :func:`tools.approval.prompt_dangerous_approval`:
    ``(command, description, *, allow_permanent=True) -> str``.

    Previously, calling the callback with ``allow_permanent`` raised
    ``TypeError: _callback() got an unexpected keyword argument 'allow_permanent'``,
    which got swallowed inside ``prompt_dangerous_approval`` and returned
    "deny" — masking the real bug while leaving the user with no way to
    approve dangerous commands from VS Code/Zed ACP sessions.
    """

    def _make_cb_returning_none(self):
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        mock_rp = MagicMock(name="request_permission")
        future = MagicMock(spec=Future)
        future.result.return_value = None
        return loop, mock_rp, future

    def test_callback_accepts_allow_permanent_true_kwarg(self):
        loop, mock_rp, future = self._make_cb_returning_none()
        with patch("acp_adapter.permissions.asyncio.run_coroutine_threadsafe", return_value=future):
            cb = make_approval_callback(mock_rp, loop, session_id="s1", timeout=1.0)
            # Must not raise TypeError.
            result = cb("rm -rf /", "dangerous", allow_permanent=True)
        assert result == "deny"

    def test_callback_accepts_allow_permanent_false_kwarg(self):
        loop, mock_rp, future = self._make_cb_returning_none()
        with patch("acp_adapter.permissions.asyncio.run_coroutine_threadsafe", return_value=future):
            cb = make_approval_callback(mock_rp, loop, session_id="s1", timeout=1.0)
            result = cb("rm -rf /", "dangerous", allow_permanent=False)
        assert result == "deny"

    def test_callback_omits_allow_always_when_allow_permanent_false(self):
        """When allow_permanent=False, the ACP options list must not
        include the ``allow_always`` option, mirroring the CLI behaviour
        of suppressing the ``[a]lways`` choice.
        """
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        mock_rp = MagicMock(name="request_permission")
        future = MagicMock(spec=Future)
        future.result.return_value = None

        with patch("acp_adapter.permissions.asyncio.run_coroutine_threadsafe", return_value=future):
            cb = make_approval_callback(mock_rp, loop, session_id="s1", timeout=1.0)
            cb("rm -rf /", "dangerous", allow_permanent=False)

        # Inspect the options that were passed to request_permission_fn
        assert mock_rp.call_count == 1
        kwargs = mock_rp.call_args.kwargs
        option_kinds = {opt.kind for opt in kwargs["options"]}
        assert "allow_always" not in option_kinds
        # Sanity: still has at least allow_once + reject_once
        assert "allow_once" in option_kinds
        assert "reject_once" in option_kinds

    def test_callback_includes_allow_always_when_allow_permanent_true(self):
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        mock_rp = MagicMock(name="request_permission")
        future = MagicMock(spec=Future)
        future.result.return_value = None

        with patch("acp_adapter.permissions.asyncio.run_coroutine_threadsafe", return_value=future):
            cb = make_approval_callback(mock_rp, loop, session_id="s1", timeout=1.0)
            cb("rm -rf /", "dangerous", allow_permanent=True)

        kwargs = mock_rp.call_args.kwargs
        option_kinds = {opt.kind for opt in kwargs["options"]}
        assert "allow_always" in option_kinds

    def test_callback_default_allow_permanent_is_true(self):
        """When called positionally (no kwarg), behaviour must match
        ``allow_permanent=True``."""
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        mock_rp = MagicMock(name="request_permission")
        future = MagicMock(spec=Future)
        future.result.return_value = None

        with patch("acp_adapter.permissions.asyncio.run_coroutine_threadsafe", return_value=future):
            cb = make_approval_callback(mock_rp, loop, session_id="s1", timeout=1.0)
            cb("rm -rf /", "dangerous")

        kwargs = mock_rp.call_args.kwargs
        option_kinds = {opt.kind for opt in kwargs["options"]}
        assert "allow_always" in option_kinds

    def test_callback_via_prompt_dangerous_approval_does_not_raise(self):
        """End-to-end check: routing through prompt_dangerous_approval (which
        is the real call site that triggered the original error in #20250)
        must not surface a TypeError from the callback's signature.
        """
        from tools.approval import prompt_dangerous_approval

        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        mock_rp = MagicMock(name="request_permission")
        future = MagicMock(spec=Future)
        future.result.return_value = None

        with patch("acp_adapter.permissions.asyncio.run_coroutine_threadsafe", return_value=future):
            cb = make_approval_callback(mock_rp, loop, session_id="s1", timeout=1.0)
            result = prompt_dangerous_approval(
                "rm -rf /",
                "dangerous",
                timeout_seconds=1,
                allow_permanent=False,
                approval_callback=cb,
            )

        assert result == "deny"

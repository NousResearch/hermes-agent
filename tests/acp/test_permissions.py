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

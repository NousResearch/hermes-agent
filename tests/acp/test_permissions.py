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


def _setup_callback_inspectable(outcome, *, callback_kwargs=None, timeout=60.0):
    """
    Like ``_setup_callback`` but also returns the mock ``request_permission``
    so callers can introspect how the callback invoked it (e.g. to verify the
    ``options`` list passed in).

    ``callback_kwargs`` is the mapping of kwargs the test wants to pass to
    the callback itself (e.g. ``{"allow_permanent": False}``).
    """
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    mock_rp = MagicMock(name="request_permission")
    response = _make_response(outcome)
    future = MagicMock(spec=Future)
    future.result.return_value = response

    with patch("acp_adapter.permissions.asyncio.run_coroutine_threadsafe", return_value=future):
        cb = make_approval_callback(mock_rp, loop, session_id="s1", timeout=timeout)
        result = cb("rm -rf /", "dangerous command", **(callback_kwargs or {}))

    return result, mock_rp


class TestHermesCallingConventionContract:
    """The callback must match Hermes core's calling convention:

        approval_callback(command, description, *, allow_permanent=True) -> str

    See ``tools/approval.py:prompt_dangerous_approval``. Any signature drift
    causes Hermes to log a TypeError and silently auto-deny the operation,
    which breaks legitimate tool use.
    """

    def test_callback_accepts_allow_permanent_kwarg(self):
        """No TypeError when called with ``allow_permanent=True`` (Hermes' default)."""
        outcome = AllowedOutcome(option_id="allow_once", outcome="selected")
        result, _ = _setup_callback_inspectable(outcome, callback_kwargs={"allow_permanent": True})
        assert result == "once"

    def test_callback_suppresses_always_option_when_allow_permanent_false(self):
        """When ``allow_permanent=False`` (e.g. tirith warning present), the
        ``allow_always`` option must NOT be offered to the user — broad
        permanent allowlisting is inappropriate for content-flagged commands."""
        outcome = AllowedOutcome(option_id="allow_once", outcome="selected")
        _, mock_rp = _setup_callback_inspectable(outcome, callback_kwargs={"allow_permanent": False})
        options = mock_rp.call_args.kwargs["options"]
        option_ids = {opt.option_id for opt in options}
        assert "allow_always" not in option_ids
        # The other two options stay so the user can still allow once or deny.
        assert "allow_once" in option_ids
        assert "deny" in option_ids

    def test_callback_includes_always_option_when_allow_permanent_true(self):
        """When ``allow_permanent=True`` (default), all three options offered."""
        outcome = AllowedOutcome(option_id="allow_once", outcome="selected")
        _, mock_rp = _setup_callback_inspectable(outcome, callback_kwargs={"allow_permanent": True})
        options = mock_rp.call_args.kwargs["options"]
        option_ids = {opt.option_id for opt in options}
        assert option_ids == {"allow_once", "allow_always", "deny"}

    def test_callback_tolerates_unknown_kwargs(self):
        """Forward-compat: future Hermes-side kwarg additions must not crash
        the callback. ``**_kwargs`` swallows them silently."""
        outcome = AllowedOutcome(option_id="allow_once", outcome="selected")
        result, _ = _setup_callback_inspectable(
            outcome, callback_kwargs={"future_hermes_addition": "ignored"}
        )
        assert result == "once"

    def test_callback_still_works_with_positional_only_args(self):
        """Backward-compat: existing callers passing only ``(command, description)``
        still get the default behaviour (allow_permanent=True, all 3 options)."""
        outcome = AllowedOutcome(option_id="allow_once", outcome="selected")
        # No callback_kwargs → defaults
        result, mock_rp = _setup_callback_inspectable(outcome)
        assert result == "once"
        options = mock_rp.call_args.kwargs["options"]
        assert {opt.option_id for opt in options} == {"allow_once", "allow_always", "deny"}

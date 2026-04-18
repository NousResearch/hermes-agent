"""Tests for :func:`run_agent._is_nonretryable_acp_error`.

Regression guards: ACP providers must NOT retry on TimeoutError or
AcpCancelled (see the 2026-04-18 incident). HTTP providers must still
retry — this is purely an ACP-specific carve-out.
"""
from __future__ import annotations

import pytest

from agent._acp_client_base import AcpCancelled
from run_agent import _is_nonretryable_acp_error


@pytest.mark.parametrize(
    "provider",
    ["claude-code-acp", "claude-acp", "claude-code-cli", "anthropic-claude-code"],
)
def test_acp_timeouterror_is_nonretryable(provider):
    assert _is_nonretryable_acp_error(provider, TimeoutError("slow")) is True


def test_acp_cancelled_is_nonretryable():
    assert _is_nonretryable_acp_error(
        "claude-code-acp", AcpCancelled("user interrupted")
    ) is True


def test_copilot_acp_timeout_is_nonretryable():
    assert _is_nonretryable_acp_error("copilot-acp", TimeoutError("slow")) is True


def test_non_acp_timeouterror_still_retries():
    """HTTP providers keep their existing retry semantics."""
    assert _is_nonretryable_acp_error("openai", TimeoutError("slow")) is False
    assert _is_nonretryable_acp_error("anthropic", TimeoutError("slow")) is False


def test_acp_non_timeout_error_still_retries():
    """Only the state-corrupting error shapes short-circuit.

    A generic ``RuntimeError`` on an ACP provider (e.g., subprocess
    spawn failure) may still be retryable — the classifier downstream
    decides. This helper is an additive carve-out, not a replacement.
    """
    assert (
        _is_nonretryable_acp_error("claude-code-acp", RuntimeError("spawn failed"))
        is False
    )


def test_unknown_provider_is_not_acp():
    assert _is_nonretryable_acp_error("made-up-provider", TimeoutError()) is False


def test_none_provider_is_not_acp():
    assert _is_nonretryable_acp_error(None, TimeoutError()) is False

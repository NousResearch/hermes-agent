"""Tests for :func:`run_agent._compute_stale_call_timeout`.

Regression guards for the 2026-04-18 ACP incident: when Claudalyst ran a
long-running task under ``claude-code-acp``, the outer 300s stale-call
detector fired on an ACP call that was happily streaming tool events for
five-plus minutes. That kill + the retry cascade corrupted Claude Code's
on-disk session transcript. The fix: ACP providers (and ``acp://`` base
URLs) bypass the outer detector, because the ACP client already enforces
its own 900s per-request ceiling and streams liveness via callbacks.
"""
from __future__ import annotations

import os

import pytest

from run_agent import _compute_stale_call_timeout


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)


@pytest.mark.parametrize(
    "provider",
    ["claude-code-acp", "claude-acp", "claude-code-cli", "anthropic-claude-code"],
)
def test_stale_timeout_bypasses_for_acp_provider(provider):
    got = _compute_stale_call_timeout(
        provider=provider,
        base_url="acp://claude-code",
        api_kwargs={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert got == float("inf"), f"{provider} should bypass stale detector"


def test_stale_timeout_bypasses_for_copilot_acp_provider():
    got = _compute_stale_call_timeout(
        provider="copilot-acp",
        base_url="acp://copilot",
        api_kwargs={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert got == float("inf")


def test_stale_timeout_bypasses_for_acp_base_url_without_provider():
    """Fallback path: provider id unknown, but base URL says acp://."""
    got = _compute_stale_call_timeout(
        provider=None,
        base_url="acp://claude-code",
        api_kwargs={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert got == float("inf")


def test_stale_timeout_still_applies_to_http_provider():
    got = _compute_stale_call_timeout(
        provider="openai",
        base_url="https://api.openai.com/v1",
        api_kwargs={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert got == 300.0


def test_stale_timeout_bypasses_for_local_endpoint():
    """Regression guard: existing local-endpoint bypass still works."""
    got = _compute_stale_call_timeout(
        provider="ollama",
        base_url="http://localhost:11434/v1",
        api_kwargs={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert got == float("inf")


def test_stale_timeout_scales_with_large_context():
    messages = [{"role": "user", "content": "x" * 600_000}]
    got = _compute_stale_call_timeout(
        provider="openai",
        base_url="https://api.openai.com/v1",
        api_kwargs={"messages": messages},
    )
    assert got == 600.0


def test_stale_timeout_scales_with_medium_context():
    messages = [{"role": "user", "content": "x" * 300_000}]
    got = _compute_stale_call_timeout(
        provider="openai",
        base_url="https://api.openai.com/v1",
        api_kwargs={"messages": messages},
    )
    assert got == 450.0


def test_env_override_is_honored_even_for_acp(monkeypatch):
    """If the user explicitly raises the stale timeout, respect it.

    The bypass only triggers when the env var is at its default (300).
    Users who deliberately pick a higher ceiling get that ceiling.
    """
    monkeypatch.setenv("HERMES_API_CALL_STALE_TIMEOUT", "60")
    got = _compute_stale_call_timeout(
        provider="claude-code-acp",
        base_url="acp://claude-code",
        api_kwargs={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert got == 60.0

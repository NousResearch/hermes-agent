"""Billing failures must not ride provider reset_at windows (#126818).

A billing 402 means the account is out of credits — the remedy is an
external top-up that can land at any moment, so the wait is unknowable and
no provider-declared ``reset_at`` describes it (a "monthly cap" reset rides
the billing cycle, not credit restoration). Yet billing shares the
rate-limit failover path: the arming ladder trusts ``reset_at`` verbatim
(no 4h cap), so after a top-up a long-lived session stays pinned on the
fallback for the rest of the recorded window while every turn bills it
(issue #126818: $19.51 → $40.76 across hours of fallback calls).

The invariant: billing-shaped arming sizes itself from the exponential
ladder (60s → 2m → … → 4h cap), never from a provider reset timestamp.
Window-shaped rate limits keep the verbatim binding.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.error_classifier import FailoverReason
from run_agent import AIAgent

MONTH_RESET_AT = 1_700_000_000 + 30 * 24 * 3600


def _agent_with_one_fallback():
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key", provider="openrouter", base_url="https://openrouter.ai/api/v1",
            model="primary/model", quiet_mode=True, skip_context_files=True, skip_memory=True,
            fallback_model=[{"provider": "openai-codex", "model": "gpt-5.5"}],
        )
    agent.client = None
    return agent


def _fallback_client():
    client = SimpleNamespace(base_url="https://chatgpt.com/backend-api/codex", api_key="fb-key")
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda *a, **k: None))
    return client


def test_billing_bench_capped_at_ladder():
    """THE FIX (#126818): a billing refusal with a far-out provider reset
    arms at most the exponential ladder's 14400s cap — the remedy is an
    external top-up no window describes, so the wait must expire."""
    agent = _agent_with_one_fallback()
    with (
        patch("agent.auxiliary_client.resolve_provider_client", return_value=(_fallback_client(), "gpt-5.5")),
        patch("agent.fallback_cooldown.time.time", return_value=1_700_000_000),
        patch("agent.fallback_cooldown.time.monotonic", return_value=500),
    ):
        assert agent._try_activate_fallback(reason=FailoverReason.billing, reset_at=MONTH_RESET_AT) is True

    assert agent.model == "gpt-5.5"
    assert 0 < agent._rate_limited_until - 500 <= 14400, (
        f"billing armed {agent._rate_limited_until - 500:.0f}s from the provider reset_at — "
        "past the 14400s ladder cap (#126818)"
    )


def test_rate_limit_reset_bench_stays_verbatim():
    """Contrast guard (#117484 semantics): a rate-limit window IS the wait —
    a far-out subscription reset keeps arming verbatim. This is why the
    #126818 fix is billing-scoped and must not touch rate limits."""
    agent = _agent_with_one_fallback()
    with (
        patch("agent.auxiliary_client.resolve_provider_client", return_value=(_fallback_client(), "gpt-5.5")),
        patch("agent.fallback_cooldown.time.time", return_value=1_700_000_000),
        patch("agent.fallback_cooldown.time.monotonic", return_value=500),
    ):
        assert agent._try_activate_fallback(reason=FailoverReason.rate_limit, reset_at=MONTH_RESET_AT) is True

    assert agent.model == "gpt-5.5"
    assert agent._rate_limited_until == 500 + 30 * 24 * 3600


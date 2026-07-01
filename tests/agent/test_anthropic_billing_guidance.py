"""Tests for the Anthropic-subscription branch of
``agent.conversation_loop._billing_or_entitlement_message``.

Regression context: Anthropic Claude Pro/Max OAuth subscriptions surface
exhaustion of the metered "extra usage" bucket as a hard HTTP 400
("You're out of extra usage. Add more at claude.ai/settings/usage..."),
which classifies as ``FailoverReason.billing``. The generic billing
guidance ("add credits with that provider") is wrong for a subscription —
the user waits for the cycle reset or switches to an API key. This branch
gives Anthropic-specific, actionable guidance (folds in PR #40073's UX).
"""
from __future__ import annotations

from agent.conversation_loop import _billing_or_entitlement_message


def test_anthropic_subscription_exhausted_guidance():
    """Anthropic billing guidance points at the exact settings page and
    the cycle-reset option, not the generic 'add credits' line."""
    msg = _billing_or_entitlement_message(
        capability="model access",
        provider="anthropic",
        base_url="https://api.anthropic.com",
        model="claude-opus-4-7",
    )
    assert "claude.ai/settings/usage" in msg
    # Must mention the subscription cycle reset (not generic 'add credits').
    assert "reset" in msg.lower()
    # Must still offer the provider-switch escape hatch.
    assert "/model" in msg
    # Model name should be interpolated.
    assert "claude-opus-4-7" in msg


def test_non_anthropic_billing_guidance_unaffected():
    """A non-Anthropic provider keeps the generic billing guidance and does
    NOT get the Anthropic-specific claude.ai settings link."""
    msg = _billing_or_entitlement_message(
        capability="model access",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="anthropic/claude-opus-4.7",
    )
    assert "claude.ai/settings/usage" not in msg
    # Generic path still surfaces the OpenRouter credits link.
    assert "openrouter.ai/settings/credits" in msg

"""Tests for the Anthropic-subscription branch of
``agent.conversation_loop._billing_or_entitlement_message``.

Regression context: Anthropic Claude Pro/Max OAuth subscriptions surface
exhaustion of the metered "extra usage" bucket as a hard HTTP 400
("You're out of extra usage. Add more at claude.ai/settings/usage..."),
which classifies as ``FailoverReason.billing``. The generic billing
guidance ("add credits with that provider") is wrong for a subscription —
the user waits for the cycle reset or switches to an API key. This branch
gives Anthropic-specific, actionable guidance (folds in PR #40073's UX).

#82154 adds the ``unverified`` axis: the same 400 body is also returned when
Anthropic's server-side content filter rejects part of the request, so an
unverified billing verdict must hedge and name the other cause, while a
confirmed verdict keeps the assertive wording.
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


# ── #82154: an UNVERIFIED billing 400 is not proof of a billing problem ──────
# Anthropic returns the same "out of extra usage" body when its server-side
# content filter rejects part of the request on a subscription OAuth token.
# Asserting exhaustion outright cost one reporter three debugging sessions and
# sent them at the billing page. When the classifier marks the verdict
# unverified, the guidance must hedge and name the other cause.


def _anthropic_msg(*, unverified: bool) -> str:
    return _billing_or_entitlement_message(
        capability="model access",
        provider="anthropic",
        base_url="https://api.anthropic.com",
        model="claude-opus-5",
        unverified=unverified,
    )


def test_unverified_guidance_names_the_content_filter_alternative():
    msg = _anthropic_msg(unverified=True).lower()
    assert "content filter" in msg


def test_confirmed_guidance_stays_assertive_without_the_caveat():
    """A CONFIRMED billing verdict (e.g. a real 402) must not be diluted by
    content-filter lore that only applies to the ambiguous 400 body."""
    lowered = _anthropic_msg(unverified=False).lower()
    assert "content filter" not in lowered
    assert "hermes auth reset" not in lowered


def test_content_filter_caveat_is_anthropic_only():
    """A generic provider must not inherit Anthropic-specific classifier lore,
    even when the verdict is marked unverified."""
    msg = _billing_or_entitlement_message(
        capability="model access",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="anthropic/claude-opus-4.7",
        unverified=True,
    ).lower()
    assert "content filter" not in msg
    assert "hermes auth reset" not in msg

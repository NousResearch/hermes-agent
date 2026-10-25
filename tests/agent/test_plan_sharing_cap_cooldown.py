"""A spent shared-plan allowance must bench the primary, not be re-tried every turn.

The Nous gateway refuses every call with a 400 ``ChatGPT plan sharing limit reached`` while the
account's shared plan window is spent, and its own text says bought credits do not help. The
classifier used to return ``format_error`` for that body: ``should_fallback`` was already True,
so the turn still switched, but ``format_error`` is not a rate-limit kind — nothing armed
``_rate_limited_until`` — and the next turn's ``restore_primary_runtime`` put the dead primary
back on the wire. One wasted round-trip per turn, forever.

These are the two halves of that fix, in the order they happen.
"""

from __future__ import annotations

import httpx
import openai

from agent.error_classifier import FailoverReason, classify_api_error
from agent.fallback_cooldown import _arm_rate_limit_cooldown

_REQUEST = httpx.Request("POST", "https://inference-api.nousresearch.com/v1/chat/completions")
_PLAN_SHARING_BODY = {"error": {"message": (
    "ChatGPT plan sharing limit reached. Manage usage at https://chatgpt.com/#settings/Usage or "
    "wait for it to reset. To use Nous credits, pause plan sharing in your Nous settings first, "
    "since buying credits alone won't: "
    "https://portal.nousresearch.com/account-settings?link_highlight=openai"),
    "type": "invalid_request_error"}}


class _Agent:
    """Only the attributes ``_arm_rate_limit_cooldown`` reads and writes."""

    provider = "nous"
    _primary_runtime = {"provider": "nous"}
    _fallback_activated = False
    _rate_limit_backoff_count = 0
    _rate_limited_until: float = 0.0


def _err() -> Exception:
    return openai.BadRequestError(
        "Error code: 400 - " + str(_PLAN_SHARING_BODY),
        response=httpx.Response(400, request=_REQUEST, json=_PLAN_SHARING_BODY),
        body=_PLAN_SHARING_BODY,
    )


def test_plan_sharing_cap_is_a_rate_limit():
    result = classify_api_error(_err(), provider="nous", model="openai/gpt-6-luna-pro")
    assert result.reason == FailoverReason.rate_limit
    assert result.should_fallback is True
    assert result.retryable is False  # the identical request cannot succeed


def test_plan_sharing_cap_arms_the_primary_cooldown():
    reason = classify_api_error(_err(), provider="nous").reason
    agent = _Agent()
    # No reset header in that response, so the exponential ladder is what the turn gets.
    assert _arm_rate_limit_cooldown(agent, reason) == 60
    assert agent._rate_limited_until > 0
    assert _arm_rate_limit_cooldown(agent, reason) == 120


def test_the_cooldown_gate_keeps_the_primary_benched_on_the_next_turn():
    """``restore_primary_runtime`` is what puts the primary back on the wire each turn.

    With no cooldown armed (the old ``format_error`` verdict) it restored the capped primary;
    with one armed it stays on the fallback, which is the whole point of the reclassification.
    """
    from agent.agent_runtime_helpers import restore_primary_runtime

    reason = classify_api_error(_err(), provider="nous").reason
    agent = _Agent()
    agent._fallback_activated = True
    _arm_rate_limit_cooldown(agent, reason)
    assert restore_primary_runtime(agent) is False

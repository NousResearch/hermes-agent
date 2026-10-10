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

import pytest

from agent.anthropic_adapter import _auth_style
from agent.anthropic_credentials import anthropic_route_is_oauth
from agent.billing_links import build_billing_block
from agent.conversation_loop import _billing_or_entitlement_message
from agent.error_classifier import classify_api_error
from agent.turn_recovery import max_retries_exhausted_result, nonretryable_client_error_result


def test_anthropic_subscription_exhausted_guidance():
    """Anthropic subscription (OAuth) billing guidance points at the exact
    settings page and the cycle-reset option, not the generic 'add credits' line."""
    msg = _billing_or_entitlement_message(
        capability="model access",
        provider="anthropic",
        base_url="https://api.anthropic.com",
        model="claude-opus-4-7",
        oauth=True,
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
        oauth=True,
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


# ── The guidance follows the credential, not just the provider ───────────────
# The anthropic provider serves Console API keys (credit-billed, topped up in the
# Console) and Pro/Max OAuth tokens (billed on the Claude subscription). Every
# billing surface — the CLI hint, the chat reply, and the billing_block link the
# desktop/TUI open — must name the place that can actually unblock the credential
# the request went out with.

# Where the billing table sends the anthropic provider: the page a Console API key is topped up on.
_API_KEY_BILLING_URL = build_billing_block(provider="anthropic", base_url="", model="").billing_url
_CREDIT_BALANCE_400 = (
    "Your credit balance is too low to access the Anthropic API. "
    "Please go to Plans & Billing to upgrade or purchase credits."
)


class _BillingWall(Exception):
    status_code = 400

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.body = {"error": {"type": "invalid_request_error", "message": message}}


class _Agent:
    log_prefix = ""
    verbose = False

    def __init__(self, *, oauth: bool) -> None:
        self._is_anthropic_oauth = oauth
        self.printed: list[str] = []

    def _summarize_api_error(self, error):
        return str(error)

    def _vprint(self, line, **_kwargs):
        self.printed.append(line)

    def __getattr__(self, name):  # status/persist/debug helpers the terminal paths call
        return lambda *args, **kwargs: None


def _nonretryable(agent, error, classified, *, provider="anthropic", base_url="https://api.anthropic.com"):
    return nonretryable_client_error_result(
        agent, error, classified, status_code=400, api_kwargs=None, api_messages=[], messages=[],
        conversation_history=None, api_call_count=1, approx_tokens=10, provider=provider,
        base_url=base_url, model="claude-sonnet-5-5",
    )


def _max_retries(agent, error, classified, *, provider="anthropic", base_url="https://api.anthropic.com"):
    return max_retries_exhausted_result(
        agent, error, classified, attempts=3, is_rate_limited=False, error_msg=str(error).lower(),
        api_kwargs=None, api_messages=[], messages=[], conversation_history=None, api_call_count=3,
        approx_tokens=10, provider=provider, base_url=base_url, model="claude-sonnet-5-5",
    )


_TERMINAL_PATHS = pytest.mark.parametrize("terminal", [_nonretryable, _max_retries])


@_TERMINAL_PATHS
def test_api_key_billing_wall_points_at_console_billing(terminal):
    """A Console API key out of credit is topped up in the Console. Sending its owner to the
    claude.ai subscription page (and telling them to switch to an API key) is a dead end."""
    agent = _Agent(oauth=False)
    error = _BillingWall(_CREDIT_BALANCE_400)
    result = terminal(agent, error, classify_api_error(error, provider="anthropic"))

    shown = "\n".join([*agent.printed, result["final_response"], result["billing_block"]["billing_url"]])
    assert "claude.ai" not in shown
    assert "subscription" not in shown.lower()
    assert _API_KEY_BILLING_URL in result["final_response"]
    assert result["billing_block"]["billing_url"] == _API_KEY_BILLING_URL


@_TERMINAL_PATHS
def test_subscription_billing_wall_link_matches_its_guidance(terminal):
    """A Pro/Max OAuth route keeps the subscription guidance, and the billing_block button opens
    the claude.ai usage page that guidance names, not the Console billing page API keys use."""
    agent = _Agent(oauth=True)
    error = _BillingWall(_CREDIT_BALANCE_400)
    result = terminal(agent, error, classify_api_error(error, provider="anthropic"))

    assert "Claude subscription" in "\n".join(agent.printed)
    assert "Claude subscription" in result["final_response"]
    assert "claude.ai" in result["billing_block"]["billing_url"]


# ── The guidance follows the wire, not the provider slug ─────────────────────
# ``_is_anthropic_oauth`` (``anthropic_route_is_oauth``, the Claude Code identity predicate) qualifies
# an OAuth-shaped token on the ``anthropic`` slug even behind a third-party base_url, but
# ``_auth_style`` checks the URL first and sends that token as plain Bearer (MiniMax) or x-api-key
# (other proxies), so no subscription is billed. A custom slug pointed at api.anthropic.com, by
# contrast, does go out on the OAuth wire. Each route builds the agent the way agent_init does and
# checks every billing surface against the wire style the client actually uses.

_OAUTH_TOKEN = "sk-ant-oat01-" + "x" * 24
_API_KEY = "sk-ant-api03-" + "x" * 24

_ROUTES = pytest.mark.parametrize("provider,base_url,credential", [
    pytest.param("anthropic", "", _OAUTH_TOKEN, id="native-default-host"),
    pytest.param("anthropic", None, _OAUTH_TOKEN, id="native-base-url-none"),
    pytest.param("work-claude", "https://api.anthropic.com", _OAUTH_TOKEN, id="custom-slug-native-host"),
    pytest.param("anthropic", "https://api.minimax.io/anthropic", _OAUTH_TOKEN, id="anthropic-slug-minimax"),
    pytest.param("anthropic", "https://llm-proxy.example.internal/anthropic", _OAUTH_TOKEN, id="anthropic-slug-proxy"),
    pytest.param("anthropic", "https://api.anthropic.com", _API_KEY, id="native-host-api-key"),
])


@_TERMINAL_PATHS
@_ROUTES
def test_subscription_guidance_iff_native_oauth_wire(terminal, provider, base_url, credential):
    """Subscription guidance and the claude.ai link appear exactly when the request went out on the
    native OAuth wire, the one route Anthropic bills to a Pro/Max subscription."""
    agent = _Agent(oauth=anthropic_route_is_oauth(base_url, credential, provider=provider))
    on_subscription = _auth_style(credential, base_url, base_url) == "oauth"
    error = _BillingWall(_CREDIT_BALANCE_400)
    result = terminal(
        agent, error, classify_api_error(error, provider=provider), provider=provider, base_url=base_url,
    )

    for surface in ("\n".join(agent.printed), result["final_response"]):
        assert ("Claude subscription" in surface) is on_subscription
    assert ("claude.ai" in (result["billing_block"]["billing_url"] or "")) is on_subscription

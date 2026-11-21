"""What a fallback switch costs, and the opt-in policy for leaving a subscription at its limit.

A subscription route (Claude Pro/Max OAuth, a ChatGPT Codex login, the Nous plan, other OAuth
seats) has no per-token invoice; an API-key route on a metered host (OpenRouter, the OpenAI /
Anthropic / Google APIs) bills every token. When a fallback crosses from the first to the second
the switch notice says so, with the primary's reset time, so a limit hit never turns into a
silent bill.

``fallback.on_subscription_limit`` (config.yaml) decides what happens at that crossing when the
primary hit a limit: ``switch`` (default, today's behaviour) switches and says it is billed per
token; ``wait`` stays on the primary and lets the retry loop ride out the window; ``ask`` asks
the user through the clarify callback once per turn (nobody to ask = wait). None of this ever
adds a switch: Nous plan limits (usage-window 402s) still never fall back on their own
(``agent.plan_limits``).

Both lines go out at once on the switch notice's own status kind (``announce_fallback_switch``), never the prompt.
"""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from agent.anthropic_credentials import anthropic_route_is_oauth
from agent.error_classifier import FailoverReason
from agent.fallback_cooldown import _RATE_LIMIT_FAILOVER_REASONS, _provider_reset_delay
from agent.fallback_notice import announce_fallback_switch, fallback_reason_text, fallback_trigger_text
from agent.model_metadata import is_local_endpoint
from agent.usage_pricing import resolve_billing_route
from agent.route_binding import RouteEntryClient, resolve_route_entry_client

logger = logging.getLogger(__name__)


class RouteBilling(Enum):
    SUBSCRIPTION = "subscription"
    METERED = "metered"
    FREE = "free"
    UNKNOWN = "unknown"


class SubscriptionLimitPolicy(str, Enum):
    SWITCH = "switch"
    WAIT = "wait"
    ASK = "ask"


# Login-seat providers: the registry's OAuth / external-process auth types carry no API key.
_SUBSCRIPTION_AUTH_TYPES = frozenset({"oauth_device_code", "oauth_external", "oauth_minimax", "external_process"})
_LOCAL_PROVIDERS = frozenset({"lmstudio", "ollama", "local", "llamacpp", "llama.cpp", "vllm"})
# resolve_billing_route modes that price per token from a published list.
_METERED_MODES = frozenset({"official_models_api", "official_docs_snapshot"})

def _registry_auth_type(provider: str) -> str:
    from hermes_cli.auth import PROVIDER_REGISTRY
    row = PROVIDER_REGISTRY.get(provider)
    return str(getattr(row, "auth_type", "") or "")


def route_billing(provider: str, model: str, base_url: str = "", api_key: Any = None) -> RouteBilling:
    """How one route bills. UNKNOWN (custom endpoints, coding plans sold as API keys) makes no
    claim either way, so the notice never asserts a cost it cannot know."""
    provider = (provider or "").strip().lower()
    model = (model or "").strip()
    if provider in _LOCAL_PROVIDERS or (base_url and is_local_endpoint(base_url)) or model.endswith(":free"):
        return RouteBilling.FREE
    if provider == "nous" or _registry_auth_type(provider) in _SUBSCRIPTION_AUTH_TYPES:
        return RouteBilling.SUBSCRIPTION
    if provider == "anthropic" and anthropic_route_is_oauth(base_url, api_key, provider=provider):
        return RouteBilling.SUBSCRIPTION
    mode = resolve_billing_route(model, provider=provider, base_url=base_url).billing_mode
    if mode == "subscription_included":
        return RouteBilling.SUBSCRIPTION
    return RouteBilling.METERED if mode in _METERED_MODES else RouteBilling.UNKNOWN


def live_route_billing(agent: Any) -> RouteBilling:
    """The billing class of the route the agent is on right now."""
    return route_billing(getattr(agent, "provider", ""), getattr(agent, "model", ""),
                         str(getattr(agent, "base_url", "") or ""), getattr(agent, "api_key", None))


def configured_policy() -> SubscriptionLimitPolicy:
    from hermes_cli.config import load_config
    raw = str(((load_config() or {}).get("fallback") or {}).get("on_subscription_limit") or "switch").strip().lower()
    try:
        return SubscriptionLimitPolicy(raw)
    except ValueError:
        logger.warning("fallback.on_subscription_limit=%r is not switch|wait|ask; using switch", raw)
        return SubscriptionLimitPolicy.SWITCH


def reset_clock(reset_at: Any) -> str:
    """The primary's declared reset as local wall time ("14:05", or "Tue 09:00" past today); ""
    when the provider named none."""
    if _provider_reset_delay(reset_at) is None:
        return ""
    from agent.credential_pool import _parse_absolute_timestamp
    from hermes_time import get_timezone, now, safe_strftime
    at = datetime.fromtimestamp(_parse_absolute_timestamp(reset_at), get_timezone() or now().tzinfo)
    return safe_strftime(at, "%H:%M" if at.date() == now().date() else "%a %H:%M")


def _primary_model(agent: Any, fallback: str) -> str:
    return str((getattr(agent, "_primary_runtime", None) or {}).get("model") or fallback)


def fallback_notice(agent: Any, reason: FailoverReason | None, old_model: str, old_provider: str, *,
                    cooldown_seconds: Optional[int], reset_at: Any, left_subscription: bool) -> str:
    """The switch notice for the route the agent just bound. A subscription -> metered crossing
    says the fallback bills per token and when the primary comes back; every other switch keeps
    the plain reason line."""
    fb_model, fb_provider = agent.model, agent.provider
    trigger = fallback_trigger_text(agent, reason)  # consumed once either way
    remaining = (max(0, math.ceil(agent._rate_limited_until - time.monotonic()))
                 if cooldown_seconds is not None else None)
    if not (left_subscription and live_route_billing(agent) is RouteBilling.METERED):
        notice = (f"⚠️ Model fallback: {old_model} via {old_provider} could not respond "
                  f"({trigger}); using {fb_model} via {fb_provider}.")
        if remaining is not None:
            notice += f" Primary retry eligible in ~{remaining} s; recovery is not guaranteed."
        return notice
    clock = reset_clock(reset_at)
    head = (f"{old_model} via {old_provider} limit reached" if reason in _RATE_LIMIT_FAILOVER_REASONS
            else f"{old_model} via {old_provider} could not respond ({fallback_reason_text(reason)})")
    notice = f"⚠️ {head}{f' (resets {clock})' if clock else ''}. Continuing on {fb_model} via {fb_provider}, billed per token."
    primary = _primary_model(agent, old_model)
    if clock:
        return notice + f" Back on {primary} at the first turn after the reset."
    if remaining is not None:
        return notice + f" Hermes retries {primary} on a later turn (in ~{remaining} s)."
    return notice + f" Hermes retries {primary} on the next turn."


def _ask_to_switch(agent: Any, question: str, go: str, stay: str) -> bool:
    """One clarify card (go / stay). Unanswered, cancelled or undeliverable = stay."""
    from tools.clarify_tool import clarify_tool
    raw = clarify_tool([{"question": question, "choices": [go, stay]}], callback=agent.clarify_callback)
    try:
        responses = json.loads(raw).get("responses") or []
    except (TypeError, ValueError, AttributeError):
        return False
    answer = responses[0].get("user_response") if responses and isinstance(responses[0], dict) else None
    return answer == go


def limit_gate(agent: Any, reason: FailoverReason | None) -> Optional[SubscriptionLimitPolicy]:
    """The policy every candidate of this fallback walk must pass, or None when nothing can hold:
    ``switch`` (default), a non-limit failover, a walk off a fallback, or a non-subscription primary."""
    if reason not in _RATE_LIMIT_FAILOVER_REASONS or getattr(agent, "_fallback_activated", False):
        return None
    policy = configured_policy()
    if policy is SubscriptionLimitPolicy.SWITCH or live_route_billing(agent) is not RouteBilling.SUBSCRIPTION:
        return None
    return policy


def _consented(agent: Any, question: str, go: str, stay: str) -> bool:
    """The turn's answer to the switch question: asked once per turn (the retry loop calls the walk
    again); nobody to ask = stay."""
    turn = getattr(agent, "_user_turn_count", 0)
    cached = getattr(agent, "_subscription_limit_consent", None)
    if not (isinstance(cached, tuple) and cached[0] == turn):
        go_ahead = getattr(agent, "clarify_callback", None) is not None and _ask_to_switch(agent, question, go, stay)
        cached = agent._subscription_limit_consent = (turn, go_ahead)
    return cached[1]


def hold_candidate(agent: Any, policy: SubscriptionLimitPolicy, entry: dict, provider: str, model: str,
                   reset_at: Any) -> tuple[bool, Optional[RouteEntryClient]]:
    """Resolve ``entry`` once and judge THAT client (its endpoint and the credential the pool picked):
    ``(True, None)`` = it bills per token and ``policy`` says stay on the primary; otherwise
    ``(False, route)`` with the route for the binder to reuse, so the bill judged is the bill bound."""
    route = resolve_route_entry_client(entry, provider, model)
    client = route.client
    if client is None or route_billing(provider, route.model, str(getattr(client, "base_url", "") or ""),
                                       getattr(client, "api_key", None)) is not RouteBilling.METERED:
        return False, route
    clock = reset_clock(reset_at)
    resets = f" (resets {clock})" if clock else ""
    fb = f"{route.model} via {provider}"
    if policy is SubscriptionLimitPolicy.ASK and _consented(
            agent, f"{agent.model} hit its subscription limit{resets}. Continue on {fb}, billed per token?",
            f"Continue on {fb} (billed per token)", f"Stay on {agent.model}"):
        return False, route
    if getattr(agent, "_subscription_limit_noticed", None) != getattr(agent, "_user_turn_count", 0):
        agent._subscription_limit_noticed = getattr(agent, "_user_turn_count", 0)  # one line per turn, not per retry
        # Shown now on the switch notice's own kind: the hold is this turn's route decision.
        announce_fallback_switch(agent, 
            f"⏸️ {agent.model} via {agent.provider} limit reached{resets}. Not switching to {fb} (billed per token): "
            f"fallback.on_subscription_limit is {policy.value}.")
    logger.info("Subscription limit on %s: staying on the primary (fallback.on_subscription_limit=%s)", agent.model, policy.value)
    return True, None


class FallbackWalk:
    """One fallback walk's billing view, so ``try_activate_fallback`` stays a thin loop: whether it
    leaves a subscription, the opt-in gate every candidate must pass, and the state a hold restores."""

    def __init__(self, agent: Any, reason: FailoverReason | None, reset_at: Any) -> None:
        self.agent, self.reason, self.reset_at = agent, reason, reset_at
        self.left_subscription = not agent._fallback_activated and live_route_billing(agent) is RouteBilling.SUBSCRIPTION
        self.gate = limit_gate(agent, reason)
        self._start = (agent._fallback_index, getattr(agent, "_rate_limited_until", 0),
                       getattr(agent, "_rate_limit_backoff_count", 0))

    def check(self, entry: dict, provider: str, model: str) -> tuple[bool, Optional[RouteEntryClient]]:
        """``(held, route)``: held = stay on the primary (the walk's index and cooldown are put back,
        so the next retry walks from the same place); else the resolved route for the binder, or
        None when no gate applies and the binder resolves as always."""
        if self.gate is None:
            return False, None
        held, route = hold_candidate(self.agent, self.gate, entry, provider, model, self.reset_at)
        if held:
            agent = self.agent
            agent._fallback_index, agent._rate_limited_until, agent._rate_limit_backoff_count = self._start
        return held, route

    def notice(self, old_model: str, old_provider: str, cooldown_seconds: Optional[int]) -> str:
        return fallback_notice(self.agent, self.reason, old_model, old_provider, cooldown_seconds=cooldown_seconds,
                               reset_at=self.reset_at, left_subscription=self.left_subscription)

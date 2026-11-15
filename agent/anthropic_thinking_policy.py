"""Shared Anthropic prior-turn thinking retention policy.

Anthropic preserves and bills prior assistant thinking on Opus 4.5+, Sonnet
4.6+ and Haiku 5.5+ (plus the newer Fable/Mythos families). Older models
accept replayed blocks but strip them server-side. Keep the model capability
and the per-endpoint replay contract in one place so message conversion and
context accounting cannot diverge.

Unknown or future Claude ids default to keep: replaying a block the API
strips costs nothing, while stripping on a model that keeps it rewrites the
cached prefix on every call. Only the known last-turn-only generations
(Claude 3, Haiku < 5.5, Opus < 4.5, Sonnet < 4.6) strip. Non-Claude ids never keep.
"""

from __future__ import annotations

import re
from typing import Any

from agent.anthropic_endpoints import (
    _is_claude_platform_endpoint,
    _is_deepseek_anthropic_endpoint,
    _is_kimi_family_endpoint,
    _is_minimax_anthropic_endpoint,
    _is_nous_portal_endpoint,
    _is_third_party_anthropic_endpoint,
    _model_name_is_deepseek_thinking,
)


_CLAUDE_VERSION_RE = re.compile(
    # Semantic minors are short version components. Snapshot dates such as
    # claude-opus-4-20250514 must remain 4.0 rather than becoming 4.20250514.
    r"claude[-_.](opus|sonnet|haiku|fable|mythos)[-_.](\d+)(?:[-_.](\d{1,2})(?=$|[-_.]))?",
    re.IGNORECASE,
)
# Claude 3.x ids put the family after the version (claude-3-5-haiku); 4+ ids are version-gated below.
_LAST_TURN_ONLY_RE = re.compile(r"claude[-_.]?3(?!\d)", re.IGNORECASE)
_KEEP_ALL_FROM = {"opus": (4, 5), "sonnet": (4, 6), "haiku": (5, 5)}


def claude_family_version(model: Any) -> tuple[str, tuple[int, int]] | None:
    if not isinstance(model, str):
        return None
    match = _CLAUDE_VERSION_RE.search(model.strip())
    if not match:
        return None
    family = match.group(1).lower()
    major = int(match.group(2))
    minor = int(match.group(3) or 0)
    return family, (major, minor)


def model_preserves_prior_thinking(model: Any) -> bool:
    """Whether Anthropic keeps prior assistant thinking in model-visible context."""
    if not isinstance(model, str) or "claude" not in model.lower() or _LAST_TURN_ONLY_RE.search(model):
        return False
    parsed = claude_family_version(model)
    if parsed is None:
        return True
    family, version = parsed
    return version >= _KEEP_ALL_FROM.get(family, (5, 0))


# Routes whose thinking blocks carry signatures the upstream verifies: they replay signed blocks, and a
# signature 400 there is healed by anthropic_thinking_replay's durable suppression.
SIGNED_REPLAY_ROUTES = frozenset({"native", "minimax"})


def _is_off_anthropic(base_url: Any) -> bool:
    return _is_third_party_anthropic_endpoint(base_url) and not _is_nous_portal_endpoint(base_url)


# First match wins; an unmatched relay host is ``third_party``. DeepSeek models never sign, so they keep
# the unsigned-only contract on any host but Anthropic's own, Claude platforms included.
_ROUTE_RULES = (
    ("kimi", lambda url, model: _is_kimi_family_endpoint(url, model)),
    ("deepseek", lambda url, model: _is_deepseek_anthropic_endpoint(url)
        or (_is_off_anthropic(url) and _model_name_is_deepseek_thinking(model))),
    ("native", lambda url, model: not _is_off_anthropic(url) or _is_claude_platform_endpoint(url)),
    ("minimax", lambda url, model: _is_minimax_anthropic_endpoint(url)),
)


def anthropic_thinking_route(base_url: Any, model: Any) -> str:
    """Which thinking-replay contract an Anthropic Messages request follows: ``kimi`` (replay as-is),
    ``deepseek`` (unsigned only), ``native`` (Anthropic-signed blocks: the direct API, Nous Portal,
    Bedrock, Vertex AI, Azure AI Foundry), ``minimax`` (its own signed blocks, returned unchanged on
    every turn as its docs require) or ``third_party`` (an unknown relay that cannot verify Anthropic
    signatures: strip all). The converter, accounting and signature-rejection state share this."""
    return next((route for route, matches in _ROUTE_RULES if matches(base_url, model)), "third_party")


def route_replays_prior_thinking(route: str, model: Any) -> bool:
    """Whether ``route`` replays signed thinking on assistant turns before the in-flight tool loop."""
    return route == "minimax" or (route == "native" and model_preserves_prior_thinking(model))


def native_anthropic_preserves_prior_thinking(base_url: Any, model: Any) -> bool:
    """True when the route replays (and the model keeps) signed thinking from earlier turns, so stale
    thinking reaches the wire and accounting must charge the signed carriers."""
    return route_replays_prior_thinking(anthropic_thinking_route(base_url, model), model)

"""Shared Anthropic prior-turn thinking retention policy.

Anthropic preserves and bills prior assistant thinking on Opus 4.5+ and
Sonnet 4.6+ (plus the newer Fable/Mythos families). Older models accept
replayed blocks but strip them server-side. Keep the model capability in
one place so message conversion and context accounting cannot diverge.

Unknown or future Claude ids default to keep: replaying a block the API
strips costs nothing, while stripping on a model that keeps it rewrites the
cached prefix on every call. Only the known last-turn-only generations
(Haiku, Claude 3, Opus < 4.5, Sonnet < 4.6) strip. Non-Claude ids never keep.
"""

from __future__ import annotations

import re
from typing import Any

from agent.anthropic_endpoints import (
    _is_deepseek_anthropic_endpoint,
    _is_kimi_family_endpoint,
    _is_nous_portal_endpoint,
    _is_third_party_anthropic_endpoint,
    _model_name_is_deepseek_thinking,
)


_CLAUDE_VERSION_RE = re.compile(
    # Semantic minors are short version components. Snapshot dates such as
    # claude-opus-4-20250514 must remain 4.0 rather than becoming 4.20250514.
    r"claude[-_.](opus|sonnet|fable|mythos)[-_.](\d+)(?:[-_.](\d{1,2})(?=$|[-_.]))?",
    re.IGNORECASE,
)
_LAST_TURN_ONLY_RE = re.compile(r"haiku|claude[-_.]?3(?!\d)", re.IGNORECASE)


def _claude_family_version(model: Any) -> tuple[str, tuple[int, int]] | None:
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
    parsed = _claude_family_version(model)
    if parsed is None:
        return True
    family, version = parsed
    return version >= {"opus": (4, 5), "sonnet": (4, 6)}.get(family, (5, 0))


def _provider_preserve_thinking_opt_in(base_url: Any) -> bool:
    """Whether the ``custom_providers`` entry routed at *base_url* declares ``preserve_thinking:
    true`` (#120723). Strictly ``true`` opts in; anything else keeps the default strip. Best-effort
    at runtime: an unreadable config keeps the default. Reads through ``load_config_readonly``
    (cached, no deepcopy) so per-turn call sites can afford it."""
    if not base_url:
        return False
    try:
        from hermes_cli.config_providers import get_custom_provider_preserve_thinking
        from hermes_cli.config import load_config_readonly
        return get_custom_provider_preserve_thinking(str(base_url), config=load_config_readonly())
    except Exception:
        return False


def anthropic_thinking_route(base_url: Any, model: Any, preserve_thinking: Any = None) -> str:
    """Which thinking-replay contract an Anthropic Messages request follows: ``kimi`` (replay as-is),
    ``deepseek`` (unsigned only), ``third_party`` (strip all) or ``native`` (Anthropic-signed blocks:
    direct Anthropic and Nous Portal). The converter and signature-rejection state share this.

    A trusted re-signing proxy opts in with ``preserve_thinking`` (#120723): ``True`` maps its
    ``third_party`` route to ``native`` so the converter, the signature-rejection recovery and every
    native call site follow the same contract, per the model's own capability — preserve-prior
    models keep signed thinking on every assistant turn, last-turn-only ones keep the native
    latest-turn behavior. Kimi/DeepSeek judgments outrank the opt-in (it never rewrites those
    routes). Absent/False keeps the strip; anything not strictly ``True`` is fail-closed. When
    *preserve_thinking* is None the provider entry's opt-in is consulted best-effort."""
    is_third_party = _is_third_party_anthropic_endpoint(base_url) and not _is_nous_portal_endpoint(base_url)
    if _is_kimi_family_endpoint(base_url, model):
        return "kimi"
    if _is_deepseek_anthropic_endpoint(base_url) or (is_third_party and _model_name_is_deepseek_thinking(model)):
        return "deepseek"
    if is_third_party:
        opted_in = bool(preserve_thinking) if preserve_thinking is not None else _provider_preserve_thinking_opt_in(base_url)
        if opted_in:
            return "native"
        return "third_party"
    return "native"


def native_anthropic_preserves_prior_thinking(base_url: Any, model: Any, preserve_thinking: Any = None) -> bool:
    """True for native-contract routes (direct Anthropic/Nous Portal, plus trusted re-signing
    proxies that opt in with ``preserve_thinking``) whose model retains old thinking."""
    return anthropic_thinking_route(base_url, model, preserve_thinking) == "native" and model_preserves_prior_thinking(model)

"""Whether a reasoning-only clean stop may be promoted to the visible answer.

A field name is not provenance: OpenRouter, DeepSeek, Bedrock and several adapters all put
private chain-of-thought in ``reasoning``. Promotion is therefore limited to trusted routes,
decided from the agent's LIVE route on every call so fallback activation, primary restore and
model switches never carry a decision to another route.
"""

from __future__ import annotations

from typing import Any

from utils import base_url_host_matches

_ANSWER_IN_REASONING_CAPABILITY = "answer_in_reasoning"
_NEMOTRON_PARSER_MODEL_MARKER = "nemotron-3.5-lightning"


def answer_in_reasoning_capability(agent: Any) -> bool:
    """True when the live route may return a clean-stop reasoning payload as the answer.

    Trusted: a custom_providers per-model ``answer_in_reasoning`` opt-in or provider-level
    ``capabilities: {answer_in_reasoning: ...}`` block on the live route, or the local
    Nemotron-3.5-Lightning parser route from #109205. OpenRouter and non-chat-completions
    transports are never trusted.
    """
    base_url = str(getattr(agent, "base_url", "") or "")
    provider = str(getattr(agent, "provider", "") or "").strip().lower()
    if provider == "openrouter" or base_url_host_matches(base_url, "openrouter.ai"):
        return False

    api_mode = str(getattr(agent, "api_mode", "") or "").strip().lower()
    if api_mode and api_mode != "chat_completions":
        return False

    # Returns None on unreadable config, so no guard is needed here.
    from hermes_cli.config import _entries_for_route, get_custom_provider_model_capability

    model = str(getattr(agent, "model", "") or "")
    custom_providers = getattr(agent, "_custom_providers", None)
    configured = get_custom_provider_model_capability(
        model=model,
        base_url=base_url,
        capability=_ANSWER_IN_REASONING_CAPABILITY,
        custom_providers=custom_providers,
    )
    if configured is not None:
        return configured
    # The provider-level ``capabilities:`` block, re-read on the live route like the per-model key
    # so it holds on CLI/TUI (no constructor ``capabilities=``) and survives /model switches.
    # Several entries may share one endpoint (LiteLLM/vLLM router): only the live ``custom:<slug>``
    # entry counts; without one, every flagged sibling on the endpoint must opt in. Startup, gateway
    # and TUI keep ``provider="custom"`` and the named identity in ``requested_provider``.
    from hermes_cli.providers import custom_provider_slug

    requested = str(getattr(agent, "requested_provider", "") or "").strip().lower()
    ids = {custom_provider_slug(p) for p in (provider, requested) if p.startswith("custom:")}
    entries = list(_entries_for_route(base_url, custom_providers, None))
    live = [e for e in entries
            if custom_provider_slug(e.get("name"), e.get("provider_key")) in ids]
    flags = {v for e in live or entries
             if isinstance(v := (e.get("capabilities") or {}).get(_ANSWER_IN_REASONING_CAPABILITY), bool)}
    if flags:
        return flags == {True}

    if _NEMOTRON_PARSER_MODEL_MARKER not in model.lower():
        return False
    from agent.model_metadata import is_local_endpoint

    return is_local_endpoint(base_url)

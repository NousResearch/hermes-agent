"""Fallback wire-format resolution, moved verbatim from ``chat_completion_helpers`` so that
file keeps going down toward its code-health line cap (moved code keeps its cap; see root
AGENTS.md). Callers import these names from :mod:`agent.chat_completion_helpers`, which
re-exports them, so existing import paths are unchanged.

``_fallback_api_mode_hint`` decides the api_mode a ``fallback_providers`` entry uses;
``_fallback_api_mode_resolved`` re-detects the wire when the hint pass landed on the
``chat_completions`` default.
"""

from __future__ import annotations

from typing import Optional

from utils import base_url_host_matches, base_url_hostname


def _is_anthropic_wire_url(url: str) -> bool:
    """Same Messages-only host match as determine_api_mode() / _detect_api_mode_for_url(): api.anthropic.com,
    a /anthropic suffix, or Kimi Code's api.kimi.com/coding (its /chat/completions 404s — #77256)."""
    from hermes_cli.providers import host_mandated_api_mode
    return host_mandated_api_mode(url) == "anthropic_messages"


def _fallback_api_mode_hint(fb: dict, fb_provider: str, fb_base_url_hint: Optional[str]) -> tuple[bool, str]:
    """(explicit, api_mode) for a fallback entry from its ORIGINAL base_url: resolve_provider_client()
    rewrites a dual-surface /anthropic base to /v1, losing the Anthropic wire signal. An explicit
    ``api_mode`` always wins (even "chat_completions") and suppresses later re-detection;
    ``provider: anthropic`` without a base_url still resolves to anthropic_messages."""
    from hermes_cli.runtime_provider import _get_named_custom_provider, _parse_api_mode
    # Entries accept the same ``api_mode`` / ``transport`` spellings as ``providers.<name>``.
    explicit = _parse_api_mode(fb.get("api_mode") or fb.get("transport"))
    if explicit:
        return True, explicit
    # A named ``providers.<name>`` block declares its wire once (``api_mode``/``transport``); a
    # fallback entry naming that provider inherits it instead of being re-detected from the host
    # (#33062, #81932: an Anthropic-Messages or Responses-only relay on a plain host was downgraded
    # to chat_completions while resolve_provider_client had already built the declared client).
    if fb_provider and fb_provider not in {"custom", "moa"}:
        declared = (_get_named_custom_provider(fb_provider) or {}).get("api_mode")
        if declared:
            return True, declared
        # A plugin provider declares its wire on its registered ProviderProfile
        # (commandcode-anthropic: anthropic_messages behind a plain host the URL
        # heuristic cannot classify — #143933). The profile default equals this
        # function's own fallthrough, so only an explicit non-default wire pins
        # the hint; generic profiles keep the re-detection path below.
        from providers import get_provider_profile
        profile_api_mode = getattr(get_provider_profile(fb_provider), "api_mode", None)
        if profile_api_mode and profile_api_mode != "chat_completions":
            return True, profile_api_mode
    if fb_provider == "anthropic" or (fb_base_url_hint and _is_anthropic_wire_url(fb_base_url_hint)):
        return False, "anthropic_messages"
    return False, "chat_completions"


def _fallback_api_mode_resolved(agent, fb_provider: str, fb_model: str, fb_base_url: str) -> str:
    """Re-detect api_mode from provider / resolved base URL / model when the hint pass
    landed on the chat_completions default (never called for an explicit api_mode)."""
    if fb_provider == "openai-codex":
        return "codex_responses"
    from hermes_cli.models import opencode_model_api_mode
    from hermes_cli.runtime_provider_custom import _opencode_family_for_custom
    opencode_family = _opencode_family_for_custom(fb_provider, fb_base_url)
    if opencode_family is not None:
        # OpenCode Zen/Go/free serve Responses-only (muse-spark, gpt-*, grok-*), anthropic_messages
        # (minimax, qwen) and chat_completions models behind one provider; the primary /model path
        # already re-derives per model — the fallback wire must agree (#102148).
        return opencode_model_api_mode(opencode_family, fb_model)
    if fb_provider in {"nous", "nous-portal", "nousresearch"}:
        # Portal is dual-wire: anthropic/* must land on /v1/messages (the swap rebuilds the native client).
        from hermes_cli.providers import nous_api_mode
        return nous_api_mode(fb_model)
    if _is_anthropic_wire_url(fb_base_url):
        # Named custom providers (cron-anthropic) resolve base_url from config; the hint pass never saw it.
        return "anthropic_messages"
    if agent._is_azure_openai_url(fb_base_url):
        return "chat_completions"  # Azure serves gpt-5.x on /chat/completions — no Responses API.
    # Provider exceptions (Copilot gpt-5-mini) stay inside the requires-responses predicate.
    if agent._is_direct_openai_url(fb_base_url) or agent._provider_model_requires_responses_api(fb_model, provider=fb_provider):
        return "codex_responses"
    host = base_url_hostname(fb_base_url)
    if fb_provider == "bedrock" or (host.startswith("bedrock-runtime.") and base_url_host_matches(fb_base_url, "amazonaws.com")):
        return "bedrock_converse"
    return "chat_completions"

"""Anthropic prompt-cache policy resolution for one resolved destination.

Decides *whether* a route may carry Anthropic-style ``cache_control`` markers and *which layout*
(inner content blocks for native Messages, message envelope for OpenAI-wire proxies). Byte-verbatim
shard of ``agent/agent_runtime_helpers.py`` (2K-law fracture, #79925 / #78647); that facade
re-exports every name here, so ``agent.agent_runtime_helpers.<name>`` stays green.
"""

from __future__ import annotations

import contextlib
import copy
import logging
from typing import Any, Optional, Tuple

from utils import base_url_host_matches, base_url_hostname

logger = logging.getLogger(__name__)


def _direct_native_anthropic_tool_cache_capability(
    agent, *, provider: Optional[str] = None, base_url: Optional[str] = None,
    api_mode: Optional[str] = None, model: Optional[str] = None,
) -> bool:
    """Return whether this resolved destination accepts native tool markers."""
    eff_base_url = base_url if base_url is not None else (agent.base_url or "")
    eff_api_mode = api_mode if api_mode is not None else (agent.api_mode or "")
    return eff_api_mode == "anthropic_messages" and base_url_hostname(eff_base_url) == "api.anthropic.com"


# The cache_ttl tiers accepted by config; mirrored by agent_init's live-agent snapshot.
VALID_CACHE_TTLS = ("5m", "1h")


def cache_ttl_means_disabled(ttl: Any) -> bool:
    """True when a ``prompt_caching.cache_ttl`` value means caching off (single predicate shared
    by ``agent_init`` and the stub policy paths). Unknown values (``"2h"``, ints) are NOT a disable."""
    if ttl in VALID_CACHE_TTLS:
        return False
    return ttl is False or ttl is None or str(ttl).lower() in ("off", "false", "disabled", "no", "none")


def _raw_cache_ttl_from_config(default: Any) -> Any:
    """Raw ``prompt_caching.cache_ttl`` config value, or ``default`` when config cannot be read."""
    try:
        from hermes_cli.config import load_config_readonly
        return (load_config_readonly().get("prompt_caching", {}) or {}).get("cache_ttl", "5m")
    except Exception:
        return default


def prompt_caching_disabled_from_config() -> bool:
    """True when ``prompt_caching.cache_ttl`` is configured as off (same detection as ``agent_init``).

    Same disable detection as ``agent_init`` (via ``cache_ttl_means_disabled``) so stub-based policy paths
    (MoA slot decoration, auxiliary fallback replan) honor the same config contract without holding a live
    ``AIAgent`` (#76085 / #33555).
    """
    return cache_ttl_means_disabled(_raw_cache_ttl_from_config("5m"))


def configured_cache_ttl() -> Optional[str]:
    """Configured ``prompt_caching.cache_ttl`` tier (``5m``/``1h``), else None; mirrors
    ``agent_init`` so stub paths don't regress a configured ``1h`` to 5m."""
    ttl = _raw_cache_ttl_from_config(None)
    return ttl if ttl in VALID_CACHE_TTLS else None


def blank_cache_policy_stub(cache_disabled: Optional[bool] = None):
    """Destination-identity-blank stub for ``anthropic_prompt_cache_policy``; the sole sanctioned
    constructor so ``_cache_disabled`` is never omitted (None consults the global config)."""
    from types import SimpleNamespace
    if cache_disabled is None:
        cache_disabled = prompt_caching_disabled_from_config()
    return SimpleNamespace(provider="", base_url="", api_mode="", model="", _cache_disabled=bool(cache_disabled))


def plan_cache_sections_for_destination(
    messages: list, tools: Optional[list], *, provider: str, base_url: str, api_mode: str,
    model: str, cache_disabled: Optional[bool] = None, cache_ttl: Optional[str] = None,
    static_system_prefix: Optional[str] = None,
) -> Tuple[list, list]:
    """Plan request-local cache sections for one resolved destination (MoA / auxiliary senders):
    stripped copies (non-caching route) or a ``build_prompt_cache_plan`` layout; never mutates
    inputs. ``cache_disabled``/``cache_ttl`` default to live config so the operator's disable and
    tier are honored; ``static_system_prefix`` gives the system prompt the main loop's early breakpoint.

    ``cache_disabled`` threads the operator's ``prompt_caching.cache_ttl`` disable into the blank policy
    stub. When omitted, the live config is consulted so MoA/auxiliary paths cannot re-enable markers after
    the user turned caching off (#76085).
    """
    from agent.prompt_caching import (
        build_prompt_cache_plan, effective_cache_ttl, envelope_tool_part_cache_markers_supported,
        strip_anthropic_cache_control, strip_anthropic_tool_cache_control,
    )
    # The policy function reads agent.* only as fallbacks for kwargs we don't pass; blank_cache_policy_stub
    # is the only sanctioned stub so _cache_disabled cannot be left off again (#76085).
    stub = blank_cache_policy_stub(cache_disabled)
    dest = dict(provider=provider, base_url=base_url, api_mode=api_mode, model=model)
    should_cache, native_layout = anthropic_prompt_cache_policy(stub, **dest)
    if not should_cache:
        canonical_messages = copy.deepcopy(messages or [])
        strip_anthropic_cache_control(canonical_messages)
        return canonical_messages, strip_anthropic_tool_cache_control(tools)
    plan = build_prompt_cache_plan(
        messages, tools,
        # effective_cache_ttl resolves None → "5m"; cache-disabled agents never reach here.
        cache_ttl=effective_cache_ttl(cache_ttl, provider=provider, model=model),
        native_anthropic=native_layout,
        static_system_prefix=static_system_prefix if isinstance(static_system_prefix, str) else None,
        direct_native_tool_cache=_direct_native_anthropic_tool_cache_capability(stub, **dest),
        # LiteLLM-style envelope routes forward part-level markers into tool_result.content[] →
        # non-retryable 400.
        tool_part_markers=envelope_tool_part_cache_markers_supported(provider, base_url),
    )
    return plan.messages, plan.tools


def _is_litellm_route(provider_lower: str, base_url: str) -> bool:
    """True when a route is a LiteLLM proxy: ``litellm`` as a whole delimited token (not
    substring) in the provider id or host; a path segment never qualifies."""
    return _has_litellm_token(provider_lower, ":-_/") or _has_litellm_token(base_url_hostname(base_url), ".-")


def _has_litellm_token(value: str, delimiters: str) -> bool:
    """True when ``value`` contains ``litellm`` as a whole delimited token."""
    if not value:
        return False
    return "litellm" in value.translate(str.maketrans(delimiters, " " * len(delimiters))).split()


def _moa_aggregator_cache_policy(agent, eff_model: str) -> tuple[bool, bool]:
    """MoA virtual provider: resolve the policy from the preset's real aggregator slot (the
    virtual provider matches no caching branch and would silently lose caching)."""
    try:
        from hermes_cli.config import load_config as _load_moa_cfg
        from hermes_cli.moa_config import resolve_moa_preset
        from hermes_cli.runtime_provider import resolve_runtime_provider
        agg = resolve_moa_preset(_load_moa_cfg().get("moa") or {}, eff_model or None).get("aggregator") or {}
        agg_provider = str(agg.get("provider") or "").strip()
        agg_model = str(agg.get("model") or "").strip()
        if agg_provider and agg_model:
            agg_base_url = agg_api_mode = ""
            with contextlib.suppress(Exception):
                rt = resolve_runtime_provider(requested=agg_provider, target_model=agg_model)
                agg_base_url = rt.get("base_url") or ""
                agg_api_mode = rt.get("api_mode") or ""
            return anthropic_prompt_cache_policy(
                agent, provider=agg_provider, base_url=agg_base_url, api_mode=agg_api_mode, model=agg_model
            )
    except Exception as _moa_exc:  # pragma: no cover - defensive
        logger.debug("MoA aggregator cache-policy resolution failed: %s", _moa_exc)
    return False, False


def _route_may_be_custom(agent, eff_provider: str, provider_lower: str, eff_base_url: str) -> bool:
    """Cheap identity gate deciding whether a custom-provider capability lookup is worth running."""
    custom_providers = getattr(agent, "_custom_providers", None)
    if custom_providers:
        # Same semantics as the capability helper (normalize_route_base_url +
        # custom_provider_aliases) so spelling differences don't drop declarations.
        from hermes_cli.providers import custom_provider_aliases
        from hermes_cli.route_identity import normalize_route_base_url
        provider_ids = {provider_lower, provider_lower.removeprefix("custom:")}
        eff_url_normalized = normalize_route_base_url(eff_base_url)
        return any(
            provider_ids & custom_provider_aliases(str(entry.get("name") or ""), str(entry.get("provider_key") or ""))
            or (eff_url_normalized and normalize_route_base_url(entry.get("base_url")) == eff_url_normalized)
            for entry in custom_providers if isinstance(entry, dict)
        )
    if custom_providers is not None:
        return False  # attached empty list never matches
    # None = list not attached yet (early init or blank stub). Avoid rebuilding the list for
    # ordinary built-in routes.
    try:
        from hermes_cli.providers import get_provider
        # allow_network=False: never trigger a registry fetch from the send path; a catalog miss
        # degrades to the conservative capability lookup.
        provider_def = get_provider(eff_provider, allow_network=False)
        return provider_def is None or (
            bool(provider_def.base_url)
            and base_url_hostname(provider_def.base_url) != base_url_hostname(eff_base_url)
        )
    except Exception as _pd_exc:
        logger.debug("provider lookup failed during cache-policy pre-gate: %s", _pd_exc)
        return provider_lower.startswith("custom:")


def anthropic_prompt_cache_policy(
    agent, *, provider: Optional[str] = None, base_url: Optional[str] = None,
    api_mode: Optional[str] = None, model: Optional[str] = None,
) -> tuple[bool, bool]:
    """Decide whether to apply Anthropic prompt caching; returns ``(should_cache, use_native_layout)``.
    Native layout puts markers on inner content blocks (Anthropic wire), else on the message
    envelope (OpenRouter / OpenAI-wire proxies; Qwen/Alibaba too). The operator disable is read
    from ``_cache_disabled`` (not ``_cache_ttl``, unset during init) so it survives switches
    and restores. Branch ORDER is load-bearing (see inline notes).

    Qwen / Alibaba-family models on OpenCode, OpenCode Go, and direct Alibaba (DashScope) also honour
    Anthropic-style ``cache_control`` markers on OpenAI-wire chat completions. Upstream pi-mono #3392 / pi
    #3393 documented this for opencode-go Qwen. Without markers these providers serve zero cache hits,
    re-billing the full prompt on every turn.
    """
    if getattr(agent, "_cache_disabled", False):
        return (False, False)
    eff_provider = (provider if provider is not None else agent.provider) or ""
    eff_base_url = base_url if base_url is not None else (agent.base_url or "")
    eff_api_mode = api_mode if api_mode is not None else (agent.api_mode or "")
    eff_model = (model if model is not None else agent.model) or ""
    if eff_provider.strip().lower() == "moa":
        return _moa_aggregator_cache_policy(agent, eff_model)
    if isinstance(eff_model, dict):
        eff_model = eff_model.get('model') or eff_model.get('default') or ''
    eff_model = eff_model if isinstance(eff_model, str) else str(eff_model or '')
    model_lower = eff_model.lower()
    provider_lower = eff_provider.lower()
    is_claude = "claude" in model_lower
    # Kimi/Moonshot via OpenRouter uses the same envelope cache_control as Claude; without this it
    # serves ~1% cache hits. Family matcher covers bare k1./k2. slugs.
    # Without this branch moonshotai/kimi-k2.6 falls through to (False, False), serving ~1% cache hits on
    # 64K-token prompts and re-billing the full prompt on every turn. Observed within-turn progression with
    # cache enabled: 1% → 67% → 84% → 97% (#25970). Reuses the canonical family matcher (covers bare
    # k1./k2./k25 release slugs the substring check missed).
    from agent.anthropic_endpoints import _model_name_is_kimi_family
    is_kimi = _model_name_is_kimi_family(eff_model) or "moonshot" in model_lower
    is_openrouter = base_url_host_matches(eff_base_url, "openrouter.ai")
    # Nous Portal proxies to OpenRouter; treat as OpenRouter-equivalent for cache layout.
    is_nous_portal = base_url_host_matches(eff_base_url, "nousresearch.com")
    is_anthropic_wire = eff_api_mode == "anthropic_messages"
    is_native_anthropic = is_anthropic_wire and (
        eff_provider == "anthropic" or base_url_hostname(eff_base_url) == "api.anthropic.com"
    )
    # Honor a configured route's per-model ``prompt_caching`` capability (explicit false too); only
    # for the two transports this planner handles, not Responses/Bedrock.
    supports_cache_markers = eff_api_mode in {"anthropic_messages", "chat_completions"}
    litellm_openai_wire = (
        eff_api_mode == "chat_completions" and is_claude and _is_litellm_route(provider_lower, eff_base_url)
    )
    if supports_cache_markers and (
        is_anthropic_wire
        or litellm_openai_wire
        or _route_may_be_custom(agent, eff_provider, provider_lower, eff_base_url)
    ):
        try:
            from hermes_cli.config import get_custom_provider_model_capability
            custom_prompt_caching = get_custom_provider_model_capability(
                model=eff_model, base_url=eff_base_url, capability="prompt_caching",
                custom_providers=getattr(agent, "_custom_providers", None),
            )
            if custom_prompt_caching is not None:
                # Layout follows the transport: native Messages → inner blocks; OpenAI wire → envelope.
                return custom_prompt_caching, custom_prompt_caching and is_anthropic_wire
        except Exception as _cap_exc:
            logger.debug("custom-provider prompt_caching capability lookup failed: %s", _cap_exc)
    # MiniMax-M3 uses server-side automatic prefix caching; explicit markers are dead weight.
    # Checked BEFORE the native-Anthropic return since provider="anthropic" may point at a MiniMax
    # proxy.
    is_minimax_route = (
        provider_lower in {"minimax", "minimax-cn"}
        or base_url_host_matches(eff_base_url, "api.minimax.io")
        or base_url_host_matches(eff_base_url, "api.minimaxi.com")
    )
    if is_anthropic_wire and is_minimax_route:
        from agent.model_metadata import _model_name_suggests_minimax_m3
        if _model_name_suggests_minimax_m3(eff_model):
            return False, False
    if is_native_anthropic:
        return True, True
    # Envelope layout is OpenAI-wire only; Portal Claude on native Messages must fall through to the
    # anthropic_messages branch (inner-block markers) or it serves 0% cache hits.
    if (is_openrouter or is_nous_portal) and (is_claude or is_kimi) and not is_anthropic_wire:
        return True, False
    # Nous Portal Qwen takes the envelope path too; the alibaba-family check below only matches
    # provider=opencode/alibaba and would leave Portal traffic uncached.
    if is_nous_portal and "qwen" in model_lower:
        return True, False
    if is_anthropic_wire and is_claude:
        return True, True  # third-party Anthropic-compatible gateway
    # LiteLLM fronting Claude on the OpenAI wire supports cache_control but matched no grant above.
    # Claude-only: strict relays reject the block format for other models. Envelope layout: native
    # top-level markers are only relocated by the anthropic_messages adapter and 400 via LiteLLM.
    # Gated on chat_completions; codex_responses/bedrock_converse have their own handling.
    if litellm_openai_wire:
        return True, False
    # MiniMax's own models (M2.x) on its Anthropic-compatible endpoint support cache_control; opt
    # them in past the is_claude gate. M3 is excluded above.
    if is_anthropic_wire and is_minimax_route:
        return True, True
    # Qwen/Alibaba on OpenCode and DashScope accept envelope cache_control on the OpenAI wire
    # (pi-mono's "alibaba" cacheControlFormat). DeepSeek on OpenCode is excluded: its relay 400s on
    # block-array content. Family set/predicate shared with the effective_cache_ttl clamp.
    # Qwen/Alibaba on OpenCode (Zen/Go) and native DashScope: OpenAI-wire transport that accepts
    # Anthropic-style cache_control markers and rewards them with real cache hits. Without this branch
    # qwen3.6-plus on opencode-go reports 0% cached tokens and burns through the subscription on every turn.
    # OpenCode Zen's relay rejects the Anthropic-style content block format that cache markers produce
    # (content becomes a block array instead of a plain string), causing HTTP 400 (#77217).
    from agent.prompt_caching import ALIBABA_FAMILY_PROVIDERS, is_qwen_model
    if provider_lower in ALIBABA_FAMILY_PROVIDERS and is_qwen_model(model_lower):
        return True, False
    return False, False

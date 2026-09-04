"""Single source of truth for provider identity in Hermes Agent."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from utils import base_url_host_matches, base_url_hostname

logger = logging.getLogger(__name__)


# -- Hermes overlay: metadata models.dev doesn't provide ----------------------

@dataclass(frozen=True)
class HermesOverlay:
    """Hermes-specific provider metadata layered on top of models.dev."""

    transport: str = "openai_chat"        # openai_chat | anthropic_messages | codex_responses
    is_aggregator: bool = False
    auth_type: str = "api_key"            # api_key | oauth_device_code | oauth_external | external_process
    extra_env_vars: Tuple[str, ...] = ()  # env vars models.dev doesn't list
    base_url_override: str = ""           # override if models.dev URL is wrong/missing
    base_url_env_var: str = ""            # env var for user-custom base URL
    keyless: bool = False                 # served anonymously — no credential exists to configure


HERMES_OVERLAYS: Dict[str, HermesOverlay] = {
    "moa": HermesOverlay(
        transport="openai_chat",
        auth_type="virtual",
        base_url_override="moa://local",
    ),
    "openrouter": HermesOverlay(
        transport="openai_chat",
        is_aggregator=True,
        base_url_env_var="OPENROUTER_BASE_URL",
    ),
    "nous": HermesOverlay(
        transport="openai_chat",
        auth_type="oauth_device_code",
        base_url_override="https://inference-api.nousresearch.com/v1",
    ),
    "openai-codex": HermesOverlay(
        transport="codex_responses",
        auth_type="oauth_external",
        base_url_override="https://chatgpt.com/backend-api/codex",
    ),
    "openai-api": HermesOverlay(
        transport="codex_responses",
        base_url_override="https://api.openai.com/v1",
        base_url_env_var="OPENAI_BASE_URL",
    ),
    "xai-oauth": HermesOverlay(
        transport="codex_responses",
        auth_type="oauth_external",
        base_url_override="https://api.x.ai/v1",
        base_url_env_var="XAI_BASE_URL",
    ),
    "qwen-oauth": HermesOverlay(
        transport="openai_chat",
        auth_type="oauth_external",
        base_url_override="https://portal.qwen.ai/v1",
        base_url_env_var="HERMES_QWEN_BASE_URL",
    ),
    "lmstudio": HermesOverlay(
        transport="openai_chat",
        auth_type="api_key",
        extra_env_vars=("LM_API_KEY",),
        base_url_override="http://127.0.0.1:1234/v1",
        base_url_env_var="LM_BASE_URL",
    ),
    "copilot-acp": HermesOverlay(
        transport="codex_responses",
        auth_type="external_process",
        base_url_override="acp://copilot",
        base_url_env_var="COPILOT_ACP_BASE_URL",
    ),
    "github-copilot": HermesOverlay(
        transport="openai_chat",
        extra_env_vars=("COPILOT_GITHUB_TOKEN", "GH_TOKEN"),
    ),
    "anthropic": HermesOverlay(
        transport="anthropic_messages",
        extra_env_vars=("ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"),
    ),
    "zai": HermesOverlay(
        transport="openai_chat",
        extra_env_vars=("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"),
        base_url_env_var="GLM_BASE_URL",
    ),
    "kimi-for-coding": HermesOverlay(
        transport="openai_chat",
        base_url_env_var="KIMI_BASE_URL",
    ),
    "stepfun": HermesOverlay(
        transport="openai_chat",
        extra_env_vars=("STEPFUN_API_KEY",),
        base_url_override="https://api.stepfun.ai/step_plan/v1",
        base_url_env_var="STEPFUN_BASE_URL",
    ),
    "minimax": HermesOverlay(
        transport="anthropic_messages",
        base_url_env_var="MINIMAX_BASE_URL",
    ),
    "minimax-oauth": HermesOverlay(
        transport="anthropic_messages",
        auth_type="oauth_external",
        base_url_override="https://api.minimax.io/anthropic",
    ),
    "minimax-cn": HermesOverlay(
        transport="anthropic_messages",
        base_url_env_var="MINIMAX_CN_BASE_URL",
    ),
    "deepseek": HermesOverlay(
        transport="openai_chat",
        base_url_env_var="DEEPSEEK_BASE_URL",
    ),
    "alibaba": HermesOverlay(
        transport="openai_chat",
        base_url_env_var="DASHSCOPE_BASE_URL",
    ),
    "alibaba-coding-plan": HermesOverlay(
        transport="openai_chat",
        base_url_env_var="ALIBABA_CODING_PLAN_BASE_URL",
    ),
    "vercel": HermesOverlay(
        transport="openai_chat",
        is_aggregator=True,
    ),
    "opencode": HermesOverlay(
        transport="openai_chat",
        is_aggregator=True,
        base_url_env_var="OPENCODE_ZEN_BASE_URL",
    ),
    "opencode-go": HermesOverlay(
        transport="openai_chat",
        is_aggregator=True,
        base_url_env_var="OPENCODE_GO_BASE_URL",
    ),
    "opencode-free": HermesOverlay(
        transport="openai_chat",
        is_aggregator=True,
        base_url_override="https://opencode.ai/zen/v1",
        keyless=True,
    ),
    "kilo": HermesOverlay(
        transport="openai_chat",
        is_aggregator=True,
        base_url_env_var="KILOCODE_BASE_URL",
    ),
    "huggingface": HermesOverlay(
        transport="openai_chat",
        is_aggregator=True,
        base_url_env_var="HF_BASE_URL",
    ),
    "novita": HermesOverlay(
        transport="openai_chat",
        is_aggregator=True,
        base_url_env_var="NOVITA_BASE_URL",
    ),
    "xai": HermesOverlay(
        transport="codex_responses",
        base_url_override="https://api.x.ai/v1",
        base_url_env_var="XAI_BASE_URL",
    ),
    "nvidia": HermesOverlay(
        transport="openai_chat",
        base_url_override="https://integrate.api.nvidia.com/v1",
        base_url_env_var="NVIDIA_BASE_URL",
    ),
    "xiaomi": HermesOverlay(
        transport="openai_chat",
        base_url_env_var="XIAOMI_BASE_URL",
    ),
    "tencent-tokenhub": HermesOverlay(
        transport="openai_chat",
        base_url_env_var="TOKENHUB_BASE_URL",
    ),
    "tencent-tokenplan": HermesOverlay(
        transport="anthropic_messages",
        base_url_override="https://api.lkeap.cloud.tencent.com/plan/anthropic",
        base_url_env_var="TOKENPLAN_BASE_URL",
    ),
    "arcee": HermesOverlay(
        transport="openai_chat",
        base_url_override="https://api.arcee.ai/api/v1",
        base_url_env_var="ARCEE_BASE_URL",
    ),
    "gmi": HermesOverlay(
        transport="openai_chat",
        extra_env_vars=("GMI_API_KEY",),
        base_url_override="https://api.gmi-serving.com/v1",
        base_url_env_var="GMI_BASE_URL",
    ),
    "fireworks": HermesOverlay(
        transport="openai_chat",
        extra_env_vars=("FIREWORKS_API_KEY",),
        base_url_override="https://api.fireworks.ai/inference/v1",
    ),
    "actual": HermesOverlay(
        transport="codex_responses",
        extra_env_vars=("ACTUAL_API_KEY", "ACTUAL_BASE_URL"),
        base_url_override="https://api.actual.inc/v1",
        base_url_env_var="ACTUAL_BASE_URL",
    ),
    "upstage": HermesOverlay(
        transport="openai_chat",
        extra_env_vars=("UPSTAGE_API_KEY",),
        base_url_override="https://api.upstage.ai/v1",
        base_url_env_var="UPSTAGE_BASE_URL",
    ),
    "nebius-token-factory": HermesOverlay(
        transport="openai_chat",
        extra_env_vars=("NEBIUS_API_KEY", "NEBIUS_TOKEN_FACTORY_API_KEY"),
        base_url_override="https://api.tokenfactory.nebius.com/v1",
        base_url_env_var="NEBIUS_BASE_URL",
    ),
    "ollama-cloud": HermesOverlay(
        transport="openai_chat",
        base_url_override="https://ollama.com/v1",
        base_url_env_var="OLLAMA_BASE_URL",
    ),
    # Azure Foundry: supports both OpenAI-style and Anthropic-style endpoints.
    # The transport is determined at runtime from config.yaml model.api_mode.
    "azure-foundry": HermesOverlay(
        transport="openai_chat",  # default; overridden by api_mode in config
        base_url_env_var="AZURE_FOUNDRY_BASE_URL",
    ),
    "bedrock": HermesOverlay(
        transport="bedrock_converse",
        auth_type="aws_sdk",
    ),
    # Vertex authenticates via OAuth2 (service-account JSON / ADC), not a
    # static API key or models.dev entry — resolved specially by
    # agent/vertex_adapter.py, like bedrock's aws_sdk. Without an overlay
    # entry get_provider("vertex") returns None, which makes
    # _preserve_provider_with_base_url() in agent/auxiliary_client.py treat
    # a Vertex MoA slot's resolved (base_url, api_key) pair as an unknown
    # custom endpoint instead of "vertex" — losing the provider identity
    # that _refresh_provider_credentials() needs to re-mint an expired
    # OAuth2 token on a 401.
    "vertex": HermesOverlay(
        transport="openai_chat",
        auth_type="vertex",
    ),
}


# -- Resolved provider -------------------------------------------------------

@dataclass
class ProviderDef:
    """Complete provider definition — merged from models.dev + overlay + user config."""

    id: str
    name: str
    transport: str                        # openai_chat | anthropic_messages | codex_responses
    api_key_env_vars: Tuple[str, ...]     # all env vars to check for API key
    base_url: str = ""
    base_url_env_var: str = ""
    is_aggregator: bool = False
    auth_type: str = "api_key"
    doc: str = ""
    source: str = ""                      # "models.dev", "hermes", "user-config"


# -- Aliases ------------------------------------------------------------------
# Maps human-friendly / legacy names to canonical provider IDs.
# Uses models.dev IDs where possible.

ALIASES: Dict[str, str] = {
    # openrouter
    "openai": "openrouter",     # bare "openai" → route through aggregator

    # zai
    "glm": "zai",
    "z-ai": "zai",
    "z.ai": "zai",
    "zhipu": "zai",

    # xai
    "x-ai": "xai",
    "x.ai": "xai",
    "grok": "xai",
    "grok-oauth": "xai-oauth",
    "xai-oauth": "xai-oauth",
    "x-ai-oauth": "xai-oauth",
    "xai-grok-oauth": "xai-oauth",

    # nvidia
    "nim": "nvidia",
    "nvidia-nim": "nvidia",
    "build-nvidia": "nvidia",
    "nemotron": "nvidia",

    # kimi-for-coding (models.dev ID)
    "kimi": "kimi-for-coding",
    "kimi-coding": "kimi-for-coding",
    "kimi-coding-cn": "kimi-for-coding",
    "moonshot": "kimi-for-coding",

    # stepfun
    "step": "stepfun",
    "stepfun-coding-plan": "stepfun",

    # minimax-cn
    "minimax-china": "minimax-cn",
    "minimax_cn": "minimax-cn",

    # anthropic
    "claude": "anthropic",
    "claude-code": "anthropic",

    # github-copilot (models.dev ID)
    "copilot": "github-copilot",
    "github": "github-copilot",
    "github-copilot-acp": "copilot-acp",

    # vercel (models.dev ID for AI Gateway)
    "ai-gateway": "vercel",
    "aigateway": "vercel",
    "vercel-ai-gateway": "vercel",

    # opencode (models.dev ID for OpenCode Zen)
    "opencode-zen": "opencode",
    "zen": "opencode",

    # opencode-go
    "go": "opencode-go",
    "opencode-go-sub": "opencode-go",

    # opencode-free
    "free": "opencode-free",
    "opencode_free": "opencode-free",

    # kilo (models.dev ID for KiloCode)
    "kilocode": "kilo",
    "kilo-code": "kilo",
    "kilo-gateway": "kilo",

    # deepseek
    "deep-seek": "deepseek",

    # alibaba
    "dashscope": "alibaba",
    "aliyun": "alibaba",
    "qwen": "alibaba",
    "alibaba-cloud": "alibaba",
    "alibaba_coding": "alibaba-coding-plan",
    "alibaba-coding": "alibaba-coding-plan",
    "alibaba_coding_plan": "alibaba-coding-plan",

    # huggingface
    "hf": "huggingface",
    "hugging-face": "huggingface",
    "huggingface-hub": "huggingface",

    # novita
    "novita-ai": "novita",
    "novitaai": "novita",

    # xiaomi
    "mimo": "xiaomi",
    "xiaomi-mimo": "xiaomi",

    # tencent
    "tencent": "tencent-tokenhub",
    "tokenhub": "tencent-tokenhub",
    "tencent-cloud": "tencent-tokenhub",
    "tencentmaas": "tencent-tokenhub",
    "tokenplan": "tencent-tokenplan",
    "tencent-lkeap": "tencent-tokenplan",

    # bedrock
    "aws": "bedrock",
    "aws-bedrock": "bedrock",
    "amazon-bedrock": "bedrock",
    "amazon": "bedrock",

    # arcee
    "arcee-ai": "arcee",
    "arceeai": "arcee",

    # gmi
    "gmi-cloud": "gmi",
    "gmicloud": "gmi",

    # fireworks
    "fireworks-ai": "fireworks",
    "fw": "fireworks",

    # upstage
    "solar": "upstage",

    # Actual Computer
    "actual-computer": "actual",
    "actualcomputer": "actual",
    "aci": "actual",
    # Nebius Token Factory
    "nebius": "nebius-token-factory",
    "nebius-tokenfactory": "nebius-token-factory",
    "nebius-tf": "nebius-token-factory",
    "token-factory": "nebius-token-factory",
    "tokenfactory": "nebius-token-factory",

    # Local server aliases → virtual "local" concept (resolved via user config)
    "lmstudio": "lmstudio",
    "lm-studio": "lmstudio",
    "lm_studio": "lmstudio",
    "ollama": "custom",  # bare "ollama" = local; use "ollama-cloud" for cloud
    "vllm": "local",
    "llamacpp": "local",
    "llama.cpp": "local",
    "llama-cpp": "local",
}
ALIASES: Dict[str, str] = {alias: canon for canon, aliases in _ALIAS_GROUPS.items() for alias in aliases}


# -- Display labels for providers not in the models.dev catalog ---------------

_LABEL_OVERRIDES: Dict[str, str] = {
    "moa": "Mixture of Agents",
    "nous": "Nous Portal",
    "openai-codex": "ChatGPT or Codex Subscription",
    "copilot-acp": "GitHub Copilot ACP",
    "stepfun": "StepFun Step Plan",
    "xiaomi": "Xiaomi MiMo",
    "gmi": "GMI Cloud",
    "upstage": "Upstage Solar",
    "actual": "Actual Computer",
    "tencent-tokenhub": "Tencent TokenHub",
    "nebius-token-factory": "Nebius Token Factory",
    "tencent-tokenplan": "Tencent TokenPlan",
    "lmstudio": "LM Studio",
    "local": "Local endpoint",
    "bedrock": "AWS Bedrock",
    "vertex": "Google Vertex AI",
    "ollama-cloud": "Ollama Cloud",
    "xai-oauth": "xAI Grok OAuth (SuperGrok / Premium+)",
    "opencode-free": "OpenCode Free",
}


# -- Transport → API mode mapping ---------------------------------------------

TRANSPORT_TO_API_MODE: Dict[str, str] = {
    "openai_chat": "chat_completions", "anthropic_messages": "anthropic_messages",
    "codex_responses": "codex_responses", "bedrock_converse": "bedrock_converse",
}


# -- Helper functions ---------------------------------------------------------

def normalize_provider(name: str) -> str:
    """Resolve aliases and normalise casing to a canonical provider id."""
    key = name.strip().lower()
    return ALIASES.get(key, key)


def is_actual_route(provider: str = "", base_url: str = "") -> bool:
    """Identify Actual by provider/alias or its hosted endpoint, including custom routes."""
    return (
        normalize_provider(provider or "") == "actual"
        or base_url_hostname(base_url) == "api.actual.inc"
    )


def _models_dev_info(canonical: str, allow_network: bool = True):
    """models.dev entry or None. Single-arg call on the default path: test sites monkeypatch
    ``get_provider_info`` with single-arg lambdas."""
    try:
        from agent.models_dev import get_provider_info as _mdev_provider
        return _mdev_provider(canonical) if allow_network else _mdev_provider(canonical, allow_network=False)
    except Exception:
        return None


def _overlay_pdef(canonical, ov: HermesOverlay, name, env_vars, base_url, doc, source) -> ProviderDef:
    return ProviderDef(id=canonical, name=name, transport=ov.transport, api_key_env_vars=env_vars, base_url=base_url,
                       base_url_env_var=ov.base_url_env_var, is_aggregator=ov.is_aggregator, auth_type=ov.auth_type, doc=doc,
                       source=source)


def get_provider(name: str, *, allow_network: bool = True) -> Optional[ProviderDef]:
    """Look up a built-in provider by id or alias: models.dev catalog merged with the Hermes overlay;
    Hermes-only overlay (nous, openai-codex, …); plugin provider profiles with a concrete endpoint."""
    canonical = normalize_provider(name)
    mdev_info = _models_dev_info(canonical, allow_network)
    overlay = HERMES_OVERLAYS.get(canonical)
    if mdev_info is not None:
        ov = overlay or HermesOverlay()
        env_vars = list(mdev_info.env)
        for ev in ov.extra_env_vars:
            if ev not in env_vars:
                env_vars.append(ev)
        return _overlay_pdef(canonical, ov, mdev_info.name, tuple(env_vars), ov.base_url_override or mdev_info.api,
                             mdev_info.doc, "models.dev")
    if overlay is not None:
        # Hermes-only provider (not in models.dev)
        return ProviderDef(
            id=canonical,
            name=_LABEL_OVERRIDES.get(canonical, canonical),
            transport=overlay.transport,
            api_key_env_vars=overlay.extra_env_vars,
            base_url=overlay.base_url_override,
            base_url_env_var=overlay.base_url_env_var,
            is_aggregator=overlay.is_aggregator,
            auth_type=overlay.auth_type,
            source="hermes",
        )

    # Plugin-registered provider profiles (plugins/model-providers/<name>/).
    # Providers that ship only as plugin profiles (e.g. commandcode,
    # tencent-tokenhub) are absent from models.dev and HERMES_OVERLAYS, so
    # without this fallback they resolve as "Unknown provider" in /model,
    # --provider, and the model-switch path even though the picker lists them
    # (CANONICAL_PROVIDERS auto-extends from the same plugin registry).
    try:
        from providers import get_provider_profile as _profile

        _prof = _profile(canonical)
        # Only profiles with a concrete endpoint resolve here. Placeholder
        # profiles like ``custom`` (aliases: ollama/local/vllm) ship with an
        # empty base_url and are completed by config.yaml custom_providers —
        # resolving them here would preempt resolve_provider_full's
        # custom-provider step and collapse keyed IDs
        # (``custom:local-...``) back to a bare, endpoint-less ``custom``.
        if _prof is not None and (_prof.base_url or "").strip():
            _api_mode_to_transport = {v: k for k, v in TRANSPORT_TO_API_MODE.items()}
            _transport = _api_mode_to_transport.get(_prof.api_mode, "openai_chat")
            return ProviderDef(
                id=canonical,
                name=_prof.display_name or _prof.name or canonical,
                transport=_transport,
                api_key_env_vars=tuple(_prof.env_vars or ()),
                base_url=_prof.base_url or "",
                auth_type=_prof.auth_type or "api_key",
                source="plugin-profile",
            )
    except Exception:
        pass

    return None


def get_label(provider_id: str) -> str:
    """Human-readable display name: label override, else models.dev name, else the id."""
    canonical = normalize_provider(provider_id)
    if canonical in _LABEL_OVERRIDES:
        return _LABEL_OVERRIDES[canonical]
    pdef = get_provider(canonical)
    return pdef.name if pdef else canonical


def is_aggregator(provider: str) -> bool:
    """Return True when the provider is a multi-model aggregator."""
    provider_norm = normalize_provider(provider or "")
    if provider_norm.startswith("custom:"):
        return True
    pdef = get_provider(provider_norm)
    return pdef.is_aggregator if pdef else False


# Flat-namespace resellers (opencode-go, opencode-zen) are flagged ``is_aggregator=True`` because
# their live ``/v1/models`` returns bare model IDs ("deepseek-v4-flash") rather than
# ``vendor/model`` routing slugs — model_switch searches their flat catalog on that flag. But they
# are NOT routing aggregators: every listed model is first-party under their own subscription, so
# picker dedup (build_models_payload) must not strip a reseller's "minimax-m3" just because a
# user's custom proxy serves a same-named model. Normalized ids: "opencode-zen" -> "opencode".
_FLAT_NAMESPACE_RESELLERS: frozenset[str] = frozenset({"opencode-go", "opencode"})


def is_routing_aggregator(provider: str) -> bool:
    """True only for TRUE routing aggregators (OpenRouter, named ``custom:*`` proxies) — excludes
    flat-namespace resellers whose catalog is first-party. Use for "would selecting this model
    silently re-route away from the intended provider?" (picker dedup)."""
    provider_norm = normalize_provider(provider or "")
    if provider_norm in _FLAT_NAMESPACE_RESELLERS:
        return False
    return is_aggregator(provider_norm)


def is_official_openai_host(base_url: str) -> bool:
    """True when *base_url* points at OpenAI's official API host family. Hostname-parsed matching
    only — never substring — so lookalike hosts (``api.openai.com.attacker.test``) and path-segment
    spoofs (``proxy.test/api.openai.com/v1``) are rejected; a genuine ``*.api.openai.com``
    subdomain requires control of openai.com DNS.

    A genuine ``*.api.openai.com`` subdomain requires control of openai.com DNS, so the dot-suffix match
    does not reopen the #32243 spoofing hole. Delegates to ``utils.base_url_host_matches``, which owns the
    exact-or-dot-suffix hostname contract (userinfo/port stripped, lowercased, trailing dot removed) — one
    implementation, not two.
    """
    return base_url_host_matches(base_url, "api.openai.com")


# Exact hostnames that are Responses-API-native: api.meta.ai only achieves prompt-cache hits on
# Responses with prompt_cache_retention (chat/completions stays cache-cold); api.router.com (Ramp
# Router) keeps reasoning validation/summaries and prompt caching on /v1/responses and serves
# /v1/chat/completions as a minimal shim.
_RESPONSES_NATIVE_HOSTS: frozenset[str] = frozenset({"api.meta.ai", "api.router.com"})


def host_mandated_api_mode(base_url: str = "") -> Optional[str]:
    """Return the wire protocol a specific endpoint *requires*, or None.

    Some hosts only accept one API mode and reject the others outright:
      - api.openai.com only accepts the Responses API for its (reasoning)
        models when tools + reasoning are in play (chat/completions 400s).
      - api.meta.ai only achieves KV-cache hits on /v1/responses with
        prompt_cache_retention; /v1/chat/completions returns 0 cached
        tokens (measured 0% vs 93-99% on /responses with retention).
      - api.router.com (Ramp Router) is Responses-native: per-model
        reasoning-effort validation, reasoning summaries, and prompt
        caching live on /v1/responses; /v1/chat/completions is only a
        minimal compatibility shim translated onto it.
      - api.anthropic.com / ``…/anthropic`` suffixes speak native Messages.
      - Kimi's ``/coding`` endpoint speaks native Messages.
      - AWS Bedrock runtime hosts speak Converse.

    These are *mandatory* — a session carrying a stale api_mode (e.g. a
    /model switch that kept the previous provider's ``chat_completions``)
    must be overridden to the host's required mode, not merely filled in
    when empty. Generic / unknown endpoints return None so an explicitly
    configured api_mode on them is never clobbered.
    """
    if not base_url:
        return None
    url_lower = base_url.rstrip("/").lower()
    hostname = base_url_hostname(base_url)
    if hostname == "api.actual.inc":
        return "chat_completions"
    # Exact-hostname matching only — never bare substring — so lookalike hosts
    # (api.openai.com.attacker.test) and path-segment spoofs (proxy.test/api.openai.com/v1) are NOT treated
    # as the real endpoint. (#32243)
    if hostname == "api.kimi.com" and "/coding" in url_lower:
        return "anthropic_messages"
    if hostname == "api.anthropic.com" or url_lower.endswith("/anthropic"):
        return "anthropic_messages"
    # Official OpenAI host family (canonical + us./eu. data-residency hosts) mandates Responses;
    # the shared predicate keeps this in lockstep with catalog filtering and listing authority.
    if is_official_openai_host(base_url) or hostname in _RESPONSES_NATIVE_HOSTS:
        # Ramp Router (api.router.com) is Responses-native: reasoning-effort validation, reasoning
        # summaries, and prompt caching live on /v1/responses, and /v1/chat/completions is only a minimal
        # compatibility shim (docs.router.com/api/endpoint). Exact-hostname match per #32243.
        return "codex_responses"
    # Ramp Router (api.router.com) is Responses-native: reasoning-effort
    # validation, reasoning summaries, and prompt caching live on
    # /v1/responses, and /v1/chat/completions is only a minimal
    # compatibility shim (docs.router.com/api/endpoint). Exact-hostname
    # match per #32243.
    if hostname == "api.router.com":
        return "codex_responses"
    if hostname.startswith("bedrock-runtime.") and base_url_host_matches(base_url, "amazonaws.com"):
        return "bedrock_converse"
    return None


def nous_api_mode(model: str = "") -> str:
    """Wire protocol for a Nous Portal model. Portal serves its ``anthropic/*`` catalog on a native
    Messages route alongside OpenAI-compatible chat/completions for everything else.

    ``anthropic/*`` rides chat/completions by default for now (``nous.anthropic_wire``). Measured
    2026-09-06, 20 concurrent sessions x 6 tool calls on Fable 5.1, same account and hour: the
    native route re-wrote the previous turn on 14-20% of consecutive calls (4 runs; the cache read
    stopped at the prior breakpoint with byte-identical prefixes), chat/completions 0 of 320 pairs.
    That is 15-20% of a fan-out's cache-write bill. The cause is inside the portal's native route
    (NousResearch/api#227 carries the diagnostics); flip the default back to ``native`` when it is
    fixed. Cost of ``chat``: prior-turn thinking travels as OpenAI-style reasoning fields instead of
    signed native blocks, and cache_control scopes are translated by the portal's adapter.
    Empty/unknown model defaults to ``chat_completions`` (the historical Nous transport)."""
    if str(model or "").strip().lower().startswith("anthropic/"):
        # ``auto`` starts on chat too: it is safe on every upstream, and ``agent/nous_wire.py``
        # promotes the session to native from the first response when the upstream allows it.
        return "anthropic_messages" if _nous_anthropic_wire() == "native" else "chat_completions"
    return "chat_completions"


def _nous_anthropic_wire() -> str:
    """``nous.anthropic_wire``: ``"chat"`` (default), ``"native"``, or ``"auto"`` (chat, then per-session
    promotion decided from the first response; see ``agent/nous_wire.py``). Anything else reads as ``chat``."""
    try:
        from hermes_cli.config import load_config_readonly
        value = str(((load_config_readonly().get("nous") or {}).get("anthropic_wire")) or "chat").strip().lower()
    except Exception:
        return "chat"
    return value if value in ("native", "auto") else "chat"


def determine_api_mode(provider: str, base_url: str = "", model: str = "") -> str:
    """API mode (wire protocol) for a provider/endpoint: host-mandated mode, then Nous dual-wire
    (model-derived — the overlay alone says openai_chat and would pin Claude on the wrong wire),
    then the known provider's transport, then bedrock, else ``chat_completions``."""
    if is_actual_route(provider, base_url):
        return "chat_completions"
    mandated = host_mandated_api_mode(base_url)
    if mandated is not None:
        return mandated
    if (provider or "").strip().lower() in {"nous", "nous-portal", "nousresearch"}:
        return nous_api_mode(model)
    pdef = get_provider(provider)
    if pdef is not None:
        return TRANSPORT_TO_API_MODE.get(pdef.transport, "chat_completions")
    if provider == "bedrock":
        return "bedrock_converse"
    return "chat_completions"


# -- Provider from user config ------------------------------------------------

def _user_pdef(pid: str, name: str, base_url: str, key_env: str, transport: str = "openai_chat") -> ProviderDef:
    """``source="user-config"`` ProviderDef shared by ``providers:`` and ``custom_providers:`` entries."""
    return ProviderDef(id=pid, name=name, transport=transport, api_key_env_vars=(key_env,) if key_env else (),
                       base_url=base_url, is_aggregator=False, auth_type="api_key", source="user-config")


def resolve_user_provider(name: str, user_config: Dict[str, Any]) -> Optional[ProviderDef]:
    """Resolve a provider from the user's config.yaml ``providers:`` section."""
    entry = user_config.get(name) if isinstance(user_config, dict) and user_config else None
    if not isinstance(entry, dict):
        return None

    # Extract fields
    display_name = entry.get("name", "") or name
    api_url = entry.get("api", "") or entry.get("url", "") or entry.get("base_url", "") or ""
    key_env = entry.get("key_env") or entry.get("api_key_env") or ""
    transport = entry.get("transport", "openai_chat") or "openai_chat"

    env_vars: List[str] = []
    if key_env:
        env_vars.append(key_env)

    return ProviderDef(
        id=name,
        name=display_name,
        transport=transport,
        api_key_env_vars=tuple(env_vars),
        base_url=api_url,
        is_aggregator=False,
        auth_type="api_key",
        source="user-config",
    )


def custom_provider_slug(display_name: str, provider_key: str = "") -> str:
    """Stable ``custom:`` identity for a configured provider: keyed ``providers:`` entries use their
    config key (survives display-name changes); legacy ``custom_providers:`` entries have no key,
    so their normalized display name is the identity."""
    identity = str(provider_key or "").strip() or str(display_name or "").strip()
    normalized = identity.lower().replace(" ", "-")
    return normalized if normalized.startswith("custom:") else f"custom:{normalized}"


def custom_provider_aliases(display_name: str, provider_key: str = "") -> frozenset[str]:
    """Return every current and legacy identity accepted for one endpoint."""
    aliases: set[str] = set()
    for value in (display_name, provider_key):
        raw = str(value or "").strip().lower()
        if not raw:
            continue
        normalized = raw.replace(" ", "-")
        aliases.update({raw, normalized, custom_provider_slug(normalized)})
        if normalized.startswith("custom:"):
            suffix = normalized.split(":", 1)[1]
            if suffix:
                aliases.update({suffix, f"custom:{normalized}"})
    return frozenset(aliases)


def resolve_custom_provider(name: str, custom_providers: Optional[List[Dict[str, Any]]]) -> Optional[ProviderDef]:
    """Resolve a provider from the user's config.yaml ``custom_providers`` list. A stored bare
    ``"custom"`` (corrupt state from a prior model-switch bug) falls back to the first valid entry
    so existing configs self-heal."""
    requested = (name or "").strip().lower()
    if not requested or not custom_providers or not isinstance(custom_providers, list):
        return None
    first_valid: Optional[ProviderDef] = None
    # If the stored provider is the bare string "custom" (corrupt state from a prior model-switch bug), fall
    # back to the first custom provider entry so existing configs self-heal. (GH #17478)
    for entry in custom_providers:
        if not isinstance(entry, dict):
            continue
        display_name = (entry.get("name") or "").strip()
        api_url = (entry.get("base_url", "") or entry.get("url", "") or entry.get("api", "") or "").strip()
        if not display_name or not api_url:
            continue
        provider_key = (entry.get("provider_key") or "").strip()
        pdef = _user_pdef(custom_provider_slug(display_name, provider_key), display_name, api_url,
                          (entry.get("key_env") or "").strip())
        if first_valid is None:
            first_valid = pdef
        if requested in custom_provider_aliases(display_name, provider_key):
            return pdef
    if requested == "custom" and first_valid:
        return first_valid
    return None


def _lossy_alias_registry_pdef(raw: str, canonical: str) -> Optional[ProviderDef]:
    """Exact Hermes registry ids win over LOSSY alias collapsing (kimi-coding-cn must stay distinct
    from kimi-coding instead of collapsing through the shared models.dev alias "kimi-for-coding").
    A collapse is lossy only when MULTIPLE registry providers normalize to the same canonical name;
    single-entry rewrites ("copilot" -> "github-copilot") are correct routing and keep resolving
    through the built-in chain so overlay transports apply."""
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY as _AUTH_PROVIDER_REGISTRY
        _pcfg = _AUTH_PROVIDER_REGISTRY.get(raw)
        if _pcfg is None:
            return None
        if sum(1 for _rid in _AUTH_PROVIDER_REGISTRY if normalize_provider(_rid) == canonical) > 1:
            return ProviderDef(id=_pcfg.id, name=_pcfg.name, transport="openai_chat",
                               api_key_env_vars=tuple(_pcfg.api_key_env_vars or ()), base_url=_pcfg.inference_base_url or "",
                               source="hermes-auth-registry")
    except Exception:
        pass
    return None


def _llamacpp_pdef() -> Optional[ProviderDef]:
    """The llamacpp aliases are a real provider whenever the managed server (or a detected external
    one) resolves — reachability is the credential. Without this rung model-switch rejected the very
    provider the Local Models 'Use' flow writes to config."""
    try:
        from hermes_cli.local_runtime.endpoint import resolve_llamacpp_endpoint
        endpoint = resolve_llamacpp_endpoint(wait_for_boot_s=0)
    except Exception:
        endpoint = None
    if not endpoint:
        return None
    return ProviderDef(id="llamacpp", name="Local", transport="openai_chat", api_key_env_vars=(), base_url=endpoint["base_url"],
                       source="local-runtime")


def resolve_provider_full(name: str, user_providers: Optional[Dict[str, Any]] = None,
                          custom_providers: Optional[List[Dict[str, Any]]] = None) -> Optional[ProviderDef]:
    """Full resolution chain: user ``providers.<raw name>`` -> lossy-alias registry id -> built-in
    (models.dev + overlays) -> user providers (canonical, then raw) -> ``custom_providers`` ->
    managed llamacpp -> models.dev directly. User-defined ``providers.<name>`` is tried FIRST on
    the raw (pre-alias) name: a configured ``providers.openai`` pointing at api.openai.com must not
    be hijacked by the legacy "openai" -> "openrouter" alias."""
    canonical = normalize_provider(name)
    raw = name.strip().lower()
    if user_providers:
        user_pdef = resolve_user_provider(raw, user_providers)
        if user_pdef is not None:
            return user_pdef
    if canonical != raw:
        pdef = _lossy_alias_registry_pdef(raw, canonical)
        if pdef is not None:
            return pdef
    pdef = get_provider(canonical)
    if pdef is not None:
        return pdef
    if user_providers:
        for candidate in (canonical, raw):
            user_pdef = resolve_user_provider(candidate, user_providers)
            if user_pdef is not None:
                return user_pdef
    custom_pdef = resolve_custom_provider(name, custom_providers)
    if custom_pdef is not None:
        return custom_pdef

    # 2c. Managed local runtime: the llamacpp aliases are a real provider
    # whenever the managed server (or a detected external one) resolves —
    # no credential and no providers: entry required, the credential is
    # reachability. Without this rung the model-switch path rejected the
    # very provider the Local Models 'Use' flow writes to config
    # ("Unknown provider 'llamacpp'" from the desktop dropdown).
    if raw in ("llamacpp", "llama.cpp", "llama-cpp"):
        try:
            from hermes_cli.local_runtime.endpoint import resolve_llamacpp_endpoint

            endpoint = resolve_llamacpp_endpoint(wait_for_boot_s=0)
        except Exception:
            endpoint = None
        if endpoint:
            return ProviderDef(
                id="llamacpp",
                name="Local",
                transport="openai_chat",
                api_key_env_vars=(),
                base_url=endpoint["base_url"],
                source="local-runtime",
            )

    # 3. Try models.dev directly (for providers not in our ALIASES)
    try:
        mdev_info = _models_dev_info(canonical)
        if mdev_info is not None:
            return ProviderDef(id=canonical, name=mdev_info.name, transport="openai_chat", api_key_env_vars=mdev_info.env,
                               base_url=mdev_info.api, source="models.dev")
    except Exception:
        pass
    return None

"""
Single source of truth for provider identity in Hermes Agent.

Two data sources, merged at runtime:

1. **models.dev catalog** — 109+ providers with base URLs, env vars, display
   names, and full model metadata (context, cost, capabilities).  This is
   the primary database.

2. **Hermes overlays** — transport type, auth patterns, aggregator flags,
   and additional env vars that models.dev doesn't track.  Small dict,
   maintained here.

3. **User config** (``providers:`` section in config.yaml) — user-defined
   endpoints and overrides.  Merged on top of everything else.

Other modules import from this file.  No parallel registries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# -- Hermes overlay ----------------------------------------------------------
# Hermes-specific metadata that models.dev doesn't provide.

@dataclass(frozen=True)
class HermesOverlay:
    """Hermes-specific provider metadata layered on top of models.dev."""

    transport: str = "openai_chat"        # openai_chat | anthropic_messages | codex_responses
    is_aggregator: bool = False
    auth_type: str = "api_key"            # api_key | oauth_device_code | oauth_external | external_process
    extra_env_vars: Tuple[str, ...] = ()  # env vars models.dev doesn't list
    base_url_override: str = ""           # override if models.dev URL is wrong/missing
    base_url_env_var: str = ""            # env var for user-custom base URL


HERMES_OVERLAYS: Dict[str, HermesOverlay] = {
    "openrouter": HermesOverlay(
        transport="openai_chat",
        is_aggregator=True,
        extra_env_vars=("OPENAI_API_KEY",),
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
    "qwen-oauth": HermesOverlay(
        transport="openai_chat",
        auth_type="oauth_external",
        base_url_override="https://portal.qwen.ai/v1",
        base_url_env_var="HERMES_QWEN_BASE_URL",
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
    "minimax": HermesOverlay(
        transport="anthropic_messages",
        base_url_env_var="MINIMAX_BASE_URL",
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
    "xai": HermesOverlay(
        transport="openai_chat",
        base_url_override="https://api.x.ai/v1",
        base_url_env_var="XAI_BASE_URL",
    ),
    "xiaomi": HermesOverlay(
        transport="openai_chat",
        base_url_env_var="XIAOMI_BASE_URL",
    ),
    "arcee": HermesOverlay(
        transport="openai_chat",
        base_url_override="https://api.arcee.ai/api/v1",
        base_url_env_var="ARCEE_BASE_URL",
    ),
    "ollama-cloud": HermesOverlay(
        transport="openai_chat",
        base_url_env_var="OLLAMA_BASE_URL",
    ),
}


# -- Resolved provider -------------------------------------------------------
# The merged result of models.dev + overlay + user config.

@dataclass
class ProviderDef:
    """Complete provider definition — merged from all sources."""

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

    # kimi-for-coding (models.dev ID)
    "kimi": "kimi-for-coding",
    "kimi-coding": "kimi-for-coding",
    "kimi-coding-cn": "kimi-for-coding",
    "moonshot": "kimi-for-coding",

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

    # huggingface
    "hf": "huggingface",
    "hugging-face": "huggingface",
    "huggingface-hub": "huggingface",

    # xiaomi
    "mimo": "xiaomi",
    "xiaomi-mimo": "xiaomi",

    # bedrock
    "aws": "bedrock",
    "aws-bedrock": "bedrock",
    "amazon-bedrock": "bedrock",
    "amazon": "bedrock",

    # arcee
    "arcee-ai": "arcee",
    "arceeai": "arcee",

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


# -- Display labels -----------------------------------------------------------
# Built dynamically from models.dev + overlays.  Fallback for providers
# not in the catalog.

_LABEL_OVERRIDES: Dict[str, str] = {
    "nous": "Nous Portal",
    "openai-codex": "OpenAI Codex",
    "copilot-acp": "GitHub Copilot ACP",
    "xiaomi": "Xiaomi MiMo",
    "local": "Local endpoint",
    "bedrock": "AWS Bedrock",
    "ollama-cloud": "Ollama Cloud",
}


# -- Transport → API mode mapping ---------------------------------------------

TRANSPORT_TO_API_MODE: Dict[str, str] = {
    "openai_chat": "chat_completions",
    "anthropic_messages": "anthropic_messages",
    "codex_responses": "codex_responses",
    "bedrock_converse": "bedrock_converse",
}


# -- Helper functions ---------------------------------------------------------

def normalize_provider(name: str) -> str:
    """Resolve aliases and normalise casing to a canonical provider id.

    Returns the canonical id string.  Does *not* validate that the id
    corresponds to a known provider.
    """
    key = name.strip().lower()
    return ALIASES.get(key, key)


def get_provider(name: str) -> Optional[ProviderDef]:
    """Look up a provider by id or alias, merging all data sources.

    Resolution order:
      1. Hermes overlays (for providers not in models.dev: nous, openai-codex, etc.)
      2. models.dev catalog + Hermes overlay
      3. User-defined providers from config (TODO: Phase 4)

    Returns a fully-resolved ProviderDef or None.
    """
    canonical = normalize_provider(name)

    # Try to get models.dev data
    try:
        from agent.models_dev import get_provider_info as _mdev_provider
        mdev_info = _mdev_provider(canonical)
    except Exception:
        mdev_info = None

    overlay = HERMES_OVERLAYS.get(canonical)

    if mdev_info is not None:
        # Merge models.dev + overlay
        transport = overlay.transport if overlay else "openai_chat"
        is_agg = overlay.is_aggregator if overlay else False
        auth = overlay.auth_type if overlay else "api_key"
        base_url_env = overlay.base_url_env_var if overlay else ""
        base_url_override = overlay.base_url_override if overlay else ""

        # Combine env vars: models.dev env + hermes extra
        env_vars = list(mdev_info.env)
        if overlay and overlay.extra_env_vars:
            for ev in overlay.extra_env_vars:
                if ev not in env_vars:
                    env_vars.append(ev)

        return ProviderDef(
            id=canonical,
            name=mdev_info.name,
            transport=transport,
            api_key_env_vars=tuple(env_vars),
            base_url=base_url_override or mdev_info.api,
            base_url_env_var=base_url_env,
            is_aggregator=is_agg,
            auth_type=auth,
            doc=mdev_info.doc,
            source="models.dev",
        )

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

    return None


def get_label(provider_id: str) -> str:
    """Get a human-readable display name for a provider."""
    canonical = normalize_provider(provider_id)

    # Check label overrides first
    if canonical in _LABEL_OVERRIDES:
        return _LABEL_OVERRIDES[canonical]

    # Try models.dev
    pdef = get_provider(canonical)
    if pdef:
        return pdef.name

    return canonical




def is_aggregator(provider: str) -> bool:
    """Return True when the provider is a multi-model aggregator."""
    pdef = get_provider(provider)
    return pdef.is_aggregator if pdef else False


def determine_api_mode(provider: str, base_url: str = "") -> str:
    """Determine the API mode (wire protocol) for a provider/endpoint.

    Resolution order:
      1. Known provider → transport → TRANSPORT_TO_API_MODE.
      2. URL heuristics for unknown / custom providers.
      3. Default: 'chat_completions'.
    """
    pdef = get_provider(provider)
    if pdef is not None:
        return TRANSPORT_TO_API_MODE.get(pdef.transport, "chat_completions")

    # Direct provider checks for providers not in HERMES_OVERLAYS
    if provider == "bedrock":
        return "bedrock_converse"

    # URL-based heuristics for custom / unknown providers
    if base_url:
        url_lower = base_url.rstrip("/").lower()
        if url_lower.endswith("/anthropic") or "api.anthropic.com" in url_lower:
            return "anthropic_messages"
        if "api.openai.com" in url_lower:
            return "codex_responses"
        if "bedrock-runtime" in url_lower and "amazonaws.com" in url_lower:
            return "bedrock_converse"

    return "chat_completions"


# -- Provider from user config ------------------------------------------------

def resolve_user_provider(name: str, user_config: Dict[str, Any]) -> Optional[ProviderDef]:
    """Resolve a provider from the user's config.yaml ``providers:`` section.

    Args:
        name: Provider name as given by the user.
        user_config: The ``providers:`` dict from config.yaml.

    Returns:
        ProviderDef if found, else None.
    """
    if not user_config or not isinstance(user_config, dict):
        return None

    entry = user_config.get(name)
    if not isinstance(entry, dict):
        return None

    # Extract fields
    display_name = entry.get("name", "") or name
    api_url = entry.get("api", "") or entry.get("url", "") or entry.get("base_url", "") or ""
    key_env = entry.get("key_env", "") or ""
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


def custom_provider_slug(display_name: str) -> str:
    """Build a canonical slug for a custom_providers entry.

    Matches the convention used by runtime_provider and credential_pool
    (``custom:<normalized-name>``).  Centralised here so all call-sites
    produce identical slugs.
    """
    return "custom:" + display_name.strip().lower().replace(" ", "-")


def resolve_custom_provider(
    name: str,
    custom_providers: Optional[List[Dict[str, Any]]],
) -> Optional[ProviderDef]:
    """Resolve a provider from the user's config.yaml ``custom_providers`` list."""
    if not custom_providers or not isinstance(custom_providers, list):
        return None

    requested = (name or "").strip().lower()
    if not requested:
        return None

    for entry in custom_providers:
        if not isinstance(entry, dict):
            continue

        display_name = (entry.get("name") or "").strip()
        api_url = (
            entry.get("base_url", "")
            or entry.get("url", "")
            or entry.get("api", "")
            or ""
        ).strip()
        if not display_name or not api_url:
            continue

        slug = custom_provider_slug(display_name)
        if requested not in {display_name.lower(), slug}:
            continue

        return ProviderDef(
            id=slug,
            name=display_name,
            transport="openai_chat",
            api_key_env_vars=(),
            base_url=api_url,
            is_aggregator=False,
            auth_type="api_key",
            source="user-config",
        )

    return None


def resolve_provider_full(
    name: str,
    user_providers: Optional[Dict[str, Any]] = None,
    custom_providers: Optional[List[Dict[str, Any]]] = None,
) -> Optional[ProviderDef]:
    """Full resolution chain: built-in → models.dev → user config.

    This is the main entry point for --provider flag resolution.

    Args:
        name: Provider name or alias.
        user_providers: The ``providers:`` dict from config.yaml (optional).
        custom_providers: The ``custom_providers:`` list from config.yaml (optional).

    Returns:
        ProviderDef if found, else None.
    """
    canonical = normalize_provider(name)

    # 1. Built-in (models.dev + overlays)
    pdef = get_provider(canonical)
    if pdef is not None:
        return pdef

    # 2. User-defined providers from config
    if user_providers:
        # Try canonical name
        user_pdef = resolve_user_provider(canonical, user_providers)
        if user_pdef is not None:
            return user_pdef
        # Try original name (in case alias didn't match)
        user_pdef = resolve_user_provider(name.strip().lower(), user_providers)
        if user_pdef is not None:
            return user_pdef

    # 2b. Saved custom providers from config
    custom_pdef = resolve_custom_provider(name, custom_providers)
    if custom_pdef is not None:
        return custom_pdef

    # 3. Try models.dev directly (for providers not in our ALIASES)
    try:
        from agent.models_dev import get_provider_info as _mdev_provider
        mdev_info = _mdev_provider(canonical)
        if mdev_info is not None:
            return ProviderDef(
                id=canonical,
                name=mdev_info.name,
                transport="openai_chat",
                api_key_env_vars=mdev_info.env,
                base_url=mdev_info.api,
                source="models.dev",
            )
    except Exception:
        pass

    return None

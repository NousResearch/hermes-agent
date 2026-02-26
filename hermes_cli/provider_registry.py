"""
Central provider registry for Hermes CLI inference providers.

This module is intentionally lightweight and dependency-safe so multiple
CLI surfaces (setup, model picker, auth resolution, diagnostics) can share
the same provider metadata and resolution behavior.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from hermes_constants import OPENROUTER_BASE_URL

EnvGetter = Callable[[str], Optional[str]]


@dataclass(frozen=True)
class ProviderMeta:
    id: str
    label: str
    auth_type: str  # "oauth" or "api_key"
    default_base_url: str = ""
    api_key_env_vars: Tuple[str, ...] = ()
    base_url_env_var: Optional[str] = None
    curated_models: Tuple[str, ...] = ()
    aliases: Tuple[str, ...] = ()
    show_in_setup: bool = True
    show_in_model_picker: bool = True
    supports_openai_chat: bool = True


PROVIDERS: Dict[str, ProviderMeta] = {
    "openrouter": ProviderMeta(
        id="openrouter",
        label="OpenRouter",
        auth_type="api_key",
        default_base_url=OPENROUTER_BASE_URL,
        api_key_env_vars=("OPENROUTER_API_KEY",),
        base_url_env_var="OPENROUTER_BASE_URL",
        curated_models=(
            "anthropic/claude-opus-4.6",
            "anthropic/claude-sonnet-4.5",
            "openai/gpt-5.2",
            "google/gemini-3-pro-preview",
        ),
    ),
    "nous": ProviderMeta(
        id="nous",
        label="Nous Portal",
        auth_type="oauth",
        show_in_setup=True,
        show_in_model_picker=True,
    ),
    "zai": ProviderMeta(
        id="zai",
        label="GLM (z.ai)",
        auth_type="api_key",
        default_base_url="https://api.z.ai/api/coding/paas/v4",
        api_key_env_vars=("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"),
        base_url_env_var="GLM_BASE_URL",
        curated_models=(
            "glm-4.5-flash",
            "glm-4.5-air",
            "glm-4.5v",
            "glm-4.6",
            "glm-4.7",
            "glm-5",
        ),
        aliases=("glm", "z-ai"),
    ),
    "kimi-coding": ProviderMeta(
        id="kimi-coding",
        label="Kimi Coding",
        auth_type="api_key",
        default_base_url="https://api.kimi.com/coding/v1",
        api_key_env_vars=("KIMI_API_KEY",),
        base_url_env_var="KIMI_BASE_URL",
        curated_models=("kimi-k2-thinking", "k2p5"),
        aliases=("kimi",),
    ),
    "minimax": ProviderMeta(
        id="minimax",
        label="MiniMax",
        auth_type="api_key",
        default_base_url="https://api.minimax.io/v1",
        api_key_env_vars=("MINIMAX_API_KEY",),
        base_url_env_var="MINIMAX_BASE_URL",
        curated_models=("MiniMax-M2.1", "MiniMax-M2.5"),
    ),
    "minimax-cn": ProviderMeta(
        id="minimax-cn",
        label="MiniMax (China)",
        auth_type="api_key",
        default_base_url="https://api.minimaxi.com/v1",
        api_key_env_vars=("MINIMAX_CN_API_KEY",),
        base_url_env_var="MINIMAX_CN_BASE_URL",
        curated_models=("MiniMax-M2.1", "MiniMax-M2.5"),
    ),
    "custom": ProviderMeta(
        id="custom",
        label="Custom endpoint",
        auth_type="api_key",
        default_base_url="",
        api_key_env_vars=("OPENAI_API_KEY",),
        base_url_env_var="OPENAI_BASE_URL",
        curated_models=(),
        show_in_setup=True,
        show_in_model_picker=True,
    ),
}

_ALIAS_TO_PROVIDER: Dict[str, str] = {}
for _pid, _meta in PROVIDERS.items():
    _ALIAS_TO_PROVIDER[_pid] = _pid
    for _alias in _meta.aliases:
        _ALIAS_TO_PROVIDER[_alias.lower()] = _pid


def normalize_provider_id(provider_id: Optional[str], default: str = "auto") -> str:
    """Normalize a provider ID or alias to a canonical ID."""
    if not provider_id:
        return default
    key = provider_id.strip().lower()
    if not key:
        return default
    return _ALIAS_TO_PROVIDER.get(key, key)


def get_provider(provider_id: str) -> Optional[ProviderMeta]:
    return PROVIDERS.get(normalize_provider_id(provider_id))


def list_provider_ids(
    *,
    include_oauth: bool = True,
    include_api_key: bool = True,
    include_custom: bool = True,
) -> List[str]:
    ids: List[str] = []
    for pid, meta in PROVIDERS.items():
        if meta.auth_type == "oauth" and not include_oauth:
            continue
        if meta.auth_type == "api_key" and not include_api_key:
            continue
        if pid == "custom" and not include_custom:
            continue
        ids.append(pid)
    return ids


def list_setup_provider_ids() -> List[str]:
    return [pid for pid, meta in PROVIDERS.items() if meta.show_in_setup]


def list_model_picker_provider_ids() -> List[str]:
    return [pid for pid, meta in PROVIDERS.items() if meta.show_in_model_picker]


def iter_api_key_env_vars(provider_id: str) -> Iterable[str]:
    meta = get_provider(provider_id)
    if not meta:
        return ()
    return meta.api_key_env_vars


def resolve_provider_api_key(
    provider_id: str,
    *,
    env_get: EnvGetter = os.getenv,
    explicit_api_key: Optional[str] = None,
) -> Optional[str]:
    if explicit_api_key:
        return explicit_api_key
    for env_var in iter_api_key_env_vars(provider_id):
        value = env_get(env_var)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def resolve_provider_base_url(
    provider_id: str,
    *,
    env_get: EnvGetter = os.getenv,
    explicit_base_url: Optional[str] = None,
) -> Optional[str]:
    if explicit_base_url:
        return explicit_base_url.strip()
    meta = get_provider(provider_id)
    if not meta:
        return None
    if meta.base_url_env_var:
        env_value = env_get(meta.base_url_env_var)
        if isinstance(env_value, str) and env_value.strip():
            return env_value.strip().rstrip("/")
    if meta.default_base_url:
        return meta.default_base_url.rstrip("/")
    return None


def resolve_provider_base_url_override(
    provider_id: str,
    *,
    env_get: EnvGetter = os.getenv,
) -> Optional[str]:
    """
    Resolve provider base URL from environment override only (no provider default).
    """
    meta = get_provider(provider_id)
    if not meta or not meta.base_url_env_var:
        return None
    env_value = env_get(meta.base_url_env_var)
    if isinstance(env_value, str) and env_value.strip():
        return env_value.strip().rstrip("/")
    return None


def resolve_effective_base_url(
    provider_id: str,
    *,
    env_get: EnvGetter = os.getenv,
    explicit_base_url: Optional[str] = None,
    profile_base_url: Optional[str] = None,
    model_base_url: Optional[str] = None,
    include_openrouter_fallback: bool = True,
) -> Optional[str]:
    """
    Resolve runtime base URL with consistent precedence.

    Order:
    1) explicit CLI --base-url
    2) provider-specific base URL env override
    3) active profile base_url
    4) model-level base_url
    5) provider default base URL
    6) optional OPENROUTER fallback
    """
    if isinstance(explicit_base_url, str) and explicit_base_url.strip():
        return explicit_base_url.strip().rstrip("/")

    env_override = resolve_provider_base_url_override(provider_id, env_get=env_get)
    if env_override:
        return env_override

    if isinstance(profile_base_url, str) and profile_base_url.strip():
        return profile_base_url.strip().rstrip("/")

    if isinstance(model_base_url, str) and model_base_url.strip():
        return model_base_url.strip().rstrip("/")

    meta = get_provider(provider_id)
    if meta and meta.default_base_url:
        return meta.default_base_url.rstrip("/")

    if include_openrouter_fallback:
        return OPENROUTER_BASE_URL
    return None


def has_any_provider_key(provider_id: str, *, env_get: EnvGetter = os.getenv) -> bool:
    for env_var in iter_api_key_env_vars(provider_id):
        value = env_get(env_var)
        if isinstance(value, str) and value.strip():
            return True
    return False


def get_curated_models(provider_id: str) -> List[str]:
    meta = get_provider(provider_id)
    if not meta:
        return []
    return list(meta.curated_models)


def get_provider_model_candidates(
    provider_id: str,
    *,
    openrouter_model_loader: Optional[Callable[[], Iterable[str]]] = None,
) -> List[str]:
    """
    Return model candidates for a provider.
    OpenRouter can use a dynamic loader to fetch the full catalog.
    """
    normalized = normalize_provider_id(provider_id)
    if normalized == "openrouter" and openrouter_model_loader is not None:
        try:
            return list(openrouter_model_loader())
        except Exception:
            return get_curated_models("openrouter")
    return get_curated_models(normalized)


def should_clear_custom_endpoint_env(provider_id: str) -> bool:
    """
    Whether selecting this provider should clear OPENAI_BASE_URL/OPENAI_API_KEY.
    """
    return normalize_provider_id(provider_id) == "openrouter"


def is_supported_provider(provider_id: str) -> bool:
    return normalize_provider_id(provider_id) in PROVIDERS


def detect_auto_provider(*, env_get: EnvGetter = os.getenv) -> str:
    """
    Detect the best API-key provider from environment variables.

    Priority:
    1) Custom endpoint when OPENAI_BASE_URL is set
    2) OpenRouter
    3) zAI
    4) Kimi Coding
    5) MiniMax
    6) MiniMax CN
    7) fallback openrouter
    """
    openai_base_url = env_get("OPENAI_BASE_URL")
    if isinstance(openai_base_url, str) and openai_base_url.strip():
        return "custom"

    for pid in ("openrouter", "zai", "kimi-coding", "minimax", "minimax-cn"):
        if has_any_provider_key(pid, env_get=env_get):
            return pid
    return "openrouter"


def provider_cli_choices(*, include_auto: bool = True) -> List[str]:
    choices: List[str] = []
    if include_auto:
        choices.append("auto")
    # Keep custom out of --provider; custom is inferred from OPENAI_BASE_URL.
    choices.extend(
        pid
        for pid in list_provider_ids(include_oauth=True, include_api_key=True, include_custom=False)
    )
    return choices

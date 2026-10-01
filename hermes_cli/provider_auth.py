"""Live auth projection over provider identity; owns auth policy, never provider registry state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from hermes_cli.auth_constants import (
    CODEX_OAUTH_CLIENT_ID,
    CODEX_OAUTH_TOKEN_URL,
    DEFAULT_NOUS_CLIENT_ID,
    DEFAULT_NOUS_PORTAL_URL,
    DEFAULT_NOUS_SCOPE,
    DEVICE_CODE_GRANT_TYPE,
    MINIMAX_OAUTH_CLIENT_ID,
    MINIMAX_OAUTH_CN_BASE,
    MINIMAX_OAUTH_GLOBAL_BASE,
    MINIMAX_OAUTH_GRANT_TYPE,
    MINIMAX_OAUTH_SCOPE,
    QWEN_OAUTH_CLIENT_ID,
    QWEN_OAUTH_TOKEN_URL,
    XAI_OAUTH_CLIENT_ID,
    XAI_OAUTH_DEVICE_CODE_URL,
    XAI_OAUTH_DISCOVERY_URL,
    XAI_OAUTH_SCOPE,
)
from providers import get_provider_profile, list_providers
from providers.base import ProviderProfile


@dataclass
class ProviderConfig:
    """Auth-domain projection of one effective provider profile."""

    id: str
    name: str
    auth_type: str
    portal_base_url: str = ""
    inference_base_url: str = ""
    client_id: str = ""
    scope: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    api_key_env_vars: tuple = ()
    base_url_env_var: str = ""


@dataclass(frozen=True)
class _AuthPolicy:
    """Provider-specific auth metadata not owned by ProviderProfile."""

    portal_base_url: str = ""
    client_id: str = ""
    scope: str = ""
    extra: tuple[tuple[str, Any], ...] = ()


_AUTH_POLICY: dict[str, _AuthPolicy] = {
    "nous": _AuthPolicy(
        portal_base_url=DEFAULT_NOUS_PORTAL_URL,
        client_id=DEFAULT_NOUS_CLIENT_ID,
        scope=DEFAULT_NOUS_SCOPE,
        extra=(("grant_type", DEVICE_CODE_GRANT_TYPE),),
    ),
    "openai-codex": _AuthPolicy(
        client_id=CODEX_OAUTH_CLIENT_ID,
        extra=(
            ("issuer", "https://auth.openai.com"),
            ("token_url", CODEX_OAUTH_TOKEN_URL),
            ("grant_type", "authorization_code"),
        ),
    ),
    "xai-oauth": _AuthPolicy(
        client_id=XAI_OAUTH_CLIENT_ID,
        scope=XAI_OAUTH_SCOPE,
        extra=(
            ("discovery_url", XAI_OAUTH_DISCOVERY_URL),
            ("device_code_url", XAI_OAUTH_DEVICE_CODE_URL),
            ("grant_type", DEVICE_CODE_GRANT_TYPE),
        ),
    ),
    "qwen-oauth": _AuthPolicy(
        client_id=QWEN_OAUTH_CLIENT_ID,
        extra=(("token_url", QWEN_OAUTH_TOKEN_URL),),
    ),
    "minimax-oauth": _AuthPolicy(
        portal_base_url=MINIMAX_OAUTH_GLOBAL_BASE,
        client_id=MINIMAX_OAUTH_CLIENT_ID,
        scope=MINIMAX_OAUTH_SCOPE,
        extra=(
            ("region", "global"),
            ("cn_portal_base_url", MINIMAX_OAUTH_CN_BASE),
            ("grant_type", MINIMAX_OAUTH_GRANT_TYPE),
        ),
    ),
}

AUTH_AUTO_DETECT_ORDER: tuple[str, ...] = (
    "openai-api",
    "gemini",
    "zai",
    "kimi-coding",
    "kimi-coding-cn",
    "stepfun",
    "arcee",
    "gmi",
    "actual",
    "minimax",
    "anthropic",
    "alibaba",
    "alibaba-coding-plan",
    "minimax-cn",
    "deepseek",
    "xai",
    "nvidia",
    "ai-gateway",
    "opencode-zen",
    "opencode-go",
    "kilocode",
    "huggingface",
    "xiaomi",
    "tencent-tokenhub",
    "tencent-tokenplan",
    "ollama-cloud",
    "azure-foundry",
)

_GENERIC_AUTO_DETECT_EXCLUDED = frozenset({"copilot", "lmstudio", "openrouter"})
AUTH_COMMAND_EXCLUDED_PROVIDER_IDS = frozenset({"custom", "moa"})
CORE_MANAGED_AUTH_PROVIDER_IDS = frozenset({
    "nous", "openai-codex", "xai-oauth", "qwen-oauth", "minimax-oauth",
    "copilot", "copilot-acp", "bedrock", "vertex", "moa",
})

def _policy_extra(profile: ProviderProfile, policy: _AuthPolicy) -> dict[str, Any]:
    extra = dict(policy.extra)
    if profile.name == "minimax-oauth":
        cn_profile = get_provider_profile("minimax-cn")
        if cn_profile is not None and cn_profile.base_url:
            extra["cn_inference_base_url"] = cn_profile.base_url
    return extra


def _project(profile: ProviderProfile) -> ProviderConfig:
    policy = _AUTH_POLICY.get(profile.name, _AuthPolicy())
    return ProviderConfig(
        id=profile.name,
        name=profile.display_name or profile.name,
        auth_type=profile.auth_type,
        portal_base_url=policy.portal_base_url,
        inference_base_url=profile.base_url,
        client_id=policy.client_id,
        scope=policy.scope,
        extra=_policy_extra(profile, policy),
        api_key_env_vars=tuple(profile.env_vars or ()),
        base_url_env_var=profile.base_url_env_var or "",
    )


def get_provider_config(provider: str) -> ProviderConfig | None:
    """Return the live auth projection for a provider name or alias."""

    profile = get_provider_profile(provider)
    return _project(profile) if profile is not None else None


def iter_provider_configs() -> Iterable[ProviderConfig]:
    """Yield live auth projections for all effective canonical providers."""

    for profile in list_providers():
        yield _project(profile)


def iter_auto_detect_provider_configs() -> Iterable[ProviderConfig]:
    """Yield API-key configs in legacy credential-precedence order.

    Newer providers are appended in canonical registry order.
    """

    emitted: set[str] = set()
    for provider_id in AUTH_AUTO_DETECT_ORDER:
        config = get_provider_config(provider_id)
        if config is None:
            continue
        emitted.add(config.id)
        if (
            config.auth_type == "api_key"
            and config.api_key_env_vars
            and config.id not in _GENERIC_AUTO_DETECT_EXCLUDED
        ):
            yield config

    for config in iter_provider_configs():
        if config.id in emitted:
            continue
        if (
            config.auth_type == "api_key"
            and config.api_key_env_vars
            and config.id not in _GENERIC_AUTO_DETECT_EXCLUDED
        ):
            yield config

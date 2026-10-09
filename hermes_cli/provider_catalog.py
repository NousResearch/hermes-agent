"""Live provider presentation catalog.

Provider identity and declarations are owned by :mod:`providers`. This module adds only
presentation policy: stable picker order plus the descriptor shapes consumed by CLI, TUI,
Desktop, and web surfaces. Unknown/late plugin providers are appended in canonical registry
order, so registration is immediately visible without synchronizing a second provider list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from hermes_cli.provider_auth import get_provider_config
from providers import list_providers


# Presentation policy only. Provider existence does NOT come from this tuple.
PROVIDER_PICKER_ORDER: tuple[str, ...] = (
    "nous", "fireworks", "openrouter", "moa", "novita", "lmstudio",
    "anthropic", "openai-codex", "openai-api", "alibaba", "xai-oauth",
    "xiaomi", "tencent-tokenhub", "tencent-tokenplan", "nvidia", "copilot",
    "copilot-acp", "huggingface", "gemini", "vertex", "deepseek", "xai",
    "zai", "kimi-coding", "kimi-coding-cn", "stepfun", "minimax",
    "minimax-oauth", "minimax-cn", "ollama-cloud", "arcee", "gmi",
    "kilocode", "opencode-zen", "opencode-go", "bedrock", "azure-foundry",
    "ai-gateway", "qwen-oauth", "actual", "alibaba-cn",
    "alibaba-token-plan", "alibaba-token-plan-cn", "alibaba-coding-plan",
    "alibaba-coding-plan-cn", "commandcode", "commandcode-anthropic",
    "custom", "deepinfra", "meta-ai", "nebius-token-factory", "router",
    "upstage",
)

_ACCOUNTS_AUTH_TYPES: frozenset[str] = frozenset(
    {"oauth_device_code", "oauth_external", "oauth_minimax", "external_process"}
)

# Auth-domain types and desktop presentation are not identical: Copilot resolves
# tokens through its own auth handler but is configured by provider-owned token vars.
_PROVIDER_TAB_OVERRIDES: dict[str, str] = {"copilot": "keys"}


class ProviderEntry(NamedTuple):
    """Compact picker-facing provider row."""

    slug: str
    label: str
    tui_desc: str


@dataclass(frozen=True)
class ProviderDescriptor:
    """One effective provider as seen by provider-selection surfaces."""

    slug: str
    label: str
    description: str
    auth_type: str
    tab: str
    api_key_env_vars: tuple[str, ...]
    base_url_env_var: str
    signup_url: str
    order: int


def tab_for_auth_type(auth_type: str) -> str:
    return "accounts" if auth_type in _ACCOUNTS_AUTH_TYPES else "keys"


def _ordered_profiles():
    profiles = list(list_providers())
    policy_order = {slug: index for index, slug in enumerate(PROVIDER_PICKER_ORDER)}
    registry_order = {profile.name: index for index, profile in enumerate(profiles)}
    return sorted(
        profiles,
        key=lambda profile: (
            policy_order.get(profile.name, len(policy_order)),
            registry_order[profile.name],
        ),
    )


def _signup_url(profile, api_key_vars: tuple[str, ...]) -> str:
    if profile.signup_url:
        return profile.signup_url
    if not api_key_vars:
        return ""
    try:
        from hermes_cli.config import OPTIONAL_ENV_VARS

        return (OPTIONAL_ENV_VARS.get(api_key_vars[0]) or {}).get("url") or ""
    except Exception:
        return ""


def provider_catalog() -> list[ProviderDescriptor]:
    """Project the current canonical provider registry into stable presentation order."""

    out: list[ProviderDescriptor] = []
    for order, profile in enumerate(_ordered_profiles()):
        config = get_provider_config(profile.name)
        auth_type = (config.auth_type if config else profile.auth_type) or "api_key"
        api_key_vars = (
            tuple(config.api_key_env_vars)
            if config is not None
            else tuple(profile.env_vars or ())
        )
        base_url_var = (
            config.base_url_env_var
            if config is not None
            else (profile.base_url_env_var or "")
        )
        label = profile.display_name or profile.name
        description = profile.description or label
        out.append(
            ProviderDescriptor(
                slug=profile.name,
                label=label,
                description=description,
                auth_type=auth_type,
                tab=_PROVIDER_TAB_OVERRIDES.get(profile.name, tab_for_auth_type(auth_type)),
                api_key_env_vars=api_key_vars,
                base_url_env_var=base_url_var,
                signup_url=_signup_url(profile, api_key_vars),
                order=order,
            )
        )
    return out


def provider_entries() -> list[ProviderEntry]:
    """Compact live picker entries in the same order as :func:`provider_catalog`."""

    return [
        ProviderEntry(descriptor.slug, descriptor.label, descriptor.description)
        for descriptor in provider_catalog()
    ]


def provider_catalog_by_slug() -> dict[str, ProviderDescriptor]:
    return {descriptor.slug: descriptor for descriptor in provider_catalog()}


def provider_slugs() -> list[str]:
    """Effective provider slugs in presentation order."""

    return [descriptor.slug for descriptor in provider_catalog()]

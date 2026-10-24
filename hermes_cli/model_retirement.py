"""Provider-scoped retirement checks for configured primary models.

This module is intentionally static and model-independent.  It must remain usable
when the configured inference route cannot answer, and it never mutates config or
selects a replacement whose availability may vary by account.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from hermes_cli.providers import normalize_provider


OPENAI_CODEX_MODELS_URL = "https://developers.openai.com/codex/models"


@dataclass(frozen=True)
class PrimaryModelRetirement:
    """A provider/auth-route-specific primary-model retirement."""

    provider: str
    model: str
    retired_on: date
    scope: str
    source_url: str


@dataclass(frozen=True)
class PrimaryModelRetirementIssue:
    """A configured primary that matches a known retirement."""

    config_path: str
    provider: str
    configured_model: str
    retirement: PrimaryModelRetirement


GPT55_CODEX_RETIREMENT = PrimaryModelRetirement(
    provider="openai-codex",
    model="gpt-5.5",
    retired_on=date(2026, 10, 14),
    scope="Codex with ChatGPT sign-in; the OpenAI API is not affected",
    source_url=OPENAI_CODEX_MODELS_URL,
)

RETIRED_PRIMARY_MODELS: tuple[PrimaryModelRetirement, ...] = (
    GPT55_CODEX_RETIREMENT,
)


def _normalize_model(provider: str, model: str) -> str:
    """Return the provider-local model ID used by the retirement registry."""
    normalized = str(model or "").strip().lower()
    if provider == "openai-codex" and normalized.startswith("openai/"):
        normalized = normalized.removeprefix("openai/")
    return normalized.removesuffix("-900k")


def retired_models_for_provider(provider: str) -> set[str]:
    """Known retired model IDs for a canonical provider or provider alias."""
    canonical = normalize_provider(str(provider or ""))
    return {
        retirement.model
        for retirement in RETIRED_PRIMARY_MODELS
        if retirement.provider == canonical
    }


def is_model_retired(provider: str, model: str) -> bool:
    """Whether a provider-local model ID matches the retirement registry."""
    canonical = normalize_provider(str(provider or ""))
    normalized_model = _normalize_model(canonical, model)
    return any(
        retirement.provider == canonical and retirement.model == normalized_model
        for retirement in RETIRED_PRIMARY_MODELS
    )


def find_retired_primary_model(config: object) -> PrimaryModelRetirementIssue | None:
    """Return a retirement issue for ``model.provider``/``model.default`` only.

    Fallback, auxiliary, delegation, MoA and session overrides are deliberately
    outside this check.  Their availability and policy need separate controls;
    a primary retirement must not be hidden inside fallback readiness work.
    """
    if not isinstance(config, dict):
        return None
    model_config = config.get("model")
    if not isinstance(model_config, dict):
        return None

    configured_provider = str(model_config.get("provider") or "").strip()
    configured_model = str(
        model_config.get("default") or model_config.get("model") or ""
    ).strip()
    if not configured_provider or not configured_model:
        return None

    provider = normalize_provider(configured_provider)
    normalized_model = _normalize_model(provider, configured_model)
    for retirement in RETIRED_PRIMARY_MODELS:
        if retirement.provider == provider and retirement.model == normalized_model:
            config_key = "default" if model_config.get("default") else "model"
            return PrimaryModelRetirementIssue(
                config_path=f"model.{config_key}",
                provider=provider,
                configured_model=configured_model,
                retirement=retirement,
            )
    return None


def format_retirement_notice(issue: PrimaryModelRetirementIssue) -> str:
    """Render recovery guidance that does not depend on a working LLM."""
    retirement = issue.retirement
    retirement_date = retirement.retired_on.strftime("%B %d, %Y").replace(" 0", " ")
    return (
        f"Configured primary {issue.provider}/{retirement.model} retired on "
        f"{retirement_date} ({retirement.scope}). Choose a model available to your "
        "account and client with `hermes model --refresh`. No configuration was changed. "
        f"Source: {retirement.source_url}"
    )

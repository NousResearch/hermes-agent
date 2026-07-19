"""Bounded product contract for the first Hermes shared-metrics slice."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

SCHEMA_KEY = "hermes.metrics.schema_version"
SCHEMA_VERSION = "hermes.metrics.event.v1"
MODEL_CALL_SCOPE = "hermes.model_call"
SUBSCRIBER_NAME = "hermes.nemo_relay.shared_metrics"
PRIMARY_MODEL_CALL_ROLE = "primary"

EXECUTION_SURFACES: frozenset[str] = frozenset({
    "api",
    "batch",
    "cli",
    "desktop",
    "gateway",
    "python",
    "scheduled_task",
    "tui",
    "other",
    "unknown",
})
PROVIDER_FAMILIES: frozenset[str] = frozenset({
    "aggregator",
    "custom",
    "direct",
    "local",
    "unknown",
})
MODEL_LOCALITIES: frozenset[str] = frozenset({"local", "remote", "unknown"})
MODEL_OUTCOMES: frozenset[str] = frozenset({"cancelled", "failed", "success"})

# Shared metrics use an explicit family allowlist rather than raw model IDs or
# dynamically sourced catalog values. The latter would make the exported schema
# drift independently of this contract.
MODEL_FAMILIES: frozenset[str] = frozenset({
    "claude",
    "deepseek",
    "gemini",
    "gemma",
    "glm",
    "gpt",
    "grok",
    "kimi",
    "llama",
    "minimax",
    "mimo",
    "mistral",
    "nemotron",
    "nova",
    "qwen",
    "step",
    "trinity",
    "o1",
    "o3",
    "o4",
    "unknown",
})

_MODEL_FAMILY_PATTERN = re.compile(
    r"(?:^|[/_.:-])("
    + "|".join(
        re.escape(family)
        for family in sorted(
            MODEL_FAMILIES - {"unknown"},
            key=lambda value: len(value),
            reverse=True,
        )
    )
    + r")(?=$|[/_.:-]|\d)"
)

# These providers route across model families but are not marked as aggregators
# in Hermes's execution metadata because that flag has narrower routing/catalog
# semantics there.
_TELEMETRY_AGGREGATOR_OVERRIDES = frozenset({
    "copilot-acp",
    "github-copilot",
    "moa",
    "nous",
})

# Hermes intentionally resolves these local runtimes through the generic custom
# provider path, so canonical provider metadata cannot distinguish them alone.
_LOCAL_CUSTOM_PROVIDER_ALIASES = frozenset({"mlx", "ollama"})


def model_call_dimensions(event: Any) -> dict[str, str] | None:
    """Return package dimensions for one valid primary model-call end event."""
    metadata = getattr(event, "metadata", None)
    if not isinstance(metadata, dict) or metadata.get(SCHEMA_KEY) != SCHEMA_VERSION:
        return None
    relay_metadata = set(metadata) - {SCHEMA_KEY}
    if relay_metadata - {"otel.status_code"} or metadata.get(
        "otel.status_code", "OK"
    ) not in {"OK", "ERROR"}:
        return None
    if (
        str(getattr(event, "kind", "") or "") != "scope"
        or str(getattr(event, "category", "") or "") != "llm"
        or str(getattr(event, "name", "") or "") != MODEL_CALL_SCOPE
        or str(getattr(event, "scope_category", "") or "") != "end"
    ):
        return None
    category_profile = getattr(event, "category_profile", None)
    if not isinstance(category_profile, dict) or set(category_profile) != {
        "model_name"
    }:
        return None
    event_model_family = category_profile.get("model_name")
    if event_model_family not in MODEL_FAMILIES:
        return None
    data = getattr(event, "data", None)
    expected_fields = {
        "call_role",
        "locality",
        "model_family",
        "outcome",
        "provider_family",
    }
    if not isinstance(data, dict) or set(data) != expected_fields:
        return None
    if (
        data.get("call_role") != PRIMARY_MODEL_CALL_ROLE
        or data.get("locality") not in MODEL_LOCALITIES
        or data.get("model_family") not in MODEL_FAMILIES
        or data.get("outcome") not in MODEL_OUTCOMES
        or data.get("provider_family") not in PROVIDER_FAMILIES
    ):
        return None
    return {
        "call_role": PRIMARY_MODEL_CALL_ROLE,
        "locality": data["locality"],
        "model_family": data["model_family"],
        "outcome": data["outcome"],
        "provider_family": data["provider_family"],
    }


def execution_surface(kwargs: dict[str, Any]) -> str:
    """Normalize the safe session surface carried by the parent Relay scope."""
    value = (
        str(kwargs.get("execution_surface") or kwargs.get("platform") or "unknown")
        .strip()
        .lower()
    )
    if value in EXECUTION_SURFACES:
        return value
    if value == "api_server":
        return "api"
    if value in {"cron", "scheduler", "scheduled"}:
        return "scheduled_task"
    try:
        from hermes_cli.platforms import get_all_platforms

        if value in get_all_platforms():
            return "gateway"
    except Exception:
        pass
    if value in {"discord", "email", "slack", "telegram", "teams", "whatsapp"}:
        return "gateway"
    return "unknown" if value == "unknown" else "other"


def provider_family(kwargs: dict[str, Any]) -> str:
    """Map a Hermes provider to a bounded product category."""
    raw_provider = str(kwargs.get("provider") or "").strip().lower().replace("_", "-")
    if not raw_provider:
        return "unknown"
    if raw_provider in _LOCAL_CUSTOM_PROVIDER_ALIASES:
        return "local"
    if raw_provider == "custom" or raw_provider.startswith(("custom-", "custom:")):
        return "custom"
    provider, is_aggregator, is_known = _provider_metadata(raw_provider)
    if provider in {"lmstudio", "local"}:
        return "local"
    if is_aggregator or provider in _TELEMETRY_AGGREGATOR_OVERRIDES:
        return "aggregator"
    if provider == "custom":
        return "custom"
    return "direct" if is_known else "unknown"


def _provider_metadata(provider: str) -> tuple[str, bool, bool]:
    """Resolve provider identity without refreshing remote provider metadata."""
    try:
        from hermes_cli.models import normalize_provider as normalize_model_provider
        from hermes_cli.providers import HERMES_OVERLAYS, normalize_provider

        canonical = normalize_provider(normalize_model_provider(provider))
        overlay = HERMES_OVERLAYS.get(canonical)
        return (
            canonical,
            bool(overlay and overlay.is_aggregator),
            canonical in _known_provider_ids(),
        )
    except Exception:
        return provider, False, False


@lru_cache(maxsize=1)
def _known_provider_ids() -> frozenset[str]:
    """Cache Hermes's static provider catalog for the process lifetime."""
    try:
        from hermes_cli.provider_catalog import provider_catalog_by_slug

        return frozenset(provider_catalog_by_slug())
    except Exception:
        return frozenset()


def model_locality(kwargs: dict[str, Any]) -> str:
    """Classify local endpoints without exporting their URL."""
    return _model_locality(kwargs, provider_family(kwargs))


def _model_locality(kwargs: dict[str, Any], provider_category: str) -> str:
    base_url = kwargs.get("base_url")
    if isinstance(base_url, str) and base_url:
        try:
            from agent.model_metadata import is_local_endpoint

            if is_local_endpoint(base_url):
                return "local"
        except Exception:
            pass
    if provider_category == "local":
        return "local"
    if provider_category in {"aggregator", "direct"}:
        return "remote"
    return "unknown"


def model_call_fields(kwargs: dict[str, Any]) -> dict[str, str]:
    """Build the bounded producer fields for one logical model call."""
    provider_category = provider_family(kwargs)
    return {
        "call_role": PRIMARY_MODEL_CALL_ROLE,
        "locality": _model_locality(kwargs, provider_category),
        "model_family": model_family(kwargs),
        "provider_family": provider_category,
    }


def model_family(kwargs: dict[str, Any]) -> str:
    """Map a raw model identifier to an allowlisted family."""
    declared_family = str(kwargs.get("model_family") or "").strip().lower()
    if declared_family in MODEL_FAMILIES - {"unknown"}:
        return declared_family
    model = str(kwargs.get("response_model") or kwargs.get("model") or "").lower()
    match = _MODEL_FAMILY_PATTERN.search(model)
    return match.group(1) if match is not None else "unknown"


def model_call_outcome(kwargs: dict[str, Any]) -> str:
    """Fail closed when a terminal model-call outcome is not recognized."""
    value = str(kwargs.get("outcome") or "").lower()
    return value if value in MODEL_OUTCOMES else "failed"

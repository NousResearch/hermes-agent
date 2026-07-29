"""Bounded product contract for the first Hermes shared-metrics slice."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from agent.relay_runtime import RUNTIME_INSTANCE_KEY

SCHEMA_KEY = "hermes.metrics.schema_version"
SCHEMA_VERSION = "hermes.metrics.event.v1"
MODEL_CALL_SCOPE = "hermes.model_call"
TASK_SCOPE = "hermes.task_run"
SUBSCRIBER_NAME = "hermes.nemo_relay.shared_metrics"
PRIMARY_MODEL_CALL_ROLE = "primary"
MODEL_CALL_METRIC = "hermes.model_call.count"
TASK_STARTED_METRIC = "hermes.task_run.started"
TASK_FINISHED_METRIC = "hermes.task_run.finished"

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
TASK_OUTCOMES: frozenset[str] = frozenset({
    "cancelled",
    "failed",
    "success",
    "timed_out",
    "unknown",
})
TASK_END_REASONS: frozenset[str] = frozenset({
    "approval_denied",
    "completed",
    "failed",
    "guardrail_blocked",
    "iteration_limit",
    "system_aborted",
    "timed_out",
    "unknown",
    "user_cancelled",
})
TASK_TERMINATIONS: frozenset[str] = frozenset({
    "none",
    "system_aborted",
    "timed_out",
    "unknown",
    "user_cancelled",
})
TASK_ENTRYPOINTS: frozenset[str] = frozenset({
    "api",
    "background",
    "batch",
    "delegated",
    "gateway_message",
    "interactive",
    "other",
    "python",
    "scheduled_task",
    "unknown",
})
DURATION_BUCKETS: frozenset[str] = frozenset({
    "1s_to_5s",
    "2m_to_10m",
    "30s_to_2m",
    "5s_to_30s",
    "gte_10m",
    "lt_1s",
})
COUNT_BUCKETS: frozenset[str] = frozenset({
    "0",
    "1",
    "2",
    "3_to_5",
    "6_to_10",
    "gte_11",
})

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

_COUNTER_DIMENSION_VALUES: dict[str, dict[str, frozenset[str]]] = {
    MODEL_CALL_METRIC: {
        "call_role": frozenset({PRIMARY_MODEL_CALL_ROLE}),
        "locality": MODEL_LOCALITIES,
        "model_family": MODEL_FAMILIES,
        "outcome": MODEL_OUTCOMES,
        "provider_family": PROVIDER_FAMILIES,
    },
    TASK_STARTED_METRIC: {
        "entrypoint": TASK_ENTRYPOINTS,
        "execution_surface": EXECUTION_SURFACES,
    },
    TASK_FINISHED_METRIC: {
        "duration_bucket": DURATION_BUCKETS,
        "end_reason": TASK_END_REASONS,
        "entrypoint": TASK_ENTRYPOINTS,
        "execution_surface": EXECUTION_SURFACES,
        "model_call_count_bucket": COUNT_BUCKETS,
        "outcome": TASK_OUTCOMES,
        "retry_count_bucket": COUNT_BUCKETS,
        "termination": TASK_TERMINATIONS,
        "tool_call_count_bucket": COUNT_BUCKETS,
    },
}
COUNTER_METRICS: frozenset[str] = frozenset(_COUNTER_DIMENSION_VALUES)

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


def counter_dimensions_are_valid(
    metric_name: str,
    dimensions: dict[str, Any],
) -> bool:
    """Return whether dimensions match one closed shared-metric contract."""
    contract = _COUNTER_DIMENSION_VALUES.get(metric_name)
    if contract is None or set(dimensions) != set(contract):
        return False
    return all(
        isinstance(dimensions[field], str)
        and dimensions[field] in allowed_values
        for field, allowed_values in contract.items()
    )


def model_call_dimensions(event: Any) -> dict[str, str] | None:
    """Return package dimensions for one valid primary model-call end event."""
    metadata = getattr(event, "metadata", None)
    if not isinstance(metadata, dict) or metadata.get(SCHEMA_KEY) != SCHEMA_VERSION:
        return None
    relay_metadata = set(metadata) - {SCHEMA_KEY, RUNTIME_INSTANCE_KEY}
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
    dimensions = {
        "call_role": data.get("call_role"),
        "locality": data.get("locality"),
        "model_family": data.get("model_family"),
        "outcome": data.get("outcome"),
        "provider_family": data.get("provider_family"),
    }
    if not counter_dimensions_are_valid(MODEL_CALL_METRIC, dimensions):
        return None
    return dimensions


def task_counter(event: Any) -> tuple[str, dict[str, str]] | None:
    """Return one validated task counter from a task scope event."""
    metadata = getattr(event, "metadata", None)
    if not isinstance(metadata, dict) or metadata.get(SCHEMA_KEY) != SCHEMA_VERSION:
        return None
    relay_metadata = set(metadata) - {SCHEMA_KEY, RUNTIME_INSTANCE_KEY}
    if relay_metadata - {"otel.status_code"} or metadata.get(
        "otel.status_code", "OK"
    ) not in {"OK", "ERROR"}:
        return None
    if (
        str(getattr(event, "kind", "") or "") != "scope"
        or str(getattr(event, "category", "") or "") != "function"
        or str(getattr(event, "name", "") or "") != TASK_SCOPE
    ):
        return None
    if getattr(event, "category_profile", None) is not None:
        return None

    scope_category = str(getattr(event, "scope_category", "") or "")
    data = getattr(event, "data", None)
    if scope_category == "start":
        expected_fields = {"entrypoint", "execution_surface"}
        if not isinstance(data, dict) or set(data) != expected_fields:
            return None
        dimensions = {
            "entrypoint": data.get("entrypoint"),
            "execution_surface": data.get("execution_surface"),
        }
        if not counter_dimensions_are_valid(TASK_STARTED_METRIC, dimensions):
            return None
        return TASK_STARTED_METRIC, dimensions

    expected_fields = {
        "duration_bucket",
        "end_reason",
        "entrypoint",
        "execution_surface",
        "model_call_count_bucket",
        "outcome",
        "retry_count_bucket",
        "termination",
        "tool_call_count_bucket",
    }
    if (
        scope_category != "end"
        or not isinstance(data, dict)
        or set(data) != expected_fields
    ):
        return None
    dimensions = {field: data.get(field) for field in sorted(expected_fields)}
    if not counter_dimensions_are_valid(TASK_FINISHED_METRIC, dimensions):
        return None
    return TASK_FINISHED_METRIC, dimensions


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


def task_start_fields(kwargs: dict[str, Any]) -> dict[str, str]:
    """Build the bounded fields recorded on a task scope start event."""
    surface = execution_surface(kwargs)
    return {
        "entrypoint": task_entrypoint(kwargs, surface),
        "execution_surface": surface,
    }


def task_entrypoint(kwargs: dict[str, Any], surface: str | None = None) -> str:
    """Normalize the task dispatch owner without exporting source strings."""
    declared = str(kwargs.get("entrypoint") or "").strip().lower()
    if declared in TASK_ENTRYPOINTS:
        return declared
    resolved_surface = surface or execution_surface(kwargs)
    if kwargs.get("parent_task_id") or kwargs.get("parent_session_id"):
        return "delegated"
    return {
        "api": "api",
        "batch": "batch",
        "cli": "interactive",
        "desktop": "interactive",
        "gateway": "gateway_message",
        "python": "python",
        "scheduled_task": "scheduled_task",
        "tui": "interactive",
        "unknown": "unknown",
    }.get(resolved_surface, "other")


def task_terminal_fields(
    kwargs: dict[str, Any],
    *,
    duration_ms: int,
    model_call_count: int,
    tool_call_count: int,
    retry_count: int,
) -> dict[str, str]:
    """Build the bounded terminal payload for one task scope."""
    start_fields = task_start_fields(kwargs)
    outcome, end_reason, termination = task_terminal_state(kwargs)
    return {
        **start_fields,
        "duration_bucket": duration_bucket(duration_ms),
        "end_reason": end_reason,
        "model_call_count_bucket": count_bucket(model_call_count),
        "outcome": outcome,
        "retry_count_bucket": count_bucket(retry_count),
        "termination": termination,
        "tool_call_count_bucket": count_bucket(tool_call_count),
    }


def task_terminal_state(kwargs: dict[str, Any]) -> tuple[str, str, str]:
    """Map Hermes terminal state to bounded task outcome dimensions."""
    reason = str(kwargs.get("turn_exit_reason") or "").strip().lower()
    if kwargs.get("interrupted") or "interrupt" in reason or "cancel" in reason:
        return "cancelled", "user_cancelled", "user_cancelled"
    if "timeout" in reason or "timed_out" in reason:
        return "timed_out", "timed_out", "timed_out"
    if "max_iterations" in reason or "budget_exhausted" in reason:
        return "failed", "iteration_limit", "system_aborted"
    if "approval" in reason and ("denied" in reason or "rejected" in reason):
        return "failed", "approval_denied", "none"
    if "guardrail" in reason:
        return "failed", "guardrail_blocked", "system_aborted"
    if reason == "system_aborted":
        return "failed", "system_aborted", "system_aborted"
    if kwargs.get("completed") is True:
        return "success", "completed", "none"
    if kwargs.get("failed") is True or (reason and reason != "unknown"):
        return "failed", "failed", "none"
    return "unknown", "unknown", "unknown"


def duration_bucket(duration_ms: int) -> str:
    """Bucket a non-negative task duration into a fixed low-cardinality range."""
    value = max(0, int(duration_ms))
    if value < 1_000:
        return "lt_1s"
    if value < 5_000:
        return "1s_to_5s"
    if value < 30_000:
        return "5s_to_30s"
    if value < 120_000:
        return "30s_to_2m"
    if value < 600_000:
        return "2m_to_10m"
    return "gte_10m"


def count_bucket(count: int) -> str:
    """Bucket a non-negative per-task count into a fixed range."""
    value = max(0, int(count))
    if value <= 2:
        return str(value)
    if value <= 5:
        return "3_to_5"
    if value <= 10:
        return "6_to_10"
    return "gte_11"


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

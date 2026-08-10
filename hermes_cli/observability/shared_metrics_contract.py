"""Bounded product contract for the first Hermes shared-metrics slice."""

from __future__ import annotations

from math import isfinite
from typing import Any

from agent.relay_runtime import (
    LOGICAL_LLM_SCOPE,
    RUNTIME_INSTANCE_KEY,
    RUNTIME_SCHEMA_KEY,
    RUNTIME_SCHEMA_VERSION,
)

SCHEMA_KEY = "hermes.metrics.schema_version"
SCHEMA_VERSION = "hermes.metrics.event.v2"
MODEL_CALL_SCOPE = "hermes.model_call"
MODEL_CALL_PROFILE_MODEL = "unknown"
TASK_SCOPE = "hermes.task_run"
TOOL_CALL_SCOPE = "hermes.tool_call"
CLIENT_ACTIVE_MARK = "hermes.client.active"
TOOL_APPROVAL_MARK = "hermes.tool_approval"
SKILL_LIFECYCLE_MARK = "hermes.skill.lifecycle"
SKILL_LOAD_MARK = "hermes.skill.load"
SUBSCRIBER_NAME = "hermes.nemo_relay.shared_metrics"
CLIENT_ACTIVE_METRIC = "hermes.client.active"
LEGACY_MODEL_CALL_METRIC = "hermes.model_call.count"
MODEL_ROUTE_METRIC = "hermes.model_route.count"
TASK_STARTED_METRIC = "hermes.task_run.started"
TASK_FINISHED_METRIC = "hermes.task_run.finished"
TOOL_CALL_METRIC = "hermes.tool_call.count"
TOOL_APPROVAL_METRIC = "hermes.tool_approval.count"
SKILL_LIFECYCLE_METRIC = "hermes.skill.lifecycle.count"
SKILL_LOAD_METRIC = "hermes.skill.load.count"
MODEL_IDENTIFIER_MAX_LENGTH = 256
PROVIDER_IDENTIFIER_MAX_LENGTH = 64
_METRIC_IDENTIFIER_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyz0123456789._:/@+-"
)
_METRIC_IDENTIFIER_START_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyz0123456789"
)

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
TOOL_CATEGORIES: frozenset[str] = frozenset({
    "browser",
    "code_execution",
    "communication",
    "computer_use",
    "delegation",
    "file",
    "home_automation",
    "mcp",
    "media",
    "memory",
    "other",
    "planning",
    "project",
    "scheduler",
    "skill",
    "terminal",
    "unknown",
    "web",
})
TOOL_OUTCOMES: frozenset[str] = frozenset({
    "blocked",
    "cancelled",
    "failed",
    "success",
    "timed_out",
    "unknown",
})
TOOL_APPROVAL_OUTCOMES: frozenset[str] = frozenset({
    "approved",
    "denied",
    "not_required",
    "timed_out",
    "unknown",
})
TOOL_APPROVAL_ATTRIBUTIONS: frozenset[str] = frozenset({
    "tool_call",
    "unattributed",
})
TOOL_LATENCY_BUCKETS: frozenset[str] = frozenset({
    "100ms_to_250ms",
    "10s_to_30s",
    "1s_to_2s",
    "250ms_to_500ms",
    "2s_to_5s",
    "500ms_to_1s",
    "5s_to_10s",
    "gte_30s",
    "lt_100ms",
    "unknown",
})
TOOL_RETRY_BUCKETS: frozenset[str] = COUNT_BUCKETS | frozenset({"unknown"})
SKILL_LIFECYCLE_ACTIONS: frozenset[str] = frozenset({
    "archived",
    "created",
    "edited",
    "installed",
    "patched",
    "restored",
    "stale",
})
SKILL_PROVENANCES: frozenset[str] = frozenset({
    "agent_created",
    "external",
    "installed",
    "local",
    "unknown",
})
SKILL_REUSE_STATES: frozenset[str] = frozenset({"first_use", "reused"})
SKILL_POST_PATCH_STATES: frozenset[str] = frozenset({
    "no_new_patch",
    "not_applicable",
    "reused_after_patch",
})
CLIENT_OS_FAMILIES: frozenset[str] = frozenset({
    "linux",
    "macos",
    "unknown",
    "windows",
})
CLIENT_ARCHITECTURES: frozenset[str] = frozenset({
    "arm",
    "arm64",
    "unknown",
    "x86",
    "x86_64",
})
CLIENT_INSTALL_METHODS: frozenset[str] = frozenset({
    "docker",
    "git",
    "homebrew",
    "nixos",
    "pip",
    "unknown",
})
CLIENT_RESOURCE_KEYS: frozenset[str] = frozenset({
    "architecture",
    "hermes_version",
    "install_method",
    "os_family",
})

def client_os_family(value: Any) -> str:
    """Map a platform system name to the shared-metrics OS taxonomy."""
    normalized = str(value or "").strip().lower()
    return {
        "darwin": "macos",
        "linux": "linux",
        "macos": "macos",
        "windows": "windows",
    }.get(normalized, "unknown")


def client_architecture(value: Any) -> str:
    """Map a machine architecture to the shared-metrics taxonomy."""
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized in {"amd64", "x64", "x86_64"}:
        return "x86_64"
    if normalized in {"aarch64", "arm64"}:
        return "arm64"
    if normalized in {"i386", "i486", "i586", "i686", "x86"}:
        return "x86"
    if normalized.startswith("armv"):
        return "arm"
    return "unknown"


def client_install_method(value: Any) -> str:
    """Return an allowlisted Hermes installation method."""
    normalized = str(value or "").strip().lower()
    if normalized == "nix":
        return "nixos"
    return normalized if normalized in CLIENT_INSTALL_METHODS else "unknown"


def client_resource(
    hermes_version: Any,
    *,
    os_name: Any,
    architecture: Any,
    install_method: Any,
) -> dict[str, str]:
    """Build the bounded client resource attached to aggregate packages."""
    normalized_version = str(hermes_version or "").strip()
    if not normalized_version or len(normalized_version) > 64:
        normalized_version = "unknown"
    return {
        "architecture": client_architecture(architecture),
        "hermes_version": normalized_version,
        "install_method": client_install_method(install_method),
        "os_family": client_os_family(os_name),
    }


def client_resource_is_valid(resource: Any) -> bool:
    """Return whether a package resource exactly matches the bounded contract."""
    if not isinstance(resource, dict) or set(resource) != CLIENT_RESOURCE_KEYS:
        return False
    version = resource.get("hermes_version")
    return (
        isinstance(version, str)
        and 0 < len(version) <= 64
        and resource.get("os_family") in CLIENT_OS_FAMILIES
        and resource.get("architecture") in CLIENT_ARCHITECTURES
        and resource.get("install_method") in CLIENT_INSTALL_METHODS
    )


_LEGACY_PROVIDER_FAMILIES = frozenset({
    "aggregator",
    "custom",
    "direct",
    "local",
    "unknown",
})
_LEGACY_MODEL_LOCALITIES = frozenset({"local", "remote", "unknown"})
_LEGACY_MODEL_OUTCOMES = frozenset({"cancelled", "failed", "success"})
_LEGACY_MODEL_FAMILIES = frozenset({
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
    "o1",
    "o3",
    "o4",
    "qwen",
    "step",
    "trinity",
    "unknown",
})

_COUNTER_DIMENSION_VALUES: dict[str, dict[str, frozenset[str]]] = {
    CLIENT_ACTIVE_METRIC: {},
    # Retained only so pre-v2 pending rows remain packageable.
    LEGACY_MODEL_CALL_METRIC: {
        "call_role": frozenset({"primary"}),
        "locality": _LEGACY_MODEL_LOCALITIES,
        "model_family": _LEGACY_MODEL_FAMILIES,
        "outcome": _LEGACY_MODEL_OUTCOMES,
        "provider_family": _LEGACY_PROVIDER_FAMILIES,
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
    TOOL_CALL_METRIC: {
        "approval_outcome": TOOL_APPROVAL_OUTCOMES,
        "latency_bucket": TOOL_LATENCY_BUCKETS,
        "outcome": TOOL_OUTCOMES,
        "retry_count_bucket": TOOL_RETRY_BUCKETS,
        "tool_category": TOOL_CATEGORIES,
    },
    TOOL_APPROVAL_METRIC: {
        "attribution": TOOL_APPROVAL_ATTRIBUTIONS,
        "outcome": TOOL_APPROVAL_OUTCOMES - {"not_required"},
    },
    SKILL_LIFECYCLE_METRIC: {
        "action": SKILL_LIFECYCLE_ACTIONS,
        "provenance": SKILL_PROVENANCES,
    },
    SKILL_LOAD_METRIC: {
        "post_patch_state": SKILL_POST_PATCH_STATES,
        "provenance": SKILL_PROVENANCES,
        "reuse_state": SKILL_REUSE_STATES,
        "use_count_bucket": COUNT_BUCKETS,
    },
}
COUNTER_METRICS: frozenset[str] = frozenset({
    CLIENT_ACTIVE_METRIC,
    MODEL_ROUTE_METRIC,
    SKILL_LIFECYCLE_METRIC,
    SKILL_LOAD_METRIC,
    TASK_FINISHED_METRIC,
    TASK_STARTED_METRIC,
    TOOL_APPROVAL_METRIC,
    TOOL_CALL_METRIC,
})


def counter_dimensions_are_valid(
    metric_name: str,
    dimensions: dict[str, Any],
) -> bool:
    """Return whether dimensions match one closed shared-metric contract."""
    if metric_name == MODEL_ROUTE_METRIC:
        return (
            set(dimensions) == {"model", "provider"}
            and dimensions["model"]
            == _metric_identifier(
                dimensions["model"],
                max_length=MODEL_IDENTIFIER_MAX_LENGTH,
            )
            and dimensions["provider"]
            == _metric_identifier(
                dimensions["provider"],
                max_length=PROVIDER_IDENTIFIER_MAX_LENGTH,
            )
        )
    contract = _COUNTER_DIMENSION_VALUES.get(metric_name)
    if contract is None or set(dimensions) != set(contract):
        return False
    return all(
        isinstance(dimensions[field], str) and dimensions[field] in allowed_values
        for field, allowed_values in contract.items()
    )


def _event_metadata_is_valid(event: Any) -> bool:
    metadata = getattr(event, "metadata", None)
    if not isinstance(metadata, dict) or metadata.get(SCHEMA_KEY) != SCHEMA_VERSION:
        return False
    relay_metadata = set(metadata) - {SCHEMA_KEY, RUNTIME_INSTANCE_KEY}
    return not relay_metadata - {"otel.status_code"} and metadata.get(
        "otel.status_code", "OK"
    ) in {"OK", "ERROR"}


def client_active_counter(event: Any) -> tuple[str, dict[str, str]] | None:
    """Return the active-install counter for one empty allowlisted mark."""
    if not _event_metadata_is_valid(event):
        return None
    if (
        str(getattr(event, "kind", "") or "") != "mark"
        or str(getattr(event, "name", "") or "") != CLIENT_ACTIVE_MARK
        or getattr(event, "category", None) is not None
        or getattr(event, "scope_category", None) is not None
        or getattr(event, "category_profile", None) is not None
        or getattr(event, "data", None) != {}
    ):
        return None
    return CLIENT_ACTIVE_METRIC, {}


def model_call_dimensions(event: Any) -> dict[str, str] | None:
    """Return package dimensions for one valid logical model-call end event."""
    auxiliary = _auxiliary_model_call_dimensions(event)
    if auxiliary is not None:
        return auxiliary
    if not _event_metadata_is_valid(event):
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
    # The synthetic scope can span provider fallback. The accepted terminal
    # route is carried in the validated payload rather than this start profile.
    if category_profile.get("model_name") != MODEL_CALL_PROFILE_MODEL:
        return None
    data = getattr(event, "data", None)
    expected_fields = {"model", "provider"}
    if not isinstance(data, dict) or set(data) != expected_fields:
        return None
    dimensions = {field: data.get(field) for field in sorted(expected_fields)}
    if not counter_dimensions_are_valid(MODEL_ROUTE_METRIC, dimensions):
        return None
    return dimensions


def _auxiliary_model_call_dimensions(event: Any) -> dict[str, str] | None:
    """Project a terminal auxiliary route from its Hermes logical scope."""
    metadata = getattr(event, "metadata", None)
    if (
        not isinstance(metadata, dict)
        or metadata.get(RUNTIME_SCHEMA_KEY) != RUNTIME_SCHEMA_VERSION
    ):
        return None
    relay_metadata = set(metadata) - {
        RUNTIME_INSTANCE_KEY,
        RUNTIME_SCHEMA_KEY,
        "hermes.call_role",
    }
    if relay_metadata - {"otel.status_code"} or metadata.get(
        "otel.status_code", "OK"
    ) not in {"OK", "ERROR"}:
        return None
    call_role = metadata.get("hermes.call_role")
    if not isinstance(call_role, str) or not call_role.startswith("auxiliary:"):
        return None
    if (
        str(getattr(event, "kind", "") or "") != "scope"
        or str(getattr(event, "category", "") or "") != "function"
        or str(getattr(event, "name", "") or "") != LOGICAL_LLM_SCOPE
        or str(getattr(event, "scope_category", "") or "") != "end"
        or getattr(event, "category_profile", None) is not None
    ):
        return None
    data = getattr(event, "data", None)
    if (
        not isinstance(data, dict)
        or set(data)
        not in (
            {"model", "outcome", "provider"},
            {"model", "outcome", "provider", "response_model"},
        )
        or data.get("outcome") not in {"cancelled", "failed", "success"}
    ):
        return None
    dimensions = model_call_fields(data)
    if not counter_dimensions_are_valid(MODEL_ROUTE_METRIC, dimensions):
        return None
    return dimensions


def task_counter(event: Any) -> tuple[str, dict[str, str]] | None:
    """Return one validated task counter from a task scope event."""
    if not _event_metadata_is_valid(event):
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


def tool_call_dimensions(event: Any) -> dict[str, str] | None:
    """Return package dimensions for one allowlisted tool lifecycle end event."""
    if not _event_metadata_is_valid(event):
        return None
    if (
        str(getattr(event, "kind", "") or "") != "scope"
        or str(getattr(event, "category", "") or "") != "tool"
        or str(getattr(event, "name", "") or "") != TOOL_CALL_SCOPE
        or str(getattr(event, "scope_category", "") or "") != "end"
        or getattr(event, "category_profile", None) != {}
    ):
        return None
    data = getattr(event, "data", None)
    expected_fields = {
        "approval_outcome",
        "latency_bucket",
        "outcome",
        "retry_count_bucket",
        "tool_category",
    }
    if not isinstance(data, dict) or set(data) != expected_fields:
        return None
    dimensions = {field: data.get(field) for field in sorted(expected_fields)}
    if not counter_dimensions_are_valid(TOOL_CALL_METRIC, dimensions):
        return None
    return dimensions


def tool_approval_counter(event: Any) -> tuple[str, dict[str, str]] | None:
    """Return one validated approval counter from a safe Relay mark event."""
    if not _event_metadata_is_valid(event):
        return None
    if (
        str(getattr(event, "kind", "") or "") != "mark"
        or str(getattr(event, "name", "") or "") != TOOL_APPROVAL_MARK
        or getattr(event, "category", None) is not None
        or getattr(event, "scope_category", None) is not None
        or getattr(event, "category_profile", None) is not None
    ):
        return None
    data = getattr(event, "data", None)
    expected_fields = {"attribution", "outcome"}
    if not isinstance(data, dict) or set(data) != expected_fields:
        return None
    dimensions = {field: data.get(field) for field in sorted(expected_fields)}
    if not counter_dimensions_are_valid(TOOL_APPROVAL_METRIC, dimensions):
        return None
    return TOOL_APPROVAL_METRIC, dimensions


def skill_counter(event: Any) -> tuple[str, dict[str, str]] | None:
    """Return one validated skill lifecycle or load counter from a safe mark."""
    if not _event_metadata_is_valid(event):
        return None
    if (
        str(getattr(event, "kind", "") or "") != "mark"
        or getattr(event, "category", None) is not None
        or getattr(event, "scope_category", None) is not None
        or getattr(event, "category_profile", None) is not None
    ):
        return None

    name = str(getattr(event, "name", "") or "")
    data = getattr(event, "data", None)
    if name == SKILL_LIFECYCLE_MARK:
        metric_name = SKILL_LIFECYCLE_METRIC
        expected_fields = {"action", "provenance"}
    elif name == SKILL_LOAD_MARK:
        metric_name = SKILL_LOAD_METRIC
        expected_fields = {
            "post_patch_state",
            "provenance",
            "reuse_state",
            "use_count_bucket",
        }
    else:
        return None
    if not isinstance(data, dict) or set(data) != expected_fields:
        return None
    dimensions = {field: data.get(field) for field in sorted(expected_fields)}
    if not counter_dimensions_are_valid(metric_name, dimensions):
        return None
    return metric_name, dimensions


def skill_lifecycle_fields(kwargs: dict[str, Any]) -> dict[str, str] | None:
    """Build bounded fields for one successful non-load skill transition."""
    action = str(kwargs.get("action") or "").strip().lower()
    if action not in SKILL_LIFECYCLE_ACTIONS:
        return None
    return {
        "action": action,
        "provenance": skill_provenance(kwargs.get("provenance")),
    }


def skill_load_fields(kwargs: dict[str, Any]) -> dict[str, str] | None:
    """Build bounded skill-use fields without exporting local skill identity."""
    use_count = kwargs.get("use_count")
    reused = kwargs.get("reused")
    reuse_after_patch = kwargs.get("reuse_after_patch")
    if (
        isinstance(use_count, bool)
        or not isinstance(use_count, int)
        or use_count < 1
        or not isinstance(reused, bool)
        or not isinstance(reuse_after_patch, bool)
        or (reuse_after_patch and not reused)
    ):
        return None
    return {
        "post_patch_state": (
            "not_applicable"
            if not reused
            else "reused_after_patch"
            if reuse_after_patch
            else "no_new_patch"
        ),
        "provenance": skill_provenance(kwargs.get("provenance")),
        "reuse_state": "reused" if reused else "first_use",
        "use_count_bucket": count_bucket(use_count),
    }


def skill_provenance(value: Any) -> str:
    """Normalize producer provenance to the closed shared-metrics taxonomy."""
    normalized = str(value or "").strip().lower()
    return normalized if normalized in SKILL_PROVENANCES else "unknown"


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


def tool_category(kwargs: dict[str, Any]) -> str:
    """Map Hermes registry toolset metadata to a low-cardinality category."""
    toolset = str(kwargs.get("toolset") or "").strip().lower()
    if not toolset:
        return "unknown"
    if toolset in TOOL_CATEGORIES:
        return toolset
    if toolset.startswith("mcp"):
        return "mcp"
    if toolset.startswith("browser"):
        return "browser"
    if toolset.startswith(("image", "tts", "video", "vision")):
        return "media"
    if toolset.startswith("homeassistant"):
        return "home_automation"
    if toolset in {"clarify", "kanban", "todo"}:
        return "planning"
    if toolset == "session_search":
        return "memory"
    if toolset == "cronjob":
        return "scheduler"
    if toolset == "skills":
        return "skill"
    if toolset == "x_search":
        return "web"
    if toolset.startswith(
        ("discord", "email", "feishu", "hermes-yuanbao", "slack", "sms")
    ):
        return "communication"
    return "other"


def tool_outcome(kwargs: dict[str, Any]) -> str:
    """Normalize the terminal Hermes tool status without inspecting its result."""
    status = str(kwargs.get("status") or "").strip().lower()
    return {
        "blocked": "blocked",
        "cancelled": "cancelled",
        "error": "failed",
        "failed": "failed",
        "ok": "success",
        "success": "success",
        "timed_out": "timed_out",
        "timeout": "timed_out",
    }.get(status, "unknown")


def tool_approval_outcome(kwargs: dict[str, Any]) -> str:
    """Normalize a terminal approval choice to a bounded outcome."""
    choice = str(kwargs.get("choice") or "").strip().lower()
    if choice in {"always", "approve", "approved", "once", "session", "smart_approve"}:
        return "approved"
    if choice in {"deny", "denied", "smart_deny"}:
        return "denied"
    if choice in {"timed_out", "timeout"}:
        return "timed_out"
    return "unknown"


def tool_terminal_fields(
    kwargs: dict[str, Any],
    *,
    category: str | None = None,
    approval_outcome: str = "not_required",
    fallback_duration_ms: int | None = None,
) -> dict[str, str]:
    """Build one bounded tool-call terminal payload."""
    return {
        "approval_outcome": (
            approval_outcome
            if approval_outcome in TOOL_APPROVAL_OUTCOMES
            else "unknown"
        ),
        "latency_bucket": tool_latency_bucket(
            kwargs.get("duration_ms"),
            fallback_duration_ms=fallback_duration_ms,
        ),
        "outcome": tool_outcome(kwargs),
        "retry_count_bucket": tool_retry_bucket(kwargs.get("retry_count")),
        "tool_category": (
            category if category in TOOL_CATEGORIES else tool_category(kwargs)
        ),
    }


def tool_latency_bucket(
    value: Any,
    *,
    fallback_duration_ms: int | None = None,
) -> str:
    """Bucket a tool duration reported in milliseconds."""
    duration_ms = _non_negative_number(value)
    if duration_ms is None:
        duration_ms = _non_negative_number(fallback_duration_ms)
    if duration_ms is None:
        return "unknown"
    if duration_ms < 100:
        return "lt_100ms"
    if duration_ms < 250:
        return "100ms_to_250ms"
    if duration_ms < 500:
        return "250ms_to_500ms"
    if duration_ms < 1_000:
        return "500ms_to_1s"
    if duration_ms < 2_000:
        return "1s_to_2s"
    if duration_ms < 5_000:
        return "2s_to_5s"
    if duration_ms < 10_000:
        return "5s_to_10s"
    if duration_ms < 30_000:
        return "10s_to_30s"
    return "gte_30s"


def tool_retry_bucket(value: Any) -> str:
    """Bucket only explicit tool retries; missing relationships stay unknown."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return "unknown"
    return count_bucket(value)


def _non_negative_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return number if isfinite(number) and number >= 0 else None


def model_call_fields(kwargs: dict[str, Any]) -> dict[str, str]:
    """Return the terminal model identity and provider route known to Hermes."""
    model = _metric_identifier(
        kwargs.get("response_model"),
        max_length=MODEL_IDENTIFIER_MAX_LENGTH,
    )
    if model == "unknown":
        model = _metric_identifier(
            kwargs.get("model"),
            max_length=MODEL_IDENTIFIER_MAX_LENGTH,
        )
    return {
        "model": model,
        "provider": _metric_identifier(
            kwargs.get("provider"),
            max_length=PROVIDER_IDENTIFIER_MAX_LENGTH,
        ),
    }


def _metric_identifier(value: Any, *, max_length: int) -> str:
    """Normalize one structurally safe identifier without a product catalog."""
    if not isinstance(value, str):
        return "unknown"
    identifier = value.strip().lower()
    if (
        not identifier
        or len(identifier) > max_length
        or identifier[0] not in _METRIC_IDENTIFIER_START_CHARACTERS
        or any(
            character not in _METRIC_IDENTIFIER_CHARACTERS
            for character in identifier
        )
    ):
        return "unknown"
    return identifier

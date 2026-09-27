"""Bounded product contract for the first Hermes shared-metrics slice."""

from __future__ import annotations

from math import isfinite
from typing import Any

from agent.error_classifier import FailoverReason
from agent.relay_runtime import (
    LOGICAL_LLM_SCOPE,
    RUNTIME_INSTANCE_KEY,
    RUNTIME_SCHEMA_KEY,
    RUNTIME_SCHEMA_VERSION,
)
from hermes_cli.platforms import PLATFORMS
from toolsets import BUILTIN_TOOL_NAMES

SCHEMA_KEY = "hermes.metrics.schema_version"
SCHEMA_VERSION = "hermes.metrics.event.v3"
MODEL_CALL_SCOPE = "hermes.model_call"
MODEL_CALL_PROFILE_MODEL = "unknown"
TASK_SCOPE = "hermes.task_run"
TOOL_CALL_SCOPE = "hermes.tool_call"
CLIENT_ACTIVE_MARK = "hermes.client.active"
TOOL_APPROVAL_MARK = "hermes.tool_approval"
SKILL_LIFECYCLE_MARK = "hermes.skill.lifecycle"
SKILL_LOAD_MARK = "hermes.skill.load"
INSTALL_SNAPSHOT_MARK = "hermes.install.snapshot"
SESSION_MARK = "hermes.session"
SETUP_COMPLETED_MARK = "hermes.setup.completed"
MODEL_TOKENS_MARK = "hermes.model_tokens"
COMPRESSION_MARK = "hermes.compression"
MODEL_SWITCH_MARK = "hermes.model_switch"
FALLBACK_MARK = "hermes.fallback"
SLASH_COMMAND_MARK = "hermes.slash_command"
EXTENSION_INSTALL_MARK = "hermes.extension.install"
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
TOOL_USAGE_METRIC = "hermes.tool.usage.count"
INSTALL_SNAPSHOT_METRIC = "hermes.install.snapshot"
SESSION_METRIC = "hermes.session.count"
MILESTONE_METRIC = "hermes.install.milestone"
SETUP_COMPLETED_METRIC = "hermes.setup.completed"
MODEL_TOKENS_METRIC = "hermes.model_tokens.sum"
COMPRESSION_METRIC = "hermes.compression.count"
MODEL_SWITCH_METRIC = "hermes.model_switch.count"
FALLBACK_METRIC = "hermes.fallback.count"
SLASH_COMMAND_METRIC = "hermes.slash_command.count"
EXTENSION_INSTALL_METRIC = "hermes.extension.install.count"
MODEL_IDENTIFIER_MAX_LENGTH = 256
PROVIDER_IDENTIFIER_MAX_LENGTH = 64
_METRIC_IDENTIFIER_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789._:/@+-")
_METRIC_IDENTIFIER_START_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789")

EXECUTION_SURFACES = frozenset({
    "acp", "api", "batch", "cli", "desktop", "gateway", "python", "scheduled_task", "tui",
    "other", "unknown",
})
TASK_OUTCOMES = frozenset({"cancelled", "failed", "success", "timed_out", "unknown"})
TASK_END_REASONS = frozenset({
    "approval_denied", "completed", "failed", "guardrail_blocked", "iteration_limit",
    "system_aborted", "timed_out", "unknown", "user_cancelled",
})
TASK_TERMINATIONS = frozenset({"none", "system_aborted", "timed_out", "unknown", "user_cancelled"})
TASK_ENTRYPOINTS = frozenset({
    "api", "background", "batch", "delegated", "gateway_message", "interactive", "other", "python",
    "scheduled_task", "unknown",
})
DURATION_BUCKETS = frozenset({
    "1s_to_5s", "2m_to_10m", "30s_to_2m", "5s_to_30s", "gte_10m", "lt_1s",
})
COUNT_BUCKETS = frozenset({"0", "1", "2", "3_to_5", "6_to_10", "gte_11"})
TOOL_CATEGORIES = frozenset({
    "browser", "code_execution", "communication", "computer_use", "delegation", "file",
    "home_automation", "mcp", "media", "memory", "other", "planning", "project", "scheduler",
    "skill", "terminal", "unknown", "web",
})
TOOL_OUTCOMES = frozenset({"blocked", "cancelled", "failed", "success", "timed_out", "unknown"})
TOOL_APPROVAL_OUTCOMES = frozenset({"approved", "cancelled", "denied", "not_required", "timed_out", "unknown"})
TOOL_APPROVAL_ATTRIBUTIONS = frozenset({"tool_call", "unattributed"})
TOOL_LATENCY_BUCKETS = frozenset({
    "100ms_to_250ms", "10s_to_30s", "1s_to_2s", "250ms_to_500ms", "2s_to_5s", "500ms_to_1s",
    "5s_to_10s", "gte_30s", "lt_100ms", "unknown",
})
TOOL_RETRY_BUCKETS = COUNT_BUCKETS | frozenset({"unknown"})
SKILL_LIFECYCLE_ACTIONS = frozenset({
    "archived", "created", "edited", "installed", "patched", "restored", "stale",
})
SKILL_PROVENANCES = frozenset({"agent_created", "external", "installed", "local", "unknown"})
SKILL_REUSE_STATES = frozenset({"first_use", "reused"})
SKILL_POST_PATCH_STATES = frozenset({"no_new_patch", "not_applicable", "reused_after_patch"})
CLIENT_OS_FAMILIES = frozenset({"linux", "macos", "unknown", "windows"})
CLIENT_ARCHITECTURES = frozenset({"arm", "arm64", "unknown", "x86", "x86_64"})
CLIENT_INSTALL_METHODS = frozenset({
    "apt", "docker", "git", "home-manager", "homebrew", "nixos", "pip", "unknown",
})
CLIENT_RESOURCE_KEYS = frozenset({"architecture", "hermes_version", "install_method", "os_family"})

# ---- v3 taxonomies -------------------------------------------------------------------------
MODEL_CALL_ROLES = frozenset({"auxiliary", "primary"})
MODEL_OUTCOMES = frozenset({"cancelled", "failed", "success"})
# The classifier's own closed enum, so the exported vocabulary is exactly what recovery acts on.
MODEL_ERROR_CLASSES = frozenset(reason.value for reason in FailoverReason) | {"none"}
TASK_FAILURE_CLASSES = MODEL_ERROR_CLASSES | frozenset({
    "context_compression", "empty_response", "exception", "local_error", "other", "persistence",
    "repeated_errors", "restart_limit", "session_busy", "shutdown",
})
# Surfaces that are not messaging platforms keep their own execution_surface value.
GATEWAY_PLATFORMS = (frozenset(PLATFORMS) - {"api_server", "cli", "cron"}) | {"none", "plugin"}
TOOL_NAMES = BUILTIN_TOOL_NAMES | {"mcp", "plugin", "unknown"}
TOOL_ERROR_CLASSES = frozenset({
    "blocked", "contract_violation", "exception", "interrupted", "invalid_arguments", "none",
    "timeout", "tool_error", "unknown",
})
SIZE_BUCKETS = frozenset({
    "0", "1", "2", "3_to_5", "6_to_10", "11_to_25", "26_to_100", "101_to_250", "gte_251",
})


def _bundled_memory_providers() -> frozenset[str]:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "plugins" / "memory"
    try:
        return frozenset(p.name for p in root.iterdir() if (p / "__init__.py").is_file())
    except OSError:
        return frozenset()


MEMORY_PROVIDERS = _bundled_memory_providers() | {"builtin", "plugin"}


class _CatalogValues:
    """A closed enum backed by a public catalog (see shared_metrics_catalog), loaded on first use."""

    def __init__(self, *loaders: str, extra: frozenset[str]) -> None:
        self.loaders, self.extra = loaders, extra

    def values(self) -> frozenset[str]:
        from . import shared_metrics_catalog as catalog

        return self.extra.union(*(catalog._safe(getattr(catalog, name)) for name in self.loaders))

    def __contains__(self, value: object) -> bool:
        return value in self.extra or value in self.values()


# ---- decision-data taxonomies ----------------------------------------------------------------
SESSION_DURATION_BUCKETS = frozenset({"lt_1m", "1m_to_5m", "5m_to_30m", "30m_to_2h", "2h_to_8h", "gte_8h"})
INSTALL_AGE_BUCKETS = frozenset({
    "lt_1h", "1h_to_1d", "1d_to_7d", "7d_to_30d", "30d_to_90d", "gte_90d", "unknown",
})
MILESTONES = frozenset({
    "setup_completed", "first_task_started", "first_task_success", "first_tool_success",
    "first_gateway_message", "first_scheduled_task", "first_delegation", "first_skill_created",
    "first_skill_reused", "first_mcp_tool_success", "first_long_session",
})
SETUP_SURFACES = frozenset({"cli", "desktop", "other"})
TOKEN_TYPES = frozenset({"input", "output", "cache_read", "cache_write", "reasoning"})
TTFT_BUCKETS = frozenset({
    "lt_500ms", "500ms_to_1s", "1s_to_2s", "2s_to_5s", "5s_to_15s", "gte_15s", "not_streamed", "unknown",
})
COMPRESSION_TRIGGERS = frozenset({"auto", "manual", "other", "overflow"})
COMPRESSION_OUTCOMES = frozenset({"failed", "skipped", "success"})
CONTEXT_FILL_BUCKETS = frozenset({"lt_50", "50_to_75", "75_to_90", "90_to_100", "gte_100", "unknown"})
EXTENSION_KINDS = frozenset({"mcp_server", "plugin", "skill"})
EXTENSION_SOURCES = frozenset({"bundled", "catalog", "hub", "local", "other", "url"})
EXTENSION_OUTCOMES = frozenset({"failed", "success"})
TERMINAL_BACKENDS = frozenset({"daytona", "docker", "local", "modal", "other", "singularity", "ssh"})
AUX_TASKS = _CatalogValues("aux_task_names", extra=frozenset({"none", "other"}))
SLASH_COMMANDS = _CatalogValues("slash_command_names", extra=frozenset({"plugin", "skill", "unknown"}))
SKILL_NAMES = _CatalogValues("bundled_skill_names", extra=frozenset({"custom"}))
EXTENSION_NAMES = _CatalogValues(
    "bundled_skill_names", "mcp_catalog_names", "plugin_catalog_names", extra=frozenset({"custom"})
)
DISPLAY_LANGUAGES = _CatalogValues("display_languages", extra=frozenset({"other"}))

_ARCHITECTURE_ALIASES = {
    "amd64": "x86_64", "x64": "x86_64", "x86_64": "x86_64",
    "aarch64": "arm64", "arm64": "arm64",
    "i386": "x86", "i486": "x86", "i586": "x86", "i686": "x86", "x86": "x86",
}
_OS_FAMILIES = {"darwin": "macos", "linux": "linux", "macos": "macos", "windows": "windows"}


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _allowlisted(normalized: str, allowed: frozenset[str]) -> str:
    return normalized if normalized in allowed else "unknown"


def client_os_family(value: Any) -> str:
    """Map a platform system name to the shared-metrics OS taxonomy."""
    return _OS_FAMILIES.get(_norm(value), "unknown")


def client_architecture(value: Any) -> str:
    """Map a machine architecture to the shared-metrics taxonomy."""
    normalized = _norm(value).replace("-", "_")
    if normalized in _ARCHITECTURE_ALIASES:
        return _ARCHITECTURE_ALIASES[normalized]
    return "arm" if normalized.startswith("armv") else "unknown"


def client_install_method(value: Any) -> str:
    """Return an allowlisted Hermes installation method."""
    normalized = _norm(value)
    return _allowlisted("nixos" if normalized == "nix" else normalized, CLIENT_INSTALL_METHODS)


def client_resource(
    hermes_version: Any, *, os_name: Any, architecture: Any, install_method: Any
) -> dict[str, str]:
    """Build the bounded client resource attached to aggregate packages."""
    version = str(hermes_version or "").strip()
    return {
        "architecture": client_architecture(architecture),
        "hermes_version": version if 0 < len(version) <= 64 else "unknown",
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


_LEGACY_PROVIDER_FAMILIES = frozenset({"aggregator", "custom", "direct", "local", "unknown"})
_LEGACY_MODEL_LOCALITIES = frozenset({"local", "remote", "unknown"})
_LEGACY_MODEL_OUTCOMES = frozenset({"cancelled", "failed", "success"})
_LEGACY_MODEL_FAMILIES = frozenset({
    "claude", "deepseek", "gemini", "gemma", "glm", "gpt", "grok", "kimi", "llama", "minimax",
    "mimo", "mistral", "nemotron", "nova", "o1", "o3", "o4", "qwen", "step", "trinity",
    "unknown",
})

_COUNTER_DIMENSION_VALUES: dict[str, dict[str, frozenset[str]]] = {
    CLIENT_ACTIVE_METRIC: {},
    # Retained only so pre-v2 pending rows remain packageable.
    LEGACY_MODEL_CALL_METRIC: {
        "call_role": frozenset({"primary"}), "locality": _LEGACY_MODEL_LOCALITIES,
        "model_family": _LEGACY_MODEL_FAMILIES, "outcome": _LEGACY_MODEL_OUTCOMES,
        "provider_family": _LEGACY_PROVIDER_FAMILIES,
    },
    TASK_STARTED_METRIC: {
        "entrypoint": TASK_ENTRYPOINTS, "execution_surface": EXECUTION_SURFACES,
        "platform": GATEWAY_PLATFORMS,
    },
    TASK_FINISHED_METRIC: {
        "duration_bucket": DURATION_BUCKETS, "end_reason": TASK_END_REASONS,
        "entrypoint": TASK_ENTRYPOINTS, "execution_surface": EXECUTION_SURFACES,
        "failure_class": TASK_FAILURE_CLASSES,
        "model_call_count_bucket": COUNT_BUCKETS, "outcome": TASK_OUTCOMES,
        "platform": GATEWAY_PLATFORMS,
        "retry_count_bucket": COUNT_BUCKETS, "termination": TASK_TERMINATIONS,
        "tool_call_count_bucket": COUNT_BUCKETS,
    },
    MODEL_ROUTE_METRIC: {
        "call_role": MODEL_CALL_ROLES, "error_class": MODEL_ERROR_CLASSES, "outcome": MODEL_OUTCOMES,
        "ttft_bucket": TTFT_BUCKETS,
    },
    TOOL_CALL_METRIC: {
        "approval_outcome": TOOL_APPROVAL_OUTCOMES, "latency_bucket": TOOL_LATENCY_BUCKETS,
        "outcome": TOOL_OUTCOMES, "retry_count_bucket": TOOL_RETRY_BUCKETS,
        "tool_category": TOOL_CATEGORIES,
    },
    TOOL_USAGE_METRIC: {
        "error_class": TOOL_ERROR_CLASSES, "outcome": TOOL_OUTCOMES, "tool_name": TOOL_NAMES,
    },
    TOOL_APPROVAL_METRIC: {
        "attribution": TOOL_APPROVAL_ATTRIBUTIONS,
        "outcome": TOOL_APPROVAL_OUTCOMES - {"not_required"},
    },
    SKILL_LIFECYCLE_METRIC: {"action": SKILL_LIFECYCLE_ACTIONS, "provenance": SKILL_PROVENANCES},
    SKILL_LOAD_METRIC: {
        "post_patch_state": SKILL_POST_PATCH_STATES, "provenance": SKILL_PROVENANCES,
        "reuse_state": SKILL_REUSE_STATES, "skill_name": SKILL_NAMES, "use_count_bucket": COUNT_BUCKETS,
    },
    INSTALL_SNAPSHOT_METRIC: {
        "cron_job_count_bucket": SIZE_BUCKETS, "display_language": DISPLAY_LANGUAGES,
        "install_age_bucket": INSTALL_AGE_BUCKETS, "mcp_server_count_bucket": SIZE_BUCKETS,
        "memory_provider": MEMORY_PROVIDERS, "messaging_platform_count_bucket": SIZE_BUCKETS,
        "plugin_count_bucket": SIZE_BUCKETS, "profile_count_bucket": SIZE_BUCKETS,
        "skill_count_bucket": SIZE_BUCKETS, "terminal_backend": TERMINAL_BACKENDS,
    },
    SESSION_METRIC: {
        "active_duration_bucket": SESSION_DURATION_BUCKETS, "entrypoint": TASK_ENTRYPOINTS,
        "execution_surface": EXECUTION_SURFACES, "failed_turn_count_bucket": COUNT_BUCKETS,
        "last_outcome": TASK_OUTCOMES, "platform": GATEWAY_PLATFORMS, "turn_count_bucket": SIZE_BUCKETS,
    },
    MILESTONE_METRIC: {"install_age_bucket": INSTALL_AGE_BUCKETS, "milestone": MILESTONES},
    SETUP_COMPLETED_METRIC: {"surface": SETUP_SURFACES},
    MODEL_TOKENS_METRIC: {"aux_task": AUX_TASKS, "call_role": MODEL_CALL_ROLES, "token_type": TOKEN_TYPES},
    COMPRESSION_METRIC: {
        "context_fill_bucket": CONTEXT_FILL_BUCKETS, "outcome": COMPRESSION_OUTCOMES,
        "trigger": COMPRESSION_TRIGGERS,
    },
    MODEL_SWITCH_METRIC: {"execution_surface": EXECUTION_SURFACES},
    FALLBACK_METRIC: {"error_class": MODEL_ERROR_CLASSES},
    SLASH_COMMAND_METRIC: {"command": SLASH_COMMANDS, "execution_surface": EXECUTION_SURFACES},
    EXTENSION_INSTALL_METRIC: {
        "kind": EXTENSION_KINDS, "name": EXTENSION_NAMES, "outcome": EXTENSION_OUTCOMES,
        "source": EXTENSION_SOURCES,
    },
}
_MODEL_ROUTE_MAX_LENGTHS = {
    "model": MODEL_IDENTIFIER_MAX_LENGTH, "provider": PROVIDER_IDENTIFIER_MAX_LENGTH,
}
# metric -> {field: max length} for provider/model identifiers, validated by shape (no catalog).
_IDENTIFIER_FIELDS: dict[str, dict[str, int]] = {
    MODEL_ROUTE_METRIC: _MODEL_ROUTE_MAX_LENGTHS,
    MODEL_TOKENS_METRIC: _MODEL_ROUTE_MAX_LENGTHS,
    SETUP_COMPLETED_METRIC: {"provider": PROVIDER_IDENTIFIER_MAX_LENGTH},
    MODEL_SWITCH_METRIC: dict.fromkeys(("from_provider", "to_provider"), PROVIDER_IDENTIFIER_MAX_LENGTH),
    FALLBACK_METRIC: dict.fromkeys(("from_provider", "to_provider"), PROVIDER_IDENTIFIER_MAX_LENGTH),
    INSTALL_SNAPSHOT_METRIC: {"main_provider": PROVIDER_IDENTIFIER_MAX_LENGTH},
}
# metric -> closed dimension field set
_METRIC_FIELDS: dict[str, frozenset[str]] = {
    name: frozenset(contract) | frozenset(_IDENTIFIER_FIELDS.get(name, ()))
    for name, contract in _COUNTER_DIMENSION_VALUES.items()
}
# Older field sets still accepted at packaging so counters recorded before an upgrade drain.
_LEGACY_METRIC_FIELDS: dict[str, tuple[frozenset[str], ...]] = {
    MODEL_ROUTE_METRIC: (
        frozenset(_MODEL_ROUTE_MAX_LENGTHS), _METRIC_FIELDS[MODEL_ROUTE_METRIC] - {"ttft_bucket"},
    ),
    TASK_STARTED_METRIC: (_METRIC_FIELDS[TASK_STARTED_METRIC] - {"platform"},),
    TASK_FINISHED_METRIC: (_METRIC_FIELDS[TASK_FINISHED_METRIC] - {"failure_class", "platform"},),
    SKILL_LOAD_METRIC: (_METRIC_FIELDS[SKILL_LOAD_METRIC] - {"skill_name"},),
    INSTALL_SNAPSHOT_METRIC: (_METRIC_FIELDS[INSTALL_SNAPSHOT_METRIC] - {
        "display_language", "install_age_bucket", "main_provider", "messaging_platform_count_bucket",
        "terminal_backend",
    },),
}
COUNTER_METRICS = frozenset(_METRIC_FIELDS) - {LEGACY_MODEL_CALL_METRIC}
# Counters whose value is a summed quantity rather than an event count.
SUM_METRICS = frozenset({MODEL_TOKENS_METRIC})
_SKILL_MARK_METRICS = {
    SKILL_LIFECYCLE_MARK: SKILL_LIFECYCLE_METRIC, SKILL_LOAD_MARK: SKILL_LOAD_METRIC,
}
# Marks projected one-to-one onto a counter of the same contract.
_DECISION_MARK_METRICS = {
    SESSION_MARK: SESSION_METRIC, SETUP_COMPLETED_MARK: SETUP_COMPLETED_METRIC,
    COMPRESSION_MARK: COMPRESSION_METRIC, MODEL_SWITCH_MARK: MODEL_SWITCH_METRIC,
    FALLBACK_MARK: FALLBACK_METRIC, SLASH_COMMAND_MARK: SLASH_COMMAND_METRIC,
    EXTENSION_INSTALL_MARK: EXTENSION_INSTALL_METRIC,
}


def counter_dimensions_are_valid(metric_name: str, dimensions: dict[str, Any]) -> bool:
    """Return whether dimensions match one closed shared-metric contract (current or legacy)."""
    fields = frozenset(dimensions)
    if fields != _METRIC_FIELDS.get(metric_name) and fields not in _LEGACY_METRIC_FIELDS.get(metric_name, ()):
        return False
    identifiers = _IDENTIFIER_FIELDS.get(metric_name, {})
    contract = _COUNTER_DIMENSION_VALUES[metric_name]
    return all(
        isinstance(value := dimensions[field], str) and (
            value == _metric_identifier(value, max_length=identifiers[field]) if field in identifiers
            else value in contract[field]
        )
        for field in fields
    )


def _relay_metadata(
    event: Any, schema_key: str, schema_version: str, *extra_keys: str
) -> dict | None:
    """Return the event metadata when it carries only the allowlisted Relay keys."""
    metadata = getattr(event, "metadata", None)
    if not isinstance(metadata, dict) or metadata.get(schema_key) != schema_version:
        return None
    allowed = {schema_key, RUNTIME_INSTANCE_KEY, "otel.status_code", *extra_keys}
    if set(metadata) - allowed or metadata.get("otel.status_code", "OK") not in {"OK", "ERROR"}:
        return None
    return metadata


def _event_text(event: Any, attr: str) -> str:
    return str(getattr(event, attr, "") or "")


def _event_shape_matches(event: Any, **expected: Any) -> bool:
    """Match the coarse Relay event shape (``kind`` plus any of name/category/scope_category/
    category_profile).

    A ``str`` expectation compares against the stringified attribute; ``None`` requires the
    attribute itself to be ``None``; anything else (the ``category_profile`` dict) compares
    with plain equality. Unmentioned attributes are not checked.
    """
    for attr, value in expected.items():
        actual = _event_text(event, attr) if isinstance(value, str) else getattr(event, attr, None)
        if actual != value:
            return False
    return True


def _bounded_dimensions(metric_name: str, data: Any) -> dict[str, str] | None:
    """Project ``data`` onto the metric's closed field set, or None when it does not fit."""
    expected_fields = _METRIC_FIELDS[metric_name]
    if not isinstance(data, dict) or set(data) != expected_fields:
        return None
    dimensions = {field: data.get(field) for field in sorted(expected_fields)}
    return dimensions if counter_dimensions_are_valid(metric_name, dimensions) else None


def _valid_shape(event: Any, **shape: Any) -> bool:
    """Metadata allowlist check plus :func:`_event_shape_matches` in one step."""
    return (
        _relay_metadata(event, SCHEMA_KEY, SCHEMA_VERSION) is not None
        and _event_shape_matches(event, **shape)
    )


def _bounded_counter(metric_name: str | None, event: Any) -> tuple[str, dict[str, str]] | None:
    if metric_name is None:
        return None
    dimensions = _bounded_dimensions(metric_name, getattr(event, "data", None))
    return None if dimensions is None else (metric_name, dimensions)


def _scoped_dimensions(event: Any, metric_name: str, **shape: Any) -> dict[str, str] | None:
    """Bounded ``event.data`` for a scope *end* event of the given shape, else None."""
    if not _valid_shape(event, kind="scope", scope_category="end", **shape):
        return None
    return _bounded_dimensions(metric_name, getattr(event, "data", None))


_MARK_SHAPE = dict(kind="mark", category=None, scope_category=None, category_profile=None)


def _mark_counter(event: Any, metrics_by_mark: dict[str, str]) -> tuple[str, dict[str, str]] | None:
    """Return the bounded counter for a safe Relay mark whose name is in *metrics_by_mark*."""
    if not _valid_shape(event, **_MARK_SHAPE):
        return None
    return _bounded_counter(metrics_by_mark.get(_event_text(event, "name")), event)


def client_active_counter(event: Any) -> tuple[str, dict[str, str]] | None:
    """Return the active-install counter for one empty allowlisted mark."""
    return _mark_counter(event, {CLIENT_ACTIVE_MARK: CLIENT_ACTIVE_METRIC})


def model_call_dimensions(event: Any) -> dict[str, str] | None:
    """Return package dimensions for one valid logical model-call end event."""
    auxiliary = _auxiliary_model_call_dimensions(event)
    if auxiliary is not None:
        return auxiliary
    # The synthetic scope can span provider fallback. The accepted terminal
    # route is carried in the validated payload rather than this start profile.
    return _scoped_dimensions(
        event, MODEL_ROUTE_METRIC, category="llm", name=MODEL_CALL_SCOPE,
        category_profile={"model_name": MODEL_CALL_PROFILE_MODEL},
    )


def _auxiliary_model_call_dimensions(event: Any) -> dict[str, str] | None:
    """Project a terminal auxiliary route from its Hermes logical scope."""
    metadata = _relay_metadata(
        event, RUNTIME_SCHEMA_KEY, RUNTIME_SCHEMA_VERSION, "hermes.call_role"
    )
    call_role = (metadata or {}).get("hermes.call_role")
    data = getattr(event, "data", None)
    if (
        not isinstance(call_role, str)
        or not call_role.startswith("auxiliary:")
        or not _event_shape_matches(
            event, kind="scope", category="function", name=LOGICAL_LLM_SCOPE,
            scope_category="end", category_profile=None,
        )
        or not isinstance(data, dict)
        or set(data) - {"response_model"} != {"model", "outcome", "provider"}
        or data.get("outcome") not in _LEGACY_MODEL_OUTCOMES
    ):
        return None
    outcome = data["outcome"]
    dimensions = model_route_fields(
        data, call_role="auxiliary", outcome=outcome,
        error_class="none" if outcome == "success" else "unknown",
    )
    return dimensions if counter_dimensions_are_valid(MODEL_ROUTE_METRIC, dimensions) else None


def task_counter(event: Any) -> tuple[str, dict[str, str]] | None:
    """Return one validated task counter from a task scope event."""
    if not _valid_shape(
        event, kind="scope", category="function", name=TASK_SCOPE, category_profile=None
    ):
        return None
    phases = {"start": TASK_STARTED_METRIC, "end": TASK_FINISHED_METRIC}
    return _bounded_counter(phases.get(_event_text(event, "scope_category")), event)


def tool_call_dimensions(event: Any) -> dict[str, str] | None:
    """Return package dimensions for one allowlisted tool lifecycle end event."""
    return _tool_end_projection(event, TOOL_CALL_METRIC)


def tool_usage_dimensions(event: Any) -> dict[str, str] | None:
    """Return the per-tool usage dimensions carried by the same tool lifecycle end event."""
    return _tool_end_projection(event, TOOL_USAGE_METRIC)


_TOOL_END_FIELDS = _METRIC_FIELDS[TOOL_CALL_METRIC] | _METRIC_FIELDS[TOOL_USAGE_METRIC]


def _tool_end_projection(event: Any, metric_name: str) -> dict[str, str] | None:
    """Project the tool end event onto one counter; both projections must validate first."""
    if not _valid_shape(
        event, kind="scope", scope_category="end", category="tool", name=TOOL_CALL_SCOPE,
        category_profile={},
    ):
        return None
    data = getattr(event, "data", None)
    if not isinstance(data, dict) or set(data) != _TOOL_END_FIELDS:
        return None
    projections = {
        name: _bounded_dimensions(name, {f: data[f] for f in _METRIC_FIELDS[name]})
        for name in (TOOL_CALL_METRIC, TOOL_USAGE_METRIC)
    }
    return projections[metric_name] if all(projections.values()) else None


def install_snapshot_counter(event: Any) -> tuple[str, dict[str, str]] | None:
    """Return the daily install-configuration snapshot from a safe mark."""
    return _mark_counter(event, {INSTALL_SNAPSHOT_MARK: INSTALL_SNAPSHOT_METRIC})


def tool_approval_counter(event: Any) -> tuple[str, dict[str, str]] | None:
    """Return one validated approval counter from a safe Relay mark event."""
    return _mark_counter(event, {TOOL_APPROVAL_MARK: TOOL_APPROVAL_METRIC})


def skill_counter(event: Any) -> tuple[str, dict[str, str]] | None:
    """Return one validated skill lifecycle or load counter from a safe mark."""
    return _mark_counter(event, _SKILL_MARK_METRICS)


def decision_counter(event: Any) -> tuple[str, dict[str, str]] | None:
    """Return the counter for one session/setup/compression/switch/fallback/command/install mark."""
    return _mark_counter(event, _DECISION_MARK_METRICS)


_TOKEN_MARK_DIMENSIONS = frozenset({"aux_task", "call_role", "model", "provider"})


def model_token_counters(event: Any) -> list[tuple[str, dict[str, str], int]]:
    """Expand one token-usage mark into ``(metric, dimensions, amount)`` sums, one per token type."""
    if not _valid_shape(event, **_MARK_SHAPE) or _event_text(event, "name") != MODEL_TOKENS_MARK:
        return []
    data = getattr(event, "data", None)
    if not isinstance(data, dict) or set(data) != _TOKEN_MARK_DIMENSIONS | TOKEN_TYPES:
        return []
    base = {field: data[field] for field in _TOKEN_MARK_DIMENSIONS}
    counters = []
    for token_type in sorted(TOKEN_TYPES):
        amount = data[token_type]
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            return []
        dimensions = {**base, "token_type": token_type}
        if not counter_dimensions_are_valid(MODEL_TOKENS_METRIC, dimensions):
            return []
        if amount:
            counters.append((MODEL_TOKENS_METRIC, dimensions, amount))
    return counters


def skill_lifecycle_fields(kwargs: dict[str, Any]) -> dict[str, str] | None:
    """Build bounded fields for one successful non-load skill transition."""
    action = _norm(kwargs.get("action"))
    if action not in SKILL_LIFECYCLE_ACTIONS:
        return None
    return {"action": action, "provenance": skill_provenance(kwargs.get("provenance"))}


def skill_load_fields(kwargs: dict[str, Any]) -> dict[str, str] | None:
    """Build bounded skill-use fields without exporting local skill identity."""
    use_count, reused = kwargs.get("use_count"), kwargs.get("reused")
    reuse_after_patch = kwargs.get("reuse_after_patch")
    if (
        isinstance(use_count, bool) or not isinstance(use_count, int) or use_count < 1
        or not isinstance(reused, bool) or not isinstance(reuse_after_patch, bool)
        or (reuse_after_patch and not reused)
    ):
        return None
    return {
        "post_patch_state": (
            "not_applicable" if not reused
            else "reused_after_patch" if reuse_after_patch
            else "no_new_patch"
        ),
        "provenance": skill_provenance(kwargs.get("provenance")),
        "reuse_state": "reused" if reused else "first_use",
        "skill_name": _skill_metric_name(kwargs.get("skill_name")),
        "use_count_bucket": count_bucket(use_count),
    }


def _skill_metric_name(value: Any) -> str:
    from .shared_metrics_catalog import skill_metric_name

    return skill_metric_name(value)


def skill_provenance(value: Any) -> str:
    """Normalize producer provenance to the closed shared-metrics taxonomy."""
    return _allowlisted(_norm(value), SKILL_PROVENANCES)


_SURFACE_ALIASES = {
    "api_server": "api",
    **dict.fromkeys(("cron", "scheduler", "scheduled"), "scheduled_task"),
}
_KNOWN_GATEWAY_PLATFORMS = frozenset({"discord", "email", "slack", "telegram", "teams", "whatsapp"})


def execution_surface(kwargs: dict[str, Any]) -> str:
    """Normalize the safe session surface carried by the parent Relay scope."""
    value = _norm(kwargs.get("execution_surface") or kwargs.get("platform") or "unknown")
    if value in EXECUTION_SURFACES:
        return value
    if value in _SURFACE_ALIASES:
        return _SURFACE_ALIASES[value]
    try:
        from hermes_cli.platforms import get_all_platforms

        if value in get_all_platforms():
            return "gateway"
    except Exception:
        pass
    return "gateway" if value in _KNOWN_GATEWAY_PLATFORMS else "other"


def task_start_fields(kwargs: dict[str, Any]) -> dict[str, str]:
    """Build the bounded fields recorded on a task scope start event."""
    surface = execution_surface(kwargs)
    return {
        "entrypoint": task_entrypoint(kwargs, surface), "execution_surface": surface,
        "platform": gateway_platform(kwargs, surface),
    }


def gateway_platform(kwargs: dict[str, Any], surface: str | None = None) -> str:
    """The built-in messaging platform for a gateway task; plugin platforms stay anonymous."""
    if (surface or execution_surface(kwargs)) != "gateway":
        return "none"
    value = _norm(kwargs.get("platform"))
    return value if value in GATEWAY_PLATFORMS else "plugin"


_SURFACE_ENTRYPOINTS = {
    # An ACP session is a human in an editor (VS Code / Zed / JetBrains), same
    # dispatch shape as the other interactive surfaces.
    **dict.fromkeys(("acp", "cli", "desktop", "tui"), "interactive"),
    **{s: s for s in ("api", "batch", "python", "scheduled_task", "unknown")},
    "gateway": "gateway_message",
}


def task_entrypoint(kwargs: dict[str, Any], surface: str | None = None) -> str:
    """Normalize the task dispatch owner without exporting source strings."""
    declared = _norm(kwargs.get("entrypoint"))
    if declared in TASK_ENTRYPOINTS:
        return declared
    if kwargs.get("parent_task_id") or kwargs.get("parent_session_id"):
        return "delegated"
    return _SURFACE_ENTRYPOINTS.get(surface or execution_surface(kwargs), "other")


def task_terminal_fields(
    kwargs: dict[str, Any], *, duration_ms: int, model_call_count: int, tool_call_count: int,
    retry_count: int,
) -> dict[str, str]:
    """Build the bounded terminal payload for one task scope."""
    outcome, end_reason, termination = task_terminal_state(kwargs)
    return {
        **task_start_fields(kwargs),
        "duration_bucket": duration_bucket(duration_ms),
        "end_reason": end_reason,
        "failure_class": task_failure_class(kwargs, outcome),
        "model_call_count_bucket": count_bucket(model_call_count),
        "outcome": outcome,
        "retry_count_bucket": count_bucket(retry_count),
        "termination": termination,
        "tool_call_count_bucket": count_bucket(tool_call_count),
    }


_LOCAL_FAILURE_CLASSES = {"interpreter_shutdown": "shutdown", "session_busy": "session_busy"}
# turn_exit_reason prefix -> class, for failures that never reached the provider classifier.
_EXIT_REASON_FAILURE_CLASSES = (
    ("empty_response", "empty_response"), ("all_retries_exhausted", "empty_response"),
    ("context_compression", "context_compression"), ("compaction_", "context_compression"),
    ("ollama_runtime_context", "context_compression"),
    ("local_processing_error", "local_error"), ("repeated_outer_errors", "repeated_errors"),
    ("error_near_max_iterations", "repeated_errors"), ("session_persistence", "persistence"),
    ("redirect_restart_limit", "restart_limit"), ("rebuilt_restart_limit", "restart_limit"),
)


def task_failure_class(kwargs: dict[str, Any], outcome: str | None = None) -> str:
    """Why a failed task failed, as a closed class; ``none`` for any non-failed outcome."""
    if (outcome or task_terminal_state(kwargs)[0]) != "failed":
        return "none"
    declared = _norm(kwargs.get("failure_class"))
    if declared in TASK_FAILURE_CLASSES - {"none"}:
        return declared
    failure_reason = _norm(kwargs.get("failure_reason"))
    if failure_reason in MODEL_ERROR_CLASSES - {"none"}:
        return failure_reason
    if failure_reason in _LOCAL_FAILURE_CLASSES:
        return _LOCAL_FAILURE_CLASSES[failure_reason]
    reason = _norm(kwargs.get("turn_exit_reason"))
    return next(
        (label for prefix, label in _EXIT_REASON_FAILURE_CLASSES if reason.startswith(prefix)),
        "other",
    )


def task_terminal_state(kwargs: dict[str, Any]) -> tuple[str, str, str]:
    """Map Hermes terminal state to bounded (outcome, end_reason, termination)."""
    reason = _norm(kwargs.get("turn_exit_reason"))
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


# (exclusive upper bound, label) — ascending; the trailing label catches the rest.
_DURATION_THRESHOLDS = (
    (1_000, "lt_1s"), (5_000, "1s_to_5s"), (30_000, "5s_to_30s"),
    (120_000, "30s_to_2m"), (600_000, "2m_to_10m"),
)
_COUNT_THRESHOLDS = ((1, "0"), (2, "1"), (3, "2"), (6, "3_to_5"), (11, "6_to_10"))
_LATENCY_THRESHOLDS = (
    (100, "lt_100ms"), (250, "100ms_to_250ms"), (500, "250ms_to_500ms"), (1_000, "500ms_to_1s"),
    (2_000, "1s_to_2s"), (5_000, "2s_to_5s"), (10_000, "5s_to_10s"), (30_000, "10s_to_30s"),
)


def _bucket(value: float, thresholds: tuple[tuple[float, str], ...], last: str) -> str:
    return next((label for upper, label in thresholds if value < upper), last)


def duration_bucket(duration_ms: int) -> str:
    """Bucket a non-negative task duration into a fixed low-cardinality range."""
    return _bucket(max(0, int(duration_ms)), _DURATION_THRESHOLDS, "gte_10m")


def count_bucket(count: int) -> str:
    """Bucket a non-negative per-task count into a fixed range."""
    return _bucket(max(0, int(count)), _COUNT_THRESHOLDS, "gte_11")


_TOOL_CATEGORY_EXACT = {
    **{category: category for category in TOOL_CATEGORIES},
    "clarify": "planning", "kanban": "planning", "todo": "planning", "session_search": "memory",
    "cronjob": "scheduler", "skills": "skill", "x_search": "web",
}
_TOOL_CATEGORY_PREFIXES = (
    ("mcp", "mcp"),
    ("browser", "browser"),
    (("image", "tts", "video", "vision"), "media"),
    ("homeassistant", "home_automation"),
    (("discord", "email", "feishu", "hermes-yuanbao", "slack", "sms"), "communication"),
)


def tool_category(kwargs: dict[str, Any]) -> str:
    """Map Hermes registry toolset metadata to a low-cardinality category."""
    toolset = _norm(kwargs.get("toolset"))
    if not toolset:
        return "unknown"
    if toolset in _TOOL_CATEGORY_EXACT:
        return _TOOL_CATEGORY_EXACT[toolset]
    for prefixes, category in _TOOL_CATEGORY_PREFIXES:
        if toolset.startswith(prefixes):
            return category
    return "other"


_TOOL_STATUS_OUTCOMES = {
    **{s: s for s in ("blocked", "cancelled", "failed", "success", "timed_out")},
    "error": "failed", "ok": "success", "timeout": "timed_out",
}


def tool_outcome(kwargs: dict[str, Any]) -> str:
    """Normalize the terminal Hermes tool status without inspecting its result."""
    return _TOOL_STATUS_OUTCOMES.get(_norm(kwargs.get("status")), "unknown")


_APPROVAL_CHOICES = {
    **dict.fromkeys(
        ("always", "approve", "approved", "once", "session", "smart_approve"), "approved"
    ),
    **dict.fromkeys(("deny", "denied", "smart_deny"), "denied"),
    **dict.fromkeys(("timed_out", "timeout"), "timed_out"),
    "cancelled": "cancelled",  # prompt withdrawn / undeliverable / unanswered — not a user decision
}


def tool_approval_outcome(kwargs: dict[str, Any]) -> str:
    """Normalize a terminal approval choice to a bounded outcome."""
    return _APPROVAL_CHOICES.get(_norm(kwargs.get("choice")), "unknown")


def tool_terminal_fields(
    kwargs: dict[str, Any], *, category: str | None = None, approval_outcome: str = "not_required",
    fallback_duration_ms: int | None = None, tool_name: str | None = None,
) -> dict[str, str]:
    """Build one bounded tool-call terminal payload (feeds the category and per-tool counters)."""
    outcome = tool_outcome(kwargs)
    return {
        "approval_outcome": _allowlisted(approval_outcome, TOOL_APPROVAL_OUTCOMES),
        "error_class": tool_error_class(kwargs, outcome),
        "latency_bucket": tool_latency_bucket(
            kwargs.get("duration_ms"), fallback_duration_ms=fallback_duration_ms
        ),
        "outcome": outcome,
        "retry_count_bucket": tool_retry_bucket(kwargs.get("retry_count")),
        "tool_category": category if category in TOOL_CATEGORIES else tool_category(kwargs),
        "tool_name": tool_name if tool_name is not None and tool_name in TOOL_NAMES
        else tool_metric_name(kwargs),
    }


def tool_metric_name(kwargs: dict[str, Any]) -> str:
    """A built-in tool's own name; MCP and plugin tools collapse to their source kind."""
    name = _norm(kwargs.get("tool_name"))
    if name in BUILTIN_TOOL_NAMES:
        return name
    if not name:
        return "unknown"
    return "mcp" if name.startswith("mcp_") or tool_category(kwargs) == "mcp" else "plugin"


_TOOL_STATUS_ERROR_CLASSES = {
    "blocked": "blocked", "cancelled": "interrupted", "success": "none", "timed_out": "timeout",
}
_TOOL_ERROR_TYPES = {
    **dict.fromkeys(("keyboard_interrupt", "tool_interrupted", "user_interrupt"), "interrupted"),
    "invalid_tool_arguments": "invalid_arguments", "thread_missing_result": "exception",
    "tool_error": "tool_error", "tool_result_contract": "contract_violation",
    "tool_timeout": "timeout",
}


def tool_error_class(kwargs: dict[str, Any], outcome: str | None = None) -> str:
    """Closed failure class from Hermes's own error_type; exception class names become
    ``exception`` so plugin-defined identifiers never leave the machine."""
    outcome = outcome or tool_outcome(kwargs)
    if outcome in _TOOL_STATUS_ERROR_CLASSES:
        return _TOOL_STATUS_ERROR_CLASSES[outcome]
    if outcome != "failed":
        return "unknown"
    error_type = _norm(kwargs.get("error_type"))
    return _TOOL_ERROR_TYPES.get(error_type, "exception" if error_type else "unknown")


def tool_latency_bucket(value: Any, *, fallback_duration_ms: int | None = None) -> str:
    """Bucket a tool duration reported in milliseconds."""
    duration_ms = _non_negative_number(value)
    if duration_ms is None:
        duration_ms = _non_negative_number(fallback_duration_ms)
    if duration_ms is None:
        return "unknown"
    return _bucket(duration_ms, _LATENCY_THRESHOLDS, "gte_30s")


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
    from .shared_metrics_catalog import model_metric_name, provider_metric_name

    provider = provider_metric_name(kwargs.get("provider"))
    model = model_metric_name(kwargs.get("response_model"), provider, max_length=MODEL_IDENTIFIER_MAX_LENGTH)
    if model == "unknown":
        model = model_metric_name(kwargs.get("model"), provider, max_length=MODEL_IDENTIFIER_MAX_LENGTH)
    return {"model": model, "provider": provider}


def model_route_fields(
    kwargs: dict[str, Any], *, call_role: str, outcome: str, error_class: str,
    ttft_bucket: str = "unknown",
) -> dict[str, str]:
    """The terminal route plus how the logical call ended. ``error_class`` is the last
    classified attempt error, so ``success`` + ``rate_limit`` reads as "recovered from a 429"."""
    return {
        **model_call_fields(kwargs),
        "call_role": _allowlisted(call_role, MODEL_CALL_ROLES),
        "error_class": error_class if error_class in MODEL_ERROR_CLASSES else "unknown",
        "outcome": outcome if outcome in MODEL_OUTCOMES else "failed",
        "ttft_bucket": ttft_bucket if ttft_bucket in TTFT_BUCKETS else "unknown",
    }


def model_error_class(kwargs: dict[str, Any]) -> str:
    """The classifier's FailoverReason for one failed provider attempt."""
    reason = _norm(kwargs.get("reason"))
    return reason if reason in MODEL_ERROR_CLASSES - {"none"} else "unknown"


# (exclusive upper bound, label) for install-scale counts (skills routinely exceed 100).
_SIZE_THRESHOLDS = (
    (1, "0"), (2, "1"), (3, "2"), (6, "3_to_5"), (11, "6_to_10"), (26, "11_to_25"),
    (101, "26_to_100"), (251, "101_to_250"),
)


def size_bucket(count: int) -> str:
    """Bucket a non-negative install-scale count."""
    return _bucket(max(0, int(count)), _SIZE_THRESHOLDS, "gte_251")


def install_snapshot_fields(
    *, memory_provider: Any, mcp_servers: int, plugins: int, skills: int, cron_jobs: int,
    profiles: int, messaging_platforms: int, install_age_bucket: str, main_provider: Any,
    terminal_backend: Any, display_language: Any,
) -> dict[str, str]:
    """Bounded daily configuration snapshot: counts, closed enums and public names only."""
    from .shared_metrics_catalog import display_language_metric_name, provider_metric_name

    provider = _norm(memory_provider)
    backend = _norm(terminal_backend) or "local"
    return {
        "cron_job_count_bucket": size_bucket(cron_jobs),
        "display_language": display_language_metric_name(display_language),
        "install_age_bucket": install_age_bucket if install_age_bucket in INSTALL_AGE_BUCKETS else "unknown",
        "main_provider": provider_metric_name(main_provider) if main_provider else "none",
        "mcp_server_count_bucket": size_bucket(mcp_servers),
        "memory_provider": "builtin" if not provider
        else provider if provider in MEMORY_PROVIDERS else "plugin",
        "messaging_platform_count_bucket": size_bucket(messaging_platforms),
        "plugin_count_bucket": size_bucket(plugins),
        "profile_count_bucket": size_bucket(profiles),
        "skill_count_bucket": size_bucket(skills),
        "terminal_backend": backend if backend in TERMINAL_BACKENDS else "other",
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
        or not _METRIC_IDENTIFIER_CHARACTERS.issuperset(identifier)
    ):
        return "unknown"
    return identifier

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
_METRIC_IDENTIFIER_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789._:/@+-")
_METRIC_IDENTIFIER_START_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789")

EXECUTION_SURFACES = frozenset({
    "api", "batch", "cli", "desktop", "gateway", "python", "scheduled_task", "tui", "other",
    "unknown",
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
TOOL_APPROVAL_OUTCOMES = frozenset({"approved", "denied", "not_required", "timed_out", "unknown"})
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
    TASK_STARTED_METRIC: {"entrypoint": TASK_ENTRYPOINTS, "execution_surface": EXECUTION_SURFACES},
    TASK_FINISHED_METRIC: {
        "duration_bucket": DURATION_BUCKETS, "end_reason": TASK_END_REASONS,
        "entrypoint": TASK_ENTRYPOINTS, "execution_surface": EXECUTION_SURFACES,
        "model_call_count_bucket": COUNT_BUCKETS, "outcome": TASK_OUTCOMES,
        "retry_count_bucket": COUNT_BUCKETS, "termination": TASK_TERMINATIONS,
        "tool_call_count_bucket": COUNT_BUCKETS,
    },
    TOOL_CALL_METRIC: {
        "approval_outcome": TOOL_APPROVAL_OUTCOMES, "latency_bucket": TOOL_LATENCY_BUCKETS,
        "outcome": TOOL_OUTCOMES, "retry_count_bucket": TOOL_RETRY_BUCKETS,
        "tool_category": TOOL_CATEGORIES,
    },
    TOOL_APPROVAL_METRIC: {
        "attribution": TOOL_APPROVAL_ATTRIBUTIONS,
        "outcome": TOOL_APPROVAL_OUTCOMES - {"not_required"},
    },
    SKILL_LIFECYCLE_METRIC: {"action": SKILL_LIFECYCLE_ACTIONS, "provenance": SKILL_PROVENANCES},
    SKILL_LOAD_METRIC: {
        "post_patch_state": SKILL_POST_PATCH_STATES, "provenance": SKILL_PROVENANCES,
        "reuse_state": SKILL_REUSE_STATES, "use_count_bucket": COUNT_BUCKETS,
    },
}
_MODEL_ROUTE_MAX_LENGTHS = {
    "model": MODEL_IDENTIFIER_MAX_LENGTH, "provider": PROVIDER_IDENTIFIER_MAX_LENGTH,
}
# metric -> closed dimension field set (model-route fields are validated by shape, not allowlist)
_METRIC_FIELDS: dict[str, frozenset[str]] = {
    **{name: frozenset(contract) for name, contract in _COUNTER_DIMENSION_VALUES.items()},
    MODEL_ROUTE_METRIC: frozenset(_MODEL_ROUTE_MAX_LENGTHS),
}
COUNTER_METRICS = frozenset(_METRIC_FIELDS) - {LEGACY_MODEL_CALL_METRIC}
_SKILL_MARK_METRICS = {
    SKILL_LIFECYCLE_MARK: SKILL_LIFECYCLE_METRIC, SKILL_LOAD_MARK: SKILL_LOAD_METRIC,
}


def counter_dimensions_are_valid(metric_name: str, dimensions: dict[str, Any]) -> bool:
    """Return whether dimensions match one closed shared-metric contract."""
    if metric_name == MODEL_ROUTE_METRIC:
        return set(dimensions) == _METRIC_FIELDS[metric_name] and all(
            dimensions[field] == _metric_identifier(dimensions[field], max_length=max_length)
            for field, max_length in _MODEL_ROUTE_MAX_LENGTHS.items()
        )
    contract = _COUNTER_DIMENSION_VALUES.get(metric_name)
    if contract is None or set(dimensions) != set(contract):
        return False
    return all(
        isinstance(dimensions[field], str) and dimensions[field] in allowed_values
        for field, allowed_values in contract.items()
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
    dimensions = model_call_fields(data)
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
    return _scoped_dimensions(
        event, TOOL_CALL_METRIC, category="tool", name=TOOL_CALL_SCOPE, category_profile={}
    )


def tool_approval_counter(event: Any) -> tuple[str, dict[str, str]] | None:
    """Return one validated approval counter from a safe Relay mark event."""
    return _mark_counter(event, {TOOL_APPROVAL_MARK: TOOL_APPROVAL_METRIC})


def skill_counter(event: Any) -> tuple[str, dict[str, str]] | None:
    """Return one validated skill lifecycle or load counter from a safe mark."""
    return _mark_counter(event, _SKILL_MARK_METRICS)


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
        "use_count_bucket": count_bucket(use_count),
    }


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
    return {"entrypoint": task_entrypoint(kwargs, surface), "execution_surface": surface}


_SURFACE_ENTRYPOINTS = {
    **dict.fromkeys(("cli", "desktop", "tui"), "interactive"),
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
        "model_call_count_bucket": count_bucket(model_call_count),
        "outcome": outcome,
        "retry_count_bucket": count_bucket(retry_count),
        "termination": termination,
        "tool_call_count_bucket": count_bucket(tool_call_count),
    }


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
}


def tool_approval_outcome(kwargs: dict[str, Any]) -> str:
    """Normalize a terminal approval choice to a bounded outcome."""
    return _APPROVAL_CHOICES.get(_norm(kwargs.get("choice")), "unknown")


def tool_terminal_fields(
    kwargs: dict[str, Any], *, category: str | None = None, approval_outcome: str = "not_required",
    fallback_duration_ms: int | None = None,
) -> dict[str, str]:
    """Build one bounded tool-call terminal payload."""
    return {
        "approval_outcome": _allowlisted(approval_outcome, TOOL_APPROVAL_OUTCOMES),
        "latency_bucket": tool_latency_bucket(
            kwargs.get("duration_ms"), fallback_duration_ms=fallback_duration_ms
        ),
        "outcome": tool_outcome(kwargs),
        "retry_count_bucket": tool_retry_bucket(kwargs.get("retry_count")),
        "tool_category": category if category in TOOL_CATEGORIES else tool_category(kwargs),
    }


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
    model = _metric_identifier(kwargs.get("response_model"), max_length=MODEL_IDENTIFIER_MAX_LENGTH)
    if model == "unknown":
        model = _metric_identifier(kwargs.get("model"), max_length=MODEL_IDENTIFIER_MAX_LENGTH)
    provider = _metric_identifier(kwargs.get("provider"), max_length=PROVIDER_IDENTIFIER_MAX_LENGTH)
    return {"model": model, "provider": provider}


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

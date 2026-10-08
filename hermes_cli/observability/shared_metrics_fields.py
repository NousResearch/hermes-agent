"""Bounded field builders for the decision-data shared metrics.

Sessions, install milestones, setup completion, token volumes, compression, model switches,
fallbacks, slash commands and extension installs. Every builder takes RAW runtime values and
returns only closed-enum, bucketed or public-catalog dimensions (see shared_metrics_contract).
"""

from __future__ import annotations

from typing import Any

from . import shared_metrics_catalog as catalog
from . import shared_metrics_contract as contract
from .shared_metrics_contract import (
    MODEL_IDENTIFIER_MAX_LENGTH, PROVIDER_IDENTIFIER_MAX_LENGTH, _bucket, _metric_identifier, _norm,
)
from .shared_metrics_signals import user_created_skill

_MINUTE_MS = 60_000
_SESSION_DURATION_THRESHOLDS = (
    (_MINUTE_MS, "lt_1m"), (5 * _MINUTE_MS, "1m_to_5m"), (30 * _MINUTE_MS, "5m_to_30m"),
    (120 * _MINUTE_MS, "30m_to_2h"), (480 * _MINUTE_MS, "2h_to_8h"),
)
_HOUR_S = 3_600
_INSTALL_AGE_THRESHOLDS = (
    (_HOUR_S, "lt_1h"), (24 * _HOUR_S, "1h_to_1d"), (7 * 24 * _HOUR_S, "1d_to_7d"),
    (30 * 24 * _HOUR_S, "7d_to_30d"), (90 * 24 * _HOUR_S, "30d_to_90d"),
)
_TTFT_THRESHOLDS = (
    (0.5, "lt_500ms"), (1.0, "500ms_to_1s"), (2.0, "1s_to_2s"), (5.0, "2s_to_5s"), (15.0, "5s_to_15s"),
)
_CONTEXT_FILL_THRESHOLDS = ((50, "lt_50"), (75, "50_to_75"), (90, "75_to_90"), (100, "90_to_100"))
# A session this long is the "real use" signal (vs a one-shot try-out).
LONG_SESSION_TURNS = 6


def _number(value: Any) -> float | None:
    return contract._non_negative_number(value)


def provider_identifier(value: Any) -> str:
    return catalog.provider_metric_name(value)


def session_fields(
    start_fields: dict[str, str], *, turns: int, failed_turns: int, last_outcome: str, active_ms: int,
) -> dict[str, str]:
    """One closed session: where it ran, how many turns, how long it was active, how it ended."""
    return {
        "active_duration_bucket": _bucket(max(0, active_ms), _SESSION_DURATION_THRESHOLDS, "gte_8h"),
        "entrypoint": start_fields.get("entrypoint", "unknown"),
        "execution_surface": start_fields.get("execution_surface", "unknown"),
        "failed_turn_count_bucket": contract.count_bucket(failed_turns),
        "last_outcome": last_outcome if last_outcome in contract.TASK_OUTCOMES else "unknown",
        "platform": start_fields.get("platform", "none"),
        "turn_count_bucket": contract.size_bucket(turns),
    }


def install_age_bucket(age_seconds: Any) -> str:
    age = _number(age_seconds)
    return "unknown" if age is None else _bucket(age, _INSTALL_AGE_THRESHOLDS, "gte_90d")


def ttft_bucket(event: dict[str, Any]) -> str:
    """Time to first streamed chunk for one primary call; ``not_streamed`` when nothing streamed."""
    started, first = _number(event.get("started_at")), _number(event.get("first_chunk_at"))
    if started is None:
        return "unknown"
    if first is None:
        return "not_streamed"
    return _bucket(max(0.0, first - started), _TTFT_THRESHOLDS, "gte_15s")


def context_fill_bucket(tokens_before: Any, context_length: Any) -> str:
    tokens, length = _number(tokens_before), _number(context_length)
    if tokens is None or not length:
        return "unknown"
    return _bucket(100 * tokens / length, _CONTEXT_FILL_THRESHOLDS, "gte_100")


_USAGE_KEYS = {
    "input": "input_tokens", "output": "output_tokens", "cache_read": "cache_read_tokens",
    "cache_write": "cache_write_tokens", "reasoning": "reasoning_tokens",
}


def model_token_fields(
    usage: Any, *, model: Any, provider: Any, call_role: str, aux_task: Any = None,
) -> dict[str, Any] | None:
    """Token-usage mark data (dimensions + integer amounts); None when there is no usage."""
    if not isinstance(usage, dict):
        return None
    amounts = {}
    for token_type, key in _USAGE_KEYS.items():
        value = usage.get(key)
        amounts[token_type] = value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0
    if not any(amounts.values()):
        return None
    return {
        "aux_task": catalog.aux_task_metric_name(aux_task) if call_role == "auxiliary" else "none",
        "call_role": call_role,
        "model": catalog.model_metric_name(
            model, provider_identifier(provider), max_length=MODEL_IDENTIFIER_MAX_LENGTH,
        ),
        "provider": provider_identifier(provider),
        **amounts,
    }


def setup_completed_fields(*, surface: Any, provider: Any) -> dict[str, str]:
    value = _norm(surface)
    return {
        "provider": provider_identifier(provider) if provider else "none",
        "surface": value if value in contract.SETUP_SURFACES else "other",
    }


_COMPRESSION_TRIGGERS = {
    **dict.fromkeys(("auto", "threshold", "preflight", "auto_threshold"), "auto"),
    # The attempt log's precise automatic triggers stay one coarse bucket here.
    **dict.fromkeys(
        ("idle", "turn_start_threshold", "engine_preflight", "pre_api", "post_tool", "gateway_hygiene"), "auto",
    ),
    **dict.fromkeys(("overflow", "context_overflow", "overflow_error", "error"), "overflow"),
    **dict.fromkeys(("manual", "user", "command", "slash"), "manual"),
}
_COMPRESSION_OUTCOMES = {
    **dict.fromkeys(("success", "ok", "compressed", "completed"), "success"),
    **dict.fromkeys(("skipped", "noop", "not_needed"), "skipped"),
}


# Attempt-log classes the closed set names under an existing value: a summary model benched after repeated
# fallbacks is a pre-LLM skip to the deterministic summary, which is what ``feasibility_skip`` counts.
_COMPRESSION_FAILURE_CLASS_ALIASES = {"summary_model_benched": "feasibility_skip"}


def compression_failure_class(outcome: str, failure_class: Any) -> str:
    """The attempt log's class, collapsed onto the closed set: ``exception:<Type>`` / ``rollback:<Type>``
    keep only the prefix (the type name may be a plugin's). A success reads ``none`` unless it committed
    through a fallback (``feasibility_skip``, ``summary_generation_failed``, ``aux_model_fallback``, ...),
    which keeps that class so a fallback commit never reads as a clean one."""
    value = _norm(failure_class if isinstance(failure_class, str) else None).split(":", 1)[0]
    value = _COMPRESSION_FAILURE_CLASS_ALIASES.get(value, value)
    if outcome == "success" and value in {"", "none"}:
        return "none"
    if not value:
        return "unknown"
    return value if value in contract.COMPRESSION_FAILURE_CLASSES - {"none", "unknown"} else "other"


def compression_fields(
    *, trigger: Any, outcome: Any, tokens_before: Any, context_length: Any, failure_class: Any = None,
) -> dict[str, str]:
    outcome_value = _COMPRESSION_OUTCOMES.get(_norm(outcome), "failed")
    return {
        "context_fill_bucket": context_fill_bucket(tokens_before, context_length),
        "failure_class": compression_failure_class(outcome_value, failure_class),
        "outcome": outcome_value,
        "trigger": _COMPRESSION_TRIGGERS.get(_norm(trigger), "other"),
    }


def model_switch_fields(*, from_provider: Any, to_provider: Any, surface: Any) -> dict[str, str]:
    return {
        "execution_surface": contract.execution_surface({"execution_surface": surface}),
        "from_provider": provider_identifier(from_provider),
        "to_provider": provider_identifier(to_provider),
    }


def fallback_fields(*, from_provider: Any, to_provider: Any, reason: Any) -> dict[str, str]:
    value = getattr(reason, "value", reason)
    return {
        "error_class": contract.model_error_class({"reason": value}),
        "from_provider": provider_identifier(from_provider),
        "to_provider": provider_identifier(to_provider),
    }


def slash_command_fields(*, command: Any, surface: Any) -> dict[str, str]:
    return {
        "command": catalog.slash_command_metric_name(command),
        "execution_surface": contract.execution_surface({"execution_surface": surface}),
    }


def extension_install_fields(
    *, kind: Any, source: Any, name: Any, outcome: Any,
    # ---- iuf c1 ----
    failure_class: Any = None, registry: Any = None, error: Any = None,
    # ---- end iuf c1 ----
) -> dict[str, str] | None:
    kind_value = _norm(kind)
    if kind_value not in contract.EXTENSION_KINDS:
        return None
    source_value = _norm(source)
    outcome_value = "success" if _norm(outcome) in {"success", "ok", "installed"} else "failed"
    return {
        "kind": kind_value,
        "name": catalog.extension_metric_name(kind_value, name),
        "outcome": outcome_value,
        "source": source_value if source_value in contract.EXTENSION_SOURCES else "other",
        # ---- iuf c1 ----
        "failure_class": extension_failure_class(kind_value, outcome_value, failure_class, error),
        "registry": extension_registry(kind_value, registry),
        # ---- end iuf c1 ----
    }


# ---- iuf c1 ----
# Exceptions from these top-level packages are transport failures, whatever their class name.
_NETWORK_MODULES = frozenset({"aiohttp", "h11", "h2", "httpcore", "httpx", "requests", "ssl", "urllib3", "websockets"})


def tagged_failure_class(error: object) -> str | None:
    """The ``failure_class`` a raise site set, read statically: an attribute hook on a third-party
    exception must not raise from a classifier and replace the user's error."""
    import inspect

    tagged = inspect.getattr_static(error, "failure_class", None)
    return tagged if isinstance(tagged, str) and tagged else None


def exception_failure_class(error: BaseException) -> str:
    """Why a raised install failed, from the exception alone: the class its raise site tagged
    (``failure_class`` attribute), else its TYPE. Never reads the message, and never runs the
    exception's own attribute hooks (a raising ``__getattr__`` or property reads by type)."""
    tagged = tagged_failure_class(error)
    if tagged:
        return tagged
    import socket
    import urllib.error

    if isinstance(error, PermissionError):
        return "permission"
    if isinstance(error, (ConnectionError, TimeoutError, socket.gaierror, urllib.error.URLError)) or any(
            (getattr(cls, "__module__", "") or "").split(".", 1)[0] in _NETWORK_MODULES for cls in type(error).__mro__):
        return "network"
    if isinstance(error, OSError):
        return "filesystem_error"
    return f"exception:{type(error).__name__}"


def extension_failure_class(kind: str, outcome: str, failure_class: Any, error: Any = None) -> str:
    """``none`` on success; else the caller's class (or the raised error's), collapsed onto the kind's
    closed set: ``exception:<Type>`` keeps only its prefix, anything unlisted reads ``other``."""
    if outcome == "success":
        return "none"
    value = failure_class if isinstance(failure_class, str) and failure_class else None
    if value is None and isinstance(error, BaseException):
        value = exception_failure_class(error)
    value = _norm(value).split(":", 1)[0]
    return value if value in contract.EXTENSION_KIND_FAILURE_CLASSES[kind] - {"none"} else "other"


def extension_registry(kind: str, registry: Any) -> str:
    """The skills-hub adapter id for a skill row; plugin and MCP rows are ``none``."""
    if kind != "skill":
        return "none"
    # The hub's own alias table (`--source skills.sh`); already imported by the install it records.
    from tools.skills_hub_install import _SOURCE_ID_ALIASES

    value = _norm(registry) or "none"
    value = _SOURCE_ID_ALIASES.get(value, value)
    return value if value in contract.EXTENSION_REGISTRIES else "other"
# ---- end iuf c1 ----


def _long_session(d: dict[str, str]) -> bool:
    return d.get("turn_count_bucket") not in {"0", "1", "2", "3_to_5"}


# metric -> [(milestone, predicate over its dimensions)]
_MILESTONE_RULES = {
    contract.TASK_STARTED_METRIC: (
        ("first_task_started", lambda d: True),
        ("first_gateway_message", lambda d: d.get("platform") not in {None, "none"}),
        ("first_scheduled_task", lambda d: d.get("entrypoint") == "scheduled_task"),
    ),
    contract.TASK_FINISHED_METRIC: (("first_task_success", lambda d: d.get("outcome") == "success"),),
    contract.TOOL_USAGE_METRIC: (
        ("first_tool_success", lambda d: d.get("outcome") == "success"),
        ("first_mcp_tool_success", lambda d: d.get("outcome") == "success" and d.get("tool_name") == "mcp"),
        ("first_delegation", lambda d: d.get("outcome") == "success" and d.get("tool_name") == "delegate_task"),
    ),
    contract.SKILL_LIFECYCLE_METRIC: (("first_skill_created", user_created_skill),),
    contract.SKILL_LOAD_METRIC: (("first_skill_reused", lambda d: d.get("reuse_state") == "reused"),),
    contract.SESSION_METRIC: (("first_long_session", _long_session),),
    contract.SETUP_COMPLETED_METRIC: (("setup_completed", lambda d: True),),
}


def milestones_for(metric_name: str, dimensions: dict[str, str]) -> tuple[str, ...]:
    """Install milestones a recorded counter reaches (each is latched once per install by the store)."""
    return tuple(name for name, reached in _MILESTONE_RULES.get(metric_name, ()) if reached(dimensions))

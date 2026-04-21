"""Shared turn-scoped runtime override merge helpers.

Centralizes precedence rules for per-skill runtime defaults so the CLI,
gateway, and cron call sites do not fork their own merge logic.  The
functions here are pure — no I/O, no config reads — so they can be called
from any execution path at the same logical position: after primary /
runtime_kwargs are assembled, before ``resolve_turn_route()`` runs.

Precedence (highest → lowest): explicit lock (``/model``, cron ``job.model``)
> skill-declared defaults (after trust-cap enforcement) > session default.
See ``website/docs/developer-guide/creating-skills.md`` for the end-user
contract.
"""

from __future__ import annotations

import fnmatch
import logging
from typing import Any, Dict, List, NamedTuple, Optional

logger = logging.getLogger(__name__)


# ── Reasoning-effort ordering ────────────────────────────────────────────
# Used for the v1 clamp (skills may only reduce effort, never raise it).
_REASONING_EFFORT_RANK: Dict[str, int] = {
    "none": 0,
    "minimal": 1,
    "low": 2,
    "medium": 3,
    "high": 4,
    "xhigh": 5,
}


def _reasoning_rank(reasoning_config: Optional[Dict[str, Any]]) -> Optional[int]:
    """Return the ordinal rank of a reasoning_config dict, or None if unknown.

    Accepts the shape returned by ``hermes_constants.parse_reasoning_effort``:
    ``{"enabled": False}`` or ``{"enabled": True, "effort": <level>}``.
    """
    if not isinstance(reasoning_config, dict):
        return None
    if reasoning_config.get("enabled") is False:
        return _REASONING_EFFORT_RANK["none"]
    effort = reasoning_config.get("effort")
    if not isinstance(effort, str):
        return None
    return _REASONING_EFFORT_RANK.get(effort.lower())


# ── Result types ──────────────────────────────────────────────────────────


class LogEvent(NamedTuple):
    """Observability signal emitted during merging.

    ``name`` follows the stable ``skill_runtime_defaults.*`` prefix; callers
    emit it via their platform's logging/metrics channel.
    """

    name: str
    level: str  # "DEBUG" | "INFO" | "WARNING" | "ERROR"
    fields: Dict[str, Any]


class HardFailReason(NamedTuple):
    """Populated when a ``required`` field could not be applied."""

    skill_name: str
    failing_field: str
    reason: str


class MergeResult(NamedTuple):
    """Output of :func:`merge_turn_runtime_defaults`."""

    merged: Dict[str, Any]
    events: List[LogEvent]
    hard_fail: Optional[HardFailReason]


# ── Core merge helper ─────────────────────────────────────────────────────


def merge_turn_runtime_defaults(
    primary: Dict[str, Any],
    skill_defaults: Dict[str, Any],
    *,
    skill_name: str = "",
    explicit_lock: Optional[Dict[str, Any]] = None,
    session_default_reasoning: Optional[Dict[str, Any]] = None,
    model_allowlist: Optional[List[str]] = None,
) -> MergeResult:
    """Merge skill runtime defaults onto a primary turn config.

    Precedence (highest → lowest):
        1. ``explicit_lock`` fields (e.g. ``/model`` session override, cron
           ``job.model``) — always win; emit ``overridden_by_explicit``.
        2. ``skill_defaults`` fields after trust-cap enforcement:
             - reasoning_effort clamped to ``session_default_reasoning`` (skills
               can only reduce effort, never raise it).
             - model allowed iff ``model_allowlist`` is empty (any allowed) or
               the model matches a glob pattern in the list.
        3. ``primary`` fields (the normal session/job default).

    ``skill_defaults["required"]`` lists field names that must be applied —
    if one is dropped by clamp, allowlist, or explicit-lock, the returned
    ``hard_fail`` is populated and the caller is expected to abort the turn
    with a user-visible error.

    Args:
        primary: Base turn config to merge onto. Not mutated.
        skill_defaults: Output of ``extract_skill_runtime_defaults``.
        skill_name: Used in log events and hard-fail reasons.  Values that
            fail ``agent.skill_utils.is_safe_skill_name`` are replaced with
            the empty string so structured logs stay safe even if a caller
            forgot to sanitize.
        explicit_lock: Fields the user or operator explicitly locked for this
            turn, e.g. ``{"model": "anthropic/claude-4.6"}``.  Always beats
            skill defaults.
        session_default_reasoning: The current session/job reasoning config,
            used as the clamp ceiling for ``reasoning_effort``.
        model_allowlist: Operator-configured list of glob patterns.  Empty or
            ``None`` means any model is allowed.

    Returns:
        :class:`MergeResult` whose ``merged`` dict is safe to pass to
        ``resolve_turn_route()``.
    """
    # Defensive sanitization: structured log events emit
    # ``skill_name_sanitized`` under the assumption the value is safe.
    # Callers are asked to pre-sanitize (see docstring), but we enforce
    # the contract here so a forgetful caller cannot corrupt observability.
    from agent.skill_utils import is_safe_skill_name

    if not is_safe_skill_name(skill_name):
        skill_name = ""

    events: List[LogEvent] = []
    merged: Dict[str, Any] = dict(primary or {})
    explicit = explicit_lock or {}
    required_fields: List[str] = list(skill_defaults.get("required") or [])
    applied_fields: List[str] = []
    dropped_required: Dict[str, str] = {}  # field -> reason

    # ── reasoning_effort ──────────────────────────────────────────────
    skill_reasoning_config = skill_defaults.get("reasoning_config")
    if skill_reasoning_config is not None:
        if "reasoning_effort" in explicit or "reasoning_config" in explicit:
            events.append(
                LogEvent(
                    "skill_runtime_defaults.overridden_by_explicit",
                    "INFO",
                    {
                        "skill_name_sanitized": skill_name,
                        "field": "reasoning_effort",
                    },
                )
            )
            if "reasoning_effort" in required_fields:
                dropped_required["reasoning_effort"] = "overridden_by_explicit"
        else:
            skill_rank = _reasoning_rank(skill_reasoning_config)
            session_rank = _reasoning_rank(session_default_reasoning)
            if (
                skill_rank is not None
                and session_rank is not None
                and skill_rank > session_rank
            ):
                # Clamp: skill tried to raise above session default.
                events.append(
                    LogEvent(
                        "skill_runtime_defaults.clamped_to_session_default",
                        "INFO",
                        {
                            "skill_name_sanitized": skill_name,
                            "attempted_value": skill_defaults.get(
                                "reasoning_effort"
                            ),
                            "session_default": (
                                session_default_reasoning or {}
                            ).get("effort")
                            if (session_default_reasoning or {}).get(
                                "enabled", True
                            )
                            else "none",
                        },
                    )
                )
                if "reasoning_effort" in required_fields:
                    dropped_required["reasoning_effort"] = (
                        "clamped_to_session_default"
                    )
            else:
                merged["reasoning_effort"] = skill_defaults.get("reasoning_effort")
                merged["reasoning_config"] = skill_reasoning_config
                applied_fields.append("reasoning_effort")

    # ── model ──────────────────────────────────────────────────────────
    skill_model = skill_defaults.get("model")
    if skill_model is not None:
        if "model" in explicit:
            events.append(
                LogEvent(
                    "skill_runtime_defaults.overridden_by_explicit",
                    "INFO",
                    {
                        "skill_name_sanitized": skill_name,
                        "field": "model",
                    },
                )
            )
            if "model" in required_fields:
                dropped_required["model"] = "overridden_by_explicit"
        elif not _model_allowlisted(skill_model, model_allowlist):
            events.append(
                LogEvent(
                    "skill_runtime_defaults.model_not_allowlisted",
                    "WARNING",
                    {
                        "skill_name_sanitized": skill_name,
                        "attempted_model": skill_model,
                        "allowlist_size": len(model_allowlist or []),
                    },
                )
            )
            if "model" in required_fields:
                dropped_required["model"] = "model_not_allowlisted"
        else:
            merged["model"] = skill_model
            merged["model_locked"] = True
            merged["routing_reason"] = "skill_fixed"
            applied_fields.append("model")

    # ── required enforcement ──────────────────────────────────────────
    hard_fail: Optional[HardFailReason] = None
    for field in required_fields:
        if field in applied_fields:
            continue
        reason = dropped_required.get(field, "not_applied")
        events.append(
            LogEvent(
                "skill_runtime_defaults.required_failed",
                "ERROR",
                {
                    "skill_name_sanitized": skill_name,
                    "failing_field": field,
                    "reason": reason,
                },
            )
        )
        # First failure wins — callers surface a single user-visible error.
        if hard_fail is None:
            hard_fail = HardFailReason(
                skill_name=skill_name,
                failing_field=field,
                reason=reason,
            )

    if applied_fields and not hard_fail:
        events.append(
            LogEvent(
                "skill_runtime_defaults.applied",
                "DEBUG",
                {
                    "skill_name_sanitized": skill_name,
                    "fields_applied": applied_fields,
                },
            )
        )

    return MergeResult(merged=merged, events=events, hard_fail=hard_fail)


def _model_allowlisted(
    model: str, allowlist: Optional[List[str]]
) -> bool:
    """Return True when the allowlist is empty or the model matches a pattern.

    Patterns use glob syntax (``*``, ``?``, ``[seq]``).
    """
    if not allowlist:
        return True
    return any(fnmatch.fnmatchcase(model, pattern) for pattern in allowlist)


# ── Multi-skill merge (cron) ──────────────────────────────────────────────


def merge_multi_skill_runtime_defaults(
    defaults_list: List[Dict[str, Any]],
) -> tuple[Dict[str, Any], List[str]]:
    """Merge runtime defaults from multiple skills attached to one job.

    Iterates the list in order.  Identical values collapse silently.  A
    conflicting value drops the field entirely and surfaces a warning string
    suitable for prepending to the job prompt as
    ``[SYSTEM: Conflicting runtime defaults dropped: ...]``.

    Args:
        defaults_list: list of skill runtime_defaults dicts (the output of
            ``extract_skill_runtime_defaults``).

    Returns:
        ``(merged_defaults, conflict_warnings)``.
    """
    merged: Dict[str, Any] = {}
    conflicts: Dict[str, set] = {}
    # All required fields union across skills.
    required: List[str] = []

    for defaults in defaults_list:
        if not isinstance(defaults, dict):
            continue
        for key in ("reasoning_effort", "model"):
            value = defaults.get(key)
            if value is None:
                continue
            if key not in merged:
                merged[key] = value
                if key == "reasoning_effort":
                    cfg = defaults.get("reasoning_config")
                    if cfg is not None:
                        merged["reasoning_config"] = cfg
            elif merged[key] != value:
                conflicts.setdefault(key, set()).update({merged[key], value})

        for field in defaults.get("required") or []:
            if field not in required:
                required.append(field)

    # Drop conflicting fields from the merged dict.
    for field in list(conflicts.keys()):
        merged.pop(field, None)
        if field == "reasoning_effort":
            merged.pop("reasoning_config", None)

    warnings: List[str] = []
    if conflicts:
        fields = ", ".join(sorted(conflicts.keys()))
        warnings.append(f"Conflicting runtime defaults dropped: {fields}")
        logger.warning(
            "skill_runtime_defaults.conflict_dropped fields=%s", fields
        )

    if required:
        merged["required"] = required
    return merged, warnings


def emit_events(events: List[LogEvent]) -> None:
    """Helper: log ``LogEvent`` entries through the module logger.

    Callers that want to route events to metrics as well should iterate the
    list themselves rather than calling this helper.
    """
    for event in events:
        level = logging.getLevelName(event.level)
        if not isinstance(level, int):
            level = logging.INFO
        logger.log(level, "%s %s", event.name, event.fields)

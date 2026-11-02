"""v5 signals: which disabled built-in tools models reach for, and when an install first adopts a feature.

``hermes.tool_unavailable.count`` is counted where the agent validates the calls a model emitted: a
name Hermes ships (``toolsets.BUILTIN_TOOL_NAMES``) that this session did not enable. Any other name
stays the v4 ``unknown_tool`` quality issue only, so a plugin, MCP or hallucinated name never leaves.

``hermes.feature_adoption.count`` rides on counters the subscriber already records: the first counter
row that proves a feature was really used latches it once per install, with the owning profile's age.
Features no counter can see report through :func:`record_feature_used`.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable

from . import shared_metrics_contract as contract

logger = logging.getLogger(__name__)

_DAY_S = 86_400
_DAYS_SINCE_INSTALL_THRESHOLDS = (
    (_DAY_S, "same_day"), (7 * _DAY_S, "1d_to_7d"), (30 * _DAY_S, "7d_to_30d"), (90 * _DAY_S, "30d_to_90d"),
)


# ---- tool unavailable ------------------------------------------------------------------------

def tool_unavailable_fields(agent: Any, name: Any, issue: str, route: dict[str, str]) -> dict[str, str] | None:
    """Fields for a call to a shipped built-in the session has disabled; None for anything else.

    Background reviews, delegated children and cron jobs run with toolsets Hermes (or the parent
    model) narrowed on purpose, so their misses say nothing about what users should get by default.
    A built-in deferred behind ``tool_search`` is enabled (reachable through ``tool_call``), not missing.
    """
    if issue != "unknown_tool" or not isinstance(name, str) or name not in contract.BUILTIN_TOOL_NAMES:
        return None
    if getattr(agent, "_delegate_depth", 0) or getattr(agent, "platform", None) == "cron":
        return None
    from agent.tool_executor import _tool_search_scoped_names
    from tools.skill_provenance import is_background_review

    if is_background_review() or name in _tool_search_scoped_names(agent):
        return None
    return {**route, "tool_name": name}


# ---- feature adoption ------------------------------------------------------------------------

def _success(d: dict[str, str]) -> bool:
    return d.get("outcome") == "success"


def _tool(*names: str, prefix: str = "") -> Callable[[dict[str, str]], bool]:
    def used(d: dict[str, str]) -> bool:
        tool = d.get("tool_name") or ""
        return _success(d) and (tool in names or bool(prefix) and tool.startswith(prefix))

    return used


def user_created_skill(d: dict[str, str]) -> bool:
    """A skill created at the user's request (``agent_created`` is Hermes' own background review)."""
    return d.get("action") == "created" and d.get("provenance") != "agent_created"


_TASK_SURFACE = {"desktop": "desktop", "tui": "tui"}
# metric -> [(feature, predicate over the recorded dimensions)]. Unattended Hermes-owned work only
# counts where the user set it up (a cron job, a curator run the user started).
_FEATURE_RULES: dict[str, tuple[tuple[str, Callable[[dict[str, str]], bool]], ...]] = {
    contract.MEMORY_OP_METRIC: (("memory", lambda d: _success(d) and d.get("origin") == "foreground"),),
    contract.SKILL_LIFECYCLE_METRIC: (("skills_created", user_created_skill),),
    contract.DELEGATION_RUN_METRIC: (("delegation", lambda d: True),),
    contract.CRON_RUN_METRIC: (("cron", lambda d: d.get("outcome") in {"success", "failed"}),),
    contract.CURATOR_RUN_METRIC: (("curator", lambda d: _success(d) and d.get("trigger") == "manual"),),
    contract.TASK_STARTED_METRIC: (
        ("gateway_platform", lambda d: d.get("platform") not in {None, "none"}),
        ("desktop", lambda d: d.get("execution_surface") == "desktop"),
        ("tui", lambda d: d.get("execution_surface") == "tui"),
    ),
    contract.TOOL_USAGE_METRIC: (
        ("mcp", _tool("mcp")),
        ("plugins", _tool("plugin")),
        ("browser", _tool(prefix="browser_")),
        ("voice", _tool("text_to_speech")),
        ("kanban", _tool(prefix="kanban_")),
        ("projects", _tool("desktop_project")),
    ),
    contract.SLASH_COMMAND_METRIC: (
        ("voice", lambda d: d.get("command") == "voice"),
        ("kanban", lambda d: d.get("command") == "kanban"),
    ),
    contract.FEATURE_USED_MARK: tuple((feature, lambda d, f=feature: d.get("feature") == f) for feature in sorted(contract.FEATURES)),
}


def features_for(metric_name: str, dimensions: dict[str, str]) -> tuple[str, ...]:
    """Features a recorded counter proves were used (each is latched once per install by the store)."""
    return tuple(name for name, used in _FEATURE_RULES.get(metric_name, ()) if used(dimensions))


def days_since_install_bucket(home: Path) -> str:
    """The owning profile's age (its first session), bucketed; ``unknown`` when state.db is unreadable."""
    from .shared_metrics_snapshot import _first_session_started_at

    try:
        first = _first_session_started_at(home)
    except (sqlite3.Error, OSError, TypeError, ValueError):  # unreadable, or a non-numeric started_at
        return "unknown"
    age = contract._non_negative_number(time.time() - first) if first is not None else None
    return "unknown" if age is None else contract._bucket(age, _DAYS_SINCE_INSTALL_THRESHOLDS, "gte_90d")


def feature_used_counter(event: Any) -> tuple[str, dict[str, str]] | None:
    """A first-use fact for a feature no counter sees (never stored as a counter row itself)."""
    if not contract._valid_shape(event, **contract._MARK_SHAPE):
        return None
    if contract._event_text(event, "name") != contract.FEATURE_USED_MARK:
        return None
    data = getattr(event, "data", None)
    if not isinstance(data, dict) or set(data) != {"feature"} or data["feature"] not in contract.FEATURES:
        return None
    return contract.FEATURE_USED_MARK, {"feature": data["feature"]}


def record_feature_used(feature: str, *, hermes_home: Any = None) -> None:
    """Report one real use of ``feature`` (only the first per install becomes a row). Never raises.
    ``hermes_home`` binds the owning profile for callers outside it."""
    token = None
    try:
        if hermes_home:
            from hermes_constants import set_hermes_home_override

            token = set_hermes_home_override(str(hermes_home))
        from .relay_shared_metrics import enabled, record_process_mark

        if feature in contract.FEATURES and enabled():
            record_process_mark(contract.FEATURE_USED_MARK, {"feature": feature})
    except Exception:
        logger.debug("Shared-metrics feature use not recorded", exc_info=True)
    finally:
        if token is not None:
            from hermes_constants import reset_hermes_home_override

            reset_hermes_home_override(token)

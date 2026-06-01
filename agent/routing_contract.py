"""Runtime validation for model-emitted routing decisions.

The model may emit a human-readable line such as::

    Routing Decision: domain -> profile/skill -> inline(read-only) -> architecture_design/gpt-5.5-xhigh

This module treats that line as a lightweight route contract.  It does not
decide how to route work itself; it checks that the declared lane matches the
profile routing table and that the upcoming execution surface can actually
honor high-reasoning lanes.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

logger = logging.getLogger(__name__)

EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
_EFFORT_RE = re.compile(
    r"(?:[/@:_-])(" + "|".join(re.escape(e) for e in EFFORTS) + r")\s*$",
    re.IGNORECASE,
)

_LANE_ALIASES = {
    "architecture": "architecture_design",
    "architecture/design": "architecture_design",
    "config_patch": "simple_config_patch",
    "debugging_complex": "complex_debugging",
    "file_authoring": "file_authoring_bounded",
    "implementation": "code_implementation",
    "implementation_routine": "code_implementation",
    "research_public": "public_research",
    "review_security": "security_review",
    "synthesis": "synthesis_recommendation",
    "verification": "verification_leaf",
}

_FRONT_DOOR_LANES = {
    "front_door",
    "frontdoor",
    "current_session",
    "active_session",
}


@dataclass(frozen=True)
class DeclaredRoute:
    raw_line: str
    domain: str
    target: str
    surface: str
    lane_label: str
    lane_name: str
    declared_provider: Optional[str] = None
    declared_model: Optional[str] = None
    declared_effort: Optional[str] = None


@dataclass(frozen=True)
class RouteContractViolation:
    code: str
    message: str
    recovery_prompt: str


@dataclass(frozen=True)
class RouteContractCheck:
    declared: Optional[DeclaredRoute] = None
    lane_route: Optional[dict[str, Any]] = None
    execution_surface: str = "none"
    violation: Optional[RouteContractViolation] = None

    @property
    def ok(self) -> bool:
        return self.violation is None


@dataclass(frozen=True)
class _ToolCall:
    name: str
    args: dict[str, Any]


def active_reasoning_effort(agent: Any) -> str:
    config = getattr(agent, "reasoning_config", None)
    if isinstance(config, Mapping):
        if config.get("enabled") is False:
            return "none"
        effort = str(config.get("effort") or "").strip().lower()
        return effort or "medium"
    return "default"


def extract_declared_route(text: str | None) -> Optional[DeclaredRoute]:
    if not isinstance(text, str) or "Routing Decision:" not in text:
        return None
    for raw in text.splitlines():
        line = raw.strip()
        line = line.lstrip("> ").lstrip("- ").strip()
        if not line.startswith("Routing Decision:"):
            continue
        body = line.split("Routing Decision:", 1)[1].strip()
        parts = [p.strip() for p in re.split(r"\s*(?:→|->)\s*", body) if p.strip()]
        if len(parts) < 4:
            return None
        lane_label = " -> ".join(parts[3:]).strip()
        lane_name, provider, model, effort = _parse_lane_label(lane_label)
        return DeclaredRoute(
            raw_line=line,
            domain=parts[0],
            target=parts[1],
            surface=parts[2],
            lane_label=lane_label,
            lane_name=lane_name,
            declared_provider=provider,
            declared_model=model,
            declared_effort=effort,
        )
    return None


def check_route_contract(
    *,
    text: str | None,
    tool_calls: Iterable[Any] | None,
    active_provider: str | None,
    active_model: str | None,
    active_effort: str | None,
) -> RouteContractCheck:
    """Validate the declared Routing Decision line, when one is present."""
    declared = extract_declared_route(text)
    if declared is None or not _guard_enabled():
        return RouteContractCheck(declared=declared, execution_surface="not_declared")

    lane_route = _resolve_lane_route(declared.lane_name)
    calls = list(_coerce_tool_calls(tool_calls))

    if _normalize_lane_key(declared.lane_name) in _FRONT_DOOR_LANES:
        execution = "active_session" if _active_matches_declared_front_door(
            declared, active_provider, active_model, active_effort
        ) else "active_session_mismatch"
        if execution == "active_session_mismatch":
            return RouteContractCheck(
                declared=declared,
                execution_surface=execution,
                violation=_violation(
                    "front_door_label_mismatch",
                    declared,
                    "the `front_door` route label does not match the active session",
                ),
            )
        return RouteContractCheck(declared=declared, execution_surface=execution)

    if lane_route is None:
        # Unknown lanes remain advisory; the guard cannot safely infer intent.
        return RouteContractCheck(declared=declared, execution_surface=_execution_surface(calls))

    expected_effort = _clean_effort(lane_route.get("reasoning_effort"))
    if expected_effort and declared.declared_effort and declared.declared_effort != expected_effort:
        return RouteContractCheck(
            declared=declared,
            lane_route=lane_route,
            execution_surface=_execution_surface(calls),
            violation=_violation(
                "lane_label_effort_mismatch",
                declared,
                (
                    f"lane `{declared.lane_name}` resolves to reasoning_effort "
                    f"`{expected_effort}`, but the route line labels it "
                    f"`{declared.declared_effort}`"
                ),
                lane_route,
            ),
        )

    # Only high-reasoning lanes need hard runtime enforcement.  Lower lanes may
    # be intentionally over-resourced by the active front door, quota policy, or
    # privacy fallback.
    if expected_effort != "xhigh":
        return RouteContractCheck(
            declared=declared,
            lane_route=lane_route,
            execution_surface=_execution_surface(calls) or "advisory",
        )

    execution = _satisfying_execution_surface(
        calls,
        lane_name=_normalize_lane_key(declared.lane_name),
        lane_route=lane_route,
        active_provider=active_provider,
        active_model=active_model,
        active_effort=active_effort,
    )
    if execution == "none":
        return RouteContractCheck(
            declared=declared,
            lane_route=lane_route,
            execution_surface=execution,
            violation=_violation(
                "xhigh_lane_not_executed",
                declared,
                (
                    f"lane `{declared.lane_name}` requires xhigh reasoning, but "
                    f"the active session is `{active_provider or 'unknown'}/"
                    f"{active_model or 'unknown'}` at `{active_effort or 'default'}` "
                    "and no matching xhigh delegate_task or kanban_create route was requested"
                ),
                lane_route,
            ),
        )

    surface = declared.surface.strip().lower()
    if surface.startswith("inline") and execution != "active_session":
        return RouteContractCheck(
            declared=declared,
            lane_route=lane_route,
            execution_surface=execution,
            violation=_violation(
                "surface_label_mismatch",
                declared,
                (
                    f"the route line says `{declared.surface}`, but the "
                    f"declared xhigh lane is actually satisfied by `{execution}`"
                ),
                lane_route,
            ),
        )

    return RouteContractCheck(
        declared=declared,
        lane_route=lane_route,
        execution_surface=execution,
    )


def record_route_contract_event(
    agent: Any,
    event: str,
    check: RouteContractCheck,
) -> None:
    """Append redacted route-contract telemetry under HERMES_HOME/logs."""
    declared = check.declared
    if declared is None:
        return
    try:
        from hermes_constants import get_hermes_home

        log_dir = Path(get_hermes_home()) / "logs"
    except Exception:
        log_dir = Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser() / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": event,
            "session_id": getattr(agent, "session_id", None),
            "provider": getattr(agent, "provider", None),
            "model": getattr(agent, "model", None),
            "reasoning_effort": active_reasoning_effort(agent),
            "domain": declared.domain,
            "target": declared.target,
            "surface": declared.surface,
            "lane": declared.lane_name,
            "lane_label": declared.lane_label,
            "execution_surface": check.execution_surface,
        }
        if check.lane_route:
            payload["resolved_route"] = {
                "provider": check.lane_route.get("provider"),
                "model": check.lane_route.get("model"),
                "reasoning_effort": check.lane_route.get("reasoning_effort"),
                "route_key": check.lane_route.get("route_key"),
            }
        if check.violation:
            payload["violation"] = {
                "code": check.violation.code,
                "message": check.violation.message,
            }
        with (log_dir / "routing-contract.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        logger.debug("failed to write route-contract telemetry", exc_info=True)


def _guard_enabled() -> bool:
    return str(os.environ.get("HERMES_ROUTE_CONTRACT_GUARD", "1")).lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _parse_lane_label(label: str) -> tuple[str, Optional[str], Optional[str], Optional[str]]:
    clean = _clean_lane_label(label)
    lane_name = clean
    tail = ""
    if "/" in clean:
        lane_name, tail = clean.split("/", 1)
    lane_name = lane_name.strip().strip("`")
    provider = None
    model = None
    effort = None
    if tail:
        tail = tail.strip().strip("`")
        m = _EFFORT_RE.search(tail)
        if m:
            effort = _clean_effort(m.group(1))
            tail = tail[: m.start()].rstrip("/@:_- ")
        if tail:
            if "/" in tail:
                provider, model = tail.rsplit("/", 1)
                provider = provider.strip() or None
                model = model.strip() or None
            else:
                model = tail.strip() or None
    return lane_name, provider, model, effort


def _clean_lane_label(label: str) -> str:
    """Return a lane label stripped of prose punctuation.

    Models often emit a correct route line as a sentence, e.g.
    ``front_door/gpt-5.5-high.``. The terminal period is not part of the
    model name or effort, so remove sentence punctuation before parsing.
    """
    clean = label.strip().strip("`").strip()
    clean = clean.rstrip("`.,;!?").strip()
    # Drop a final parenthetical annotation before trimming sentence
    # punctuation. The route contract is the compact label before the note.
    clean = re.sub(r"\s+\([^)]*\)\s*$", "", clean)
    return clean.rstrip("`.,;!?").strip()


def _normalize_lane_key(value: Any) -> str:
    key = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    return _LANE_ALIASES.get(key, key)


def _clean_effort(value: Any) -> Optional[str]:
    effort = str(value or "").strip().lower()
    return effort if effort in EFFORTS else None


def _resolve_lane_route(lane_name: str) -> Optional[dict[str, Any]]:
    key = _normalize_lane_key(lane_name)
    if key in _FRONT_DOOR_LANES:
        return None
    try:
        from tools.kanban_tools import _load_model_routing_table

        table, _source = _load_model_routing_table()
        lanes = table.get("task_lanes") if isinstance(table.get("task_lanes"), Mapping) else {}
        route = lanes.get(key)
        if isinstance(route, Mapping):
            return dict(route)
    except Exception:
        logger.debug("failed to resolve model-routing lane", exc_info=True)

    if key in {
        "architecture_design",
        "security_review",
        "complex_debugging",
        "synthesis_recommendation",
        "front_door_or_uncertain",
    }:
        return {
            "provider": "openai-codex",
            "model": "gpt-5.5",
            "reasoning_effort": "xhigh",
            "route_key": "openai-codex/gpt-5.5",
        }
    return None


def _coerce_tool_calls(tool_calls: Iterable[Any] | None) -> Iterable[_ToolCall]:
    for call in tool_calls or []:
        function = getattr(call, "function", None)
        if function is None and isinstance(call, Mapping):
            function = call.get("function")
        if isinstance(function, Mapping):
            name = str(function.get("name") or "")
            raw_args = function.get("arguments")
        else:
            name = str(getattr(function, "name", "") or "")
            raw_args = getattr(function, "arguments", None)
        if not name:
            continue
        args: dict[str, Any] = {}
        if isinstance(raw_args, Mapping):
            args = dict(raw_args)
        elif isinstance(raw_args, str) and raw_args.strip():
            try:
                parsed = json.loads(raw_args)
                if isinstance(parsed, Mapping):
                    args = dict(parsed)
            except Exception:
                args = {}
        yield _ToolCall(name=name, args=args)


def _active_matches_declared_front_door(
    declared: DeclaredRoute,
    active_provider: str | None,
    active_model: str | None,
    active_effort: str | None,
) -> bool:
    if declared.declared_effort and declared.declared_effort != (active_effort or "").lower():
        return False
    if declared.declared_model and not _model_matches(declared.declared_model, active_model):
        return False
    if declared.declared_provider and not _text_matches(declared.declared_provider, active_provider):
        return False
    return True


def _satisfying_execution_surface(
    calls: list[_ToolCall],
    *,
    lane_name: str,
    lane_route: Mapping[str, Any],
    active_provider: str | None,
    active_model: str | None,
    active_effort: str | None,
) -> str:
    if _route_matches(lane_route, active_provider, active_model, active_effort):
        return "active_session"
    for call in calls:
        if call.name == "delegate_task" and _delegate_task_satisfies(
            call.args, lane_route, active_provider, active_model
        ):
            return "delegate_task"
        if call.name == "kanban_create" and _kanban_create_satisfies(
            call.args, lane_name, lane_route
        ):
            return "kanban_create"
    return "none"


def _execution_surface(calls: list[_ToolCall]) -> str:
    names = {c.name for c in calls}
    if "delegate_task" in names:
        return "delegate_task"
    if "kanban_create" in names:
        return "kanban_create"
    if names:
        return "inline_tools"
    return "inline_final"


def _delegate_task_satisfies(
    args: Mapping[str, Any],
    lane_route: Mapping[str, Any],
    active_provider: str | None,
    active_model: str | None,
) -> bool:
    top_provider = _clean_text(args.get("provider"))
    top_model = _clean_text(args.get("model"))
    top_effort = _clean_effort(args.get("reasoning_effort"))
    tasks = args.get("tasks")
    if isinstance(tasks, list) and tasks:
        for task in tasks:
            if not isinstance(task, Mapping):
                continue
            provider = _clean_text(task.get("provider")) or top_provider or active_provider
            model = _clean_text(task.get("model")) or top_model or active_model
            effort = _clean_effort(task.get("reasoning_effort")) or top_effort
            if _route_matches(lane_route, provider, model, effort):
                return True
        return False
    provider = top_provider or active_provider
    model = top_model or active_model
    return _route_matches(lane_route, provider, model, top_effort)


def _kanban_create_satisfies(
    args: Mapping[str, Any],
    lane_name: str,
    lane_route: Mapping[str, Any],
) -> bool:
    requested = _normalize_lane_key(args.get("model_routing"))
    if requested != lane_name:
        return False
    # kanban_create(model_routing=...) resolves both model and reasoning effort
    # into the task dispatch metadata.  The dispatcher is responsible for
    # carrying that effort into the worker process.
    return bool(_clean_effort(lane_route.get("reasoning_effort")))


def _route_matches(
    lane_route: Mapping[str, Any],
    provider: str | None,
    model: str | None,
    effort: str | None,
) -> bool:
    expected_effort = _clean_effort(lane_route.get("reasoning_effort"))
    if expected_effort and expected_effort != _clean_effort(effort):
        return False
    expected_model = _clean_text(lane_route.get("model"))
    if expected_model and not _model_matches(expected_model, model):
        return False
    expected_provider = _clean_text(lane_route.get("provider"))
    actual_provider = _clean_text(provider)
    if actual_provider in {"auto", "default"}:
        actual_provider = None
    if (
        expected_provider
        and actual_provider
        and not _text_matches(expected_provider, actual_provider)
    ):
        return False
    return True


def _model_matches(expected: str | None, actual: str | None) -> bool:
    if not expected or not actual:
        return False
    exp = expected.strip().lower()
    act = actual.strip().lower()
    return exp == act or exp.split("/")[-1] == act.split("/")[-1]


def _text_matches(expected: str | None, actual: str | None) -> bool:
    if not expected or not actual:
        return False
    return expected.strip().lower() == actual.strip().lower()


def _clean_text(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _violation(
    code: str,
    declared: DeclaredRoute,
    reason: str,
    lane_route: Mapping[str, Any] | None = None,
) -> RouteContractViolation:
    route_hint = ""
    if lane_route:
        provider = lane_route.get("provider") or "openai-codex"
        model = lane_route.get("model") or "gpt-5.5"
        effort = lane_route.get("reasoning_effort") or "xhigh"
        route_hint = (
            f" Use a real route such as delegate_task(provider={provider!r}, "
            f"model={model!r}, reasoning_effort={effort!r}) or "
            f"kanban_create(model_routing={declared.lane_name!r})."
        )
    message = f"Route contract violation: {reason}."
    recovery = (
        f"{message}{route_hint} If the work is only inline read-only/advisory, "
        "retry with an honest `front_door/<model>-<effort>` route label instead "
        "of an xhigh lane."
    )
    return RouteContractViolation(code=code, message=message, recovery_prompt=recovery)

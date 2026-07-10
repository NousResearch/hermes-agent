"""Runtime-owned per-turn receipt facts.

The receipt is deliberately small: it captures facts the runtime can know
without importing heavier coordination systems.  Final response formatters can
use it to add durable status metadata without asking the model to author it.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


@dataclass
class TurnReceipt:
    session_id: str
    turn_id: str
    provider: str = ""
    model: str = ""
    platform: str = ""
    route: str = "native"
    reason: str = "runtime_default"
    opencode_state: str = "unknown"  # unknown | called | not_called
    tool_names: list[str] = field(default_factory=list)
    tool_total: int = 0
    tool_failed: int = 0
    agents_count: Optional[int] = 0
    subagents_count: Optional[int] = 0
    delegation_observed: bool = False
    human_language_state: str = "unknown"  # unknown | seen | not_seen
    evidence_status: str = "unknown"  # ok | partial | failed | unknown
    api_calls: int = 0
    completed: bool = False
    failed: bool = False
    interrupted: bool = False
    exit_reason: str = "unknown"
    start_monotonic: Optional[float] = None
    elapsed_seconds: Optional[float] = None

    @classmethod
    def start(
        cls,
        *,
        session_id: str,
        turn_id: str,
        provider: str = "",
        model: str = "",
        platform: str = "",
        start_monotonic: Optional[float] = None,
    ) -> "TurnReceipt":
        receipt = cls(
            session_id=session_id or "",
            turn_id=turn_id or "",
            provider=provider or "",
            model=model or "",
            platform=platform or "",
            start_monotonic=start_monotonic,
        )
        _infer_route_from_runtime(receipt)
        return receipt


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _infer_route_from_runtime(receipt: TurnReceipt) -> None:
    provider = (receipt.provider or "").strip().lower()
    model = (receipt.model or "").strip().lower()
    if provider == "moa":
        receipt.route = "moa"
        receipt.reason = "provider_moa"
    elif provider.startswith("opencode") or model.startswith("opencode"):
        receipt.route = "opencode"
        receipt.reason = "provider_opencode"
        receipt.opencode_state = "called"


def _tool_call_name(tool_call: Any) -> str:
    if isinstance(tool_call, Mapping):
        function = tool_call.get("function")
        if isinstance(function, Mapping):
            return str(function.get("name") or "").strip()
        return str(tool_call.get("name") or "").strip()
    function = getattr(tool_call, "function", None)
    return str(getattr(function, "name", "") or getattr(tool_call, "name", "") or "").strip()


def _message_role(message: Any) -> str:
    if isinstance(message, Mapping):
        return str(message.get("role") or "")
    return str(getattr(message, "role", "") or "")


def _message_tool_calls(message: Any) -> list[Any]:
    if isinstance(message, Mapping):
        calls = message.get("tool_calls") or []
    else:
        calls = getattr(message, "tool_calls", None) or []
    return list(calls) if isinstance(calls, (list, tuple)) else []


def _failure_from_mapping(payload: Mapping[str, Any]) -> bool:
    """Return failure only from explicit, structured execution metadata."""

    if payload.get("is_error") is True or payload.get("success") is False:
        return True
    status = str(payload.get("status") or "").strip().lower()
    if status in {"error", "failed", "failure"}:
        return True
    exit_code = payload.get("exit_code")
    if exit_code is not None:
        try:
            if int(exit_code) != 0:
                return True
        except (TypeError, ValueError):
            pass
    error = payload.get("error")
    return error not in (None, "", False, 0, [], {})


def _tool_result_failed(message: Any) -> bool:
    """Classify a tool result without scanning successful prose for keywords."""

    if isinstance(message, Mapping) and _failure_from_mapping(message):
        return True
    if isinstance(message, Mapping):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)
    if isinstance(content, Mapping):
        return _failure_from_mapping(content)
    if not isinstance(content, str):
        return False
    stripped = content.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return False
    try:
        decoded = json.loads(stripped)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return isinstance(decoded, Mapping) and _failure_from_mapping(decoded)


def update_turn_receipt_from_result(
    receipt: TurnReceipt,
    *,
    completed: bool,
    failed: bool,
    interrupted: bool,
    api_calls: int,
    exit_reason: str,
    messages: list[Mapping[str, Any]] | list[Any],
    agent: Any = None,
) -> TurnReceipt:
    """Update ``receipt`` from finalizer/runtime facts."""

    if agent is not None:
        receipt.provider = str(getattr(agent, "provider", receipt.provider) or receipt.provider)
        receipt.model = str(getattr(agent, "model", receipt.model) or receipt.model)
        receipt.platform = str(getattr(agent, "platform", receipt.platform) or receipt.platform)
        _infer_route_from_runtime(receipt)

    receipt.completed = bool(completed)
    receipt.failed = bool(failed)
    receipt.interrupted = bool(interrupted)
    try:
        receipt.api_calls = int(api_calls or 0)
    except Exception:
        receipt.api_calls = 0
    receipt.exit_reason = str(exit_reason or "unknown")
    if receipt.start_monotonic is not None and receipt.elapsed_seconds is None:
        receipt.elapsed_seconds = max(0.0, time.monotonic() - receipt.start_monotonic)

    names: list[str] = []
    failures = 0
    for message in messages or []:
        if _message_role(message) == "assistant":
            for call in _message_tool_calls(message):
                name = _tool_call_name(call)
                if name:
                    names.append(name)
        elif _message_role(message) == "tool":
            if _tool_result_failed(message):
                failures += 1

    receipt.tool_names = names
    receipt.tool_total = len(names)
    receipt.tool_failed = failures
    if any(name == "delegate_task" for name in names):
        receipt.delegation_observed = True
        if receipt.agents_count == 0:
            receipt.agents_count = None
        if receipt.subagents_count == 0:
            receipt.subagents_count = None

    if failed or interrupted:
        receipt.evidence_status = "failed" if failed else "partial"
    elif completed:
        receipt.evidence_status = "ok"
    elif receipt.evidence_status == "unknown":
        receipt.evidence_status = "partial"

    return receipt


def apply_turn_facts(receipt: TurnReceipt, facts: Mapping[str, Any] | None) -> TurnReceipt:
    """Merge optional runtime fact dictionaries into ``receipt``.

    This accepts a narrow, generic shape so future MoA/coordination layers can
    pass facts without this module depending on those systems.
    """

    if not isinstance(facts, Mapping):
        return receipt

    route = _as_mapping(facts.get("route"))
    if route.get("actual"):
        receipt.route = str(route["actual"])
    if route.get("reason"):
        receipt.reason = str(route["reason"])

    opencode = _as_mapping(facts.get("opencode"))
    if "state" in opencode:
        receipt.opencode_state = str(opencode.get("state") or "unknown")
    elif "observed" in opencode:
        receipt.opencode_state = "called" if bool(opencode.get("observed")) else "not_called"

    tools = _as_mapping(facts.get("tools"))
    if tools:
        names = tools.get("names")
        if isinstance(names, (list, tuple)):
            receipt.tool_names = [str(name) for name in names if str(name)]
        if "total" in tools:
            try:
                receipt.tool_total = int(tools.get("total") or 0)
            except Exception:
                pass
        elif receipt.tool_names:
            receipt.tool_total = len(receipt.tool_names)
        if "failed" in tools:
            try:
                receipt.tool_failed = int(tools.get("failed") or 0)
            except Exception:
                pass

    delegation = _as_mapping(facts.get("delegation")) or _as_mapping(facts.get("agents"))
    if delegation:
        if "observed" in delegation:
            receipt.delegation_observed = bool(delegation.get("observed"))
        if "agents" in delegation:
            try:
                _raw_agents = delegation.get("agents")
                receipt.agents_count = int(_raw_agents) if _raw_agents is not None else None
            except Exception:
                receipt.agents_count = None
        if "subagents" in delegation:
            try:
                _raw_subagents = delegation.get("subagents")
                receipt.subagents_count = int(_raw_subagents) if _raw_subagents is not None else None
            except Exception:
                receipt.subagents_count = None

    human = _as_mapping(facts.get("human_language")) or _as_mapping(facts.get("human"))
    if human:
        if "state" in human:
            receipt.human_language_state = str(human.get("state") or "unknown")
        elif "observed" in human:
            receipt.human_language_state = "seen" if bool(human.get("observed")) else "not_seen"

    evidence = _as_mapping(facts.get("evidence"))
    if evidence.get("level"):
        receipt.evidence_status = str(evidence.get("level") or "unknown")
    elif evidence.get("status"):
        receipt.evidence_status = str(evidence.get("status") or "unknown")

    return receipt

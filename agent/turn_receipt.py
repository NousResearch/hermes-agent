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
    mechanism_segments: list[str] = field(default_factory=list)
    agents_count: Optional[int] = 0
    subagents_count: Optional[int] = 0
    coordination_count: Optional[int] = 0
    coordination_breakdown: dict[str, Optional[int]] = field(default_factory=dict)
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


def _tool_call_id(tool_call: Any) -> str:
    if isinstance(tool_call, Mapping):
        return str(tool_call.get("id") or tool_call.get("call_id") or "").strip()
    return str(getattr(tool_call, "id", "") or getattr(tool_call, "call_id", "") or "").strip()


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


def _message_tool_call_id(message: Any) -> str:
    if isinstance(message, Mapping):
        return str(message.get("tool_call_id") or message.get("call_id") or "").strip()
    return str(getattr(message, "tool_call_id", "") or getattr(message, "call_id", "") or "").strip()


def _message_explicit_tool_names(message: Any) -> set[str]:
    if isinstance(message, Mapping):
        raw_names = (message.get("name"), message.get("tool_name"))
    else:
        raw_names = (getattr(message, "name", ""), getattr(message, "tool_name", ""))
    return {str(name).strip() for name in raw_names if str(name or "").strip()}


def _message_content(message: Any) -> Any:
    if isinstance(message, Mapping):
        return message.get("content")
    return getattr(message, "content", None)


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


def _add_mechanism_segment(receipt: TurnReceipt, segment: str) -> None:
    text = str(segment or "").strip()
    if text and text not in receipt.mechanism_segments:
        receipt.mechanism_segments.append(text)


def _count_strings(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        return len([item for item in value if str(item)])
    return 0


def _int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _normalize_coordination_count(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return max(0, int(value))
    except Exception:
        return None


def _recompute_coordination_count(receipt: TurnReceipt) -> None:
    if "explicit" in receipt.coordination_breakdown:
        receipt.coordination_count = _normalize_coordination_count(
            receipt.coordination_breakdown.get("explicit")
        )
        return
    total = 0
    for count in receipt.coordination_breakdown.values():
        normalized = _normalize_coordination_count(count)
        if normalized is None:
            receipt.coordination_count = None
            return
        total += normalized
    receipt.coordination_count = total


_AUTHORITATIVE_COORDINATION_COMPONENTS = {"explicit", "moa", "omo", "delegate_task"}


def _has_explicit_coordination_total(receipt: TurnReceipt) -> bool:
    return "explicit" in receipt.coordination_breakdown


def _has_authoritative_coordination_component(receipt: TurnReceipt) -> bool:
    return any(component in receipt.coordination_breakdown for component in _AUTHORITATIVE_COORDINATION_COMPONENTS)


def _set_coordination_component(receipt: TurnReceipt, key: str, count: Any) -> None:
    component = str(key or "").strip().lower()
    if not component:
        return
    if component == "explicit":
        receipt.coordination_breakdown.clear()
        receipt.coordination_breakdown[component] = _normalize_coordination_count(count)
        _recompute_coordination_count(receipt)
        return
    if _has_explicit_coordination_total(receipt):
        _recompute_coordination_count(receipt)
        return
    if component == "delegation" and _has_authoritative_coordination_component(receipt):
        receipt.coordination_breakdown.pop("delegation", None)
        _recompute_coordination_count(receipt)
        return
    if component in _AUTHORITATIVE_COORDINATION_COMPONENTS:
        receipt.coordination_breakdown.pop("delegation", None)
    receipt.coordination_breakdown[component] = _normalize_coordination_count(count)
    _recompute_coordination_count(receipt)


def _clear_coordination_component(receipt: TurnReceipt, key: str) -> None:
    component = str(key or "").strip().lower()
    if component in receipt.coordination_breakdown:
        del receipt.coordination_breakdown[component]
        _recompute_coordination_count(receipt)


def _decode_tool_result_payload(content: Any) -> Mapping[str, Any] | None:
    if isinstance(content, Mapping):
        return content
    if not isinstance(content, str):
        return None
    stripped = content.strip()
    if not stripped:
        return None
    try:
        decoded = json.loads(stripped)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return decoded if isinstance(decoded, Mapping) else None


def _delegate_task_result_count(message: Any) -> Optional[int]:
    payload = _decode_tool_result_payload(_message_content(message))
    if not isinstance(payload, Mapping):
        return None
    results = payload.get("results")
    if isinstance(results, list):
        return len(results)
    status = str(payload.get("status") or "").strip().lower()
    mode = str(payload.get("mode") or "").strip().lower()
    if status == "dispatched" and mode == "background":
        count = payload.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            return None
        return count
    return None


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
    assistant_tool_names_by_call_id: dict[str, set[str]] = {}
    delegate_call_ids: list[str] = []
    seen_delegate_call_ids: set[str] = set()
    duplicate_delegate_call_id = False
    delegate_calls_without_id = 0
    for message in messages or []:
        if _message_role(message) == "assistant":
            for call in _message_tool_calls(message):
                name = _tool_call_name(call)
                call_id = _tool_call_id(call)
                if name:
                    names.append(name)
                if call_id and name:
                    assistant_tool_names_by_call_id.setdefault(call_id, set()).add(name)
                if name == "delegate_task":
                    if call_id:
                        if call_id in seen_delegate_call_ids:
                            duplicate_delegate_call_id = True
                        seen_delegate_call_ids.add(call_id)
                        delegate_call_ids.append(call_id)
                    else:
                        delegate_calls_without_id += 1
        elif _message_role(message) == "tool":
            if _tool_result_failed(message):
                failures += 1

    receipt.tool_names = names
    receipt.tool_total = len(names)
    receipt.tool_failed = failures

    delegate_call_count = len(delegate_call_ids) + delegate_calls_without_id
    if delegate_call_count:
        receipt.delegation_observed = True
        if receipt.agents_count == 0:
            receipt.agents_count = None
        if receipt.subagents_count == 0:
            receipt.subagents_count = None

        delegate_result_total = 0
        delegate_result_unknown = False
        delegate_results_by_call_id: dict[str, list[Optional[int]]] = {
            call_id: [] for call_id in seen_delegate_call_ids
        }
        anonymous_delegate_results: list[Optional[int]] = []
        for message in messages or []:
            if _message_role(message) != "tool":
                continue
            call_id = _message_tool_call_id(message)
            explicit_tool_names = _message_explicit_tool_names(message)
            explicit_non_delegate = any(name != "delegate_task" for name in explicit_tool_names)
            explicit_delegate = "delegate_task" in explicit_tool_names
            if call_id and call_id in delegate_results_by_call_id:
                if explicit_non_delegate:
                    delegate_results_by_call_id[call_id].append(None)
                else:
                    delegate_results_by_call_id[call_id].append(_delegate_task_result_count(message))
            elif explicit_delegate and not explicit_non_delegate:
                anonymous_delegate_results.append(_delegate_task_result_count(message))

        delegate_call_id_name_conflict = any(
            "delegate_task" in call_names and len(call_names) > 1
            for call_names in assistant_tool_names_by_call_id.values()
        )
        if duplicate_delegate_call_id or delegate_call_id_name_conflict:
            delegate_result_unknown = True

        for result_counts in delegate_results_by_call_id.values():
            if len(result_counts) != 1 or result_counts[0] is None:
                delegate_result_unknown = True
            else:
                delegate_result_total += result_counts[0]

        if delegate_calls_without_id:
            if delegate_calls_without_id != 1 or len(anonymous_delegate_results) != 1:
                delegate_result_unknown = True
            else:
                for result_count in anonymous_delegate_results:
                    if result_count is None:
                        delegate_result_unknown = True
                    else:
                        delegate_result_total += result_count
        elif anonymous_delegate_results:
            delegate_result_unknown = True

        if delegate_result_unknown:
            _set_coordination_component(receipt, "delegate_task", None)
        else:
            _set_coordination_component(receipt, "delegate_task", delegate_result_total)
    else:
        _clear_coordination_component(receipt, "delegate_task")

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

    mechanisms = facts.get("mechanisms") or facts.get("mechanism_segments")
    if isinstance(mechanisms, str):
        _add_mechanism_segment(receipt, mechanisms)
    elif isinstance(mechanisms, (list, tuple)):
        for segment in mechanisms:
            _add_mechanism_segment(receipt, str(segment))

    moa_failed_refs_observed = False
    explicit_coordination_summary = False
    moa = _as_mapping(facts.get("moa"))
    if moa:
        reference_count = _count_strings(moa.get("reference_models"))
        aggregator_count = 1 if str(moa.get("aggregator_model") or "").strip() else 0
        if "reference_count" in moa:
            reference_count = _int_or_zero(moa.get("reference_count"))
        if "aggregator_count" in moa:
            aggregator_count = _int_or_zero(moa.get("aggregator_count"))
        failed_count = max(0, _int_or_zero(moa.get("failed_count"))) if "failed_count" in moa else 0
        if "reference_total" in moa:
            reference_total = max(0, _int_or_zero(moa.get("reference_total")))
        else:
            reference_total = reference_count + failed_count
        reference_total = max(reference_total, reference_count + failed_count)
        observed = bool(moa.get("observed")) or reference_count > 0 or failed_count > 0 or aggregator_count > 0
        if observed:
            if failed_count > 0:
                segment = f"MoA {reference_count}/{reference_total}+{aggregator_count}"
                moa_failed_refs_observed = True
            else:
                segment = f"MoA {reference_count}+{aggregator_count}"
            _add_mechanism_segment(receipt, segment)
            _set_coordination_component(receipt, "moa", reference_count + aggregator_count)
            if receipt.route == "native":
                receipt.route = "moa"
                receipt.reason = "provider_moa"

    omo = _as_mapping(facts.get("omo"))
    if omo:
        parent_count = 1 if str(omo.get("parent_session_id") or "").strip() else 0
        descendant_count = _count_strings(omo.get("descendant_session_ids") or [])
        if "parent_count" in omo:
            parent_count = _int_or_zero(omo.get("parent_count"))
        if "descendant_count" in omo:
            descendant_count = _int_or_zero(omo.get("descendant_count"))
        observed = bool(omo.get("observed")) or parent_count > 0 or descendant_count > 0
        if observed:
            _add_mechanism_segment(receipt, f"OMO {parent_count}+{descendant_count}")
            _set_coordination_component(receipt, "omo", parent_count + descendant_count)

    coordination = _as_mapping(facts.get("coordination"))
    if coordination:
        raw_modes = coordination.get("modes") or []
        if isinstance(raw_modes, str):
            modes = {raw_modes.strip().lower()} if raw_modes.strip() else set()
        elif isinstance(raw_modes, (list, tuple, set)):
            modes = {str(mode).strip().lower() for mode in raw_modes if str(mode).strip()}
        else:
            modes = set()
        breakdown = _as_mapping(coordination.get("breakdown"))
        if "omo" in modes:
            parent_count = _int_or_zero(breakdown.get("omo_parent"))
            descendant_count = _int_or_zero(breakdown.get("omo_descendants"))
            if parent_count or descendant_count:
                _add_mechanism_segment(receipt, f"OMO {parent_count}+{descendant_count}")
                _set_coordination_component(receipt, "omo", parent_count + descendant_count)
        if "moa" in modes:
            reference_count = _int_or_zero(
                breakdown.get("moa_references", breakdown.get("reference_count"))
            )
            raw_aggregator_count = breakdown.get("moa_aggregators")
            if raw_aggregator_count is None:
                raw_aggregator_count = breakdown.get("moa_aggregator")
            if raw_aggregator_count is None:
                raw_aggregator_count = breakdown.get("aggregator_count")
            aggregator_count = _int_or_zero(raw_aggregator_count)
            if reference_count or aggregator_count:
                _set_coordination_component(receipt, "moa", reference_count + aggregator_count)
        if "agents" in coordination:
            explicit_coordination_summary = True
            _set_coordination_component(receipt, "explicit", coordination.get("agents"))
            _clear_coordination_component(receipt, "delegation")

    delegation = _as_mapping(facts.get("delegation")) or _as_mapping(facts.get("agents"))
    if delegation:
        if "observed" in delegation:
            receipt.delegation_observed = bool(delegation.get("observed"))
        raw_agents = delegation.get("agents") if "agents" in delegation else None
        raw_subagents = delegation.get("subagents") if "subagents" in delegation else None
        if "agents" in delegation:
            try:
                receipt.agents_count = int(raw_agents) if raw_agents is not None else None
            except Exception:
                receipt.agents_count = None
        if "subagents" in delegation:
            try:
                receipt.subagents_count = int(raw_subagents) if raw_subagents is not None else None
            except Exception:
                receipt.subagents_count = None
        if explicit_coordination_summary:
            _clear_coordination_component(receipt, "delegation")
        else:
            agents_count = _normalize_coordination_count(raw_agents)
            subagents_count = _normalize_coordination_count(raw_subagents)
            has_agent_count = raw_agents is not None and agents_count is not None
            has_subagent_count = raw_subagents is not None and subagents_count is not None
            observed = bool(delegation.get("observed")) or has_agent_count or has_subagent_count
            if observed:
                if has_subagent_count:
                    _set_coordination_component(receipt, "delegation", subagents_count)
                elif has_agent_count:
                    _set_coordination_component(receipt, "delegation", agents_count)
                else:
                    _set_coordination_component(receipt, "delegation", None)
            else:
                _clear_coordination_component(receipt, "delegation")

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

    if moa_failed_refs_observed and not receipt.failed and not receipt.interrupted:
        current_evidence = (receipt.evidence_status or "unknown").strip().lower()
        if current_evidence not in {"failed", "fail", "error", "errored"}:
            receipt.evidence_status = "partial"

    return receipt

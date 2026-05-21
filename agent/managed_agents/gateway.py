"""Delegation gateway for managed agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from agent.session_event_log import EventLog

from .event_log import ManagedAgentEventLog
from .permissions import DelegationPermissionError, PermissionGuard, PermissionSnapshot
from .policy import PolicyDecision, PolicyEngine
from .registry import AgentRegistry, AgentSpec, RiskLevel
from .router import RoutingDecision


class DelegationGatewayError(ValueError):
    """Raised when delegation cannot proceed."""


@dataclass(frozen=True, slots=True)
class TaskResult:
    task_id: str
    agent_id: str
    status: str
    summary: str = ""
    accepted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def _normalize_list(values: Any) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):
        return [values.strip()] if values.strip() else []
    if isinstance(values, Iterable):
        return [str(item).strip() for item in values if str(item).strip()]
    value = str(values).strip()
    return [value] if value else []


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _stringify_status(value: Any) -> str:
    value = str(value or "").strip()
    return value or "completed"


@dataclass(slots=True)
class DelegationGateway:
    registry: AgentRegistry
    policy_engine: PolicyEngine
    event_log: EventLog
    runtime: Any | None = None
    permission_guard: PermissionGuard = field(default_factory=PermissionGuard)

    def delegate(
        self,
        task_card: Any,
        *,
        routing_decision: RoutingDecision | Mapping[str, Any] | None = None,
        allowed_files: Iterable[str] | None = None,
        forbidden_actions: Iterable[str] | None = None,
        requested_tools: Iterable[str] | None = None,
        human_approved: bool = False,
        runtime: Any | None = None,
    ) -> TaskResult:
        if task_card is None:
            raise DelegationGatewayError("TaskCard is required")

        task_id = str(_get_attr(task_card, "task_id", "")).strip()
        if not task_id:
            raise DelegationGatewayError("TaskCard is missing task_id")

        session_id = str(_get_attr(task_card, "session_id", "")).strip() or "unknown"
        target_agent_id = self.permission_guard.resolve_target_agent_id(task_card, routing_decision)
        agent = self.registry.get(target_agent_id)
        risk_level = self._resolve_risk_level(task_card, routing_decision)

        policy_task = self._build_policy_task(task_card, routing_decision, target_agent_id, risk_level)
        policy_decision = self.policy_engine.evaluate(policy_task)
        managed_event_log = ManagedAgentEventLog(self.event_log)
        managed_event_log.log_policy_evaluated(
            task_id=task_id,
            session_id=session_id,
            agent_id=target_agent_id,
            risk_level=risk_level.value,
            decision=policy_decision.outcome,
            reason=str(policy_decision.winner),
            metadata={"policy_record": dict(policy_decision.record)},
        )
        if policy_decision.outcome == "deny":
            raise DelegationGatewayError(
                f"Policy denied delegation for {task_id}: {policy_decision.winner}"
            )

        try:
            snapshot = self.permission_guard.build_snapshot(
                agent,
                risk_level=risk_level,
                allowed_files=allowed_files,
                forbidden_actions=forbidden_actions,
                requested_tools=requested_tools,
                human_approved=human_approved,
                metadata={"policy_record": dict(policy_decision.record)},
            )
        except DelegationPermissionError as exc:
            managed_event_log.log_tool_permission_denied(
                task_id=task_id,
                session_id=session_id,
                agent_id=target_agent_id,
                tool_name=", ".join(_normalize_list(requested_tools)),
                reason=str(exc),
                metadata={"risk_level": risk_level.value},
            )
            raise

        execution_plan = _get_attr(task_card, "execution_plan", None)
        dispatch_mode = _get_attr(
            routing_decision,
            "mode",
            _get_attr(execution_plan, "mode", "single_agent"),
        )
        dispatch_event = self.event_log.log_dispatch_decision(
            task_id=task_id,
            session_id=session_id,
            mode=str(dispatch_mode),
            agents=[target_agent_id],
            reason=str(_get_attr(routing_decision, "reason", "")),
        )
        self.event_log.log_agent_called(
            task_id=task_id,
            session_id=session_id,
            agent_name=target_agent_id,
            prompt_preview=self._build_prompt_preview(task_card),
        )
        managed_event_log.log_task_delegated(
            task_id=task_id,
            session_id=session_id,
            from_agent="hermes",
            to_agent=target_agent_id,
            risk_level=risk_level.value,
            reason=str(_get_attr(routing_decision, "reason", "")),
            metadata={"dispatch_event_id": dispatch_event.event_id},
        )

        runtime_result = self._invoke_runtime(
            runtime or self.runtime,
            agent,
            task_card,
            {
                "allowed_files": list(snapshot.allowed_files),
                "forbidden_actions": list(snapshot.forbidden_actions),
                "requested_tools": list(snapshot.requested_tools),
                "risk_level": snapshot.risk_level,
                "human_approved": snapshot.human_approved,
                "policy_record": dict(policy_decision.record),
                "dispatch_event_id": dispatch_event.event_id,
            },
        )
        normalized = self._normalize_runtime_result(runtime_result, task_id, target_agent_id)

        self.event_log.log_agent_result(
            task_id=task_id,
            session_id=session_id,
            agent_name=target_agent_id,
            result_summary=normalized.summary,
            success=normalized.accepted,
        )
        managed_event_log.log_task_result_received(
            task_id=task_id,
            session_id=session_id,
            from_agent=target_agent_id,
            status=normalized.status,
            summary=normalized.summary,
            metadata={"accepted": normalized.accepted},
        )
        return normalized

    def _resolve_risk_level(
        self,
        task_card: Any,
        routing_decision: RoutingDecision | Mapping[str, Any] | None,
    ) -> RiskLevel:
        if routing_decision is not None:
            value = _get_attr(routing_decision, "risk_level", None)
            if value is not None:
                return RiskLevel.from_raw(value)
        value = _get_attr(task_card, "risk_level", None)
        if value is not None:
            return RiskLevel.from_raw(value)
        return RiskLevel.R0

    def _build_policy_task(
        self,
        task_card: Any,
        routing_decision: RoutingDecision | Mapping[str, Any] | None,
        target_agent_id: str,
        risk_level: RiskLevel,
    ) -> dict[str, Any]:
        intent = _get_attr(task_card, "compiled_intent", None)
        execution_plan = _get_attr(task_card, "execution_plan", None)
        action_type = _get_attr(task_card, "action_type", None)
        if action_type is None and routing_decision is not None:
            action_type = _get_attr(routing_decision, "action_type", None)
        return {
            "task_id": str(_get_attr(task_card, "task_id", "unknown")),
            "task_category": str(_get_attr(intent, "task_category", "other")),
            "risk_level": risk_level.value,
            "task_kind": str(_get_attr(intent, "real_task", "")),
            "owner_agent": target_agent_id,
            "route_mode": str(_get_attr(routing_decision, "mode", _get_attr(execution_plan, "mode", "single_agent"))),
            "action_type": action_type,
            "user_override": _get_attr(task_card, "user_override", None),
        }

    def _build_prompt_preview(self, task_card: Any) -> str:
        raw_request = str(_get_attr(task_card, "raw_user_request", "")).strip()
        if raw_request:
            return raw_request[:200]
        intent = _get_attr(task_card, "compiled_intent", None)
        return str(_get_attr(intent, "real_task", ""))[:200]

    def _invoke_runtime(self, runtime: Any, agent: AgentSpec, task_card: Any, context: Mapping[str, Any]) -> Any:
        if runtime is None:
            raise DelegationGatewayError("Delegation runtime is not configured")

        if callable(runtime):
            return runtime(agent, task_card, context)

        invoke = getattr(runtime, "invoke", None)
        if callable(invoke):
            try:
                return invoke(agent, task_card, context)
            except TypeError:
                return invoke(agent=agent, task_card=task_card, context=context)

        raise DelegationGatewayError("Delegation runtime must be callable or expose invoke()")

    def _normalize_runtime_result(
        self,
        runtime_result: Any,
        task_id: str,
        agent_id: str,
    ) -> TaskResult:
        if isinstance(runtime_result, TaskResult):
            return runtime_result

        if isinstance(runtime_result, Mapping):
            payload = dict(runtime_result)
        else:
            payload = {
                "status": _get_attr(runtime_result, "status", "completed"),
                "summary": _get_attr(runtime_result, "summary", ""),
                "accepted": _get_attr(runtime_result, "accepted", None),
                "metadata": _get_attr(runtime_result, "metadata", {}),
            }

        status = _stringify_status(payload.get("status"))
        summary = str(payload.get("summary") or payload.get("result_summary") or "")
        accepted = payload.get("accepted")
        if accepted is None:
            accepted = status in {"completed", "done", "success", "ok"}
        metadata = dict(payload.get("metadata") or {})
        metadata.setdefault("runtime_status", status)
        return TaskResult(
            task_id=task_id,
            agent_id=agent_id,
            status=status,
            summary=summary,
            accepted=bool(accepted),
            metadata=metadata,
        )

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from agent.managed_agents.gateway import (
    DelegationGateway,
    DelegationGatewayError,
    TaskResult,
)
from agent.managed_agents.permissions import DelegationPermissionError
from agent.managed_agents.policy import PolicyEngine
from agent.managed_agents.registry import (
    AgentRegistry,
    AgentSpec,
    AgentStatus,
    PermissionMode,
    RiskLevel,
)
from agent.managed_agents.router import RoutingDecision
from agent.session_event_log import EventLog
from agent.task_card import ExecutionPlan, TaskCard, CompiledIntent


def _write_policy(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "policy.yaml"
    path.write_text(textwrap.dedent(body).strip() + "\n", encoding="utf-8")
    return path


class FakeRuntime:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def invoke(self, agent, task_card, context):
        self.calls.append((agent, task_card, context))
        return self.result


def _make_gateway(tmp_path: Path, *, registry: AgentRegistry | None = None, policy: PolicyEngine | None = None) -> DelegationGateway:
    registry = registry or AgentRegistry(
        version="2026-05-21",
        agents={
            "claude": AgentSpec(
                agent_id="claude",
                name="Claude Code",
                role="lead_implementer",
                tools=("file", "terminal"),
                permission=PermissionMode.ASK,
                can_delegate=False,
                capabilities=("code_edit",),
                risk_allowed=frozenset({RiskLevel.R0, RiskLevel.R1, RiskLevel.R2, RiskLevel.R3}),
                status=AgentStatus.ACTIVE,
            ),
            "codex": AgentSpec(
                agent_id="codex",
                name="Codex",
                role="principal_engineer",
                tools=("file",),
                permission=PermissionMode.READ_ONLY,
                can_delegate=False,
                capabilities=("code_review",),
                risk_allowed=frozenset({RiskLevel.R0, RiskLevel.R1, RiskLevel.R2, RiskLevel.R3, RiskLevel.R4}),
                status=AgentStatus.ACTIVE,
            ),
        },
    )

    policy = policy or PolicyEngine(
        version="2026-05-21",
        priority_order=(
            "safety",
            "user_explicit_instruction",
            "soul_global_policy",
            "managed_agents_policy",
            "router_policy",
            "skill_policy",
            "agent_preference",
        ),
        rules=(),
    )
    return DelegationGateway(
        registry=registry,
        policy_engine=policy,
        event_log=EventLog(db_path=tmp_path / "events.db"),
        runtime=None,
    )


def _make_task_card(*, task_id: str = "task-1", session_id: str = "sess-1", task_category: str = "feature", agents: list[str] | None = None) -> TaskCard:
    return TaskCard(
        task_id=task_id,
        session_id=session_id,
        raw_user_request="build managed agents",
        compiled_intent=CompiledIntent(real_task="build managed agents", task_category=task_category),
        execution_plan=ExecutionPlan(mode="single_agent", agents=agents or ["claude"], delegation_reason="route"),
        status="pending",
    )


def test_delegate_success_logs_dispatch_call_and_result(tmp_path):
    gateway = _make_gateway(tmp_path)
    runtime = FakeRuntime({"status": "completed", "summary": "done", "accepted": True})
    task_card = _make_task_card()
    decision = RoutingDecision(
        task_id="task-1",
        mode="single_agent",
        agents=["claude"],
        reason="default_feature_implementation",
        risk_level="R2",
    )

    result = gateway.delegate(task_card, routing_decision=decision, runtime=runtime)

    assert isinstance(result, TaskResult)
    assert result.agent_id == "claude"
    assert result.status == "completed"
    assert result.accepted is True
    assert runtime.calls[0][2]["risk_level"] == "R2"
    assert runtime.calls[0][2]["policy_record"]["outcome"] == "allow"
    events = gateway.event_log.get_events_for_task("task-1")
    assert [event["type"] for event in events] == [
        "policy_evaluated",
        "dispatch_decision",
        "agent_called",
        "task_delegated",
        "agent_result",
        "task_result_received",
    ]


def test_missing_task_card_rejected(tmp_path):
    gateway = _make_gateway(tmp_path)

    with pytest.raises(DelegationGatewayError, match="TaskCard is required"):
        gateway.delegate(None, runtime=FakeRuntime({}))


def test_policy_deny_blocks_delegation(tmp_path):
    policy = PolicyEngine(
        version="2026-05-21",
        priority_order=(
            "safety",
            "user_explicit_instruction",
            "soul_global_policy",
            "managed_agents_policy",
            "router_policy",
            "skill_policy",
            "agent_preference",
        ),
        rules=(
            {
                "id": "deny_blocked_task",
                "when": {"task_category": "blocked"},
                "decision": "deny",
                "reason": "policy",
            },
        ),
    )
    gateway = _make_gateway(tmp_path, policy=policy)

    task_card = _make_task_card(task_id="task-2", task_category="blocked")
    task_card.action_type = "delete_file"
    decision = RoutingDecision(task_id="task-2", mode="single_agent", agents=["claude"], reason="blocked", risk_level="R1")

    with pytest.raises(DelegationGatewayError, match="Policy denied delegation"):
        gateway.delegate(task_card, routing_decision=decision, runtime=FakeRuntime({}))


def test_r4_requires_human_approval(tmp_path):
    gateway = _make_gateway(tmp_path)
    task_card = _make_task_card(task_id="task-3")
    decision = RoutingDecision(task_id="task-3", mode="single_agent", agents=["claude"], reason="risk", risk_level="R4")

    with pytest.raises(DelegationPermissionError, match="requires human approval"):
        gateway.delegate(task_card, routing_decision=decision, runtime=FakeRuntime({}))

    events = gateway.event_log.get_events_for_task("task-3")
    assert [event["type"] for event in events] == [
        "policy_evaluated",
        "tool_permission_denied",
    ]


def test_read_only_agent_cannot_use_write_tools(tmp_path):
    registry = AgentRegistry(
        version="2026-05-21",
        agents={
            "codex": AgentSpec(
                agent_id="codex",
                name="Codex",
                role="principal_engineer",
                tools=("terminal",),
                permission=PermissionMode.READ_ONLY,
                can_delegate=False,
                capabilities=("code_review",),
                risk_allowed=frozenset({RiskLevel.R0, RiskLevel.R1, RiskLevel.R2}),
                status=AgentStatus.ACTIVE,
            )
        },
    )
    gateway = _make_gateway(tmp_path, registry=registry)
    task_card = _make_task_card(task_id="task-4", agents=["codex"])
    decision = RoutingDecision(task_id="task-4", mode="single_agent", agents=["codex"], reason="review", risk_level="R1")

    with pytest.raises(DelegationPermissionError, match="cannot use write-capable tools"):
        gateway.delegate(
            task_card,
            routing_decision=decision,
            requested_tools=["terminal"],
            runtime=FakeRuntime({}),
        )


def test_can_delegate_true_is_rejected(tmp_path):
    registry = AgentRegistry(
        version="2026-05-21",
        agents={
            "claude": AgentSpec(
                agent_id="claude",
                name="Claude Code",
                role="lead_implementer",
                tools=("file",),
                permission=PermissionMode.ASK,
                can_delegate=True,
                capabilities=("code_edit",),
                risk_allowed=frozenset({RiskLevel.R0, RiskLevel.R1, RiskLevel.R2}),
                status=AgentStatus.ACTIVE,
            )
        },
    )
    gateway = _make_gateway(tmp_path, registry=registry)
    task_card = _make_task_card(task_id="task-5", agents=["claude"])
    decision = RoutingDecision(task_id="task-5", mode="single_agent", agents=["claude"], reason="delegate", risk_level="R1")

    with pytest.raises(DelegationPermissionError, match="cannot delegate further"):
        gateway.delegate(task_card, routing_decision=decision, runtime=FakeRuntime({}))


def test_mapping_task_card_and_execution_plan_supported(tmp_path):
    gateway = _make_gateway(tmp_path)
    runtime = FakeRuntime({"status": "completed", "summary": "done", "accepted": True})
    task_card = {
        "task_id": "task-6",
        "session_id": "sess-1",
        "raw_user_request": "implement feature",
        "compiled_intent": {"real_task": "implement feature", "task_category": "feature"},
        "execution_plan": {"mode": "single_agent", "agents": ["claude"]},
        "risk_level": "R1",
    }

    result = gateway.delegate(task_card, runtime=runtime)

    assert result.agent_id == "claude"
    assert result.accepted is True
    events = gateway.event_log.get_events_for_task("task-6")
    dispatch_event = next(event for event in events if event["type"] == "dispatch_decision")
    assert dispatch_event["payload"]["mode"] == "single_agent"

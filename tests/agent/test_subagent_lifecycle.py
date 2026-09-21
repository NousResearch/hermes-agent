"""Contract tests for the public plugin subagent lifecycle API."""

import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.subagent_lifecycle import (
    SubagentLaunchRequest,
    SubagentLifecycleError,
    SubagentLifecycleService,
    SubagentState,
    bind_subagent_parent,
    get_active_subagent_parent,
)
from agent.worker_contract import (
    ContextProfile,
    ExecutionEnvelope,
    JobContract,
    MemoryPolicy,
    PrivacyPolicy,
    RoleSeparation,
    WorkforceGovernance,
    WorkerConstitution,
)


class FakeChild:
    def __init__(self, ident="sa-test"):
        self._subagent_id = ident
        self._delegate_role = "leaf"
        self._delegate_depth = 1
        self.provider = "test"
        self.model = "test-model"
        self.interrupted = False
        self.interrupt_kind = None
        self.interrupt_message = None
        self.tool_reason = None

    def interrupt(self, _reason):
        self.interrupted = True
        self.interrupt_kind = "soft"

    def hard_interrupt(self, reason, *, tool_reason=None):
        self.interrupted = True
        self.interrupt_kind = "hard"
        self.interrupt_message = reason
        self.tool_reason = tool_reason


def _active_workforce_contracts():
    now = datetime.now(timezone.utc)
    stamp = lambda value: value.isoformat().replace("+00:00", "Z")
    constitution = WorkerConstitution(
        profile="researcher",
        values=("truthful evidence", "bounded autonomy"),
        authority="research-only",
        forbidden_actions=("submit external side effects",),
        required_evidence=("research", "diagnostic"),
        escalation_path="operator-review",
    )
    contract = JobContract(
        name="stellaris-research-assignment",
        worker_profile="researcher",
        job="analyze the assigned research question",
        scope=("current task", "approved sources"),
        authority="research-only",
        obligations=("separate observations from conclusions",),
        granted_at=stamp(now - timedelta(seconds=1)),
        expires_at=stamp(now + timedelta(minutes=5)),
    )
    return constitution, contract


@pytest.fixture
def lifecycle(monkeypatch):
    parent = SimpleNamespace(session_id="parent-1", enabled_toolsets=["file"])
    counter = iter(range(1000))

    def build(**_kwargs):
        return FakeChild(f"sa-{next(counter)}")

    def run(_index, _goal, child, _parent):
        for _ in range(20):
            if child.interrupted:
                return {
                    "status": "interrupted",
                    "summary": None,
                    "api_calls": 0,
                    "duration_seconds": 0,
                }
            time.sleep(0.002)
        return {
            "status": "completed",
            "summary": "safe summary",
            "api_calls": 1,
            "duration_seconds": 0.01,
        }

    monkeypatch.setattr("tools.delegate_tool._build_child_agent", build)
    monkeypatch.setattr("tools.delegate_tool._run_single_child", run)
    return SubagentLifecycleService(lambda: parent)


def test_cancel_is_cooperative_and_forged_handle_is_unknown(lifecycle):
    handle = lifecycle.launch(SubagentLaunchRequest(goal="x"))
    assert lifecycle.cancel(handle, reason="test").accepted
    terminal = lifecycle.wait(handle, timeout_seconds=1)
    assert terminal.state is SubagentState.CANCELLED
    forged = handle.__class__(**{**handle.to_dict(), "capability": "forged"})
    assert lifecycle.status(forged).state is SubagentState.UNKNOWN
    assert lifecycle.result(forged).error_classification == "UNKNOWN_HANDLE"
    other_parent = SimpleNamespace(session_id="different-parent")
    other_service = SubagentLifecycleService(lambda: other_parent)
    assert other_service.status(handle).state is SubagentState.UNKNOWN


def test_cancel_uses_explicit_hard_interrupt(lifecycle):
    handle = lifecycle.launch(SubagentLaunchRequest(goal="x"))
    record = lifecycle._record(handle)
    assert record is not None and record.agent is not None

    assert lifecycle.cancel(handle, reason="explicit user cancel").accepted

    assert record.agent.interrupt_kind == "hard"
    assert "explicit user cancel" in record.agent.interrupt_message
    assert record.agent.tool_reason == "subagent cancellation requested"
    lifecycle.wait(handle, timeout_seconds=1)


def test_launch_propagates_active_workforce_contracts_to_handle_and_child(lifecycle):
    constitution, contract = _active_workforce_contracts()
    request = SubagentLaunchRequest(
        goal="work under a governed assignment",
        constitution=constitution,
        job_contract=contract,
    )

    handle = lifecycle.launch(request)
    record = lifecycle._record(handle)

    assert handle.worker_profile == "researcher"
    assert handle.constitution == constitution
    assert handle.job_contract == contract
    assert record is not None
    assert record.agent._worker_constitution == constitution
    assert record.agent._job_contract == contract
    restored = handle.from_dict(handle.to_dict())
    assert restored.constitution == constitution
    assert restored.job_contract == contract
    lifecycle.wait(handle, timeout_seconds=1)


def test_workforce_context_preserves_existing_context_and_adds_rules():
    constitution, contract = _active_workforce_contracts()
    request = SubagentLaunchRequest(
        goal="use governed context",
        context="Use the repository's existing conventions.",
        constitution=constitution,
        job_contract=contract,
    )

    rendered = SubagentLifecycleService._workforce_context(request)

    assert rendered.startswith("Use the repository's existing conventions.")
    assert "GOVERNED WORKFORCE ASSIGNMENT:" in rendered
    assert '"worker_profile": "researcher"' in rendered


def test_launch_rejects_contract_profile_mismatch(lifecycle):
    constitution, contract = _active_workforce_contracts()
    mismatched = JobContract(**{
        **contract.__dict__,
        "worker_profile": "operator",
    })

    with pytest.raises(SubagentLifecycleError, match="worker_profile"):
        lifecycle.launch(
            SubagentLaunchRequest(
                goal="reject a mismatched assignment",
                constitution=constitution,
                job_contract=mismatched,
            )
        )


def test_launch_rejects_expired_workforce_contract(lifecycle):
    constitution, contract = _active_workforce_contracts()
    expired = JobContract(**{
        **contract.__dict__,
        "granted_at": "2020-01-01T00:00:00Z",
        "expires_at": "2020-01-01T00:01:00Z",
    })

    with pytest.raises(SubagentLifecycleError, match="active"):
        lifecycle.launch(
            SubagentLaunchRequest(
                goal="reject an expired assignment",
                constitution=constitution,
                job_contract=expired,
            )
        )


def test_launch_propagates_governance_controls_and_roundtrips_handle(lifecycle):
    governance = WorkforceGovernance(
        memory=MemoryPolicy(purpose="task-local evidence"),
        execution=ExecutionEnvelope(lane="simulation"),
        roles=RoleSeparation(planner_id="planner", critic_id="critic"),
        context=ContextProfile(
            profile="researcher",
            assumptions=("source is current",),
            assumption_warnings=("verify source freshness",),
        ),
        privacy=PrivacyPolicy(allowed_scopes=("task",)),
    )
    handle = lifecycle.launch(
        SubagentLaunchRequest(goal="governed research", governance=governance)
    )
    record = lifecycle._record(handle)

    assert handle.governance == governance
    assert record is not None
    assert record.agent._workforce_governance == governance
    assert "GOVERNED WORKFORCE ASSIGNMENT:" in lifecycle._workforce_context(
        SubagentLaunchRequest(goal="governed research", governance=governance)
    )
    assert handle.from_dict(handle.to_dict()).governance == governance
    lifecycle.wait(handle, timeout_seconds=1)


def test_launch_rejects_unsafe_governance_before_child_creation(lifecycle):
    governance = WorkforceGovernance(
        execution=ExecutionEnvelope(lane="production"),
        roles=RoleSeparation(planner_id="planner", executor_id="executor"),
    )
    with pytest.raises(SubagentLifecycleError, match="critic"):
        lifecycle.launch(
            SubagentLaunchRequest(goal="must be reviewed", governance=governance)
        )


def test_public_lifecycle_runs_host_aggregation(monkeypatch):
    memory = Mock()
    parent = SimpleNamespace(
        session_id="parent-aggregate",
        enabled_toolsets=["file"],
        _memory_manager=memory,
        _current_turn_id="turn-1",
        session_estimated_cost_usd=1.0,
        session_cost_source="none",
        session_cost_status="unknown",
    )
    child = FakeChild("sa-aggregate")
    child.session_id = "child-session"
    hook = Mock()

    monkeypatch.setattr(
        "tools.delegate_tool._build_child_agent", lambda **_kwargs: child
    )
    monkeypatch.setattr(
        "tools.delegate_tool._run_single_child",
        lambda *_args, **_kwargs: {
            "task_index": 0,
            "status": "completed",
            "summary": "aggregated",
            "api_calls": 1,
            "duration_seconds": 0.25,
            "_child_role": "leaf",
            "_child_cost_usd": 2.5,
        },
    )
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", hook)

    service = SubagentLifecycleService(lambda: parent)
    handle = service.launch(SubagentLaunchRequest(goal="aggregate me"))
    assert service.wait(handle, timeout_seconds=1).state is SubagentState.SUCCEEDED

    memory.on_delegation.assert_called_once_with(
        task="aggregate me", result="aggregated", child_session_id="child-session"
    )
    hook.assert_called_once_with(
        "subagent_stop",
        parent_session_id="parent-aggregate",
        parent_turn_id="turn-1",
        child_session_id="child-session",
        child_role="leaf",
        child_summary="aggregated",
        child_status="completed",
        # Redacted tool history rides the shared finalization pipeline
        # (#62011/#72403); empty here because the fabricated result carries
        # no tool_trace.
        tool_call_history=[],
        duration_ms=250,
    )
    assert parent.session_estimated_cost_usd == 3.5
    assert parent.session_cost_source == "subagent"
    assert parent.session_cost_status == "estimated"


def test_agent_turn_binds_and_clears_lifecycle_parent(monkeypatch):
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    observed = []

    def run_conversation(parent, *_args, **_kwargs):
        observed.append(get_active_subagent_parent())
        return {"final_response": "ok"}

    monkeypatch.setattr("agent.conversation_loop.run_conversation", run_conversation)

    assert agent.run_conversation("hello") == {"final_response": "ok"}
    assert observed == [agent]
    assert get_active_subagent_parent() is None

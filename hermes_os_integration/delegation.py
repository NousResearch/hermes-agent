"""Delegation prototype for routing Hermes OS tasks to worker agents."""

from dataclasses import dataclass
from typing import Callable, List

from .contracts import AgentRequest, AgentResponse, DelegationRequest
from .runtime_policies import RuntimePolicy, evaluate_runtime_policy
from .registry import get_agent, select_agent_kind
from .wrapper import RuntimeWrapper


@dataclass(frozen=True)
class DelegationResult:
    request: AgentRequest
    response: AgentResponse
    persisted_outputs: List[str]


class DelegationEngine:
    def __init__(self, wrapper=None, persist_result=None, policy: RuntimePolicy | None = None, persist_audit=None):
        self.wrapper = wrapper or RuntimeWrapper()
        self.persist_result = persist_result or (lambda response: ["hermes-os://dry-run/result/%s" % response.task_id])
        self.policy = policy or RuntimePolicy()
        self.persist_audit = persist_audit or (lambda audit: None)

    def delegate(self, request):
        agent_kind = select_agent_kind(request.task_type)
        agent = get_agent(agent_kind)
        runtime_provider = agent.runtime_provider if agent else "official-hermes-agent"
        agent_request = AgentRequest(
            task_id=request.task_id,
            project_id=request.project_id,
            agent_kind=agent_kind,
            prompt=request.prompt,
            working_directory=request.working_directory,
            runtime_provider="dry-run" if request.dry_run else runtime_provider,
            dry_run=request.dry_run or not request.opt_in_runtime,
        )
        action = "write" if request.task_type in {"coding", "implementation", "deployment"} else "read"
        decision = evaluate_runtime_policy(
            action=action,
            estimated_cost_usd=0.0,
            retry_count=0,
            approved=False,
            policy=self.policy,
        )
        self.persist_audit(decision.audit)
        if not decision.allowed:
            response = AgentResponse(
                task_id=request.task_id,
                status="blocked",
                output="Runtime delegation blocked by policy.",
                errors=[],
            )
            return DelegationResult(
                request=agent_request,
                response=response,
                persisted_outputs=self.persist_result(response),
            )
        response = self.wrapper.run(agent_request)
        persisted = self.persist_result(response)
        return DelegationResult(
            request=agent_request,
            response=response,
            persisted_outputs=persisted,
        )


def delegate_task(payload, wrapper=None):
    request = DelegationRequest(
        task_id=str(payload["task_id"]),
        project_id=str(payload["project_id"]),
        task_type=str(payload.get("task_type", "research")),
        prompt=str(payload["prompt"]),
        working_directory=str(payload["working_directory"]),
        opt_in_runtime=bool(payload.get("opt_in_runtime", False)),
        dry_run=bool(payload.get("dry_run", True)),
    )
    return DelegationEngine(wrapper=wrapper).delegate(request)

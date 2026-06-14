"""Agent registry and task-type mapping for Hermes OS delegation."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class AgentDefinition:
    kind: str
    purpose: str
    inputs: List[str]
    outputs: List[str]
    allowed_tools: List[str]
    runtime_provider: str


AGENT_REGISTRY: Dict[str, AgentDefinition] = {
    "research": AgentDefinition(
        kind="research",
        purpose="Gather evidence and synthesize research notes.",
        inputs=["prompt", "project_context", "sources"],
        outputs=["research_summary", "evidence", "open_questions"],
        allowed_tools=["web", "filesystem", "docs"],
        runtime_provider="official-hermes-agent",
    ),
    "coding": AgentDefinition(
        kind="coding",
        purpose="Implement scoped code changes.",
        inputs=["task", "repo_context", "constraints"],
        outputs=["patch", "test_results", "summary"],
        allowed_tools=["filesystem", "terminal", "git"],
        runtime_provider="official-hermes-agent",
    ),
    "testing": AgentDefinition(
        kind="testing",
        purpose="Run and interpret validation commands.",
        inputs=["test_plan", "repo_context"],
        outputs=["test_results", "failure_analysis"],
        allowed_tools=["terminal", "filesystem"],
        runtime_provider="official-hermes-agent",
    ),
    "review": AgentDefinition(
        kind="review",
        purpose="Review code, docs, or execution artifacts.",
        inputs=["diff", "requirements"],
        outputs=["findings", "risks", "recommendations"],
        allowed_tools=["filesystem", "git"],
        runtime_provider="official-hermes-agent",
    ),
    "documentation": AgentDefinition(
        kind="documentation",
        purpose="Create or update project documentation.",
        inputs=["topic", "source_material"],
        outputs=["docs_patch", "summary"],
        allowed_tools=["filesystem", "docs"],
        runtime_provider="official-hermes-agent",
    ),
    "deployment": AgentDefinition(
        kind="deployment",
        purpose="Prepare deployment plans and runbooks.",
        inputs=["environment", "release_context"],
        outputs=["runbook", "checks", "risks"],
        allowed_tools=["filesystem", "terminal"],
        runtime_provider="official-hermes-agent",
    ),
    "kalshi-research": AgentDefinition(
        kind="kalshi-research",
        purpose="Research prediction-market buckets with evidence trails.",
        inputs=["bucket", "market_context"],
        outputs=["evidence", "hypotheses", "portfolio_notes"],
        allowed_tools=["web", "research", "filesystem"],
        runtime_provider="official-hermes-agent",
    ),
    "experiment": AgentDefinition(
        kind="experiment",
        purpose="Run controlled experiments and capture results.",
        inputs=["experiment_plan", "metrics"],
        outputs=["observations", "result_summary"],
        allowed_tools=["terminal", "filesystem"],
        runtime_provider="deepseek-direct",
    ),
}


TASK_TYPE_TO_AGENT = {
    "research": "research",
    "coding": "coding",
    "test": "testing",
    "testing": "testing",
    "review": "review",
    "docs": "documentation",
    "documentation": "documentation",
    "deployment": "deployment",
    "kalshi": "kalshi-research",
    "experiment": "experiment",
}


def get_agent(kind):
    return AGENT_REGISTRY.get(kind)


def select_agent_kind(task_type):
    return TASK_TYPE_TO_AGENT.get(str(task_type).lower(), "research")

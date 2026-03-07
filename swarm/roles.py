from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AgentRole:
    name: str
    description: str
    default_model: str
    system_prompt_suffix: str = ""
    tools_allowed: list[str] = field(default_factory=list)
    max_tokens: int = 4096
    temperature: float = 0.0


PLANNER = AgentRole(
    name="planner",
    description="Decomposes tasks into subtasks, plans execution order, assigns roles.",
    default_model="claude-opus",
    system_prompt_suffix=(
        "You are a planning agent. Break the user's goal into concrete, "
        "dependency-ordered subtasks. Output a structured task graph. "
        "Do not execute tasks yourself."
    ),
    tools_allowed=["task_create", "task_list", "task_update"],
    max_tokens=8192,
    temperature=0.2,
)

EXECUTOR = AgentRole(
    name="executor",
    description="Runs tools, writes code, does the work.",
    default_model="gemini-flash",
    system_prompt_suffix=(
        "You are an executor agent. Carry out the assigned task using "
        "available tools. Be precise and produce artifacts."
    ),
    tools_allowed=[
        "bash", "read", "write", "edit", "glob", "grep",
        "web_fetch", "web_search",
    ],
    max_tokens=4096,
    temperature=0.0,
)

CRITIC = AgentRole(
    name="critic",
    description="Reviews executor output, identifies issues, suggests improvements.",
    default_model="claude-opus",
    system_prompt_suffix=(
        "You are a critic agent. Review the provided output for correctness, "
        "completeness, style, and edge cases. List concrete issues and fixes. "
        "Do not rewrite the output yourself."
    ),
    tools_allowed=["read", "glob", "grep"],
    max_tokens=4096,
    temperature=0.3,
)

VERIFIER = AgentRole(
    name="verifier",
    description="Scores and validates results against acceptance criteria.",
    default_model="claude-sonnet",
    system_prompt_suffix=(
        "You are a verifier agent. Check whether the output satisfies the "
        "acceptance criteria. Return a pass/fail verdict with justification "
        "and a confidence score 0-1."
    ),
    tools_allowed=["read", "bash", "glob", "grep"],
    max_tokens=2048,
    temperature=0.0,
)

MERGER = AgentRole(
    name="merger",
    description="Resolves conflicts and merges multi-agent outputs.",
    default_model="claude-sonnet",
    system_prompt_suffix=(
        "You are a merger agent. Given multiple outputs that may conflict, "
        "produce a single coherent result that preserves the best parts of each."
    ),
    tools_allowed=["read", "write", "edit"],
    max_tokens=8192,
    temperature=0.1,
)

RESEARCHER = AgentRole(
    name="researcher",
    description="Web search, doc reading, information gathering.",
    default_model="gemini-flash",
    system_prompt_suffix=(
        "You are a research agent. Gather information relevant to the query "
        "using search and document reading. Summarise findings with sources."
    ),
    tools_allowed=["web_search", "web_fetch", "read", "glob", "grep"],
    max_tokens=4096,
    temperature=0.1,
)

_ROLES: dict[str, AgentRole] = {
    r.name: r
    for r in [PLANNER, EXECUTOR, CRITIC, VERIFIER, MERGER, RESEARCHER]
}


def get_role(name: str) -> AgentRole:
    role = _ROLES.get(name)
    if role is None:
        raise KeyError(f"Unknown role: {name!r}. Available: {list(_ROLES)}")
    return role


def list_roles() -> list[AgentRole]:
    return list(_ROLES.values())

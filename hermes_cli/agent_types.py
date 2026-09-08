"""Predefined task-specific agent configuration.

This module owns the static schema and validation helpers for the configured
agents that higher-level session/orchestrator code can invoke.  It deliberately
keeps the allowed set small: users may enable/disable or tune these predefined
categories, but arbitrary new task-agent identifiers are rejected until the
product explicitly supports them.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping


PREDEFINED_TASK_AGENT_IDS = frozenset({
    "research",
    "implementation",
    "review",
    "documentation",
})

SUPPORTED_RUNTIME_KEYS = frozenset({
    # Model/provider routing mirrors delegation.* and normal runtime provider
    # resolution for OpenAI-compatible and native providers.
    "model",
    "provider",
    "base_url",
    "api_key",
    "api_mode",
    "reasoning_effort",
    # Runtime controls already supported by the agent/delegation path.
    "max_iterations",
    "timeout_seconds",
    "max_summary_chars",
    "enabled_toolsets",
    "disabled_toolsets",
    "skills",
})

PREDEFINED_TASK_AGENT_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "id": "research",
        "enabled": True,
        "purpose": "Collect and synthesize source-backed information before implementation or decisions.",
        "invocation": {
            "entrypoint": "delegate_task",
            "task_category": "research",
            "parameters": {
                "required": ["prompt"],
                "optional": ["context", "limit", "sources", "deliverable"],
            },
        },
        "runtime": {
            "enabled_toolsets": ["web", "file", "terminal", "skills"],
            "skills": ["deep-research"],
            "model": "",
            "provider": "",
            "base_url": "",
            "api_key": "",
            "api_mode": "",
            "reasoning_effort": "",
            "max_iterations": 50,
            "timeout_seconds": 0,
            "max_summary_chars": 24000,
        },
    },
    {
        "id": "implementation",
        "enabled": True,
        "purpose": "Build or modify the requested artifact using repository tools and verification checks.",
        "invocation": {
            "entrypoint": "delegate_task",
            "task_category": "implementation",
            "parameters": {
                "required": ["prompt"],
                "optional": ["context", "workdir", "tests", "constraints"],
            },
        },
        "runtime": {
            "enabled_toolsets": ["file", "terminal", "search", "skills", "todo"],
            "skills": ["test-driven-development"],
            "model": "",
            "provider": "",
            "base_url": "",
            "api_key": "",
            "api_mode": "",
            "reasoning_effort": "",
            "max_iterations": 50,
            "timeout_seconds": 0,
            "max_summary_chars": 24000,
        },
    },
    {
        "id": "review",
        "enabled": True,
        "purpose": "Evaluate completed work for correctness, regressions, security, and readiness before handoff.",
        "invocation": {
            "entrypoint": "delegate_task",
            "task_category": "review",
            "parameters": {
                "required": ["prompt"],
                "optional": ["context", "diff", "tests", "risk_focus"],
            },
        },
        "runtime": {
            "enabled_toolsets": ["file", "terminal", "search", "skills"],
            "skills": ["github-code-review", "requesting-code-review"],
            "model": "",
            "provider": "",
            "base_url": "",
            "api_key": "",
            "api_mode": "",
            "reasoning_effort": "",
            "max_iterations": 50,
            "timeout_seconds": 0,
            "max_summary_chars": 24000,
        },
    },
    {
        "id": "documentation",
        "enabled": True,
        "purpose": "Produce user-facing or integration documentation from the implemented behavior and verified examples.",
        "invocation": {
            "entrypoint": "delegate_task",
            "task_category": "documentation",
            "parameters": {
                "required": ["prompt"],
                "optional": ["context", "audience", "format", "examples"],
            },
        },
        "runtime": {
            "enabled_toolsets": ["file", "search", "skills"],
            "skills": [],
            "model": "",
            "provider": "",
            "base_url": "",
            "api_key": "",
            "api_mode": "",
            "reasoning_effort": "",
            "max_iterations": 50,
            "timeout_seconds": 0,
            "max_summary_chars": 24000,
        },
    },
]


@dataclass(frozen=True)
class TaskAgentDefinition:
    """Resolved task-agent definition consumed by invocation integrations."""

    id: str
    purpose: str
    enabled: bool
    invocation: Dict[str, Any]
    runtime: Dict[str, Any]


@dataclass(frozen=True)
class TaskAgentConfigIssue:
    """A schema validation issue for task_agents configuration."""

    severity: str
    message: str
    hint: str


class TaskAgentConfigError(ValueError):
    """Raised when task agent definitions cannot be loaded safely."""

    def __init__(self, issues: Iterable[TaskAgentConfigIssue]):
        self.issues = list(issues)
        super().__init__("; ".join(issue.message for issue in self.issues))


def default_task_agents_config() -> Dict[str, Any]:
    """Return a fresh default task_agents config section."""

    return {"definitions": copy.deepcopy(PREDEFINED_TASK_AGENT_DEFINITIONS)}


def validate_task_agents_config(config: Mapping[str, Any]) -> List[TaskAgentConfigIssue]:
    """Validate the task_agents section in a loaded config mapping."""

    issues: List[TaskAgentConfigIssue] = []
    section = config.get("task_agents")
    if section is None:
        return issues
    if not isinstance(section, Mapping):
        return [TaskAgentConfigIssue(
            "error",
            f"task_agents should be a dict, got {type(section).__name__}",
            "Use task_agents.definitions as a list of predefined agent definitions.",
        )]

    definitions = section.get("definitions")
    if definitions is None:
        issues.append(TaskAgentConfigIssue(
            "error",
            "task_agents.definitions is required",
            "Add a definitions list containing only approved predefined task-agent ids.",
        ))
        return issues
    if not isinstance(definitions, list):
        issues.append(TaskAgentConfigIssue(
            "error",
            f"task_agents.definitions should be a list, got {type(definitions).__name__}",
            "Change to: task_agents:\n  definitions:\n    - id: research",
        ))
        return issues

    seen_ids: Dict[str, int] = {}
    seen_purposes: Dict[str, str] = {}
    for index, entry in enumerate(definitions):
        prefix = f"task_agents.definitions[{index}]"
        if not isinstance(entry, Mapping):
            issues.append(TaskAgentConfigIssue(
                "error",
                f"{prefix} should be a dict, got {type(entry).__name__}",
                "Each definition needs id, purpose, enabled, invocation, and optional runtime.",
            ))
            continue

        agent_id = entry.get("id")
        if not isinstance(agent_id, str) or not agent_id.strip():
            issues.append(TaskAgentConfigIssue("error", f"{prefix}.id must be a non-empty string", "Use one of: " + ", ".join(sorted(PREDEFINED_TASK_AGENT_IDS))))
            continue
        if agent_id not in PREDEFINED_TASK_AGENT_IDS:
            issues.append(TaskAgentConfigIssue(
                "error",
                f"Unknown task_agents id '{agent_id}'",
                "Only predefined task-agent ids are supported: " + ", ".join(sorted(PREDEFINED_TASK_AGENT_IDS)),
            ))
        if agent_id in seen_ids:
            issues.append(TaskAgentConfigIssue(
                "error",
                f"Duplicate task_agents id '{agent_id}'",
                f"Keep exactly one definition for '{agent_id}'. First seen at index {seen_ids[agent_id]}.",
            ))
        else:
            seen_ids[agent_id] = index

        if not isinstance(entry.get("enabled"), bool):
            issues.append(TaskAgentConfigIssue(
                "error",
                f"{prefix}.enabled must be a boolean",
                "Set enabled to true or false.",
            ))

        purpose = entry.get("purpose")
        if not isinstance(purpose, str) or not purpose.strip():
            issues.append(TaskAgentConfigIssue(
                "error",
                f"{prefix}.purpose must be a non-empty string",
                "Describe the distinct job this predefined agent performs.",
            ))
        else:
            normalized_purpose = " ".join(purpose.lower().split())
            if normalized_purpose in seen_purposes:
                issues.append(TaskAgentConfigIssue(
                    "error",
                    f"task_agents '{seen_purposes[normalized_purpose]}' and '{agent_id}' share the same purpose",
                    "Each configured task-agent type must have a distinct purpose.",
                ))
            else:
                seen_purposes[normalized_purpose] = agent_id

        invocation = entry.get("invocation")
        if not isinstance(invocation, Mapping):
            issues.append(TaskAgentConfigIssue(
                "error",
                f"{prefix}.invocation must be a dict",
                "Invocation must specify entrypoint, task_category, and parameters.",
            ))
        else:
            if invocation.get("entrypoint") != "delegate_task":
                issues.append(TaskAgentConfigIssue(
                    "error",
                    f"{prefix}.invocation.entrypoint must be 'delegate_task'",
                    "Configured task agents currently invoke through the existing delegate_task path.",
                ))
            if invocation.get("task_category") != agent_id:
                issues.append(TaskAgentConfigIssue(
                    "error",
                    f"{prefix}.invocation.task_category must match id '{agent_id}'",
                    "Keep task_category equal to the stable task-agent id.",
                ))
            parameters = invocation.get("parameters")
            if not isinstance(parameters, Mapping):
                issues.append(TaskAgentConfigIssue(
                    "error",
                    f"{prefix}.invocation.parameters must be a dict",
                    "Use parameters.required and parameters.optional lists for integration-time validation.",
                ))
            else:
                required = parameters.get("required", [])
                optional = parameters.get("optional", [])
                if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
                    issues.append(TaskAgentConfigIssue("error", f"{prefix}.invocation.parameters.required must be a list of strings", "List required invocation input names."))
                elif "prompt" not in required:
                    issues.append(TaskAgentConfigIssue("error", f"{prefix}.invocation.parameters.required must include 'prompt'", "Task-specific agents need a prompt to execute."))
                if not isinstance(optional, list) or not all(isinstance(item, str) for item in optional):
                    issues.append(TaskAgentConfigIssue("error", f"{prefix}.invocation.parameters.optional must be a list of strings", "List optional invocation input names."))

        runtime = entry.get("runtime", {})
        if runtime is None:
            runtime = {}
        if not isinstance(runtime, Mapping):
            issues.append(TaskAgentConfigIssue(
                "error",
                f"{prefix}.runtime must be a dict when provided",
                "Runtime may contain model/provider overrides, toolsets, skills, and limits.",
            ))
        else:
            unknown_runtime = sorted(set(runtime) - SUPPORTED_RUNTIME_KEYS)
            if unknown_runtime:
                issues.append(TaskAgentConfigIssue(
                    "error",
                    f"{prefix}.runtime has unsupported keys {unknown_runtime}",
                    "Supported runtime keys: " + ", ".join(sorted(SUPPORTED_RUNTIME_KEYS)),
                ))
            for string_key in ("model", "provider", "base_url", "api_key", "api_mode", "reasoning_effort"):
                if string_key in runtime and not isinstance(runtime[string_key], str):
                    issues.append(TaskAgentConfigIssue("error", f"{prefix}.runtime.{string_key} must be a string", "Use an empty string to inherit the session setting."))
            for int_key in ("max_iterations", "timeout_seconds", "max_summary_chars"):
                if int_key in runtime:
                    value = runtime[int_key]
                    if not isinstance(value, int) or value < 0:
                        issues.append(TaskAgentConfigIssue("error", f"{prefix}.runtime.{int_key} must be a non-negative integer", "Use 0 for the existing runtime default/unlimited behavior where supported."))
            for list_key in ("enabled_toolsets", "disabled_toolsets", "skills"):
                if list_key in runtime and not (isinstance(runtime[list_key], list) and all(isinstance(item, str) for item in runtime[list_key])):
                    issues.append(TaskAgentConfigIssue("error", f"{prefix}.runtime.{list_key} must be a list of strings", "Use YAML list syntax, e.g. enabled_toolsets: [web, file]."))

    return issues


def load_task_agent_definitions(config: Mapping[str, Any]) -> Dict[str, TaskAgentDefinition]:
    """Return validated task-agent definitions keyed by stable id.

    Raises TaskAgentConfigError when the config contains errors.  Warnings are
    currently not produced by the validator; if that changes, callers can still
    load when all issues are non-error.
    """

    issues = validate_task_agents_config(config)
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        raise TaskAgentConfigError(errors)

    section = config.get("task_agents") or default_task_agents_config()
    definitions = section.get("definitions", []) if isinstance(section, Mapping) else []
    loaded: Dict[str, TaskAgentDefinition] = {}
    for entry in definitions:
        runtime = dict(entry.get("runtime") or {})
        loaded[entry["id"]] = TaskAgentDefinition(
            id=entry["id"],
            purpose=entry["purpose"],
            enabled=entry["enabled"],
            invocation=dict(entry["invocation"]),
            runtime=runtime,
        )
    return loaded

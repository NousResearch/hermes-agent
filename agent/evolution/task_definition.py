"""Task definition model for the Evolution Engine.

A task is a unit of work the agent attempts, with defined success criteria.
Tasks are defined in YAML files stored under ``~/.hermes/evolution/tasks/``.

Example task definition:

.. code-block:: yaml

    name: fix-login-bug
    description: "Fix the login redirect loop when session expires"
    domain: software-development
    complexity: 5
    success_criteria:
      - type: test_pass
        command: "pytest tests/test_login.py -v"
      - type: file_exists
        path: "fixes/login_fix.patch"
    environment:
      cwd: "/home/user/project"
      sandbox: docker
    tools_allowed:
      - terminal
      - read_file
      - write_file
      - search_files
      - patch
    timeout_seconds: 600
    max_turns: 30
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TaskStatus(str, Enum):
    PENDING = "pending"
    ATTEMPTING = "attempting"
    EVALUATING = "evaluating"
    ANALYZING = "analyzing"
    IMPROVING = "improving"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    EXHAUSTED = "exhausted"  # max iterations reached without success
    CANCELLED = "cancelled"


class SuccessCriterionType(str, Enum):
    TEST_PASS = "test_pass"          # Run a command, check exit code 0
    FILE_EXISTS = "file_exists"       # Check a file was created
    CONTENT_MATCH = "content_match"   # grep/regex against file content
    COMMAND_OUTPUT = "command_output" # Check command output matches expected
    LLM_JUDGE = "llm_judge"          # Use auxiliary model as judge
    MANUAL = "manual"                 # Human must verify


class ImprovementActionType(str, Enum):
    SKILL_CREATE = "skill_create"
    SKILL_PATCH = "skill_patch"
    TOOL_CREATE = "tool_create"
    TOOL_MODIFY = "tool_modify"
    PROMPT_MODIFY = "prompt_modify"
    MEMORY_UPDATE = "memory_update"
    STRATEGY_CHANGE = "strategy_change"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class SuccessCriterion:
    """A single success condition for a task."""
    type: SuccessCriterionType
    command: Optional[str] = None        # For test_pass, command_output
    path: Optional[str] = None           # For file_exists
    pattern: Optional[str] = None        # For content_match (regex)
    expected_output: Optional[str] = None # For command_output
    rubric: Optional[str] = None         # For llm_judge (scoring rubric)
    weight: float = 1.0                  # Relative weight in composite score

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"type": self.type.value, "weight": self.weight}
        if self.command is not None:
            d["command"] = self.command
        if self.path is not None:
            d["path"] = self.path
        if self.pattern is not None:
            d["pattern"] = self.pattern
        if self.expected_output is not None:
            d["expected_output"] = self.expected_output
        if self.rubric is not None:
            d["rubric"] = self.rubric
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SuccessCriterion":
        return cls(
            type=SuccessCriterionType(d["type"]),
            command=d.get("command"),
            path=d.get("path"),
            pattern=d.get("pattern"),
            expected_output=d.get("expected_output"),
            rubric=d.get("rubric"),
            weight=float(d.get("weight", 1.0)),
        )


@dataclass
class TaskDefinition:
    """A complete task specification for the Evolution Engine."""

    name: str
    description: str
    success_criteria: List[SuccessCriterion] = field(default_factory=list)
    domain: str = "general"
    complexity: int = 1  # 1-14 scale (AI4Work-inspired)
    environment: Dict[str, Any] = field(default_factory=dict)
    tools_allowed: List[str] = field(default_factory=list)
    tools_denied: List[str] = field(default_factory=list)
    timeout_seconds: int = 600
    max_turns: int = 30
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Runtime fields (not serialized)
    task_id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:12]}")
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "domain": self.domain,
            "complexity": self.complexity,
            "success_criteria": [c.to_dict() for c in self.success_criteria],
            "environment": self.environment,
            "tools_allowed": self.tools_allowed,
            "tools_denied": self.tools_denied,
            "timeout_seconds": self.timeout_seconds,
            "max_turns": self.max_turns,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskDefinition":
        criteria = [SuccessCriterion.from_dict(c) for c in d.get("success_criteria", [])]
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            success_criteria=criteria,
            domain=d.get("domain", "general"),
            complexity=int(d.get("complexity", 1)),
            environment=d.get("environment", {}),
            tools_allowed=list(d.get("tools_allowed", [])),
            tools_denied=list(d.get("tools_denied", [])),
            timeout_seconds=int(d.get("timeout_seconds", 600)),
            max_turns=int(d.get("max_turns", 30)),
            metadata=d.get("metadata", {}),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "TaskDefinition":
        """Load a task definition from a YAML file."""
        with open(path, encoding="utf-8-sig") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Task YAML must be a mapping, got {type(data).__name__}: {path}")
        return cls.from_dict(data)

    def to_yaml(self, path: Path) -> None:
        """Write the task definition to a YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def validate(self) -> List[str]:
        """Return list of validation errors (empty = valid)."""
        errors = []
        if not self.name or not self.name.strip():
            errors.append("name is required")
        if not self.success_criteria:
            errors.append("at least one success_criterion is required")
        if self.complexity < 1 or self.complexity > 14:
            errors.append(f"complexity must be 1-14, got {self.complexity}")
        if self.timeout_seconds < 1:
            errors.append(f"timeout_seconds must be >= 1, got {self.timeout_seconds}")
        if self.max_turns < 1:
            errors.append(f"max_turns must be >= 1, got {self.max_turns}")
        for i, criterion in enumerate(self.success_criteria):
            if criterion.type == SuccessCriterionType.TEST_PASS and not criterion.command:
                errors.append(f"criterion {i}: test_pass requires 'command'")
            if criterion.type == SuccessCriterionType.FILE_EXISTS and not criterion.path:
                errors.append(f"criterion {i}: file_exists requires 'path'")
            if criterion.type == SuccessCriterionType.CONTENT_MATCH and not criterion.pattern:
                errors.append(f"criterion {i}: content_match requires 'pattern'")
        return errors


# ---------------------------------------------------------------------------
# Task bank management
# ---------------------------------------------------------------------------


def get_task_dir() -> Path:
    """Return the task storage directory."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "evolution" / "tasks"


def list_tasks() -> List[TaskDefinition]:
    """List all defined tasks in the task bank."""
    task_dir = get_task_dir()
    if not task_dir.is_dir():
        return []
    tasks = []
    for yaml_file in sorted(task_dir.glob("*.yaml")):
        try:
            tasks.append(TaskDefinition.from_yaml(yaml_file))
        except Exception as e:
            logger.warning("Failed to load task %s: %s", yaml_file, e)
    return tasks


def load_task(name: str) -> Optional[TaskDefinition]:
    """Load a task by name."""
    task_file = get_task_dir() / f"{name}.yaml"
    if not task_file.exists():
        return None
    return TaskDefinition.from_yaml(task_file)


def save_task(task: TaskDefinition) -> Path:
    """Persist a task definition. Returns the file path."""
    path = get_task_dir() / f"{task.name}.yaml"
    task.to_yaml(path)
    return path


def delete_task(name: str) -> bool:
    """Delete a task definition. Returns True if deleted."""
    task_file = get_task_dir() / f"{name}.yaml"
    if task_file.exists():
        task_file.unlink()
        return True
    return False

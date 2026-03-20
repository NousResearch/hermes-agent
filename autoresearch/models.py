"""Normalized AutoResearch configuration models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class CommandConfig:
    """Commands and result paths for candidate validation/evaluation."""

    evaluation: str
    result_json: str
    validation: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SecondaryMetricRule:
    """Selector guardrail for metrics that must not regress too far."""

    metric: str
    min_delta: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SelectionConfig:
    """Selector behavior for ranking and keeping candidates."""

    primary_metric: str
    goal: str = "maximize"
    secondary_metrics: list[SecondaryMetricRule] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_metric": self.primary_metric,
            "goal": self.goal,
            "secondary_metrics": [rule.to_dict() for rule in self.secondary_metrics],
        }


@dataclass(frozen=True)
class InterestingRule:
    """One boolean rule used to decide whether a report is worth writing."""

    metric: str
    op: str
    value: Any

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InterestingConfig:
    """Report/publication interestingness policy."""

    mode: str = "all"
    rules: list[InterestingRule] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "rules": [rule.to_dict() for rule in self.rules],
        }


@dataclass(frozen=True)
class EditableMarker:
    """Marker-bounded editable region inside a file."""

    file: str
    start: str
    end: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnchorConfig:
    """Seed/benchmark candidate declared in a family config."""

    candidate_id: str
    label: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MutationConfig:
    """Candidate generation policy."""

    mode: str
    population: int = 8
    survivors: int = 3
    parameter_space: dict[str, list[Any]] = field(default_factory=dict)
    max_mutations_per_candidate: int = 2
    prompt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level workspace AutoResearch config."""

    project_id: str
    description: str
    root: Path
    default_cwd: str = "."
    datasets: list[Any] = field(default_factory=list)
    benchmarks: list[str] = field(default_factory=list)
    evaluator: Optional[CommandConfig] = None
    report_output_dir: str = "research"
    publish_target: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "description": self.description,
            "root": str(self.root),
            "default_cwd": self.default_cwd,
            "datasets": list(self.datasets),
            "benchmarks": list(self.benchmarks),
            "evaluator": self.evaluator.to_dict() if self.evaluator else None,
            "report_output_dir": self.report_output_dir,
            "publish_target": self.publish_target,
        }


@dataclass(frozen=True)
class FamilyConfig:
    """One bounded search family inside a project."""

    family_id: str
    thesis: str
    project_root: Path
    commands: CommandConfig
    mutation: MutationConfig
    selection: SelectionConfig
    interesting: InterestingConfig
    anchors: list[AnchorConfig]
    editable_files: list[str] = field(default_factory=list)
    editable_markers: list[EditableMarker] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "thesis": self.thesis,
            "project_root": str(self.project_root),
            "commands": self.commands.to_dict(),
            "mutation": self.mutation.to_dict(),
            "selection": self.selection.to_dict(),
            "interesting": self.interesting.to_dict(),
            "anchors": [anchor.to_dict() for anchor in self.anchors],
            "editable_files": list(self.editable_files),
            "editable_markers": [marker.to_dict() for marker in self.editable_markers],
        }


@dataclass
class WorkspaceInfo:
    """Runtime metadata for one candidate workspace."""

    path: Path
    method: str
    source_root: Path
    repo_root: Optional[Path] = None
    branch: Optional[str] = None
    editable_base_contents: dict[str, Optional[str]] = field(default_factory=dict)
    snapshot: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "method": self.method,
            "source_root": str(self.source_root),
            "repo_root": str(self.repo_root) if self.repo_root else None,
            "branch": self.branch,
        }

"""Structured research state models."""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SpecLevel = Literal["zero_spec", "partial_spec", "full_spec"]
ResearchPhase = Literal[
    "bootstrap",
    "literature",
    "ideation",
    "planning",
    "data",
    "training",
    "evaluation",
    "reporting",
    "completed",
]
RunMethod = Literal["sft", "dpo", "ppo"]
RunStatus = Literal[
    "pending",
    "pending_approval",
    "queued",
    "running",
    "stopping",
    "stopped",
    "completed",
    "failed",
]
InboxStatus = Literal["unread", "read"]


def now_utc() -> str:
    """Return an ISO timestamp in UTC."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(value: str, default: str = "project") -> str:
    """Create a filesystem-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or default


def make_id(prefix: str) -> str:
    """Create a short prefixed identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class ResearchBaseModel(BaseModel):
    """Base model with permissive extra fields."""

    model_config = ConfigDict(extra="allow")


class ResearchProject(ResearchBaseModel):
    project_id: str = Field(default_factory=lambda: make_id("proj"))
    name: str
    cwd: str
    workspace_dir: str
    created_at: str = Field(default_factory=now_utc)
    updated_at: str = Field(default_factory=now_utc)
    objective: str = ""
    user_brief: str = ""
    spec_level: SpecLevel = "partial_spec"
    zero_spec_strategy: str = "novel_idea_first"
    phase: ResearchPhase = "bootstrap"
    status: str = "active"
    active_run_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchIdea(ResearchBaseModel):
    idea_id: str = Field(default_factory=lambda: make_id("idea"))
    title: str
    summary: str
    rationale: str = ""
    novelty_score: float | None = None
    feasibility_score: float | None = None
    cost_score: float | None = None
    evaluation_score: float | None = None
    chosen: bool = False
    source: str = "agent"
    created_at: str = Field(default_factory=now_utc)


class Hypothesis(ResearchBaseModel):
    hypothesis_id: str = Field(default_factory=lambda: make_id("hyp"))
    statement: str
    success_metric: str
    stop_condition: str
    rationale: str = ""
    linked_idea_id: str | None = None
    status: str = "draft"
    created_at: str = Field(default_factory=now_utc)


class ExperimentSpec(ResearchBaseModel):
    experiment_id: str = Field(default_factory=lambda: make_id("exp"))
    title: str
    hypothesis_id: str
    objective: str
    method: RunMethod = "sft"
    dataset_path: str = ""
    evaluation_plan: list[str] = Field(default_factory=list)
    success_metric: str
    stop_condition: str
    status: str = "draft"
    run_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=now_utc)
    updated_at: str = Field(default_factory=now_utc)


class RunSpec(ResearchBaseModel):
    run_id: str = Field(default_factory=lambda: make_id("run"))
    project_id: str
    experiment_id: str | None = None
    method: RunMethod = "sft"
    base_model: str
    dataset_path: str
    field_map: dict[str, str] = Field(default_factory=dict)
    batch_size: int = 32
    learning_rate: float = 1e-4
    lora_rank: int = 32
    num_epochs: int = 1
    max_steps: int | None = None
    save_every: int = 50
    output_name: str = "final"
    checkpoint_name: str = "final"
    beta: float = 0.1
    sampling_params: dict[str, Any] = Field(default_factory=dict)
    optimizer: dict[str, Any] = Field(default_factory=dict)
    estimated_cost_usd: float | None = None
    status: RunStatus = "pending"
    resume_from: str | None = None
    resume_step: int = 0
    reference_model_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=now_utc)
    updated_at: str = Field(default_factory=now_utc)


class LoopCheckpoint(ResearchBaseModel):
    checkpoint_id: str = Field(default_factory=lambda: make_id("ckpt"))
    project_id: str
    reason: str
    summary: str
    next_action: str = ""
    continuation_job_id: str | None = None
    created_at: str = Field(default_factory=now_utc)
    metadata: dict[str, Any] = Field(default_factory=dict)


class InboxItem(ResearchBaseModel):
    item_id: str = Field(default_factory=lambda: make_id("inbox"))
    project_id: str
    title: str
    summary: str
    details: str = ""
    status: InboxStatus = "unread"
    created_at: str = Field(default_factory=now_utc)
    metadata: dict[str, Any] = Field(default_factory=dict)


EvalStatus = Literal["pending", "running", "completed", "failed"]


class EvalResult(ResearchBaseModel):
    eval_id: str = Field(default_factory=lambda: make_id("eval"))
    project_id: str
    run_id: str | None = None
    experiment_id: str | None = None
    benchmark: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    num_fewshot: int = 0
    model_args: str = ""
    status: EvalStatus = "pending"
    error: str = ""
    created_at: str = Field(default_factory=now_utc)


class LiteratureRecord(ResearchBaseModel):
    record_id: str = Field(default_factory=lambda: make_id("lit"))
    paper_id: str = ""
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    abstract: str = ""
    url: str = ""
    citation_count: int | None = None
    source: str = ""
    tags: list[str] = Field(default_factory=list)
    summary: str = ""
    created_at: str = Field(default_factory=now_utc)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetRecord(ResearchBaseModel):
    dataset_id: str = Field(default_factory=lambda: make_id("ds"))
    project_id: str = ""
    name: str = ""
    path: str = ""
    format: str = "jsonl"
    num_rows: int = 0
    columns: list[str] = Field(default_factory=list)
    source_action: str = ""
    parent_dataset_id: str | None = None
    source_paths: list[str] = Field(default_factory=list)
    quality_report: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=now_utc)


class GraphEdge(ResearchBaseModel):
    edge_id: str = Field(default_factory=lambda: make_id("edge"))
    source_type: str
    source_id: str
    target_type: str
    target_id: str
    relation: str
    weight: float = 1.0
    created_at: str = Field(default_factory=now_utc)

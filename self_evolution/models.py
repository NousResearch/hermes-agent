"""
Self Evolution Plugin — Data Models
=====================================
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import json
import time


def _now() -> float:
    return time.time()


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


# ── Quality Scoring ──────────────────────────────────────────────────────

@dataclass
class QualityScore:
    session_id: str
    composite: float = 0.0
    completion_rate: float = 0.0
    efficiency_score: float = 0.0
    cost_efficiency: float = 0.0
    satisfaction_proxy: float = 0.0
    task_category: str = ""
    model: str = ""
    created_at: float = field(default_factory=_now)

    def to_db_row(self) -> dict:
        return {
            "session_id": self.session_id,
            "composite_score": self.composite,
            "completion_rate": self.completion_rate,
            "efficiency_score": self.efficiency_score,
            "cost_efficiency": self.cost_efficiency,
            "satisfaction_proxy": self.satisfaction_proxy,
            "task_category": self.task_category,
            "model": self.model,
            "created_at": self.created_at,
        }


# ── Error Analysis ───────────────────────────────────────────────────────

@dataclass
class ToolFailure:
    tool_name: str
    error_type: str
    count: int
    sessions_affected: List[str] = field(default_factory=list)
    example_session: str = ""


@dataclass
class RetryPattern:
    session_id: str
    tool_name: str
    attempt_count: int
    final_outcome: str  # "success" | "failure" | "abandoned"


@dataclass
class ErrorAnalysis:
    tool_failures: List[ToolFailure] = field(default_factory=list)
    retry_patterns: List[RetryPattern] = field(default_factory=list)
    incomplete_sessions: List[str] = field(default_factory=list)
    user_corrections: int = 0
    correction_examples: List[str] = field(default_factory=list)
    api_error_count: int = 0
    api_error_types: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = []
        if self.tool_failures:
            lines.append(f"工具失败: {len(self.tool_failures)} 种工具出错")
            for tf in self.tool_failures[:5]:
                lines.append(f"  - {tf.tool_name}: {tf.count}次 ({tf.error_type})")
        if self.retry_patterns:
            retries = len(self.retry_patterns)
            lines.append(f"重复重试: {retries} 次")
        if self.incomplete_sessions:
            lines.append(f"未完成session: {len(self.incomplete_sessions)} 个")
        if self.user_corrections:
            lines.append(f"用户纠正: {self.user_corrections} 次")
        if self.api_error_count:
            lines.append(f"API错误: {self.api_error_count} 次")
        return "\n".join(lines)


# ── Time Waste Analysis ──────────────────────────────────────────────────

@dataclass
class ToolDuration:
    tool_name: str
    total_duration_ms: int
    call_count: int
    avg_duration_ms: float


@dataclass
class RepeatedOperation:
    description: str
    count: int
    sessions: List[str] = field(default_factory=list)
    wasted_ms: int = 0


@dataclass
class WasteAnalysis:
    slowest_tools: List[ToolDuration] = field(default_factory=list)
    repeated_operations: List[RepeatedOperation] = field(default_factory=list)
    inefficient_sessions: List[str] = field(default_factory=list)
    shortcut_opportunities: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = []
        if self.slowest_tools:
            lines.append("耗时最长的工具:")
            for td in self.slowest_tools[:5]:
                lines.append(f"  - {td.tool_name}: 平均{td.avg_duration_ms:.0f}ms ({td.call_count}次)")
        if self.repeated_operations:
            lines.append(f"重复操作: {len(self.repeated_operations)} 种")
            for ro in self.repeated_operations[:5]:
                lines.append(f"  - {ro.description}: {ro.count}次")
        if self.inefficient_sessions:
            lines.append(f"低效session: {len(self.inefficient_sessions)} 个")
        if self.shortcut_opportunities:
            lines.append(f"可优化路径: {len(self.shortcut_opportunities)} 个")
        return "\n".join(lines)


# ── Code Change Analysis ──────────────────────────────────────────────────

@dataclass
class CommitInfo:
    hash_short: str
    subject: str
    body: str = ""
    author: str = ""
    timestamp: float = 0.0
    files_changed: int = 0
    insertions: int = 0
    deletions: int = 0
    file_list: List[str] = field(default_factory=list)


@dataclass
class CodeChangeAnalysis:
    commits: List[CommitInfo] = field(default_factory=list)
    total_commits: int = 0
    total_insertions: int = 0
    total_deletions: int = 0
    total_files_changed: int = 0
    authors: List[str] = field(default_factory=list)
    change_categories: Dict[str, int] = field(default_factory=dict)
    areas_touched: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.commits:
            return "代码更新: 无新提交"
        lines = [
            f"代码更新: {self.total_commits} commits, "
            f"+{self.total_insertions}/-{self.total_deletions} lines, "
            f"{self.total_files_changed} files changed",
        ]
        if self.change_categories:
            cats = ", ".join(f"{k}: {v}" for k, v in self.change_categories.items())
            lines.append(f"提交类型分布: {cats}")
        if self.areas_touched:
            lines.append(f"涉及模块: {', '.join(self.areas_touched)}")
        lines.append("主要变更:")
        for c in self.commits[:8]:
            lines.append(f"  - {c.subject} ({c.hash_short}, +{c.insertions}/-{c.deletions})")
        return "\n".join(lines)


# ── Reflection Report ────────────────────────────────────────────────────

@dataclass
class ReflectionReport:
    period_start: float
    period_end: float
    sessions_analyzed: int = 0
    avg_score: float = 0.0
    error_summary: str = ""
    waste_summary: str = ""
    worst_patterns: List[str] = field(default_factory=list)
    best_patterns: List[str] = field(default_factory=list)
    tool_insights: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    code_change_summary: str = ""
    model_used: str = ""
    created_at: float = field(default_factory=_now)

    def to_db_row(self) -> dict:
        return {
            "period_start": self.period_start,
            "period_end": self.period_end,
            "sessions_analyzed": self.sessions_analyzed,
            "avg_score": self.avg_score,
            "error_summary": self.error_summary,
            "waste_summary": self.waste_summary,
            "worst_patterns": json.dumps(self.worst_patterns, ensure_ascii=False),
            "best_patterns": json.dumps(self.best_patterns, ensure_ascii=False),
            "tool_insights": json.dumps(self.tool_insights, ensure_ascii=False),
            "recommendations": json.dumps(self.recommendations, ensure_ascii=False),
            "code_change_summary": self.code_change_summary,
            "model_used": self.model_used,
            "created_at": self.created_at,
        }


# ── Evolution Proposal ───────────────────────────────────────────────────

@dataclass
class Proposal:
    id: str
    proposal_type: str  # skill | strategy | memory | tool_preference | code_improvement
    title: str
    description: str
    expected_impact: str = ""
    risk_assessment: str = "low"
    rollback_plan: str = ""
    status: str = "pending_approval"
    report_id: Optional[int] = None
    user_feedback: str = ""
    created_at: float = field(default_factory=_now)
    resolved_at: Optional[float] = None

    def to_db_row(self) -> dict:
        return {
            "id": self.id,
            "report_id": self.report_id,
            "proposal_type": self.proposal_type,
            "title": self.title,
            "description": self.description,
            "expected_impact": self.expected_impact,
            "risk_assessment": self.risk_assessment,
            "rollback_plan": self.rollback_plan,
            "status": self.status,
            "user_feedback": self.user_feedback,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
        }


# ── Improvement Unit (A/B Test Tracking) ─────────────────────────────────

@dataclass
class ImprovementUnit:
    id: str
    proposal_id: str
    change_type: str
    version: int = 0
    baseline_score: float = 0.0
    current_score: float = 0.0
    sessions_sampled: int = 0
    min_sessions: int = 10
    min_improvement: float = 0.05
    max_regression: float = 0.10
    status: str = "active"  # active | promoted | reverted
    created_at: float = field(default_factory=_now)
    resolved_at: Optional[float] = None

    @property
    def should_revert(self) -> bool:
        return (
            self.sessions_sampled >= 3
            and (self.baseline_score - self.current_score) > self.max_regression
        )

    @property
    def should_promote(self) -> bool:
        return (
            self.sessions_sampled >= self.min_sessions
            and (self.current_score - self.baseline_score) >= self.min_improvement
        )

    def to_db_row(self) -> dict:
        return {
            "id": self.id,
            "proposal_id": self.proposal_id,
            "change_type": self.change_type,
            "version": self.version,
            "baseline_score": self.baseline_score,
            "current_score": self.current_score,
            "sessions_sampled": self.sessions_sampled,
            "min_sessions": self.min_sessions,
            "min_improvement": self.min_improvement,
            "max_regression": self.max_regression,
            "status": self.status,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
        }


# ── Strategy Rule ────────────────────────────────────────────────────────

@dataclass
class StrategyCondition:
    field: str
    operator: str  # regex_match | contains | equals | not_contains
    pattern: str


@dataclass
class StrategyRule:
    id: str
    name: str
    strategy_type: str  # hint | avoid | prefer
    description: str
    conditions: List[StrategyCondition] = field(default_factory=list)
    hint_text: str = ""
    severity: str = "medium"  # high | medium | low
    enabled: bool = True
    version: int = 1
    source: str = "learned"  # learned | manual | default
    created_at: float = field(default_factory=_now)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "strategy_type": self.strategy_type,
            "description": self.description,
            "conditions": [
                {"field": c.field, "operator": c.operator, "pattern": c.pattern}
                for c in self.conditions
            ],
            "hint_text": self.hint_text,
            "severity": self.severity,
            "enabled": self.enabled,
            "version": self.version,
            "source": self.source,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> StrategyRule:
        conditions = [
            StrategyCondition(field=c["field"], operator=c["operator"], pattern=c["pattern"])
            for c in d.get("conditions", [])
        ]
        return cls(
            id=d["id"],
            name=d["name"],
            strategy_type=d.get("strategy_type", "hint"),
            description=d.get("description", ""),
            conditions=conditions,
            hint_text=d.get("hint_text", ""),
            severity=d.get("severity", "medium"),
            enabled=d.get("enabled", True),
            version=d.get("version", 1),
            source=d.get("source", "learned"),
            created_at=d.get("created_at", _now()),
        )

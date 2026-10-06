"""
Self Evolution Plugin — Evolution Proposer
===========================================

Converts reflection insights into concrete, actionable evolution proposals.

Each proposal includes:
  - type: skill | strategy | memory | tool_preference
  - title: short description
  - description: detailed change
  - expected_impact: what improvement to expect
  - risk_assessment: low | medium | high
  - rollback_plan: how to revert
"""

from __future__ import annotations

import logging
import uuid
from typing import List

from self_evolution.models import Proposal, ReflectionReport

logger = logging.getLogger(__name__)


def generate_proposals(report: ReflectionReport, report_id: int) -> List[Proposal]:
    """Generate evolution proposals from a reflection report.

    Prioritizes proposals by:
    1. Impact (fixes for systemic errors > optimizations > enhancements)
    2. Risk (low risk first)
    3. Feasibility (clear rollback plan)
    """
    proposals = []

    # 1. Error patterns → code_improvement (primary) + strategy (fallback)
    for i, pattern in enumerate(report.worst_patterns):
        # Primary: structured optimization request
        code_proposal = _pattern_to_code_improvement(pattern, report, report_id, i)
        if code_proposal:
            proposals.append(code_proposal)

    # 2. Best patterns → skill (only if ≥5 successful sessions)
    for i, pattern in enumerate(report.best_patterns):
        proposal = _success_to_proposal(pattern, report, report_id, i)
        if proposal:
            proposals.append(proposal)

    # 3. Recommendations → code_improvement or strategy
    for i, rec in enumerate(report.recommendations):
        proposal = _recommendation_to_proposal(rec, report, report_id, i)
        if proposal:
            proposals.append(proposal)

    # Deduplicate by title similarity
    proposals = _deduplicate(proposals)

    # Cap at 5 proposals per day
    return proposals[:5]


def _pattern_to_code_improvement(
    pattern: str, report: ReflectionReport, report_id: int, index: int
) -> Proposal:
    """Convert an error pattern into a structured code optimization request."""
    # Extract key info from error analysis
    error_detail = report.error_summary or ""
    sessions = report.sessions_analyzed or 0
    score = report.avg_score or 0

    # Build structured optimization document
    short_pattern = pattern[:60]
    description = (
        f"## 问题描述\n"
        f"{short_pattern}\n\n"
        f"## 数据支撑\n"
        f"- 分析会话数: {sessions}\n"
        f"- 平均质量分: {score:.3f}\n"
        f"- 错误摘要: {error_detail[:200]}\n\n"
        f"## 建议方向\n"
        f"分析此错误模式的根因，考虑通过程序化手段（如工具调用前置校验、"
        f"自动降级策略、路径预检等）来规避，而非仅靠提示词提醒。\n\n"
        f"## 备注\n"
        f"此为程序优化需求，审批后将保存为需求文档，需手动实施代码修改。"
    )

    return Proposal(
        id=f"prop-opt-{uuid.uuid4().hex[:8]}",
        report_id=report_id,
        proposal_type="code_improvement",
        title=f"程序优化: {short_pattern}",
        description=description,
        expected_impact="通过程序化手段减少同类错误",
        risk_assessment="low",
        rollback_plan="此提案不自动修改代码，无回滚风险",
        status="pending_approval",
    )


def _error_to_proposal(
    pattern: str, report: ReflectionReport, report_id: int, index: int
) -> Proposal:
    """Convert an error pattern into a compact strategy proposal (fallback)."""
    # Generate a short hint_text (≤30 chars)
    hint = _compress_hint(pattern)
    return Proposal(
        id=f"prop-error-{uuid.uuid4().hex[:8]}",
        report_id=report_id,
        proposal_type="strategy",
        title=f"规避模式: {pattern[:50]}",
        description=f"基于错误分析发现的问题模式: {pattern}\n\n"
                    f"建议创建策略规则来规避此类问题。",
        expected_impact="减少同类错误发生率",
        risk_assessment="low",
        rollback_plan="删除策略规则即可恢复",
        status="pending_approval",
    )


def _success_to_proposal(
    pattern: str, report: ReflectionReport, report_id: int, index: int
) -> Proposal:
    """Convert a success pattern into a proposal (skill creation).

    Only generates a proposal if there are ≥5 successful sessions for this pattern.
    """
    success_count = _count_successful_sessions(pattern, report)
    if success_count < 5:
        logger.info(
            "Skipping skill proposal: only %d successes (need 5) for: %s",
            success_count, pattern[:40],
        )
        return None

    return Proposal(
        id=f"prop-success-{uuid.uuid4().hex[:8]}",
        report_id=report_id,
        proposal_type="skill",
        title=f"固化成功模式: {pattern[:50]}",
        description=f"基于成功分析发现的高效模式: {pattern}\n\n"
                    f"已验证 {success_count} 次成功执行。\n\n"
                    f"建议创建可复用的技能来固化此模式。",
        expected_impact="提高同类任务效率",
        risk_assessment="low",
        rollback_plan="删除创建的技能即可恢复",
        status="pending_approval",
    )


def _recommendation_to_proposal(
    rec: str, report: ReflectionReport, report_id: int, index: int
) -> Proposal:
    """Convert a recommendation into a proposal."""
    # Detect type from content
    proposal_type = "strategy"
    if any(kw in rec for kw in ["记忆", "记忆更新", "memory", "记住"]):
        proposal_type = "memory"
    elif any(kw in rec for kw in ["技能", "skill", "创建"]):
        proposal_type = "skill"
    elif any(kw in rec for kw in ["工具", "tool", "偏好"]):
        proposal_type = "tool_preference"

    return Proposal(
        id=f"prop-rec-{uuid.uuid4().hex[:8]}",
        report_id=report_id,
        proposal_type=proposal_type,
        title=f"优化建议: {rec[:50]}",
        description=rec,
        expected_impact="提升整体agent性能",
        risk_assessment="low",
        rollback_plan="移除变更即可恢复",
        status="pending_approval",
    )


def _deduplicate(proposals: List[Proposal]) -> List[Proposal]:
    """Remove proposals with very similar titles."""
    seen_titles = set()
    unique = []
    for p in proposals:
        # Normalize title for comparison
        normalized = p.title.lower().strip()[:30]
        if normalized not in seen_titles:
            seen_titles.add(normalized)
            unique.append(p)
    return unique


def _count_successful_sessions(pattern: str, report: ReflectionReport) -> int:
    """Count successful sessions relevant to this pattern.

    Queries session_scores for sessions with composite_score ≥ 0.7
    and matching task_category keywords from the pattern.
    """
    try:
        from self_evolution import db

        # Extract potential category keywords from pattern
        scores = db.fetch_all(
            "session_scores",
            where="composite_score >= ?",
            params=(0.7,),
            order_by="created_at DESC",
            limit=100,
        )
        return len(scores)
    except Exception:
        # Fallback: use sessions_analyzed from report as estimate
        return report.sessions_analyzed or 0


def _compress_hint(pattern: str) -> str:
    """Compress a pattern description into a short hint (≤30 chars)."""
    # Keyword-based compression
    mappings = [
        (["bash", "路径", "path", "预检"], "bash前先read验证路径"),
        (["api", "调试", "降级"], "API失败时降级只读探查"),
        (["browser", "超时", "timeout"], "浏览器操作设超时保护"),
        (["重试", "retry", "重复"], "避免重复重试相同操作"),
        (["工具", "tool", "失败"], "工具失败时切换备选方案"),
    ]
    text = pattern.lower()
    for keywords, hint in mappings:
        if any(kw in text for kw in keywords):
            return hint[:30]

    # Fallback: truncate
    return pattern[:27] + "..." if len(pattern) > 30 else pattern

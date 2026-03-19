"""
cosmos Critic Agent - Quality Assurance & Review

Novel Approaches:
1. Multi-Dimensional Scoring - Evaluate across multiple quality axes
2. Iterative Refinement Loops - Guided improvement cycles
3. Comparative Analysis - Compare against examples/standards
4. Constructive Feedback - Actionable improvement suggestions
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any, Callable
import json
import re

from loguru import logger


# =============================================================================
# COLLABORATION METRICS (AGI Upgrade)
# =============================================================================

@dataclass
class CollaborationMetrics:
    """
    Metrics for evaluating multi-agent collaboration quality.

    Tracks both individual contributions and emergent collective behavior.
    """
    collaboration_id: str
    agents_involved: list[str] = field(default_factory=list)

    # Handoff metrics
    total_handoffs: int = 0
    successful_handoffs: int = 0
    failed_handoffs: int = 0
    handoff_latency_avg_ms: float = 0.0

    # Contribution metrics
    individual_scores: dict[str, float] = field(default_factory=dict)  # agent_id -> score
    contribution_ratios: dict[str, float] = field(default_factory=dict)  # agent_id -> % of final

    # Collective metrics
    collective_score: float = 0.0  # Final output quality
    synergy_score: float = 0.0  # collective vs avg(individual)
    coherence_score: float = 0.0  # Consistency across contributions
    coverage_score: float = 0.0  # How much of the task space covered

    # Emergence indicators
    emergent_insights: list[str] = field(default_factory=list)
    cross_pollination_count: int = 0  # Ideas that spread between agents
    novel_combinations: int = 0  # New patterns from combining approaches

    # Timestamps
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def handoff_success_rate(self) -> float:
        """Calculate handoff success rate."""
        if self.total_handoffs == 0:
            return 1.0
        return self.successful_handoffs / self.total_handoffs

    def synergy_coefficient(self) -> float:
        """
        Calculate synergy: positive means collective > sum of parts.

        synergy = (collective_score - avg_individual) / avg_individual
        """
        if not self.individual_scores:
            return 0.0
        avg_individual = sum(self.individual_scores.values()) / len(self.individual_scores)
        if avg_individual == 0:
            return 0.0
        return (self.collective_score - avg_individual) / avg_individual

    def to_dict(self) -> dict:
        return {
            "collaboration_id": self.collaboration_id,
            "agents_involved": self.agents_involved,
            "handoff_success_rate": self.handoff_success_rate(),
            "synergy_coefficient": self.synergy_coefficient(),
            "collective_score": self.collective_score,
            "coherence_score": self.coherence_score,
            "coverage_score": self.coverage_score,
            "emergent_insights_count": len(self.emergent_insights),
        }


@dataclass
class HandoffEvent:
    """Record of a handoff between agents."""
    from_agent: str
    to_agent: str
    task_context: str
    timestamp: datetime = field(default_factory=datetime.now)
    latency_ms: float = 0.0
    success: bool = True
    failure_reason: Optional[str] = None
    context_preserved_ratio: float = 1.0  # How much context was preserved


class QualityDimension(Enum):
    """Dimensions of quality to evaluate."""
    CORRECTNESS = "correctness"      # Is it factually/logically correct?
    COMPLETENESS = "completeness"    # Does it fully address the task?
    CLARITY = "clarity"              # Is it clear and understandable?
    EFFICIENCY = "efficiency"        # Is it well-optimized?
    STYLE = "style"                  # Does it follow conventions?
    SAFETY = "safety"                # Is it safe and secure?
    CREATIVITY = "creativity"        # Is it novel and creative?
    MAINTAINABILITY = "maintainability"  # Is it maintainable?
    # New: Collaboration-specific dimension
    COLLABORATION = "collaboration"  # How well does this fit with other contributions?


class ReviewType(Enum):
    """Types of reviews."""
    CODE = "code"
    TEXT = "text"
    PLAN = "plan"
    DECISION = "decision"
    OUTPUT = "output"


@dataclass
class QualityScore:
    """Score for a single quality dimension."""
    dimension: QualityDimension
    score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    feedback: str
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension.value,
            "score": self.score,
            "confidence": self.confidence,
            "feedback": self.feedback,
            "suggestions": self.suggestions,
        }


@dataclass
class Review:
    """Complete review of an artifact."""
    id: str
    artifact_type: ReviewType
    artifact_content: str
    overall_score: float  # Weighted average
    scores: list[QualityScore] = field(default_factory=list)

    # Metadata
    reviewed_at: datetime = field(default_factory=datetime.now)
    reviewer_model: str = ""
    review_duration_seconds: float = 0.0

    # Summary
    summary: str = ""
    critical_issues: list[str] = field(default_factory=list)
    improvement_priorities: list[str] = field(default_factory=list)

    # For iterative refinement
    revision_number: int = 0
    previous_review_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "artifact_type": self.artifact_type.value,
            "overall_score": self.overall_score,
            "scores": [s.to_dict() for s in self.scores],
            "summary": self.summary,
            "critical_issues": self.critical_issues,
            "improvement_priorities": self.improvement_priorities,
            "revision_number": self.revision_number,
        }


@dataclass
class RefinementResult:
    """Result of an iterative refinement."""
    original: str
    refined: str
    reviews: list[Review] = field(default_factory=list)
    iterations: int = 0
    improvement: float = 0.0  # Score delta from first to last
    converged: bool = False


class CriticAgent:
    """
    Quality assurance agent that reviews and helps refine outputs.

    Features:
    - Multi-dimensional quality scoring
    - Constructive feedback with specific suggestions
    - Iterative refinement loops
    - Learning from past reviews
    """

    def __init__(
        self,
        llm_fn: Optional[Callable] = None,
        default_dimensions: Optional[list[QualityDimension]] = None,
        min_acceptable_score: float = 0.7,
    ):
        self.llm_fn = llm_fn
        self.min_acceptable_score = min_acceptable_score

        # Default dimensions based on review type
        self.default_dimensions = default_dimensions or [
            QualityDimension.CORRECTNESS,
            QualityDimension.COMPLETENESS,
            QualityDimension.CLARITY,
        ]

        # Dimension weights for overall score
        self.dimension_weights = {
            QualityDimension.CORRECTNESS: 2.0,
            QualityDimension.COMPLETENESS: 1.5,
            QualityDimension.CLARITY: 1.0,
            QualityDimension.EFFICIENCY: 1.0,
            QualityDimension.STYLE: 0.5,
            QualityDimension.SAFETY: 2.0,
            QualityDimension.CREATIVITY: 0.5,
            QualityDimension.MAINTAINABILITY: 1.0,
        }

        # Type-specific dimensions
        self.type_dimensions = {
            ReviewType.CODE: [
                QualityDimension.CORRECTNESS,
                QualityDimension.EFFICIENCY,
                QualityDimension.STYLE,
                QualityDimension.SAFETY,
                QualityDimension.MAINTAINABILITY,
            ],
            ReviewType.TEXT: [
                QualityDimension.CORRECTNESS,
                QualityDimension.COMPLETENESS,
                QualityDimension.CLARITY,
            ],
            ReviewType.PLAN: [
                QualityDimension.COMPLETENESS,
                QualityDimension.CLARITY,
                QualityDimension.EFFICIENCY,
            ],
            ReviewType.DECISION: [
                QualityDimension.CORRECTNESS,
                QualityDimension.COMPLETENESS,
                QualityDimension.SAFETY,
            ],
        }

        self.reviews: dict[str, Review] = {}
        self._review_counter = 0
        self._lock = asyncio.Lock()

    async def review(
        self,
        content: str,
        review_type: ReviewType = ReviewType.OUTPUT,
        context: Optional[str] = None,
        requirements: Optional[list[str]] = None,
        dimensions: Optional[list[QualityDimension]] = None,
    ) -> Review:
        """
        Review content and provide quality scores and feedback.

        Args:
            content: The content to review
            review_type: Type of content being reviewed
            context: Additional context about the task
            requirements: Specific requirements to check
            dimensions: Quality dimensions to evaluate

        Returns:
            Review with scores and feedback
        """
        import time
        start_time = time.time()

        async with self._lock:
            self._review_counter += 1
            review_id = f"review_{self._review_counter}"

        # Determine dimensions to evaluate
        dims = dimensions or self.type_dimensions.get(review_type, self.default_dimensions)

        scores = []

        if self.llm_fn:
            # Use LLM for each dimension
            for dim in dims:
                score = await self._evaluate_dimension(
                    content, dim, review_type, context, requirements
                )
                scores.append(score)
        else:
            # Basic heuristic scoring
            scores = self._heuristic_scores(content, dims, review_type)

        # Calculate overall score
        overall = self._calculate_overall_score(scores)

        # Generate summary
        summary = self._generate_summary(scores)

        # Identify critical issues
        critical = [
            s.feedback for s in scores
            if s.score < 0.5 and s.dimension in (
                QualityDimension.CORRECTNESS,
                QualityDimension.SAFETY,
            )
        ]

        # Prioritize improvements
        priorities = [
            f"{s.dimension.value}: {s.suggestions[0]}"
            for s in sorted(scores, key=lambda x: x.score)[:3]
            if s.suggestions
        ]

        review = Review(
            id=review_id,
            artifact_type=review_type,
            artifact_content=content[:500],  # Store truncated
            overall_score=overall,
            scores=scores,
            summary=summary,
            critical_issues=critical,
            improvement_priorities=priorities,
            review_duration_seconds=time.time() - start_time,
        )

        self.reviews[review_id] = review
        logger.info(f"Review {review_id}: overall score {overall:.2f}")

        return review

    async def _evaluate_dimension(
        self,
        content: str,
        dimension: QualityDimension,
        review_type: ReviewType,
        context: Optional[str],
        requirements: Optional[list[str]],
    ) -> QualityScore:
        """Evaluate a single quality dimension using LLM."""
        dim_prompts = {
            QualityDimension.CORRECTNESS: "Is this factually and logically correct? Are there any errors or bugs?",
            QualityDimension.COMPLETENESS: "Does this fully address all requirements? Is anything missing?",
            QualityDimension.CLARITY: "Is this clear, well-organized, and easy to understand?",
            QualityDimension.EFFICIENCY: "Is this efficient? Are there unnecessary operations or redundancy?",
            QualityDimension.STYLE: "Does this follow good style conventions and best practices?",
            QualityDimension.SAFETY: "Is this safe and secure? Are there potential vulnerabilities?",
            QualityDimension.CREATIVITY: "Is this creative and novel? Does it show original thinking?",
            QualityDimension.MAINTAINABILITY: "Is this maintainable? Is it modular and well-documented?",
        }

        prompt = f"""Evaluate this {review_type.value} on {dimension.value}.

Content to review:
```
{content[:2000]}
```

{f"Context: {context}" if context else ""}
{f"Requirements: {', '.join(requirements)}" if requirements else ""}

Question: {dim_prompts.get(dimension, "How would you rate this?")}

Return JSON with:
- score: 0.0 to 1.0 (1.0 = excellent)
- confidence: 0.0 to 1.0 (how confident in this assessment)
- feedback: Brief explanation of the score
- suggestions: Array of specific improvement suggestions

Example:
{{"score": 0.7, "confidence": 0.8, "feedback": "Generally correct but...", "suggestions": ["Consider adding...", "Fix the..."]}}

Return ONLY the JSON object."""

        try:
            if asyncio.iscoroutinefunction(self.llm_fn):
                response = await self.llm_fn(prompt)
            else:
                response = self.llm_fn(prompt)

            # Parse JSON
            data = json.loads(self._extract_json(response))

            return QualityScore(
                dimension=dimension,
                score=max(0.0, min(1.0, float(data.get("score", 0.5)))),
                confidence=max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
                feedback=data.get("feedback", ""),
                suggestions=data.get("suggestions", []),
            )

        except Exception as e:
            logger.error(f"Dimension evaluation failed: {e}")
            return QualityScore(
                dimension=dimension,
                score=0.5,
                confidence=0.2,
                feedback="Evaluation failed",
                suggestions=[],
            )

    def _heuristic_scores(
        self,
        content: str,
        dimensions: list[QualityDimension],
        review_type: ReviewType,
    ) -> list[QualityScore]:
        """Generate heuristic-based scores when LLM is unavailable."""
        scores = []

        for dim in dimensions:
            score = 0.5
            feedback = ""
            suggestions = []

            if dim == QualityDimension.COMPLETENESS:
                # Check length as proxy
                if len(content) > 100:
                    score = min(0.9, 0.5 + len(content) / 1000)
                feedback = f"Content length: {len(content)} characters"

            elif dim == QualityDimension.CLARITY:
                # Check sentence structure
                sentences = content.split('.')
                avg_len = sum(len(s) for s in sentences) / len(sentences) if sentences else 0
                if 30 <= avg_len <= 100:
                    score = 0.8
                elif avg_len < 30:
                    score = 0.7
                    suggestions.append("Consider more detailed sentences")
                else:
                    score = 0.6
                    suggestions.append("Consider breaking into shorter sentences")
                feedback = f"Average sentence length: {avg_len:.0f} characters"

            elif dim == QualityDimension.STYLE:
                # Check for common issues
                issues = []
                if "  " in content:
                    issues.append("Double spaces found")
                if content != content.strip():
                    issues.append("Leading/trailing whitespace")
                score = max(0.3, 1.0 - len(issues) * 0.1)
                feedback = "Style check: " + (", ".join(issues) if issues else "OK")
                suggestions = issues

            else:
                feedback = "Heuristic evaluation (limited without LLM)"

            scores.append(QualityScore(
                dimension=dim,
                score=score,
                confidence=0.3,  # Low confidence for heuristics
                feedback=feedback,
                suggestions=suggestions,
            ))

        return scores

    def _calculate_overall_score(self, scores: list[QualityScore]) -> float:
        """Calculate weighted overall score."""
        if not scores:
            return 0.0

        weighted_sum = 0.0
        weight_total = 0.0

        for score in scores:
            weight = self.dimension_weights.get(score.dimension, 1.0)
            # Weight by confidence too
            effective_weight = weight * score.confidence
            weighted_sum += score.score * effective_weight
            weight_total += effective_weight

        return weighted_sum / weight_total if weight_total > 0 else 0.5

    def _generate_summary(self, scores: list[QualityScore]) -> str:
        """Generate a text summary from scores."""
        if not scores:
            return "No evaluation available"

        overall = self._calculate_overall_score(scores)

        if overall >= 0.9:
            quality = "Excellent"
        elif overall >= 0.7:
            quality = "Good"
        elif overall >= 0.5:
            quality = "Acceptable"
        else:
            quality = "Needs improvement"

        best = max(scores, key=lambda s: s.score)
        worst = min(scores, key=lambda s: s.score)

        return (
            f"{quality} overall ({overall:.0%}). "
            f"Strongest: {best.dimension.value} ({best.score:.0%}). "
            f"Weakest: {worst.dimension.value} ({worst.score:.0%})."
        )

    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response."""
        # Find JSON object
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            return text[start:end]
        return '{}'

    async def refine(
        self,
        content: str,
        review_type: ReviewType = ReviewType.OUTPUT,
        refiner_fn: Optional[Callable] = None,
        max_iterations: int = 3,
        target_score: float = 0.8,
        context: Optional[str] = None,
    ) -> RefinementResult:
        """
        Iteratively refine content until quality threshold is met.

        Args:
            content: Content to refine
            review_type: Type of content
            refiner_fn: Function to apply refinements (content, feedback) -> refined
            max_iterations: Maximum refinement cycles
            target_score: Stop when this score is achieved
            context: Additional context

        Returns:
            RefinementResult with history and final content
        """
        result = RefinementResult(
            original=content,
            refined=content,
        )

        current = content
        prev_review_id = None

        for i in range(max_iterations):
            # Review current version
            review = await self.review(
                current,
                review_type=review_type,
                context=context,
            )
            review.revision_number = i
            review.previous_review_id = prev_review_id
            result.reviews.append(review)

            logger.info(f"Refinement iteration {i+1}: score {review.overall_score:.2f}")

            # Check if target met
            if review.overall_score >= target_score:
                result.converged = True
                break

            # Apply refinements
            if refiner_fn:
                feedback = self._compile_feedback(review)
                try:
                    if asyncio.iscoroutinefunction(refiner_fn):
                        current = await refiner_fn(current, feedback)
                    else:
                        current = refiner_fn(current, feedback)
                except Exception as e:
                    logger.error(f"Refinement failed: {e}")
                    break
            elif self.llm_fn:
                # Use LLM for refinement
                current = await self._llm_refine(current, review, review_type)
            else:
                # Can't refine without functions
                break

            prev_review_id = review.id
            result.iterations = i + 1

        result.refined = current
        result.improvement = (
            result.reviews[-1].overall_score - result.reviews[0].overall_score
            if len(result.reviews) > 1 else 0.0
        )

        return result

    def _compile_feedback(self, review: Review) -> str:
        """Compile review into actionable feedback."""
        feedback_parts = [
            f"Overall score: {review.overall_score:.0%}",
            "",
            "Areas to improve:",
        ]

        for score in sorted(review.scores, key=lambda s: s.score):
            if score.score < 0.8 and score.suggestions:
                feedback_parts.append(f"- {score.dimension.value}: {score.suggestions[0]}")

        if review.critical_issues:
            feedback_parts.extend([
                "",
                "Critical issues:",
                *[f"- {issue}" for issue in review.critical_issues],
            ])

        return "\n".join(feedback_parts)

    async def _llm_refine(
        self,
        content: str,
        review: Review,
        review_type: ReviewType,
    ) -> str:
        """Use LLM to apply refinements."""
        feedback = self._compile_feedback(review)

        prompt = f"""Improve this {review_type.value} based on the feedback.

Original:
```
{content}
```

Feedback:
{feedback}

Return the improved version only, no explanations."""

        try:
            if asyncio.iscoroutinefunction(self.llm_fn):
                response = await self.llm_fn(prompt)
            else:
                response = self.llm_fn(prompt)

            # Clean up response
            refined = response.strip()

            # Remove markdown code blocks if present
            if refined.startswith("```"):
                lines = refined.split("\n")
                if len(lines) > 2:
                    refined = "\n".join(lines[1:-1])

            return refined

        except Exception as e:
            logger.error(f"LLM refinement failed: {e}")
            return content

    async def compare(
        self,
        artifacts: list[str],
        review_type: ReviewType = ReviewType.OUTPUT,
        context: Optional[str] = None,
    ) -> dict:
        """
        Compare multiple artifacts and rank them.

        Returns ranking with scores and analysis.
        """
        reviews = []

        for artifact in artifacts:
            review = await self.review(artifact, review_type, context)
            reviews.append(review)

        # Rank by score
        ranked = sorted(
            enumerate(reviews),
            key=lambda x: x[1].overall_score,
            reverse=True,
        )

        best_idx = ranked[0][0]
        worst_idx = ranked[-1][0]

        comparison = {
            "ranking": [
                {
                    "rank": i + 1,
                    "index": idx,
                    "score": review.overall_score,
                    "summary": review.summary,
                }
                for i, (idx, review) in enumerate(ranked)
            ],
            "best": {
                "index": best_idx,
                "score": reviews[best_idx].overall_score,
                "strengths": [
                    s.dimension.value for s in reviews[best_idx].scores
                    if s.score >= 0.8
                ],
            },
            "worst": {
                "index": worst_idx,
                "score": reviews[worst_idx].overall_score,
                "weaknesses": [
                    s.dimension.value for s in reviews[worst_idx].scores
                    if s.score < 0.5
                ],
            },
        }

        return comparison

    async def verify_requirements(
        self,
        content: str,
        requirements: list[str],
        review_type: ReviewType = ReviewType.OUTPUT,
    ) -> dict:
        """
        Verify content meets specific requirements.

        Returns per-requirement pass/fail with details.
        """
        results = {
            "passed": 0,
            "failed": 0,
            "requirements": [],
        }

        if self.llm_fn:
            for req in requirements:
                result = await self._check_requirement(content, req, review_type)
                results["requirements"].append(result)
                if result["met"]:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
        else:
            # Simple keyword matching
            for req in requirements:
                keywords = req.lower().split()
                content_lower = content.lower()
                met = any(kw in content_lower for kw in keywords if len(kw) > 3)

                results["requirements"].append({
                    "requirement": req,
                    "met": met,
                    "confidence": 0.3,
                    "evidence": "Keyword match" if met else "Keywords not found",
                })
                if met:
                    results["passed"] += 1
                else:
                    results["failed"] += 1

        results["pass_rate"] = results["passed"] / len(requirements) if requirements else 0

        return results

    async def _check_requirement(
        self,
        content: str,
        requirement: str,
        review_type: ReviewType,
    ) -> dict:
        """Check if content meets a specific requirement."""
        prompt = f"""Does this {review_type.value} meet the following requirement?

Requirement: {requirement}

Content:
```
{content[:2000]}
```

Return JSON:
{{"met": true/false, "confidence": 0.0-1.0, "evidence": "explanation"}}"""

        try:
            if asyncio.iscoroutinefunction(self.llm_fn):
                response = await self.llm_fn(prompt)
            else:
                response = self.llm_fn(prompt)

            data = json.loads(self._extract_json(response))

            return {
                "requirement": requirement,
                "met": data.get("met", False),
                "confidence": data.get("confidence", 0.5),
                "evidence": data.get("evidence", ""),
            }

        except Exception as e:
            logger.error(f"Requirement check failed: {e}")
            return {
                "requirement": requirement,
                "met": False,
                "confidence": 0.1,
                "evidence": f"Check failed: {e}",
            }

    def get_stats(self) -> dict:
        """Get critic statistics."""
        if not self.reviews:
            return {"total_reviews": 0}

        scores = [r.overall_score for r in self.reviews.values()]

        return {
            "total_reviews": len(self.reviews),
            "avg_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "by_type": {
                t.value: len([r for r in self.reviews.values() if r.artifact_type == t])
                for t in ReviewType
            },
        }

    # =========================================================================
    # COLLABORATION METRICS (AGI Upgrade)
    # =========================================================================

    def _init_collaboration_tracking(self):
        """Initialize collaboration tracking state."""
        if not hasattr(self, '_collaborations'):
            self._collaborations: dict[str, CollaborationMetrics] = {}
            self._handoff_log: list[HandoffEvent] = []
            self._agent_performance: dict[str, dict] = {}  # agent_id -> stats

    def start_collaboration(
        self,
        collaboration_id: str,
        agents: list[str],
    ) -> CollaborationMetrics:
        """
        Start tracking a new collaboration session.

        Args:
            collaboration_id: Unique identifier for this collaboration
            agents: List of agent IDs participating

        Returns:
            CollaborationMetrics instance for tracking
        """
        self._init_collaboration_tracking()

        metrics = CollaborationMetrics(
            collaboration_id=collaboration_id,
            agents_involved=agents,
            started_at=datetime.now(),
        )

        self._collaborations[collaboration_id] = metrics
        logger.info(f"Started collaboration tracking: {collaboration_id} with {len(agents)} agents")

        return metrics

    def record_handoff(
        self,
        collaboration_id: str,
        from_agent: str,
        to_agent: str,
        task_context: str,
        success: bool = True,
        latency_ms: float = 0.0,
        context_preserved_ratio: float = 1.0,
        failure_reason: Optional[str] = None,
    ) -> HandoffEvent:
        """
        Record a handoff between agents.

        Args:
            collaboration_id: The collaboration this handoff belongs to
            from_agent: Source agent ID
            to_agent: Target agent ID
            task_context: Brief description of what was handed off
            success: Whether the handoff succeeded
            latency_ms: Time taken for handoff
            context_preserved_ratio: How much context was preserved (0-1)
            failure_reason: Reason if handoff failed

        Returns:
            HandoffEvent record
        """
        self._init_collaboration_tracking()

        event = HandoffEvent(
            from_agent=from_agent,
            to_agent=to_agent,
            task_context=task_context,
            timestamp=datetime.now(),
            latency_ms=latency_ms,
            success=success,
            failure_reason=failure_reason,
            context_preserved_ratio=context_preserved_ratio,
        )

        self._handoff_log.append(event)

        # Update collaboration metrics
        if collaboration_id in self._collaborations:
            metrics = self._collaborations[collaboration_id]
            metrics.total_handoffs += 1
            if success:
                metrics.successful_handoffs += 1
            else:
                metrics.failed_handoffs += 1

            # Update running average of latency
            n = metrics.total_handoffs
            metrics.handoff_latency_avg_ms = (
                (metrics.handoff_latency_avg_ms * (n - 1) + latency_ms) / n
            )

        return event

    async def record_contribution(
        self,
        collaboration_id: str,
        agent_id: str,
        contribution: str,
        review_type: ReviewType = ReviewType.OUTPUT,
    ) -> QualityScore:
        """
        Record and evaluate an individual agent's contribution.

        Args:
            collaboration_id: The collaboration this contribution belongs to
            agent_id: The contributing agent
            contribution: The content contributed
            review_type: Type of contribution

        Returns:
            QualityScore for this contribution
        """
        self._init_collaboration_tracking()

        # Review the contribution
        review = await self.review(contribution, review_type)

        # Store individual score
        if collaboration_id in self._collaborations:
            self._collaborations[collaboration_id].individual_scores[agent_id] = review.overall_score

        # Update agent performance tracking
        if agent_id not in self._agent_performance:
            self._agent_performance[agent_id] = {
                "total_contributions": 0,
                "avg_score": 0.0,
                "scores": [],
            }

        stats = self._agent_performance[agent_id]
        stats["total_contributions"] += 1
        stats["scores"].append(review.overall_score)
        stats["avg_score"] = sum(stats["scores"]) / len(stats["scores"])

        logger.debug(f"Agent {agent_id} contribution score: {review.overall_score:.2f}")

        return QualityScore(
            dimension=QualityDimension.COLLABORATION,
            score=review.overall_score,
            confidence=0.8,
            feedback=review.summary,
            suggestions=review.improvement_priorities,
        )

    async def evaluate_collective_output(
        self,
        collaboration_id: str,
        collective_output: str,
        individual_contributions: Optional[dict[str, str]] = None,
        review_type: ReviewType = ReviewType.OUTPUT,
    ) -> CollaborationMetrics:
        """
        Evaluate the collective output and calculate synergy.

        Args:
            collaboration_id: The collaboration being evaluated
            collective_output: The final combined output
            individual_contributions: Optional dict of agent_id -> contribution
            review_type: Type of output

        Returns:
            Updated CollaborationMetrics with synergy scores
        """
        self._init_collaboration_tracking()

        if collaboration_id not in self._collaborations:
            self.start_collaboration(collaboration_id, [])

        metrics = self._collaborations[collaboration_id]

        # Review collective output
        collective_review = await self.review(collective_output, review_type)
        metrics.collective_score = collective_review.overall_score

        # Review individual contributions if provided
        if individual_contributions:
            for agent_id, contribution in individual_contributions.items():
                if agent_id not in metrics.individual_scores:
                    await self.record_contribution(
                        collaboration_id, agent_id, contribution, review_type
                    )

        # Calculate synergy
        metrics.synergy_score = metrics.synergy_coefficient()

        # Evaluate coherence (consistency across contributions)
        if len(metrics.individual_scores) > 1:
            scores = list(metrics.individual_scores.values())
            mean_score = sum(scores) / len(scores)
            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            # Coherence is inverse of variance (high coherence = low variance)
            metrics.coherence_score = max(0, 1 - (variance * 4))  # Scale variance

        # Calculate contribution ratios
        total_chars = sum(
            len(c) for c in (individual_contributions or {}).values()
        )
        if total_chars > 0 and individual_contributions:
            for agent_id, contribution in individual_contributions.items():
                metrics.contribution_ratios[agent_id] = len(contribution) / total_chars

        # Check for emergent insights (collective better than best individual)
        if metrics.individual_scores:
            best_individual = max(metrics.individual_scores.values())
            if metrics.collective_score > best_individual * 1.1:  # 10% better
                metrics.emergent_insights.append(
                    f"Collective ({metrics.collective_score:.2f}) exceeded best "
                    f"individual ({best_individual:.2f}) by "
                    f"{((metrics.collective_score / best_individual) - 1) * 100:.1f}%"
                )

        # Mark completion
        metrics.completed_at = datetime.now()

        logger.info(
            f"Collaboration {collaboration_id} evaluated: "
            f"collective={metrics.collective_score:.2f}, "
            f"synergy={metrics.synergy_score:+.2f}"
        )

        return metrics

    async def detect_cross_pollination(
        self,
        collaboration_id: str,
        agent_outputs: dict[str, list[str]],
    ) -> int:
        """
        Detect ideas that spread between agents (cross-pollination).

        Args:
            collaboration_id: The collaboration
            agent_outputs: Dict of agent_id -> list of outputs over time

        Returns:
            Count of cross-pollinated ideas detected
        """
        self._init_collaboration_tracking()

        if collaboration_id not in self._collaborations:
            return 0

        metrics = self._collaborations[collaboration_id]
        cross_pollination_count = 0

        # Extract key phrases from each agent's outputs
        agent_phrases: dict[str, set[str]] = {}

        for agent_id, outputs in agent_outputs.items():
            phrases = set()
            for output in outputs:
                # Simple phrase extraction (could use NLP in production)
                words = output.lower().split()
                # Extract 2-3 word phrases
                for i in range(len(words) - 1):
                    if len(words[i]) > 3 and len(words[i + 1]) > 3:
                        phrases.add(f"{words[i]} {words[i + 1]}")
            agent_phrases[agent_id] = phrases

        # Find phrases that appear in multiple agents' later outputs
        agents = list(agent_outputs.keys())
        for i, agent1 in enumerate(agents):
            for agent2 in agents[i + 1:]:
                # Check for shared phrases
                shared = agent_phrases[agent1] & agent_phrases[agent2]
                cross_pollination_count += len(shared)

        metrics.cross_pollination_count = cross_pollination_count

        logger.debug(
            f"Collaboration {collaboration_id}: "
            f"{cross_pollination_count} cross-pollinated concepts detected"
        )

        return cross_pollination_count

    def get_collaboration_metrics(self, collaboration_id: str) -> Optional[CollaborationMetrics]:
        """Get metrics for a specific collaboration."""
        self._init_collaboration_tracking()
        return self._collaborations.get(collaboration_id)

    def get_agent_collaboration_stats(self, agent_id: str) -> dict:
        """Get collaboration statistics for a specific agent."""
        self._init_collaboration_tracking()

        if agent_id not in self._agent_performance:
            return {"agent_id": agent_id, "total_contributions": 0}

        stats = self._agent_performance[agent_id]

        # Calculate handoff stats for this agent
        handoffs_initiated = sum(
            1 for h in self._handoff_log if h.from_agent == agent_id
        )
        handoffs_received = sum(
            1 for h in self._handoff_log if h.to_agent == agent_id
        )
        handoff_success_rate = (
            sum(1 for h in self._handoff_log if h.from_agent == agent_id and h.success)
            / handoffs_initiated if handoffs_initiated > 0 else 1.0
        )

        return {
            "agent_id": agent_id,
            "total_contributions": stats["total_contributions"],
            "avg_contribution_score": stats["avg_score"],
            "handoffs_initiated": handoffs_initiated,
            "handoffs_received": handoffs_received,
            "handoff_success_rate": handoff_success_rate,
            "recent_scores": stats["scores"][-5:],  # Last 5 scores
        }

    def get_collaboration_summary(self) -> dict:
        """Get summary of all collaborations."""
        self._init_collaboration_tracking()

        if not self._collaborations:
            return {"total_collaborations": 0}

        all_metrics = list(self._collaborations.values())
        completed = [m for m in all_metrics if m.completed_at]

        avg_synergy = (
            sum(m.synergy_score for m in completed) / len(completed)
            if completed else 0.0
        )
        avg_collective = (
            sum(m.collective_score for m in completed) / len(completed)
            if completed else 0.0
        )

        return {
            "total_collaborations": len(all_metrics),
            "completed_collaborations": len(completed),
            "avg_synergy_score": avg_synergy,
            "avg_collective_score": avg_collective,
            "total_handoffs": len(self._handoff_log),
            "handoff_success_rate": (
                sum(1 for h in self._handoff_log if h.success)
                / len(self._handoff_log) if self._handoff_log else 1.0
            ),
            "agents_tracked": len(self._agent_performance),
            "emergent_insights_total": sum(
                len(m.emergent_insights) for m in all_metrics
            ),
        }

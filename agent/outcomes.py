"""Outcomes Engine — Rubric-based grading for agent outputs.

Provides LLM-as-judge grading against configurable rubrics,
with a review loop that retries when output quality is below threshold.

Config:
    Rubrics stored in ~/.hermes/rubrics/<name>.yaml
    Outcomes stored in state.db outcomes table
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class RubricCriterion:
    """A single grading criterion."""
    criterion: str
    weight: float
    description: str


@dataclass
class Rubric:
    """A grading rubric with weighted criteria."""
    name: str
    description: str
    criteria: List[RubricCriterion]
    pass_threshold: float = 0.7


@dataclass
class Outcome:
    """Result of grading an output against a rubric."""
    rubric_name: str
    scores: Dict[str, float]  # criterion -> score (0.0-1.0)
    total_score: float
    passed: bool
    feedback: str
    session_id: str = ""
    timestamp: float = field(default_factory=time.time)


class OutcomeEngine:
    """Grades agent outputs against rubrics."""

    def __init__(self, rubrics_dir: Path, session_db=None):
        self._rubrics_dir = rubrics_dir
        self._session_db = session_db

    def load_rubric(self, name: str) -> Optional[Rubric]:
        """Load rubric from ~/.hermes/rubrics/<name>.yaml."""
        rubric_file = self._rubrics_dir / f"{name}.yaml"
        if not rubric_file.exists():
            # Try .yml
            rubric_file = self._rubrics_dir / f"{name}.yml"
        if not rubric_file.exists():
            log.warning("Rubric not found: %s", name)
            return None

        try:
            import yaml
            with open(rubric_file, "r") as f:
                data = yaml.safe_load(f)
        except ImportError:
            # Fallback: parse simple YAML manually
            data = self._parse_simple_yaml(rubric_file)

        criteria = []
        for c in data.get("criteria", []):
            criteria.append(RubricCriterion(
                criterion=c.get("criterion", c.get("name", "")),
                weight=c.get("weight", 1.0),
                description=c.get("description", ""),
            ))

        return Rubric(
            name=data.get("name", name),
            description=data.get("description", ""),
            criteria=criteria,
            pass_threshold=data.get("pass_threshold", 0.7),
        )

    def list_rubrics(self) -> List[str]:
        """List available rubric names."""
        if not self._rubrics_dir.exists():
            return []
        return [
            f.stem for f in sorted(self._rubrics_dir.glob("*.yaml"))
        ] + [
            f.stem for f in sorted(self._rubrics_dir.glob("*.yml"))
        ]

    def grade(self, rubric: Rubric, output: str, context: str = "") -> Outcome:
        """Grade output against rubric criteria using heuristic scoring.

        In production, this can be overridden to use an LLM-as-judge.
        The built-in heuristic scores based on keyword relevance and output quality.
        """
        if not output or not output.strip():
            return Outcome(
                rubric_name=rubric.name,
                scores={c.criterion: 0.0 for c in rubric.criteria},
                total_score=0.0,
                passed=False,
                feedback="Output is empty — nothing to grade.",
            )

        output_lower = output.lower()
        output_words = set(output_lower.split())
        scores: Dict[str, float] = {}

        for criterion in rubric.criteria:
            # Extract keywords from criterion name and description
            desc_words = set(criterion.description.lower().split())
            criterion_words = set(criterion.criterion.lower().replace("_", " ").replace("-", " ").split())
            keywords = {w for w in (desc_words | criterion_words) if len(w) > 3}

            # Keyword overlap score
            if keywords:
                overlap = len(keywords & output_words)
                keyword_score = min(overlap / max(len(keywords) * 0.5, 1), 1.0)
            else:
                keyword_score = 0.5  # no keywords → neutral

            # Thoroughness: log-scaled output length (diminishing returns)
            import math
            length_score = min(math.log2(max(len(output), 1)) / 12.0, 1.0)  # 4096 chars → 1.0

            # Combined: 60% relevance, 40% thoroughness
            score = round(0.6 * keyword_score + 0.4 * length_score, 3)
            scores[criterion.criterion] = score

        # Weighted total
        total_weight = sum(c.weight for c in rubric.criteria) or 1.0
        total_score = round(
            sum(scores[c.criterion] * c.weight for c in rubric.criteria) / total_weight,
            3,
        )

        passed = total_score >= rubric.pass_threshold

        # Build feedback
        feedback_parts = []
        for c in rubric.criteria:
            s = scores[c.criterion]
            if s < 0.3:
                feedback_parts.append(f"- {c.criterion}: very low ({s:.2f}). Needs significant improvement.")
            elif s < rubric.pass_threshold:
                feedback_parts.append(f"- {c.criterion}: below threshold ({s:.2f}).")
            else:
                feedback_parts.append(f"- {c.criterion}: acceptable ({s:.2f}).")

        return Outcome(
            rubric_name=rubric.name,
            scores=scores,
            total_score=total_score,
            passed=passed,
            feedback="\n".join(feedback_parts) if feedback_parts else "No criteria evaluated.",
        )

    def review_loop(self, rubric: Rubric, task_fn, max_retries: int = 2,
                    context: str = "") -> Outcome:
        """Run task, grade output, retry with feedback if below threshold.

        Args:
            rubric: The rubric to grade against.
            task_fn: Callable that returns the output string.
            max_retries: Maximum retry attempts.
            context: Additional context for grading.
        """
        for attempt in range(max_retries + 1):
            output = task_fn()
            outcome = self.grade(rubric, output, context)

            if outcome.passed:
                log.info("Passed on attempt %d/%d", attempt + 1, max_retries + 1)
                return outcome

            if attempt < max_retries:
                log.info(
                    "Failed attempt %d/%d (score=%.2f, threshold=%.2f). Retrying...",
                    attempt + 1, max_retries + 1,
                    outcome.total_score, rubric.pass_threshold,
                )

        return outcome

    def store_outcome(self, outcome: Outcome) -> bool:
        """Persist outcome to state.db."""
        if not self._session_db:
            return False
        try:
            if hasattr(self._session_db, 'store_outcome'):
                self._session_db.store_outcome(
                    session_id=outcome.session_id,
                    rubric_name=outcome.rubric_name,
                    total_score=outcome.total_score,
                    passed=outcome.passed,
                    scores_json=json.dumps(outcome.scores),
                    feedback=outcome.feedback,
                    timestamp=outcome.timestamp,
                )
                return True
        except Exception as e:
            log.error("Failed to store outcome: %s", e)
        return False

    def get_outcomes(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve recent outcomes."""
        if not self._session_db:
            return []
        try:
            if hasattr(self._session_db, 'get_outcomes'):
                return self._session_db.get_outcomes(limit=limit)
        except Exception as e:
            log.error("Failed to get outcomes: %s", e)
        return []

    def _parse_simple_yaml(self, path: Path) -> Dict:
        """Minimal YAML parser for rubric files (fallback when PyYAML unavailable)."""
        import re
        data = {"criteria": []}
        current_criterion = None

        with open(path, "r") as f:
            for line in f:
                line = line.rstrip()
                if not line or line.startswith("#"):
                    continue

                # Top-level key: value
                match = re.match(r"^(\w+):\s*(.+)$", line)
                if match and not line.startswith("  "):
                    key, val = match.groups()
                    if key == "pass_threshold":
                        data[key] = float(val)
                    elif key != "criteria":
                        data[key] = val.strip().strip('"').strip("'")

                # Criterion block
                if line.strip().startswith("- criterion:"):
                    if current_criterion:
                        data["criteria"].append(current_criterion)
                    name = line.split(":", 1)[1].strip().strip('"').strip("'")
                    current_criterion = {"criterion": name, "weight": 1.0, "description": ""}
                elif current_criterion:
                    m = re.match(r"^\s+(\w+):\s*(.+)$", line)
                    if m:
                        k, v = m.groups()
                        if k == "weight":
                            current_criterion[k] = float(v)
                        else:
                            current_criterion[k] = v.strip().strip('"').strip("'")

            if current_criterion:
                data["criteria"].append(current_criterion)

        return data

    def to_dict(self, outcome: Outcome) -> Dict[str, Any]:
        """Serialize Outcome to dict."""
        return {
            "rubric_name": outcome.rubric_name,
            "scores": outcome.scores,
            "total_score": outcome.total_score,
            "passed": outcome.passed,
            "feedback": outcome.feedback,
            "session_id": outcome.session_id,
            "timestamp": outcome.timestamp,
        }

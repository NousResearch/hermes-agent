"""Kanban size gate - deterministic pre-flight estimator for card complexity.

This module provides a cheap, deterministic estimator that does NOT invoke an LLM.
It measures card complexity using structural signals: acceptance criteria count,
distinct files/subsystems named, and distinct deliverables.

The gate hooks into two paths:
1. **Card creation** (kanban_db.create_task): if a card is oversized and triage=False,
   it is routed to triage for decomposition instead of going straight to ready.
2. **Decomposition** (kanban_db_graph.decompose_triage_task): every emitted child
   is measured, and oversized children are rejected with a typed error.

Estimator formula:
    score = (ac_count * 3) + (file_count * 2) + (deliverable_count * 1)

Where:
- ac_count: number of acceptance criteria items (numbered lists, bullet points
  in "Acceptance Criteria" sections)
- file_count: number of distinct files/subsystems named (paths with extensions
  like .py, .ts, .md, etc.)
- deliverable_count: number of distinct deliverables (sections with "##" headers
  that imply separate work units)

Default threshold: 30 (configurable via kanban.size_gate.threshold in config.yaml)

A card scoring above the threshold is considered oversized and must be decomposed
before dispatch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# Default threshold for the size gate. Cards scoring above this are oversized.
# Configurable via kanban.size_gate.threshold in config.yaml.
DEFAULT_SIZE_GATE_THRESHOLD = 30

# Weights for the estimator formula
WEIGHT_ACCEPTANCE_CRITERIA = 3
WEIGHT_FILE = 2
WEIGHT_DELIVERABLE = 1


@dataclass
class SizeEstimate:
    """Result of estimating a card's size."""

    score: int
    acceptance_criteria_count: int
    file_count: int
    deliverable_count: int

    def is_oversized(self, threshold: int) -> bool:
        """Return True if this estimate exceeds the threshold."""
        return self.score > threshold


class OversizedChildError(Exception):
    """Raised when a decomposed child exceeds the size gate threshold.

    This is a typed failure that the decomposer must raise (not silently pass)
    when a child task is too large for a single worker.
    """

    def __init__(self, child_index: int, child_title: str, estimate: SizeEstimate, threshold: int):
        self.child_index = child_index
        self.child_title = child_title
        self.estimate = estimate
        self.threshold = threshold
        super().__init__(
            f"Child task {child_index} ({child_title!r}) is oversized: "
            f"score={estimate.score} exceeds threshold={threshold}. "
            f"Break it down further before emitting."
        )


def estimate_card_size(title: Optional[str], body: Optional[str]) -> SizeEstimate:
    """Estimate the size of a card from its title and body.

    This is a deterministic, LLM-free estimator that counts structural signals:
    - Acceptance criteria items (numbered lists, bullets in AC sections)
    - Distinct files/subsystems named (paths with extensions)
    - Distinct deliverables (## headers implying separate work units)

    Returns a SizeEstimate with the computed score and breakdown.
    """
    if not body:
        # Minimal card with just a title
        return SizeEstimate(score=0, acceptance_criteria_count=0, file_count=0, deliverable_count=0)

    text = body
    ac_count = 0
    file_count = 0
    deliverable_count = 0

    # Count acceptance criteria items
    # Look for numbered lists (1., 2., etc.) or bullet points (-, *) in AC sections
    # or standalone numbered lists that look like criteria
    ac_section_match = re.search(
        r"(?i)(acceptance\s+criteria|acceptance|criteria)\s*:?\s*\n(.*?)(?=\n##|\Z)",
        text,
        re.DOTALL,
    )
    if ac_section_match:
        ac_section = ac_section_match.group(2)
        # Count numbered items (1., 2., etc.) or bullets (-, *)
        ac_count = len(re.findall(r"^\s*(\d+\.|[-*])\s+", ac_section, re.MULTILINE))
    else:
        # No explicit AC section - count standalone numbered lists that look like criteria
        # Look for sequences of numbered items
        numbered_blocks = re.findall(r"(?:^|\n)(\s*\d+\..+(?:\n\s*\d+\..+)*)", text)
        for block in numbered_blocks:
            items = re.findall(r"^\s*\d+\.", block, re.MULTILINE)
            if len(items) >= 3:  # At least 3 items to count as AC
                ac_count += len(items)

    # Count distinct files/subsystems named
    # Look for file paths with extensions (.py, .ts, .md, .yaml, etc.)
    file_pattern = r"\b[\w\-/]+\.(?:py|ts|tsx|js|jsx|md|yaml|yml|json|toml|sql|sh|rb|go|rs|java|c|cpp|h|hpp)\b"
    files = set(re.findall(file_pattern, text, re.IGNORECASE))
    file_count = len(files)

    # Count distinct deliverables (## headers)
    # Each ## header implies a separate work unit or section
    headers = re.findall(r"^##\s+(.+)$", text, re.MULTILINE)
    deliverable_count = len(headers)

    # Compute the weighted score
    score = (
        ac_count * WEIGHT_ACCEPTANCE_CRITERIA
        + file_count * WEIGHT_FILE
        + deliverable_count * WEIGHT_DELIVERABLE
    )

    return SizeEstimate(
        score=score,
        acceptance_criteria_count=ac_count,
        file_count=file_count,
        deliverable_count=deliverable_count,
    )


def load_size_gate_threshold() -> int:
    """Load the size gate threshold from config, falling back to the default.

    Config key: kanban.size_gate.threshold
    """
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
        kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
        threshold = kanban_cfg.get("size_gate", {}).get("threshold")
        if isinstance(threshold, (int, float)) and threshold > 0:
            return int(threshold)
    except Exception:
        pass
    return DEFAULT_SIZE_GATE_THRESHOLD


def should_route_to_triage(title: Optional[str], body: Optional[str]) -> tuple[bool, SizeEstimate]:
    """Determine if a card should be routed to triage based on its size.

    Returns (should_route, estimate) where should_route is True if the card
    is oversized and should be decomposed before dispatch.
    """
    threshold = load_size_gate_threshold()
    estimate = estimate_card_size(title, body)
    return estimate.is_oversized(threshold), estimate


def record_size_estimate(conn, task_id: str, estimate: SizeEstimate) -> None:
    """Record the size estimate on the card as a task event.

    This makes the estimate machine-readable for later audits and the dispatcher.
    """
    from hermes_cli import kanban_db as kb

    try:
        kb._append_event(
            conn,
            task_id,
            "size_estimate",
            {
                "score": estimate.score,
                "acceptance_criteria_count": estimate.acceptance_criteria_count,
                "file_count": estimate.file_count,
                "deliverable_count": estimate.deliverable_count,
            },
        )
    except Exception:
        # Recording failure should not block the card
        pass

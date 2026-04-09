"""Structured review engine for completed tasks.

Analyzes a task-completion payload and produces a typed review result
indicating what follow-up actions (memory save, skill save) are warranted.

Also provides policy-based memory writeback: conservative, deterministic
routing of classified memory-write candidates to the built-in memory store.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory write candidate categories and routing
# ---------------------------------------------------------------------------

#: Valid categories for memory write candidates.
MEMORY_WRITE_CATEGORIES = frozenset({
    "user_facts",
    "environment_facts",
    "workflow_facts",
    "ignore",
})

#: Deterministic mapping from category to built-in memory target.
#: ``"ignore"`` is intentionally absent — it produces no write.
_CATEGORY_TO_TARGET: Dict[str, str] = {
    "user_facts": "user",
    "environment_facts": "memory",
    "workflow_facts": "memory",
}


@dataclass(frozen=True)
class MemoryWriteCandidate:
    """A candidate fact for built-in memory writeback.

    Attributes:
        category: One of :data:`MEMORY_WRITE_CATEGORIES`.
        content: The text to persist.
        source: Where this candidate originated (for debugging/logging).
    """

    category: str
    content: str
    source: str = ""


@dataclass(frozen=True)
class TaskReviewResult:
    """Typed, immutable result from reviewing a completed task.

    Attributes:
        should_review_memory: Whether the task warrants a memory review pass.
        should_review_skills: Whether the task warrants a skill review pass.
        review_reasons: Human-readable reasons explaining why review is needed.
        payload: The original task-completion payload (kept for downstream use).
        memory_write_candidates: Classified candidates for built-in memory
            writeback.  May be empty even when *should_review_memory* is True
            (the background LLM review handles the rest).
    """

    should_review_memory: bool
    should_review_skills: bool
    review_reasons: List[str]
    payload: Dict[str, Any]
    memory_write_candidates: List[MemoryWriteCandidate] = field(default_factory=list)


def review_completed_task(task_payload: Dict[str, Any]) -> TaskReviewResult:
    """Analyze a completed-task payload and return a structured review result.

    This is a pure function — it reads the payload and produces a decision
    without side effects.  The caller decides what to do with the result.

    Args:
        task_payload: Dict produced by ``AIAgent._build_task_completion_payload``.

    Returns:
        A :class:`TaskReviewResult` with review recommendations.

    Raises:
        ValueError: If ``task_payload`` is ``None`` or missing required keys.
    """
    if not task_payload:
        raise ValueError("task_payload must be a non-empty dict")

    trigger_reasons: List[str] = task_payload.get("trigger_reasons") or []
    tools_used: List[str] = task_payload.get("tools_used") or []

    reasons: List[str] = []
    should_memory = False
    should_skills = False

    # Explicit memory request from user always triggers memory review.
    if "explicit_memory_request" in trigger_reasons:
        should_memory = True
        reasons.append("user explicitly asked to remember/save")

    # Tool use indicates a non-trivial task — worth checking for skill patterns
    # and also for memory-worthy context (user preferences revealed during work).
    if "tool_used" in trigger_reasons:
        should_skills = True
        reasons.append("tools were used during the task")

        # Memory-specific tools hint that the user interacted with their
        # knowledge store — review for additional memory-worthy content.
        _memory_tools = {"memory", "memory_manage", "user_profile"}
        if _memory_tools & set(tools_used):
            should_memory = True
            reasons.append("memory/profile tools were invoked")

    # If we have no reasons at all, nothing to review.
    if not reasons:
        return TaskReviewResult(
            should_review_memory=False,
            should_review_skills=False,
            review_reasons=[],
            payload=task_payload,
        )

    # Extract deterministic memory-write candidates from the payload.
    candidates = extract_memory_candidates(task_payload)

    return TaskReviewResult(
        should_review_memory=should_memory,
        should_review_skills=should_skills,
        review_reasons=reasons,
        payload=task_payload,
        memory_write_candidates=candidates,
    )


# ---------------------------------------------------------------------------
# Memory candidate extraction (conservative / deterministic)
# ---------------------------------------------------------------------------

_MEMORY_PREFIXES = (
    "remember this:",
    "remember that:",
    "save this:",
    "save that:",
    "remember this",
    "remember that",
    "save this",
    "save that",
)


def _strip_memory_prefix(text: str) -> str:
    """Strip common 'remember/save' prefixes from a user message.

    Returns the remainder after stripping, or the original text if no
    recognized prefix is found.
    """
    lower = text.lower().strip()
    for prefix in _MEMORY_PREFIXES:
        if lower.startswith(prefix):
            remainder = text[len(prefix):].strip()
            if remainder:
                return remainder
    return text


def extract_memory_candidates(
    task_payload: Dict[str, Any],
) -> List[MemoryWriteCandidate]:
    """Extract deterministic memory-write candidates from a task payload.

    Conservative by design — only produces candidates when there is clear,
    unambiguous signal in the payload.  Prefers no-write over speculative
    write.

    Returns:
        A (possibly empty) list of :class:`MemoryWriteCandidate`.
    """
    if not task_payload:
        return []

    candidates: List[MemoryWriteCandidate] = []
    trigger_reasons: List[str] = task_payload.get("trigger_reasons") or []

    # ── Explicit memory request ──────────────────────────────────────────
    # The user said "remember this" / "save that" — persist the user's own
    # statement as a user fact.  We strip the command prefix so the stored
    # entry reads naturally.
    if "explicit_memory_request" in trigger_reasons:
        user_msg = (task_payload.get("original_user_message") or "").strip()
        if user_msg:
            content = _strip_memory_prefix(user_msg)
            if content:
                candidates.append(MemoryWriteCandidate(
                    category="user_facts",
                    content=content,
                    source="explicit_memory_request",
                ))

    return candidates


# ---------------------------------------------------------------------------
# Centralized memory writeback
# ---------------------------------------------------------------------------

def _content_key(category: str, content: str) -> str:
    """Produce a normalized key for same-turn duplicate detection."""
    return f"{category}:{content.strip().lower()}"


def apply_memory_writeback(
    candidates: List[MemoryWriteCandidate],
    memory_store: Any,
    *,
    written_keys: Optional[Set[str]] = None,
) -> List[MemoryWriteCandidate]:
    """Route classified candidates to the built-in memory store.

    Deterministic routing:
    - ``user_facts``        → ``memory_store.add("user", ...)``
    - ``environment_facts`` → ``memory_store.add("memory", ...)``
    - ``workflow_facts``    → ``memory_store.add("memory", ...)``
    - ``ignore``            → no write

    Duplicate suppression: normalizes ``(category, content)`` into a key
    and skips any candidate whose key already appears in *written_keys*.
    New writes are added to *written_keys* in place so the caller can
    accumulate across multiple calls within the same turn.

    Args:
        candidates: Classified memory-write candidates.
        memory_store: A ``MemoryStore`` instance (or compatible object with
            an ``add(target, content)`` method returning a dict with a
            ``"success"`` key).
        written_keys: Mutable set for cross-call duplicate suppression.
            Created internally if ``None``.

    Returns:
        The subset of *candidates* that were successfully written.
    """
    if not candidates or memory_store is None:
        return []

    if written_keys is None:
        written_keys = set()

    written: List[MemoryWriteCandidate] = []
    for candidate in candidates:
        # Skip ignored or unrecognized categories.
        if candidate.category not in _CATEGORY_TO_TARGET:
            continue

        # Same-turn duplicate suppression.
        key = _content_key(candidate.category, candidate.content)
        if key in written_keys:
            logger.debug("Duplicate suppressed for key: %s", key)
            continue

        target = _CATEGORY_TO_TARGET[candidate.category]
        try:
            result = memory_store.add(target, candidate.content)
        except Exception:
            logger.debug(
                "memory_store.add(%s) raised for category=%s",
                target, candidate.category, exc_info=True,
            )
            continue

        if result.get("success"):
            written_keys.add(key)
            written.append(candidate)
            logger.debug(
                "Memory writeback: %s → %s (%s)",
                candidate.category, target, candidate.source,
            )
        else:
            logger.debug(
                "Memory writeback skipped for %s: %s",
                candidate.category, result.get("error", "unknown"),
            )

    return written

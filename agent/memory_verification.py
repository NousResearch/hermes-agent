"""Writer-isolated verification gate for memory consolidation (issue #112102).

Extraction at a session boundary turns one writer's claim into durable knowledge:
whatever the extractor decided to store *became* the truth, and the same branch
both proposed and judged it — so a bad inference re-derived itself from the store
on every later session. RSIAgent's evidence says that hole is structural rather
than prompt-level (a writer reviewing its own reasoning re-derives its own
mistakes, and its verifier is deliberately isolated from the actor), so the
structure is what this module fixes:

* the writer hands over CANDIDATES, never a verdict;
* the verifier is handed a projection built from :data:`CANDIDATE_FIELDS` only,
  so writer-private fields (reasoning, transcript, rationale, ...) cannot reach
  it even by accident — they are never read;
* only an explicit approval is written: a missing verifier, a raised error and an
  unrecognized verdict shape all DROP the candidate.

OFF by default (``memory.verify_consolidation`` in the ``memory`` config section),
so existing memory writes are unchanged until a deployment opts in — and opting
in without wiring a verifier stores nothing rather than storing unchecked claims.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from utils import is_truthy_value

logger = logging.getLogger(__name__)

# The ONLY candidate fields the verifier may see. Anything else the writer keeps on
# its candidate dict (reasoning, transcript, rationale, chain-of-thought) stays
# writer-private and is never forwarded.
CANDIDATE_FIELDS = ("content", "target", "category")


@dataclass(frozen=True)
class CandidateView:
    """What the verifier judges: the proposed entry and its routing, nothing else."""

    content: str
    target: str = "memory"
    category: str = "general"


@dataclass(frozen=True)
class ConsolidationVerdict:
    """A verifier's decision. ``approved=False`` (like every error path) means: do not write."""

    approved: bool
    reason: str = ""


def consolidation_verification_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """``memory.verify_consolidation`` — False unless explicitly truthy (conservative default).

    ``config`` is the resolved config dict; ``None`` loads it (a string ``"false"``
    from YAML/env must not read as on, hence :func:`is_truthy_value`).
    """
    from tools.memory_tool import get_builtin_memory_config  # lazy: registry registration at import

    return is_truthy_value(get_builtin_memory_config(config).get("verify_consolidation"), default=False)


def _view(candidate: Dict[str, Any]) -> CandidateView:
    """Project a candidate onto the allowlist — this is where isolation is enforced."""
    return CandidateView(
        content=str(candidate.get("content") or ""),
        target=str(candidate.get("target") or "memory"),
        category=str(candidate.get("category") or "general"),
    )


def verify_candidates(
    candidates: Sequence[Dict[str, Any]],
    verifier: Optional[Callable[[CandidateView, Tuple[str, ...]], Any]] = None,
    *,
    evidence: Sequence[str] = (),
    config: Optional[Dict[str, Any]] = None,
    enabled: Optional[bool] = None,
) -> Tuple[List[Dict[str, Any]], List[Tuple[Dict[str, Any], str]]]:
    """Split *candidates* into ``(approved, dropped)`` before anything is persisted.

    ``approved`` holds the ORIGINAL candidate dicts, so the writer keeps fields the
    verifier never saw; ``dropped`` holds ``(candidate, reason)`` pairs. Gate off
    (the default) approves everything and never calls *verifier*. *evidence* is the
    objective material the verifier may judge against (tool output, file content,
    a probe result) — never the session transcript or the writer's reasoning. A
    verifier returns :class:`ConsolidationVerdict` (or a plain bool); anything else,
    including an exception, drops the candidate.
    """
    candidates = list(candidates)
    if enabled is None:
        enabled = consolidation_verification_enabled(config)
    if not enabled:
        return candidates, []
    if verifier is None:
        logger.warning(
            "memory.verify_consolidation is on but no consolidation verifier is wired; "
            "dropping %d candidate(s) (fail closed)", len(candidates))
        return [], [(c, "no consolidation verifier configured") for c in candidates]
    frozen_evidence = tuple(str(e) for e in evidence)
    approved: List[Dict[str, Any]] = []
    dropped: List[Tuple[Dict[str, Any], str]] = []
    for candidate in candidates:
        try:
            verdict = verifier(_view(candidate), frozen_evidence)
        except Exception as e:
            dropped.append((candidate, f"verifier raised: {e}"))
            continue
        if isinstance(verdict, ConsolidationVerdict):
            ok, reason = bool(verdict.approved), verdict.reason
        elif isinstance(verdict, bool):
            ok, reason = verdict, ""
        else:
            ok, reason = False, f"unrecognized verdict shape: {type(verdict).__name__}"
        if ok:
            approved.append(candidate)
        else:
            dropped.append((candidate, reason or "not approved"))
    return approved, dropped

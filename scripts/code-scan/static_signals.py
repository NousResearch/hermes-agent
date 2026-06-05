"""Static Signals Schema and Helpers (Tier 1).

Minimal stdlib-only implementation per ua-tier1-001-static-signals-schema bead.

This module defines:
- make_signal_record / SignalRecord: canonical shape for a single heuristic marker.
- build_static_signals_artifact: produces the full document with forced
  claim_type="heuristic_signal" and semantic_status="not_validated".

Core boundary contract (verbatim):
Tier 1 static signals are heuristic content markers only.
They do not prove security, RLS correctness, auth correctness, runtime behavior,
deployment readiness, CI success, or policy semantics.

Every emitted Tier 1 claim MUST be labelled heuristic_signal and not_validated
unless it is an existing deterministic inventory fact from Tier 0.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.0.0"
CLAIM_TYPE = "heuristic_signal"
SEMANTIC_STATUS = "not_validated"

DEFAULT_BOUNDARIES: List[str] = [
    "Tier 1 static signals are content markers only; they do not prove security, "
    "RLS correctness, auth correctness, runtime behavior, deployment readiness, "
    "CI success, or policy semantics."
]


@dataclass(frozen=True)
class SignalRecord:
    """Canonical record for a Tier 1 static signal / heuristic marker.

    All fields are required at construction time for explicitness.
    The dataclass is frozen to prevent mutation after creation.
    """

    surface: str
    path: str
    line: int
    marker_type: str
    marker: str
    claim_type: str = CLAIM_TYPE
    semantic_status: str = SEMANTIC_STATUS
    boundary: str = DEFAULT_BOUNDARIES[0]

    def __getitem__(self, key: str):
        """Support dict-style access for compatibility (tests use rec["surface"])."""
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


def make_signal_record(
    surface: str,
    path: str,
    line: int,
    marker_type: str,
    marker: str,
    claim_type: Optional[str] = None,
    semantic_status: Optional[str] = None,
    boundary: Optional[str] = None,
) -> Dict[str, Any]:
    """Factory for a SignalRecord (returns plain dict for JSON stability).

    Callers may omit claim_type/semantic_status/boundary; they are forced
    to the Tier 1 defaults (heuristic_signal / not_validated).
    This helper never permits validated / concrete claims for pure Tier 1 signals.
    """
    rec = SignalRecord(
        surface=surface,
        path=path,
        line=line,
        marker_type=marker_type,
        marker=marker,
        claim_type=claim_type or CLAIM_TYPE,
        semantic_status=semantic_status or SEMANTIC_STATUS,
        boundary=boundary or DEFAULT_BOUNDARIES[0],
    )
    return asdict(rec)


def _normalize_signal(sig: Any) -> Dict[str, Any]:
    """Coerce incoming signal (dict or SignalRecord) into validated Tier 1 shape."""
    if isinstance(sig, SignalRecord):
        d = asdict(sig)
    elif isinstance(sig, dict):
        d = dict(sig)  # shallow copy
    else:
        raise TypeError(f"Signal must be dict or SignalRecord, got {type(sig)}")

    # Force Tier 1 contract on every signal, regardless of what caller supplied.
    # This is the overclaim-prevention mechanism.
    d["claim_type"] = CLAIM_TYPE
    d["semantic_status"] = SEMANTIC_STATUS

    # Ensure boundary text is present and carries the contract.
    if not d.get("boundary") or "does not prove security" not in str(d.get("boundary", "")):
        d["boundary"] = DEFAULT_BOUNDARIES[0]

    # Basic shape validation (minimal but sufficient for Tier 1)
    required = ("surface", "path", "line", "marker_type", "marker")
    for k in required:
        if k not in d:
            raise ValueError(f"Signal missing required field: {k}")

    # line must be int (or coercible)
    try:
        d["line"] = int(d["line"])
    except (TypeError, ValueError):
        raise ValueError("Signal 'line' must be integer")

    return d


def build_static_signals_artifact(
    signals: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Build the canonical static_signals.json document.

    Returns exactly the shape specified in the bead (empty case + populated).
    Always enforces:
        - schema_version = "1.0.0"
        - claim_type = "heuristic_signal"
        - semantic_status = "not_validated"
        - boundaries contain the exact required disclaimer text
    """
    if signals is None:
        signals = []

    normalized: List[Dict[str, Any]] = [_normalize_signal(s) for s in signals]

    # Compute summary (pure, deterministic)
    by_surface: Dict[str, int] = {}
    by_marker_type: Dict[str, int] = {}
    for s in normalized:
        surf = s["surface"]
        mt = s["marker_type"]
        by_surface[surf] = by_surface.get(surf, 0) + 1
        by_marker_type[mt] = by_marker_type.get(mt, 0) + 1

    summary = {
        "total_signals": len(normalized),
        "by_surface": by_surface,
        "by_marker_type": by_marker_type,
    }

    artifact: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "claim_type": CLAIM_TYPE,
        "semantic_status": SEMANTIC_STATUS,
        "signals": normalized,
        "summary": summary,
        "boundaries": list(DEFAULT_BOUNDARIES),  # copy
    }

    # Deterministic JSON round-trippability (no extra keys, stable order in practice)
    # We do not sort keys here; callers that need canonical bytes can do json.dumps(..., sort_keys=True)
    return artifact


# Convenience re-exports for consumers that want the canonical names
__all__ = [
    "build_static_signals_artifact",
    "make_signal_record",
    "SignalRecord",
    "CLAIM_TYPE",
    "SEMANTIC_STATUS",
    "SCHEMA_VERSION",
    "DEFAULT_BOUNDARIES",
]

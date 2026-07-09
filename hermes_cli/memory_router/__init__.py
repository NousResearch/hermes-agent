"""Thin Memory Router for Hermes (Phase 1).

The Router is the single entry point for all memory access. It:

1. CLASSIFIES a request by intent (rule-based, deterministic, no LLM).
2. SELECTS an available registered capability for that intent.
3. DISPATCHES the request to the capability's handler callable.
4. LOGS every routing decision (intent -> capability) for auditability.

The Router contains NO memory content, performs NO interpretation, calls NO
LLM, and bakes in NO project-specific knowledge. It only routes. This is the
"thin waist" between callers (agents) and the concrete storage layers
(L1 identity files, L5 SQLite+FTS5 index, etc.).

Out of scope for Phase 1 (deliberately NOT implemented here):
  - auto-extraction, summarization, embeddings
  - L2 project folders, L3 archive, L4 ADRs, L7 Graphiti, Holographic
  - any write path (remember/ingest)
"""

from __future__ import annotations

from .classify import classify
from .intents import INTENT_METADATA, Intent
from .provenance import (
    SearchResult,
    format_result,
    format_results,
)
from .registry import Capability, CapabilityRegistry
from .router import MemoryRouter, RouteResult

__all__ = [
    "Intent",
    "INTENT_METADATA",
    "Capability",
    "CapabilityRegistry",
    "SearchResult",
    "format_result",
    "format_results",
    "MemoryRouter",
    "RouteResult",
    "classify",
]

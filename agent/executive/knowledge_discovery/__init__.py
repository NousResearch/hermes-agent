"""Knowledge Discovery module for Executive v2 (B1 wiring).

Public surface (re-exported from engine.py and flag.py):

- Data classes (frozen value objects):
    KnowledgeQuery, KnowledgeHitV2, KnowledgeCitation, ConflictRecord,
    ProvenanceEnvelope, FreshnessPolicy
- Output dataclass:
    EvidencePack
- Engine (DI-based, no global state):
    EvidencePackEngine, _AuditSink
- Constants:
    SCHEMA_VERSION, ALLOWED_SOURCES, SOURCE_PRIORITY, SOURCE_TTL_DAYS,
    PREFIXES, SUMMARY_TEXT_MAX_LEN, MAX_HITS_TOTAL_DEFAULT,
    TOP_N_CITATIONS_DEFAULT, MAX_HITS_PER_SOURCE_DEFAULT,
    TIMEOUT_SECONDS_DEFAULT, SNIPPET_MAX_LEN, TITLE_MAX_LEN,
    HIT_ID_MAX_LEN, QUOTE_MAX_LEN, STATEMENT_MAX_LEN,
    IMPACT_MAX_LEN, RECOMMENDATION_MAX_LEN, STALE_PENALTY,
    UNKNOWN_PENALTY, CONFLICT_PENALTY_CAP, CONFLICT_PENALTY_PER_ITEM
- Internal helpers (re-exported for the canary shim and tests):
    _rank_hits, _detect_conflicts, _build_citations, _build_summary,
    _make_provenance, _make_hit_v2, _make_freshness,
    _classify_conflict, _conflict_id, _citation_fingerprint,
    _hit_fingerprint, _canonical_json, _sha256_hex, _clamp, _tokenize,
    _jaccard, _now_iso8601, _iso_mtime
- State meta helpers:
    get_state_meta_key
- Default-OFF flag resolver:
    resolve_knowledge_discovery_enabled

Hermeticity invariants (grep-enforced by tests):
- No subprocess, urllib, requests, httpx, socket, aiohttp, ssl imports
- No LLM client imports (openai, anthropic, etc.)
- No real GBrain / Obsidian imports (real adapters live in adapters.py
  and are wired via DI in Gate D / E2E)
- All side effects routed through injected storage + audit_sink
"""

from .engine import (
    EvidencePack,
    KnowledgeQuery,
    KnowledgeHitV2,
    KnowledgeCitation,
    ConflictRecord,
    ProvenanceEnvelope,
    FreshnessPolicy,
    EvidencePackEngine,
    _AuditSink,
    SCHEMA_VERSION,
    ALLOWED_SOURCES,
    ALLOWED_FRESHNESS,
    ALLOWED_RETRIEVAL_MODES,
    ALLOWED_SEVERITY,
    ALLOWED_PROVENANCE_SOURCE_TYPES,
    ALLOWED_CONFLICT_TYPES,
    ALLOWED_RESOLUTION_STATUSES,
    SOURCE_PRIORITY,
    SOURCE_TTL_DAYS,
    PREFIXES,
    SUMMARY_TEXT_MAX_LEN,
    MAX_HITS_TOTAL_DEFAULT,
    TOP_N_CITATIONS_DEFAULT,
    MAX_HITS_PER_SOURCE_DEFAULT,
    TIMEOUT_SECONDS_DEFAULT,
    SNIPPET_MAX_LEN,
    TITLE_MAX_LEN,
    HIT_ID_MAX_LEN,
    QUOTE_MAX_LEN,
    STATEMENT_MAX_LEN,
    IMPACT_MAX_LEN,
    RECOMMENDATION_MAX_LEN,
    STALE_PENALTY,
    UNKNOWN_PENALTY,
    CONFLICT_PENALTY_CAP,
    CONFLICT_PENALTY_PER_ITEM,
    FINGERPRINT_RE,
    CITATION_ID_RE,
    CONFLICT_ID_RE,
    LINE_RANGE_RE,
    _rank_hits,
    _detect_conflicts,
    _build_citations,
    _build_summary,
    _make_provenance,
    _make_hit_v2,
    _make_freshness,
    _classify_conflict,
    _conflict_id,
    _citation_fingerprint,
    _hit_fingerprint,
    _canonical_json,
    _sha256_hex,
    _clamp,
    _tokenize,
    _jaccard,
    _now_iso8601,
    _iso_mtime,
    get_state_meta_key,
)
from .flag import resolve_knowledge_discovery_enabled

__all__ = [
    # Data classes
    "EvidencePack",
    "KnowledgeQuery",
    "KnowledgeHitV2",
    "KnowledgeCitation",
    "ConflictRecord",
    "ProvenanceEnvelope",
    "FreshnessPolicy",
    # Engine
    "EvidencePackEngine",
    "_AuditSink",
    # Constants
    "SCHEMA_VERSION",
    "ALLOWED_SOURCES",
    "ALLOWED_FRESHNESS",
    "ALLOWED_RETRIEVAL_MODES",
    "ALLOWED_SEVERITY",
    "ALLOWED_PROVENANCE_SOURCE_TYPES",
    "ALLOWED_CONFLICT_TYPES",
    "ALLOWED_RESOLUTION_STATUSES",
    "SOURCE_PRIORITY",
    "SOURCE_TTL_DAYS",
    "PREFIXES",
    "SUMMARY_TEXT_MAX_LEN",
    "MAX_HITS_TOTAL_DEFAULT",
    "TOP_N_CITATIONS_DEFAULT",
    "MAX_HITS_PER_SOURCE_DEFAULT",
    "TIMEOUT_SECONDS_DEFAULT",
    "SNIPPET_MAX_LEN",
    "TITLE_MAX_LEN",
    "HIT_ID_MAX_LEN",
    "QUOTE_MAX_LEN",
    "STATEMENT_MAX_LEN",
    "IMPACT_MAX_LEN",
    "RECOMMENDATION_MAX_LEN",
    "STALE_PENALTY",
    "UNKNOWN_PENALTY",
    "CONFLICT_PENALTY_CAP",
    "CONFLICT_PENALTY_PER_ITEM",
    "FINGERPRINT_RE",
    "CITATION_ID_RE",
    "CONFLICT_ID_RE",
    "LINE_RANGE_RE",
    # Helpers (re-exported for the canary shim)
    "_rank_hits",
    "_detect_conflicts",
    "_build_citations",
    "_build_summary",
    "_make_provenance",
    "_make_hit_v2",
    "_make_freshness",
    "_classify_conflict",
    "_conflict_id",
    "_citation_fingerprint",
    "_hit_fingerprint",
    "_canonical_json",
    "_sha256_hex",
    "_clamp",
    "_tokenize",
    "_jaccard",
    "_now_iso8601",
    "_iso_mtime",
    "get_state_meta_key",
    # Flag
    "resolve_knowledge_discovery_enabled",
]

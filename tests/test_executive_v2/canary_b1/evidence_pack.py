"""Re-export shim — production module moved to agent.executive.knowledge_discovery.

This file exists for backwards compatibility with the canary tests
(tests/test_executive_v2/canary_b1/test_*.py) and the B1 hermetic tests
(tests/test_executive_v2/b1_tests/test_*.py) which import from this path.

The implementation was lifted to
``agent.executive/knowledge_discovery/engine.py`` in Gate C (B1
Knowledge Discovery integration). The behavior is byte-identical: same
classes, same constants, same helpers. New wiring code (the
``ObjectiveEngine.discover_evidence_pack`` glue, the
``ExecutionContractV1`` extended fields, the planner gate) imports from
the production module, not from this shim.

This shim also re-imports the ``time`` and ``re`` modules from the
engine so that the canary's conftest fixtures can still monkeypatch
``time.monotonic`` and re-look-up module-level constants.
"""

import time as time  # re-exported for canary conftest monkeypatch
import re as re      # re-exported for canary tests that use re.compile

from agent.executive.knowledge_discovery.engine import (  # noqa: F401
    # Data classes
    EvidencePack,
    KnowledgeQuery,
    KnowledgeHitV2,
    KnowledgeCitation,
    ConflictRecord,
    ProvenanceEnvelope,
    FreshnessPolicy,
    # Engine
    EvidencePackEngine,
    _AuditSink,
    # Constants
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
    # Helpers
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

"""EvidencePack v1 + EvidencePackEngine — production module.

Lifted from tests/test_executive_v2/canary_b1/evidence_pack.py in
Gate C (B1 Knowledge Discovery integration). Self-contained: does NOT
import from any production code path. Does NOT touch GBrain, Obsidian,
the real state.db, network, or subprocess. All side effects are routed
through an injected `in_memory_storage` and (optionally) an injected
`audit_sink` callable.

The engine is OFF by default (HERMES_EXECUTIVE_KNOWLEDGE_DISCOVERY_ENABLED=0;
resolved via agent.executive.knowledge_discovery.flag).

Hermeticity invariants (preserved from the canary):
- No subprocess / urllib / requests / httpx / socket / aiohttp / ssl imports
- No LLM client imports
- No real GBrain / Obsidian imports (real adapters live in adapters.py
  and are wired via DI; Gate D / E2E)
- No state.db / audit log writes (the production wiring uses the same
  in-memory injection as the canary)

The canary file tests/test_executive_v2/canary_b1/evidence_pack.py is
preserved as a thin re-export shim for backward compatibility with the
existing 130 canary + 57 b1_tests that import from that path.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterable, Optional

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

SCHEMA_VERSION = "evidence_pack.v1"
MAX_HITS_TOTAL_DEFAULT = 20
TOP_N_CITATIONS_DEFAULT = 10
MAX_HITS_PER_SOURCE_DEFAULT = 5
TIMEOUT_SECONDS_DEFAULT = 30.0
SUMMARY_TEXT_MAX_LEN = 2000
SNIPPET_MAX_LEN = 500
TITLE_MAX_LEN = 256
HIT_ID_MAX_LEN = 512
QUOTE_MAX_LEN = 1000
STATEMENT_MAX_LEN = 500
IMPACT_MAX_LEN = 500
RECOMMENDATION_MAX_LEN = 500

STALE_PENALTY = 0.50
UNKNOWN_PENALTY = 0.60
CONFLICT_PENALTY_CAP = 0.10
CONFLICT_PENALTY_PER_ITEM = 0.05

# Source priority per freshness_and_ranking_policy §2.2
SOURCE_PRIORITY = {
    "contract": 1.00,
    "policy": 0.95,
    "gbrain": 0.85,
    "obsidian": 0.75,
    "report": 0.65,
}

# Per-source TTL in days per freshness_and_ranking_policy §1.1
SOURCE_TTL_DAYS = {
    "policy": 30,
    "contract": 30,
    "report": 90,
    "gbrain": 14,
    "obsidian": 14,
}

ALLOWED_SOURCES = set(SOURCE_TTL_DAYS.keys())
ALLOWED_RETRIEVAL_MODES = {
    "metadata_only", "snippet", "full_document",
    "semantic_search", "keyword_search",
}
ALLOWED_FRESHNESS = {"current", "recent", "stale", "unknown"}
ALLOWED_SEVERITY = {"low", "medium", "high"}
ALLOWED_CONFLICT_TYPES = {
    "policy_vs_goal", "memory_vs_evidence", "evidence_vs_evidence",
    "freshness", "scope", "identity", "unknown",
}
ALLOWED_RESOLUTION_STATUSES = {
    "unresolved", "resolved_by_policy", "resolved_by_newer_evidence",
    "requires_human", "requires_expert",
}
ALLOWED_PROVENANCE_SOURCE_TYPES = ALLOWED_SOURCES | {"kg", "claims", "evidence"}

FINGERPRINT_RE = re.compile(r"^[a-f0-9]{64}$")
CITATION_ID_RE = re.compile(r"^cite:[a-f0-9]{8,16}$")
CONFLICT_ID_RE = re.compile(r"^conflict:[a-f0-9]{8,16}$")
LINE_RANGE_RE = re.compile(r"^[0-9]+-[0-9]+$")

# Recommendation prefixes (degradation / readiness flags)
PREFIXES = (
    "[READY_FOR_STRATEGY]",
    "[READY_WITH_CAVEATS]",
    "[REQUIRES_HUMAN]",
    "[REQUIRES_MORE_INFO]",
    "[NEEDS_EXPERT_REVIEW]",
    "[DEGRADED_FRESHNESS]",
    "[VAULT_STALE]",
)


# ─────────────────────────────────────────────────────────────────────
# Time helpers (deterministic; overridable via monkeypatch)
# ─────────────────────────────────────────────────────────────────────


def _now_iso8601() -> str:
    """Default: real UTC ISO 8601. Tests monkeypatch this."""
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _iso_mtime(path: Any) -> str:
    """Best-effort ISO 8601 mtime for filesystem-mtime fixtures."""
    try:
        mtime = path.stat().st_mtime
        return _dt.datetime.fromtimestamp(mtime, tz=_dt.timezone.utc).isoformat()
    except OSError:
        return _now_iso8601()


# ─────────────────────────────────────────────────────────────────────
# Canonical JSON / hashing helpers
# ─────────────────────────────────────────────────────────────────────


def _canonical_json(payload: Any) -> str:
    """Canonical JSON with sorted keys, ensure_ascii=False, dataclass-safe."""
    return json.dumps(
        payload, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), default=str,
    )


def _sha256_hex(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _hit_fingerprint(source: str, hit_id: str, snippet: str) -> str:
    return _sha256_hex({"source": source, "hit_id": hit_id, "snippet": snippet})


def _citation_fingerprint(citation_id: str, statement: str, source_uri: str) -> str:
    return _sha256_hex({
        "citation_id": citation_id,
        "statement": statement,
        "source_uri": source_uri,
    })


def _conflict_id(items: tuple[str, ...], conflict_type: str) -> str:
    raw = _sha256_hex({"items": sorted(items), "conflict_type": conflict_type})
    return f"conflict:{raw[:16]}"


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _tokenize(text: str) -> set[str]:
    out: set[str] = set()
    for tok in (text or "").lower().split():
        if len(tok) >= 3:
            out.add(tok)
    return out


# ─────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProvenanceEnvelope:
    """Every KnowledgeHitV2 MUST carry this. read_only is hardcoded True."""

    producer: str
    produced_at: str            # ISO 8601 UTC
    source_type: str            # enum
    source_uri: str
    retrieval_mode: str         # enum
    read_only: bool = True      # ALWAYS True
    hash_sha256: Optional[str] = None
    quote: Optional[str] = None
    line_range: Optional[str] = None


@dataclass(frozen=True)
class FreshnessPolicy:
    """Freshness metadata for a single hit."""

    observed_at: str
    source_updated_at: str
    staleness_days: int
    freshness: str              # current | recent | stale | unknown
    freshness_score: float      # [0.0, 1.0]


@dataclass(frozen=True)
class KnowledgeHitV2:
    """v2 hit — superset of v0.1 KnowledgeHit."""

    source: str
    hit_id: str
    title: str
    relevance_score: float
    snippet: str
    location: str
    fingerprint: str
    created_at: str
    provenance: ProvenanceEnvelope
    freshness: FreshnessPolicy
    effective_score: float = 0.0


@dataclass(frozen=True)
class KnowledgeCitation:
    citation_id: str            # cite:<hex8-16>
    statement: str
    source_uri: str
    source_type: str
    fingerprint: str
    relevance_score: float
    freshness_score: float
    confidence: float


@dataclass(frozen=True)
class ConflictRecord:
    conflict_id: str            # conflict:<hex8-16>
    conflict_type: str          # enum
    severity: str               # low | medium | high
    items: tuple[str, ...]      # hit_ids
    impact: str
    recommended_resolution: str
    resolution_status: str      # enum
    detected_at: str


@dataclass(frozen=True)
class KnowledgeQuery:
    objective_id: str
    objective_text: str
    goal_class: str = "OTHER"
    risk_profile: str = "low"
    complexity: str = "S"
    max_hits_per_source: int = MAX_HITS_PER_SOURCE_DEFAULT
    max_hits_total: int = MAX_HITS_TOTAL_DEFAULT
    timeout_seconds: float = TIMEOUT_SECONDS_DEFAULT
    sources_requested: tuple[str, ...] = ("policy", "contract", "report", "gbrain", "obsidian")
    schema_version: str = SCHEMA_VERSION

    def fingerprint(self) -> str:
        return _sha256_hex({
            "objective_id": self.objective_id,
            "objective_text": self.objective_text,
            "goal_class": self.goal_class,
            "risk_profile": self.risk_profile,
            "complexity": self.complexity,
            "max_hits_per_source": self.max_hits_per_source,
            "timeout_seconds": self.timeout_seconds,
            "sources_requested": sorted(self.sources_requested),
            "schema_version": self.schema_version,
        })


@dataclass
class EvidencePack:
    objective_id: str
    query_fingerprint: str
    sources_queried: list[str] = field(default_factory=list)
    sources_failed: list[str] = field(default_factory=list)
    hits: list[KnowledgeHitV2] = field(default_factory=list)
    citations: list[KnowledgeCitation] = field(default_factory=list)
    conflicts: list[ConflictRecord] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    overall_freshness_score: float = 0.0
    overall_confidence: float = 0.0
    summary_text: str = ""
    summary_fingerprint: str = ""
    duration_ms: int = 0
    created_at: str = ""
    schema_version: str = SCHEMA_VERSION
    is_idempotent_reuse: bool = False
    total_hits: int = 0

    def to_dict(self) -> dict:
        d = {
            "objective_id": self.objective_id,
            "query_fingerprint": self.query_fingerprint,
            "sources_queried": list(self.sources_queried),
            "sources_failed": list(self.sources_failed),
            "hits": [_hit_to_dict(h) for h in self.hits],
            "citations": [_citation_to_dict(c) for c in self.citations],
            "conflicts": [_conflict_to_dict(c) for c in self.conflicts],
            "missing_information": list(self.missing_information),
            "overall_freshness_score": float(self.overall_freshness_score),
            "overall_confidence": float(self.overall_confidence),
            "summary_text": self.summary_text,
            "summary_fingerprint": self.summary_fingerprint,
            "duration_ms": int(self.duration_ms),
            "created_at": self.created_at,
            "schema_version": self.schema_version,
        }
        return d


def _hit_to_dict(h: KnowledgeHitV2) -> dict:
    return {
        "source": h.source,
        "hit_id": h.hit_id,
        "title": h.title[:TITLE_MAX_LEN],
        "relevance_score": float(_clamp(h.relevance_score)),
        "snippet": h.snippet[:SNIPPET_MAX_LEN],
        "location": h.location,
        "fingerprint": h.fingerprint,
        "created_at": h.created_at,
        "provenance": {
            "producer": h.provenance.producer,
            "produced_at": h.provenance.produced_at,
            "source_type": h.provenance.source_type,
            "source_uri": h.provenance.source_uri,
            "retrieval_mode": h.provenance.retrieval_mode,
            "read_only": bool(h.provenance.read_only),
            "hash_sha256": h.provenance.hash_sha256,
            "quote": (h.provenance.quote or "")[:QUOTE_MAX_LEN] or None,
            "line_range": h.provenance.line_range,
        },
        "freshness": {
            "observed_at": h.freshness.observed_at,
            "source_updated_at": h.freshness.source_updated_at,
            "staleness_days": int(h.freshness.staleness_days),
            "freshness": h.freshness.freshness,
            "freshness_score": float(_clamp(h.freshness.freshness_score)),
        },
        "effective_score": float(_clamp(h.effective_score)),
    }


def _citation_to_dict(c: KnowledgeCitation) -> dict:
    return {
        "citation_id": c.citation_id,
        "statement": c.statement[:STATEMENT_MAX_LEN],
        "source_uri": c.source_uri,
        "source_type": c.source_type,
        "fingerprint": c.fingerprint,
        "relevance_score": float(_clamp(c.relevance_score)),
        "freshness_score": float(_clamp(c.freshness_score)),
        "confidence": float(_clamp(c.confidence)),
    }


def _conflict_to_dict(c: ConflictRecord) -> dict:
    return {
        "conflict_id": c.conflict_id,
        "conflict_type": c.conflict_type,
        "severity": c.severity,
        "items": list(c.items),
        "impact": c.impact[:IMPACT_MAX_LEN],
        "recommended_resolution": c.recommended_resolution[:RECOMMENDATION_MAX_LEN],
        "resolution_status": c.resolution_status,
        "detected_at": c.detected_at,
    }


# ─────────────────────────────────────────────────────────────────────
# Freshness calculator
# ─────────────────────────────────────────────────────────────────────


def _make_freshness(
    *,
    observed_at: str,
    source_updated_at: Optional[str],
    ttl_days: int,
) -> FreshnessPolicy:
    """Compute FreshnessPolicy from observed_at + source_updated_at + TTL."""
    if source_updated_at is None:
        return FreshnessPolicy(
            observed_at=observed_at,
            source_updated_at="1970-01-01T00:00:00+00:00",
            staleness_days=0,
            freshness="unknown",
            freshness_score=0.5,
        )
    obs = _dt.datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    upd = _dt.datetime.fromisoformat(source_updated_at.replace("Z", "+00:00"))
    delta_days = (obs - upd).days
    days = max(0, delta_days)
    ttl_half = ttl_days / 2.0

    if days <= ttl_half:
        freshness = "current"
        if ttl_half > 0:
            freshness_score = 1.0 - 0.05 * (days / ttl_half)
        else:
            freshness_score = 1.0
    elif days <= ttl_days:
        freshness = "recent"
        if ttl_half > 0:
            freshness_score = 0.95 - 0.30 * ((days - ttl_half) / ttl_half)
        else:
            freshness_score = 0.95
    elif days <= ttl_days * 2:
        freshness = "stale"
        if ttl_days > 0:
            freshness_score = 0.65 - 0.45 * ((days - ttl_days) / ttl_days)
        else:
            freshness_score = 0.20
    else:
        freshness = "stale"
        freshness_score = 0.20

    return FreshnessPolicy(
        observed_at=observed_at,
        source_updated_at=source_updated_at,
        staleness_days=days,
        freshness=freshness,
        freshness_score=_clamp(freshness_score),
    )


def _make_provenance(
    source: str,
    source_uri: str,
    *,
    retrieval_mode: str = "metadata_only",
    hash_sha256: Optional[str] = None,
    quote: Optional[str] = None,
    line_range: Optional[str] = None,
    observed_at: str = "",
    producer: str = "",
) -> ProvenanceEnvelope:
    if retrieval_mode not in ALLOWED_RETRIEVAL_MODES:
        retrieval_mode = "metadata_only"
    return ProvenanceEnvelope(
        producer=producer or f"fake_{source}_provider_v1",
        produced_at=observed_at or _now_iso8601(),
        source_type=source if source in ALLOWED_PROVENANCE_SOURCE_TYPES else "evidence",
        source_uri=source_uri,
        retrieval_mode=retrieval_mode,
        read_only=True,
        hash_sha256=hash_sha256,
        quote=(quote or "")[:QUOTE_MAX_LEN] or None,
        line_range=line_range,
    )


def _make_hit_v2(
    source: str,
    hit_id: str,
    title: str,
    relevance_score: float,
    snippet: str,
    *,
    source_uri: str,
    source_updated_at: Optional[str],
    retrieval_mode: str = "metadata_only",
    quote: Optional[str] = None,
    line_range: Optional[str] = None,
    hash_sha256: Optional[str] = None,
    observed_at: str,
    ttl_days: int,
    created_at: Optional[str] = None,
    location: Optional[str] = None,
) -> KnowledgeHitV2:
    fp = _hit_fingerprint(source, hit_id, snippet)
    provenance = _make_provenance(
        source, source_uri,
        retrieval_mode=retrieval_mode,
        hash_sha256=hash_sha256,
        quote=quote, line_range=line_range,
        observed_at=observed_at,
    )
    freshness = _make_freshness(
        observed_at=observed_at,
        source_updated_at=source_updated_at,
        ttl_days=ttl_days,
    )
    return KnowledgeHitV2(
        source=source,
        hit_id=hit_id,
        title=title[:TITLE_MAX_LEN],
        relevance_score=_clamp(relevance_score),
        snippet=snippet[:SNIPPET_MAX_LEN],
        location=location or source_uri,
        fingerprint=fp,
        created_at=created_at or source_updated_at or observed_at,
        provenance=provenance,
        freshness=freshness,
        effective_score=0.0,
    )


# ─────────────────────────────────────────────────────────────────────
# Conflict detection
# ─────────────────────────────────────────────────────────────────────


def _detect_conflicts(hits: list[KnowledgeHitV2], observed_at: str) -> list[ConflictRecord]:
    """Pairwise O(N²) conflict detector on top-K hits.

    Conflict types and severities per conflict_resolution_policy.md:
    * policy_vs_goal        — medium/high
    * memory_vs_evidence    — medium
    * evidence_vs_evidence  — low (same band) / medium (cross band)
    * freshness             — low (delta > 30d)
    * scope                 — low
    * identity              — medium
    * unknown               — low (fallback)
    """
    conflicts: list[ConflictRecord] = []
    if not hits:
        return conflicts

    # We only do pairwise on the first MAX_HITS_TOTAL_DEFAULT to bound O(N²)
    top = hits[:MAX_HITS_TOTAL_DEFAULT]
    for i in range(len(top)):
        a = top[i]
        for j in range(i + 1, len(top)):
            b = top[j]
            detected = _classify_conflict(a, b)
            if detected is None:
                continue
            ctype, severity, impact, rec = detected
            cid = _conflict_id((a.hit_id, b.hit_id), ctype)
            conflicts.append(ConflictRecord(
                conflict_id=cid,
                conflict_type=ctype,
                severity=severity,
                items=(a.hit_id, b.hit_id),
                impact=impact,
                recommended_resolution=rec,
                resolution_status="unresolved",
                detected_at=observed_at,
            ))
    return conflicts


def _classify_conflict(
    a: KnowledgeHitV2, b: KnowledgeHitV2
) -> Optional[tuple[str, str, str, str]]:
    """Return (type, severity, impact, recommended_resolution) or None."""
    # Freshness delta: a/b from same source, very different updated_at
    if (a.source == b.source
            and abs(a.freshness.staleness_days - b.freshness.staleness_days) > 30):
        return (
            "freshness", "low",
            f"source_updated_at delta > 30d between {a.hit_id} and {b.hit_id}",
            "use newer source_updated_at; archive older",
        )

    # Identity: same hit_id, different source_uri
    # IMPORTANT: this check is BEFORE memory_vs_evidence because identity
    # is the more specific classification (same entity, different URI is
    # an identity issue, not a generic memory-vs-evidence contradiction).
    if a.hit_id == b.hit_id and a.provenance.source_uri != b.provenance.source_uri:
        return (
            "identity", "medium",
            f"duplicate hit_id {a.hit_id} with different uris",
            "dedup by source_uri; keep higher-priority source",
        )

    # Memory vs evidence: gbrain vs obsidian
    pair = {a.source, b.source}
    if pair == {"gbrain", "obsidian"} or pair == {"gbrain", "report"}:
        return (
            "memory_vs_evidence", "medium",
            f"contradiction between {a.source} and {b.source}",
            "resolve by policy; default to higher source priority",
        )

    # Policy vs goal: policy/contract contradictions
    if pair == {"policy", "gbrain"} or pair == {"policy", "obsidian"}:
        return (
            "policy_vs_goal", "high",
            f"policy decision contradicted by {a.source if a.source != 'policy' else b.source}",
            "requires_human; flag in human_gate_audit",
        )

    # Evidence vs evidence: same source, freshness band delta
    if a.source == b.source and a.freshness.freshness != b.freshness.freshness:
        return (
            "evidence_vs_evidence", "medium",
            f"same source {a.source} cross-band freshness",
            "prefer current; demote stale",
        )

    # Scope: same hit_id family (prefix), different token overlap pattern
    a_prefix = a.hit_id.rsplit("/", 1)[0] if "/" in a.hit_id else a.hit_id
    b_prefix = b.hit_id.rsplit("/", 1)[0] if "/" in b.hit_id else b.hit_id
    if (a_prefix == b_prefix
            and a.source != b.source
            and _jaccard(_tokenize(a.snippet), _tokenize(b.snippet)) < 0.30):
        return (
            "scope", "low",
            f"same prefix {a_prefix} but divergent content",
            "split into sub-scopes; mark both as candidates",
        )

    return None


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# ─────────────────────────────────────────────────────────────────────
# Ranker
# ─────────────────────────────────────────────────────────────────────


def _rank_hits(
    hits: list[KnowledgeHitV2],
    *,
    top_k: int = MAX_HITS_TOTAL_DEFAULT,
    per_source_cap: int = MAX_HITS_PER_SOURCE_DEFAULT,
) -> list[KnowledgeHitV2]:
    """Score, dedup by fingerprint, cap per source, return top-K sorted."""
    if not hits:
        return []

    # Score (effective_score = relevance × freshness × source_priority × penalties)
    scored: list[KnowledgeHitV2] = []
    for h in hits:
        sp = SOURCE_PRIORITY.get(h.source, 0.50)
        score = h.relevance_score * h.freshness.freshness_score * sp
        # STALE_PENALTY when freshness_score < 0.30
        if h.freshness.freshness_score < 0.30:
            score *= STALE_PENALTY
        # UNKNOWN_PENALTY when freshness=unknown
        if h.freshness.freshness == "unknown":
            score *= UNKNOWN_PENALTY
        # Demote hits missing provenance (shouldn't happen in canary, but safe)
        if h.provenance is None:
            score = 0.0
        # Carry provenance + freshness; only effective_score changes
        scored.append(KnowledgeHitV2(
            source=h.source,
            hit_id=h.hit_id,
            title=h.title,
            relevance_score=h.relevance_score,
            snippet=h.snippet,
            location=h.location,
            fingerprint=h.fingerprint,
            created_at=h.created_at,
            provenance=h.provenance,
            freshness=h.freshness,
            effective_score=_clamp(score),
        ))

    # Dedup by exact fingerprint
    seen_fp: set[str] = set()
    deduped: list[KnowledgeHitV2] = []
    for h in sorted(scored, key=lambda x: -x.effective_score):
        if h.fingerprint in seen_fp:
            continue
        seen_fp.add(h.fingerprint)
        deduped.append(h)

    # Near-dup Jaccard ≥ 0.85 → drop lower score
    final: list[KnowledgeHitV2] = []
    for h in deduped:
        tokens_h = _tokenize(h.snippet)
        is_dup = False
        for kept in final:
            if _jaccard(tokens_h, _tokenize(kept.snippet)) >= 0.85:
                is_dup = True
                break
        if not is_dup:
            final.append(h)

    # Per-source cap
    capped: list[KnowledgeHitV2] = []
    src_count: dict[str, int] = {}
    for h in final:
        c = src_count.get(h.source, 0)
        if c >= per_source_cap:
            continue
        capped.append(h)
        src_count[h.source] = c + 1

    # Top-K
    capped.sort(key=lambda x: -x.effective_score)
    return capped[:top_k]


# ─────────────────────────────────────────────────────────────────────
# Citations
# ─────────────────────────────────────────────────────────────────────


def _build_citations(
    hits: list[KnowledgeHitV2],
    top_n: int = TOP_N_CITATIONS_DEFAULT,
    observed_at: str = "",
) -> list[KnowledgeCitation]:
    """Build top-N citations from top-K hits, sorted desc by score."""
    out: list[KnowledgeCitation] = []
    for h in hits[:top_n]:
        statement = h.snippet[:STATEMENT_MAX_LEN]
        fp = _citation_fingerprint(
            citation_id="",  # placeholder; we want fingerprint on (statement, source_uri)
            statement=statement,
            source_uri=h.provenance.source_uri,
        )
        # citation_id derived from fingerprint
        cid = f"cite:{fp[:12]}"
        # Now recompute fingerprint with the real citation_id
        real_fp = _citation_fingerprint(
            citation_id=cid, statement=statement, source_uri=h.provenance.source_uri
        )
        confidence = _clamp(h.effective_score)
        out.append(KnowledgeCitation(
            citation_id=cid,
            statement=statement,
            source_uri=h.provenance.source_uri,
            source_type=h.source,
            fingerprint=real_fp,
            relevance_score=_clamp(h.relevance_score),
            freshness_score=_clamp(h.freshness.freshness_score),
            confidence=confidence,
        ))
    out.sort(key=lambda c: -(c.relevance_score * c.freshness_score * c.confidence))
    return out


# ─────────────────────────────────────────────────────────────────────
# Summary text builder
# ─────────────────────────────────────────────────────────────────────


def _build_summary(
    hits: list[KnowledgeHitV2],
    conflicts: list[ConflictRecord],
    overall_freshness: float,
    overall_confidence: float,
) -> str:
    """Compute summary text with degradation / readiness prefix."""
    if not hits:
        return "(no relevant knowledge found)"

    high_conflicts = [c for c in conflicts if c.severity == "high"]
    med_conflicts = [c for c in conflicts if c.severity == "medium"]

    prefix: str
    if high_conflicts:
        prefix = "[REQUIRES_HUMAN]"
    elif overall_freshness < 0.5:
        prefix = "[DEGRADED_FRESHNESS]"
    elif overall_confidence < 0.4:
        prefix = "[REQUIRES_MORE_INFO]"
    elif med_conflicts:
        prefix = "[READY_WITH_CAVEATS]"
    else:
        prefix = "[READY_FOR_STRATEGY]"

    body = (
        f"found {len(hits)} hits "
        f"avg_freshness={overall_freshness:.2f} "
        f"confidence={overall_confidence:.2f} "
        f"conflicts={len(conflicts)}"
    )
    text = f"{prefix} {body}"
    if len(text) > SUMMARY_TEXT_MAX_LEN:
        text = text[:SUMMARY_TEXT_MAX_LEN]
    return text


# ─────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────


class _AuditSink:
    """In-memory audit sink. Real audit log NEVER touched."""

    def __init__(self) -> None:
        self._events: list[dict] = []

    def emit(self, event: dict) -> None:
        self._events.append(dict(event))

    def get_events(self) -> list[dict]:
        return list(self._events)


class EvidencePackEngine:
    """Hermetic canary engine. Default-off flag respected.

    Parameters
    ----------
    sources : dict[str, callable]
        Maps source name → callable(query, *, max_hits, observed_at) -> list[KnowledgeHitV2].
        All 5 fake sources should be passed in by tests.
    storage : optional
        In-memory storage (FakeDB-like). Used only by `discover()` to persist
        a single state_meta key per objective. NEVER touches ~/.hermes/state.db.
    audit_sink : optional
        In-memory audit capture. NEVER touches ~/.hermes/audit/*.
    """

    STATE_META_PREFIX = "objective_knowledge_discovery:"
    STATE_META_KEY_VERSION = "v2"

    def __init__(
        self,
        sources: Optional[dict[str, Callable[..., list[KnowledgeHitV2]]]] = None,
        storage: Any = None,
        audit_sink: Optional[_AuditSink] = None,
    ) -> None:
        # Default-off gate (inert when env=0; we instantiate regardless for canary)
        self._enabled_flag = os.environ.get(
            "HERMES_EXECUTIVE_KNOWLEDGE_DISCOVERY_ENABLED", "0"
        ) == "1"
        self._sources: dict[str, Callable[..., list[KnowledgeHitV2]]] = sources or {}
        self._storage = storage
        self._audit_sink = audit_sink if audit_sink is not None else _AuditSink()
        self._monotonic = time.monotonic

    # ── public API ──────────────────────────────────────────────

    def dry_run(
        self,
        objective_id: str,
        objective_text: str,
        *,
        goal_class: str = "OTHER",
        risk_profile: str = "low",
        complexity: str = "S",
        sources_requested: Optional[tuple[str, ...]] = None,
        max_hits_per_source: int = MAX_HITS_PER_SOURCE_DEFAULT,
        max_hits_total: int = MAX_HITS_TOTAL_DEFAULT,
        timeout_seconds: float = TIMEOUT_SECONDS_DEFAULT,
    ) -> EvidencePack:
        """Build an EvidencePack without persisting anywhere."""
        t0 = self._monotonic()
        observed_at = _now_iso8601()
        if not objective_text:
            objective_text = ""
        if max_hits_per_source < 1:
            max_hits_per_source = 1
        if max_hits_total < 1:
            max_hits_total = 1
        if timeout_seconds < 0:
            timeout_seconds = TIMEOUT_SECONDS_DEFAULT

        sources = sources_requested or tuple(self._sources.keys())
        # Filter to known sources only
        sources = tuple(s for s in sources if s in ALLOWED_SOURCES)
        query = KnowledgeQuery(
            objective_id=objective_id,
            objective_text=objective_text,
            goal_class=goal_class,
            risk_profile=risk_profile,
            complexity=complexity,
            max_hits_per_source=max_hits_per_source,
            max_hits_total=max_hits_total,
            timeout_seconds=timeout_seconds,
            sources_requested=sources,
        )

        all_hits: list[KnowledgeHitV2] = []
        sources_queried: list[str] = []
        sources_failed: list[str] = []
        missing: list[str] = []

        for src in sources:
            provider = self._sources.get(src)
            if provider is None:
                missing.append(f"provider for source {src!r} not registered")
                continue
            try:
                hits = provider(
                    query,
                    max_hits=max_hits_per_source,
                    observed_at=observed_at,
                )
            except Exception:
                sources_failed.append(src)
                continue
            sources_queried.append(src)
            for h in hits:
                if h.provenance is None:
                    missing.append(
                        f"hit {h.hit_id} missing provenance (demoted)"
                    )
                    continue
                all_hits.append(h)

        # Bound objective_text for spec
        if len(objective_text) > 10_000:
            objective_text = objective_text[:10_000]
            missing.append("objective_text clamped to 10000 chars")

        # Rank, dedup, cap
        ranked = _rank_hits(
            all_hits,
            top_k=max_hits_total,
            per_source_cap=max_hits_per_source,
        )

        # Conflicts
        conflicts = _detect_conflicts(ranked, observed_at)
        for c in conflicts:
            if c.severity == "high":
                self._audit_sink.emit({
                    "gate_type": "knowledge_conflict",
                    "severity": "high",
                    "conflict_id": c.conflict_id,
                    "objective_id": objective_id,
                    "detected_at": c.detected_at,
                })

        # Aggregate scores
        if ranked:
            overall_freshness = sum(
                h.freshness.freshness_score for h in ranked
            ) / len(ranked)
        else:
            overall_freshness = 0.0

        if ranked:
            relevance_avg = sum(h.relevance_score for h in ranked) / len(ranked)
            freshness_avg = overall_freshness
            unique_sources = len({h.source for h in ranked})
            corroboration = 0.5 + 0.5 * min(unique_sources / 5, 1.0)
            sp_avg = sum(SOURCE_PRIORITY.get(h.source, 0.5) for h in ranked) / len(ranked)
            med_high = sum(1 for c in conflicts if c.severity in ("medium", "high"))
            conflict_penalty = min(CONFLICT_PENALTY_CAP, CONFLICT_PENALTY_PER_ITEM * med_high)
            overall_confidence = _clamp(
                0.35 * relevance_avg
                + 0.20 * freshness_avg
                + 0.15 * sp_avg
                + 0.20 * corroboration
                - 0.10 * (1.0 if conflict_penalty > 0 else 0.0)
            )
        else:
            overall_confidence = 0.0
            corroboration = 0.0

        # Citations
        citations = _build_citations(ranked, observed_at=observed_at)

        # Summary
        summary_text = _build_summary(
            ranked, conflicts, overall_freshness, overall_confidence
        )

        # Fingerprints
        qfp = query.fingerprint()
        sfp = _sha256_hex({
            "objective_id": objective_id,
            "sources_queried": sorted(sources_queried),
            "hits_fingerprints_sorted": sorted(h.fingerprint for h in ranked),
            "sources_failed": sorted(sources_failed),
            "schema_version": SCHEMA_VERSION,
        })

        t1 = self._monotonic()
        duration_ms = max(0, int((t1 - t0) * 1000))

        return EvidencePack(
            objective_id=objective_id,
            query_fingerprint=qfp,
            sources_queried=sources_queried,
            sources_failed=sources_failed,
            hits=ranked,
            citations=citations,
            conflicts=conflicts,
            missing_information=missing,
            overall_freshness_score=overall_freshness,
            overall_confidence=overall_confidence,
            summary_text=summary_text,
            summary_fingerprint=sfp,
            duration_ms=duration_ms,
            created_at=observed_at,
            schema_version=SCHEMA_VERSION,
            is_idempotent_reuse=False,
            total_hits=len(ranked),
        )

    def discover(
        self,
        objective_id: str,
        objective_text: str,
        **kwargs: Any,
    ) -> EvidencePack:
        """dry_run + persist single state_meta key (if storage present). Idempotent."""
        pack = self.dry_run(objective_id, objective_text, **kwargs)
        if self._storage is not None and hasattr(self._storage, "_state_meta"):
            key = f"{self.STATE_META_PREFIX}{objective_id}:{self.STATE_META_KEY_VERSION}"
            existing = self._state_meta_get(key)
            if existing is not None and existing.get("query_fingerprint") == pack.query_fingerprint:
                pack.is_idempotent_reuse = True
                return pack
            self._state_meta_set(key, pack.to_dict())
        return pack

    def rollback(self, objective_id: str) -> bool:
        """Idempotent: returns True if something was deleted, False otherwise."""
        if self._storage is None or not hasattr(self._storage, "_state_meta"):
            return False
        key = f"{self.STATE_META_PREFIX}{objective_id}:{self.STATE_META_KEY_VERSION}"
        existing = self._state_meta_get(key)
        if existing is None:
            return False
        self._state_meta_delete(key)
        return True

    # ── state_meta adapter (works with both ObjectiveStateStorage and FakeDB) ──

    def _state_meta_get(self, key: str) -> Optional[dict]:
        s = self._storage
        # FakeDB-like
        if hasattr(s, "_state_meta") and isinstance(getattr(s, "_state_meta", None), dict):
            raw = s._state_meta.get(key)
            if raw is None:
                return None
            if isinstance(raw, (bytes, str)):
                try:
                    return json.loads(raw)
                except (TypeError, ValueError):
                    return None
            if isinstance(raw, dict):
                return dict(raw)
        # ObjectiveStateStorage-like (set/get_meta)
        if hasattr(s, "get_meta"):
            try:
                raw = s.get_meta(key)
            except Exception:
                return None
            if raw is None:
                return None
            if isinstance(raw, str):
                try:
                    return json.loads(raw)
                except ValueError:
                    return None
            if isinstance(raw, dict):
                return dict(raw)
        return None

    def _state_meta_set(self, key: str, value: dict) -> None:
        s = self._storage
        if hasattr(s, "_state_meta") and isinstance(getattr(s, "_state_meta", None), dict):
            s._state_meta[key] = value
            return
        if hasattr(s, "set_meta"):
            s.set_meta(key, json.dumps(value))
            return

    def _state_meta_delete(self, key: str) -> None:
        s = self._storage
        if hasattr(s, "_state_meta") and isinstance(getattr(s, "_state_meta", None), dict):
            s._state_meta.pop(key, None)
            return
        if hasattr(s, "delete_meta"):
            s.delete_meta(key)
            return


# ─────────────────────────────────────────────────────────────────────
# Helper: detect KnowledgeQuery shape on state_storage (for backwards compat)
# ─────────────────────────────────────────────────────────────────────


def get_state_meta_key(objective_id: str) -> str:
    return f"{EvidencePackEngine.STATE_META_PREFIX}{objective_id}:{EvidencePackEngine.STATE_META_KEY_VERSION}"

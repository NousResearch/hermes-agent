"""EvidencePack v1 + EvidencePackEngine — production module.

Standalone evidence-pack core. It performs no real source, state,
network, subprocess, or model access. External behavior is supplied
through injected sources, storage, and audit sinks.

The engine is an explicitly invoked standalone core. It has no
configuration or environment-variable activation path.

Hermeticity invariants (preserved from the canary):
- No subprocess / urllib / requests / httpx / socket / aiohttp / ssl imports
- No LLM client imports
- No real GBrain / Obsidian imports (real adapters live in adapters.py
  and are wired via DI; Gate D / E2E)
- No state.db / audit log writes (the production wiring uses the same
  in-memory injection as the canary)

Production integration and real source adapters are intentionally
outside this standalone core and must be supplied by explicit consumers.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
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
SUMMARY_TEXT_MAX_LEN = 2000
SNIPPET_MAX_LEN = 500
TITLE_MAX_LEN = 256
HIT_ID_MAX_LEN = 512
QUOTE_MAX_LEN = 1000
STATEMENT_MAX_LEN = 500
IMPACT_MAX_LEN = 500
RECOMMENDATION_MAX_LEN = 500
OBJECTIVE_TEXT_MAX_LEN = 10_000

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
    * policy_vs_goal        — high (when a hit pair satisfies the
                              explicit-claim content rule)
    * memory_vs_evidence    — medium
    * evidence_vs_evidence  — low / medium
    * freshness             — low (delta > 30d)
    * scope                 — low
    * identity              — medium

    Conflict detection is content-aware. A source combination alone is NEVER
    sufficient to declare a conflict; a positive conflict must demonstrate:
    (1) the same normalized subject, (2) the same normalized attribute,
    (3) explicit polarity opposition OR explicit numeric values with the
    SAME explicit unit, and (4) intact provenance on both hits.
    Order of input is irrelevant — the pair is canonicalised before evaluation.

    Unparseable prose, free-form negation, and identifier-like numerics all
    default to no_conflict. False negatives are preferred over false positives.
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
            # ConflictRecord.items preserves canonical provenance for both
            # hits; the tuple order is sorted so input order cannot perturb
            # the identity of the conflict.
            cid = _conflict_id(_canonical_pair(a, b), ctype)
            conflicts.append(ConflictRecord(
                conflict_id=cid,
                conflict_type=ctype,
                severity=severity,
                items=_canonical_pair(a, b),
                impact=impact,
                recommended_resolution=rec,
                resolution_status="unresolved",
                detected_at=observed_at,
            ))
    return conflicts


# Closed polarity pairs (positive, negative). Recognition requires both
# halves to appear as exact normalized values of parsed explicit claims.
_POLARITY_PAIRS: tuple[tuple[str, ...], ...] = (
    ("approved", "rejected"),
    ("allowed", "forbidden"),
    ("enabled", "disabled"),
    ("active", "inactive"),
    ("pass", "fail"),
    ("true", "false"),
    ("yes", "no"),
    ("open", "closed"),
    ("on", "off"),
    ("successful", "unsuccessful"),
)

_POLARITY_WORDS: frozenset[str] = frozenset(
    w for pair in _POLARITY_PAIRS for w in pair
)

# Identifier-like attributes — numerics on these never produce a
# conflict. Auditable exclusion set: date / datetime / timestamp,
# version / revision, id / identifier, phone / telephone, port,
# ip / address, serial / model, build / commit, ticket / issue,
# account / postal / zip, plus the legacy token ``incidents`` and any
# attribute ending in an excluded suffix.
_NUMERIC_EXCLUDED_WORDS: frozenset[str] = frozenset({
    "date", "datetime", "timestamp",
    "version", "revision",
    "id", "identifier",
    "phone", "telephone",
    "port",
    "ip", "address",
    "serial", "model",
    "build", "commit",
    "ticket", "issue",
    "account", "postal", "zip",
    "incidents",
})
_NUMERIC_EXCLUDED_SUFFIXES: tuple[str, ...] = (
    "_id", "_version", "_date", "_port", "_phone",
    "_serial", "_build", "_ticket", "_issue",
)

# Ranker near-duplicate threshold (snippet-token Jaccard). Above this the
# pair is collapsed unless the explicit-claim opposition exception applies.
_NEAR_DUP_JACCARD = 0.85

# Explicit-claim grammar: ``subject attribute = value`` or
# ``subject attribute: value``. Subject and attribute are each a single
# token ``[a-z][a-z0-9_-]*``; value is the rest of the line, trimmed.
_EXPLICIT_CLAIM_RE = re.compile(
    r"^([a-z][a-z0-9_-]*)\s+([a-z][a-z0-9_-]*)\s*[:=]\s*(.+?)\s*$"
)

# Numeric value grammar: integer / decimal followed by an optional unit
# token (letters, %).
_NUMERIC_VALUE_RE = re.compile(r"^(\d+(?:\.\d+)?)(?:\s+([a-z%]+))?$")


def _canonical_pair(a: KnowledgeHitV2, b: KnowledgeHitV2) -> tuple[str, str]:
    """Return the hit_id pair in canonical (sorted) order.

    Ensures conflict detection is independent of input order.
    """
    return (a.hit_id, b.hit_id) if a.hit_id <= b.hit_id else (b.hit_id, a.hit_id)


def _parse_explicit_claim(text: Optional[str]) -> Optional[tuple[str, str, str]]:
    """Parse ``subject attribute = value`` or ``subject attribute: value``.

    Returns ``(subject, attribute, value)`` (all lower-cased and trimmed)
    or ``None``. The parser is purely structural — no NLP fallback, no
    token fallback, no title/context concatenation.
    """
    if not text:
        return None
    s = text.strip().lower()
    if not s:
        return None
    m = _EXPLICIT_CLAIM_RE.match(s)
    if not m:
        return None
    subject, attribute, value = m.group(1), m.group(2), m.group(3).strip()
    if not value:
        return None
    return (subject, attribute, value)


def _hit_claim(h: KnowledgeHitV2) -> Optional[tuple[str, str, str]]:
    """Inspect snippet first; fall back to ``provenance.quote`` if present.

    Never synthesises a claim from title or arbitrary context prose.
    """
    claim = _parse_explicit_claim(h.snippet)
    if claim is not None:
        return claim
    return _parse_explicit_claim(h.provenance.quote)


def _parse_numeric_value(value: str) -> Optional[tuple[str, str]]:
    """Parse a numeric value, returning ``(number, unit)`` or ``None``.

    ``unit`` is the empty string when no explicit unit is present.
    """
    m = _NUMERIC_VALUE_RE.match(value.strip())
    if not m:
        return None
    return (m.group(1), m.group(2) or "")


def _is_identifier_like_attribute(attr: str) -> bool:
    """True when ``attr`` represents an identifier-like class.

    Identifier-like attributes must never produce numeric conflicts —
    distinct identifiers are not contradictory measurements.
    """
    if attr in _NUMERIC_EXCLUDED_WORDS:
        return True
    for suffix in _NUMERIC_EXCLUDED_SUFFIXES:
        if attr.endswith(suffix):
            return True
    return False


def _polarity_opposite(a: str, b: str) -> bool:
    """True iff ``(a, b)`` are opposite members of one closed polarity pair."""
    a = a.strip()
    b = b.strip()
    for pair in _POLARITY_PAIRS:
        if (a == pair[0] and b == pair[1]) or (a == pair[1] and b == pair[0]):
            return True
    return False


def _explicit_claims_conflict(
    claim_a: tuple[str, str, str],
    claim_b: tuple[str, str, str],
) -> Optional[str]:
    """Return ``"polarity"`` / ``"numeric"`` for a content conflict, else ``None``.

    Conflict requires identical subject + identical attribute + differing
    values, and either an opposite closed-polarity pair OR comparable
    numerics with the SAME explicit unit. Identifier-like attributes
    suppress numeric conflicts.
    """
    sa, aa, va = claim_a
    sb, ab, vb = claim_b
    if sa != sb or aa != ab:
        return None
    if va == vb:
        return None
    if (
        va in _POLARITY_WORDS
        and vb in _POLARITY_WORDS
        and _polarity_opposite(va, vb)
    ):
        return "polarity"
    if _is_identifier_like_attribute(aa):
        return None
    na = _parse_numeric_value(va)
    nb = _parse_numeric_value(vb)
    if na is None or nb is None:
        return None
    if na[1] != nb[1]:
        return None
    if na[0] == nb[0]:
        return None
    return "numeric"


def _content_aware_conflict(
    a: KnowledgeHitV2, b: KnowledgeHitV2
) -> Optional[tuple[str, str]]:
    """Conflict iff both hits parse to explicit opposing claims.

    Source combination only SELECTS the conflict type / severity; it
    never proves a conflict.
    """
    ca = _hit_claim(a)
    cb = _hit_claim(b)
    if ca is None or cb is None:
        return None
    if _explicit_claims_conflict(ca, cb) is None:
        return None
    pair = {a.source, b.source}
    if "policy" in pair:
        return ("policy_vs_goal", "high")
    if pair == {"gbrain", "obsidian"} or pair == {"gbrain", "report"}:
        return ("memory_vs_evidence", "medium")
    if a.source == b.source:
        return ("evidence_vs_evidence", "medium")
    return ("evidence_vs_evidence", "low")


def _ranker_preserve_opposing_claims(
    h1: KnowledgeHitV2, h2: KnowledgeHitV2
) -> bool:
    """Ranker near-duplicate exception: keep both if claims oppose.

    Returns ``True`` iff both hits parse to explicit claims on the same
    subject + same attribute and the values differ as either
    closed-polarity opposites or numeric with identical explicit units.
    The function does NOT itself declare a conflict.
    """
    c1 = _hit_claim(h1)
    c2 = _hit_claim(h2)
    if c1 is None or c2 is None:
        return False
    return _explicit_claims_conflict(c1, c2) is not None


def _classify_conflict(
    a: KnowledgeHitV2, b: KnowledgeHitV2
) -> Optional[tuple[str, str, str, str]]:
    """Return (type, severity, impact, recommended_resolution) or None.

    Order of evaluation:
      1. Freshness delta — same source, very different updated_at.
      2. Identity — same hit_id with different source_uri (canonical
         identity rule; takes precedence over content-aware conflict).
      3. Cross-band freshness on the same source.
      4. Content-aware conflict rule (subject + attribute + incompatibility).
      5. Scope — same hit_id prefix, divergent content.

    The source-pair-only ``memory_vs_evidence`` and ``policy_vs_goal``
    heuristics have been replaced by ``_content_aware_conflict``. A source
    combination is never sufficient to declare a conflict.
    """
    # Freshness delta: a/b from same source, very different updated_at.
    if (a.source == b.source
            and abs(a.freshness.staleness_days - b.freshness.staleness_days) > 30):
        return (
            "freshness", "low",
            f"source_updated_at delta > 30d between {a.hit_id} and {b.hit_id}",
            "use newer source_updated_at; archive older",
        )

    # Identity: same hit_id, different source_uri.
    # IMPORTANT: this check is BEFORE the content-aware rule because
    # identity is the more specific classification (same entity, different
    # URI is an identity issue, not a content contradiction).
    if a.hit_id == b.hit_id and a.provenance.source_uri != b.provenance.source_uri:
        return (
            "identity", "medium",
            f"duplicate hit_id {a.hit_id} with different uris",
            "dedup by source_uri; keep higher-priority source",
        )

    # Evidence vs evidence: same source, freshness band delta.
    if a.source == b.source and a.freshness.freshness != b.freshness.freshness:
        return (
            "evidence_vs_evidence", "medium",
            f"same source {a.source} cross-band freshness",
            "prefer current; demote stale",
        )

    # Content-aware conflict rule. Replaces the old source-pair-only
    # memory_vs_evidence / policy_vs_goal heuristics.
    content_result = _content_aware_conflict(a, b)
    if content_result is not None:
        ctype, severity = content_result
        impact = (
            f"explicit incompatibility between {a.source} and {b.source} "
            f"on shared subject/attribute"
        )
        if ctype == "policy_vs_goal":
            rec = "requires_human; flag in human_gate_audit"
        elif ctype == "memory_vs_evidence":
            rec = "resolve by policy; default to higher source priority"
        else:
            rec = "prefer higher-priority source; reconcile via policy"
        return (ctype, severity, impact, rec)

    # Scope: same hit_id family (prefix), low token overlap, different source.
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

    # Near-dup Jaccard ≥ 0.85 → drop lower score. The Jaccard check only
    # applies across DIFFERENT sources — within a single source the
    # fingerprint dedup already collapses true duplicates (each hit
    # has a unique hit_id per document), and separate documents that
    # happen to render the same sentence must be preserved so the
    # conflict detector can evaluate them.
    #
    # Cross-source exception: when both snippets parse to explicit
    # opposing claims on the same subject + same attribute
    # (closed-polarity opposites OR comparable numerics with identical
    # explicit units), the ranker keeps BOTH hits so the conflict
    # detector can evaluate the pair. Same-value claims remain subject
    # to normal duplicate suppression. The exception is narrow —
    # arbitrary prose with different numbers is NOT preserved.
    final: list[KnowledgeHitV2] = []
    for h in deduped:
        tokens_h = _tokenize(h.snippet)
        is_dup = False
        for kept in final:
            if kept.source == h.source:
                continue  # fingerprint dedup already handled within source
            if _jaccard(tokens_h, _tokenize(kept.snippet)) < _NEAR_DUP_JACCARD:
                continue
            if _ranker_preserve_opposing_claims(h, kept):
                continue  # opposing explicit claims — keep both
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
        self._sources: dict[str, Callable[..., list[KnowledgeHitV2]]] = sources or {}
        self._storage = storage
        self._audit_sink = audit_sink if audit_sink is not None else _AuditSink()
        self._monotonic = time.monotonic

    # ── public API ──────────────────────────────────────────────

    def _effective_objective_text(self, objective_text: str) -> str:
        """Return the normalized objective text used by every downstream stage."""
        return (objective_text or "")[:OBJECTIVE_TEXT_MAX_LEN]

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
    ) -> EvidencePack:
        """Build an EvidencePack without persisting anywhere."""
        t0 = self._monotonic()
        observed_at = _now_iso8601()
        raw_objective_text = objective_text or ""
        objective_text = self._effective_objective_text(raw_objective_text)
        objective_text_was_clamped = len(raw_objective_text) > OBJECTIVE_TEXT_MAX_LEN
        if max_hits_per_source < 1:
            max_hits_per_source = 1
        if max_hits_total < 1:
            max_hits_total = 1

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

        if objective_text_was_clamped:
            missing.append(
                f"objective_text clamped to {OBJECTIVE_TEXT_MAX_LEN} chars"
            )

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

    # ── public metadata API (idempotency contract) ───────────────────────────

    def get_meta(self, key: str) -> Optional[dict]:
        """Public API: retrieve metadata by key.

        Returns None if key does not exist or storage is unavailable.
        Detects storage by public methods only (get_meta), not by private attributes.
        """
        if self._storage is None:
            return None
        # Detect storage by public get_meta method only
        if not hasattr(self._storage, "get_meta") or not callable(self._storage.get_meta):
            return None
        try:
            raw = self._storage.get_meta(key)
        except Exception:
            return None
        if raw is None:
            return None
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except (TypeError, ValueError):
                return None
        if isinstance(raw, dict):
            return dict(raw)
        return None

    def set_meta(self, key: str, value: dict) -> None:
        """Public API: store metadata by key.

        Detects storage by public methods only (set_meta), not by private attributes.
        Raises RuntimeError if storage is unavailable or write fails.
        """
        if self._storage is None:
            raise RuntimeError("No storage configured")
        if not hasattr(self._storage, "set_meta") or not callable(self._storage.set_meta):
            raise RuntimeError("Storage does not support set_meta")
        try:
            self._storage.set_meta(key, json.dumps(value))
        except Exception as e:
            raise RuntimeError(f"Failed to set metadata: {e}")

    # ── discover (idempotent) ─────────────────────────────────────────────

    def discover(
        self,
        objective_id: str,
        objective_text: str,
        **kwargs: Any,
    ) -> EvidencePack:
        """dry_run + persist single metadata key (if storage supports get_meta/set_meta). Idempotent."""
        # Check cache BEFORE running dry_run to avoid unnecessary provider calls
        if self._storage is not None and hasattr(self._storage, "get_meta") and callable(self._storage.get_meta):
            key = f"{self.STATE_META_PREFIX}{objective_id}:{self.STATE_META_KEY_VERSION}"
            existing = self.get_meta(key)
            if existing is not None and existing.get("query_fingerprint") == self._compute_fingerprint(objective_id, objective_text, **kwargs):
                # Reconstruct EvidencePack from cached metadata
                pack = EvidencePack(
                    objective_id=existing.get("objective_id", objective_id),
                    query_fingerprint=existing.get("query_fingerprint", ""),
                    sources_queried=existing.get("sources_queried", []),
                    sources_failed=existing.get("sources_failed", []),
                    hits=[],
                    citations=[],
                    conflicts=[],
                    missing_information=existing.get("missing_information", []),
                    overall_freshness_score=existing.get("overall_freshness_score", 0.0),
                    overall_confidence=existing.get("overall_confidence", 0.0),
                    summary_text=existing.get("summary_text", ""),
                    summary_fingerprint=existing.get("summary_fingerprint", ""),
                    duration_ms=existing.get("duration_ms", 0),
                    created_at=existing.get("created_at", ""),
                    schema_version=existing.get("schema_version", SCHEMA_VERSION),
                    is_idempotent_reuse=True,
                    total_hits=existing.get("total_hits", 0),
                )
                return pack

        # Cache miss - run dry_run and persist
        pack = self.dry_run(objective_id, objective_text, **kwargs)
        if self._storage is not None and hasattr(self._storage, "get_meta") and callable(self._storage.get_meta):
            key = f"{self.STATE_META_PREFIX}{objective_id}:{self.STATE_META_KEY_VERSION}"
            self.set_meta(key, pack.to_dict())
        return pack

    def _compute_fingerprint(self, objective_id: str, objective_text: str, **kwargs) -> str:
        """Compute query fingerprint for cache key matching."""
        goal_class = kwargs.get("goal_class", "OTHER")
        risk_profile = kwargs.get("risk_profile", "low")
        complexity = kwargs.get("complexity", "S")
        max_hits_per_source = kwargs.get("max_hits_per_source", MAX_HITS_PER_SOURCE_DEFAULT)
        sources_requested = kwargs.get("sources_requested", tuple(self._sources.keys()))
        objective_text = self._effective_objective_text(objective_text)
        return _sha256_hex({
            "objective_id": objective_id,
            "objective_text": objective_text,
            "goal_class": goal_class,
            "risk_profile": risk_profile,
            "complexity": complexity,
            "max_hits_per_source": max_hits_per_source,
            "sources_requested": sorted(sources_requested) if sources_requested else [],
            "schema_version": SCHEMA_VERSION,
        })

    def rollback(self, objective_id: str) -> bool:
        """Idempotent: returns True if something was deleted, False otherwise.

        Uses public get_meta/set_meta for detection and deletion.
        """
        if self._storage is None:
            return False
        # Detect storage by public methods only
        if not hasattr(self._storage, "get_meta") or not callable(self._storage.get_meta):
            return False
        key = f"{self.STATE_META_PREFIX}{objective_id}:{self.STATE_META_KEY_VERSION}"
        existing = self.get_meta(key)
        if existing is None:
            return False
        # Use delete_meta if available, otherwise set to None to mark as deleted
        if hasattr(self._storage, "delete_meta") and callable(self._storage.delete_meta):
            try:
                self._storage.delete_meta(key)
            except Exception:
                return False
        else:
            # Fallback: set to empty to indicate deletion
            try:
                self.set_meta(key, {"_deleted": True})
            except Exception:
                return False
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

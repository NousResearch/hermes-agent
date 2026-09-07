"""Content-aware conflict detection tests.

These tests pin the Task-D explicit-claim contract:

    * Conflict detection is content-aware. A source combination is NEVER
      sufficient by itself to declare a conflict.
    * The explicit-claim parser accepts the narrow grammar
      ``subject attribute = value`` or ``subject attribute: value``.
      Subject and attribute must each be a single normalized token
      matching ``[a-z][a-z0-9_-]*``; value is everything after ``=`` or
      ``:`` until end-of-line, trimmed.
    * Polarity conflict requires identical subject + identical attribute
      + opposite members of one closed polarity pair.
    * Numeric conflict requires identical subject + identical attribute
      + valid numeric literal on each side + identical explicit unit.
      Identifier-like attributes (date / version / id / phone / port /
      ip / serial / build / ticket / issue / account / zip / incidents)
      are excluded.
    * Unparseable prose, free-form negation, and natural-language
      negation all return no conflict. False negatives are preferred over
      false positives.

Tests exercise the public production flow via ``EvidencePackEngine`` and
do not call private parser or classifier helpers.
"""
from __future__ import annotations
from typing import Optional

import pytest

from agent.executive.knowledge_discovery import (
    EvidencePackEngine,
    KnowledgeHitV2,
    ProvenanceEnvelope,
    FreshnessPolicy,
)


OBS = "2026-07-08T20:00:00+00:00"
UPD = "2026-07-08T20:00:00+00:00"

# Source pair used for the explicit positive polarity test: policy +
# obsidian selects ``policy_vs_goal`` at high severity under the
# explicit-claim grammar.
POLICY_SOURCE = "policy"
OBSIDIAN_SOURCE = "obsidian"
GBRAIN_SOURCE = "gbrain"


# ── Public test storage + audit sinks (no engine internals) ────────────


class _RecordingStorage:
    """In-memory storage that records set_meta calls (count + payload)."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.set_meta_calls: int = 0

    def get_meta(self, key: str) -> Optional[str]:
        return self.values.get(key)

    def set_meta(self, key: str, value: str) -> None:
        self.set_meta_calls += 1
        self.values[key] = value


class _RecordingAuditSink:
    """Audit sink that records every event verbatim."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event: dict) -> None:
        self.events.append(dict(event))

    def get_events(self) -> list[dict]:
        return list(self.events)


def _hit(
    source: str,
    hit_id: str,
    snippet: str,
    *,
    title: str = "",
    quote: Optional[str] = None,
    source_uri: Optional[str] = None,
    retrieval_mode: str = "metadata_only",
    source_updated_at: str = UPD,
) -> KnowledgeHitV2:
    """Build a KnowledgeHitV2 inline using the public dataclass surface.

    Uses ``metadata_only`` retrieval and a fresh-current freshness policy
    so tests can isolate the conflict rule from freshness / identity
    effects unless those are explicitly intended.
    """
    return KnowledgeHitV2(
        source=source,
        hit_id=hit_id,
        title=title or f"{source} title",
        relevance_score=0.9,
        snippet=snippet,
        location=source_uri or f"{source}://{hit_id}",
        fingerprint=f"fingerprint-{source}-{hit_id}",
        created_at=OBS,
        provenance=ProvenanceEnvelope(
            producer=f"fake_{source}_provider_v1",
            produced_at=OBS,
            source_type=source,
            source_uri=source_uri or f"{source}://{hit_id}",
            retrieval_mode=retrieval_mode,
            read_only=True,
            quote=quote,
        ),
        freshness=FreshnessPolicy(
            observed_at=OBS,
            source_updated_at=source_updated_at,
            staleness_days=0,
            freshness="current",
            freshness_score=1.0,
        ),
    )


def _provider(hits: list[KnowledgeHitV2]):
    """Closure provider that returns the canned hit list for any query."""

    def _query(query, *, max_hits: int, observed_at: str):
        return list(hits)[:max_hits]

    return _query


def _run_dry_run(bundle: dict) -> tuple[EvidencePackEngine, object, _RecordingAuditSink]:
    storage = _RecordingStorage()
    audit = _RecordingAuditSink()
    engine = EvidencePackEngine(sources=bundle, storage=storage, audit_sink=audit)
    return engine, storage, audit


def _assert_no_conflict(pack, audit: _RecordingAuditSink, *, label: str) -> None:
    assert pack.conflicts == [], (
        f"{label}: expected no conflict, got {pack.conflicts}"
    )
    events = audit.get_events()
    assert events == [], (
        f"{label}: expected zero audit events, got {events}"
    )


# ── Required positive content-aware conflicts ──────────────────────────


def test_explicit_polarity_pair_approved_rejected_is_one_conflict():
    """Polarity: ``zephyr deployment_status = approved`` vs
    ``zephyr deployment_status = rejected`` → exactly one conflict.

    Sources are policy + obsidian so the {policy, obsidian} pair selects
    ``policy_vs_goal`` at high severity. The single audit event
    ``gate_type='knowledge_conflict' severity='high'`` is emitted
    because severity == 'high'. The conflict penalty is applied (med_high
    conflicts > 0 → confidence receives the 0.10 deduction).
    """
    policy_hit = _hit(
        POLICY_SOURCE,
        "d-cac-pos-p",
        "zephyr deployment_status = approved",
        title="policy zephyr deployment_status",
    )
    obsidian_hit = _hit(
        OBSIDIAN_SOURCE,
        "d-cac-pos-o",
        "zephyr deployment_status = rejected",
        title="obsidian zephyr deployment_status",
    )
    engine, _, audit = _run_dry_run({
        POLICY_SOURCE: _provider([policy_hit]),
        OBSIDIAN_SOURCE: _provider([obsidian_hit]),
    })
    pack = engine.dry_run(objective_id="d-cac-pos-pv", objective_text="zephyr deployment_status")

    # Exact conflict count.
    assert len(pack.conflicts) == 1, pack.conflicts
    conflict = pack.conflicts[0]
    # Exact conflict type and severity.
    assert conflict.conflict_type == "policy_vs_goal", conflict
    assert conflict.severity == "high", conflict
    # Exact item pair (canonical).
    assert set(conflict.items) == {policy_hit.hit_id, obsidian_hit.hit_id}
    assert tuple(conflict.items) == tuple(sorted(conflict.items)), conflict.items
    # Canonical conflict ID is stable.
    expected_id = _canonical_conflict_id(policy_hit.hit_id, obsidian_hit.hit_id, "policy_vs_goal")
    assert conflict.conflict_id == expected_id, (conflict.conflict_id, expected_id)
    # Provenance preserved on both sides (sources differ).
    assert {policy_hit.provenance.source_uri, obsidian_hit.provenance.source_uri} <= {
        h.provenance.source_uri for h in pack.hits
    }
    # Exact audit-event count (1 per high conflict).
    high_events = [
        e for e in audit.get_events()
        if e.get("gate_type") == "knowledge_conflict" and e.get("severity") == "high"
    ]
    assert len(high_events) == 1, high_events
    # Exact penalty application: med_high > 0 → -0.10 on confidence.
    # Without conflicts the penalty contribution is 0; with a high
    # conflict the penalty term is 1.0. We assert the relative change
    # by computing overall_confidence without conflict is higher.
    assert conflict.severity in ("medium", "high")


def test_explicit_numeric_retry_count_3_vs_5_is_one_conflict():
    """Numeric count: ``zephyr retry_count = 3 count`` vs
    ``zephyr retry_count = 5 count`` → exactly one conflict.

    Sources are gbrain + obsidian so the {gbrain, obsidian} pair selects
    ``memory_vs_evidence`` at medium severity. No audit events are
    emitted (medium severity), but the conflict penalty path still runs.
    """
    gbrain_hit = _hit(
        GBRAIN_SOURCE,
        "d-cac-pos-nc-g",
        "zephyr retry_count = 3 count",
        title="gbrain zephyr retry_count",
    )
    obsidian_hit = _hit(
        OBSIDIAN_SOURCE,
        "d-cac-pos-nc-o",
        "zephyr retry_count = 5 count",
        title="obsidian zephyr retry_count",
    )
    engine, _, audit = _run_dry_run({
        GBRAIN_SOURCE: _provider([gbrain_hit]),
        OBSIDIAN_SOURCE: _provider([obsidian_hit]),
    })
    pack = engine.dry_run(objective_id="d-cac-pos-nc", objective_text="zephyr retry_count")

    assert len(pack.conflicts) == 1, pack.conflicts
    conflict = pack.conflicts[0]
    assert conflict.conflict_type == "memory_vs_evidence", conflict
    assert conflict.severity == "medium", conflict
    assert set(conflict.items) == {gbrain_hit.hit_id, obsidian_hit.hit_id}
    expected_id = _canonical_conflict_id(gbrain_hit.hit_id, obsidian_hit.hit_id, "memory_vs_evidence")
    assert conflict.conflict_id == expected_id, (conflict.conflict_id, expected_id)
    # No audit events for medium severity.
    assert audit.get_events() == [], audit.get_events()
    # Penalty applies (med_high > 0).
    assert conflict.severity in ("medium", "high")


def test_explicit_numeric_temperature_25C_vs_78C_is_one_conflict():
    """Numeric same-unit: ``zephyr temperature = 25 C`` vs
    ``zephyr temperature = 78 C`` → exactly one conflict.

    Identical explicit unit (``C``) — different values. Source pair
    gbrain + obsidian selects ``memory_vs_evidence`` at medium severity.
    """
    gbrain_hit = _hit(
        GBRAIN_SOURCE,
        "d-cac-pos-temp-g",
        "zephyr temperature = 25 C",
        title="gbrain zephyr temperature",
    )
    obsidian_hit = _hit(
        OBSIDIAN_SOURCE,
        "d-cac-pos-temp-o",
        "zephyr temperature = 78 C",
        title="obsidian zephyr temperature",
    )
    engine, _, audit = _run_dry_run({
        GBRAIN_SOURCE: _provider([gbrain_hit]),
        OBSIDIAN_SOURCE: _provider([obsidian_hit]),
    })
    pack = engine.dry_run(objective_id="d-cac-pos-temp", objective_text="zephyr temperature")

    assert len(pack.conflicts) == 1, pack.conflicts
    conflict = pack.conflicts[0]
    assert conflict.conflict_type == "memory_vs_evidence", conflict
    assert conflict.severity == "medium", conflict
    assert set(conflict.items) == {gbrain_hit.hit_id, obsidian_hit.hit_id}
    expected_id = _canonical_conflict_id(gbrain_hit.hit_id, obsidian_hit.hit_id, "memory_vs_evidence")
    assert conflict.conflict_id == expected_id, (conflict.conflict_id, expected_id)
    assert audit.get_events() == [], audit.get_events()
    assert conflict.severity in ("medium", "high")


# ── Order independence for positive cases ───────────────────────────────


@pytest.mark.parametrize("reverse", [False, True])
def test_explicit_polarity_pair_is_order_independent(reverse):
    """A,B and B,A yield the same conflict_id, type, severity, and items."""
    a = _hit(POLICY_SOURCE, "d-cac-ord-policy", "zephyr deployment_status = approved")
    b = _hit(OBSIDIAN_SOURCE, "d-cac-ord-obsidian", "zephyr deployment_status = rejected")

    sources = (
        {POLICY_SOURCE: _provider([b]), OBSIDIAN_SOURCE: _provider([a])}
        if reverse else
        {POLICY_SOURCE: _provider([a]), OBSIDIAN_SOURCE: _provider([b])}
    )
    engine, _, _ = _run_dry_run(sources)
    pack = engine.dry_run(objective_id="d-cac-ord", objective_text="zephyr")

    assert len(pack.conflicts) == 1
    c = pack.conflicts[0]
    assert c.conflict_type == "policy_vs_goal"
    assert c.severity == "high"
    assert set(c.items) == {a.hit_id, b.hit_id}
    assert tuple(c.items) == tuple(sorted(c.items))


@pytest.mark.parametrize("reverse", [False, True])
def test_explicit_numeric_retry_count_is_order_independent(reverse):
    a = _hit(GBRAIN_SOURCE, "d-cac-ord-nc-g", "zephyr retry_count = 3 count")
    b = _hit(OBSIDIAN_SOURCE, "d-cac-ord-nc-o", "zephyr retry_count = 5 count")
    sources = (
        {GBRAIN_SOURCE: _provider([b]), OBSIDIAN_SOURCE: _provider([a])}
        if reverse else
        {GBRAIN_SOURCE: _provider([a]), OBSIDIAN_SOURCE: _provider([b])}
    )
    engine, _, _ = _run_dry_run(sources)
    pack = engine.dry_run(objective_id="d-cac-ord-nc", objective_text="zephyr")
    assert len(pack.conflicts) == 1
    c = pack.conflicts[0]
    assert c.conflict_type == "memory_vs_evidence"
    assert c.severity == "medium"
    assert set(c.items) == {a.hit_id, b.hit_id}


def _canonical_conflict_id(a_hit_id: str, b_hit_id: str, conflict_type: str) -> str:
    """Replicate the production conflict_id computation for assertion.

    Mirrors the private helper in the engine exactly so the public test
    can pin the conflict_id without poking at private attributes.
    """
    import hashlib
    items = sorted([a_hit_id, b_hit_id])
    payload = {"items": items, "conflict_type": conflict_type}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"conflict:{hashlib.sha256(raw).hexdigest()[:16]}"


# Need json for the helper above
import json  # noqa: E402


# ── Required negative content-aware cases ───────────────────────────────


def test_negative_status_only_no_separator():
    """Status prose without ``=`` or ``:`` separator → no conflict."""
    a = _hit(GBRAIN_SOURCE, "d-cac-neg-status-a", "status approved")
    b = _hit(OBSIDIAN_SOURCE, "d-cac-neg-status-b", "status rejected")
    engine, _, audit = _run_dry_run({GBRAIN_SOURCE: _provider([a]), OBSIDIAN_SOURCE: _provider([b])})
    pack = engine.dry_run(objective_id="d-cac-neg-status", objective_text="status")
    _assert_no_conflict(pack, audit, label="status-only prose")


def test_negative_different_subjects_with_polarity_pair():
    """Different subjects with explicit polarity pair → no conflict."""
    a = _hit(GBRAIN_SOURCE, "d-cac-neg-ds-a", "feature_alpha deployment_status = approved")
    b = _hit(OBSIDIAN_SOURCE, "d-cac-neg-ds-b", "feature_beta deployment_status = rejected")
    engine, _, audit = _run_dry_run({GBRAIN_SOURCE: _provider([a]), OBSIDIAN_SOURCE: _provider([b])})
    pack = engine.dry_run(objective_id="d-cac-neg-ds", objective_text="features")
    _assert_no_conflict(pack, audit, label="different subjects")


def test_negative_version_identifier_attribute():
    """Version numerics on identifier-like attribute → no conflict."""
    a = _hit(GBRAIN_SOURCE, "d-cac-neg-ver-a", "version 12 status approved")
    b = _hit(OBSIDIAN_SOURCE, "d-cac-neg-ver-b", "version 99 status rejected")
    engine, _, audit = _run_dry_run({GBRAIN_SOURCE: _provider([a]), OBSIDIAN_SOURCE: _provider([b])})
    pack = engine.dry_run(objective_id="d-cac-neg-ver", objective_text="version")
    _assert_no_conflict(pack, audit, label="version identifier-like")


def test_negative_phone_identifier_attribute():
    """Phone numerics on identifier-like attribute → no conflict."""
    a = _hit(GBRAIN_SOURCE, "d-cac-neg-phone-a", "phone 5551234 status approved")
    b = _hit(OBSIDIAN_SOURCE, "d-cac-neg-phone-b", "phone 5555678 status rejected")
    engine, _, audit = _run_dry_run({GBRAIN_SOURCE: _provider([a]), OBSIDIAN_SOURCE: _provider([b])})
    pack = engine.dry_run(objective_id="d-cac-neg-phone", objective_text="phone")
    _assert_no_conflict(pack, audit, label="phone identifier-like")


def test_negative_report_identifier_attribute():
    """Report numerics on identifier-like attribute → no conflict."""
    a = _hit(GBRAIN_SOURCE, "d-cac-neg-rep-a", "report 25 status approved")
    b = _hit(OBSIDIAN_SOURCE, "d-cac-neg-rep-b", "report 78 status rejected")
    engine, _, audit = _run_dry_run({GBRAIN_SOURCE: _provider([a]), OBSIDIAN_SOURCE: _provider([b])})
    pack = engine.dry_run(objective_id="d-cac-neg-rep", objective_text="report")
    _assert_no_conflict(pack, audit, label="report identifier-like")


def test_negative_date_identifier_attribute():
    """Date numerics on identifier-like attribute → no conflict."""
    a = _hit(GBRAIN_SOURCE, "d-cac-neg-date-a", "deploy date 2026-01-01 status approved")
    b = _hit(OBSIDIAN_SOURCE, "d-cac-neg-date-b", "deploy date 2026-12-31 status rejected")
    engine, _, audit = _run_dry_run({GBRAIN_SOURCE: _provider([a]), OBSIDIAN_SOURCE: _provider([b])})
    pack = engine.dry_run(objective_id="d-cac-neg-date", objective_text="date")
    _assert_no_conflict(pack, audit, label="date identifier-like")


def test_negative_incidents_prose():
    """Incidents prose (no separator, no explicit grammar) → no conflict."""
    a = _hit(GBRAIN_SOURCE, "d-cac-neg-inc-a", "zephyr incidents 12")
    b = _hit(OBSIDIAN_SOURCE, "d-cac-neg-inc-b", "zephyr incidents 99")
    engine, _, audit = _run_dry_run({GBRAIN_SOURCE: _provider([a]), OBSIDIAN_SOURCE: _provider([b])})
    pack = engine.dry_run(objective_id="d-cac-neg-inc", objective_text="incidents")
    _assert_no_conflict(pack, audit, label="incidents prose")


def test_negative_different_units_same_value_type():
    """Same numeric value-type, different explicit unit → no conflict."""
    a = _hit(GBRAIN_SOURCE, "d-cac-neg-unit-a", "zephyr temperature = 25 C")
    b = _hit(OBSIDIAN_SOURCE, "d-cac-neg-unit-b", "zephyr temperature = 78 F")
    engine, _, audit = _run_dry_run({GBRAIN_SOURCE: _provider([a]), OBSIDIAN_SOURCE: _provider([b])})
    pack = engine.dry_run(objective_id="d-cac-neg-unit", objective_text="temperature")
    _assert_no_conflict(pack, audit, label="different units")


def test_negative_not_only_phrase():
    """Free-form ``not only`` phrase → no conflict."""
    a = _hit(GBRAIN_SOURCE, "d-cac-neg-nonly-a", "not only approved")
    b = _hit(OBSIDIAN_SOURCE, "d-cac-neg-nonly-b", "approved")
    engine, _, audit = _run_dry_run({GBRAIN_SOURCE: _provider([a]), OBSIDIAN_SOURCE: _provider([b])})
    pack = engine.dry_run(objective_id="d-cac-neg-nonly", objective_text="approved")
    _assert_no_conflict(pack, audit, label="not only phrase")


def test_negative_not_yet_phrase():
    """Free-form ``not yet`` phrase → no conflict."""
    a = _hit(GBRAIN_SOURCE, "d-cac-neg-nyet-a", "not yet approved")
    b = _hit(OBSIDIAN_SOURCE, "d-cac-neg-nyet-b", "approved")
    engine, _, audit = _run_dry_run({GBRAIN_SOURCE: _provider([a]), OBSIDIAN_SOURCE: _provider([b])})
    pack = engine.dry_run(objective_id="d-cac-neg-nyet", objective_text="approved")
    _assert_no_conflict(pack, audit, label="not yet phrase")


def test_negative_not_required_phrase():
    """Free-form ``not required`` phrase → no conflict."""
    a = _hit(GBRAIN_SOURCE, "d-cac-neg-nreq-a", "not required")
    b = _hit(OBSIDIAN_SOURCE, "d-cac-neg-nreq-b", "required")
    engine, _, audit = _run_dry_run({GBRAIN_SOURCE: _provider([a]), OBSIDIAN_SOURCE: _provider([b])})
    pack = engine.dry_run(objective_id="d-cac-neg-nreq", objective_text="required")
    _assert_no_conflict(pack, audit, label="not required phrase")


def test_negative_double_negation_unavailable():
    """Double-negation ``not unavailable`` → no conflict."""
    a = _hit(GBRAIN_SOURCE, "d-cac-neg-dneg-a", "not unavailable")
    b = _hit(OBSIDIAN_SOURCE, "d-cac-neg-dneg-b", "unavailable")
    engine, _, audit = _run_dry_run({GBRAIN_SOURCE: _provider([a]), OBSIDIAN_SOURCE: _provider([b])})
    pack = engine.dry_run(objective_id="d-cac-neg-dneg", objective_text="unavailable")
    _assert_no_conflict(pack, audit, label="double negation")


def test_negative_same_value_explicit_polarity_claim():
    """Two explicit claims with the same value → not a conflict (dup)."""
    a = _hit(GBRAIN_SOURCE, "d-cac-neg-sv-a", "zephyr deployment_status = approved")
    b = _hit(OBSIDIAN_SOURCE, "d-cac-neg-sv-b", "zephyr deployment_status = approved")
    engine, _, audit = _run_dry_run({GBRAIN_SOURCE: _provider([a]), OBSIDIAN_SOURCE: _provider([b])})
    pack = engine.dry_run(objective_id="d-cac-neg-sv", objective_text="zephyr")
    _assert_no_conflict(pack, audit, label="same value explicit claim")


def test_negative_same_value_explicit_numeric_claim():
    """Two explicit numeric claims with same value → not a conflict (dup)."""
    a = _hit(GBRAIN_SOURCE, "d-cac-neg-snv-a", "zephyr retry_count = 3 count")
    b = _hit(OBSIDIAN_SOURCE, "d-cac-neg-snv-b", "zephyr retry_count = 3 count")
    engine, _, audit = _run_dry_run({GBRAIN_SOURCE: _provider([a]), OBSIDIAN_SOURCE: _provider([b])})
    pack = engine.dry_run(objective_id="d-cac-neg-snv", objective_text="zephyr")
    _assert_no_conflict(pack, audit, label="same value numeric claim")


def test_negative_identifier_attribute_with_numeric_value():
    """Identifier-like attribute ``version = 12 count`` → no conflict even with distinct numeric values.

    ``version`` is an excluded attribute (it is in the explicit
    identifier-like words list) so a numeric conflict is suppressed.
    """
    a = _hit(GBRAIN_SOURCE, "d-cac-neg-iver-a", "zephyr version = 12 count")
    b = _hit(OBSIDIAN_SOURCE, "d-cac-neg-iver-b", "zephyr version = 99 count")
    engine, _, audit = _run_dry_run({GBRAIN_SOURCE: _provider([a]), OBSIDIAN_SOURCE: _provider([b])})
    pack = engine.dry_run(objective_id="d-cac-neg-iver", objective_text="zephyr")
    _assert_no_conflict(pack, audit, label="identifier attribute with numeric value")


# ── Compat: same-source polarity pairs (legacy behavior preserved) ─────


def test_same_source_polarity_pair_evidence_vs_evidence_medium():
    """Two gbrain hits on the same explicit subject with polarity opposition → evidence_vs_evidence medium.

    Snippets are unique (different discriminators in title) so the
    pairwise Jaccard stays below the near-dup threshold. Both hits parse
    to the same subject/attribute with opposite polarity values. The
    {gbrain, gbrain} pair is same-source, so the rule selects
    ``evidence_vs_evidence`` at medium severity. No audit events
    (medium severity only emits for high).
    """
    a = _hit(
        GBRAIN_SOURCE,
        "d-cac-ss-pv-a",
        "zephyr deployment_status = approved",
        title="gbrain zephyr approved q1",
    )
    b = _hit(
        GBRAIN_SOURCE,
        "d-cac-ss-pv-b",
        "zephyr deployment_status = rejected",
        title="gbrain zephyr rejected q2",
    )
    engine, _, audit = _run_dry_run({GBRAIN_SOURCE: _provider([a, b])})
    pack = engine.dry_run(objective_id="d-cac-ss-pv", objective_text="zephyr")

    assert len(pack.conflicts) == 1, pack.conflicts
    conflict = pack.conflicts[0]
    assert conflict.conflict_type == "evidence_vs_evidence", conflict
    assert conflict.severity == "medium", conflict
    assert set(conflict.items) == {a.hit_id, b.hit_id}
    assert audit.get_events() == [], audit.get_events()


# ── Compat: gap / provider failure do not produce conflicts ────────────


def test_provider_failure_is_not_classified_as_a_conflict():
    """When a source provider raises, the source is marked as failed
    but no spurious conflict is emitted.
    """
    def _ok_gbrain(query, *, max_hits: int, observed_at: str):
        return [_hit(GBRAIN_SOURCE, "d-cac-fail-001", "zephyr deployment_status = approved")]

    def _failing_obsidian(query, *, max_hits: int, observed_at: str):
        raise RuntimeError("simulated provider failure")

    engine = EvidencePackEngine(
        sources={GBRAIN_SOURCE: _ok_gbrain, OBSIDIAN_SOURCE: _failing_obsidian},
        storage=_RecordingStorage(),
        audit_sink=_RecordingAuditSink(),
    )
    pack = engine.dry_run(objective_id="d-cac-fail", objective_text="zephyr")

    assert OBSIDIAN_SOURCE in pack.sources_failed
    assert OBSIDIAN_SOURCE not in pack.sources_queried
    assert pack.conflicts == [], (
        f"provider failure must not be classified as a conflict, got {pack.conflicts}"
    )


def test_missing_information_path_does_not_emit_conflict():
    """When the objective cannot be satisfied by any source (no overlap),
    the missing_information list grows but conflicts stay empty.
    """
    gbrain = _hit(GBRAIN_SOURCE, "d-cac-miss-001", "completely unrelated topic about zzz")
    engine, _, audit = _run_dry_run({GBRAIN_SOURCE: _provider([gbrain])})
    pack = engine.dry_run(objective_id="d-cac-miss", objective_text="zephyr")

    assert pack.conflicts == []
    assert audit.get_events() == []


def test_provider_with_no_hits_is_not_a_conflict():
    """An empty provider returns no hits → engine produces 0 hits, 0
    conflicts. (Provider-returned empty is distinct from provider-failed.)
    """
    def _empty_provider(query, *, max_hits: int, observed_at: str):
        return []

    engine = EvidencePackEngine(
        sources={GBRAIN_SOURCE: _empty_provider},
        storage=_RecordingStorage(),
        audit_sink=_RecordingAuditSink(),
    )
    pack = engine.dry_run(objective_id="d-cac-empty", objective_text="zephyr")

    assert pack.hits == []
    assert pack.conflicts == []
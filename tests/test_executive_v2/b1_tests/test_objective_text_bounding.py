"""Behavioral coverage for the canonical objective_text bound."""
from __future__ import annotations

import json
from typing import Any

from agent.executive.knowledge_discovery import (
    EvidencePackEngine,
    FreshnessPolicy,
    KnowledgeHitV2,
    ProvenanceEnvelope,
)


CANONICAL_OBJECTIVE_TEXT_LIMIT = 10_000
OBSERVED_AT = "2026-07-08T20:00:00+00:00"


class RecordingStorage:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.set_meta_calls = 0

    def get_meta(self, key: str) -> str | None:
        return self.values.get(key)

    def set_meta(self, key: str, value: str) -> None:
        self.set_meta_calls += 1
        self.values[key] = value


class RecordingAuditSink:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event: dict[str, Any]) -> None:
        self.events.append(dict(event))

    def get_events(self) -> list[dict[str, Any]]:
        return list(self.events)


def _hit(source: str, hit_id: str, *, uri: str | None = None) -> KnowledgeHitV2:
    observed = OBSERVED_AT
    return KnowledgeHitV2(
        source=source,
        hit_id=hit_id,
        title=f"{source} title",
        relevance_score=0.9,
        snippet=f"{source} snippet",
        location=uri or f"{source}://{hit_id}",
        fingerprint=f"fingerprint-{source}-{hit_id}",
        created_at=observed,
        provenance=ProvenanceEnvelope(
            producer=f"fake_{source}_provider_v1",
            produced_at=observed,
            source_type=source,
            source_uri=uri or f"{source}://{hit_id}",
            retrieval_mode="metadata_only",
            read_only=True,
        ),
        freshness=FreshnessPolicy(
            observed_at=observed,
            source_updated_at=observed,
            staleness_days=0,
            freshness="current",
            freshness_score=1.0,
        ),
    )


def _recording_provider(calls: list[str], source: str = "gbrain"):
    def provider(query, *, max_hits: int, observed_at: str):
        calls.append(query.objective_text)
        return [_hit(source, f"hit-{len(calls)}")]

    return provider


def test_short_text_is_preserved_and_provider_receives_same_text():
    calls: list[str] = []
    engine = EvidencePackEngine(
        sources={"gbrain": _recording_provider(calls)},
    )

    pack = engine.dry_run("short", "  texto corto  ")

    assert calls == ["  texto corto  "]
    assert len(calls[0]) < CANONICAL_OBJECTIVE_TEXT_LIMIT
    assert pack.query_fingerprint == engine.dry_run("short", "  texto corto  ").query_fingerprint


def test_exact_limit_is_preserved_and_limit_plus_one_is_bounded():
    calls: list[str] = []
    engine = EvidencePackEngine(
        sources={"gbrain": _recording_provider(calls)},
    )

    exact = "é" * CANONICAL_OBJECTIVE_TEXT_LIMIT
    over = exact + "SUFFIX"
    engine.dry_run("exact", exact)
    engine.dry_run("over", over)

    assert calls[0] == exact
    assert calls[1] == exact
    assert len(calls[1]) == CANONICAL_OBJECTIVE_TEXT_LIMIT
    assert "SUFFIX" not in calls[1]


def test_empty_text_follows_existing_empty_normalization():
    calls: list[str] = []
    engine = EvidencePackEngine(
        sources={"gbrain": _recording_provider(calls)},
    )

    engine.dry_run("empty", "")

    assert calls == [""]


def test_unicode_bound_is_by_characters_not_utf8_bytes():
    calls: list[str] = []
    engine = EvidencePackEngine(
        sources={"gbrain": _recording_provider(calls)},
    )
    text = "🙂" * (CANONICAL_OBJECTIVE_TEXT_LIMIT + 1)

    engine.dry_run("unicode", text)

    assert len(calls[0]) == CANONICAL_OBJECTIVE_TEXT_LIMIT
    assert calls[0] == text[:CANONICAL_OBJECTIVE_TEXT_LIMIT]
    assert len(calls[0].encode("utf-8")) > CANONICAL_OBJECTIVE_TEXT_LIMIT


def test_fingerprint_uses_effective_text():
    calls: list[str] = []
    engine = EvidencePackEngine(
        sources={"gbrain": _recording_provider(calls)},
    )
    prefix = "p" * CANONICAL_OBJECTIVE_TEXT_LIMIT

    pack_a = engine.dry_run("same", prefix + "-suffix-a")
    pack_b = engine.dry_run("same", prefix + "-suffix-b")
    pack_inside = engine.dry_run("same", prefix[:-1] + "q")

    assert pack_a.query_fingerprint == pack_b.query_fingerprint
    assert pack_a.query_fingerprint != pack_inside.query_fingerprint


def test_discover_reuses_cache_for_different_discarded_suffix():
    calls: list[str] = []
    storage = RecordingStorage()
    audit = RecordingAuditSink()
    engine = EvidencePackEngine(
        sources={"gbrain": _recording_provider(calls)},
        storage=storage,
        audit_sink=audit,
    )
    prefix = "objective " + "x" * (CANONICAL_OBJECTIVE_TEXT_LIMIT - len("objective "))

    first = engine.discover("idem", prefix + "-first-suffix")
    first_events = len(audit.get_events())
    first_writes = storage.set_meta_calls
    second = engine.discover("idem", prefix + "-second-suffix")

    assert calls == [prefix]
    assert first.is_idempotent_reuse is False
    assert second.is_idempotent_reuse is True
    assert len(audit.get_events()) == first_events
    assert storage.set_meta_calls == first_writes == 1
    assert second.query_fingerprint == first.query_fingerprint
    metadata = json.loads(storage.values["objective_knowledge_discovery:idem:v2"])
    serialized_metadata = json.dumps(metadata)
    assert "first-suffix" not in serialized_metadata
    assert "second-suffix" not in serialized_metadata


def test_audit_events_do_not_include_discarded_objective_suffix():
    calls: list[str] = []
    audit = RecordingAuditSink()
    engine = EvidencePackEngine(
        sources={
            "policy": _recording_provider(calls, "policy"),
            "obsidian": _recording_provider(calls, "obsidian"),
        },
        audit_sink=audit,
    )
    prefix = "q" * CANONICAL_OBJECTIVE_TEXT_LIMIT

    engine.dry_run("audit", prefix + "-discarded")

    assert calls == [prefix, prefix]
    serialized_events = json.dumps(audit.get_events(), ensure_ascii=False)
    assert "discarded" not in serialized_events
    assert prefix not in serialized_events

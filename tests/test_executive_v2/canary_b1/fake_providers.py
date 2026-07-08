"""Fake knowledge sources for the B-1 Knowledge Discovery canary.

Five in-memory sources (policy, contract, report, gbrain, obsidian) that
return canned `KnowledgeHitV2` objects with provenance + freshness populated.

**HERMETICITY GUARANTEES** (enforced via AST scan in
`test_fake_providers_no_forbidden_imports`):

* No subprocess (no `os.system`, `subprocess.run`, `subprocess.Popen`).
* No network (no `urllib`, `requests`, `httpx`, `socket`, `ssl`, `aiohttp`).
* No LLM provider imports (`anthropic`, `openai`, `litellm`, `ollama`).
* No EIL imports (`agent.executive_integration`).
* No `gbrain`, `obsidian`, `notebooklm` CLIs.
* No `agent.executive.objective_engine` (caller; canary never invokes it).
* `FakeReportSource` reads ONLY `tmp_path` markdown files; it never walks
  the real reports dir, real Obsidian vault, or any user path.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from tests.test_executive_v2.canary_b1.evidence_pack import (
    _make_hit_v2,
    _tokenize,
    _clamp,
    _iso_mtime,
    KnowledgeHitV2,
    SOURCE_TTL_DAYS,
)


# ─────────────────────────────────────────────────────────────────────
# Producer name registry (mirrors fake_provider_contract.md §2)
# ─────────────────────────────────────────────────────────────────────


PRODUCER_NAME = {
    "policy": "fake_policy_provider_v1",
    "contract": "fake_contract_provider_v1",
    "report": "fake_report_provider_v1",
    "gbrain": "fake_gbrain_provider_v1",
    "obsidian": "fake_obsidian_provider_v1",
}


# ─────────────────────────────────────────────────────────────────────
# Specs (canned response shape)
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FakeProviderSpec:
    """Canned response shape for a fake source.

    `hits` is a tuple of dicts. Each dict MAY contain:
        hit_id, title, snippet, relevance_score, source_updated_at,
        quote, line_range, hash_sha256, created_at, location.
    `query_exception`: if set, the provider raises this on query().
    `is_available`: if False, the provider returns [] (not registered).
    """

    name: str
    hits: tuple[dict, ...] = field(default_factory=tuple)
    is_available: bool = True
    query_exception: Optional[BaseException] = None


# ─────────────────────────────────────────────────────────────────────
# Default fixture specs
# ─────────────────────────────────────────────────────────────────────


def default_gbrain_spec() -> FakeProviderSpec:
    return FakeProviderSpec(
        name="gbrain",
        hits=(
            {
                "hit_id": "fake-gb-entity-001",
                "title": "GBrain entity: Knowledge Discovery v0.1 (canary fixture)",
                "relevance_score": 0.85,
                "snippet": "Knowledge Discovery canary hermetic fixture",
                "source_updated_at": "2026-07-07T20:00:00+00:00",  # 1 day stale
            },
            {
                "hit_id": "fake-gb-entity-002",
                "title": "GBrain entity: Promotion gate criteria (canary fixture)",
                "relevance_score": 0.72,
                "snippet": "Promotion gate criteria for B-1 closure",
                "source_updated_at": "2026-06-11T20:00:00+00:00",  # 27 days stale
            },
            {
                "hit_id": "fake-gb-entity-003",
                "title": "GBrain entity: Hermetic test pattern (canary fixture)",
                "relevance_score": 0.60,
                "snippet": "Hermetic test pattern with fake providers",
                "source_updated_at": None,
            },
        ),
    )


def default_obsidian_spec() -> FakeProviderSpec:
    return FakeProviderSpec(
        name="obsidian",
        hits=(
            {
                "hit_id": "Diario/2026-06-01.md",
                "title": "Diario 2026-06-01 (canary fixture)",
                "relevance_score": 0.78,
                "snippet": "Knowledge discovery notes from yesterday",
                "source_updated_at": "2026-06-01T20:00:00+00:00",  # 37 days stale
            },
            {
                "hit_id": "Sistema/Asistente/architecture.md",
                "title": "Architecture notes (canary fixture)",
                "relevance_score": 0.70,
                "snippet": "Architecture for knowledge discovery",
                "source_updated_at": "2026-06-11T20:00:00+00:00",  # 27 days stale
            },
            {
                "hit_id": "Sistema/Asistente/index.md",
                "title": "Index (canary fixture)",
                "relevance_score": 0.55,
                "snippet": "Index of system assistant files",
                "source_updated_at": "2026-06-22T20:00:00+00:00",  # 16 days stale
            },
            {
                "hit_id": "Diario/2026-07-08.md",
                "title": "Diario 2026-07-08 (canary fixture)",
                "relevance_score": 0.65,
                "snippet": "Today's notes on knowledge discovery",
                "source_updated_at": "2026-07-08T20:00:00+00:00",  # current
            },
        ),
    )


def default_reports_spec(tmp_path: Path) -> FakeProviderSpec:
    target = tmp_path / "matching_report.md"
    target.write_text(
        "# Matching Report\n\nKnowledge discovery canary hermetic fixture report.\n",
        encoding="utf-8",
    )
    _pin_mtime(target, "2026-07-07T20:00:00+00:00")
    unrelated = tmp_path / "unrelated_report.md"
    unrelated.write_text(
        "# Unrelated\n\nCats and dogs.\n",
        encoding="utf-8",
    )
    _pin_mtime(unrelated, "2026-07-08T20:00:00+00:00")
    return FakeProviderSpec(
        name="report",
        hits=(
            {
                "path": target,
                "title": "matching_report.md",
                "relevance_score": 0.80,
                "snippet": "Knowledge discovery canary hermetic fixture report",
                "source_updated_at": "2026-07-07T20:00:00+00:00",
            },
        ),
        is_available=True,
    )


def empty_spec(name: str) -> FakeProviderSpec:
    return FakeProviderSpec(name=name, hits=(), is_available=True)


def unavailable_spec(name: str) -> FakeProviderSpec:
    return FakeProviderSpec(name=name, hits=(), is_available=False)


def failing_spec(name: str, exc: BaseException) -> FakeProviderSpec:
    return FakeProviderSpec(name=name, hits=(), is_available=True, query_exception=exc)


def _pin_mtime(path: Path, iso_utc: str) -> None:
    dt = _dt.datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
    ts = dt.timestamp()
    import os
    os.utime(str(path), (ts, ts))


# ─────────────────────────────────────────────────────────────────────
# Provider callables (consumed by EvidencePackEngine)
# ─────────────────────────────────────────────────────────────────────


def gbrain_provider(
    spec: FakeProviderSpec,
) -> Callable[..., list[KnowledgeHitV2]]:
    def _query(query, *, max_hits: int = 5, observed_at: str) -> list[KnowledgeHitV2]:
        if spec.query_exception is not None:
            raise spec.query_exception
        if not spec.is_available or not spec.hits:
            return []
        q_tokens = _tokenize(query.objective_text)
        if not q_tokens:
            return []
        out: list[KnowledgeHitV2] = []
        for h in spec.hits:
            overlap = q_tokens & _tokenize(
                h.get("snippet", "") + " " + h.get("title", "")
            )
            if not overlap:
                continue
            score = float(h.get("relevance_score", 0.5))
            snippet = h.get("snippet", "")
            out.append(_make_hit_v2(
                source="gbrain",
                hit_id=h["hit_id"],
                title=h.get("title", ""),
                relevance_score=score,
                snippet=snippet,
                source_uri=f"gbrain://entity/{h['hit_id']}",
                source_updated_at=h.get("source_updated_at"),
                retrieval_mode="semantic_search",
                quote=snippet[:200] or None,
                observed_at=observed_at,
                ttl_days=SOURCE_TTL_DAYS["gbrain"],
                created_at=h.get("source_updated_at") or observed_at,
                location=f"gbrain://entity/{h['hit_id']}",
            ))
        out.sort(key=lambda x: -x.relevance_score)
        return out[:max_hits]
    return _query


def obsidian_provider(
    spec: FakeProviderSpec,
) -> Callable[..., list[KnowledgeHitV2]]:
    def _query(query, *, max_hits: int = 5, observed_at: str) -> list[KnowledgeHitV2]:
        if not spec.is_available or not spec.hits:
            return []
        q_tokens = _tokenize(query.objective_text)
        if not q_tokens:
            return []
        out: list[KnowledgeHitV2] = []
        for h in spec.hits:
            overlap = q_tokens & _tokenize(
                h.get("snippet", "") + " " + h.get("title", "")
            )
            if not overlap:
                continue
            score = float(h.get("relevance_score", 0.5))
            snippet = h.get("snippet", "")
            out.append(_make_hit_v2(
                source="obsidian",
                hit_id=h["hit_id"],
                title=h.get("title", ""),
                relevance_score=score,
                snippet=snippet,
                source_uri=f"file://obsidian/{h['hit_id']}",
                source_updated_at=h.get("source_updated_at"),
                retrieval_mode="snippet",
                quote=snippet[:200] or None,
                line_range="1-50",
                hash_sha256=hashlib.sha256(snippet.encode("utf-8")).hexdigest(),
                observed_at=observed_at,
                ttl_days=SOURCE_TTL_DAYS["obsidian"],
                created_at=h.get("source_updated_at") or observed_at,
                location=f"file://obsidian/{h['hit_id']}",
            ))
        out.sort(key=lambda x: -x.relevance_score)
        return out[:max_hits]
    return _query


def report_provider(
    spec: FakeProviderSpec,
) -> Callable[..., list[KnowledgeHitV2]]:
    def _query(query, *, max_hits: int = 5, observed_at: str) -> list[KnowledgeHitV2]:
        if not spec.is_available:
            return []
        q_tokens = _tokenize(query.objective_text)
        if not q_tokens:
            return []
        out: list[KnowledgeHitV2] = []
        for h in spec.hits:
            path = h.get("path")
            if path is None:
                # In-memory canned hit with no file backing
                snippet = h.get("snippet", "")
                overlap = q_tokens & _tokenize(snippet + " " + h.get("title", ""))
                if not overlap:
                    continue
                score = float(h.get("relevance_score", 0.5))
                out.append(_make_hit_v2(
                    source="report",
                    hit_id=h.get("hit_id", "report-canned"),
                    title=h.get("title", ""),
                    relevance_score=score,
                    snippet=snippet,
                    source_uri=h.get("source_uri", "report://canned"),
                    source_updated_at=h.get("source_updated_at"),
                    retrieval_mode="full_document",
                    quote=snippet[:200] or None,
                    line_range="1-1",
                    hash_sha256=h.get("hash_sha256"),
                    observed_at=observed_at,
                    ttl_days=SOURCE_TTL_DAYS["report"],
                    created_at=h.get("source_updated_at") or observed_at,
                    location=h.get("source_uri", "report://canned"),
                ))
                continue

            try:
                content = Path(path).read_text(encoding="utf-8", errors="ignore")
            except (OSError, IOError):
                continue
            overlap = q_tokens & _tokenize(content)
            if not overlap:
                continue
            score = _clamp(len(overlap) / max(len(q_tokens), 1))
            snippet = " ".join(content.split())[:200]
            out.append(_make_hit_v2(
                source="report",
                hit_id=str(path),
                title=Path(path).name,
                relevance_score=score,
                snippet=snippet,
                source_uri=str(path),
                source_updated_at=_iso_mtime(path),
                retrieval_mode="full_document",
                quote=snippet[:200] or None,
                line_range="1-1",
                hash_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                observed_at=observed_at,
                ttl_days=SOURCE_TTL_DAYS["report"],
                created_at=_iso_mtime(path),
                location=str(path),
            ))
        out.sort(key=lambda x: -x.relevance_score)
        return out[:max_hits]
    return _query


def policy_provider(
    spec: FakeProviderSpec,
) -> Callable[..., list[KnowledgeHitV2]]:
    """Pure in-memory policy decision provider (no real state.db)."""
    def _query(query, *, max_hits: int = 5, observed_at: str) -> list[KnowledgeHitV2]:
        if not spec.is_available or not spec.hits:
            return []
        q_tokens = _tokenize(query.objective_text)
        if not q_tokens:
            return []
        out: list[KnowledgeHitV2] = []
        for h in spec.hits:
            decision_text = " ".join([
                str(h.get("goal_class", "")),
                " ".join(map(str, h.get("warnings", ()))),
                str(h.get("decision_fingerprint", "")),
            ]).lower()
            overlap = q_tokens & _tokenize(decision_text)
            if not overlap:
                continue
            score = _clamp(len(overlap) / max(len(q_tokens), 1))
            warnings = h.get("warnings", ())
            snippet = (
                " ".join(warnings[:3])[:200]
                if warnings else
                f"risk_level={h.get('risk_level', 'medium')}"
            )
            out.append(_make_hit_v2(
                source="policy",
                hit_id=h.get("hit_id", "policy-canned"),
                title=h.get("title", f"Policy decision (canary fixture)"),
                relevance_score=score,
                snippet=snippet,
                source_uri=f"state_meta[objective_policy_decision:{h.get('hit_id', 'canned')}]",
                source_updated_at=h.get("source_updated_at"),
                retrieval_mode="metadata_only",
                observed_at=observed_at,
                ttl_days=SOURCE_TTL_DAYS["policy"],
                created_at=h.get("source_updated_at"),
                location=f"state_meta[objective_policy_decision:{h.get('hit_id', 'canned')}]",
            ))
        out.sort(key=lambda x: -x.relevance_score)
        return out[:max_hits]
    return _query


def contract_provider(
    spec: FakeProviderSpec,
) -> Callable[..., list[KnowledgeHitV2]]:
    """Pure in-memory execution contract provider (no real state.db)."""
    def _query(query, *, max_hits: int = 5, observed_at: str) -> list[KnowledgeHitV2]:
        if not spec.is_available or not spec.hits:
            return []
        q_tokens = _tokenize(query.objective_text)
        if not q_tokens:
            return []
        out: list[KnowledgeHitV2] = []
        for h in spec.hits:
            success = h.get("success_criteria", []) or []
            success_text = " ".join(str(x) for x in success)
            contract_text = " ".join([
                str(h.get("risk_score", "")),
                " ".join(str(x) for x in (h.get("hard_constraints", []) or [])),
                " ".join(str(x) for x in (h.get("soft_constraints", []) or [])),
                success_text,
            ]).lower()
            overlap = q_tokens & _tokenize(contract_text)
            if not overlap:
                continue
            score = _clamp(len(overlap) / max(len(q_tokens), 1))
            snippet = (
                f"risk_score={float(h.get('risk_score', 0.0)):.3f}; "
                f"criteria={success_text[:120]}"
            )[:200]
            out.append(_make_hit_v2(
                source="contract",
                hit_id=h.get("hit_id", "contract-canned"),
                title=h.get("title", "Execution Contract (canary fixture)"),
                relevance_score=score,
                snippet=snippet,
                source_uri=f"state_meta[objective:<{h.get('hit_id', 'canned')}>].contract",
                source_updated_at=h.get("source_updated_at"),
                retrieval_mode="metadata_only",
                observed_at=observed_at,
                ttl_days=SOURCE_TTL_DAYS["contract"],
                created_at=h.get("source_updated_at"),
                location=f"state_meta[objective:<{h.get('hit_id', 'canned')}>].contract",
            ))
        out.sort(key=lambda x: -x.relevance_score)
        return out[:max_hits]
    return _query


# ─────────────────────────────────────────────────────────────────────
# Bundle factory
# ─────────────────────────────────────────────────────────────────────


def make_provider_bundle(
    *,
    gbrain_spec: Optional[FakeProviderSpec] = None,
    obsidian_spec: Optional[FakeProviderSpec] = None,
    report_spec: Optional[FakeProviderSpec] = None,
    policy_spec: Optional[FakeProviderSpec] = None,
    contract_spec: Optional[FakeProviderSpec] = None,
) -> dict[str, Callable[..., list[KnowledgeHitV2]]]:
    """Return a dict of source_name → provider_callable.

    The engine consumes this dict directly. Sources with `is_available=False`
    are still registered but will be excluded by the engine if
    `sources_requested` filters them; otherwise they yield empty hits.
    """
    bundle: dict[str, Callable[..., list[KnowledgeHitV2]]] = {}
    if gbrain_spec is not None:
        bundle["gbrain"] = gbrain_provider(gbrain_spec)
    if obsidian_spec is not None:
        bundle["obsidian"] = obsidian_provider(obsidian_spec)
    if report_spec is not None:
        bundle["report"] = report_provider(report_spec)
    if policy_spec is not None:
        bundle["policy"] = policy_provider(policy_spec)
    if contract_spec is not None:
        bundle["contract"] = contract_provider(contract_spec)
    return bundle

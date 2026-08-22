"""Phase 1.5 Knowledge Discovery — engine facade.

Consumes the three immediate knowledge sources (designed in
``hermes_knowledge_discovery_executive_design_readonly``):

* ``policy``   — ``state_meta[objective_policy_decision:*]``
* ``report``   — ``~/.hermes/reports/**/*.md``  (read-only)
* ``contract`` — ``state_meta[objective:*].contract``

The deferred sources (GBrain, Obsidian, KnowledgeGraph, Claims,
Evidence, NotebookLM) are NOT consulted by this module. They will
require separate future designs.

Forbidden APIs (PROHIBITED in this module):

* Any write to ``~/.hermes/reports/``
* Any write to GBrain / Obsidian / NotebookLM
* Any network / urllib / httpx / requests / aiohttp
* Any subprocess / os.system / subprocess.run / subprocess.Popen
* Any provider / anthropic / openai / litellm / auxiliary_client
* Any EIL / ExecutiveLauncher / ExecutiveIntegrationRouter
* Any execution of ObjectiveEngine / Planner / Policy / Kanban /
  Worker (this module is pure compute + a single state_meta write)

The engine has three modes:

* ``dry_run``      — pure compute; produces a KnowledgeDiscoveryReport
                     with no state_meta writes.
* ``discover``     — re-uses a persisted report if its
                     ``query_fingerprint`` matches (idempotent);
                     otherwise builds and persists.
* ``rollback``     — best-effort, idempotent state_meta cleanup of the
                     single ``objective_knowledge_discovery:<oid>`` key.

All side effects are scoped to a single state_meta key per objective.
The component is hermetic and deterministic.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol

from .state_storage import ObjectiveStateStorage

log = logging.getLogger(__name__)


SCHEMA_VERSION = "knowledge_discovery.v1"


# ── Data classes ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class KnowledgeHit:
    """A single hit returned by a KnowledgeProvider."""

    source: str                       # "policy" | "report" | "contract"
    hit_id: str                       # objective_id (policies/contracts)
                                     # or report path (reports)
    title: str
    relevance_score: float            # [0.0, 1.0]
    snippet: str                      # ≤ 200 chars, whitespace-collapsed
    location: str                     # state_meta key or filesystem path
    fingerprint: str                  # sha256 of (source, hit_id, snippet)
    created_at: str                   # ISO 8601

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "hit_id": self.hit_id,
            "title": self.title,
            "relevance_score": float(self.relevance_score),
            "snippet": self.snippet,
            "location": self.location,
            "fingerprint": self.fingerprint,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeHit":
        return cls(
            source=str(data.get("source", "")),
            hit_id=str(data.get("hit_id", "")),
            title=str(data.get("title", "")),
            relevance_score=float(data.get("relevance_score", 0.0) or 0.0),
            snippet=str(data.get("snippet", "")),
            location=str(data.get("location", "")),
            fingerprint=str(data.get("fingerprint", "")),
            created_at=str(data.get("created_at", "")),
        )


@dataclass(frozen=True)
class KnowledgeQuery:
    """Input contract for Knowledge Discovery queries."""

    objective_id: str
    objective_text: str
    goal_class: str
    risk_profile: str
    complexity: str
    max_hits_per_source: int
    timeout_seconds: float
    sources_requested: tuple[str, ...]

    def fingerprint(self) -> str:
        """sha256 of canonical inputs (idempotency key)."""
        canonical = json.dumps(
            {
                "objective_id": self.objective_id,
                "objective_text": self.objective_text,
                "goal_class": self.goal_class,
                "risk_profile": self.risk_profile,
                "complexity": self.complexity,
                "max_hits_per_source": int(self.max_hits_per_source),
                "timeout_seconds": float(self.timeout_seconds),
                "sources_requested": sorted(self.sources_requested),
                "schema_version": SCHEMA_VERSION,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class KnowledgeDiscoveryReport:
    """Output report of Knowledge Discovery."""

    objective_id: str
    knowledge_query_fingerprint: str
    sources_queried: tuple[str, ...]
    sources_failed: tuple[str, ...]
    hits_by_source: tuple[KnowledgeHit, ...]
    summary_text: str
    summary_fingerprint: str
    factual_grounding_score: float   # [0.0, 1.0]
    total_hits: int
    duration_ms: int
    created_at: str
    created_by: str
    schema_version: str = SCHEMA_VERSION
    is_idempotent_reuse: bool = False

    def to_dict(self) -> dict:
        return {
            "objective_id": self.objective_id,
            "knowledge_query_fingerprint": self.knowledge_query_fingerprint,
            "sources_queried": list(self.sources_queried),
            "sources_failed": list(self.sources_failed),
            "hits_by_source": [h.to_dict() for h in self.hits_by_source],
            "summary_text": self.summary_text,
            "summary_fingerprint": self.summary_fingerprint,
            "factual_grounding_score": float(self.factual_grounding_score),
            "total_hits": int(self.total_hits),
            "duration_ms": int(self.duration_ms),
            "created_at": self.created_at,
            "created_by": self.created_by,
            "schema_version": self.schema_version,
            "is_idempotent_reuse": bool(self.is_idempotent_reuse),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeDiscoveryReport":
        return cls(
            objective_id=str(data.get("objective_id", "")),
            knowledge_query_fingerprint=str(
                data.get("knowledge_query_fingerprint", "")
            ),
            sources_queried=tuple(data.get("sources_queried") or ()),
            sources_failed=tuple(data.get("sources_failed") or ()),
            hits_by_source=tuple(
                KnowledgeHit.from_dict(h)
                for h in (data.get("hits_by_source") or [])
            ),
            summary_text=str(data.get("summary_text", "")),
            summary_fingerprint=str(data.get("summary_fingerprint", "")),
            factual_grounding_score=float(
                data.get("factual_grounding_score", 0.0) or 0.0
            ),
            total_hits=int(data.get("total_hits", 0) or 0),
            duration_ms=int(data.get("duration_ms", 0) or 0),
            created_at=str(data.get("created_at", "")),
            created_by=str(data.get("created_by", "")),
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            is_idempotent_reuse=bool(data.get("is_idempotent_reuse", False)),
        )


# ── Provider protocol ───────────────────────────────────────────────


class KnowledgeProvider(Protocol):
    """Read-only knowledge provider interface.

    Future adapters (GBrain, Obsidian, KnowledgeGraph, etc.)
    implement this Protocol. The 3 built-in providers below are
    implemented directly (not as adapters) because they read from
    state.db and the local filesystem, both of which are in-process.
    """

    name: str

    def query(
        self,
        query: KnowledgeQuery,
        *,
        max_hits: int = 5,
    ) -> list[KnowledgeHit]: ...

    def is_available(self) -> bool: ...


# ── Helpers ──────────────────────────────────────────────────────────


def _hit_fingerprint(source: str, hit_id: str, snippet: str) -> str:
    canonical = json.dumps(
        {"source": source, "hit_id": hit_id, "snippet": snippet},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _iso_mtime(path: Path) -> str:
    import datetime as _dt

    ts = path.stat().st_mtime
    return _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).isoformat()


def _now_iso8601() -> str:
    import datetime as _dt

    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _tokenize(text: str) -> set[str]:
    """Whitespace split + lowercase; ignore very short tokens."""
    out: set[str] = set()
    for tok in (text or "").lower().split():
        if len(tok) >= 3:
            out.add(tok)
    return out


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


# ── Built-in providers (3 immediate sources) ─────────────────────────


class _PolicyProvider:
    """Reads state_meta[objective_policy_decision:*]."""

    name = "policy"

    def __init__(self, storage: ObjectiveStateStorage) -> None:
        self._storage = storage

    def is_available(self) -> bool:
        return True

    def query(
        self,
        query: KnowledgeQuery,
        *,
        max_hits: int = 5,
    ) -> list[KnowledgeHit]:
        query_tokens = _tokenize(query.objective_text)
        if not query_tokens:
            return []
        hits: list[KnowledgeHit] = []
        # Iterate active objective IDs via list_active(); filter
        # the policy decision per id. list_active returns ids whose
        # state_meta[objective:<id>] exists; we then check for the
        # policy decision key.
        for oid in self._storage.list_active():
            if oid == query.objective_id:
                continue
            decision = self._storage.get_objective_policy_decision(oid)
            if decision is None:
                continue
            decision_text = " ".join(
                [
                    str(getattr(decision, "goal_class", "") or ""),
                    " ".join(map(str, decision.warnings or ())),
                    str(decision.decision_fingerprint),
                ]
            ).lower()
            decision_tokens = _tokenize(decision_text)
            overlap = query_tokens & decision_tokens
            if not overlap:
                continue
            score = _clamp(len(overlap) / max(len(query_tokens), 1))
            snippet = " ".join(decision.warnings[:3])[:200] if decision.warnings else (
                f"risk_level={decision.risk_level.name if hasattr(decision.risk_level, 'name') else decision.risk_level}"
            )
            hits.append(
                KnowledgeHit(
                    source="policy",
                    hit_id=oid,
                    title=(
                        f"Policy decision {oid[:12]} "
                        f"(risk={decision.risk_level.name if hasattr(decision.risk_level, 'name') else decision.risk_level})"
                    ),
                    relevance_score=score,
                    snippet=snippet,
                    location=f"state_meta[objective_policy_decision:{oid}]",
                    fingerprint=_hit_fingerprint("policy", oid, snippet),
                    created_at=str(decision.created_at),
                )
            )
        hits.sort(key=lambda h: -h.relevance_score)
        return hits[:max_hits]


class _ReportProvider:
    """Reads ~/.hermes/reports/**/*.md (read-only)."""

    name = "report"

    def __init__(self, reports_root: Optional[Path] = None) -> None:
        self._root = reports_root or Path(
            os.environ.get("HERMES_REPORTS_ROOT", str(Path.home() / ".hermes" / "reports"))
        )

    def is_available(self) -> bool:
        return self._root.exists() and self._root.is_dir()

    def query(
        self,
        query: KnowledgeQuery,
        *,
        max_hits: int = 5,
    ) -> list[KnowledgeHit]:
        query_tokens = _tokenize(query.objective_text)
        if not query_tokens:
            return []
        if not self.is_available():
            return []
        hits: list[KnowledgeHit] = []
        for md_path in self._root.glob("**/*.md"):
            try:
                content = md_path.read_text(encoding="utf-8", errors="ignore")
            except (OSError, IOError):
                continue
            content_tokens = _tokenize(content)
            overlap = query_tokens & content_tokens
            if not overlap:
                continue
            score = _clamp(len(overlap) / max(len(query_tokens), 1))
            snippet = " ".join(content.split())[:200]
            hits.append(
                KnowledgeHit(
                    source="report",
                    hit_id=str(md_path),
                    title=md_path.name,
                    relevance_score=score,
                    snippet=snippet,
                    location=str(md_path),
                    fingerprint=_hit_fingerprint("report", str(md_path), snippet),
                    created_at=_iso_mtime(md_path),
                )
            )
        hits.sort(key=lambda h: -h.relevance_score)
        return hits[:max_hits]


class _ContractProvider:
    """Reads state_meta[objective:*].contract for prior objectives."""

    name = "contract"

    def __init__(self, storage: ObjectiveStateStorage) -> None:
        self._storage = storage

    def is_available(self) -> bool:
        return True

    def query(
        self,
        query: KnowledgeQuery,
        *,
        max_hits: int = 5,
    ) -> list[KnowledgeHit]:
        query_tokens = _tokenize(query.objective_text)
        if not query_tokens:
            return []
        hits: list[KnowledgeHit] = []
        for oid in self._storage.list_active():
            if oid == query.objective_id:
                continue
            state = self._storage.load(oid)
            if state is None or not isinstance(state.contract, dict):
                continue
            contract = state.contract
            success_criteria = " ".join(
                str(x) for x in (contract.get("success_criteria") or [])
            )
            risk_components = contract.get("risk_components") or {}
            contract_text = " ".join(
                [
                    str(contract.get("risk_score", "")),
                    " ".join(str(x) for x in (contract.get("hard_constraints") or [])),
                    " ".join(str(x) for x in (contract.get("soft_constraints") or [])),
                    success_criteria,
                    json.dumps(risk_components, sort_keys=True) if risk_components else "",
                ]
            ).lower()
            contract_tokens = _tokenize(contract_text)
            overlap = query_tokens & contract_tokens
            if not overlap:
                continue
            score = _clamp(len(overlap) / max(len(query_tokens), 1))
            snippet = f"risk_score={contract.get('risk_score', 0.0):.3f}; "
            snippet += f"criteria={success_criteria[:120]}"
            hits.append(
                KnowledgeHit(
                    source="contract",
                    hit_id=oid,
                    title=f"Execution Contract {oid[:12]}",
                    relevance_score=score,
                    snippet=snippet[:200],
                    location=f"state_meta[objective:<{oid}>].contract",
                    fingerprint=_hit_fingerprint("contract", oid, snippet[:200]),
                    created_at=str(state.created_at or ""),
                )
            )
        hits.sort(key=lambda h: -h.relevance_score)
        return hits[:max_hits]


# ── Engine facade ─────────────────────────────────────────────────────


class KnowledgeDiscoveryEngine:
    """Read-only knowledge discovery engine.

    Side effects (only):
    - state_meta[objective_knowledge_discovery:<objective_id>] is
      written ONCE per objective, AFTER the report is built.

    Does NOT:
    - Modify any input source.
    - Spawn workers.
    - Make network calls.
    - Invoke Kanban.
    - Activate Executive v2 or EIL.
    - Invoke any LLM, GBrain, Obsidian, NotebookLM, or provider.
    """

    SCHEMA_VERSION = SCHEMA_VERSION

    def __init__(
        self,
        *,
        storage: Optional[ObjectiveStateStorage] = None,
        reports_root: Optional[Path] = None,
        max_hits_per_source: int = 5,
        timeout_seconds: float = 30.0,
        sources: Optional[Iterable[str]] = None,
        created_by: str = "executive_v2_knowledge_discovery",
    ) -> None:
        self._storage = storage or ObjectiveStateStorage()
        self._max_hits_per_source = max(1, int(max_hits_per_source))
        self._timeout_seconds = float(timeout_seconds)
        self._created_by = str(created_by)
        self._providers: list[KnowledgeProvider] = []
        requested = list(sources) if sources is not None else [
            "policy", "report", "contract",
        ]
        for name in requested:
            if name == "policy":
                self._providers.append(_PolicyProvider(self._storage))
            elif name == "report":
                self._providers.append(_ReportProvider(reports_root))
            elif name == "contract":
                self._providers.append(_ContractProvider(self._storage))
            # GBrain, Obsidian, NotebookLM, KG, Claims, Evidence:
            # NOT implemented in this canary. They will require
            # separate future designs and will plug in here when
            # available.

    # ── Query building ─────────────────────────────────────────────

    @staticmethod
    def build_query(
        objective_id: str,
        objective_text: str,
        *,
        goal_class: str = "OTHER",
        risk_profile: str = "low",
        complexity: str = "S",
        max_hits_per_source: int = 5,
        timeout_seconds: float = 30.0,
        sources: Optional[Iterable[str]] = None,
    ) -> KnowledgeQuery:
        sources_requested = tuple(sources) if sources is not None else (
            "policy", "report", "contract",
        )
        return KnowledgeQuery(
            objective_id=objective_id,
            objective_text=objective_text,
            goal_class=goal_class,
            risk_profile=risk_profile,
            complexity=complexity,
            max_hits_per_source=max_hits_per_source,
            timeout_seconds=timeout_seconds,
            sources_requested=sources_requested,
        )

    # ── Mode 1: dry_run (pure) ────────────────────────────────────

    def dry_run(
        self,
        objective_id: str,
        objective_text: str,
        *,
        goal_class: str = "OTHER",
        risk_profile: str = "low",
        complexity: str = "S",
    ) -> KnowledgeDiscoveryReport:
        """Pure compute: build report. NO state_meta writes."""
        import time as _time

        query = self.build_query(
            objective_id,
            objective_text,
            goal_class=goal_class,
            risk_profile=risk_profile,
            complexity=complexity,
            max_hits_per_source=self._max_hits_per_source,
            timeout_seconds=self._timeout_seconds,
            sources=[p.name for p in self._providers],
        )
        start = _time.monotonic()
        all_hits: list[KnowledgeHit] = []
        sources_queried: list[str] = []
        sources_failed: list[str] = []
        for provider in self._providers:
            sources_queried.append(provider.name)
            try:
                if not provider.is_available():
                    log.info("kd: provider %s not available; skipping", provider.name)
                    continue
                hits = provider.query(query, max_hits=self._max_hits_per_source)
                all_hits.extend(hits)
            except Exception as exc:
                log.warning("kd: provider %s failed: %s", provider.name, exc)
                sources_failed.append(provider.name)
        duration_ms = int((_time.monotonic() - start) * 1000)
        report = self._build_report(
            query=query,
            hits=tuple(all_hits),
            sources_queried=tuple(sources_queried),
            sources_failed=tuple(sources_failed),
            duration_ms=duration_ms,
            is_idempotent_reuse=False,
        )
        return report

    # ── Mode 2: discover (idempotent + persist) ────────────────────

    def discover(
        self,
        objective_id: str,
        objective_text: str,
        *,
        goal_class: str = "OTHER",
        risk_profile: str = "low",
        complexity: str = "S",
    ) -> KnowledgeDiscoveryReport:
        """Re-use persisted report if its query_fingerprint matches;
        otherwise build and persist.
        """
        query = self.build_query(
            objective_id,
            objective_text,
            goal_class=goal_class,
            risk_profile=risk_profile,
            complexity=complexity,
            max_hits_per_source=self._max_hits_per_source,
            timeout_seconds=self._timeout_seconds,
            sources=[p.name for p in self._providers],
        )
        persisted = self._storage.get_objective_knowledge_discovery(objective_id)
        if persisted is not None:
            if persisted.knowledge_query_fingerprint == query.fingerprint():
                # Idempotent: return a copy with is_idempotent_reuse=True.
                return KnowledgeDiscoveryReport(
                    objective_id=persisted.objective_id,
                    knowledge_query_fingerprint=persisted.knowledge_query_fingerprint,
                    sources_queried=persisted.sources_queried,
                    sources_failed=persisted.sources_failed,
                    hits_by_source=persisted.hits_by_source,
                    summary_text=persisted.summary_text,
                    summary_fingerprint=persisted.summary_fingerprint,
                    factual_grounding_score=persisted.factual_grounding_score,
                    total_hits=persisted.total_hits,
                    duration_ms=persisted.duration_ms,
                    created_at=persisted.created_at,
                    created_by=persisted.created_by,
                    schema_version=persisted.schema_version,
                    is_idempotent_reuse=True,
                )
        # Not idempotent: build fresh.
        report = self.dry_run(
            objective_id,
            objective_text,
            goal_class=goal_class,
            risk_profile=risk_profile,
            complexity=complexity,
        )
        # Single side effect: persist the report.
        self._storage.set_objective_knowledge_discovery(report)
        return report

    # ── Mode 3: rollback ───────────────────────────────────────────

    def rollback(self, objective_id: str) -> bool:
        """Best-effort, idempotent cleanup of the single state_meta key."""
        return self._storage.delete_objective_knowledge_discovery(objective_id)

    # ── Internal helpers ────────────────────────────────────────────

    def _build_report(
        self,
        *,
        query: KnowledgeQuery,
        hits: tuple[KnowledgeHit, ...],
        sources_queried: tuple[str, ...],
        sources_failed: tuple[str, ...],
        duration_ms: int,
        is_idempotent_reuse: bool,
    ) -> KnowledgeDiscoveryReport:
        # Aggregate scoring.
        total_score = sum(h.relevance_score for h in hits)
        # factual_grounding_score: bounded aggregator.
        # 0.0 if no hits; else min(1.0, total_score / 10).
        factual_grounding = 0.0 if not hits else _clamp(total_score / 10.0)

        # summary_text: bounded human-readable string.
        if hits:
            top = hits[:3]
            summary_parts = [
                f"Found {len(hits)} relevant knowledge hit(s) "
                f"from {len({h.source for h in hits})} source(s)."
            ]
            for h in top:
                summary_parts.append(
                    f"- [{h.source}] {h.title} (score={h.relevance_score:.2f})"
                )
            summary_text = "\n".join(summary_parts)[:2000]
        else:
            summary_text = "(no relevant knowledge found)"

        # summary_fingerprint: sha256 of canonical inputs.
        canonical = json.dumps(
            {
                "objective_id": query.objective_id,
                "sources_queried": sorted(sources_queried),
                "hits_by_source": [
                    {"source": h.source, "fingerprint": h.fingerprint}
                    for h in hits
                ],
                "sources_failed": sorted(sources_failed),
                "schema_version": SCHEMA_VERSION,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        summary_fingerprint = hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()

        return KnowledgeDiscoveryReport(
            objective_id=query.objective_id,
            knowledge_query_fingerprint=query.fingerprint(),
            sources_queried=sources_queried,
            sources_failed=sources_failed,
            hits_by_source=hits,
            summary_text=summary_text,
            summary_fingerprint=summary_fingerprint,
            factual_grounding_score=factual_grounding,
            total_hits=len(hits),
            duration_ms=duration_ms,
            created_at=_now_iso8601(),
            created_by=self._created_by,
            schema_version=SCHEMA_VERSION,
            is_idempotent_reuse=is_idempotent_reuse,
        )


# ── Module-level helpers ────────────────────────────────────────────


def knowledge_discovery_dry_run(
    objective_id: str,
    objective_text: str,
    *,
    storage: Optional[ObjectiveStateStorage] = None,
    reports_root: Optional[Path] = None,
    goal_class: str = "OTHER",
    risk_profile: str = "low",
    complexity: str = "S",
    max_hits_per_source: int = 5,
    timeout_seconds: float = 30.0,
) -> KnowledgeDiscoveryReport:
    """Pure: build report. NO state_meta writes."""
    eng = KnowledgeDiscoveryEngine(
        storage=storage,
        reports_root=reports_root,
        max_hits_per_source=max_hits_per_source,
        timeout_seconds=timeout_seconds,
    )
    return eng.dry_run(
        objective_id,
        objective_text,
        goal_class=goal_class,
        risk_profile=risk_profile,
        complexity=complexity,
    )


def knowledge_discovery_discover(
    objective_id: str,
    objective_text: str,
    *,
    storage: Optional[ObjectiveStateStorage] = None,
    reports_root: Optional[Path] = None,
    goal_class: str = "OTHER",
    risk_profile: str = "low",
    complexity: str = "S",
    max_hits_per_source: int = 5,
    timeout_seconds: float = 30.0,
) -> KnowledgeDiscoveryReport:
    """Re-use persisted if query_fingerprint matches; else build + persist."""
    eng = KnowledgeDiscoveryEngine(
        storage=storage,
        reports_root=reports_root,
        max_hits_per_source=max_hits_per_source,
        timeout_seconds=timeout_seconds,
    )
    return eng.discover(
        objective_id,
        objective_text,
        goal_class=goal_class,
        risk_profile=risk_profile,
        complexity=complexity,
    )


def knowledge_discovery_rollback(
    objective_id: str,
    *,
    storage: Optional[ObjectiveStateStorage] = None,
) -> bool:
    """Best-effort, idempotent cleanup."""
    eng = KnowledgeDiscoveryEngine(storage=storage)
    return eng.rollback(objective_id)


__all__ = [
    "SCHEMA_VERSION",
    "KnowledgeHit",
    "KnowledgeQuery",
    "KnowledgeDiscoveryReport",
    "KnowledgeProvider",
    "KnowledgeDiscoveryEngine",
    "knowledge_discovery_dry_run",
    "knowledge_discovery_discover",
    "knowledge_discovery_rollback",
]
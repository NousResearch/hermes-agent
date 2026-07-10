"""Phase 5 — AdrProvider (Layer 4 decision memory).

This module is the SOLE owner of ADR storage resolution. No ADR path is
hardcoded anywhere else in the codebase; every filesystem location derives
from :meth:`AdrProvider._project_dir` / :meth:`AdrProvider._adr_path`.

Design (docs/memory/memory-architecture.md §17):
- Markdown is the source of truth. Each ADR is one markdown file under
  ``<hermes_home>/memory/adr/<project-key>/NNN-title.md`` (global/system
  ADRs live under ``_system/``).
- SQLite remains a derived index. ADR markdown is auto-indexed by the
  existing MemoryIndex because it lives under ``memory/`` — no indexer
  change required.
- No LLM extraction, no automatic acceptance, no embeddings, no
  Graphiti/Holographic.

Trust boundary (§17.5):
- Hermes MAY draft (status=proposed) autonomously -> non-authoritative
  suggestion artifact.
- Only a HUMAN approval flips status to accepted|deprecated|superseded.
  :meth:`accept` requires an ``approved_by`` identity; it writes
  provenance (approved_by/approved_at) and the supersession back-links.
- :meth:`decision` (read) returns ONLY accepted ADRs. A proposed draft is
  NEVER returned as a decision. This is the testable trust boundary.

Write provenance (per directive):
- created_by / created_at  -> set on draft()
- approved_by / approved_at -> set on accept()
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from hermes_cli.memory_api.errors import CapabilityError
from hermes_cli.memory_api.protocols import (
    CapabilityStatus,
    DecisionRecord,
    MemoryProvider,
    MemoryResult,
    SearchResultLike,
)
from hermes_cli.memory_index.indexer import MemoryIndex

_ADR_DIRNAME = "adr"
_SYSTEM_PROJECT = "_system"
_STATUS_ACCEPTED = "accepted"
_STATUS_PROPOSED = "proposed"
_VALID_STATUSES = {"proposed", "accepted", "deprecated", "superseded"}
_FRONTMATTER_KEYS = (
    "id", "title", "status", "date", "project", "decision_maker",
    "proposed_by", "related_components", "supersedes", "superseded_by",
    "tags", "created_by", "created_at", "approved_by", "approved_at",
)


def _now_iso() -> str:
    # Full precision (microseconds) so same-second acceptances still order
    # deterministically for `recent()`. Human-readable enough in frontmatter.
    return datetime.now(timezone.utc).isoformat()


def _slug(project: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", project.lower()).strip("-") or _SYSTEM_PROJECT


class AdrProvider:
    """Structural MemoryProvider over markdown ADRs (Layer 4)."""

    def __init__(self, hermes_home: Optional[Path] = None) -> None:
        self.hermes_home = Path(hermes_home) if hermes_home else self._default_home()
        self._index = MemoryIndex(hermes_home=self.hermes_home)

    @staticmethod
    def _default_home() -> Path:
        import os

        return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))

    # -- path resolution (THE ONLY place ADR paths are constructed) --------
    @property
    def adr_root(self) -> Path:
        return self.hermes_home / "memory" / _ADR_DIRNAME

    def _project_dir(self, project_key: str) -> Path:
        return self.adr_root / _slug(project_key or _SYSTEM_PROJECT)

    def _adr_path(self, project_key: str, number: int, title: str) -> Path:
        slug = _slug(title)
        return self._project_dir(project_key) / f"{number:03d}-{slug}.md"

    # -- metadata -------------------------------------------------------
    @property
    def name(self) -> str:
        return "adr"

    @property
    def layers(self) -> list[str]:
        return ["L4"]

    def status(self) -> CapabilityStatus:
        # ADR provider is a real, file-backed source of truth (not a derived
        # cache). It is "available" as long as the home resolves.
        return CapabilityStatus.AVAILABLE

    # -- read operations (DECISION intent) ------------------------------
    def decision(
        self,
        id: Optional[str] = None,
        topic: Optional[str] = None,
        project: Optional[str] = None,
        include_proposed: bool = False,
    ) -> list[DecisionRecord]:
        """Return ADRs. By default ONLY accepted ADRs are returned.

        Acceptable as decisions. Proposed drafts are excluded unless
        ``include_proposed=True`` (used by lifecycle tooling, not by the
        trust-boundary read path).
        """
        recs = self._load_all(project_key=_slug(project) if project else None)
        if id:
            recs = [r for r in recs if r.id == id]
        if topic:
            toks = [t for t in re.findall(r"[A-Za-z0-9_]+", topic.lower())]
            recs = [
                r for r in recs
                if any(t in (r.title + r.context + r.decision).lower() for t in toks)
            ]
        if not include_proposed:
            recs = [r for r in recs if r.status == _STATUS_ACCEPTED]
        return recs

    def get(self, id: str) -> Optional[DecisionRecord]:
        """Exact ADR lookup by global id '<project>/NNN' or 'NNN'."""
        project_key: Optional[str] = None
        num_str: str = id
        if "/" in id:
            project_key, num_str = id.split("/", 1)
        project_key = _slug(project_key) if project_key else None
        recs = self._load_all(project_key=project_key)
        for r in recs:
            if r.id == id or r.id.endswith("/" + num_str) or r.id == num_str:
                return r
        return None

    def search(self, topic: str, *, project: Optional[str] = None) -> list[DecisionRecord]:
        return self.decision(topic=topic, project=project)

    def by_project(self, project: str) -> list[DecisionRecord]:
        return self.decision(project=project)

    def recent(self, *, project: Optional[str] = None, limit: int = 10) -> list[DecisionRecord]:
        recs = self.decision(project=project)
        recs.sort(key=lambda r: r.date or "", reverse=True)
        return recs[:limit]

    # -- structural Protocol compatibility (search/recent/archive) -------
    def search_files(self, query: str, *, limit: int = 10, scope: Optional[str] = None) -> list[SearchResultLike]:
        # So the provider is usable behind the generic MemoryProvider shape.
        # ADRs are primarily surfaced via decision(); this offers keyword
        # retrieval over accepted+proposed titles/bodies via the existing
        # SQLite index (ADRs live under memory/ already).
        return self._index.search(query, limit=limit, scope="L4" if scope is None else scope)

    def recent_files(self, *, limit: int = 10) -> list[SearchResultLike]:
        return self._index.recent(limit=limit)

    def archive(self, *args: Any, **kwargs: Any) -> list[SearchResultLike]:
        # ADR provider does not serve archive (L3). Explicit non-support.
        raise CapabilityError("archive", "ADR provider does not serve L3 archive", layer="L3", provider=self.name)

    # -- write operations (draft/accept only; never silent) ------------
    def draft(
        self,
        title: str,
        *,
        context: str,
        decision: str,
        alternatives: str = "",
        reasoning: str = "",
        consequences: str = "",
        related_components: Optional[list[str]] = None,
        project: str = _SYSTEM_PROJECT,
        proposed_by: str = "hermes",
        tags: Optional[list[str]] = None,
    ) -> DecisionRecord:
        """Hermes-autonomous write. Creates a PROPOSED draft (non-authoritative)."""
        project_key = _slug(project)
        number = self._next_number(project_key)
        created_at = _now_iso()
        rec = DecisionRecord(
            id=f"{project_key}/{number:03d}",
            title=title,
            status=_STATUS_PROPOSED,
            project=project_key,
            context=context,
            decision=decision,
            consequences=consequences,
            source=str(self._adr_path(project_key, number, title)),
            date=created_at,
            # provenance
            proposed_by=proposed_by,
            created_by=proposed_by,
            created_at=created_at,
            related_components=related_components or [],
            tags=tags or [],
        )
        self._write(rec, body_sections={
            "Alternatives considered": alternatives,
            "Reasoning": reasoning,
        })
        return rec

    def accept(
        self,
        id: str,
        *,
        approved_by: str,
        supersedes: Optional[list[str]] = None,
        status: str = _STATUS_ACCEPTED,
    ) -> DecisionRecord:
        """Human-gated authority transition. REQUIRES approved_by.

        Writes approved_by/approved_at provenance and the supersession
        back-links (this ADR's supersedes, and the old ADR's
        superseded_by) in the SAME event.
        """
        rec = self.get(id)
        if rec is None:
            raise CapabilityError("accept", f"no ADR with id={id!r}", layer="L4", provider=self.name)
        if status not in _VALID_STATUSES:
            raise CapabilityError("accept", f"invalid status {status!r}", layer="L4", provider=self.name)
        approved_at = _now_iso()
        rec.status = status
        rec.approved_by = approved_by
        rec.approved_at = approved_at
        rec.date = approved_at  # decision date = when it was accepted
        rec.decision_maker = approved_by
        if supersedes:
            rec.supersedes = supersedes
            for old_id in supersedes:
                self._backlink(old_id, rec.id)
        self._write(rec, body_sections={})
        return rec

    def remember(self, content: str, *, layer: str, **meta: Any) -> MemoryResult:
        # The ADR provider's authoritative write is accept(); a free-form
        # remember() is not how ADRs are created. Refuse explicitly.
        raise CapabilityError(
            "remember",
            "ADRs are created via draft()+accept(), not free-form remember()",
            layer=layer,
            provider=self.name,
        )

    def project(self, name: str) -> Any:  # L2 not served here
        return None

    # -- internals ------------------------------------------------------
    def _next_number(self, project_key: str) -> int:
        d = self._project_dir(project_key)
        if not d.is_dir():
            return 1
        max_n = 0
        for p in d.glob("*.md"):
            m = re.match(r"(\d+)-", p.name)
            if m:
                max_n = max(max_n, int(m.group(1)))
        return max_n + 1

    def _load_all(self, project_key: Optional[str] = None) -> list[DecisionRecord]:
        if not self.adr_root.is_dir():
            return []  # no ADRs yet -> empty, never crash
        roots = [self._project_dir(project_key)] if project_key else [d for d in self.adr_root.iterdir() if d.is_dir()]
        recs: list[DecisionRecord] = []
        for root in roots:
            if not root.is_dir():
                continue
            for p in sorted(root.glob("*.md")):
                rec = self._parse(p)
                if rec is not None:
                    recs.append(rec)
        return recs

    def _parse(self, path: Path) -> Optional[DecisionRecord]:
        raw = path.read_text(encoding="utf-8", errors="replace")
        fm, body = self._split_frontmatter(raw)
        meta = self._parse_frontmatter(fm)
        project_key = path.parent.name
        m = re.match(r"(\d+)-", path.name)
        number = int(m.group(1)) if m else 0
        rid = meta.get("id") or f"{project_key}/{number:03d}"
        sections = self._split_sections(body)
        return DecisionRecord(
            id=rid,
            title=meta.get("title", path.stem),
            status=meta.get("status", "proposed"),
            project=meta.get("project", ""),
            context=sections.get("Context", ""),
            decision=sections.get("Decision", meta.get("decision", "")),
            consequences=sections.get("Consequences", meta.get("consequences", "")),
            source=str(path),
            date=meta.get("date", ""),
            decision_maker=meta.get("decision_maker", ""),
            proposed_by=meta.get("proposed_by", meta.get("created_by", "")),
            related_components=meta.get("related_components", []) or [],
            supersedes=meta.get("supersedes", []) or [],
            superseded_by=meta.get("superseded_by", []) or [],
            tags=meta.get("tags", []) or [],
            # provenance
            created_by=meta.get("created_by", meta.get("proposed_by", "")),
            created_at=meta.get("created_at", meta.get("date", "")),
            approved_by=meta.get("approved_by", ""),
            approved_at=meta.get("approved_at", ""),
        )

    def _write(self, rec: DecisionRecord, *, body_sections: dict[str, str]) -> None:
        path = Path(rec.source)
        path.parent.mkdir(parents=True, exist_ok=True)
        fm = self._build_frontmatter(rec)
        body = self._build_body(rec, body_sections)
        path.write_text(fm + "\n" + body, encoding="utf-8")

    def _backlink(self, old_id: str, new_id: str) -> None:
        old = self.get(old_id)
        if old is None:
            return
        old.superseded_by = list(dict.fromkeys((old.superseded_by or []) + [new_id]))
        self._write(old, body_sections={})

    # -- markdown helpers ----------------------------------------------
    @staticmethod
    def _split_frontmatter(raw: str):
        if not raw.startswith("---"):
            return "", raw
        m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", raw, re.DOTALL)
        if not m:
            return "", raw
        return m.group(1), m.group(2)

    @staticmethod
    def _parse_frontmatter(fm: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for line in fm.splitlines():
            if ":" not in line:
                continue
            k, _, v = line.partition(":")
            k = k.strip()
            v = v.strip()
            if v.startswith("[") and v.endswith("]"):
                inner = v[1:-1].strip()
                out[k] = [x.strip().strip("'\"") for x in inner.split(",") if x.strip()] if inner else []
            else:
                out[k] = v.strip("'\"")
        return out

    @staticmethod
    def _split_sections(body: str) -> dict[str, str]:
        sections: dict[str, str] = {}
        cur = None
        buf: list[str] = []
        for line in body.splitlines():
            m = re.match(r"^##\s+(.*)$", line)
            if m:
                if cur is not None:
                    sections[cur] = "\n".join(buf).strip()
                cur = m.group(1).strip()
                buf = []
            elif cur is not None:
                buf.append(line)
        if cur is not None:
            sections[cur] = "\n".join(buf).strip()
        return sections

    @staticmethod
    def _build_frontmatter(rec: DecisionRecord) -> str:
        def fmt_list(xs: list) -> str:
            if not xs:
                return "[]"
            return "[" + ", ".join(f"'{x}'" for x in xs) + "]"

        lines = [
            "---",
            f"id: '{rec.id}'",
            f"title: '{rec.title}'",
            f"status: '{rec.status}'",
            f"date: '{rec.date}'",
            f"project: '{rec.id.split('/')[0]}'",
            f"decision_maker: '{rec.decision_maker}'",
            f"proposed_by: '{rec.proposed_by}'",
            f"related_components: {fmt_list(rec.related_components)}",
            f"supersedes: {fmt_list(rec.supersedes)}",
            f"superseded_by: {fmt_list(rec.superseded_by)}",
            f"tags: {fmt_list(rec.tags)}",
            f"created_by: '{rec.created_by}'",
            f"created_at: '{rec.created_at}'",
        ]
        if rec.approved_by:
            lines.append(f"approved_by: '{rec.approved_by}'")
        if rec.approved_at:
            lines.append(f"approved_at: '{rec.approved_at}'")
        lines.append("---")
        return "\n".join(lines)

    @staticmethod
    def _build_body(rec: DecisionRecord, extra: dict[str, str]) -> str:
        parts = [
            f"## Context",
            rec.context.strip(),
            f"## Decision",
            rec.decision.strip(),
        ]
        for heading, text in extra.items():
            if text:
                parts.append(f"## {heading}")
                parts.append(text.strip())
        parts.append("## Consequences")
        parts.append(rec.consequences.strip())
        parts.append("## Related components")
        parts.append(", ".join(rec.related_components) or "_none_")
        parts.append("## Supersession")
        sup = rec.supersedes or []
        sup_by = rec.superseded_by or []
        if sup:
            parts.append("Supersedes: " + ", ".join(sup))
        if sup_by:
            parts.append("Superseded by: " + ", ".join(sup_by))
        if not sup and not sup_by:
            parts.append("_no supersession_")
        return "\n\n".join(parts) + "\n"

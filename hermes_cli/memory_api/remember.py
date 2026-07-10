"""Phase 7 — RememberProvider (Layer 1 curated memory).

This module is the SOLE owner of L1 REMEMBER storage resolution. No remember
path is hardcoded anywhere else in the codebase; every filesystem location
derives from the path helpers below.

Design (docs/memory/memory-architecture.md, Intent.REMEMBER):
- Markdown is the source of truth. Each memory is one markdown file under
  ``<hermes_home>/memory/remember/<slug>.md`` while PROPOSED, and is relocated
  to ``<hermes_home>/memory/remember/accepted/<slug>.md`` when a human accepts
  it. The curated identity files (MEMORY.md / USER.md / SOUL.md / IDENTITY.md)
  are NEVER read or written by this provider — they stay human-authored.
- Authority model (approved option B): Hermes MAY draft (status=proposed) a
  non-authoritative suggestion artifact. Only a HUMAN approval (accept(),
  requires ``approved_by``) flips status to accepted. A proposed draft is
  NEVER returned as established memory.
- SQLite remains a derived index; remember markdown lives under ``memory/`` so
  it is auto-indexed — no indexer change required.
- No LLM extraction, no automatic acceptance, no embeddings, no
  Graphiti/Holographic.

Writes are never silent: accept() / draft() raise CapabilityError (never a
fake success) if the file does not actually persist.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from hermes_cli.memory_api.errors import CapabilityError
from hermes_cli.memory_api.protocols import (
    CapabilityStatus,
    MemoryProvider,
    RememberRecord,
)


_REMEMBER_DIRNAME = "remember"
_ACCEPTED_DIRNAME = "accepted"
_STATUS_PROPOSED = "proposed"
_STATUS_ACCEPTED = "accepted"
_VALID_STATUSES = {_STATUS_PROPOSED, _STATUS_ACCEPTED}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(text: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", (text or "note").lower()).strip("-")
    return base or "note"


class RememberProvider:
    """Structural MemoryProvider over markdown L1 memories (Layer 1)."""

    def __init__(self, hermes_home: Optional[Path] = None) -> None:
        self.hermes_home = Path(hermes_home) if hermes_home else self._default_home()

    @staticmethod
    def _default_home() -> Path:
        import os

        return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))

    # -- path resolution (THE ONLY place remember paths are constructed) ----
    @property
    def remember_root(self) -> Path:
        return self.hermes_home / "memory" / _REMEMBER_DIRNAME

    def _proposed_dir(self) -> Path:
        return self.remember_root

    def _accepted_dir(self) -> Path:
        return self.remember_root / _ACCEPTED_DIRNAME

    def _path_for(self, slug: str, *, accepted: bool) -> Path:
        if accepted:
            return self._accepted_dir() / f"{slug}.md"
        return self._proposed_dir() / f"{slug}.md"

    # -- metadata -------------------------------------------------------
    @property
    def name(self) -> str:
        return "remember"

    @property
    def layers(self) -> list[str]:
        return ["L1"]

    def status(self) -> CapabilityStatus:
        # File-backed source of truth; available as long as the home resolves.
        return CapabilityStatus.AVAILABLE

    # -- read operations (structural Protocol compatibility) --------------
    def search(self, query: str, *, limit: int = 10, scope: Optional[str] = None) -> list:
        # Remember is not a FTS5 backend. Explicit non-support — established
        # memories are surfaced via remember_established(), not search().
        raise CapabilityError("search", "RememberProvider does not serve search", layer="L1", provider=self.name)

    def archive(self, *args: Any, **kwargs: Any) -> list:
        raise CapabilityError("archive", "RememberProvider does not serve L3 archive", layer="L3", provider=self.name)

    def recent(self, *, limit: int = 10) -> list:
        raise CapabilityError("recent", "RememberProvider does not serve recent", layer="L3", provider=self.name)

    def project(self, name: str) -> Any:
        return None

    def decision(self, *args: Any, **kwargs: Any) -> list:
        raise CapabilityError("decision", "RememberProvider does not serve L4 decisions", layer="L4", provider=self.name)

    def remember(self, content: str, *, layer: str, **meta: Any) -> RememberRecord:
        # The provider's authoritative write path is draft(); free-form
        # remember() is refused (callers must use draft/accept). Mirrors the
        # ADR provider, which refuses remember() in favor of draft()+accept().
        raise CapabilityError(
            "remember",
            "L1 memory is written via draft()+accept(), not free-form remember()",
            layer=layer,
            provider=self.name,
        )

    # -- write (draft / accept) ------------------------------------------
    def draft(self, content: str, *, topic: str = "", proposed_by: str = "hermes", **meta: Any) -> RememberRecord:
        """Hermes-autonomous write. Creates a PROPOSED draft (non-authoritative).

        Writes ``<root>/<slug>.md`` with status=proposed. The curated identity
        files are never touched. Returns the RememberRecord (with source path).
        """
        if not content or not content.strip():
            raise CapabilityError("draft", "refusing to persist empty memory", layer="L1", provider=self.name)
        slug = _slug(topic or content[:40])
        slug = self._unique_slug(slug, accepted=False)
        created_at = _now_iso()
        rec = RememberRecord(
            id=f"remember/{slug}",
            slug=slug,
            content=content,
            status=_STATUS_PROPOSED,
            layer="L1",
            topic=topic,
            source=str(self._path_for(slug, accepted=False)),
            proposed_by=proposed_by,
            created_at=created_at,
        )
        self._write(rec, base_dir=self._proposed_dir())
        return rec

    def accept(self, id: str, *, approved_by: str, **meta: Any) -> RememberRecord:
        """Human-gated authority transition. REQUIRES approved_by.

        Loads the PROPOSED draft, flips status to accepted, writes provenance
        (approved_by/approved_at), and RELOCATES the file to the accepted/
        store. Raises CapabilityError if the id is unknown, already accepted,
        or the write does not actually persist.
        """
        if not approved_by:
            raise CapabilityError("accept", "acceptance requires an approved_by identity", layer="L1", provider=self.name)
        slug = self._slug_from_id(id)
        proposed_path = self._path_for(slug, accepted=False)
        accepted_path = self._path_for(slug, accepted=True)
        if not proposed_path.is_file():
            # Maybe already accepted?
            if accepted_path.is_file():
                raise CapabilityError("accept", f"remember {id!r} is already accepted", layer="L1", provider=self.name)
            raise CapabilityError("accept", f"no proposed remember with id={id!r}", layer="L1", provider=self.name)
        rec = self._parse(proposed_path)
        if rec is None:
            raise CapabilityError("accept", f"could not parse remember {id!r}", layer="L1", provider=self.name)
        approved_at = _now_iso()
        rec.status = _STATUS_ACCEPTED
        rec.approved_by = approved_by
        rec.approved_at = approved_at
        rec.id = f"remember/{slug}"
        rec.source = str(accepted_path)
        # Write to accepted/ FIRST (verify), then remove the proposed file.
        try:
            self._write(rec, base_dir=self._accepted_dir())
        except OSError as exc:
            raise CapabilityError("accept", f"failed to persist accepted memory: {exc}", layer="L1", provider=self.name) from exc
        if not accepted_path.is_file():
            raise CapabilityError("accept", "accepted memory was not written", layer="L1", provider=self.name)
        try:
            proposed_path.unlink()
        except OSError:
            pass  # relocation best-effort; the authoritative copy is in accepted/
        return rec

    # -- listing / reads ------------------------------------------------
    def list(self, *, status: Optional[str] = None) -> list[RememberRecord]:
        """Return all remember records, optionally filtered by status.

        Proposed drafts come from ``remember/<slug>.md``; accepted from
        ``remember/accepted/<slug>.md``. Both are returned with their true
        status. Callers that want ONLY established memory should filter
        status='accepted' (or call remember_established()).
        """
        out: list[RememberRecord] = []
        out.extend(self._load_dir(self._proposed_dir(), expected_status=_STATUS_PROPOSED))
        out.extend(self._load_dir(self._accepted_dir(), expected_status=_STATUS_ACCEPTED))
        if status:
            out = [r for r in out if r.status == status]
        return out

    def remember_established(self) -> list[RememberRecord]:
        """Return ONLY accepted (human-approved) memories — the trust boundary.

        Proposed drafts are excluded by construction. This is what callers
        should inject as established L1 memory.
        """
        return self.list(status=_STATUS_ACCEPTED)

    # -- internals ------------------------------------------------------
    def _slug_from_id(self, id: str) -> str:
        if id.startswith("remember/"):
            id = id[len("remember/"):]
        return id

    def _unique_slug(self, slug: str, *, accepted: bool) -> str:
        # Avoid clobbering an existing proposed file at the same slug; accept/
        # dir collisions are impossible because accepted is the destination.
        candidate = slug
        n = 1
        while self._path_for(candidate, accepted=accepted).is_file():
            n += 1
            candidate = f"{slug}-{n}"
        return candidate

    def _load_dir(self, d: Path, *, expected_status: str) -> list[RememberRecord]:
        if not d.is_dir():
            return []
        recs: list[RememberRecord] = []
        for p in sorted(d.glob("*.md")):
            rec = self._parse(p)
            if rec is None:
                continue
            # Trust the stored status, but never let a moved file lie about it.
            rec.status = expected_status if rec.status != expected_status else rec.status
            recs.append(rec)
        return recs

    def _parse(self, path: Path) -> Optional[RememberRecord]:
        raw = path.read_text(encoding="utf-8", errors="replace")
        fm, body = self._split_frontmatter(raw)
        meta = self._parse_frontmatter(fm)
        slug = path.stem
        return RememberRecord(
            id=meta.get("id") or f"remember/{slug}",
            slug=slug,
            content=(body or meta.get("content") or "").strip(),
            status=meta.get("status", _STATUS_PROPOSED),
            layer=meta.get("layer", "L1"),
            topic=meta.get("topic", ""),
            source=str(path),
            proposed_by=meta.get("proposed_by", ""),
            created_at=meta.get("created_at", ""),
            approved_by=meta.get("approved_by", ""),
            approved_at=meta.get("approved_at", ""),
        )

    def _write(self, rec: RememberRecord, *, base_dir: Path) -> None:
        base_dir.mkdir(parents=True, exist_ok=True)
        fm = self._build_frontmatter(rec)
        body = rec.content.strip() or "_no content_"
        path = base_dir / f"{rec.slug}.md"
        path.write_text(fm + "\n\n" + body + "\n", encoding="utf-8")

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
    def _parse_frontmatter(fm: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for line in fm.splitlines():
            if ":" not in line:
                continue
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip().strip("'\"")
        return out

    @staticmethod
    def _build_frontmatter(rec: RememberRecord) -> str:
        lines = [
            "---",
            f"id: '{rec.id}'",
            f"slug: '{rec.slug}'",
            f"status: '{rec.status}'",
            f"layer: '{rec.layer}'",
            f"topic: '{rec.topic}'",
            f"proposed_by: '{rec.proposed_by}'",
            f"created_at: '{rec.created_at}'",
        ]
        if rec.approved_by:
            lines.append(f"approved_by: '{rec.approved_by}'")
        if rec.approved_at:
            lines.append(f"approved_at: '{rec.approved_at}'")
        lines.append("---")
        return "\n".join(lines)

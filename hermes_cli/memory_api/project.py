"""Phase 6 — ProjectProvider (Layer 2 project memory).

This module is the SOLE owner of project-state storage resolution. No
project path is hardcoded anywhere else in the codebase; every filesystem
location derives from :meth:`ProjectProvider._project_dir` /
:meth:`ProjectProvider._status_path`.

Design (docs/memory/memory-architecture.md §18):
- Markdown is the source of truth. Each project's state is one markdown
  file under ``<hermes_home>/memory/projects/<project-key>/STATUS.md``.
- L2 describes the PRESENT, not the PAST. It links to history (ADRs /
  Archive / Search) by reference, never copies it.
- Authority model (§18.2, approved option B): L2 stays human-curated.
  Hermes MAY propose updates, but MUST NEVER modify STATUS.md without
  explicit human approval. ``set()`` is the only write path and is invoked
  only by an explicit human command (CLI / code). ``propose_*`` returns an
  in-memory draft and writes NOTHING.

Trust boundary (mirrors ADR §17.5 in spirit):
- ``get()`` returns the accepted, on-disk STATE.md (the authoritative
  source) — it never synthesizes project state from Archive/ADR/Search.
- ``set()`` raises :class:`CapabilityError` if the write does not actually
  persist (no silent success).

No LLM extraction, no automatic STATUS generation, no embeddings, no
Graphiti/Holographic.

Statelessness invariant (docs/memory/memory-architecture.md §18.11): providers own storage
access but hold NO authoritative mutable in-memory state. All truth lives on
disk (STATUS.md). Two reads of the same project must always agree; nothing is
cached or mutated in memory across calls.
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
    NextAction,
    ProjectState,
)


_PROJECT_DIRNAME = "projects"
_SYSTEM_PROJECT = "_system"
_VALID_STATUSES = {"active", "paused", "blocked", "done", "archived"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(project: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", project.lower()).strip("-") or _SYSTEM_PROJECT


class ProjectProvider:
    """Structural MemoryProvider over markdown project state (Layer 2)."""

    def __init__(self, hermes_home: Optional[Path] = None) -> None:
        self.hermes_home = Path(hermes_home) if hermes_home else self._default_home()

    @staticmethod
    def _default_home() -> Path:
        import os

        return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))

    # -- path resolution (THE ONLY place L2 paths are constructed) -------
    @property
    def projects_root(self) -> Path:
        return self.hermes_home / "memory" / _PROJECT_DIRNAME

    def _project_dir(self, project_key: str) -> Path:
        return self.projects_root / _slug(project_key or _SYSTEM_PROJECT)

    def _status_path(self, project_key: str) -> Path:
        return self._project_dir(project_key) / "STATUS.md"

    # -- metadata -------------------------------------------------------
    @property
    def name(self) -> str:
        return "project"

    @property
    def layers(self) -> list[str]:
        return ["L2"]

    def status(self) -> CapabilityStatus:
        # File-backed source of truth; available as long as the home resolves.
        return CapabilityStatus.AVAILABLE

    # -- read operations (PROJECT_STATE intent) -------------------------
    def get(self, project: str) -> Optional[ProjectState]:
        """Return the curated project state from STATUS.md, or None if absent."""
        path = self._status_path(project)
        if not path.is_file():
            return None
        return self._parse(path)

    def project(self, name: str) -> Optional[ProjectState]:
        # Protocol method; L2 read resolves to get().
        return self.get(name)

    def search_files(self, query: str, *, limit: int = 10, scope: Optional[str] = None) -> list:
        # L2 is not a search backend. Explicit non-support.
        raise CapabilityError("search", "ProjectProvider does not serve search", layer="L2", provider=self.name)

    def recent_files(self, *, limit: int = 10) -> list:
        raise CapabilityError("recent", "ProjectProvider does not serve recent", layer="L2", provider=self.name)

    def archive(self, *args: Any, **kwargs: Any) -> list:
        raise CapabilityError("archive", "ProjectProvider does not serve L3 archive", layer="L3", provider=self.name)

    def decision(self, *args: Any, **kwargs: Any) -> list:
        raise CapabilityError("decision", "ProjectProvider does not serve L4 decisions", layer="L4", provider=self.name)

    def remember(self, content: str, *, layer: str, **meta: Any):
        # L2 authoritative write is set(); free-form remember() is refused.
        raise CapabilityError(
            "remember",
            "Project state is written via set(), not free-form remember()",
            layer=layer,
            provider=self.name,
        )

    # -- write (human-gated; the ONLY persistence path) -----------------
    def set(self, state: ProjectState, *, updated_by: Optional[str] = None) -> ProjectState:
        """Persist a ProjectState to STATUS.md. The only write path.

        Raises CapabilityError (never silent) if the file is not actually
        written. Caller (CLI / explicit code) is responsible for human
        authorization — Hermes never calls this autonomously.
        """
        if state.status and state.status not in _VALID_STATUSES:
            raise CapabilityError("set", f"invalid status {state.status!r}", layer="L2", provider=self.name)
        path = self._status_path(state.project)
        # Slug collision guard: if a STATUS.md already exists at this slug's
        # path under a DIFFERENT project key, two distinct projects have
        # resolved to the same storage directory. Prefer explicit failure over
        # ambiguous overwrite (forward-review requirement: thousands of projects
        # => cheap collision insurance; no silent resolution).
        if path.is_file():
            existing = self._parse(path)
            if existing is not None and existing.project != state.project:
                raise CapabilityError(
                    "set",
                    f"slug collision: '{state.project}' and existing "
                    f"'{existing.project}' both resolve to directory "
                    f"{path.parent.name!r}; refusing ambiguous write",
                    layer="L2", provider=self.name,
                )
        # Ensure directory exists before attempting write.
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._write(state, updated_by=updated_by)
        except OSError as exc:
            raise CapabilityError("set", f"failed to persist STATUS.md: {exc}", layer="L2", provider=self.name)
        # Verify the write actually landed (no silent success).
        if not path.is_file():
            raise CapabilityError("set", "STATUS.md was not written", layer="L2", provider=self.name)
        reread = self._parse(path)
        if reread is None or reread.project != state.project:
            raise CapabilityError("set", "STATUS.md persist verification failed", layer="L2", provider=self.name)
        return reread

    def propose_update(
        self,
        project: str,
        *,
        status: Optional[str] = None,
        next_actions: Optional[list[NextAction]] = None,
        blockers: Optional[list[str]] = None,
        goals: Optional[list[str]] = None,
        narrative: Optional[str] = None,
    ) -> ProjectState:
        """Hermes-autonomous SUGGESTION. Writes NOTHING.

        Returns an in-memory ProjectState marked as a proposal. The human
        must accept it (via an explicit set()) before it touches disk. This
        is the only Hermes-initiated path, and it is strictly non-persistent.
        """
        base = self.get(project)
        proposed = base or ProjectState(project=_slug(project), title=project)
        if status is not None:
            proposed.status = status
        if next_actions is not None:
            proposed.next_actions = list(next_actions)
        if blockers is not None:
            proposed.blockers = list(blockers)
        if goals is not None:
            proposed.goals = list(goals)
        if narrative is not None:
            proposed.narrative = narrative
        # Mark as proposal so consumers never mistake it for accepted state.
        proposed.source = ""  # not on disk
        return proposed

    # -- internals ------------------------------------------------------
    def _parse(self, path: Path) -> Optional[ProjectState]:
        raw = path.read_text(encoding="utf-8", errors="replace")
        fm, body = self._split_frontmatter(raw)
        meta = self._parse_frontmatter(fm)
        project_key = meta.get("project") or path.parent.name
        sections = self._split_sections(body)
        narrative = sections.get("Current state", "")
        links_raw = meta.get("links", {}) or {}
        # links may be stored as a dict already (from _build_frontmatter) or
        # as a list of "kind: [..]" lines; normalize defensively.
        links: dict[str, list[str]] = {}
        if isinstance(links_raw, dict):
            for k, v in links_raw.items():
                links[k] = list(v) if isinstance(v, (list, tuple)) else [str(v)]
        next_actions: list[NextAction] = []
        raw_actions = meta.get("next_actions") or []
        if isinstance(raw_actions, list):
            for a in raw_actions:
                if isinstance(a, dict):
                    next_actions.append(NextAction(
                        what=str(a.get("what", "")),
                        owner=str(a.get("owner", "unassigned")),
                        blocked_by=list(a.get("blocked_by", []) or []),
                    ))
        return ProjectState(
            project=project_key,
            title=meta.get("title", path.parent.name),
            status=meta.get("status", ""),
            updated_at=meta.get("updated_at", ""),
            updated_by=meta.get("updated_by", ""),
            owners=list(meta.get("owners", []) or []),
            next_actions=next_actions,
            goals=list(meta.get("goals", []) or []),
            blockers=list(meta.get("blockers", []) or []),
            open_questions=list(meta.get("open_questions", []) or []),
            links=links,
            last_verified=meta.get("last_verified", ""),
            verified_by=meta.get("verified_by", ""),
            narrative=narrative,
            source=str(path),
        )

    def _write(self, state: ProjectState, *, updated_by: Optional[str] = None) -> None:
        path = self._status_path(state.project)
        path.parent.mkdir(parents=True, exist_ok=True)
        fm = self._build_frontmatter(state, updated_by=updated_by)
        body = self._build_body(state)
        path.write_text(fm + "\n" + body, encoding="utf-8")

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
            if k == "next_actions":
                # Robust encoding: "what::owner::blk1,blk2"
                inner = v[1:-1].strip() if v.startswith("[") else v
                actions = []
                for item in inner.split(","):
                    item = item.strip().strip("'\"")
                    if not item:
                        continue
                    what, _, rest = item.partition("::")
                    owner, _, blk = rest.partition("::")
                    blk_list = [b.strip() for b in blk.split(",") if b.strip()]
                    actions.append({"what": what.strip(), "owner": owner.strip() or "unassigned", "blocked_by": blk_list})
                out[k] = actions
            elif v.startswith("[") and v.endswith("]"):
                inner = v[1:-1].strip()
                out[k] = [x.strip().strip("'\"") for x in inner.split(",") if x.strip()] if inner else []
            elif v.startswith("{") and v.endswith("}"):
                # inline dict (links) — keep as raw mapping after splitting
                inner = v[1:-1].strip()
                d: dict[str, Any] = {}
                for pair in inner.split(","):
                    if ":" in pair:
                        dk, _, dv = pair.partition(":")
                        d[dk.strip()] = [x.strip().strip("'\"") for x in dv.strip().strip("[]").split(",") if x.strip()]
                out[k] = d
            elif k == "next_actions":
                # Robust encoding: "what::owner::blk1,blk2"
                inner = v[1:-1].strip() if v.startswith("[") else v
                actions = []
                for item in inner.split(","):
                    item = item.strip().strip("'\"")
                    if not item:
                        continue
                    what, _, rest = item.partition("::")
                    owner, _, blk = rest.partition("::")
                    blk_list = [b.strip() for b in blk.split(",") if b.strip()]
                    actions.append({"what": what.strip(), "owner": owner.strip() or "unassigned", "blocked_by": blk_list})
                out[k] = actions
            else:
                out[k] = v.strip("'\"")
        return out

    @staticmethod
    def _split_sections(body: str) -> dict[str, str]:
        sections: dict[str, str] = {}
        cur: Optional[str] = None
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
    def _build_frontmatter(state: ProjectState, *, updated_by: Optional[str] = None) -> str:
        def fmt_list(xs: list) -> str:
            xs = list(xs or [])
            if not xs:
                return "[]"
            return "[" + ", ".join(f"'{x}'" for x in xs) + "]"

        def fmt_actions(actions: list[NextAction]) -> str:
            # Robust single-line encoding: 'what::owner::blk1,blk2'
            if not actions:
                return "[]"
            items = []
            for a in actions:
                blk = ",".join(a.blocked_by)
                items.append(f"'{a.what}::{a.owner}::{blk}'")
            return "[" + ", ".join(items) + "]"

        def fmt_links(links: dict[str, list[str]]) -> str:
            if not links:
                return "{}"
            parts = []
            for k, v in links.items():
                parts.append(f"{k}: {fmt_list(v)}")
            return "{" + ", ".join(parts) + "}"

        lines = [
            "---",
            f"project: '{state.project}'",
            f"title: '{state.title or state.project}'",
            f"status: '{state.status}'",
            f"updated_at: '{_now_iso()}'",
            f"updated_by: '{updated_by or state.updated_by}'",
            f"owners: {fmt_list(state.owners)}",
            f"next_actions: {fmt_actions(state.next_actions)}",
            f"goals: {fmt_list(state.goals)}",
            f"blockers: {fmt_list(state.blockers)}",
            f"open_questions: {fmt_list(state.open_questions)}",
            f"links: {fmt_links(state.links)}",
            f"last_verified: '{state.last_verified}'",
            f"verified_by: '{state.verified_by}'",
            "---",
        ]
        return "\n".join(lines)

    @staticmethod
    def _build_body(state: ProjectState) -> str:
        parts = ["## Current state", state.narrative.strip() or "_no narrative_"]
        return "\n\n".join(parts) + "\n"

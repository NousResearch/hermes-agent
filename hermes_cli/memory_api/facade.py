"""Phase 4 — MemoryAPI facade (the stable front door).

Callers use ONLY this class. Internally it delegates EVERY operation — reads
AND writes — to the registered :class:`~hermes_cli.memory_router.router.MemoryRouter`
capabilities. The facade holds NO concrete provider instances and never
imports a storage backend; it speaks only intents. This keeps callers
backend-agnostic: adding or swapping a provider touches the Router's registry,
not the facade or any caller.

The Router decides *routing*; providers own *storage behavior*. The API defines
*available operations* and normalizes every result into a provenance-bearing
:class:`~hermes_cli.memory_api.protocols.MemoryResult`.

Context assembly is STRUCTURED, not concatenated: results are bucketed by
source category (identity / project / decision / recent / other), each entry
keeping full provenance. The API performs NO LLM reasoning or ranking.

Writes are never silent: an unsupported/unavailable write raises
:class:`~hermes_cli.memory_api.errors.CapabilityError`. No "success" is ever
reported for an unpersisted write (docs/memory/memory-architecture.md §16.4).
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from pathlib import Path

from hermes_cli.memory_api.errors import CapabilityError, UnsupportedCapability
from hermes_cli.memory_api.protocols import (
    ContextBundle,
    DecisionRecord,
    MemoryResult,
    ProjectState,
    NextAction,
    RememberRecord,
)
from hermes_cli.memory_router.provenance import SearchResult
from hermes_cli.memory_router.router import MemoryRouter
from hermes_cli.memory_router.intents import Intent


class MemoryAPI:
    """Stable, versioned interface between callers and memory internals.

    The facade is a THIN translation layer: it maps caller intent -> Router
    intent -> normalized result. It owns no storage, no provider instances,
    and no routing logic.
    """

    def __init__(self, router: Optional[MemoryRouter] = None) -> None:
        self._router = router or MemoryRouter()

    # -- read operations -------------------------------------------------
    def search(self, query: str, *, limit: int = 10, scope: Optional[str] = None) -> list[MemoryResult]:
        res = self._router.search(query, limit=limit, scope=scope)
        return self._normalize(res.results, provider="router", intent="historical")

    def archive(self, query: str = "", *, limit: int = 10, session_id: Optional[str] = None) -> list[MemoryResult]:
        res = self._router.archive(query=query, limit=limit, session_id=session_id)
        return self._normalize(res.results, provider="router", intent="archive")

    def recent(self, *, limit: int = 10) -> list[MemoryResult]:
        res = self._router.recent(limit=limit)
        return self._normalize(res.results, provider="router", intent="recent")

    # -- structured context (no LLM ranking; categories preserved) -------
    def context(self, query: str, *, limit_per_category: int = 5, project: Optional[str] = None) -> ContextBundle:
        """Assemble cross-layer context as a STRUCTURED bundle.

        Fan-out is DECLARATIVE (Invariant B): for each capability registered
        with the Router, if it opts in via ``contributes_to_context`` we fold
        its contribution into the ContextBundle slot named by its
        ``context_category``. A capability participates in context ONLY when a
        human/reviewed decision has set the opt-in flag — merely existing in
        the registry never implies context participation (the guardrail that
        stops "more backends" from silently changing what Hermes loads).

        The four product-opted capabilities and their slots are:
          - "identity"  (L1-identity)  -> identity pointers (via router)
          - "recent"    (L3-archive)   -> recent archive activity
          - "decision"  (L4-adr)       -> accepted ADRs
          - "project"   (L2-project)    -> the single active project

        No reasoning/ranking is performed here — callers assemble the prompt.
        """
        bundle = ContextBundle()
        for cap in self._router.registry.list():
            if not cap.in_context():
                # Either unavailable, or opted out (the default). A capability
                # contributes nothing to context unless explicitly opted in.
                continue
            category = cap.context_category
            if category == "identity":
                ident = self._router.search(query, limit=limit_per_category, scope="L1-identity")
                bundle.identity = self._normalize(ident.results, provider="router", intent="identity")
            elif category == "recent":
                bundle.recent = self._normalize(
                    self._router.recent(limit=limit_per_category).results,
                    provider="router",
                    intent="recent",
                )
            elif category == "decision":
                # Route through the facade's own decision() so it benefits from
                # the Router's DECISION intent and the trust boundary (only
                # ACCEPTED ADRs surface; proposed drafts are excluded). If no L4
                # backend is wired, decision() returns [] gracefully.
                try:
                    decs = self.decision()  # accepted, newest-first
                except Exception:  # noqa: BLE001 — one layer must not break context
                    decs = []
                for d in decs:
                    bundle.decision.append(
                        MemoryResult(
                            source=d.source, provider="adr", layer="L4",
                            retrieval_method="adr", content=d.decision or d.title,
                            extra={"id": d.id, "title": d.title, "status": d.status,
                                   "project": d.project},
                        )
                    )
            elif category == "project":
                # Single active project only (Phase 6 design §18.4). If no L2
                # exists, bundle.project stays empty (never fabricated).
                try:
                    resolved = self._resolve_current_project(project)
                    if resolved:
                        ps = self.project(resolved)
                        if ps is not None:
                            bundle.project.append(
                                MemoryResult(
                                    source=ps.source, provider="project", layer="L2",
                                    retrieval_method="project",
                                    content=ps.narrative or ps.status,
                                    extra={
                                        "project": ps.project, "title": ps.title,
                                        "status": ps.status, "owners": ps.owners,
                                        "next_actions": [
                                            {"what": a.what, "owner": a.owner, "blocked_by": a.blocked_by}
                                            for a in ps.next_actions
                                        ],
                                        "blockers": ps.blockers, "goals": ps.goals,
                                        "open_questions": ps.open_questions,
                                        "links": ps.links,
                                        "last_verified": ps.last_verified,
                                        "verified_by": ps.verified_by,
                                        "updated_at": ps.updated_at,
                                    },
                                )
                            )
                except Exception:  # noqa: BLE001 — one layer must not break context
                    pass
            # Unknown / unhandled category: ignored (defensive; opt-ins are
            # reviewed and limited to the four above).
        return bundle

    # -- write operations (explicit; never silent) -----------------------
    def remember(self, content: str, *, layer: str, **meta: Any) -> MemoryResult:
        """Persist a memory. Routes through the Router's REMEMBER intent.

        Only the L1 layer has a writable provider; a write to any other layer
        raises CapabilityError — never a silent no-op. For L1, this creates a
        PROPOSED (non-authoritative) memory; accept it via
        :meth:`accept_remember` to make it established.
        """
        if layer != "L1":
            raise CapabilityError("remember", "no writable provider for this layer", layer=layer)
        res = self._router.remember(content, layer=layer, **meta)
        if not res.ok:
            raise CapabilityError("remember", "no writable provider for this layer", layer=layer)
        raw = res.results
        if isinstance(raw, list) and raw:
            raw = raw[0]
        if isinstance(raw, RememberRecord):
            return MemoryResult(
                source=raw.source,
                provider="remember",
                layer=raw.layer,
                retrieval_method="draft",
                content=raw.content,
                intent="remember",
                extra={"id": raw.id, "status": raw.status, "topic": raw.topic},
            )
        raise CapabilityError("remember", "write returned no result", layer=layer)

    def project(self, project: Optional[str] = None) -> Optional[ProjectState]:
        """Return curated L2 project state, or None if none exists.

        Does NOT raise when no record exists — context() and CLI treat a
        missing L2 as "no current project state" (never fabricated). Routes
        through the Router's PROJECT_STATE intent so the L2-project provider
        is the sole storage owner.
        """
        resolved = self._resolve_current_project(project)
        if not resolved:
            return None
        res = self._router.project(project=resolved)
        # route() returns a RouteResult; the L2-project handle returns the
        # ProjectState (or None) directly, not wrapped in a list. Normalize.
        raw = res.results
        if isinstance(raw, ProjectState):
            return raw
        for r in raw or []:
            if isinstance(r, ProjectState):
                return r
        return None

    def set_project(self, state: ProjectState, *, updated_by: str) -> ProjectState:
        """Human-gated persistence of project state. The ONLY L2 write path.

        Hermes never calls this autonomously. Raises CapabilityError (never
        silent) if the write does not actually persist. Routed through the
        Router's PROJECT_STATE intent (single owner, no facade-held provider).
        """
        res = self._router.project(project=state.project, method="set", state=state, updated_by=updated_by)
        raw = res.results
        if isinstance(raw, ProjectState):
            return raw
        for r in raw or []:
            if isinstance(r, ProjectState):
                return r
        raise CapabilityError("set_project", "L2 write returned no state", layer="L2", provider="project")

    def propose_project(self, project: str, **kwargs: Any) -> ProjectState:
        """Hermes-autonomous SUGGESTION. Writes NOTHING (per authority model B).

        Returns an in-memory ProjectState marked as a proposal; the human must
        accept it via :meth:`set_project` before it touches disk. Routed
        through the Router's PROJECT_STATE intent (method="propose").
        """
        res = self._router.project(method="propose", project=project, kwargs=kwargs)
        raw = res.results
        if isinstance(raw, ProjectState):
            return raw
        for r in raw or []:
            if isinstance(r, ProjectState):
                return r
        raise CapabilityError("propose_project", "provider returned no proposal", layer="L2", provider="project")

    def _resolve_current_project(self, explicit: Optional[str] = None) -> Optional[str]:
        """Resolve the "active" project using the approved precedence:

        1. explicit project parameter
        2. configured current project (config.yaml: memory.current_project)
        3. current git repository detection (cwd inside a repo)
        4. None  (never infer/fabricate beyond these rules)
        """
        if explicit:
            return explicit
        # (2) configured current project
        try:
            from hermes_cli.config import cfg_get, load_config

            cfg = load_config()
            configured = cfg_get(cfg, "memory", "current_project", default="") or ""
            if configured:
                return configured
        except Exception:  # noqa: BLE001 — config failure must not break resolution
            pass
        # (3) git repository detection (best-effort, cwd only)
        try:
            import subprocess

            top = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=str(Path.cwd()), capture_output=True, text=True, timeout=5,
            )
            if top.returncode == 0 and top.stdout.strip():
                name = Path(top.stdout.strip()).name
                if name:
                    return name
        except Exception:  # noqa: BLE001 — git absent/non-repo must fall through
            pass
        # (4) nothing determinable -> None (preferred over guessing)
        return None

    def decision(
        self,
        id: Optional[str] = None,
        topic: Optional[str] = None,
        project: Optional[str] = None,
    ) -> list[DecisionRecord]:
        """Return accepted ADRs. Routes through the Router's DECISION intent.

        Per design §17.2 the Router owns routing; the facade forwards to the
        L4-adr capability. Only ACCEPTED ADRs come back — proposed drafts
        are excluded at the provider read boundary (trust boundary, §17.5).
        """
        res = self._router.decision(id=id, topic=topic, project=project)
        out: list[DecisionRecord] = []
        for r in res.results or []:
            if isinstance(r, DecisionRecord):
                out.append(r)
        # No id/topic => the "recent" case (design §17.2): newest-first.
        if id is None and topic is None:
            out.sort(key=lambda r: r.date or "", reverse=True)
        return out

    # -- write operations (human-gated authority), all routed via Router ---
    def draft_decision(
        self,
        title: str,
        *,
        context: str,
        decision: str,
        alternatives: str = "",
        reasoning: str = "",
        consequences: str = "",
        related_components: Optional[list[str]] = None,
        project: str = "_system",
        proposed_by: str = "hermes",
        tags: Optional[list[str]] = None,
    ) -> DecisionRecord:
        """Hermes-autonomous write: creates a PROPOSED draft (non-authoritative).

        Routed through the Router's DECISION intent (method="draft") — the
        facade holds no AdrProvider instance.
        """
        res = self._router.draft_decision(
            title,
            context=context,
            decision=decision,
            alternatives=alternatives,
            reasoning=reasoning,
            consequences=consequences,
            related_components=related_components,
            project=project,
            proposed_by=proposed_by,
            tags=tags,
        )
        raw = res.results
        if isinstance(raw, DecisionRecord):
            return raw
        for r in raw or []:
            if isinstance(r, DecisionRecord):
                return r
        raise CapabilityError("draft_decision", "ADR draft returned no record", layer="L4", provider="adr")

    def accept_decision(
        self,
        id: str,
        *,
        approved_by: str,
        supersedes: Optional[list[str]] = None,
        status: str = "accepted",
    ) -> DecisionRecord:
        """Human-gated authority transition. REQUIRES approved_by.

        Writes approved_by/approved_at provenance + supersession back-links.
        Routed through the Router's DECISION intent (method="accept").
        """
        res = self._router.accept_decision(
            id, approved_by=approved_by, supersedes=supersedes, status=status
        )
        raw = res.results
        if isinstance(raw, DecisionRecord):
            return raw
        for r in raw or []:
            if isinstance(r, DecisionRecord):
                return r
        raise CapabilityError("accept_decision", "ADR accept returned no record", layer="L4", provider="adr")

    # -- write operations (human-gated authority), all routed via Router ---  # noqa: E501
    def draft_remember(
        self,
        content: str,
        *,
        topic: str = "",
        layer: str = "L1",
        proposed_by: str = "hermes",
    ) -> "RememberRecord":
        """Hermes-autonomous write: creates a PROPOSED L1 memory (non-authoritative).

        Routed through the Router's REMEMBER intent (method="draft"). The
        curated identity files are never touched; the draft is written to a
        staging markdown file and is NOT surfaced as established memory until
        a human calls :meth:`accept_remember`.
        """
        res = self._router.route(Intent.REMEMBER, "draft", content=content, topic=topic, proposed_by=proposed_by, layer=layer)
        raw = res.results
        if isinstance(raw, RememberRecord):
            return raw
        for r in raw or []:
            if isinstance(r, RememberRecord):
                return r
        raise CapabilityError("draft_remember", "remember draft returned no record", layer="L1", provider="remember")

    def accept_remember(self, id: str, *, approved_by: str) -> "RememberRecord":
        """Human-gated authority transition. REQUIRES approved_by.

        Flips a PROPOSED memory to accepted and relocates it to the accepted/
        store. Routed through the Router's REMEMBER intent (method="accept").
        """
        res = self._router.route(Intent.REMEMBER, "accept", id=id, approved_by=approved_by, layer="L1")
        raw = res.results
        if isinstance(raw, RememberRecord):
            return raw
        for r in raw or []:
            if isinstance(r, RememberRecord):
                return r
        raise CapabilityError("accept_remember", "remember accept returned no record", layer="L1", provider="remember")

    def list_remember(self, *, status: Optional[str] = None) -> list["RememberRecord"]:
        """List L1 memories, optionally filtered by status (proposed|accepted)."""
        res = self._router.route(Intent.REMEMBER, "list", status=status, layer="L1")
        raw = res.results
        if isinstance(raw, list):
            return [r for r in raw if isinstance(r, RememberRecord)]
        return []

    def remember_established(self) -> list["RememberRecord"]:
        """Return ONLY accepted (human-approved) L1 memories — the trust boundary.

        Proposed drafts are excluded by construction. This is what callers
        should treat as established memory.
        """
        res = self._router.route(Intent.REMEMBER, "established", layer="L1")
        raw = res.results
        if isinstance(raw, list):
            return [r for r in raw if isinstance(r, RememberRecord)]
        return []

    # -- internals -------------------------------------------------------
    def _normalize(self, results: Iterable[Any], *, provider: str, intent: Optional[str]) -> list[MemoryResult]:
        out: list[MemoryResult] = []
        for r in results or []:
            if isinstance(r, SearchResult):
                out.append(MemoryResult.from_search_result(r, provider, intent=intent))
            elif hasattr(r, "source_file"):
                out.append(MemoryResult.from_search_result(r, provider, intent=intent))
            else:
                # Best-effort: treat as raw content with minimal provenance.
                out.append(
                    MemoryResult(
                        source=str(getattr(r, "source_file", "")),
                        provider=provider,
                        layer=getattr(r, "memory_layer", ""),
                        retrieval_method=getattr(r, "retrieval_method", "unknown"),
                        content=str(r),
                        intent=intent,
                    )
                )
        return out

    def capability_status(self) -> dict[str, str]:
        """Snapshot of every routed capability's readiness (graceful degrad.)."""
        return self._router.capability_status()

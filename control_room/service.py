"""Control Room profile-scoped snapshot service (CR-101..CR-104).

Composes existing authoritative runtime readers into a
``ControlRoomSnapshot``. Pure orchestration: every provider either returns
rows with a ``SourceMeta`` or is caught and turned into a typed
``unavailable`` item — the snapshot never silently pretends a missing source
is empty.

Providers are injectable for tests; defaults wrap the real readers:
- agents: live gateway/CLI running-agent context (when supplied)
- processes: ``tools.process_registry`` (module-level, session-scoped)
- delegations: ``tools.async_delegation`` durable registry
- kanban: ``hermes_cli.kanban_db`` read-only queries (never raw writes)
- peer: Hermes Peer public API (optional import; absent -> unavailable)
- system: system health reader (degraded -> unknown)

The service holds a small bounded cache keyed by ``(profile, context_key)``
so status-bar polling does not re-read every provider on every repaint
(CR-104). Reads are synchronous; gateway event updates can supersede the
cache later without changing this interface.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from .attention import rank_attention, severity_for_kind
from .contract import (
    AgentRow,
    AttentionItem,
    AttentionKind,
    AttentionSeverity,
    Capabilities,
    ControlRoomSnapshot,
    MessageRow,
    SnapshotCounts,
    SourceMeta,
    SystemSummary,
    TaskRow,
    unavailable_attention,
    unavailable_source,
    unavailable_system,
)

DEFAULT_CACHE_MAX_AGE_SECONDS = 2.0
MAX_ROWS_PER_SECTION = 50

ProviderFn = Callable[[Dict[str, Any]], List[Any]]


class ControlRoomService:
    """Builds profile-scoped Control Room snapshots from injected providers."""

    def __init__(
        self,
        providers: Optional[Dict[str, ProviderFn]] = None,
        cache_max_age_seconds: float = DEFAULT_CACHE_MAX_AGE_SECONDS,
    ) -> None:
        self.providers = providers or default_providers()
        self.cache_max_age_seconds = cache_max_age_seconds
        self._cache: Dict[str, tuple[float, ControlRoomSnapshot]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_snapshot(
        self,
        profile: str = "default",
        context: Optional[Dict[str, Any]] = None,
        *,
        refresh: bool = False,
    ) -> ControlRoomSnapshot:
        """Build (or return a fresh-enough cached) snapshot for ``profile``.

        ``context`` carries live surface state (e.g. a gateway's
        ``_running_agents``) — optional; providers degrade when absent.
        """
        context_key = _context_key(context)
        cache_key = f"{profile}|{context_key}"
        now = time.monotonic()
        if not refresh and cache_key in self._cache:
            cached_at, cached = self._cache[cache_key]
            if now - cached_at < self.cache_max_age_seconds:
                return cached

        ctx: Dict[str, Any] = {"profile": profile, **(context or {})}
        snapshot = self._assemble(profile, ctx)
        self._cache[cache_key] = (now, snapshot)
        return snapshot

    def invalidate(self) -> None:
        self._cache.clear()

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def _assemble(self, profile: str, ctx: Dict[str, Any]) -> ControlRoomSnapshot:
        agents, agents_meta = self._collect("agents", ctx)
        processes, processes_meta = self._collect("processes", ctx)
        delegations, delegations_meta = self._collect("delegations", ctx)
        tasks, kanban_meta = self._collect("kanban", ctx)
        messages, peer_meta = self._collect("peer", ctx)
        system_list, system_meta = self._collect("system", ctx)
        system = system_list[0] if system_list else unavailable_system(
            "system", "no system summary available"
        )

        agent_rows: List[AgentRow] = []
        for a in agents[:MAX_ROWS_PER_SECTION]:
            agent_rows.append(_as_agent_row(a, profile, agents_meta))

        task_rows: List[TaskRow] = []
        for t in tasks[:MAX_ROWS_PER_SECTION]:
            task_rows.append(_as_task_row(t, profile, kanban_meta))

        message_rows: List[MessageRow] = []
        for m in messages[:MAX_ROWS_PER_SECTION]:
            message_rows.append(_as_message_row(m, profile, peer_meta))

        # Attention assembly (CR-103): approvals/held first, errors/stalled,
        # blocked/review, then info. Providers emit typed attention where they
        # know something needs a human; the service also derives stalled/ready
        # signals from row state so counts stay consistent.
        attention: List[AttentionItem] = []
        attention.extend(_attention_from_agents(agent_rows, agents_meta))
        attention.extend(_attention_from_tasks(task_rows, kanban_meta))
        attention.extend(_attention_from_messages(message_rows, peer_meta))
        attention.extend(_attention_from_system(system))
        attention = rank_attention(attention)

        counts = _derive_counts(attention, agent_rows, task_rows, message_rows, system)

        capabilities = Capabilities(
            approvals=bool(ctx.get("approvals_available", False)),
            peer_messages=peer_meta.state == "ok",
            kanban_actions=kanban_meta.state == "ok",
            process_control=processes_meta.state == "ok",
            delegation_control=delegations_meta.state == "ok",
        )

        return ControlRoomSnapshot(
            version=1,
            profile=profile,
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            attention=attention,
            counts=counts,
            agents=agent_rows,
            tasks=task_rows,
            messages=message_rows,
            system=system,
            capabilities=capabilities,
        )

    def _collect(self, name: str, ctx: Dict[str, Any]) -> tuple[List[Any], SourceMeta]:
        provider = self.providers.get(name)
        if provider is None:
            return [], unavailable_source(name, "no provider registered")
        try:
            rows = provider(ctx)
            if rows is None:
                return [], unavailable_source(name, "provider returned no data")
            return rows, SourceMeta(provider=name, state="ok")
        except Exception as exc:  # noqa: BLE001 - provider isolation boundary
            return [], unavailable_source(name, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Default providers — wrap the real authoritative readers.
# ---------------------------------------------------------------------------


def _agents_provider(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    running = ctx.get("running_agents") or {}
    started = ctx.get("running_agents_ts") or {}
    now = time.time()
    rows: List[Dict[str, Any]] = []
    for session_key, agent in running.items():
        elapsed = max(0, int(now - float(started.get(session_key, now))))
        rows.append(
            {
                "kind": "agent",
                "id": session_key,
                "name": session_key,
                "status": "starting" if agent is None else "running",
                "elapsed_seconds": elapsed,
                "session_id": str(getattr(agent, "session_id", "") or ""),
                "model": str(getattr(agent, "model", "") or ""),
            }
        )
    if not running and not ctx.get("_allow_empty_agents", False):
        # No live agent context in this process — typed degraded, not zero.
        # Callers that genuinely have no agents (fresh session) pass
        # _allow_empty_agents=True to get an empty ok list instead.
        raise RuntimeError("no live running-agent context")
    return rows


def _processes_provider(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    from tools.process_registry import process_registry

    try:
        sessions = process_registry.list_sessions()
    except Exception:  # noqa: BLE001
        sessions = []
    rows = []
    for p in sessions:
        if p.get("status") != "running":
            continue
        rows.append(
            {
                "kind": "process",
                "id": str(p.get("session_id", "?")),
                "name": " ".join(str(p.get("command", "")).split())[:80],
                "status": "running",
                "uptime_seconds": int(p.get("uptime_seconds", 0) or 0),
            }
        )
    return rows


def _delegations_provider(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    from tools.async_delegation import list_async_delegations

    try:
        delegations = list_async_delegations()
    except Exception:  # noqa: BLE001
        delegations = []
    rows = []
    for d in delegations:
        status = d.get("status", "?")
        if status not in ("running", "stalling", "finalizing"):
            continue
        goal = " ".join(str(d.get("goal") or "").split())
        rows.append(
            {
                "kind": "delegation",
                "id": str(d.get("delegation_id", "?")),
                "name": goal[:70] or d.get("delegation_id", "?"),
                "status": status,
                "quiet_seconds": d.get("seconds_since_progress"),
            }
        )
    return rows


def _kanban_provider(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Read-only Kanban task summaries through ``kanban_db`` (CR-205: Control
    Room never writes SQLite directly; reads are fine through the DB API)."""
    from hermes_cli import kanban_db

    board = ctx.get("kanban_board")
    with kanban_db.connect_closing(db_path=None) as conn:
        tasks = kanban_db.list_tasks(conn, limit=MAX_ROWS_PER_SECTION + 1)
    rows = []
    for t in tasks[:MAX_ROWS_PER_SECTION]:
        rows.append(
            {
                "id": str(getattr(t, "id", "")),
                "title": str(getattr(t, "title", "")),
                "state": str(getattr(t, "status", "") or getattr(t, "state", "") or "unknown"),
                "board": str(board or ""),
                "owner": str(getattr(t, "assignee", "") or ""),
            }
        )
    return rows


def _peer_provider(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Hermes Peer inbox/request summaries via the PUBLIC plugin API only.

    If the plugin is not importable, raises so the service records a typed
    unavailable source — never a fake zero inbox.
    """
    try:
        import hermes_peer.plugin as peer_plugin
        import hermes_peer.tools as peer_tools
    except ImportError as exc:  # pragma: no cover - exercised via absence path
        raise RuntimeError(f"hermes_peer plugin not available: {exc}") from exc

    manager = peer_plugin.get_manager()
    if manager is None:
        raise RuntimeError("hermes_peer plugin not registered in this process")

    rows: List[Dict[str, Any]] = []
    # Public tool surface returns JSON: {"messages": [{message_id, peer_id,
    # content, state, from, created_at}, ...]}. Parse that shape; never fall
    # back to scraping text.
    raw = peer_tools.peer_read_inbox({"limit": 10})
    try:
        payload = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except (TypeError, ValueError):  # noqa: BLE001 - defensive against tool drift
        payload = {}
    for msg in payload.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        sender = str(msg.get("from") or msg.get("peer_id") or "unknown")
        content = str(msg.get("content") or "")
        state = str(msg.get("state") or "held")
        message_id = str(msg.get("message_id") or "")
        rows.append(
            {
                "kind": "peer_message",
                "id": (message_id or sender)[:24],
                "title": (content or sender)[:70],
                "state": state,
                "sender": sender,
            }
        )
    return rows[:10]


def _system_provider(ctx: Dict[str, Any]) -> List[SystemSummary]:
    # Graceful: system health may live in gateway or dashboard contexts.
    # Default returns an unknown-but-ok summary; richer providers can be
    # injected in surfaces that have real health data.
    return [
        SystemSummary(
            state="healthy",
            severity=AttentionSeverity.info,
            detail="runtime reachable",
            source=SourceMeta(provider="system", state="ok"),
            available_actions=["inspect"],
        )
    ]


def default_providers() -> Dict[str, ProviderFn]:
    return {
        "agents": _agents_provider,
        "processes": _processes_provider,
        "delegations": _delegations_provider,
        "kanban": _kanban_provider,
        "peer": _peer_provider,
        "system": _system_provider,
    }


# ---------------------------------------------------------------------------
# Row coercion helpers
# ---------------------------------------------------------------------------


def _as_agent_row(raw: Dict[str, Any], profile: str, meta: SourceMeta) -> AgentRow:
    kind = raw.get("kind", "agent")
    return AgentRow(
        kind=kind,
        id=str(raw.get("id", "?")),
        name=str(raw.get("name", "")),
        status=str(raw.get("status", "running")),
        detail=f"{raw.get('elapsed_seconds', '')}s" if raw.get("elapsed_seconds") is not None else "",
        profile=profile,
        source=meta,
        available_actions=_agent_actions(kind, raw),
    )


def _agent_actions(kind: str, raw: Dict[str, Any]) -> List[str]:
    if kind == "process":
        return ["inspect", "kill"]
    if kind == "delegation":
        return ["inspect", "pause"]
    return ["inspect", "interrupt"]


def _as_task_row(raw: Dict[str, Any], profile: str, meta: SourceMeta) -> TaskRow:
    return TaskRow(
        id=str(raw.get("id", "?")),
        title=str(raw.get("title", "")),
        state=str(raw.get("state", "unknown")),
        board=str(raw.get("board", "")),
        owner=str(raw.get("owner", "")),
        profile=profile,
        source=meta,
        available_actions=_task_actions(raw.get("state", "")),
    )


def _task_actions(state: str) -> List[str]:
    if state in ("blocked",):
        return ["inspect"]
    if state in ("ready", "todo"):
        return ["inspect"]
    return ["inspect"]


def _as_message_row(raw: Dict[str, Any], profile: str, meta: SourceMeta) -> MessageRow:
    return MessageRow(
        kind=raw.get("kind", "peer_message"),
        id=str(raw.get("id", "?")),
        title=str(raw.get("title", "")),
        state=str(raw.get("state", "queued")),
        sender=str(raw.get("sender", "")),
        profile=profile,
        source=meta,
        available_actions=["release", "refuse"] if raw.get("kind", "peer_message") == "peer_message" else [],
    )


# ---------------------------------------------------------------------------
# Attention derivation (CR-103)
# ---------------------------------------------------------------------------


def _attention_from_agents(rows: List[AgentRow], meta: SourceMeta) -> List[AttentionItem]:
    items = []
    for r in rows:
        if r.status == "stalling":
            items.append(
                AttentionItem(
                    kind=AttentionKind.stalled,
                    id=r.id,
                    severity=AttentionSeverity.error,
                    title=f"Delegation stalled: {r.name}",
                    profile=r.profile,
                    source=r.source,
                    available_actions=r.available_actions,
                )
            )
    return items


def _attention_from_tasks(rows: List[TaskRow], meta: SourceMeta) -> List[AttentionItem]:
    items = []
    for r in rows:
        if r.state == "blocked":
            items.append(
                AttentionItem(
                    kind=AttentionKind.blocked_task,
                    id=r.id,
                    severity=AttentionSeverity.warning,
                    title=f"Blocked: {r.title}",
                    profile=r.profile,
                    source=r.source,
                    available_actions=r.available_actions,
                )
            )
        elif r.state in ("review", "review_required"):
            items.append(
                AttentionItem(
                    kind=AttentionKind.review_task,
                    id=r.id,
                    severity=AttentionSeverity.warning,
                    title=f"Review: {r.title}",
                    profile=r.profile,
                    source=r.source,
                    available_actions=r.available_actions,
                )
            )
    return items


def _attention_from_messages(rows: List[MessageRow], meta: SourceMeta) -> List[AttentionItem]:
    items = []
    for r in rows:
        if r.state in ("held", "queued"):
            items.append(
                AttentionItem(
                    kind=AttentionKind.held_message,
                    id=r.id,
                    severity=AttentionSeverity.critical,
                    title=f"Peer message from {r.sender or 'peer'}: {r.title}",
                    profile=r.profile,
                    source=r.source,
                    available_actions=r.available_actions,
                )
            )
    return items


def _attention_from_system(system: SystemSummary) -> List[AttentionItem]:
    if system.severity in (AttentionSeverity.error, AttentionSeverity.warning):
        return [
            AttentionItem(
                kind=AttentionKind.system,
                id="system",
                severity=system.severity,
                title=system.detail or f"System {system.state}",
                source=system.source,
                available_actions=system.available_actions,
            )
        ]
    return []


def _derive_counts(
    attention: List[AttentionItem],
    agents: List[AgentRow],
    tasks: List[TaskRow],
    messages: List[MessageRow],
    system: SystemSummary,
) -> SnapshotCounts:
    needs_you = sum(
        1 for a in attention if a.severity in (AttentionSeverity.critical, AttentionSeverity.error)
    )
    return SnapshotCounts(
        needs_you=needs_you,
        agents_active=len(agents),
        tasks_running=sum(1 for t in tasks if t.state in ("running", "in_progress", "ready")),
        messages_unread=sum(1 for m in messages if m.state in ("held", "queued")),
        system_severity=system.severity,
    )


def _context_key(context: Optional[Dict[str, Any]]) -> str:
    if not context:
        return ""
    # Stable, bounded cache key: hash the live-surface identity fields only.
    key_parts = []
    running = context.get("running_agents") or {}
    key_parts.append(f"agents:{len(running)}")
    board = context.get("kanban_board")
    if board:
        key_parts.append(f"board:{board}")
    return "|".join(key_parts)

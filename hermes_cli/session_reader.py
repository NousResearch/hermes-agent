"""Read-only, profile-bound view of the session store for plugins (``ctx.sessions``) and scripts
(``hermes sessions show --json``).

Every call opens the profile's ``state.db`` with ``mode=ro`` and returns plain JSON-safe dicts in a
documented projection, so a reader never holds a ``SessionDB``, never sees system prompts, model
config or billing endpoints, and never writes. Column names in the projection are the public
contract; the table layout behind them is not.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

SESSION_FIELDS = (
    "id", "title", "source", "created_source", "model", "billing_provider", "parent_session_id",
    "started_at", "ended_at", "end_reason", "message_count", "tool_call_count", "api_call_count",
    "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens",
    "estimated_cost_usd", "actual_cost_usd", "cost_status", "cwd", "git_branch", "git_repo_root",
    "profile_name", "pinned", "archived",
)
LISTED_FIELDS = ("preview", "last_active")
MESSAGE_FIELDS = ("id", "role", "content", "tool_name", "tool_call_id", "tool_calls", "timestamp", "finish_reason")
USAGE_FIELDS = (
    "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens",
    "api_call_count", "estimated_cost_usd", "actual_cost_usd",
)
_BOOL_FIELDS = ("pinned", "archived")


def _delegate_from(row: dict[str, Any]) -> Optional[str]:
    raw = row.get("model_config")
    cfg = raw if isinstance(raw, dict) else None
    if isinstance(raw, str) and raw:
        try:
            cfg = json.loads(raw)
        except ValueError:
            cfg = None
    value = cfg.get("_delegate_from") if isinstance(cfg, dict) else None
    return value if isinstance(value, str) and value else None


def project_session(row: dict[str, Any]) -> dict[str, Any]:
    """The public session projection: ``SESSION_FIELDS`` plus ``delegate_from`` (the parent a
    subagent session was spawned by, else None) and, for listed rows, ``preview``/``last_active``."""
    out = {name: row.get(name) for name in SESSION_FIELDS}
    for name in _BOOL_FIELDS:
        out[name] = bool(out[name])
    out["delegate_from"] = _delegate_from(row)
    for name in LISTED_FIELDS:
        if name in row:
            out[name] = row[name]
    return out


def project_message(row: dict[str, Any]) -> dict[str, Any]:
    return {name: row.get(name) for name in MESSAGE_FIELDS}


class SessionReader:
    """Read-only session queries against one Hermes home's ``state.db``."""

    def __init__(self, home: Path) -> None:
        self.home = Path(home)

    @property
    def db_path(self) -> Path:
        return self.home / "state.db"

    @contextmanager
    def _db(self) -> Iterator[Any]:
        """Open ``mode=ro`` under this home's scope (store readers consult the profile's config);
        yields None when the profile never created a store."""
        from hermes_cli.plugins_loader import _plugin_home_scope
        from hermes_state import SessionDB

        if not self.db_path.exists():
            yield None
            return
        with _plugin_home_scope(self.home):
            db = SessionDB(db_path=self.db_path, read_only=True)
            try:
                yield db
            finally:
                db.close()

    def get(self, session_id: str) -> Optional[dict[str, Any]]:
        """One session by id or unambiguous id prefix, else None."""
        with self._db() as db:
            row = _resolved_row(db, session_id)
            return project_session(row) if row else None

    def recent(self, *, source: Optional[str] = None, limit: int = 20,
               include_archived: bool = False) -> list[dict[str, Any]]:
        """Top-level sessions, most recently started first (compression chains collapse to their
        tip, as in ``hermes sessions list``); each row adds ``preview`` and ``last_active``. ``source``
        means what ``hermes sessions list --source`` means: ``cli`` also matches a gateway row stored
        as ``local`` that the CLI created."""
        from hermes_cli.gateway_session_store import SurfaceLabelReader

        with self._db() as db:
            if db is None:
                return []
            rows = SurfaceLabelReader(db).list_sessions_rich(source=source, limit=max(0, int(limit)), include_archived=include_archived)
            return [project_session(r) for r in rows]

    def lineage(self, session_id: str) -> list[str]:
        """Compression lineage of one conversation, root first through the live tip."""
        with self._db() as db:
            return db.get_compression_lineage(session_id) if db else []

    def subagents(self, session_id: str) -> list[dict[str, Any]]:
        """Every delegate-subagent session spawned under the conversation (any segment of its
        compression lineage), recursively, with their own compression continuations; rebuild the
        tree from each row's ``delegate_from``/``parent_session_id``. Ordered by start time."""
        with self._db() as db:
            return _subagents(db, session_id) if db else []

    def usage(self, session_id: str, *, include_subagents: bool = False) -> Optional[dict[str, Any]]:
        """Token and cost totals for the whole conversation (its compression lineage), optionally
        with every session ``subagents()`` lists; ``session_ids`` names the rows summed."""
        with self._db() as db:
            return _usage(db, session_id, include_subagents) if db else None

    def detail(self, session_id: str) -> Optional[dict[str, Any]]:
        """``{"session", "lineage", "subagents", "usage"}`` for one id or unique prefix, read over one
        open of the store; None when no session matches or it is deleted between the reads."""
        with self._db() as db:
            row = _resolved_row(db, session_id)
            if row is None:
                return None
            sid = row["id"]
            detail = {"session": project_session(row), "lineage": db.get_compression_lineage(sid),
                      "subagents": _subagents(db, sid), "usage": _usage(db, sid, True)}
            # Ids are never reused, so a row still present here was present for every read above.
            return detail if db.get_session(sid) else None

    def messages(self, session_id: str, *, limit: Optional[int] = 50, latest: bool = True,
                 include_ancestors: bool = False) -> list[dict[str, Any]]:
        """Active transcript rows in chronological order (``latest`` pages back from the newest);
        ``include_ancestors`` reads across the compression lineage."""
        with self._db() as db:
            if db is None:
                return []
            rows = db.get_messages(session_id, limit=limit, latest=latest, include_ancestors=include_ancestors)
            return [project_message(r) for r in rows]

    def latest_todo(self, session_id: str) -> Optional[list[dict[str, Any]]]:
        """The session's latest ``todo`` list (across its lineage), else None."""
        with self._db() as db:
            raw = db.get_latest_todo_result(session_id) if db else None
        try:
            todos = json.loads(raw).get("todos") if raw else None
        except (ValueError, AttributeError):
            return None
        return todos if isinstance(todos, list) else None


def _resolved_row(db: Any, session_id: str) -> Optional[dict[str, Any]]:
    resolved = db.resolve_session_id(session_id) if db and session_id else None
    return db.get_session(resolved) if resolved else None


def _subagents(db: Any, session_id: str) -> list[dict[str, Any]]:
    rows = [db.get_session(sid) for sid in _subagent_ids(db, _lineage_ids(db, session_id))]
    return sorted((project_session(r) for r in rows if r), key=lambda r: r["started_at"] or 0)


def _usage(db: Any, session_id: str, include_subagents: bool) -> Optional[dict[str, Any]]:
    if not db.get_session(session_id):
        return None
    ids = _lineage_ids(db, session_id)
    if include_subagents:
        ids += _subagent_ids(db, ids)
    rows = [r for r in (db.get_session(sid) for sid in ids) if r]
    totals: dict[str, Any] = {name: 0 for name in USAGE_FIELDS}
    for name in ("estimated_cost_usd", "actual_cost_usd"):
        totals[name] = None if all(r.get(name) is None for r in rows) else 0.0
    for row in rows:
        for name in USAGE_FIELDS:
            if row.get(name) is not None:
                totals[name] += row[name]
    return {"session_ids": [r["id"] for r in rows], **totals}


def _lineage_ids(db: Any, session_id: str) -> list[str]:
    return db.get_compression_lineage(session_id) or [session_id]


def _subagent_ids(db: Any, lineage: list[str]) -> list[str]:
    """Delegates under *lineage* and the compression continuations of each, closed over each other
    to a fixed point (the walk ``delete_targets`` runs): a delegate's continuation may carry no
    ``_delegate_from`` marker, and its own delegates hang off whichever segment spawned them."""
    from hermes_state_sessions import _collect_delegate_child_ids, _expand_compression_lineage_ids

    roots = set(lineage)
    seen, found = set(roots), []
    frontier = list(lineage)
    with db._read_ctx() as conn:
        while frontier:
            delegates = [sid for sid in frontier if sid not in roots]
            step = _collect_delegate_child_ids(conn, frontier) + _expand_compression_lineage_ids(conn, delegates)
            frontier = [sid for sid in dict.fromkeys(step) if sid not in seen]
            seen.update(frontier)
            found += frontier
    return found

"""Persistent, Hermes-owned dev-todo store keyed by a stable ``key``.

Journal proposes dev todos every morning into
``~/.hermes/contracts/journal_brief.latest.json``, keyed by a per-day ``id``
(e.g. ``jrl-2026-07-15-01``) that changes daily, so nothing proposed there can
ever be marked "done" — tomorrow it is a new id. This module is the fix: a
SQLite table keyed by a *stable* ``key`` (added on the journal side in
separate work) that Hermes owns across days. The morning run reconciles the
day's contract into this store (:func:`upsert_from_contract`); Discord /
journal close a todo out (:func:`close_todo`); ``OPEN_KEYS`` /
``CLOSED_KEYS`` injection back into journal's prompt comes from
:func:`get_open_keys` / :func:`get_closed_keys`.

DB file: ``$HERMES_HOME/todos.db`` (WAL). Every public function here is safe
to call repeatedly and never raises on malformed caller data — it validates
defensively and returns empty/error results instead of crashing the
scheduled path.

No LLM/agent-client imports: stdlib + sqlite3 + hermes_cli.sqlite_util only.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import sqlite3
from pathlib import Path

from hermes_cli.sqlite_util import add_column_if_missing, write_txn

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _hermes_home() -> Path:
    home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(home)


def todos_db_path() -> Path:
    """The ``todos.db`` path (``$HERMES_HOME/todos.db``)."""
    return _hermes_home() / "todos.db"


def _audit_log_path() -> Path:
    return _hermes_home() / "todo-audit.log"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS todos (
    key             TEXT PRIMARY KEY,
    text            TEXT NOT NULL,
    category        TEXT,
    priority        TEXT,
    source_dates    TEXT,
    recurring       INTEGER NOT NULL DEFAULT 0,
    confidence      REAL,
    status          TEXT NOT NULL DEFAULT 'open',
    snoozed_until   TEXT,
    first_seen      TEXT NOT NULL,
    last_seen       TEXT NOT NULL,
    closed_at       TEXT,
    origin          TEXT NOT NULL DEFAULT 'journal'
);

CREATE INDEX IF NOT EXISTS idx_todos_status ON todos(status);
"""

# Columns that may be introduced after v1; re-applied idempotently on every
# open (mirrors hermes_cli.projects_db._migrate_add_optional_columns) so an
# older todos.db upgrades in place instead of needing a hand-run migration.
_OPTIONAL_TODO_COLUMNS: dict[str, str] = {
    "category": "category TEXT",
    "priority": "priority TEXT",
    "source_dates": "source_dates TEXT",
    "recurring": "recurring INTEGER NOT NULL DEFAULT 0",
    "confidence": "confidence REAL",
    "snoozed_until": "snoozed_until TEXT",
    "closed_at": "closed_at TEXT",
    "origin": "origin TEXT NOT NULL DEFAULT 'journal'",
}


def _migrate_todos(conn: sqlite3.Connection) -> None:
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(todos)")}
    for col, ddl in _OPTIONAL_TODO_COLUMNS.items():
        if col not in cols:
            add_column_if_missing(conn, "todos", col, ddl)


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    """Open (and initialize if needed) the todos DB.

    WAL mode, row_factory=sqlite3.Row. Schema init is idempotent
    (``CREATE TABLE IF NOT EXISTS`` + additive migrations) so it is safe to
    call on every operation.
    """
    path = db_path if db_path is not None else todos_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError as exc:
        _log.warning("todos.db: could not enable WAL (%s); continuing with default mode", exc)
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(_SCHEMA_SQL)
    _migrate_todos(conn)
    return conn


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _coerce_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_todo_item(item: object) -> dict | None:
    """Normalize one journal-contract todo item.

    Returns ``None`` for anything that cannot be classified (not a dict, or
    missing/empty ``key``) — the caller is expected to have already filtered
    to items with a stable ``key``, this is just the defensive backstop so a
    malformed entry cannot crash the morning run.
    """
    if not isinstance(item, dict):
        return None
    key = item.get("key")
    if not isinstance(key, str) or not key.strip():
        return None
    text = item.get("text") or item.get("content") or ""
    source_dates = item.get("source_dates")
    if not isinstance(source_dates, list):
        source_dates = []
    source_dates = [d for d in source_dates if isinstance(d, str)]
    return {
        "key": key,
        "text": _coerce_str(text) or "",
        "category": _coerce_str(item.get("category")),
        "priority": _coerce_str(item.get("priority")),
        "source_dates": source_dates,
        "recurring": bool(item.get("recurring")),
        "confidence": _coerce_float(item.get("confidence")),
    }


def _dedupe_by_key(items: list[dict]) -> list[dict]:
    """Last-write-wins de-dupe, preserving first-seen order."""
    by_key: dict[str, dict] = {}
    for item in items:
        by_key[item["key"]] = item
    return list(by_key.values())


def _load_existing(conn: sqlite3.Connection, keys: list[str]) -> dict[str, sqlite3.Row]:
    if not keys:
        return {}
    placeholders = ",".join("?" for _ in keys)
    rows = conn.execute(
        f"SELECT * FROM todos WHERE key IN ({placeholders})", keys
    ).fetchall()
    return {row["key"]: row for row in rows}


def _empty_plan() -> dict:
    return {
        "inserted": [],
        "refreshed": [],
        "reopened": [],
        "ignored_snoozed": [],
        "ignored_closed": [],
    }


def _classify(
    normalized_todos: list[dict], for_date: str, existing: dict[str, sqlite3.Row]
) -> tuple[dict, dict[str, tuple[str, dict]]]:
    """Pure classification: the single place the branching rules live.

    Returns ``(plan, actions)`` where ``plan`` is the summary dict returned
    to callers and ``actions`` maps key -> ("insert"|"refresh"|"reopen",
    normalized_item) for the keys that need a write.
    """
    plan = _empty_plan()
    actions: dict[str, tuple[str, dict]] = {}
    for item in normalized_todos:
        key = item["key"]
        row = existing.get(key)
        if row is None:
            plan["inserted"].append(key)
            actions[key] = ("insert", item)
            continue
        status = row["status"]
        if status == "open":
            plan["refreshed"].append(key)
            actions[key] = ("refresh", item)
        elif status == "snoozed":
            snoozed_until = row["snoozed_until"]
            if snoozed_until and snoozed_until <= for_date:
                plan["reopened"].append(key)
                actions[key] = ("reopen", item)
            else:
                plan["ignored_snoozed"].append(key)
        else:
            # 'done' / 'dismissed' / any unrecognized terminal status —
            # belt-and-braces backstop, caller is also expected to have
            # excluded closed keys upstream via CLOSED_KEYS injection.
            plan["ignored_closed"].append(key)
    return plan, actions


def _apply_actions(
    conn: sqlite3.Connection, actions: dict[str, tuple[str, dict]], for_date: str
) -> None:
    for key, (action, item) in actions.items():
        source_dates_json = json.dumps(item["source_dates"])
        recurring_int = 1 if item["recurring"] else 0
        if action == "insert":
            conn.execute(
                "INSERT INTO todos "
                "(key, text, category, priority, source_dates, recurring, "
                " confidence, status, snoozed_until, first_seen, last_seen, "
                " closed_at, origin) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'open', NULL, ?, ?, NULL, 'journal')",
                (
                    key,
                    item["text"],
                    item["category"],
                    item["priority"],
                    source_dates_json,
                    recurring_int,
                    item["confidence"],
                    for_date,
                    for_date,
                ),
            )
        elif action == "refresh":
            conn.execute(
                "UPDATE todos SET text=?, category=?, priority=?, source_dates=?, "
                "recurring=?, confidence=?, last_seen=? WHERE key=?",
                (
                    item["text"],
                    item["category"],
                    item["priority"],
                    source_dates_json,
                    recurring_int,
                    item["confidence"],
                    for_date,
                    key,
                ),
            )
        elif action == "reopen":
            conn.execute(
                "UPDATE todos SET status='open', snoozed_until=NULL, text=?, "
                "category=?, priority=?, source_dates=?, recurring=?, confidence=?, "
                "last_seen=? WHERE key=?",
                (
                    item["text"],
                    item["category"],
                    item["priority"],
                    source_dates_json,
                    recurring_int,
                    item["confidence"],
                    for_date,
                    key,
                ),
            )


# ---------------------------------------------------------------------------
# Public API — morning-run reconciliation
# ---------------------------------------------------------------------------


def plan_upsert_from_contract(todos: list[dict], for_date: str) -> dict:
    """Read-only classification of ``todos`` against the store — for --dry-run.

    Same branching rules as :func:`upsert_from_contract`, no writes. Returns
    ``{"inserted": [...], "refreshed": [...], "reopened": [...],
    "ignored_snoozed": [...], "ignored_closed": [...]}``.
    """
    normalized = _dedupe_by_key(
        [n for n in (_normalize_todo_item(t) for t in (todos or [])) if n is not None]
    )
    for_date = str(for_date or "")
    conn = connect()
    try:
        existing = _load_existing(conn, [n["key"] for n in normalized])
        plan, _actions = _classify(normalized, for_date, existing)
        return plan
    finally:
        conn.close()


def upsert_from_contract(todos: list[dict], for_date: str) -> dict:
    """Morning-run entrypoint: reconcile a journal contract's ``todos[]``.

    ``todos`` items are assumed to already carry a stable ``key`` (that
    validation is the caller's job); items without one are skipped
    defensively rather than raising. See module docstring / :func:`_classify`
    for the per-key branching rules. Returns the same plan shape as
    :func:`plan_upsert_from_contract`.
    """
    normalized = _dedupe_by_key(
        [n for n in (_normalize_todo_item(t) for t in (todos or [])) if n is not None]
    )
    for_date = str(for_date or "")
    conn = connect()
    try:
        existing = _load_existing(conn, [n["key"] for n in normalized])
        plan, actions = _classify(normalized, for_date, existing)
        if actions:
            with write_txn(conn):
                _apply_actions(conn, actions, for_date)
        return plan
    finally:
        conn.close()


def reopen_expired_snoozes(today: str) -> list[str]:
    """Flip any snoozed row whose ``snoozed_until <= today`` back to open.

    Independent of :func:`upsert_from_contract` (which only reopens a
    snoozed key when it reappears in that day's contract) — this covers
    snoozed todos that do NOT reappear tomorrow but must still auto-reopen.
    Returns the list of keys reopened.
    """
    today = str(today or "")
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT key FROM todos WHERE status='snoozed' AND snoozed_until IS NOT NULL "
            "AND snoozed_until <= ?",
            (today,),
        ).fetchall()
        keys = [row["key"] for row in rows]
        if keys:
            with write_txn(conn):
                conn.execute(
                    "UPDATE todos SET status='open', snoozed_until=NULL "
                    "WHERE status='snoozed' AND snoozed_until IS NOT NULL AND snoozed_until <= ?",
                    (today,),
                )
        return keys
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Public API — reads for Discord / journal injection
# ---------------------------------------------------------------------------

_PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def get_open_todos() -> list[dict]:
    """Open (non-snoozed, non-closed) todos with the full display field set.

    Same rows/ordering as :func:`get_open_keys` (priority high/medium/low,
    unknown last; ``last_seen`` descending within a priority) but includes
    ``priority``, ``recurring``, ``source_dates`` (parsed back into a list)
    and ``first_seen`` — the fields the morning-brief renderer needs for the
    glyph/recency columns. :func:`get_open_keys` is a thin projection of
    this down to ``{"key", "text"}`` for the journal ``OPEN_KEYS`` injection.
    """
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT key, text, priority, recurring, source_dates, first_seen, last_seen "
            "FROM todos WHERE status='open'"
        ).fetchall()
    finally:
        conn.close()
    items: list[dict] = []
    for r in rows:
        try:
            source_dates = json.loads(r["source_dates"]) if r["source_dates"] else []
        except (TypeError, ValueError):
            source_dates = []
        if not isinstance(source_dates, list):
            source_dates = []
        items.append(
            {
                "key": r["key"],
                "text": r["text"],
                "priority": r["priority"],
                "recurring": bool(r["recurring"]),
                "source_dates": [d for d in source_dates if isinstance(d, str)],
                "first_seen": r["first_seen"],
                "last_seen": r["last_seen"],
            }
        )
    items.sort(key=lambda d: d["last_seen"] or "", reverse=True)
    items.sort(key=lambda d: _PRIORITY_ORDER.get((d["priority"] or "").lower(), 3))
    return items


def get_open_keys() -> list[dict]:
    """Open (non-snoozed, non-closed) todos as ``[{"key":..., "text":...}]``.

    Ordered by priority (high, medium, low; unknown priorities last) then
    ``last_seen`` descending within a priority. This is what gets exported
    for the Discord select menu / journal's ``OPEN_KEYS`` injection.
    """
    return [{"key": d["key"], "text": d["text"]} for d in get_open_todos()]


def get_closed_keys() -> list[str]:
    """All keys with status in ('done', 'dismissed') — for CLOSED_KEYS injection."""
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT key FROM todos WHERE status IN ('done', 'dismissed')"
        ).fetchall()
    finally:
        conn.close()
    return [r["key"] for r in rows]


# ---------------------------------------------------------------------------
# Public API — closing a todo out
# ---------------------------------------------------------------------------

_ACTION_TO_STATUS = {"done": "done", "dismiss": "dismissed", "snooze": "snoozed"}


def close_todo(
    key: str, action: str, source: str, snooze_until: str | None = None
) -> dict:
    """Close (done/dismiss) or snooze a todo. Never raises.

    Idempotent: re-closing an already-closed key with the *same* action is a
    no-op that still returns success; re-closing with a *different* action
    updates to the new terminal state. Closing an unknown key returns
    ``{"ok": False, "error": "unknown key: <key>"}``. Every attempt
    (including no-op repeats) is appended to the audit log via
    :func:`record_audit`.
    """
    key = str(key or "")
    action = str(action or "")
    if action not in _ACTION_TO_STATUS:
        return {"ok": False, "error": f"invalid action: {action}"}
    if action == "snooze" and not snooze_until:
        return {"ok": False, "error": "snooze requires snooze_until"}

    new_status = _ACTION_TO_STATUS[action]
    conn = connect()
    try:
        row = conn.execute("SELECT 1 FROM todos WHERE key=?", (key,)).fetchone()
        if row is None:
            return {"ok": False, "error": f"unknown key: {key}"}

        now = _now_iso()
        with write_txn(conn):
            if new_status == "snoozed":
                conn.execute(
                    "UPDATE todos SET status='snoozed', snoozed_until=?, closed_at=NULL "
                    "WHERE key=?",
                    (snooze_until, key),
                )
            else:
                conn.execute(
                    "UPDATE todos SET status=?, snoozed_until=NULL, closed_at=? WHERE key=?",
                    (new_status, now, key),
                )
    finally:
        conn.close()

    record_audit(key, action, source, timestamp=now)
    return {"ok": True, "status": new_status}


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


def record_audit(key: str, action: str, source: str, timestamp: str | None = None) -> None:
    """Append one JSONL entry to ``$HERMES_HOME/todo-audit.log``.

    Never raises on write failure — logs a WARNING instead so a log-write
    hiccup cannot block the caller's user-facing response.
    """
    entry = {
        "key": key,
        "action": action,
        "source": source,
        "ts": timestamp or _now_iso(),
    }
    line = json.dumps(entry, separators=(",", ":")) + "\n"
    path = _audit_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as exc:
        _log.warning("todo audit log write failed: %s", exc)

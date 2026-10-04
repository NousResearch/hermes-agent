"""Durable origin references: which board tasks a conversation ordered.

A navigation index, never a source of truth. Each ``(origin session, board, task)`` triple is one
independent ``state_meta`` key in the OWNING profile's ``state.db`` (``kanban_origin:<session>:<board>:<task>``),
so concurrent writers never read-merge-write a shared value. The board database stays authoritative for
every task fact: the reader below re-opens only the boards the index names, read-only and without
init/migration, and re-verifies each ref against the board's own ``tasks.session_id`` / ``tui``
subscription before reporting it linked.

Writers (best-effort — an index failure never changes task creation or subscription semantics):

* ``index_created_task`` — ``kanban_create``, for the session verified in the calling profile's
  ``state.db``, plus the ``tui`` subscriptions the new task carries (a worker-created task inherits its
  owner's; those encode the ORIGINAL owner's profile and are verified in THAT profile's store, never the
  worker's).
* ``index_subscription`` — ``add_notify_sub`` for ``platform="tui"`` with an explicit ``notifier_profile``.

Tasks created before this index existed are not discovered and nothing sweeps boards to backfill them:
re-subscribing the task (``hermes kanban notify-subscribe``) indexes it. The board slug comes from the
connection's own file, never from the ambient current board; a connection that is not a known board file
(e.g. a custom ``HERMES_KANBAN_DB``) is simply not indexed.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import closing, nullcontext
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

KEY_PREFIX = "kanban_origin:"

# Reader bounds. Every bound is reported through ``truncated``; none silently skips an indexed board.
MAX_SEED_SESSIONS = 64
MAX_LINEAGE_IDS = 256
MAX_REFS = 200

_TERMINAL = ("done", "archived")


def origin_key(session_id: str, board: str, task_id: str) -> str:
    return f"{KEY_PREFIX}{session_id}:{board}:{task_id}"


def _parse_key(key: str, session_id: str) -> Optional[tuple[str, str]]:
    """``(board, task_id)`` when *key* belongs to exactly *session_id* (a prefix scan for ``a`` also
    sees ``a:b``'s keys; board slugs and task ids never contain ``:``)."""
    head, _, task_id = key[len(KEY_PREFIX):].rpartition(":")
    sid, sep, board = head.rpartition(":")
    return (board, task_id) if sep and sid == session_id and board and task_id else None


def _meta_fields(value: Any) -> tuple[int, str]:
    """``(indexed_at, source)`` from a ref's value. The value only annotates a ref that the key already
    proves; anything unreadable or of another JSON type degrades to ``(0, "")`` and the ref is kept."""
    try:
        meta = json.loads(value)
    except (TypeError, ValueError):
        return 0, ""
    if not isinstance(meta, dict):
        return 0, ""
    at, src = meta.get("at"), meta.get("src")
    return (at if isinstance(at, int) and not isinstance(at, bool) and at > 0 else 0), (src if isinstance(src, str) else "")


# --- writers -----------------------------------------------------------------

def board_slug_for_connection(conn: sqlite3.Connection) -> Optional[str]:
    """The slug of the board *conn* is actually open on, or None when its file is not a known board."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    raw = kbc._main_db_file(conn)
    if not raw:
        return None
    try:
        path = Path(raw).resolve()
        if path == (kb.kanban_home() / "kanban.db").resolve():
            return kb.DEFAULT_BOARD
        if path.name == "kanban.db" and path.parent.parent == kb.boards_root().resolve():
            return kb._normalize_board_slug(path.parent.name)
    except (OSError, ValueError):
        pass
    return None


def _owner_state_db(profile: str) -> Optional[Path]:
    """An EXISTING profile's ``state.db``. A missing/invalid/tombstoned named profile yields None —
    never the default profile's store."""
    from hermes_cli.profiles import get_profile_dir, normalize_profile_name, profile_exists

    try:
        canon = normalize_profile_name(profile)
        if not profile_exists(canon):
            return None
        path = get_profile_dir(canon) / "state.db"
    except (OSError, ValueError):
        return None
    return path if path.exists() else None


def _record(state_db: Path, session_id: str, board: str, task_id: str, source: str) -> bool:
    """Index one ref when *session_id* exists in *state_db*; False (never raises) otherwise."""
    from hermes_state_registry import acquire, release_or_close

    db = None
    try:
        db = acquire(state_db)
        if db.get_session(session_id) is None:
            return False
        db.set_meta(
            origin_key(session_id, board, task_id),
            json.dumps({"at": int(time.time()), "src": source}, separators=(",", ":")),
        )
        return True
    except Exception:
        logger.debug("kanban origin index write failed for %s on %s", task_id, board, exc_info=True)
        return False
    finally:
        if db is not None:
            release_or_close(db)


def _index_subscriptions(
    conn: sqlite3.Connection, task_id: str, board: str, extra_session_id: Optional[str],
) -> None:
    from hermes_cli import kanban_db_notify as kbn

    for sub in kbn.list_notify_subs(conn, task_id):
        profile = (sub.get("notifier_profile") or "").strip()
        if (sub.get("platform") or "").lower() != "tui" or not profile or not sub.get("chat_id"):
            continue
        state_db = _owner_state_db(profile)
        if state_db is None:
            continue
        for session_id in dict.fromkeys(filter(None, (sub["chat_id"], extra_session_id))):
            _record(state_db, session_id, board, task_id, "sub")


def index_created_task(
    conn: sqlite3.Connection, task_id: str, *, session_id: Optional[str], inherited_session: bool,
) -> None:
    """Index a task ``kanban_create`` just wrote. ``session_id`` is the stamped origin;
    ``inherited_session`` means it came from the creating worker's own task (not verified in the worker's
    store), so it is indexed only through the owner profiles the inherited subscriptions name."""
    try:
        board = board_slug_for_connection(conn)
        if not board:
            return
        if session_id and not inherited_session:
            from hermes_constants import get_hermes_home

            state_db = get_hermes_home() / "state.db"
            if state_db.exists():
                _record(state_db, session_id, board, task_id, "create")
        _index_subscriptions(conn, task_id, board, session_id)
    except Exception:
        logger.debug("kanban origin indexing failed for %s", task_id, exc_info=True)


def index_subscription(
    conn: sqlite3.Connection, *, task_id: str, platform: str, chat_id: str, notifier_profile: Optional[str],
) -> None:
    """Index an explicit ``tui`` subscription under its ``notifier_profile``'s store."""
    profile = (notifier_profile or "").strip()
    if platform.lower() != "tui" or not profile or not chat_id:
        return
    try:
        board = board_slug_for_connection(conn)
        state_db = _owner_state_db(profile) if board else None
        if board and state_db is not None:
            _record(state_db, chat_id, board, task_id, "sub")
    except Exception:
        logger.debug("kanban origin subscription indexing failed for %s", task_id, exc_info=True)


# --- reader ------------------------------------------------------------------

_CLOCK_SKEW = 1


def _wall_time(value: Any, now: int) -> Optional[int]:
    """A plausible wall-clock second: an int that is neither negative nor in the future."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > now + _CLOCK_SKEW:
        return None
    return value


def classify_activity(
    task: Any, run: Optional[dict], spawned_at: Any, open_parents: int, now: int, claim_window: int,
) -> tuple[str, str]:
    """``(activity, evidence)`` for one task from the board's own rows.

    Only evidence belonging to the CURRENT run counts as execution: its heartbeat, or its spawn (a pid
    on the open run plus the ``spawned``/``worker_registered`` event time — ``worker_started_at`` is a
    process-identity fingerprint, never a time). A task-level heartbeat left by an earlier run, a live
    claim alone (a reservation), and timestamps that are negative or in the future prove nothing. A
    heartbeat or spawn older than the claim window is ``stale``; no usable evidence is ``unknown``.
    """
    status = task.status
    if status == "running":
        started = _wall_time((run or {}).get("started_at"), now)
        open_run = run is not None and run.get("ended_at") is None
        beats = [_wall_time((run or {}).get("last_heartbeat_at"), now)] if open_run else []
        task_beat = _wall_time(task.last_heartbeat_at, now)
        if open_run and started is not None and task_beat is not None and task_beat >= started:
            beats.append(task_beat)  # the task-level beat is only this run's if it postdates the claim
        beat = max((b for b in beats if b is not None), default=None)
        spawn = _wall_time(spawned_at, now) if open_run and run.get("worker_pid") else None
        if beat is not None:
            return ("background", "heartbeat") if now - beat <= claim_window else ("stale", "heartbeat_old")
        if spawn is not None:
            return ("background", "spawn") if now - spawn <= claim_window else ("stale", "spawn_old")
        claim = _wall_time(task.claim_expires, now + claim_window + _CLOCK_SKEW)
        if claim is not None and claim > now:
            return "reserved", "claim"
        return "unknown", "none"
    if status == "blocked":
        if task.block_kind == "needs_input":
            return "needs-input", "block_kind"
        if task.block_kind == "dependency":
            return "waiting", "block_kind"
        return "blocked", "block_kind"
    if status == "review":
        return "review", "status"
    if status == "todo":
        return ("waiting", "parents") if open_parents else ("queued", "status")
    if status in ("ready", "triage", "scheduled"):
        return "queued", "status"
    if status in _TERMINAL:
        return status, "status"
    return "unknown", "status"


def _open_readonly(board: str) -> tuple[Optional[sqlite3.Connection], str]:
    """Read-only handle on an existing board file — no init, no migration, no file creation."""
    from hermes_cli import kanban_db as kb

    try:
        path = kb.kanban_db_path(board=board)
        if not path.exists():
            return None, "board_missing"
        conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=2.0)
        conn.row_factory = sqlite3.Row
        return conn, "ok"
    except (OSError, ValueError, sqlite3.Error):
        return None, "board_unreadable"


def _task_summary(task: Any, activity: str, evidence: str) -> dict[str, Any]:
    return {
        "id": task.id, "title": task.title, "status": task.status, "assignee": task.assignee,
        "block_kind": task.block_kind, "created_at": task.created_at, "completed_at": task.completed_at,
        "last_heartbeat_at": task.last_heartbeat_at, "claim_expires": task.claim_expires,
        "current_run_id": task.current_run_id, "activity": activity, "activity_evidence": evidence,
    }


def _read_ref(
    conn: sqlite3.Connection, task_id: str, chain: frozenset, profile: str, now: int, claim_window: int,
) -> tuple[str, Optional[dict]]:
    from hermes_cli import kanban_db as kb

    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if row is None:
        return "task_missing", None
    task = kb.Task.from_row(row)
    linked = task.session_id in chain
    if not linked:
        marks = ",".join("?" * len(chain))
        linked = conn.execute(
            f"SELECT 1 FROM kanban_notify_subs WHERE task_id = ? AND LOWER(platform) = 'tui' "
            f"AND chat_id IN ({marks}) AND LOWER(COALESCE(notifier_profile, '')) = ? LIMIT 1",
            (task_id, *sorted(chain), profile),
        ).fetchone() is not None
    if not linked:
        return "not_associated", None
    run, spawned_at = None, None
    if task.current_run_id:
        found = conn.execute(
            "SELECT worker_pid, started_at, last_heartbeat_at, ended_at FROM task_runs WHERE id = ?",
            (task.current_run_id,)).fetchone()
        run = dict(found) if found else None
        if run:
            spawned_at = conn.execute(
                "SELECT MAX(created_at) FROM task_events WHERE run_id = ? AND kind IN ('spawned', 'worker_registered')",
                (task.current_run_id,)).fetchone()[0]
    open_parents = conn.execute(
        "SELECT COUNT(*) FROM task_links l JOIN tasks p ON p.id = l.parent_id "
        "WHERE l.child_id = ? AND p.status NOT IN ('done', 'archived')", (task_id,)).fetchone()[0]
    return "ok", _task_summary(task, *classify_activity(task, run, spawned_at, int(open_parents), now, claim_window))


def origin_tasks(session_ids: Iterable[str], *, now: Optional[int] = None) -> dict[str, Any]:
    """Linked tasks for the conversations *session_ids* name, resolved in the CURRENT profile scope.

    Lineage is expanded here from this profile's own ``state.db`` (the caller's aliases are only seeds),
    so an id that exists in another profile contributes nothing. Reads only indexed boards.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli.profiles import current_profile_name
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB

    now = int(time.time()) if now is None else now
    profile = (current_profile_name("default") or "default").strip().lower()
    seeds = list(dict.fromkeys(s.strip() for s in session_ids if isinstance(s, str) and s.strip()))
    truncated = {"sessions": len(seeds) > MAX_SEED_SESSIONS, "lineage": False, "refs": False, "total_refs": 0}
    seeds = seeds[:MAX_SEED_SESSIONS]
    result: dict[str, Any] = {
        "profile": profile, "now": now, "refs": [], "unknown_sessions": [], "truncated": truncated}

    state_db = get_hermes_home() / "state.db"
    if not state_db.exists():
        result["unknown_sessions"] = seeds
        return result

    chains: dict[str, frozenset] = {}
    refs: list[tuple[int, str, str, str, str]] = []
    # Read-only attach: a GET never creates, migrates or write-locks a profile's store. A store that
    # cannot be read raises (the route answers 5xx, which the UI shows as unavailable) — never an empty list.
    db = SessionDB(db_path=state_db, read_only=True)
    try:
        for seed in seeds:
            if db.get_session(seed) is None:
                result["unknown_sessions"].append(seed)
                continue
            chain = frozenset(db.get_compression_lineage(seed) or [seed]) | {seed}
            for sid in chain:
                chains[sid] = chain
        if len(chains) > MAX_LINEAGE_IDS:
            truncated["lineage"] = True
        for sid in sorted(chains)[:MAX_LINEAGE_IDS]:
            for key, value in db.list_meta_prefix(f"{KEY_PREFIX}{sid}:"):
                parsed = _parse_key(key, sid)
                if parsed:
                    at, src = _meta_fields(value)
                    refs.append((at, sid, parsed[0], parsed[1], src))
    finally:
        db.close()

    newest: dict[tuple[str, str], tuple[int, str, str, str, str]] = {}
    for ref in sorted(refs, reverse=True):
        newest.setdefault((ref[2], ref[3]), ref)
    ordered = sorted(newest.values(), reverse=True)
    truncated["total_refs"] = len(ordered)
    truncated["refs"] = len(ordered) > MAX_REFS
    ordered = ordered[:MAX_REFS]

    claim_window = kb._resolve_claim_ttl_seconds()
    by_board: dict[str, list] = {}
    for ref in ordered:
        by_board.setdefault(ref[2], []).append(ref)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for board, board_refs in by_board.items():
        conn, state = _open_readonly(board)
        with closing(conn) if conn is not None else nullcontext():
            for at, sid, _board, task_id, src in board_refs:
                evidence, task = state, None
                if conn is not None:
                    try:
                        evidence, task = _read_ref(conn, task_id, chains[sid], profile, now, claim_window)
                    except sqlite3.Error:
                        evidence = "board_unreadable"
                out[(board, task_id)] = {
                    "origin_session_id": sid, "board": board, "task_id": task_id, "indexed_at": at,
                    "source": src, "evidence": evidence, "task": task}
    result["refs"] = [out[(r[2], r[3])] for r in ordered]
    return result

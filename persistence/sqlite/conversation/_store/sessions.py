"""Session rows: create, end and reopen, metadata and model config, lifecycle flags, listing and counting, delete."""


import json
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence
from session_rules.activity import ActivityProvenance, bound_activity_description, normalize_activity_provenance
from hermes_state_errors import SessionActiveWriteGuardError
from hermes_state_common import _shape_preview
from persistence.sqlite._holder import StoreHolder
from persistence.sqlite._schema_ddl import _RESET_CHILD_SQL, _RESET_END_REASONS, _legacy_reset_child_sql, _sql_json_extract
from persistence.sqlite.conversation._store.sql_common import (
    _LISTABLE_CHILD_SQL, _RECOVERABLE_END_REASONS, _RECOVERABLE_END_REASONS_SQL, _non_continuation_child_sql,
    _sql_preview_raw, QUEUED_PROMPT_METADATA_KEY, _sql_in_window, _sql_session_last_active,
    _sql_session_last_active_by_id, escape_like as _escape_like, _SQL_IN_CHUNK, _id_chunks,
    _placeholders as _session_ids_placeholders, _source_match_clause, launch_stub_filter_sql,
)
from persistence.sqlite.conversation._store.compression import _UNSET_CREATED_SQL


logger = logging.getLogger("hermes_state")


def _delegate_from_json(col: str = "model_config") -> str:
    return _sql_json_extract(col, "$._delegate_from")


_MODEL_CONFIG_ROW_MISSING = object()


_LINEAGE_CTE_SQL = f"""
            WITH RECURSIVE
              ancestors(id) AS (
                SELECT ?
                UNION
                SELECT parent.id
                FROM ancestors a
                JOIN sessions child ON child.id = a.id
                JOIN sessions parent ON parent.id = child.parent_session_id
                WHERE parent.end_reason = 'compression'
{_non_continuation_child_sql('child.', 'parent.id')}
              ),
              descendants(id) AS (
                SELECT ?
                UNION
                SELECT child.id
                FROM descendants d
                JOIN sessions parent ON parent.id = d.id
                JOIN sessions child ON child.parent_session_id = parent.id
                WHERE parent.end_reason = 'compression'
{_non_continuation_child_sql('child.', 'parent.id')}
              ),
              lineage(id) AS (
                SELECT id FROM ancestors
                UNION
                SELECT id FROM descendants
              )"""


def _parse_model_config(raw: Any) -> dict[str, Any]:
    """Tolerant ``model_config`` decode: JSON text or dict -> dict copy; anything else -> {}."""
    if isinstance(raw, str) and raw.strip():
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
    return dict(raw) if isinstance(raw, dict) else {}


def _cwd_prefix_clause(cwd_prefix: str) -> tuple[str, list[str]]:
    prefix = cwd_prefix.rstrip("/\\") or cwd_prefix
    # ``_``/``%`` are LIKE wildcards but ordinary path characters: unescaped, a
    # prefix also matches sibling directories. The ``=`` arm keeps the raw prefix.
    esc = _escape_like(prefix)
    return (
        "(s.cwd = ? OR s.cwd LIKE ? ESCAPE '\\' OR s.cwd LIKE ? ESCAPE '\\')",
        [prefix, f"{esc}/%", f"{esc}\\\\%"],
    )


def _workspace_key_clause(key: str) -> tuple[str, list[str]]:
    """WHERE for ``workspace_key(row) == key``: git_repo_root equals ``key``, or (rows predating
    per-session git metadata) cwd is at/under ``key``."""
    prefix = key.rstrip("/\\") or key
    cwd_clause, cwd_params = _cwd_prefix_clause(prefix)
    return (
        f"(s.git_repo_root = ? OR (COALESCE(s.git_repo_root, '') = '' AND {cwd_clause}))",
        [prefix, *cwd_params],
    )


def _where_sql(clauses: list[str], lead: str = "") -> str:
    """``WHERE a AND b`` (with *lead* prefix) or "" when there are no clauses."""
    return f"{lead}WHERE {' AND '.join(clauses)}" if clauses else ""


def _session_filter_where(
    *, exclude_children: bool = False, source: str | None = None, sources: list[str] | None = None,
    session_key: str | None = None, exclude_sources: list[str] | None = None, cwd_prefix: str | None = None,
    min_message_count: int = 0, archived_only: bool = False, include_archived: bool = False,
    include_subagents: bool = False, local_created_sources: Sequence[str] | None = None,
) -> tuple[list[str], list[Any]]:
    """Shared ``sessions s`` WHERE builder so counts line up with listed rows. ``exclude_children``
    hides sub-agent runs and compression continuations but keeps branch/reset children
    (``_LISTABLE_CHILD_SQL``); ``include_subagents`` re-admits the sub-agent runs only
    (``sessions.show_subagents``). Clause order is part of the SQL text contract."""
    where: list[str] = []
    params: list[Any] = []
    if exclude_children and include_subagents:
        where.append(f"({_LISTABLE_CHILD_SQL} OR {_delegate_from_json('s.model_config')} IS NOT NULL)")
    elif exclude_children:
        where += [_LISTABLE_CHILD_SQL, f"{_delegate_from_json('s.model_config')} IS NULL"]
    # Show roots and user-visible branch/reset sessions, while still hiding sub-agent runs and compression
    # continuations. All four carry parent_session_id, so the shared predicate classifies the edge from
    # stable markers plus legacy-compatible parent metadata. Branch sessions are identified two ways, OR'd
    # for robustness: 1. A stable ``_branched_from`` marker in model_config, written by /branch at creation
    # time. This survives the parent being reopened and re-ended with a different end_reason (e.g.
    # tui_shutdown overwriting 'branched'), which otherwise hides the branch — see issue #20856. 2. The
    # legacy heuristic (parent ended with 'branched' before the child started), covering branch sessions
    # created before the marker existed.
    include_sources = [source] if source else list(sources or [])
    source_clause, source_params = _source_match_clause(include_sources, local_created_sources)
    for clause, values in (
        (source_clause, source_params),
        ("s.session_key = ?", [session_key] if session_key else []),
        (f"s.source NOT IN ({_session_ids_placeholders(exclude_sources or ())})", exclude_sources or []),
        (_cwd_prefix_clause(cwd_prefix) if cwd_prefix else ("", [])),
        ("s.message_count >= ?", [min_message_count] if min_message_count > 0 else []),
    ):
        if values:
            where.append(clause)
            params.extend(values)
    if archived_only:
        where.append("s.archived = 1")
    elif not include_archived:
        where.append("s.archived = 0")
    return where, params


def _expand_compression_lineage_ids(conn, frontier: list[str]) -> list[str]:
    """One both-direction compression-edge step from *frontier*: forward through every
    continuation child and backward through the compression parent, using the SAME edge
    definition as ``_CHAIN_STEP_SQL`` / the list-projection CTE (parent ended with
    ``reason='compression'``, child is not a branch / reset / delegate / tool session).
    Branch/reset children are NOT continuations — they are their own conversations."""
    ids = [sid for sid in frontier if sid]
    if not ids:
        return []
    out: list[str] = []
    for chunk in _id_chunks(ids, _SQL_IN_CHUNK // 4):  # each id bound 4x below
        ph = _session_ids_placeholders(chunk)
        child_filter = (
            f"{_sql_json_extract('child.model_config', '$._branched_from')} IS NULL "
            f"AND {_sql_json_extract('child.model_config', '$._delegate_from')} IS NULL "
            f"AND NOT ({_RESET_CHILD_SQL.format(a='child')}) "
            "AND COALESCE(child.source, '') != 'tool'"
        )
        out.extend(row["id"] for row in conn.execute(
            # forward: every continuation of a compression-ended frontier row
            f"SELECT child.id FROM sessions parent "
            f"JOIN sessions child ON child.parent_session_id = parent.id "
            f"WHERE parent.id IN ({ph}) AND parent.end_reason = 'compression' AND {child_filter}",
            chunk,
        ).fetchall())
        out.extend(row["id"] for row in conn.execute(
            # backward: the compression parent of a frontier row
            f"SELECT parent.id FROM sessions child "
            f"JOIN sessions parent ON parent.id = child.parent_session_id "
            f"WHERE child.id IN ({ph}) AND parent.end_reason = 'compression' "
            f"AND {child_filter}",
            chunk,
        ).fetchall())
    return out


def _collect_delegate_child_ids(conn, parent_ids: list[str]) -> list[str]:
    """Delegate-subagent ids (``_delegate_from`` marker, walked recursively) to cascade-delete with
    *parent_ids*; untagged children stay orphaned, not deleted."""
    df = _delegate_from_json()
    seeds = {sid for sid in parent_ids if sid}
    # Seed visited with the parents: a marker chain can loop back onto a parent,
    # which would then be collected as its own descendant. Never return parents.
    # A delegation marker chain can loop back onto a parent — a cycle, or a parent that is also another
    # parent's delegate child when several ids are deleted at once — and without this guard that parent
    # would be collected as one of its own descendants and cascade-deleted along with all of its messages.
    # Callers delete the parents separately, so parents must never appear in the returned child set.
    # (#49148)
    found: set[str] = set(seeds)
    frontier = list(seeds)
    while frontier:
        next_frontier: list[str] = []
        for chunk in _id_chunks(frontier, _SQL_IN_CHUNK // 2):  # each id is bound twice below
            ph = _session_ids_placeholders(chunk)
            cursor = conn.execute(
                f"SELECT id FROM sessions WHERE {df} IN ({ph}) "
                f"OR (parent_session_id IN ({ph}) AND {df} IS NOT NULL)", chunk + chunk,
            )
            for row in cursor.fetchall():
                if row["id"] not in found:
                    found.add(row["id"])
                    next_frontier.append(row["id"])
        frontier = next_frontier
    return [sid for sid in found if sid not in seeds]


def _delete_delegate_children(conn, parent_ids: list[str], *, media_root: str) -> list[str]:
    ids = _collect_delegate_child_ids(conn, parent_ids)
    from persistence.sqlite.conversation._store.retirement import retire_sessions
    retire_sessions(conn, ids, media_root=media_root)
    for chunk in _id_chunks(ids):
        ph = _session_ids_placeholders(chunk)
        conn.execute(f"DELETE FROM messages WHERE session_id IN ({ph})", chunk)
        # FK safety: orphan any untagged stragglers pointing at a doomed row.
        conn.execute(f"UPDATE sessions SET parent_session_id = NULL WHERE parent_session_id IN ({ph})", chunk)
        conn.execute(f"DELETE FROM sessions WHERE id IN ({ph})", chunk)
    return ids


INTERNAL_LISTING_SOURCES = ("kanban", "tool", "oneshot")


SESSION_STATUS_COMPLETE = "complete"


SESSION_STATUS_INTERRUPTED = "interrupted"


SESSION_STATUS_ERROR = "error"


_ERROR_FINISH_REASONS = frozenset({"error", "agent_error", "content_filter"})


def classify_session_status(role: Optional[str], has_tool_calls: bool, finish_reason: Optional[str]) -> str:
    """Error finish → ``error``; assistant with pending tool_calls or a trailing user/tool row →
    ``interrupted``; otherwise ``complete`` (benign default: pickers must not alarm on unknown shapes)."""
    if (finish_reason or "").strip().lower() in _ERROR_FINISH_REASONS:
        return SESSION_STATUS_ERROR
    r = (role or "").strip().lower()
    if r in {"user", "tool"} or (r == "assistant" and has_tool_calls):
        return SESSION_STATUS_INTERRUPTED
    return SESSION_STATUS_COMPLETE


_SAME_KEY_NAMESPACE_SQL = (
    "p.session_key IS NULL OR sessions.session_key IS NULL"
    " OR substr(p.session_key, 1, instr(substr(p.session_key, 7), ':') + 6)"
    "  = substr(sessions.session_key, 1, instr(substr(sessions.session_key, 7), ':') + 6)"
)


_UPSERT_KEEP_EXISTING_SQL = ",\n".join(
    f"                       {col} = COALESCE(sessions.{col}, excluded.{col})" for col in (
        "session_key", "chat_id", "chat_type", "thread_id", "parent_session_id", "cwd", "profile_name",
        "transport_profile", "git_repo_root", "origin_json", "display_name",
    )
) + ",\n" + (
    # Accounting may insert before the creator ('unknown') and a routing write carries 'local':
    # neither is creation provenance (_UNSET_CREATED_SQL), so a later creator replaces them and
    # 'local' never replaces anything. A real source, once stamped, is preserved (#56439).
    "                       created_source = CASE WHEN "
    + _UNSET_CREATED_SQL.format(col="sessions.created_source")
    + " THEN COALESCE(NULLIF(excluded.created_source, 'local'), sessions.created_source)"
    " ELSE sessions.created_source END"
)


def _inherit_col_sql(col: str, extra: str = "") -> str:
    """``col = COALESCE(sessions.col, (SELECT p.col FROM parent))`` (whitespace is part of the SQL text)."""
    pad = " " * (30 + len(col))
    return (
        f"{col} = COALESCE(sessions.{col},\n{pad}(SELECT p.{col} FROM sessions p\n"
        f"{pad}  WHERE p.id = sessions.parent_session_id{extra}))"
    )


_INHERIT_SEP = ",\n" + " " * 27


_INHERIT_PARENT_META_SQL = (
    "UPDATE sessions\n                       SET "
    + _INHERIT_SEP.join((
        *(_inherit_col_sql(c) for c in ("cwd", "git_repo_root", "git_branch")),
        _inherit_col_sql("profile_name", "\n" + " " * 46 + f"AND ({_SAME_KEY_NAMESPACE_SQL})"),
    ))
    + "\n                     WHERE id = ? AND parent_session_id IS NOT NULL"
)


_FORK_EDGE_EXCLUSION_SQL = "".join(
    f"\n                       AND COALESCE({_sql_json_extract('model_config', f'$.{marker}')}, '')"
    "\n                           != parent_session_id"
    for marker in ("_delegate_from", "_branched_from")
)


_INHERIT_PARENT_ROUTING_SQL = (
    "UPDATE sessions\n                       SET "
    + _INHERIT_SEP.join(_inherit_col_sql(c) for c in (
        "user_id", "session_key", "chat_id", "chat_type", "thread_id", "display_name", "origin_json",
        "transport_profile",
    ))
    + "\n                     WHERE id = ? AND parent_session_id IS NOT NULL\n"
    "                       AND EXISTS (\n"
    "                           SELECT 1 FROM sessions p\n"
    "                           WHERE p.id = sessions.parent_session_id\n"
    "                             AND p.end_reason = 'compression'\n"
    "                       )"
    + _FORK_EDGE_EXCLUSION_SQL
)


def _own_profile_name(holder: StoreHolder) -> Optional[str]:
    """The profile owning THIS store, from ``db_path`` alone (``<root>/state.db`` → default,
    ``<root>/profiles/<name>/state.db`` → name): a gateway serving a NON-launch profile opens that
    profile's store. None outside the profile tree — NULL beats a fabricated owner."""
    try:
        from hermes_constants import get_default_hermes_root
        root = get_default_hermes_root().resolve()
        parent = Path(holder.db_path).resolve().parent
        if parent == root:
            return "default"
        is_profile_dir = parent.parent == root / "profiles"
        if is_profile_dir and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", parent.name):
            return parent.name
    except Exception:  # health: allow BLE001 -- an underivable profile is None
        logger.debug("own-profile derivation failed", exc_info=True)
    return None


def _inherit_parent_session_metadata(conn, session_id: str) -> None:
    """NULL-fill a child's cwd/git/profile from its parent (profile_name only within the same
    ``agent:<ns>:`` namespace). Gateway routing columns are inherited ONLY by compression forks
    (a crash before the gateway re-records the peer would strand the child unroutable); delegate
    and branch children must NOT inherit them (peer recovery could repoint traffic into a
    subagent's session), including when their parent row itself ended on compression — a long
    batch outlives its coordinator's rotation, and two live rows holding one routing key is the
    shape reported in #92859."""
    conn.execute(_INHERIT_PARENT_META_SQL, (session_id,))
    conn.execute(_INHERIT_PARENT_ROUTING_SQL, (session_id,))


def _insert_session_row(
    holder: StoreHolder, session_id: str, source: str, model: str | None = None, model_config: dict[str, Any] | None = None,
    system_prompt: str | None = None, user_id: str | None = None, session_key: Optional[str] = None,
    chat_id: str | None = None, chat_type: str | None = None, thread_id: str | None = None,
    parent_session_id: str | None = None, cwd: str | None = None, profile_name: Optional[str] = None,
    git_repo_root: str | None = None, origin_json: str | None = None, display_name: str | None = None,
    transport_profile: Optional[str] = None,
) -> None:
    """Upsert a session row, never overwriting what an earlier writer set (the gateway creates a
    bare row before create_session carries the real model/prompt) — the one exception is the
    token-accounting guard's placeholder ``source='unknown'``, which a later writer's real surface
    replaces (#111999): once minted, that placeholder otherwise labelled a real session anonymous
    for life, because this upsert is the only writer that could correct it. chat_id/thread_id scope gateway
    /resume (IDOR). Children backfill from the parent; a missing profile_name is stamped with THIS
    store's own (NULL reads as unowned).

    When ``parent_session_id`` is set (compression fork, delegate/subagent spawn, branch continuation)
    and this row's own ``cwd``/``git_repo_root``/ ``git_branch``/``profile_name`` are still NULL after
    the insert, they are backfilled from the parent row. Callers of ``create_session`` for a child
    session historically didn't propagate these fields themselves (e.g. the compression-fork path), so a
    lineage could silently lose its working directory and drop out of the project sidebar every time it
    forked (#64709), or lose its owning profile and be aggregated as "default" every time it rotated or
    branched (the cross-profile session-jump bug). This only fills NULLs — an explicit value on the
    child is never overwritten. For compression forks specifically (parent ended with
    ``end_reason='compression'``), the gateway origin columns
    (``user_id``/``session_key``/``chat_id``/``chat_type``/
    ``thread_id``/``display_name``/``origin_json``) are inherited too, so a crash before the gateway
    re-records the peer can't strand the child without a recoverable routing mapping (#59527).
    When the caller passes no ``profile_name`` at all, the row is stamped with THIS store's own profile
    (:meth:`_own_profile_name`) instead of NULL. Every ``state.db`` belongs to exactly one profile — the
    same single-match contract :meth:`backfill_null_session_profiles` relies on — so the stamp is
    derivation, not a guess. Rows minted NULL after that one-shot #94724 backfill ran stayed NULL
    forever, and profile-keyed consumers (desktop sidebar scope matching, ``@session:<profile>/<id>``
    deep links, the fail-closed owner ladder) treat NULL as unowned: the session vanishes from the
    sidebar even though its transcript is intact (#99222). Stores outside the profile tree (explicit
    ``db_path`` in tests, ad-hoc copies) derive nothing and keep NULL — never guess.
    """
    from persistence.sqlite.conversation._store.worker_lifecycle import insert_session_row_in_transaction
    params = dict(session_id=session_id, source=source, model=model, model_config=model_config,
                  system_prompt=system_prompt, user_id=user_id, session_key=session_key,
                  chat_id=chat_id, chat_type=chat_type, thread_id=thread_id,
                  parent_session_id=parent_session_id, cwd=cwd, profile_name=profile_name,
                  git_repo_root=git_repo_root, origin_json=origin_json, display_name=display_name,
                  transport_profile=transport_profile)
    holder._execute_write(lambda conn: insert_session_row_in_transaction(holder, conn, **params),
                        patience_s=holder._TRANSCRIPT_WRITE_PATIENCE_S)


def create_session(holder: StoreHolder, session_id: str, source: str, **kwargs) -> str:
    """Create (upsert) a session record. Returns the session_id."""
    _insert_session_row(holder, session_id, source, **kwargs)
    return session_id


_ORPHAN_ADOPTION_MAX_GAP_S = 900.0


_NON_CONTINUATION_CHILD_FILTER_SQL = _non_continuation_child_sql("{alias}")


def end_session(holder: StoreHolder, session_id: str, end_reason: str) -> None:
    """Mark a session ended; the first end_reason wins (a compression split must keep
    ``'compression'`` even if a stale end_session() lands later); reopen_session() to re-end."""
    holder._execute_write(lambda conn: _end_and_bump(holder,
        conn, "UPDATE sessions SET ended_at = ?, end_reason = ? WHERE id = ? AND ended_at IS NULL",
        (time.time(), end_reason, session_id), session_id, end_reason,
    ))


def _end_and_bump(holder: StoreHolder, conn, sql: str, params: tuple, session_id: str, reason: str) -> int:
    """Run an end-stamp UPDATE; only a boundary this call actually wrote advances the
    conversation generation (a no-op must not rotate the peer). Returns rowcount."""
    changed = conn.execute(sql, params).rowcount
    if changed:
        _bump_conversation_generation(holder, conn, session_id, reason)
    return changed


def _retire_undrained_queue_rows_conn(conn, session_id: str) -> int:
    """The still-marked == never-drained accept-row UPDATE (#125577), on a caller's connection."""
    return conn.execute(
        "UPDATE messages SET active = 0 "
        "WHERE session_id = ? AND role = 'user' AND active = 1 "
        f"AND COALESCE({_sql_json_extract('display_metadata', '$.' + QUEUED_PROMPT_METADATA_KEY)}, 0) = 1",
        (session_id,),
    ).rowcount


def retire_undrained_queue_rows(holder: StoreHolder, session_id: str) -> int:
    """Deactivate busy-queue accept rows that never drained (#125577): a restart discarded the
    in-memory queue, so nothing re-placed/deactivated the row written at accept time and
    alternation repair would glue the never-run prompt into the previous turn's user message.
    The drain's replacement row carries no marker, so still-marked == never drained; a
    dispatched-then-interrupted turn's row is never marked (its discard is handled in-process,
    #123532). Deactivated, never deleted — the row stays as history.

    Split out of :meth:`reopen_session` so the read-only mount paths (#85303: resume,
    lazy watch, hydration) can retire stale accept rows WITHOUT clearing ``ended_at`` —
    after a restart the marked row would otherwise stay active and visible on every
    mount until the next send. Idempotent; returns the number of rows retired.
    """
    return holder._execute_write(
        lambda conn: _retire_undrained_queue_rows_conn(conn, session_id))


def reopen_session(holder: StoreHolder, session_id: str) -> None:
    """Clear ended_at/end_reason so a session can be resumed; first freeze markerless legacy reset
    children, skipping explicit fork/delegate provenance and children that predate the parent itself.
    The guard compares against the parent's started_at, not its current ended_at: a parent that was
    reopened and re-ended later still owns reset children from its earlier boundaries."""
    def _do(conn):
        _retire_undrained_queue_rows_conn(conn, session_id)
        conn.execute(
            "UPDATE sessions AS child SET model_config = json_set("
            "COALESCE(child.model_config, '{}'), '$._reset_from', child.parent_session_id) "
            f"WHERE child.parent_session_id = ? AND {_sql_json_extract('child.model_config', '$._reset_from')} IS NULL "
            f"AND {_sql_json_extract('child.model_config', '$._branched_from')} IS NULL "
            f"AND {_sql_json_extract('child.model_config', '$._delegate_from')} IS NULL "
            "AND COALESCE(child.source, '') != 'tool' "
            "AND child.started_at >= (SELECT p.started_at FROM sessions p WHERE p.id = child.parent_session_id) "
            f"AND {_legacy_reset_child_sql('child', _session_ids_placeholders(_RESET_END_REASONS))}",
            (session_id, *_RESET_END_REASONS),
        )
        conn.execute(
            "UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = ?", (session_id,),
        )
        # Resuming re-activates the chat: drop the idle sweep's archive (never a manual one).
        _unarchive_auto_archived_lineage(conn, session_id)
    holder._execute_write(_do)


def promote_to_session_reset(holder: StoreHolder, session_id: str, reason: str = "session_reset") -> bool:
    """Durably mark an intentional reset boundary on live rows or rows with a *recoverable* accidental
    end_reason (explicit boundaries are preserved): an ``agent_close`` row left recoverable would be
    resurrected by stale-route recovery. Keep in sync with find_latest_gateway_session_for_peer.

    Plain ``end_session()`` is NOT sufficient for reset boundaries: it no-ops on an already-ended row,
    so a row that agent cleanup already closed as ``agent_close`` would stay recoverable and stale-route
    recovery would resurrect the reset session with its full history (#61220, #61993, #63539).
    """
    if not session_id:
        return False
    now = time.time()
    # /new and policy auto-resets promote rather than end_session, so the
    # generation advances here too — same transaction, only when written.
    try:
        return bool(holder._execute_write(lambda conn: _end_and_bump(holder,
            conn, "UPDATE sessions SET ended_at = ?, end_reason = ? WHERE id = ? AND (ended_at IS NULL "
            f"OR end_reason IN ({_RECOVERABLE_END_REASONS_SQL}))",
            (now, reason, session_id), session_id, reason,
        )))
    except Exception:  # health: allow BLE001 -- a failed end answers False
        return False


def update_session_cwd(
    holder: StoreHolder, session_id: str, cwd: str, git_branch: Optional[str] = None,
    git_repo_root: Optional[str] = None, replace_git_meta: bool = False,
) -> Optional[int]:
    """Persist the authoritative cwd and claim a Git metadata generation. git fields are written
    only when non-empty (a probe failure never clobbers a value) except under ``replace_git_meta``
    (a workspace MOVE overwrites the old repo identity). Async probes publish with the returned
    generation so an older worker cannot overwrite a newer claim (A -> B -> A)."""
    if not session_id or not cwd:
        return None
    branch = (git_branch or "").strip()
    repo_root = (git_repo_root or "").strip()
    def _do(conn):
        current = conn.execute("SELECT cwd FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if current is None:
            return None
        sets = ["cwd = ?", "git_metadata_generation = COALESCE(git_metadata_generation, 0) + 1"]
        params: list[Any] = [cwd]
        if current[0] != cwd or replace_git_meta:
            sets.extend(("git_branch = ?", "git_repo_root = ?"))
            params.extend((branch or None, repo_root or None))
        else:  # same cwd: only overwrite with captured (non-empty) values
            for col, val in (("git_branch", branch), ("git_repo_root", repo_root)):
                if val:
                    sets.append(f"{col} = ?")
                    params.append(val)
        conn.execute(f"UPDATE sessions SET {', '.join(sets)} WHERE id = ?", [*params, session_id])
        row = conn.execute(
            "SELECT git_metadata_generation FROM sessions WHERE id = ?", (session_id,),
        ).fetchone()
        return None if row is None else int(row[0])
    return holder._execute_write(_do)


def publish_session_git_metadata(
    holder: StoreHolder, session_id: str, cwd: str, generation: int, git_branch: Optional[str] = None,
    git_repo_root: Optional[str] = None,
) -> bool:
    """Publish async Git enrichment only while its cwd claim is current."""
    valid_generation = isinstance(generation, int) and not isinstance(generation, bool) and generation >= 1
    if not session_id or not cwd or not valid_generation:
        return False
    fields = [
        (col, val) for col, val in (
            ("git_branch", (git_branch or "").strip()), ("git_repo_root", (git_repo_root or "").strip()),
        ) if val
    ]
    if not fields:
        return False
    return holder._write_rowcount(
        f"UPDATE sessions SET {', '.join(f'{col} = ?' for col, _ in fields)} "
        "WHERE id = ? AND cwd = ? AND git_metadata_generation = ?",
        [val for _, val in fields] + [session_id, cwd, generation],
    ) == 1


def backfill_repo_roots(holder: StoreHolder, cwd_to_root: dict[str, str]) -> None:
    """Backfill git repo roots for cwds without one; never clobbers a recorded root."""
    pairs = [(root, cwd) for cwd, root in cwd_to_root.items() if root and cwd]
    if pairs:
        holder._write_sql(
            "UPDATE sessions SET git_repo_root = ? WHERE cwd = ? AND COALESCE(git_repo_root, '') = ''",
            pairs, many=True,
        )


def touch_session_activity(
    holder: StoreHolder, session_id: str, ts: Optional[float] = None, *, description: Optional[str] = None,
    provenance: Optional[ActivityProvenance] = None,
) -> None:
    """Stamp durable mid-turn activity (observation-only; rate-limited by the caller) so surfaces see
    activity before any message row lands. Never moves ``last_activity_at`` backwards.

    Called (rate-limited) from ``AIAgent._touch_activity`` so gateway/CLI surfaces and stall consumers
    observe API/tool/compaction activity even when no new message row has been written yet (#72016 /
    #72039).
    """
    if not session_id:
        return
    when = float(ts if ts is not None else time.time())
    holder._write_sql(
        "UPDATE sessions SET last_activity_at = ?, "
        "last_activity_description = ?, last_activity_provenance = ? "
        "WHERE id = ? AND (last_activity_at IS NULL OR last_activity_at < ?)",
        (
            when, bound_activity_description(description),
            normalize_activity_provenance(provenance).value, session_id, when,
        ),
        patience_s=holder._ACTIVITY_WRITE_PATIENCE_S,
    )


def clear_session_activity_labels(holder: StoreHolder, session_id: str) -> None:
    """Clear activity labels after a turn (``last_activity_at`` is kept so idle / watchdog clocks stay
    continuous). A no-op clear skips the write transaction.

    Description and provenance are observation labels for *what was happening at* that timestamp during
    an active turn; once the turn is idle they must not keep advertising "compressing" / "executing
    tool" (#72039).
    Response-critical-path contract (#76354 review S1): runs in the turn's ``finally``; a no-op clear
    (labels already empty) skips the write transaction entirely, and a real clear uses the same short
    sub-second busy budget as :meth:`touch_session_activity` instead of the full routine write patience.
    """
    if not session_id:
        return
    try:
        row = holder._read_one(
            "SELECT last_activity_description, last_activity_provenance FROM sessions WHERE id = ?",
            (session_id,),
        )
    except sqlite3.Error:
        row = None
    if row is not None and not row[0] and (not row[1] or row[1] == ActivityProvenance.UNKNOWN.value):
        return
    holder._write_sql(
        "UPDATE sessions SET last_activity_description = ?, last_activity_provenance = ? WHERE id = ?",
        ("", ActivityProvenance.UNKNOWN.value, session_id), patience_s=holder._ACTIVITY_WRITE_PATIENCE_S,
    )


def update_session_meta(
    holder: StoreHolder, session_id: str, model_config_json: str, model: Optional[str] = None,
) -> None:
    """Update model_config and (COALESCE) optionally model."""
    holder.flush_token_counts()  # barrier against queued token deltas — see update_session_model
    holder._write_sql(
        "UPDATE sessions SET model_config = ?, model = COALESCE(?, model) WHERE id = ?",
        (model_config_json, model, session_id),
    )


def update_system_prompt(holder: StoreHolder, session_id: str, system_prompt: Optional[str]) -> None:
    """Store the full assembled system prompt snapshot."""
    def _do(conn):
        conn.execute(
            "UPDATE sessions SET system_prompt_hash = ?, system_prompt = NULL WHERE id = ?",
            (holder._store_system_prompt(conn, system_prompt), session_id),
        )
        _delete_unreferenced_system_prompts(conn)
    holder._execute_write(_do)


def update_session_tool_names(holder: StoreHolder, session_id: str, pin: Any) -> None:
    """Persist the session's ``tools[]`` pin (JSON-serializable) so a rebuilt AIAgent sends the
    same bytes; ``None`` clears. The array repeats across sessions like a system prompt does, so it
    is stored in the same content-addressed ``system_prompts`` table and the column holds its hash
    (legacy rows: an inline JSON name list); ``get_session`` resolves either."""
    payload = json.dumps(pin) if pin is not None else None
    def _do(conn):
        conn.execute("UPDATE sessions SET tool_names = ? WHERE id = ?",
                     (holder._store_system_prompt(conn, payload), session_id))
        _delete_unreferenced_system_prompts(conn)
    holder._execute_write(_do)


def update_session_model(
    holder: StoreHolder, session_id: str, model: str, provider: Optional[str] = None, *,
    base_url: Optional[str] = None, api_mode: Optional[str] = None,
) -> None:
    """Set the model after a mid-session /model switch (unconditionally) and drop any Browser
    runtime lock (lineage markers survive).

    Route writers never touch the stored prompt; ``_stored_prompt_matches_runtime`` decides staleness.

    When *provider* is given the whole route is written, in both shapes resume reads (top-level
    keys for the TUI/Desktop, ``gateway_runtime`` for the CLI), so a later resume recombines the
    model with the provider that serves it (#79536). ``base_url``/``api_mode`` are always
    replaced then (``None`` deletes): the previous provider's endpoint must not survive a switch,
    or resume sends the new provider's model to the old host. Callers without provider knowledge
    leave the stored route untouched.
    """
    # Flush first: a still-queued pre-switch delta applied after this UPDATE would trip the
    # first_accounted_route overwrite and resurrect the old route.
    holder.flush_token_counts()
    patch: dict[str, Any] = {"browser_model_lock": None}
    if model:
        patch["model"] = model
    if provider:
        route = {"provider": provider, "base_url": base_url or None, "api_mode": api_mode or None}
        patch.update(route, gateway_runtime=route)
    _write_model_config_patch(holder,
        session_id, patch, "UPDATE sessions SET model = ?, model_config = ? WHERE id = ?",
        lambda merged: (model, merged, session_id),
    )


def _write_model_config_patch(
    holder: StoreHolder, session_id: str, patch: dict[str, Any],
    sql: str = "UPDATE sessions SET model_config = ? WHERE id = ?",
    params: Optional[Callable[[Optional[str]], tuple]] = None,
) -> None:
    """Merge ``patch`` into model_config then run ``sql`` with ``params(merged)`` in one write
    transaction; no-op when the row doesn't exist."""
    def _do(conn):
        merged = _merge_model_config_json(holder, conn, session_id, patch)
        if merged is _MODEL_CONFIG_ROW_MISSING:
            return
        conn.execute(sql, params(merged) if params else (merged, session_id))
    holder._execute_write(_do)


def _merge_model_config_json(
    holder: StoreHolder, conn, session_id: str, patch: dict[str, Any], *, on_missing: str = "skip",
):
    """SELECT + tolerant-parse + merge ``patch`` into model_config (the one place that keeps
    ``_branched_from``/``_delegate_from`` alive); ``None`` deletes a key. Returns serialized JSON
    (``None`` when empty) or ``_MODEL_CONFIG_ROW_MISSING`` (``on_missing="raise"`` → ValueError)."""
    row = conn.execute("SELECT model_config FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if row is None:
        if on_missing == "raise":
            raise ValueError(f"Session not found: {session_id}")
        return _MODEL_CONFIG_ROW_MISSING
    config = _parse_model_config(row[0])
    for key, value in patch.items():
        if value is None:
            config.pop(key, None)
        else:
            config[key] = value
    return json.dumps(config) if config else None


def patch_session_model_config(holder: StoreHolder, session_id: str, patch: dict[str, Any]) -> None:
    """Merge ``patch`` into model_config atomically (``None`` removes a key);
    no-op when the row or patch is empty."""
    if not session_id or not patch:
        return
    _write_model_config_patch(holder, session_id, patch)


def get_session_model_config_value(holder: StoreHolder, session_id: str, key: str, default: Any = None) -> Any:
    """Read one key out of a session's model_config JSON (tolerant parse)."""
    session = get_session(holder, session_id) or {}
    return _parse_model_config(session.get("model_config")).get(key, default)


def update_session_runtime_lock(
    holder: StoreHolder, session_id: str, *, model: Optional[str] = None, provider: Optional[str] = None,
    model_options: Optional[dict[str, Any]] = None, route_source: Optional[str] = None,
    confirmed: bool = False,
) -> None:
    """Persist a Browser / API-client runtime lock into model_config (lineage markers survive).
    Route writers never touch the stored prompt; ``_stored_prompt_matches_runtime`` decides staleness."""
    lock = {
        "provider": provider or "", "model": model or "", "model_options": model_options or {},
        "route_source": route_source or "", "confirmed": bool(confirmed), "updated_at": time.time(),
    }
    _write_model_config_patch(holder,
        session_id, {"browser_model_lock": lock},
        """UPDATE sessions SET
                   model_config = ?,
                   model = COALESCE(?, model)
                   WHERE id = ?""",
        lambda merged: (merged, model, session_id),
    )


def set_session_yolo(holder: StoreHolder, session_id: str, enabled: bool) -> None:
    """Persist the per-session YOLO flag so ``/yolo`` survives ``--resume``; no-op without a row."""
    if not session_id:
        return
    _write_model_config_patch(holder, session_id, {"yolo_mode": bool(enabled)})


def session_yolo_enabled(session_meta: Optional[dict[str, Any]]) -> bool:
    """Persisted YOLO flag; False on any parse failure (resume must never enable the bypass)."""
    return bool(_parse_model_config((session_meta or {}).get("model_config")).get("yolo_mode"))


def get_session(holder: StoreHolder, session_id: str) -> Optional[dict[str, Any]]:
    """Get a session by ID (drains queued token deltas first so cost readers see exact totals)."""
    holder.flush_for_read()
    row = holder._read_one(
        "SELECT s.*, COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved, "
        "COALESCE(tp.prompt, s.tool_names) AS _tool_names_resolved "
        "FROM sessions s LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash "
        "LEFT JOIN system_prompts tp ON tp.hash = s.tool_names WHERE s.id = ?",
        (session_id,),
    )
    return _session_row_dict(row) if row else None


def get_session_list_row(holder: StoreHolder, session_id: str) -> Optional[dict[str, Any]]:
    """Picker columns only: the live listing skips ``get_session``'s prompt/tool JOIN and flush."""
    row = holder._read_one(
        "SELECT id, source, title, started_at, ended_at, message_count, model, hidden, archived "
        "FROM sessions WHERE id = ?",
        (session_id,),
    )
    return dict(row) if row else None


def get_recent_session_model_route(holder: StoreHolder, session_id: str) -> Optional[dict[str, Any]]:
    """Most recently used main-loop model route as one coherent per-call tuple
    (``session_model_usage`` keeps model+provider together; ``sessions`` mixes route changes).
    Recency, not lifetime call count: on a long session a route retired weeks ago can hold the
    highest ``api_call_count`` forever, and /status and /usage would keep calling it current.
    ``rowid DESC`` breaks same-timestamp ties toward the route that first appeared later; without
    it SQLite's temp-sort order is unspecified and the retired route can win."""
    holder.flush_for_read()
    row = holder._read_one(
        """SELECT model, billing_provider, billing_base_url, billing_mode,
                      api_call_count
                 FROM session_model_usage
                WHERE session_id = ?
                  AND task = ''
                  AND model <> 'unknown'
                  AND billing_provider <> ''
                ORDER BY last_seen DESC, rowid DESC
                LIMIT 1""",
        (session_id,),
    )
    return dict(row) if row else None


def resolve_session_id(holder: StoreHolder, session_id_or_prefix: str) -> Optional[str]:
    """Exact id, else the single unambiguous prefix match, else None."""
    exact = get_session(holder, session_id_or_prefix)
    if exact:
        return exact["id"]
    matches = holder._read_all(
        "SELECT id FROM sessions WHERE id LIKE ? ESCAPE '\\' ORDER BY started_at DESC LIMIT 2",
        (f"{_escape_like(session_id_or_prefix)}%",),
    )
    return matches[0]["id"] if len(matches) == 1 else None


def backfill_null_session_profiles(holder: StoreHolder, profile_name: str) -> int:
    """Stamp this store's own profile onto legacy ``profile_name IS NULL`` rows, which the fail-closed
    owner ladder cannot route. Never overwrites a non-NULL owner. Returns rows stamped.

    Sessions created before the durable-ownership work (#95407 lineage) carry ``profile_name = NULL``.
    On single-backend installs that was harmless, but once a Desktop registers a second connection the
    fail-closed owner ladder (which is correct for new sessions) can no longer route those rows anywhere
    — every pre-campaign session becomes unresumable after upgrade (#94724, field report).
    """
    stamp = (profile_name or "").strip()
    if not stamp:
        return 0
    return int(holder._write_rowcount(
        """UPDATE sessions
               SET profile_name = ?
             WHERE profile_name IS NULL OR TRIM(profile_name) = ''""",
        (stamp,),
    ) or 0)


def backfill_acp_session_cwd(holder: StoreHolder) -> int:
    """Promote ``model_config.cwd`` into the cwd column for ACP rows lacking one.

    ACP sessions minted before the adapter populated the column still carry
    their workspace inside ``model_config``, written by the same adapter that
    knew the real directory — so this is a record being promoted, not a guess.
    Only fills NULL/empty; an explicit column value always wins. Returns the
    number of rows changed.
    """
    return int(holder._write_rowcount(
        """UPDATE sessions
                  SET cwd = json_extract(model_config, '$.cwd')
                WHERE source = 'acp'
                  AND COALESCE(cwd, '') = ''
                  AND json_valid(model_config)
                  AND COALESCE(json_extract(model_config, '$.cwd'), '') != ''""",
        (),
    ) or 0)


def _set_lineage_column(holder: StoreHolder, column: str, session_id: str, value: Any, *,
                        extra_set_sql: str = "") -> bool:
    """Set one ``sessions`` column across a whole compression lineage: Desktop projects roots
    forward to their tip, so updating only the tip would let the root resurrect it on refresh.
    *extra_set_sql* (trusted literal, ``, col = expr``) rides the same UPDATE."""
    return bool(holder._execute_write(
        lambda conn: _set_lineage_column_in_transaction(holder,
            conn, column, session_id, value, extra_set_sql=extra_set_sql)
    ))


def _set_lineage_column_in_transaction(holder: StoreHolder, conn, column: str, session_id: str, value: Any, *,
                                       extra_set_sql: str = ""):
    """Return affected IDs so authority revisions share this exact lineage selector. A plain
    ``archived`` write (canonical mutation, user/CLI/API) is the DELIBERATE archive and clears
    the idle sweep's ``auto_archived`` provenance, same as :meth:`set_session_archived`."""
    if column == "archived" and not extra_set_sql:
        extra_set_sql = ", auto_archived = 0"
    return [row[0] for row in conn.execute(
        _LINEAGE_CTE_SQL + f"""
            UPDATE sessions
            SET {column} = ?{extra_set_sql}
            WHERE id IN (SELECT id FROM lineage)
            RETURNING id
            """,
        (session_id, session_id, value),
    ).fetchall()]


def set_session_archived(holder: StoreHolder, session_id: str, archived: bool, *, notify: bool = True) -> tuple[bool, tuple[str, ...]]:
    """Soft-hide (or unhide) a session and its compression lineage; messages are kept.
    This is the DELIBERATE archive (user, CLI, API): it clears the ``auto_archived``
    provenance, so re-activation never un-hides it on the user's behalf. A real
    unarchived -> archived flip fires ``on_session_delete`` (``reason="archive"``) after commit;
    ``notify=False`` is for housekeeping that hides a row whose session lives on elsewhere."""
    def _do(conn):
        row = conn.execute("SELECT archived FROM sessions WHERE id = ?", (session_id,)).fetchone()
        ids = _set_lineage_column_in_transaction(holder,
            conn, "archived", session_id, int(archived), extra_set_sql=", auto_archived = 0")
        return ids, bool(archived and ids and row is not None and not row[0])
    ids, newly_archived = holder._execute_write(_do)
    return bool(ids), tuple(ids) if newly_archived and notify else ()


def _auto_archive_lineage(holder: StoreHolder, session_id: str) -> bool:
    """The idle sweep's archive: like :meth:`set_session_archived` but stamps
    ``auto_archived`` on the rows IT hides. A row already archived keeps its provenance, so a
    deliberately archived ancestor is never relabelled as sweep-owned (SQLite evaluates every
    SET expression against the pre-update row)."""
    return _set_lineage_column(holder,
        "archived", session_id, 1,
        extra_set_sql=", auto_archived = CASE WHEN archived = 0 THEN 1 ELSE auto_archived END")


def _unarchive_auto_archived_lineage(conn, session_id: str) -> bool:
    """Un-hide a lineage the idle sweep archived, on *conn* (caller's write txn). A tip that is
    published (compression) or reopened (resume) under it is live again, and the listing admits
    a lineage by its ROOT's flag, so leaving the sweep's stamp would hide an active chat (#117713).
    A lineage with ANY deliberately archived row (``archived = 1 AND auto_archived = 0``) is left
    alone: a manual archive stays until the user un-archives it. True when rows were un-hidden."""
    params = (session_id, session_id)
    manual = conn.execute(
        _LINEAGE_CTE_SQL + """
            SELECT 1 FROM sessions
            WHERE id IN (SELECT id FROM lineage) AND archived <> 0 AND COALESCE(auto_archived, 0) = 0
            LIMIT 1
            """, params).fetchone()
    if manual is not None:
        return False
    return conn.execute(
        _LINEAGE_CTE_SQL + """
            UPDATE sessions SET archived = 0, auto_archived = 0
            WHERE id IN (SELECT id FROM lineage) AND archived <> 0
            """, params).rowcount > 0


RECOVERABLE_END_REASONS = _RECOVERABLE_END_REASONS


def unarchive_recoverable_session(holder: StoreHolder, session_id: str) -> bool:
    """Un-archive a session archived by a recoverable accident (ws_orphan_reap, agent_close);
    deliberate archives are left alone. True when un-archived.

    Registry-style lookups (Bot Mode's canonical "Bot Chat") use this to resurrect a row the ws-orphan
    reaper (``ws_orphan_reap``) or older agent cleanup (``agent_close``) archived: those ends are
    accidents, not user intent, so the identity-scoped canonical chat must survive them (#92687).
    Sessions archived with no end_reason or an explicit boundary reason (user archived deliberately,
    ``session_reset``, …) are left untouched — returns ``False`` for those, ``True`` only when the row
    was archived for a recoverable reason and is now un-archived (whole compression lineage, via
    :meth:`set_session_archived`).
    """
    if not session_id:
        return False
    try:
        row = get_session(holder, session_id)
    except Exception:  # health: allow BLE001 -- an unreadable row is not unarchived
        return False
    if not row or not row.get("archived"):
        return False
    # The accidental stamp lives on the live TIP; judge recoverability there.
    tip = row
    try:
        tip_id = get_compression_tip(holder, session_id) or session_id
        if tip_id != session_id:
            tip = get_session(holder, tip_id) or row
    except Exception:  # health: allow BLE001 S110 -- a failed tip lookup checks the row itself
        pass
    if (tip.get("end_reason") or "") not in RECOVERABLE_END_REASONS:
        return False
    if not set_session_archived(holder, session_id, False)[0]:
        return False
    # Clear the accidental end stamp, or a LATER deliberate archive (which never
    # writes end_reason) would auto-resurrect on the next lookup.
    holder._write_sql(
        "UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = ?", (tip["id"],),
    )
    return True


def set_session_pinned(holder: StoreHolder, session_id: str, pinned: bool) -> bool:
    """Pin/unpin a session and its compression lineage (pins are exempt from the auto_archive sweep).
    Pinning also clears ``hidden``: a pin means "keep this visible", and a hidden+pinned row is
    otherwise absent from both the default listing and the pinned back-fill (see #106171).
    Exempt the canonical Bot Chat (hidden + exact registry title): the desktop contract keeps it
    hidden and reachable only through the bot row, and unhiding it would also disable the
    rename guard in ``_set_session_title`` that protects its identity (see review on #106180)."""
    result = _set_lineage_column(holder, "pinned", session_id, int(pinned))
    if pinned:
        row = get_session(holder, session_id)
        is_canonical_bot_chat = bool(row) and bool(row.get("hidden")) and (
            (row.get("title") or "") == CANONICAL_BOT_CHAT_TITLE
        )
        if not is_canonical_bot_chat:
            _set_lineage_column(holder, "hidden", session_id, 0)
    return result


def set_session_hidden(holder: StoreHolder, session_id: str, hidden: bool) -> bool:
    """Hide/unhide a session and its compression lineage from the default listing; still resumable."""
    return _set_lineage_column(holder, "hidden", session_id, int(hidden))


def set_session_read(holder: StoreHolder, session_id: str, read: bool = True) -> bool:
    """Mark read/unread across the compression lineage. ``last_read_at`` is a watermark: unread when
    activity postdates it (no write on the message path). NULL = never tracked = read; 0 = unread."""
    return _set_lineage_column(holder, "last_read_at", session_id, time.time() if read else 0.0)


def session_unread(session_row: dict[str, Any]) -> bool:
    """Unread = activity postdates the ``last_read_at`` watermark (NULL = read)."""
    last_read = session_row.get("last_read_at")
    if last_read is None:
        return False
    last_active = session_row.get("last_active") or session_row.get("started_at")
    return float(last_active or 0) > float(last_read)


_SESSION_COMPACT_EXCLUDED = frozenset(
    {"system_prompt", "system_prompt_hash", "git_metadata_generation"}
)


def _chain_search_where(where_sql: str, id_needle: str, search_needle: str) -> tuple[str, list[Any]]:
    """Extend ``where_sql`` with the id_query / search_query filters: a row is admitted when its own
    id or any id in its forward compression chain matches (search also matches titles and a
    punctuation-stripped form so ``an94`` finds ``AN-94``); chain membership bounds the LIKE."""
    params: list[Any] = []
    clauses: list[str] = []
    def like(needle: str) -> str:
        return f"%{_escape_like(needle)}%"
    if id_needle:
        clauses.append(
            "EXISTS (SELECT 1 FROM chain cq        WHERE cq.root_id = s.id"
            "          AND LOWER(cq.cur_id) LIKE ? ESCAPE '\\')"
        )
        params.append(like(id_needle))
    if search_needle:
        compact_needle = re.sub(r"[\W_]+", "", search_needle)
        search_clause = (
            "EXISTS (SELECT 1 FROM chain cq JOIN sessions cs ON cs.id = cq.cur_id"
            " WHERE cq.root_id = s.id AND (LOWER(COALESCE(cs.title, '')) LIKE ? ESCAPE '\\'"
            " OR LOWER(cq.cur_id) LIKE ? ESCAPE '\\'"
        )
        params.extend([like(search_needle)] * 2)
        if compact_needle:
            search_clause += (
                " OR REPLACE(REPLACE(REPLACE(REPLACE(LOWER(COALESCE(cs.title, '')),"
                " '-', ''), '_', ''), '.', ''), ' ', '') LIKE ? ESCAPE '\\'"
            )
            params.append(like(compact_needle))
        clauses.append(search_clause + "))")
    if not clauses:
        return where_sql, params
    combined = " AND ".join(clauses)
    return (f"{where_sql} AND {combined}" if where_sql else f"WHERE {combined}"), params


def list_recent_sessions_bounded(
    holder: StoreHolder,
    *,
    limit: int = 20,
    exclude_sources: list[str] | None = None,
    timeout_seconds: float = 3.0,
    candidate_limit: int | None = None,
    lineage_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Latency-bounded recent-conversation browse (``session_search()``): preselect a small candidate set
    from the indexed durable activity timestamp (fallback ``started_at``), resolve only those across
    compression ancestry/chains, then hydrate activity/previews for that bounded set. Lineage traversal
    uses ``UNION`` plus a total-row ceiling so a corrupt cycle or a deep/branching lineage cannot defeat
    the bound; a lineage that hits the ceiling before a terminal root/tip is omitted, not expanded.
    A cooperative SQLite progress deadline interrupts sustained work past ``timeout_seconds`` and raises
    ``TimeoutError`` (cheap statements may finish between callbacks). Supports only the agent-tool
    browse filters; rich callers keep using :meth:`list_sessions_rich`."""
    limit = max(1, int(limit))
    timeout_seconds = max(0.0, float(timeout_seconds))
    if candidate_limit is None:
        candidate_limit = max(128, limit * 8)
    candidate_limit = max(limit, min(int(candidate_limit), 2048))
    if lineage_limit is None:
        lineage_limit = min(8192, candidate_limit * 8)
    lineage_limit = max(candidate_limit, min(int(lineage_limit), 8192))

    candidate_clauses = [
        "s.archived = 0",
        "s.hidden = 0",
        f"{_delegate_from_json('s.model_config')} IS NULL",
    ]
    candidate_params: list[Any] = []
    if exclude_sources:
        placeholders = ",".join("?" for _ in exclude_sources)
        candidate_clauses.append(f"s.source NOT IN ({placeholders})")
        candidate_params.extend(exclude_sources)
    candidate_where = " AND ".join(candidate_clauses)

    # A compression continuation is an implementation edge, unlike /new
    # reset and /branch children which are independent user-visible
    # conversations.  The same predicate is used in both directions so a
    # candidate tip maps to its logical root and the root maps back to the
    # freshest live tip.
    compression_parent_edge = f"""
            parent.end_reason = 'compression'
            AND child.parent_session_id = parent.id
            AND json_extract(
                COALESCE(child.model_config, '{{}}'), '$._branched_from'
            ) IS NULL
            AND {_delegate_from_json('child.model_config')} IS NULL
            AND COALESCE(child.source, '') != 'tool'
        """

    query = f"""
            WITH RECURSIVE
            recent_candidates(id) AS (
                SELECT s.id
                FROM sessions s
                WHERE {candidate_where}
                ORDER BY COALESCE(s.last_activity_at, s.started_at) DESC,
                         s.started_at DESC, s.id DESC
                LIMIT ?
            ),
            ancestors(candidate_id, cur_id) AS (
                SELECT id, id FROM recent_candidates
                UNION
                SELECT a.candidate_id, parent.id
                FROM ancestors a
                JOIN sessions child ON child.id = a.cur_id
                JOIN sessions parent ON {compression_parent_edge}
                LIMIT ?
            ),
            candidate_roots(root_id) AS (
                SELECT DISTINCT a.cur_id
                FROM ancestors a
                JOIN sessions child ON child.id = a.cur_id
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM sessions parent
                    WHERE {compression_parent_edge}
                )
            ),
            chain(root_id, cur_id) AS (
                SELECT root_id, root_id FROM candidate_roots
                UNION
                SELECT c.root_id, child.id
                FROM chain c
                JOIN sessions parent ON parent.id = c.cur_id
                JOIN sessions child ON {compression_parent_edge}
                LIMIT ?
            ),
            chain_rows AS (
                SELECT
                    c.root_id,
                    c.cur_id,
                    {_sql_session_last_active_by_id('c.cur_id')} AS activity,
                    CASE WHEN EXISTS (
                        SELECT 1
                        FROM sessions parent
                        JOIN sessions child ON {compression_parent_edge}
                        WHERE parent.id = c.cur_id
                    ) THEN 0 ELSE 1 END AS is_tip
                FROM chain c
            ),
            ranked_tips AS (
                SELECT root_id, cur_id, activity,
                       ROW_NUMBER() OVER (
                           PARTITION BY root_id
                           ORDER BY activity DESC, cur_id DESC
                       ) AS rank_in_root
                FROM chain_rows
                WHERE is_tip = 1
            )
            SELECT
                tip.id,
                tip.source,
                tip.model,
                COALESCE(tip.title, s.title) AS title,
                s.started_at AS started_at,
                tip.ended_at,
                tip.end_reason,
                tip.message_count,
                tip.tool_call_count,
                rt.activity AS last_active,
                {_sql_preview_raw('tip.id')},
                CASE WHEN s.id != tip.id THEN s.id ELSE NULL END
                    AS _lineage_root_id
            FROM ranked_tips rt
            JOIN sessions s ON s.id = rt.root_id
            JOIN sessions tip ON tip.id = rt.cur_id
            WHERE rt.rank_in_root = 1
              AND s.archived = 0
              AND s.hidden = 0
              AND {_LISTABLE_CHILD_SQL}
              AND {_delegate_from_json('s.model_config')} IS NULL
            ORDER BY rt.activity DESC, s.started_at DESC, tip.id DESC
            LIMIT ?
        """
    params = candidate_params + [
        candidate_limit,
        lineage_limit,
        lineage_limit,
        limit,
    ]
    deadline = time.monotonic() + timeout_seconds
    interrupted_by_deadline = False

    def _deadline_progress_handler() -> int:
        nonlocal interrupted_by_deadline
        if time.monotonic() >= deadline:
            interrupted_by_deadline = True
            return 1
        return 0

    try:
        with holder._read_ctx() as conn:
            conn.set_progress_handler(_deadline_progress_handler, 1000)
            try:
                rows = conn.execute(query, params).fetchall()
            finally:
                conn.set_progress_handler(None, 0)
    except sqlite3.OperationalError as exc:
        if interrupted_by_deadline and "interrupt" in str(exc).lower():
            raise TimeoutError(
                f"recent-session browse exceeded {timeout_seconds:g}s deadline"
            ) from exc
        raise

    sessions = []
    for row in rows:
        session = _session_row_dict(row)
        session["preview"] = _shape_preview(session.pop("_preview_raw", ""))
        session["unread"] = session_unread(session)
        sessions.append(session)
    return sessions


def _list_row(row: sqlite3.Row) -> dict[str, Any]:
    """Project a list_sessions_rich row: shape the preview, drop internal ordering columns."""
    s = _session_row_dict(row)
    s["preview"] = _shape_preview(s.pop("_preview_raw", ""))
    s.pop("_effective_last_active", None)
    return s


def list_sessions_rich(
    holder: StoreHolder, source: str | None = None, sources: list[str] | None = None, exclude_sources: list[str] | None = None,
    cwd_prefix: str | None = None, limit: int = 20, offset: int = 0, include_children: bool = False,
    min_message_count: int = 0, project_compression_tips: bool = True,
    order_by_last_active: bool = False, include_archived: bool = False, archived_only: bool = False,
    id_query: str | None = None, search_query: str | None = None, compact_rows: bool = False,
    include_pinned: bool = False, session_key: str | None = None, include_hidden: bool = False,
    include_subagents: bool = False, local_created_sources: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """List sessions with preview and ``last_active`` in one query. ``order_by_last_active`` sorts
    by the chain TIP via a recursive CTE (the only path honouring ``id_query`` / ``search_query``);
    ``include_pinned`` back-fills pins the page missed, still obeying the other
    filters except archived: a pin is an explicit keep, so a pinned row stamped
    archived must still return. ``local_created_sources`` restricts ``source='local'``
    rows to those surfaces before LIMIT/OFFSET."""
    holder.flush_for_read()  # rows carry token/cost totals
    where_clauses, params = _session_filter_where(
        exclude_children=not include_children, source=source, sources=sources, session_key=session_key,
        exclude_sources=exclude_sources, cwd_prefix=cwd_prefix, min_message_count=min_message_count,
        archived_only=archived_only, include_archived=include_archived, include_subagents=include_subagents,
        local_created_sources=local_created_sources,
    )
    # The archived-only view is the recovery surface for rows that dropped out of every
    # default list: a session that is archived AND hidden (Bot Mode marks its sessions
    # hidden) must still be reachable there, or nothing but direct DB access can bring
    # it back (#90946).
    if not include_hidden and not archived_only:
        where_clauses.append("s.hidden = 0")
    heads = {}
    from persistence.sqlite.conversation._store.local import exclude_superseded_segments
    if not include_children:
        # A local reset lineage is one conversation (hermes_state_local.local_lineage_index):
        # earlier segments leave the page in SQL, before LIMIT/OFFSET, not after it.
        heads = exclude_superseded_segments(holder, where_clauses, params)
    where_sql = _where_sql(where_clauses)
    # Shared projection head of the three list queries (whitespace is part of the SQL text).
    select_head = (
        f"SELECT {_compact_session_cols(holder.db_path.parent) if compact_rows else 's.*'}"
        + ("" if compact_rows else ", COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved")
        + f",\n                    {_sql_preview_raw()},\n                    "
    )
    prompt_join = (
        "" if compact_rows else "LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash"
    )
    from_sessions = f"FROM sessions s\n                {prompt_join}"
    if order_by_last_active:
        # The CTE walks compression-continuation edges forward from the admitted
        # rows; MAX over the chain gives effective_last_active in SQL. Do NOT
        # require child.started_at >= parent.ended_at: races insert the
        # continuation before ended_at is written.
        #
        # The page is ranked and cut on ids alone (``page``); only its rows then
        # pay for the projection. Projecting in the ranked query made SQLite
        # build every admitted row's full result — s.*, the resolved system
        # prompt, the preview subquery over message content — before LIMIT:
        # 4-11s per page at 25k rows, for LIMIT 1 as much as LIMIT 200.
        #
        # Unless a search needs every row's chain (it matches a row's own id and
        # title through it), only rows that HAVE a child seed the walk. A
        # childless row's chain is itself alone, so its MAX is just its own
        # last_active, which the outer COALESCE reads directly. Seeding every
        # admitted row made the CTE + GROUP BY materialise the whole table on
        # every page: ~12s of a 13s list on a 25k-session store where 1 row had
        # a child, at LIMIT 1 or 200.
        id_needle = (id_query or "").strip().lower()
        search_needle = (search_query or "").strip().lower()
        seed_where = where_sql if (id_needle or search_needle) else _where_sql(where_clauses + [
            "EXISTS (SELECT 1 FROM sessions _seed_c WHERE _seed_c.parent_session_id = s.id)"])
        outer_where, id_params = _chain_search_where(where_sql, id_needle, search_needle)
        query = f"""
                WITH RECURSIVE chain(root_id, cur_id) AS (
                    SELECT s.id, s.id FROM sessions s {seed_where}
                    UNION ALL
                    SELECT c.root_id, child.id
                    FROM chain c
                    JOIN sessions parent ON parent.id = c.cur_id
                    JOIN sessions child ON child.parent_session_id = c.cur_id
                    WHERE parent.end_reason = 'compression'
                      AND {_sql_json_extract('child.model_config', '$._branched_from')} IS NULL
                      AND {_sql_json_extract('child.model_config', '$._delegate_from')} IS NULL
                      AND NOT ({_RESET_CHILD_SQL.format(a='child')})
                      AND COALESCE(child.source, '') != 'tool'
                ),
                chain_max AS (
                    SELECT
                        root_id,
                        MAX({_sql_session_last_active_by_id("cur_id")}) AS effective_last_active
                    FROM chain
                    GROUP BY root_id
                ),
                page AS (
                    SELECT s.id AS page_id,
                        COALESCE(cm.effective_last_active, {_sql_session_last_active("s")},
                                 {_sql_in_window("s.started_at")}) AS effective_last_active
                    FROM sessions s
                    LEFT JOIN chain_max cm ON cm.root_id = s.id
                    {outer_where}
                    ORDER BY effective_last_active DESC, s.started_at DESC, s.id DESC
                    LIMIT ? OFFSET ?
                )
                {select_head}{_sql_session_last_active("s")} AS last_active,
                    page.effective_last_active AS _effective_last_active
                FROM page
                JOIN sessions s ON s.id = page.page_id
                {prompt_join}
                ORDER BY _effective_last_active DESC, s.started_at DESC, s.id DESC
            """
        params = params + params + id_params + [limit, offset]  # WHERE binds twice (seed + outer)
    else:
        query = f"""
                {select_head}{_sql_session_last_active("s")} AS last_active
                {from_sessions}
                {where_sql}
                ORDER BY s.started_at DESC
                LIMIT ? OFFSET ?
            """
        params.extend([limit, offset])
    sessions = [_list_row(row) for row in holder._read_all(query, params)]
    # Pinned back-fill runs BEFORE compression projection so a back-filled root
    # projects to its tip like any other row. Do not inherit the archived
    # constraint: the sidebar lists with include_archived=False, and a pin
    # must stay reachable even when that row is also archived.
    if include_pinned:
        seen_ids = {s["id"] for s in sessions}
        pinned_clauses, pinned_params = _session_filter_where(
            exclude_children=not include_children, source=source, sources=sources,
            session_key=session_key, exclude_sources=exclude_sources, cwd_prefix=cwd_prefix,
            min_message_count=min_message_count, archived_only=False, include_archived=True,
            include_subagents=include_subagents, local_created_sources=local_created_sources,
        )
        if not include_hidden and not archived_only:
            pinned_clauses.append("s.hidden = 0")
        if heads:
            exclude_superseded_segments(holder, pinned_clauses, pinned_params)
        pinned_clauses.append("s.pinned = 1")
        pinned_where = _where_sql(pinned_clauses)
        pinned_query = f"""
                {select_head}{_sql_session_last_active("s")} AS last_active
                {from_sessions}
                {pinned_where}
                ORDER BY s.started_at DESC
            """
        for row in holder._read_all(pinned_query, pinned_params):
            s = _list_row(row)
            if s["id"] not in seen_ids:
                seen_ids.add(s["id"])
                sessions.append(s)
    if project_compression_tips and not include_children:
        from persistence.sqlite.conversation._store.session_list import project_compression_tips
        sessions = project_compression_tips(holder, sessions, compact_rows)
    if heads:
        from persistence.sqlite.conversation._store.local import annotate_local_lineages
        sessions = annotate_local_lineages(sessions, heads)
    # last_read_at is lineage-stamped, so root and tip watermarks agree.
    for s in sessions:
        s["unread"] = session_unread(s)
    return sessions


def session_lifecycle_statuses(holder: StoreHolder, session_ids: list[str]) -> dict[str, str]:
    """``{session_id: status}`` from each session's LAST message row (``'empty'`` when none); one
    query, MAX(id) per session joined back — never scans transcripts."""
    ids = [sid for sid in (session_ids or []) if sid]
    if not ids:
        return {}
    statuses: dict[str, str] = {sid: "empty" for sid in ids}
    rows = holder._read_all(f"""
            SELECT m.session_id, m.role,
                   m.tool_calls IS NOT NULL AS has_tool_calls,
                   m.finish_reason
            FROM messages m
            JOIN (
                SELECT session_id, MAX(id) AS max_id
                FROM messages
                WHERE session_id IN ({_session_ids_placeholders(ids)})
                GROUP BY session_id
            ) latest ON m.id = latest.max_id
        """, ids)
    for row in rows:
        statuses[row["session_id"]] = classify_session_status(
            role=row["role"], has_tool_calls=bool(row["has_tool_calls"]),
            finish_reason=row["finish_reason"],
        )
    return statuses


def _is_explicit_branch_session(holder: StoreHolder, session_id: str) -> bool:
    """Copied user-facing branch (``_branched_from``)? Branches own a copied transcript;
    compression continuations need the parent's archived rows."""
    if not session_id:
        return False
    row = holder._read_one("SELECT model_config FROM sessions WHERE id = ?", (session_id,))
    return row is not None and bool(_parse_model_config(row[0]).get("_branched_from"))


def _session_lineage_root_to_tip(holder: StoreHolder, session_id: str) -> list[str]:
    if not session_id:
        return [session_id]
    chain: list[str] = []
    current = session_id
    with holder._read_ctx() as conn:
        while current and current not in chain and len(chain) < 100:
            chain.append(current)
            row = conn.execute(
                "SELECT parent_session_id FROM sessions WHERE id = ?", (current,),
            ).fetchone()
            if row is None:
                break
            current = row[0]
    return list(reversed(chain)) or [session_id]


def search_sessions(
    holder: StoreHolder, source: str | Sequence[str] | None = None, limit: int = 20, offset: int = 0,
    workspace_key: str | None = None, local_created_sources: Sequence[str] | None = None,
    skip_launch_stubs: bool = False,
) -> list[dict[str, Any]]:
    """Sessions MRU-first with a computed ``last_active``; ``workspace_key`` scopes to one workspace
    so ``hermes -c``/``--resume`` picks its last session. ``source`` may be one label or several.
    ``local_created_sources`` restricts ``source='local'`` rows before LIMIT/OFFSET, and
    ``skip_launch_stubs`` drops empty launch stubs (launch_stub_filter_sql) before it too."""
    where_clauses = [launch_stub_filter_sql(holder)] if skip_launch_stubs else []
    params: list = []
    if source:
        sources = [source] if isinstance(source, str) else list(source)
        clause, bound = _source_match_clause(sources, local_created_sources)
        where_clauses.append(clause)
        params.extend(bound)
    if workspace_key:
        ws_clause, ws_params = _workspace_key_clause(workspace_key)
        where_clauses.append(ws_clause)
        params.extend(ws_params)
    return [_session_row_dict(row) for row in holder._read_all(
        "SELECT s.*, COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved, "
        f"{_sql_session_last_active('s')} AS last_active "
        "FROM sessions s LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash "
        f"{_where_sql(where_clauses, ' ')} "
        "ORDER BY last_active DESC, s.started_at DESC, s.id DESC LIMIT ? OFFSET ?",
        [*params, limit, offset],
    )]


def session_count(
    holder: StoreHolder, source: str | None = None, sources: list[str] | None = None, cwd_prefix: str | None = None,
    min_message_count: int = 0, include_archived: bool = False, archived_only: bool = False,
    exclude_children: bool = False, exclude_sources: list[str] | None = None, include_subagents: bool = False,
    local_created_sources: Sequence[str] | None = None,
) -> int:
    """Count sessions with list_sessions_rich's filters so a paired "load more" total matches."""
    where_clauses, params = _session_filter_where(
        exclude_children=exclude_children, source=source, sources=sources,
        exclude_sources=exclude_sources, cwd_prefix=cwd_prefix, min_message_count=min_message_count,
        archived_only=archived_only, include_archived=include_archived, include_subagents=include_subagents,
        local_created_sources=local_created_sources,
    )
    if exclude_children:
        from persistence.sqlite.conversation._store.local import exclude_superseded_segments
        exclude_superseded_segments(holder, where_clauses, params)
    return holder._read_one(f"SELECT COUNT(*) FROM sessions s{_where_sql(where_clauses, ' ')}", params)[0]


def session_count_ge(holder: StoreHolder, n: int = 1) -> bool:
    """At least N sessions exist (archived included); LIMIT short-circuits session_count()'s scan."""
    return len(holder._read_all("SELECT 1 FROM sessions LIMIT ?", (n,))) >= n


def session_count_by_source(
    holder: StoreHolder, *, include_archived: bool = False, archived_only: bool = False,
    exclude_children: bool = False,
) -> dict[str, int]:
    """``{source: count}`` via one GROUP BY; ``exclude_children`` mirrors listing visibility."""
    where_clauses, params = _session_filter_where(
        exclude_children=exclude_children, archived_only=archived_only,
        include_archived=include_archived,
    )
    if exclude_children:
        from persistence.sqlite.conversation._store.local import exclude_superseded_segments
        exclude_superseded_segments(holder, where_clauses, params)
    with holder._read_ctx() as conn:
        if holder._conn is None:
            raise RuntimeError("SessionDB connection is closed")
        rows = conn.execute(
            "SELECT COALESCE(NULLIF(s.source, ''), 'cli') AS source, COUNT(*) AS count "
            f"FROM sessions s{_where_sql(where_clauses, ' ')} "
            "GROUP BY COALESCE(NULLIF(s.source, ''), 'cli') ORDER BY count DESC", params,
        ).fetchall()
    return {str(row["source"]): int(row["count"] or 0) for row in rows}


def declared_scope_identity(holder: StoreHolder, session_id: str) -> tuple[bool, str]:
    """(is_fork_child, source) in ONE read (prompt_cache_scope needs both from the same row).
    Missing row → (False, ""); DB errors propagate (fail closed).

    ``agent/prompt_cache_scope.py`` needs both to resolve a host-declared conversation scope, and both
    live on the same ``sessions`` row; asking for them separately read that row twice per resolution
    (@teknium1 on 98811). The marker rules stay here, beside :meth:`is_explicit_fork_child`, instead of
    being re-implemented by the caller. See #98811.
    """
    session = get_session(holder, session_id)
    if not session:
        return False, ""
    return _is_explicit_fork_child_row(holder, session), str(session.get("source") or "").strip()


def get_session_delete_targets(holder: StoreHolder, session_id: str) -> list[str]:
    """Rows :meth:`delete_session` would remove: the session, then its recursive delegate children
    (branch/compression children are orphaned, not deleted)."""
    with holder._read_ctx() as conn:
        if not conn.execute("SELECT 1 FROM sessions WHERE id = ? LIMIT 1", (session_id,)).fetchone():
            return []
        # Use the borrowed read connection, never self._conn: handing the shared writer connection to a
        # helper here executes on it without self._lock — the same unsynchronized-read class as
        # #99349/#90734.
        from persistence.sqlite.conversation._store.local import owned_lineage_ids
        scope = owned_lineage_ids(conn, session_id)
        delegate_ids = _collect_delegate_child_ids(conn, scope)
    return [session_id, *sorted(set(scope) - {session_id}), *sorted(delegate_ids)]


def _expand_compression_lineage(holder: StoreHolder, conn, session_ids: list[str]) -> list[str]:
    """Expand *session_ids* to every member of their compression chains.

    The dashboard sessions list surfaces one row per logical conversation: compression
    roots are projected forward to their latest continuation, and the surfaced row
    carries the *tip's* id (see :meth:`list_sessions_rich`). Deleting only that tip row
    leaves the root behind, and the next list re-projects the root onto the previous
    link — the "deleted" conversation reappears (#57543).

    Walks compression-continuation edges in BOTH directions from the given ids:
    backward to the chain root, and forward through every continuation — including
    stale sibling continuations (e.g. ``ws_orphan_reap`` leftovers), which would
    otherwise be orphaned into brand-new visible top-level rows by the delete itself.
    The edge definition matches ``_CHAIN_STEP_SQL`` / ``get_compression_tip`` and the
    list-projection CTE: parent ended with ``reason='compression'``, child is not a
    branch / reset / delegate / tool session. Branch, reset and delegate children are
    deliberately NOT pulled in — a branch/reset is its own conversation, and delegates
    are handled by the existing cascade in the delete paths.

    Runs inside the caller's write transaction (*conn*). The walk is defensively
    bounded like ``get_compression_chain`` — ``seen`` guards cycles.
    """
    seen = set(session_ids)
    frontier = list(session_ids)
    for _ in range(_CHAIN_CAP):
        if not frontier:
            break
        targets = _expand_compression_lineage_ids(conn, frontier)
        frontier = [sid for sid in targets if sid not in seen]
        seen.update(frontier)
    # A local reset lineage (creation id + every segment, compression children included) is one
    # listed conversation: deleting any segment must not strand the others without an owner.
    from persistence.sqlite.conversation._store.local import local_conversation_ids
    for sid in list(seen):
        seen.update(local_conversation_ids(conn, sid))
    return list(seen)


def delete_session(
    holder: StoreHolder, session_id: str, sessions_dir: Optional[Path] = None,
    expected_delete_ids: Optional[list[str]] = None,
    expected_display_messages: Optional[dict[str, list[dict[str, Any]]]] = None,
    exclude_active_write_guards: bool = False,
    include_compression_chain: bool = False, *, media_root: str,
) -> tuple[str, ...]:
    """Delete a session and its messages; delegate children cascade, branch/compression children
    are orphaned. Optional expected ids fence delegate drift; expected display snapshots fence
    transcript drift. Both checks run inside the same write transaction as deletion.
    With ``exclude_active_write_guards``, raises :class:`SessionActiveWriteGuardError` if the row
    is protected by an active turn lease or compression lock.

    With ``include_compression_chain=True`` the whole compression chain the session belongs to
    is deleted with it (root and every continuation) — the dashboard delete endpoints use this
    variant because their list rows carry chain-*tip* ids; deleting only that physical row
    would leave the rest of the chain to resurface on the next list (#57543). Default False so
    callers that operate on a single physical session keep the historic contract. Routing
    through :meth:`delete_sessions` keeps the chain-aware guard/counting semantics in one
    place; its return value counts the *requested* ids that existed, so ``> 0`` is exactly
    "the session was found"."""
    if include_compression_chain:
        return delete_sessions(holder,
            [session_id], sessions_dir=sessions_dir,
            exclude_active_write_guards=exclude_active_write_guards,
            include_compression_chain=True, media_root=media_root,
        )[1]
    removed_ids: list[str] = []
    expected_ids = set(expected_delete_ids) if expected_delete_ids is not None else None
    def _do(conn):
        if conn.execute("SELECT 1 FROM sessions WHERE id = ? LIMIT 1", (session_id,)).fetchone() is None:
            return False
        # A local reset/compression conversation deletes whole from any segment: the listed row
        # is its latest segment, and a lone segment delete strands the owner's policy and FIFO.
        from persistence.sqlite.conversation._store.local import owned_lineage_ids
        scope = owned_lineage_ids(conn, session_id)
        target_ids = (
            [*scope, *_collect_delegate_child_ids(conn, scope)]
            if exclude_active_write_guards or expected_ids is not None else None
        )
        if exclude_active_write_guards and _guarded_ids(holder, conn, target_ids):
            # Delegate children cascade with the root, so a guard on any of them refuses too.
            raise SessionActiveWriteGuardError(
                f"session '{session_id}' (or a delegate child) has an active turn lease or compression lock"
            )
        if expected_ids is not None and expected_ids != set(target_ids):
            return False
        if expected_display_messages is not None and any(
            _display_messages_from_conn(holder, conn, covered_id) != expected
            for covered_id, expected in expected_display_messages.items()
        ):
            return False
        from persistence.sqlite.conversation._store.retirement import retire_sessions
        retire_sessions(conn, scope, media_root=media_root)
        delegate_ids = _delete_delegate_children(conn, scope, media_root=media_root)
        for sid in scope:  # orphan remaining children (branches) so FK is satisfied
            conn.execute("UPDATE sessions SET parent_session_id = NULL WHERE parent_session_id = ?", (sid,))
            conn.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
        conn.executemany("DELETE FROM sessions WHERE id = ?", [(sid,) for sid in scope])
        _delete_unreferenced_system_prompts(conn)
        removed_ids.extend([session_id, *sorted(set(scope) - {session_id}), *sorted(delegate_ids)])
        return True
    holder._execute_write(_do)
    return tuple(removed_ids)


def discard_unadmitted_session(holder: StoreHolder, session_id: str, *, media_root: str) -> tuple[str, ...]:
    """Abort a row this same flow just created and nobody has been admitted against (seed-copy
    compensation): remove it WITHOUT the retirement fence so the id can be lazily recreated.
    A row with any admission or worker receipt is a real session and takes the fenced
    :meth:`delete_session` path instead."""
    def _do(conn):
        if conn.execute(
            "SELECT 1 FROM session_admissions WHERE target_session_id=? UNION ALL "
            "SELECT 1 FROM worker_executions WHERE session_id=? LIMIT 1", (session_id, session_id)).fetchone():
            return None
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cur = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return (session_id,) if cur.rowcount > 0 else ()
    result = holder._execute_write(_do)
    if result is None:
        return delete_session(holder, session_id, media_root=media_root)
    return result


def delete_session_if_empty(holder: StoreHolder, session_id: str, sessions_dir: Optional[Path] = None, *, media_root: str) -> tuple[str, ...]:
    """Delete *session_id* only if it has no messages, no title and no children; check and delete
    share one transaction so a concurrent flush can't be lost. A row under an active turn lease
    or compression lock is never a candidate: the emptiness predicate reads committed state, so a
    row whose first turn is already leased but not yet flushed would be deleted mid-turn
    (#123583). A guarded row is simply not deleted (returns ``False``), like any non-empty row."""
    def _do(conn):
        from persistence.sqlite.conversation._store.local import owned_lineage_ids
        if _guarded_ids(holder, conn, [session_id]) or owned_lineage_ids(conn, session_id) != [session_id]:
            return False  # an empty reset tip is still its conversation's current target
        eligible = conn.execute(
            """
                SELECT 1 FROM sessions
                WHERE id = ?
                  AND title IS NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM messages WHERE messages.session_id = sessions.id
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM sessions child
                      WHERE child.parent_session_id = sessions.id
                  )
                """,
            (session_id,),
        ).fetchone()
        if eligible is None:
            return False
        # Same BEGIN IMMEDIATE transaction as the check above: retire the ledger rows
        # (ON DELETE RESTRICT) and delete without re-evaluating eligibility. Housekeeping, not
        # a user delete: its fence ages out at the horizon (an untitled row is never a Bot Chat).
        from persistence.sqlite.conversation._store.retirement import retire_sessions
        retire_sessions(conn, [session_id], expiring=[session_id], media_root=media_root)
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        _delete_unreferenced_system_prompts(conn)
        return True
    return (session_id,) if holder._execute_write(_do) else ()


def delete_sessions(
    holder: StoreHolder, session_ids: list[str], sessions_dir: Optional[Path] = None,
    exclude_active_write_guards: bool = False, skipped_ids: Optional[list[str]] = None,
    include_compression_chain: bool = False, *, media_root: str,
) -> tuple[int, tuple[str, ...]]:
    """Bulk delete with :meth:`delete_session` semantics per row, in ONE transaction. Unknown ids
    are skipped (UI selection can race another tab's delete). With ``exclude_active_write_guards``, rows
    protected by an active turn lease or compression lock are skipped and, when given, appended
    to ``skipped_ids`` so callers can tell the user. Returns the number deleted.

    With ``include_compression_chain=True``, each selected id is expanded to its full
    compression chain (root + every continuation, including stale sibling continuations)
    via :meth:`_expand_compression_lineage` — the dashboard rows carry chain-tip ids, so
    without the expansion a "deleted" conversation resurfaces as the previous chain link on
    the next list reload (#57543). The returned count still reflects the *requested* rows,
    not the expanded chain links, so the caller's toast matches the user's selection."""
    unique_ids = list({sid for sid in session_ids or () if isinstance(sid, str) and sid})
    if not unique_ids:
        return 0, ()
    removed_ids: list[str] = []
    def _do(conn):
        existing = [row["id"] for chunk in _id_chunks(unique_ids) for row in conn.execute(
            f"SELECT id FROM sessions WHERE id IN ({_session_ids_placeholders(chunk)})", chunk,
        ).fetchall()]
        if not existing:
            return 0
        from persistence.sqlite.conversation._store.local import owned_lineage_ids

        def expand(ids):
            # A local reset/compression segment deletes as its whole conversation (delete_session).
            if include_compression_chain:
                return _expand_compression_lineage(holder, conn, ids)
            return list(dict.fromkeys(x for sid in ids for x in owned_lineage_ids(conn, sid)))
        # Expand BEFORE the guard pass so a guarded lineage member (active turn lease /
        # compression lock anywhere in it) keeps its selected root listed too.
        expanded = expand(existing)
        guard_pool = [*expanded, *_collect_delegate_child_ids(conn, expanded)]
        if exclude_active_write_guards:
            # A root is skipped when it or any delegate child it would cascade is guarded, so the
            # cascade below never deletes a guarded row reported back as kept. When the chain
            # expansion is on, a guard on ANY lineage member keeps its selected root: deleting
            # a torn chain would re-surface it (the root re-projects onto a surviving link).
            # One batched check first; per-root attribution only when something is guarded.
            active_ids: set = set()
            if _guarded_ids(holder, conn, guard_pool):
                if include_compression_chain:
                    for sid in existing:
                        member_lineage = set(_expand_compression_lineage(holder, conn, [sid]))
                        member_lineage.add(sid)
                        if _guarded_ids(holder, conn, [*member_lineage,
                                *_collect_delegate_child_ids(conn, list(member_lineage))]):
                            active_ids.add(sid)
                else:
                    active_ids = {
                        sid for sid in existing
                        if _guarded_ids(holder, conn, [*(members := expand([sid])),
                                                    *_collect_delegate_child_ids(conn, members)])
                    }
            existing = [sid for sid in existing if sid not in active_ids]
            if skipped_ids is not None:
                skipped_ids.extend(sorted(active_ids))
            if not existing:
                return 0
            expanded = expand(existing)
        from persistence.sqlite.conversation._store.retirement import retire_sessions
        retire_sessions(conn, expanded, media_root=media_root)
        delegate_ids = _delete_delegate_children(conn, expanded, media_root=media_root)
        for chunk in _id_chunks(expanded):
            ph = _session_ids_placeholders(chunk)
            conn.execute(  # orphan children whose parent is in the kill list (FK)
                f"UPDATE sessions SET parent_session_id = NULL WHERE parent_session_id IN ({ph})", chunk,
            )
            conn.execute(f"DELETE FROM messages WHERE session_id IN ({ph})", chunk)
            conn.execute(f"DELETE FROM sessions WHERE id IN ({ph})", chunk)
        _delete_unreferenced_system_prompts(conn)
        removed_ids.extend([*expanded, *delegate_ids])
        # Count the ids the CALLER asked for, not the expanded chain links —
        # the dashboard toast should say "3 deleted" when the user selected
        # 3 rows, however many physical links those rows had.
        return len(existing)
    count = holder._execute_write(_do)
    return count, tuple(removed_ids)


_EMPTY_SESSION_WHERE = (
    "message_count = 0 AND ended_at IS NOT NULL AND archived = 0 AND NOT EXISTS ("
    "SELECT 1 FROM messages WHERE messages.session_id = sessions.id)"
)


def count_empty_sessions(holder: StoreHolder) -> int:
    """Count of empty, ended, non-archived sessions; ended_at guards a fresh session's first message."""
    return holder._read_one(f"SELECT COUNT(*) FROM sessions WHERE {_EMPTY_SESSION_WHERE}")[0]


def delete_empty_sessions(holder: StoreHolder, sessions_dir: Optional[Path] = None, *, media_root: str) -> tuple[str, ...]:
    """Delete every empty, ended, non-archived session in one transaction, orphaning (not cascading)
    children; transcript files are swept too."""
    removed_ids: list[str] = []
    def _do(conn):
        session_ids = {row["id"] for row in conn.execute(
            f"SELECT id FROM sessions WHERE {_EMPTY_SESSION_WHERE}"
        ).fetchall()}
        from persistence.sqlite.conversation._store.retirement import retire_prunable
        session_ids = retire_prunable(conn, sorted(session_ids), media_root=media_root)
        if not session_ids:
            return 0
        for chunk in _id_chunks(session_ids):
            ph = _session_ids_placeholders(chunk)
            conn.execute(f"UPDATE sessions SET parent_session_id = NULL WHERE parent_session_id IN ({ph})", chunk)
            # DELETE FROM messages: a row inserted between the SELECT and here
            # would otherwise dangle (clean FK state).
            conn.execute(f"DELETE FROM messages WHERE session_id IN ({ph})", chunk)
            conn.execute(f"DELETE FROM sessions WHERE id IN ({ph})", chunk)
            removed_ids.extend(chunk)
        _delete_unreferenced_system_prompts(conn)
        return len(session_ids)
    holder._execute_write(_do)
    return tuple(removed_ids)


# Sibling store modules import this module's copies in turn, so their copies are imported after these.
from persistence.sqlite.conversation._store.compression import _CHAIN_CAP, get_compression_tip
from persistence.sqlite.conversation._store.core import CANONICAL_BOT_CHAT_TITLE, _delete_unreferenced_system_prompts, _session_row_dict
from persistence.sqlite.conversation._store.messages import _bump_conversation_generation, _display_messages_from_conn, _is_explicit_fork_child_row
from persistence.sqlite.conversation._store.portability import _compact_session_cols
from persistence.sqlite.conversation._store.write_guards import _guarded_ids

"""Session lifecycle mixin for SessionDB: row upsert/inheritance, lifecycle
flags (end/reopen/archive/pin/hide/read), model_config patching, listing and
counting, delete cascades, and the auto-archive sweep."""

import json
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.session_activity import (
    ActivityProvenance, bound_activity_description, normalize_activity_provenance,
)
from hermes_state_common import (
    _LISTABLE_CHILD_SQL, _NOT_BRANCH_MARKER_CHILD_SQL, _NOT_DELEGATE_MARKER_CHILD_SQL,
    _PREVIEW_ELIGIBLE_SQL,
    _PREVIEW_RAW_SELECT, _RECOVERABLE_END_REASONS,
    _RECOVERABLE_END_REASONS_SQL, _RESET_END_REASONS, _legacy_reset_child_sql, _shape_preview,
    _sql_session_last_active, _sql_session_last_active_by_id, escape_like as _escape_like,
    _placeholders as _session_ids_placeholders,
)

# caplog tests pin the "hermes_state" logger name.
logger = logging.getLogger("hermes_state")


def workspace_key(row: Dict[str, Any]) -> Optional[str]:
    """Workspace grouping key: git repo root, else cwd, else None (branch excluded: a checkout must not
    fragment history)."""
    return (row.get("git_repo_root") or "").strip() or (row.get("cwd") or "").strip() or None


def _delegate_from_json(col: str = "model_config") -> str:
    return f"json_extract(COALESCE({col}, '{{}}'), '$._delegate_from')"


# _merge_model_config_json's "no such row" result — distinct from the legal None
# ("merged config is empty → store NULL").
_MODEL_CONFIG_ROW_MISSING = object()


def _parse_model_config(raw: Any) -> Dict[str, Any]:
    """Tolerant ``model_config`` decode: JSON text or dict -> dict copy; anything else -> {}."""
    if isinstance(raw, str) and raw.strip():
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
    return dict(raw) if isinstance(raw, dict) else {}


def _cwd_prefix_clause(cwd_prefix: str) -> Tuple[str, List[str]]:
    prefix = cwd_prefix.rstrip("/\\") or cwd_prefix
    # ``_``/``%`` are LIKE wildcards but ordinary path characters: unescaped, a
    # prefix also matches sibling directories. The ``=`` arm keeps the raw prefix.
    esc = _escape_like(prefix)
    return (
        "(s.cwd = ? OR s.cwd LIKE ? ESCAPE '\\' OR s.cwd LIKE ? ESCAPE '\\')",
        [prefix, f"{esc}/%", f"{esc}\\\\%"],
    )


def _workspace_key_clause(key: str) -> Tuple[str, List[str]]:
    """WHERE for ``workspace_key(row) == key``: git_repo_root equals ``key``, or (rows predating
    per-session git metadata) cwd is at/under ``key``."""
    prefix = key.rstrip("/\\") or key
    cwd_clause, cwd_params = _cwd_prefix_clause(prefix)
    return (
        f"(s.git_repo_root = ? OR (COALESCE(s.git_repo_root, '') = '' AND {cwd_clause}))",
        [prefix, *cwd_params],
    )


# First user message of a session, shaped by _shape_preview() in Python.
# The indentation is part of the list_sessions_rich SQL text.
_PREVIEW_COL_SQL = f"""COALESCE(
                        (SELECT {_PREVIEW_RAW_SELECT}
                         FROM messages m
                         WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                           AND {_PREVIEW_ELIGIBLE_SQL}
                         ORDER BY m.timestamp, m.id LIMIT 1),
                        ''
                    ) AS _preview_raw"""


def _where_sql(clauses: List[str], lead: str = "") -> str:
    """``WHERE a AND b`` (with *lead* prefix) or "" when there are no clauses."""
    return f"{lead}WHERE {' AND '.join(clauses)}" if clauses else ""


def _session_filter_where(
    *, exclude_children: bool = False, source: str = None, sources: List[str] = None,
    session_key: str = None, exclude_sources: List[str] = None, cwd_prefix: str = None,
    min_message_count: int = 0, archived_only: bool = False, include_archived: bool = False,
) -> Tuple[List[str], List[Any]]:
    """Shared ``sessions s`` WHERE builder so counts line up with listed rows. ``exclude_children``
    hides sub-agent runs and compression continuations but keeps branch/reset children
    (``_LISTABLE_CHILD_SQL``). Clause order is part of the SQL text contract."""
    where: List[str] = []
    params: List[Any] = []
    if exclude_children:
        where += [_LISTABLE_CHILD_SQL, _NOT_DELEGATE_MARKER_CHILD_SQL.format(a="s")]
    # Show roots and user-visible branch/reset sessions, while still hiding sub-agent runs and compression
    # continuations. All four carry parent_session_id, so the shared predicate classifies the edge from
    # stable markers plus legacy-compatible parent metadata. Branch sessions are identified two ways, OR'd
    # for robustness: 1. A stable ``_branched_from`` marker in model_config, written by /branch at creation
    # time. This survives the parent being reopened and re-ended with a different end_reason (e.g.
    # tui_shutdown overwriting 'branched'), which otherwise hides the branch — see issue #20856. 2. The
    # legacy heuristic (parent ended with 'branched' before the child started), covering branch sessions
    # created before the marker existed.
    include_sources = [source] if source else list(sources or [])
    for clause, values in (
        (f"s.source IN ({_session_ids_placeholders(include_sources)})", include_sources),
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


def _collect_delegate_child_ids(conn, parent_ids: List[str]) -> List[str]:
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
        ph = _session_ids_placeholders(frontier)
        cursor = conn.execute(
            f"SELECT id FROM sessions WHERE {df} IN ({ph}) "
            f"OR (parent_session_id IN ({ph}) AND {df} IS NOT NULL)", frontier + frontier,
        )
        frontier = [row["id"] for row in cursor.fetchall() if row["id"] not in found]
        found.update(frontier)
    return [sid for sid in found if sid not in seeds]


def _delete_delegate_children(conn, parent_ids: List[str]) -> List[str]:
    ids = _collect_delegate_child_ids(conn, parent_ids)
    if ids:
        ph = _session_ids_placeholders(ids)
        conn.execute(f"DELETE FROM messages WHERE session_id IN ({ph})", ids)
        # FK safety: orphan any untagged stragglers pointing at a doomed row.
        conn.execute(f"UPDATE sessions SET parent_session_id = NULL WHERE parent_session_id IN ({ph})", ids)
        conn.execute(f"DELETE FROM sessions WHERE id IN ({ph})", ids)
    return ids


# Lifecycle statuses surfaced by session pickers; classified from the final
# message row ONLY so it stays O(1) per session.
SESSION_STATUS_COMPLETE = "complete"
SESSION_STATUS_INTERRUPTED = "interrupted"
SESSION_STATUS_ERROR = "error"
SESSION_STATUS_EMPTY = "empty"

# finish_reason values meaning the turn ended in a provider/agent error.
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


# Parent→child profile_name inheritance fence: keyless rows inherit freely; two
# ``agent:<ns>:...`` keyed rows must agree on the namespace.
# ``agent:<ns>:...`` gateway keys encode the profile namespace; a keyless row (CLI / subagent lineage)
# carries none and inherits freely. Two keyed rows must agree on ``agent:<ns>:`` — a default child
# (``agent:main:``) forked from a sibling profile's row must not be durably mislabelled as that profile's.
# See #88381.
_SAME_KEY_NAMESPACE_SQL = (
    "p.session_key IS NULL OR sessions.session_key IS NULL"
    " OR substr(p.session_key, 1, instr(substr(p.session_key, 7), ':') + 6)"
    "  = substr(sessions.session_key, 1, instr(substr(sessions.session_key, 7), ':') + 6)"
)


# Upsert tail of _insert_session_row: every routing/metadata column keeps the value an
# earlier writer set (whitespace is part of the SQL text).
_UPSERT_KEEP_EXISTING_SQL = ",\n".join(
    f"                       {col} = COALESCE(sessions.{col}, excluded.{col})" for col in (
        "session_key", "chat_id", "chat_type", "thread_id", "parent_session_id", "cwd", "profile_name",
        "git_repo_root", "origin_json", "display_name",
    )
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
_INHERIT_PARENT_ROUTING_SQL = (
    "UPDATE sessions\n                       SET "
    + _INHERIT_SEP.join(_inherit_col_sql(c) for c in (
        "user_id", "session_key", "chat_id", "chat_type", "thread_id", "display_name", "origin_json",
    ))
    + "\n                     WHERE id = ? AND parent_session_id IS NOT NULL\n"
    "                       AND EXISTS (\n"
    "                           SELECT 1 FROM sessions p\n"
    "                           WHERE p.id = sessions.parent_session_id\n"
    "                             AND p.end_reason = 'compression'\n"
    "                       )"
)


class SessionSessionsMixin:
    """Session rows: create/inherit, lifecycle flags, model_config, listing, deletion."""

    def _own_profile_name(self) -> Optional[str]:
        """The profile owning THIS store, from ``db_path`` alone (``<root>/state.db`` → default,
        ``<root>/profiles/<name>/state.db`` → name): a gateway serving a NON-launch profile opens that
        profile's store. None outside the profile tree — NULL beats a fabricated owner."""
        try:
            from hermes_constants import get_default_hermes_root
            root = get_default_hermes_root().resolve()
            parent = Path(self.db_path).resolve().parent
            if parent == root:
                return "default"
            is_profile_dir = parent.parent == root / "profiles"
            if is_profile_dir and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", parent.name):
                return parent.name
        except Exception:
            logger.debug("own-profile derivation failed", exc_info=True)
        return None

    @staticmethod
    def _inherit_parent_session_metadata(conn, session_id: str) -> None:
        """NULL-fill a child's cwd/git/profile from its parent (profile_name only within the same
        ``agent:<ns>:`` namespace). Gateway routing columns are inherited ONLY by compression forks
        (a crash before the gateway re-records the peer would strand the child unroutable); delegate
        children must NOT inherit them (peer recovery could repoint traffic into a subagent's session)."""
        conn.execute(_INHERIT_PARENT_META_SQL, (session_id,))
        conn.execute(_INHERIT_PARENT_ROUTING_SQL, (session_id,))

    def _insert_session_row(
        self, session_id: str, source: str, model: str = None, model_config: Dict[str, Any] = None,
        system_prompt: str = None, user_id: str = None, session_key: Optional[str] = None,
        chat_id: str = None, chat_type: str = None, thread_id: str = None,
        parent_session_id: str = None, cwd: str = None, profile_name: Optional[str] = None,
        git_repo_root: str = None, origin_json: str = None, display_name: str = None,
    ) -> None:
        """Upsert a session row, never overwriting what an earlier writer set (the gateway creates a
        bare row before create_session carries the real model/prompt). chat_id/thread_id scope gateway
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
        if not (profile_name or "").strip():
            profile_name = self._own_profile_name()
        def _do(conn):
            system_prompt_hash = self._store_system_prompt(conn, system_prompt)
            conn.execute(
                """INSERT INTO sessions (
                   id, source, user_id, session_key, chat_id, chat_type, thread_id,
                   model, model_config, system_prompt, system_prompt_hash,
                   parent_session_id, cwd, profile_name, git_repo_root,
                   origin_json, display_name, started_at
                )
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       model = COALESCE(sessions.model, excluded.model),
                       model_config = CASE
                           WHEN excluded.model_config IS NOT NULL
                                AND json_type(
                                    sessions.model_config, '$._reset_from'
                                ) IS NOT NULL
                                AND json_remove(
                                    sessions.model_config, '$._reset_from'
                                ) = '{}'
                           THEN json_set(
                               excluded.model_config,
                               '$._reset_from',
                               json_extract(
                                   sessions.model_config, '$._reset_from'
                               )
                           )
                           ELSE COALESCE(
                               sessions.model_config, excluded.model_config
                           )
                       END,
                       system_prompt_hash = COALESCE(
                           sessions.system_prompt_hash,
                           excluded.system_prompt_hash
                       ),
                       system_prompt = CASE
                           WHEN sessions.system_prompt_hash IS NULL
                                AND excluded.system_prompt_hash IS NOT NULL
                           THEN NULL
                           ELSE sessions.system_prompt
                       END,
""" + _UPSERT_KEEP_EXISTING_SQL,
                (
                    session_id, source, user_id, session_key, chat_id, chat_type, thread_id, model,
                    json.dumps(model_config) if model_config else None, system_prompt_hash,
                    parent_session_id, cwd, profile_name, git_repo_root, origin_json, display_name,
                    time.time(),
                ),
            )
            if system_prompt_hash is not None:
                self._delete_unreferenced_system_prompts(conn)
            if parent_session_id:
                self._inherit_parent_session_metadata(conn, session_id)
        # Transcript-critical: a failed row creation aborts the turn.
        self._execute_write(_do, patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S)

    def create_session(self, session_id: str, source: str, **kwargs) -> str:
        """Create (upsert) a session record. Returns the session_id."""
        self._insert_session_row(session_id, source, **kwargs)
        return session_id

    def ensure_session(self, session_id: str, source: str = "unknown", model: str = None, **kwargs) -> str:
        """Ensure a session row exists (upsert). Accepts optional kwargs."""
        self._insert_session_row(session_id, source, model=model, **kwargs)
        return session_id

    def set_expiry_finalized(self, session_id: str, finalized: bool = True) -> None:
        """Mirror ``SessionEntry.expiry_finalized`` so it survives a lost sessions.json.

        See #9006.
        """
        if not session_id:
            return
        self._write_sql(
            "UPDATE sessions SET expiry_finalized = ? WHERE id = ?", (1 if finalized else 0, session_id),
        )

    def find_session_by_origin(
        self, *, platform: str, chat_id: str, thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """Most recent live session_id for source + chat_id (+ thread_id). With ``user_id`` exact sender
        matches win; several distinct users and no match → None (never another participant's session)."""
        if not platform or chat_id in (None, ""):
            return None
        query = """
            SELECT id, user_id, started_at FROM sessions
            WHERE LOWER(source) = LOWER(?)
              AND session_key IS NOT NULL
              AND chat_id = ?
              AND ended_at IS NULL
        """
        params: list = [platform, str(chat_id)]
        if thread_id is not None:
            query += " AND COALESCE(thread_id, '') = ?"
            params.append(str(thread_id))
        rows = [dict(r) for r in self._read_all(query + " ORDER BY started_at DESC", params)]
        if not rows:
            return None
        if user_id:
            exact = [r for r in rows if str(r.get("user_id") or "") == str(user_id)]
            if exact:
                return str(exact[0]["id"])
            if len(rows) > 1:
                return None
        elif len({u for u in (str(r.get("user_id") or "").strip() for r in rows) if u}) > 1:
            return None
        return str(rows[0]["id"])

    # Orphaned gateway-session repair: widest plausible gap between a keyed predecessor going
    # quiet and its unkeyed successor (incident was ~60s; 15 min without spanning conversations).
    _ORPHAN_ADOPTION_MAX_GAP_S = 900.0

    # Children that are NOT compression continuations (branches, delegates, tool sessions). Markers
    # are bound to the queried parent id: continuations inherit model_config verbatim, so
    # presence-matching misclassified them as delegates.
    _NON_CONTINUATION_CHILD_FILTER_SQL = (
        "  AND COALESCE(json_extract(COALESCE({alias}model_config, '{{}}'),"
        " '$._branched_from'), '') != ?\n"
        "  AND COALESCE(json_extract(COALESCE({alias}model_config, '{{}}'),"
        " '$._delegate_from'), '') != ?\n  AND COALESCE({alias}source, '') != 'tool'\n"
    )

    def end_session(self, session_id: str, end_reason: str) -> None:
        """Mark a session ended; the first end_reason wins (a compression split must keep
        ``'compression'`` even if a stale end_session() lands later); reopen_session() to re-end."""
        self._execute_write(lambda conn: self._end_and_bump(
            conn, "UPDATE sessions SET ended_at = ?, end_reason = ? WHERE id = ? AND ended_at IS NULL",
            (time.time(), end_reason, session_id), session_id, end_reason,
        ))

    def _end_and_bump(self, conn, sql: str, params: tuple, session_id: str, reason: str) -> int:
        """Run an end-stamp UPDATE; only a boundary this call actually wrote advances the
        conversation generation (a no-op must not rotate the peer). Returns rowcount."""
        child_ids = {
            row["id"]
            for row in conn.execute(
                "SELECT id FROM sessions WHERE parent_session_id = ?", (session_id,)
            ).fetchall()
        }
        topology_ids = {session_id, *child_ids}
        roots_before = self._display_roots_from_conn(conn, topology_ids)
        changed = conn.execute(sql, params).rowcount
        if changed:
            self._bump_conversation_generation(conn, session_id, reason)
            self._invalidate_display_topology(conn, roots_before, topology_ids)
        return changed

    def reopen_session(self, session_id: str) -> None:
        """Clear ended_at/end_reason so a session can be resumed; first stamp markerless legacy reset
        children that depend on the parent's mutable end_reason (WHERE shared with the listing predicate
        so they cannot drift)."""
        def _do(conn):
            child_ids = {
                row["id"]
                for row in conn.execute(
                    "SELECT id FROM sessions WHERE parent_session_id = ?", (session_id,)
                ).fetchall()
            }
            topology_ids = {session_id, *child_ids}
            roots_before = self._display_roots_from_conn(conn, topology_ids)
            conn.execute(
                "UPDATE sessions AS child SET model_config = json_set("
                "COALESCE(child.model_config, '{}'), '$._reset_from', child.parent_session_id) "
                "WHERE child.parent_session_id = ? AND json_extract(COALESCE(child.model_config, '{}'), "
                "                 '$._reset_from') IS NULL "
                f"AND {_legacy_reset_child_sql('child', _session_ids_placeholders(_RESET_END_REASONS))}",
                (session_id, *_RESET_END_REASONS),
            )
            updated = conn.execute(
                "UPDATE sessions SET ended_at = NULL, end_reason = NULL "
                "WHERE id = ? AND (ended_at IS NOT NULL OR end_reason IS NOT NULL)",
                (session_id,),
            )
            if updated.rowcount == 1:
                self._invalidate_display_topology(conn, roots_before, topology_ids)
        self._execute_write(_do)

    def promote_to_session_reset(self, session_id: str, reason: str = "session_reset") -> bool:
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
            return bool(self._execute_write(lambda conn: self._end_and_bump(
                conn, "UPDATE sessions SET ended_at = ?, end_reason = ? WHERE id = ? AND (ended_at IS NULL "
                f"OR end_reason IN ({_RECOVERABLE_END_REASONS_SQL}))",
                (now, reason, session_id), session_id, reason,
            )))
        except Exception:
            return False

    def update_session_cwd(
        self, session_id: str, cwd: str, git_branch: Optional[str] = None,
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
            params: List[Any] = [cwd]
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
        return self._execute_write(_do)

    def publish_session_git_metadata(
        self, session_id: str, cwd: str, generation: int, git_branch: Optional[str] = None,
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
        return self._write_rowcount(
            f"UPDATE sessions SET {', '.join(f'{col} = ?' for col, _ in fields)} "
            "WHERE id = ? AND cwd = ? AND git_metadata_generation = ?",
            [val for _, val in fields] + [session_id, cwd, generation],
        ) == 1

    def backfill_repo_roots(self, cwd_to_root: Dict[str, str]) -> None:
        """Backfill git repo roots for cwds without one; never clobbers a recorded root."""
        pairs = [(root, cwd) for cwd, root in cwd_to_root.items() if root and cwd]
        if pairs:
            self._write_sql(
                "UPDATE sessions SET git_repo_root = ? WHERE cwd = ? AND COALESCE(git_repo_root, '') = ''",
                pairs, many=True,
            )

    def touch_session_activity(
        self, session_id: str, ts: Optional[float] = None, *, description: Optional[str] = None,
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
        self._write_sql(
            "UPDATE sessions SET last_activity_at = ?, "
            "last_activity_description = ?, last_activity_provenance = ? "
            "WHERE id = ? AND (last_activity_at IS NULL OR last_activity_at < ?)",
            (
                when, bound_activity_description(description),
                normalize_activity_provenance(provenance).value, session_id, when,
            ),
            patience_s=self._ACTIVITY_WRITE_PATIENCE_S,
        )

    # Observation-only write: never let it ride the full routine write-patience budget (#76354 review S1).
    # Under contention a heartbeat that waits ~20s would delay the response-critical path it is merely
    # observing; give up after a sub-second budget instead (the next due window retries naturally).
    def clear_session_activity_labels(self, session_id: str) -> None:
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
            row = self._read_one(
                "SELECT last_activity_description, last_activity_provenance FROM sessions WHERE id = ?",
                (session_id,),
            )
        except sqlite3.Error:
            row = None
        if row is not None and not row[0] and (not row[1] or row[1] == ActivityProvenance.UNKNOWN.value):
            return
        self._write_sql(
            "UPDATE sessions SET last_activity_description = ?, last_activity_provenance = ? WHERE id = ?",
            ("", ActivityProvenance.UNKNOWN.value, session_id), patience_s=self._ACTIVITY_WRITE_PATIENCE_S,
        )

    def update_session_meta(
        self, session_id: str, model_config_json: str, model: Optional[str] = None,
    ) -> None:
        """Update model_config and (COALESCE) optionally model."""
        self.flush_token_counts()  # barrier against queued token deltas — see update_session_model
        self._write_sql(
            "UPDATE sessions SET model_config = ?, model = COALESCE(?, model) WHERE id = ?",
            (model_config_json, model, session_id),
        )

    def update_system_prompt(self, session_id: str, system_prompt: Optional[str]) -> None:
        """Store the full assembled system prompt snapshot."""
        def _do(conn):
            conn.execute(
                "UPDATE sessions SET system_prompt_hash = ?, system_prompt = NULL WHERE id = ?",
                (self._store_system_prompt(conn, system_prompt), session_id),
            )
            self._delete_unreferenced_system_prompts(conn)
        self._execute_write(_do)

    def update_session_tool_names(self, session_id: str, tool_names: Optional[List[str]]) -> None:
        """Persist the resolved ``tools[]`` name order so a rebuilt AIAgent can't fork the cached tool
        prefix on a flipped check_fn verdict; ``None`` clears."""
        payload = json.dumps(list(tool_names)) if tool_names is not None else None
        self._write_sql("UPDATE sessions SET tool_names = ? WHERE id = ?", (payload, session_id))

    def update_session_model(self, session_id: str, model: str, provider: Optional[str] = None) -> None:
        """Set the model after a mid-session /model switch (unconditionally), null system_prompt so
        stale Model:/Provider: footers rebuild, and drop any Browser runtime lock (lineage markers
        survive). *provider* is merged into model_config so resume recombines model and provider.

        When *provider* is given, it is merged into ``model_config`` alongside the model (``$.model`` /
        ``$.provider``) so a later resume recombines the persisted model with the provider that actually
        serves it instead of the config.yaml primary provider (#79536). Callers without provider knowledge
        leave any stored provider untouched.
        """
        # Flush first: a still-queued pre-switch delta applied after this UPDATE would trip the
        # first_accounted_route overwrite and resurrect the old route.
        self.flush_token_counts()
        patch: Dict[str, Any] = {"browser_model_lock": None}
        if model:
            patch["model"] = model
        if provider:
            patch["provider"] = provider
        self._write_model_config_patch(
            session_id, patch, "UPDATE sessions SET model = ?, model_config = ?, "
            "system_prompt = NULL, system_prompt_hash = NULL WHERE id = ?",
            lambda merged: (model, merged, session_id),
        )

    def _write_model_config_patch(
        self, session_id: str, patch: Dict[str, Any],
        sql: str = "UPDATE sessions SET model_config = ? WHERE id = ?",
        params: Optional[Callable[[Optional[str]], tuple]] = None,
    ) -> None:
        """Merge ``patch`` into model_config then run ``sql`` with ``params(merged)`` in one write
        transaction; no-op when the row doesn't exist. Custom ``sql`` (prompt-nulling) also GCs prompts."""
        def _do(conn):
            merged = self._merge_model_config_json(conn, session_id, patch)
            if merged is _MODEL_CONFIG_ROW_MISSING:
                return
            conn.execute(sql, params(merged) if params else (merged, session_id))
            if params is not None:
                self._delete_unreferenced_system_prompts(conn)
        self._execute_write(_do)

    def _merge_model_config_json(
        self, conn, session_id: str, patch: Dict[str, Any], *, on_missing: str = "skip",
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

    def patch_session_model_config(self, session_id: str, patch: Dict[str, Any]) -> None:
        """Merge ``patch`` into model_config atomically (``None`` removes a key);
        no-op when the row or patch is empty."""
        if not session_id or not patch:
            return
        self._write_model_config_patch(session_id, patch)

    def get_session_model_config_value(self, session_id: str, key: str, default: Any = None) -> Any:
        """Read one key out of a session's model_config JSON (tolerant parse)."""
        session = self.get_session(session_id) or {}
        return _parse_model_config(session.get("model_config")).get(key, default)

    def update_session_runtime_lock(
        self, session_id: str, *, model: Optional[str] = None, provider: Optional[str] = None,
        model_options: Optional[Dict[str, Any]] = None, route_source: Optional[str] = None,
        confirmed: bool = False,
    ) -> None:
        """Persist a Browser / API-client runtime lock into model_config (lineage markers survive); null
        system_prompt so cached footers cannot lie."""
        lock = {
            "provider": provider or "", "model": model or "", "model_options": model_options or {},
            "route_source": route_source or "", "confirmed": bool(confirmed), "updated_at": time.time(),
        }
        self._write_model_config_patch(
            session_id, {"browser_model_lock": lock},
            """UPDATE sessions SET
                   model_config = ?,
                   model = COALESCE(?, model),
                   system_prompt = NULL,
                   system_prompt_hash = NULL
                   WHERE id = ?""",
            lambda merged: (merged, model, session_id),
        )

    def set_session_yolo(self, session_id: str, enabled: bool) -> None:
        """Persist the per-session YOLO flag so ``/yolo`` survives ``--resume``; no-op without a row."""
        if not session_id:
            return
        self._write_model_config_patch(session_id, {"yolo_mode": bool(enabled)})

    @staticmethod
    def session_yolo_enabled(session_meta: Optional[Dict[str, Any]]) -> bool:
        """Persisted YOLO flag; False on any parse failure (resume must never enable the bypass)."""
        return bool(_parse_model_config((session_meta or {}).get("model_config")).get("yolo_mode"))

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a session by ID (drains queued token deltas first so cost readers see exact totals)."""
        self.flush_token_counts()
        row = self._read_one(
            "SELECT s.*, COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved "
            "FROM sessions s LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash WHERE s.id = ?",
            (session_id,),
        )
        return self._session_row_dict(row) if row else None

    def get_dominant_session_model_route(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Main-loop model route that served most API calls (``session_model_usage`` keeps the coherent
        per-call tuple; ``sessions`` mixes route changes)."""
        self.flush_token_counts()
        row = self._read_one(
            """SELECT model, billing_provider, billing_base_url, billing_mode,
                      api_call_count
                 FROM session_model_usage
                WHERE session_id = ?
                  AND task = ''
                  AND model <> 'unknown'
                  AND billing_provider <> ''
                ORDER BY api_call_count DESC,
                         (input_tokens + output_tokens + cache_read_tokens +
                          cache_write_tokens + reasoning_tokens) DESC,
                         last_seen DESC
                LIMIT 1""",
            (session_id,),
        )
        return dict(row) if row else None

    def resolve_session_id(self, session_id_or_prefix: str) -> Optional[str]:
        """Exact id, else the single unambiguous prefix match, else None."""
        self.flush_token_counts()
        with self._read_ctx() as conn:
            return self._resolve_session_id_from_conn(conn, session_id_or_prefix)

    def backfill_null_session_profiles(self, profile_name: str) -> int:
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
        return int(self._write_rowcount(
            """UPDATE sessions
               SET profile_name = ?
             WHERE profile_name IS NULL OR TRIM(profile_name) = ''""",
            (stamp,),
        ) or 0)

    def _set_lineage_column(self, column: str, session_id: str, value: Any) -> bool:
        """Set one ``sessions`` column across a whole compression lineage: Desktop projects roots
        forward to their tip, so updating only the tip would let the root resurrect it on refresh."""
        return self._write_rowcount(
            f"""
            WITH RECURSIVE
              ancestors(id) AS (
                SELECT ?
                UNION
                SELECT parent.id
                FROM ancestors a
                JOIN sessions child ON child.id = a.id
                JOIN sessions parent ON parent.id = child.parent_session_id
                WHERE parent.end_reason = 'compression'
              ),
              descendants(id) AS (
                SELECT ?
                UNION
                SELECT child.id
                FROM descendants d
                JOIN sessions parent ON parent.id = d.id
                JOIN sessions child ON child.parent_session_id = parent.id
                WHERE parent.end_reason = 'compression'
              ),
              lineage(id) AS (
                SELECT id FROM ancestors
                UNION
                SELECT id FROM descendants
              )
            UPDATE sessions
            SET {column} = ?
            WHERE id IN (SELECT id FROM lineage)
            """,
            (session_id, session_id, value),
        ) > 0

    def set_session_archived(self, session_id: str, archived: bool) -> bool:
        """Soft-hide (or unhide) a session and its compression lineage; messages are kept."""
        return self._set_lineage_column("archived", session_id, int(archived))

    # Accidental end reasons recovery treats as resumable (also interpolated into
    # the recovery/promotion SQL so literals cannot drift).
    RECOVERABLE_END_REASONS = _RECOVERABLE_END_REASONS

    def unarchive_recoverable_session(self, session_id: str) -> bool:
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
            row = self.get_session(session_id)
        except Exception:
            return False
        if not row or not row.get("archived"):
            return False
        # The accidental stamp lives on the live TIP; judge recoverability there.
        tip = row
        try:
            tip_id = self.get_compression_tip(session_id) or session_id
            if tip_id != session_id:
                tip = self.get_session(tip_id) or row
        except Exception:
            pass
        if (tip.get("end_reason") or "") not in self.RECOVERABLE_END_REASONS:
            return False
        if not self.set_session_archived(session_id, False):
            return False
        # Clear the accidental end stamp, or a LATER deliberate archive (which never
        # writes end_reason) would auto-resurrect on the next lookup.
        self._write_sql(
            "UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = ?", (tip["id"],),
        )
        return True

    def set_session_pinned(self, session_id: str, pinned: bool) -> bool:
        """Pin/unpin a session and its compression lineage (pins are exempt from the auto_archive sweep)."""
        return self._set_lineage_column("pinned", session_id, int(pinned))

    def set_session_hidden(self, session_id: str, hidden: bool) -> bool:
        """Hide/unhide a session and its compression lineage from the default listing; still resumable."""
        return self._set_lineage_column("hidden", session_id, int(hidden))

    def set_session_read(self, session_id: str, read: bool = True) -> bool:
        """Mark read/unread across the compression lineage. ``last_read_at`` is a watermark: unread when
        activity postdates it (no write on the message path). NULL = never tracked = read; 0 = unread."""
        return self._set_lineage_column("last_read_at", session_id, time.time() if read else 0.0)

    @staticmethod
    def session_unread(session_row: Dict[str, Any]) -> bool:
        """Unread = activity postdates the ``last_read_at`` watermark (NULL = read)."""
        last_read = session_row.get("last_read_at")
        if last_read is None:
            return False
        last_active = session_row.get("last_active") or session_row.get("started_at")
        return float(last_active or 0) > float(last_read)

    # compact_rows excludes only payload-heavy blobs no list consumer renders.
    _SESSION_COMPACT_EXCLUDED = frozenset(
        {"system_prompt", "system_prompt_hash", "git_metadata_generation"}
    )
    _session_compact_cols_sql: Optional[str] = None

    @staticmethod
    def _chain_search_where(where_sql: str, id_needle: str, search_needle: str) -> Tuple[str, List[Any]]:
        """Extend ``where_sql`` with the id_query / search_query filters: a row is admitted when its own
        id or any id in its forward compression chain matches (search also matches titles and a
        punctuation-stripped form so ``an94`` finds ``AN-94``); chain membership bounds the LIKE."""
        params: List[Any] = []
        clauses: List[str] = []
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

    def _project_compression_tips(self, sessions: List[Dict[str, Any]], compact_rows: bool) -> List[Dict[str, Any]]:
        """Replace each compression root's surfaced fields with its live tip's (root ``started_at`` kept
        for stable ordering), one batched query. ``_lineage_ids`` carries every chain id (a tile may
        hold a MIDDLE segment's id)."""
        chain_by_root: Dict[str, List[str]] = {}  # only roots whose tip differs from themselves
        for s in sessions:
            if s.get("end_reason") == "compression":
                chain = self.get_compression_chain(s["id"])
                if chain and chain[-1] != s["id"]:
                    chain_by_root[s["id"]] = chain
        tip_rows = (
            self._get_session_rich_rows_batch(
                {chain[-1] for chain in chain_by_root.values()}, compact_rows=compact_rows,
            ) if chain_by_root else {}
        )
        projected = []
        for s in sessions:
            chain = chain_by_root.get(s["id"])
            tip_row = tip_rows.get(chain[-1]) if chain else None
            if not tip_row:
                projected.append(s)
                continue
            merged = dict(s)
            for key in (
                "id", "ended_at", "end_reason", "message_count", "tool_call_count", "title", "last_active",
                "preview", "model", "system_prompt", "cwd", "git_branch", "git_repo_root",
            ):
                if key in tip_row:
                    merged[key] = tip_row[key]
            merged["_lineage_root_id"] = s["id"]
            merged["_lineage_ids"] = chain
            projected.append(merged)
        return projected

    def list_recent_sessions_bounded(
        self,
        *,
        limit: int = 20,
        exclude_sources: List[str] = None,
        timeout_seconds: float = 3.0,
        candidate_limit: int = None,
        lineage_limit: int = None,
    ) -> List[Dict[str, Any]]:
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
        candidate_params: List[Any] = []
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
                tip.title,
                s.started_at AS started_at,
                tip.ended_at,
                tip.end_reason,
                tip.message_count,
                tip.tool_call_count,
                rt.activity AS last_active,
                COALESCE(
                    (SELECT {_PREVIEW_RAW_SELECT}
                     FROM messages m
                     WHERE m.session_id = tip.id
                       AND m.role = 'user'
                       AND m.content IS NOT NULL
                       AND {_PREVIEW_ELIGIBLE_SQL}
                     ORDER BY m.timestamp, m.id LIMIT 1),
                    ''
                ) AS _preview_raw,
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
            with self._read_ctx() as conn:
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
            session = self._session_row_dict(row)
            session["preview"] = _shape_preview(session.pop("_preview_raw", ""))
            session["unread"] = self.session_unread(session)
            sessions.append(session)
        return sessions

    @classmethod
    def _list_row(cls, row: sqlite3.Row) -> Dict[str, Any]:
        """Project a list_sessions_rich row: shape the preview, drop internal ordering columns."""
        s = cls._session_row_dict(row)
        s["preview"] = _shape_preview(s.pop("_preview_raw", ""))
        s.pop("_effective_last_active", None)
        return s

    def list_sessions_rich(
        self, source: str = None, sources: List[str] = None, exclude_sources: List[str] = None,
        cwd_prefix: str = None, limit: int = 20, offset: int = 0, include_children: bool = False,
        min_message_count: int = 0, project_compression_tips: bool = True,
        order_by_last_active: bool = False, include_archived: bool = False, archived_only: bool = False,
        id_query: str = None, search_query: str = None, compact_rows: bool = False,
        include_pinned: bool = False, session_key: str = None, include_hidden: bool = False,
    ) -> List[Dict[str, Any]]:
        """List sessions with preview and ``last_active`` in one query. ``order_by_last_active`` sorts
        by the chain TIP via a recursive CTE (the only path honouring ``id_query`` / ``search_query``);
        ``include_pinned`` back-fills pins the page missed, still obeying the other filters."""
        self.flush_token_counts()  # rows carry token/cost totals
        where_clauses, params = _session_filter_where(
            exclude_children=not include_children, source=source, sources=sources, session_key=session_key,
            exclude_sources=exclude_sources, cwd_prefix=cwd_prefix, min_message_count=min_message_count,
            archived_only=archived_only, include_archived=include_archived,
        )
        if not include_hidden:
            where_clauses.append("s.hidden = 0")
        where_sql = _where_sql(where_clauses)
        base_where_params = list(params)  # pinned back-fill reuses the WHERE before LIMIT/OFFSET
        # Shared projection head of the three list queries (whitespace is part of the SQL text).
        select_head = (
            f"SELECT {self._compact_session_cols() if compact_rows else 's.*'}"
            + ("" if compact_rows else ", COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved")
            + f",\n                    {_PREVIEW_COL_SQL},\n                    "
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
            outer_where, id_params = self._chain_search_where(
                where_sql, (id_query or "").strip().lower(), (search_query or "").strip().lower(),
            )
            query = f"""
                WITH RECURSIVE chain(root_id, cur_id) AS (
                    SELECT s.id, s.id FROM sessions s {where_sql}
                    UNION ALL
                    SELECT c.root_id, child.id
                    FROM chain c
                    JOIN sessions parent ON parent.id = c.cur_id
                    JOIN sessions child ON child.parent_session_id = c.cur_id
                    WHERE parent.end_reason = 'compression'
                      AND {_NOT_BRANCH_MARKER_CHILD_SQL.format(a='child')}
                      AND {_NOT_DELEGATE_MARKER_CHILD_SQL.format(a='child')}
                      AND COALESCE(child.source, '') != 'tool'
                ),
                chain_max AS (
                    SELECT
                        root_id,
                        MAX({_sql_session_last_active_by_id("cur_id")}) AS effective_last_active
                    FROM chain
                    GROUP BY root_id
                )
                {select_head}{_sql_session_last_active("s")} AS last_active,
                    COALESCE(cm.effective_last_active, s.started_at) AS _effective_last_active
                FROM sessions s
                LEFT JOIN chain_max cm ON cm.root_id = s.id
                {prompt_join}
                {outer_where}
                ORDER BY _effective_last_active DESC, s.started_at DESC, s.id DESC
                LIMIT ? OFFSET ?
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
        sessions = [self._list_row(row) for row in self._read_all(query, params)]
        # Pinned back-fill runs BEFORE compression projection so a back-filled root
        # projects to its tip like any other row.
        if include_pinned:
            seen_ids = {s["id"] for s in sessions}
            pinned_where = f"{where_sql} AND s.pinned = 1" if where_sql else "WHERE s.pinned = 1"
            pinned_query = f"""
                {select_head}COALESCE(
                        (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id),
                        s.started_at
                    ) AS last_active
                {from_sessions}
                {pinned_where}
                ORDER BY s.started_at DESC
            """
            for row in self._read_all(pinned_query, base_where_params):
                s = self._list_row(row)
                if s["id"] not in seen_ids:
                    seen_ids.add(s["id"])
                    sessions.append(s)
        if project_compression_tips and not include_children:
            sessions = self._project_compression_tips(sessions, compact_rows)
        # last_read_at is lineage-stamped, so root and tip watermarks agree.
        for s in sessions:
            s["unread"] = self.session_unread(s)
        return sessions

    def session_lifecycle_statuses(self, session_ids: List[str]) -> Dict[str, str]:
        """``{session_id: status}`` from each session's LAST message row (``'empty'`` when none); one
        query, MAX(id) per session joined back — never scans transcripts."""
        ids = [sid for sid in (session_ids or []) if sid]
        if not ids:
            return {}
        statuses: Dict[str, str] = {sid: "empty" for sid in ids}
        rows = self._read_all(f"""
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

    def assert_export_safe(self, session_id: str, max_messages: Optional[int] = None) -> int:
        """Active row count of this segment, or raise SessionExportTooLargeError (the LIMITed subquery
        stops once the bound is exceeded). ``None`` resolves ``sessions.max_export_messages``; 0 disables
        the guard."""
        from hermes_state import SessionExportTooLargeError, resolved_max_export_messages
        if max_messages is None:
            max_messages = resolved_max_export_messages()
        if max_messages < 0:
            raise ValueError("max_messages must be non-negative")
        if max_messages == 0:
            return 0
        row = self._read_one(
            "SELECT COUNT(*) FROM (SELECT 1 FROM messages WHERE session_id = ? AND active = 1 LIMIT ?)",
            (session_id, max_messages + 1),
        )
        message_count = int(row[0] if row else 0)
        if message_count > max_messages:
            raise SessionExportTooLargeError(session_id, message_count, max_messages)
        return message_count

    def _is_explicit_branch_session(self, session_id: str) -> bool:
        """Copied user-facing branch (``_branched_from``)? Branches own a copied transcript;
        compression continuations need the parent's archived rows."""
        if not session_id:
            return False
        row = self._read_one(
            "SELECT parent_session_id, model_config FROM sessions WHERE id = ?", (session_id,)
        )
        if row is None or not row["parent_session_id"]:
            return False
        return _parse_model_config(row["model_config"]).get("_branched_from") == row["parent_session_id"]

    def _session_lineage_root_to_tip(self, session_id: str) -> List[str]:
        if not session_id:
            return [session_id]
        chain: List[str] = []
        current = session_id
        with self._read_ctx() as conn:
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
        self, source: str = None, limit: int = 20, offset: int = 0, workspace_key: str = None,
    ) -> List[Dict[str, Any]]:
        """Sessions MRU-first with a computed ``last_active``; ``workspace_key`` scopes to one workspace
        so ``hermes -c``/``--resume`` picks its last session."""
        where_clauses = []
        params: list = []
        if source:
            where_clauses.append("s.source = ?")
            params.append(source)
        if workspace_key:
            ws_clause, ws_params = _workspace_key_clause(workspace_key)
            where_clauses.append(ws_clause)
            params.extend(ws_params)
        return [self._session_row_dict(row) for row in self._read_all(
            "SELECT s.*, COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved, "
            f"{_sql_session_last_active('s')} AS last_active "
            "FROM sessions s LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash "
            f"{_where_sql(where_clauses, ' ')} "
            "ORDER BY last_active DESC, s.started_at DESC, s.id DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        )]

    def session_count(
        self, source: str = None, sources: List[str] = None, cwd_prefix: str = None,
        min_message_count: int = 0, include_archived: bool = False, archived_only: bool = False,
        exclude_children: bool = False, exclude_sources: List[str] = None,
    ) -> int:
        """Count sessions with list_sessions_rich's filters so a paired "load more" total matches."""
        where_clauses, params = _session_filter_where(
            exclude_children=exclude_children, source=source, sources=sources,
            exclude_sources=exclude_sources, cwd_prefix=cwd_prefix, min_message_count=min_message_count,
            archived_only=archived_only, include_archived=include_archived,
        )
        return self._read_one(f"SELECT COUNT(*) FROM sessions s{_where_sql(where_clauses, ' ')}", params)[0]

    def session_count_ge(self, n: int = 1) -> bool:
        """At least N sessions exist (archived included); LIMIT short-circuits session_count()'s scan."""
        return len(self._read_all("SELECT 1 FROM sessions LIMIT ?", (n,))) >= n

    def session_count_by_source(
        self, *, include_archived: bool = False, archived_only: bool = False,
        exclude_children: bool = False,
    ) -> Dict[str, int]:
        """``{source: count}`` via one GROUP BY; ``exclude_children`` mirrors listing visibility."""
        where_clauses, params = _session_filter_where(
            exclude_children=exclude_children, archived_only=archived_only,
            include_archived=include_archived,
        )
        with self._read_ctx() as conn:
            if self._conn is None:
                raise RuntimeError("SessionDB connection is closed")
            rows = conn.execute(
                "SELECT COALESCE(NULLIF(s.source, ''), 'cli') AS source, COUNT(*) AS count "
                f"FROM sessions s{_where_sql(where_clauses, ' ')} "
                "GROUP BY COALESCE(NULLIF(s.source, ''), 'cli') ORDER BY count DESC", params,
            ).fetchall()
        return {str(row["source"]): int(row["count"] or 0) for row in rows}

    def declared_scope_identity(self, session_id: str) -> Tuple[bool, str]:
        """(is_fork_child, source) in ONE read (prompt_cache_scope needs both from the same row).
        Missing row → (False, ""); DB errors propagate (fail closed).

        ``agent/prompt_cache_scope.py`` needs both to resolve a host-declared conversation scope, and both
        live on the same ``sessions`` row; asking for them separately read that row twice per resolution
        (@teknium1 on 98811). The marker rules stay here, beside :meth:`is_explicit_fork_child`, instead of
        being re-implemented by the caller. See #98811.
        """
        session = self.get_session(session_id)
        if not session:
            return False, ""
        return self._is_explicit_fork_child_row(session), str(session.get("source") or "").strip()

    @staticmethod
    def _remove_session_files(sessions_dir: Optional[Path], session_id: str) -> None:
        """Remove ``<id>.json``/``.jsonl`` and gateway ``request_dump_<id>_*.json``; OSError is swallowed
        so a filesystem hiccup never blocks a DB operation."""
        if sessions_dir is None:
            return
        targets = [sessions_dir / f"{session_id}{suffix}" for suffix in (".json", ".jsonl")]
        try:
            targets.extend(sessions_dir.glob(f"request_dump_{session_id}_*.json"))
        except OSError:
            pass
        for p in targets:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

    def get_session_delete_targets(self, session_id: str) -> List[str]:
        """Rows :meth:`delete_session` would remove: the session, then its recursive delegate children
        (branch/compression children are orphaned, not deleted)."""
        with self._read_ctx() as conn:
            if not conn.execute("SELECT 1 FROM sessions WHERE id = ? LIMIT 1", (session_id,)).fetchone():
                return []
            # Use the borrowed read connection, never self._conn: handing the shared writer connection to a
            # helper here executes on it without self._lock — the same unsynchronized-read class as
            # #99349/#90734.
            delegate_ids = _collect_delegate_child_ids(conn, [session_id])
        return [session_id, *sorted(delegate_ids)]

    def delete_session(
        self, session_id: str, sessions_dir: Optional[Path] = None,
        expected_delete_ids: Optional[List[str]] = None,
    ) -> bool:
        """Delete a session and its messages; delegate children cascade, branch/compression children
        are orphaned. *expected_delete_ids*: proceed only if parent + delegate cascade still equals that
        set (re-walked inside the transaction on purpose: export-before-delete fails closed)."""
        removed_ids: List[str] = []
        expected_ids = set(expected_delete_ids) if expected_delete_ids is not None else None
        def _do(conn):
            if conn.execute("SELECT 1 FROM sessions WHERE id = ? LIMIT 1", (session_id,)).fetchone() is None:
                return False
            delegate_ids = _collect_delegate_child_ids(conn, [session_id])
            actual_ids = {session_id, *delegate_ids}
            if expected_ids is not None and expected_ids != actual_ids:
                return False
            roots_before = self._capture_display_ancestor_roots(conn, actual_ids)
            surviving_child_ids = self._surviving_children_for_deleted_sessions(conn, actual_ids)
            removed_ids.extend(_delete_delegate_children(conn, [session_id]))
            conn.execute(  # orphan remaining children (branches) so FK is satisfied
                "UPDATE sessions SET parent_session_id = NULL WHERE parent_session_id = ?", (session_id,),
            )
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            self._delete_unreferenced_system_prompts(conn)
            self._invalidate_display_topology(conn, roots_before, surviving_child_ids)
            removed_ids.append(session_id)
            return True
        deleted = self._execute_write(_do)
        for sid in removed_ids:
            self._remove_session_files(sessions_dir, sid)
        return bool(deleted)

    def delete_session_if_empty(self, session_id: str, sessions_dir: Optional[Path] = None) -> bool:
        """Delete *session_id* only if it has no messages, no title and no children; check and delete
        share one transaction so a concurrent flush can't be lost."""
        def _do(conn):
            cursor = conn.execute(
                """
                DELETE FROM sessions
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
            )
            if cursor.rowcount > 0:
                self._delete_unreferenced_system_prompts(conn)
            return cursor.rowcount > 0
        deleted = self._execute_write(_do)
        if deleted:
            self._remove_session_files(sessions_dir, session_id)
        return deleted

    def delete_sessions(self, session_ids: List[str], sessions_dir: Optional[Path] = None) -> int:
        """Bulk delete with :meth:`delete_session` semantics per row, in ONE transaction. Unknown ids
        are skipped (UI selection can race another tab's delete). Returns the number deleted."""
        unique_ids = list({sid for sid in session_ids or () if isinstance(sid, str) and sid})
        if not unique_ids:
            return 0
        removed_ids: list[str] = []
        def _do(conn):
            existing = [row["id"] for row in conn.execute(
                f"SELECT id FROM sessions WHERE id IN ({_session_ids_placeholders(unique_ids)})",
                unique_ids,
            ).fetchall()]
            if not existing:
                return 0
            ph = _session_ids_placeholders(existing)
            delegate_ids = _collect_delegate_child_ids(conn, existing)
            deleted_ids = {*existing, *delegate_ids}
            roots_before = self._capture_display_ancestor_roots(conn, deleted_ids)
            surviving_child_ids = self._surviving_children_for_deleted_sessions(conn, deleted_ids)
            removed_ids.extend(_delete_delegate_children(conn, existing))
            conn.execute(  # orphan children whose parent is in the kill list (FK)
                f"UPDATE sessions SET parent_session_id = NULL WHERE parent_session_id IN ({ph})", existing,
            )
            conn.execute(f"DELETE FROM messages WHERE session_id IN ({ph})", existing)
            conn.execute(f"DELETE FROM sessions WHERE id IN ({ph})", existing)
            self._delete_unreferenced_system_prompts(conn)
            self._invalidate_display_topology(conn, roots_before, surviving_child_ids)
            removed_ids.extend(existing)
            return len(existing)
        count = self._execute_write(_do)
        for sid in removed_ids:
            self._remove_session_files(sessions_dir, sid)
        return count

    # Shared by count_empty_sessions / delete_empty_sessions so badge and sweep agree. message_count
    # counts live rows only (rewind/compaction keep dropped turns as active = 0): NOT EXISTS is authority.
    # The ``NOT EXISTS`` probe is the authority; : ``message_count = 0`` stays as a cheap prefilter. Same
    # shape as every : other emptiness guard in this module. (#95868)
    _EMPTY_SESSION_WHERE = (
        "message_count = 0 AND ended_at IS NOT NULL AND archived = 0 AND NOT EXISTS ("
        "SELECT 1 FROM messages WHERE messages.session_id = sessions.id)"
    )

    def count_empty_sessions(self) -> int:
        """Count of empty, ended, non-archived sessions; ended_at guards a fresh session's first message."""
        return self._read_one(f"SELECT COUNT(*) FROM sessions WHERE {self._EMPTY_SESSION_WHERE}")[0]

    def delete_empty_sessions(self, sessions_dir: Optional[Path] = None) -> int:
        """Delete every empty, ended, non-archived session in one transaction, orphaning (not cascading)
        children; transcript files are swept too."""
        removed_ids: list[str] = []
        def _do(conn):
            session_ids = {row["id"] for row in conn.execute(
                f"SELECT id FROM sessions WHERE {self._EMPTY_SESSION_WHERE}"
            ).fetchall()}
            if not session_ids:
                return 0
            conn.execute(
                "UPDATE sessions SET parent_session_id = NULL "
                f"WHERE parent_session_id IN ({_session_ids_placeholders(session_ids)})", list(session_ids),
            )
            for sid in session_ids:
                # DELETE FROM messages: a row inserted between the SELECT and here
                # would otherwise dangle (clean FK state).
                conn.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
                conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
                removed_ids.append(sid)
            self._delete_unreferenced_system_prompts(conn)
            return len(session_ids)
        count = self._execute_write(_do)
        for sid in removed_ids:
            self._remove_session_files(sessions_dir, sid)
        return count

    def archive_sessions(
        self, older_than_days: Optional[float] = None, source: str = None, **filters,
    ) -> int:
        """Bulk soft-hide with prune_sessions' filter surface, via set_session_archived so each lineage
        flips as a unit; idempotent. Returns matches."""
        filters.setdefault("archived", False)
        rows = self.list_prune_candidates(older_than_days=older_than_days, source=source, **filters)
        for row in rows:
            self.set_session_archived(row["id"], True)
        return len(rows)

    def maybe_auto_archive(
        self, idle_days: float = 3, min_interval_hours: int = 24, exclude_pinned: bool = True,
    ) -> Dict[str, Any]:
        """Idempotent, non-destructive auto-archive of sessions idle for ``idle_days``; state_meta
        ``last_auto_archive`` gates runs within ``min_interval_hours``. Never raises."""
        result: Dict[str, Any] = {"skipped": False, "archived": 0}
        try:
            now = time.time()
            try:
                last = float(self.get_meta("last_auto_archive") or 0.0)
            except (TypeError, ValueError):
                last = 0.0  # corrupt meta; treat as no prior run
            if last and now - last < min_interval_hours * 3600:
                result["skipped"] = True
                return result
            archived = result["archived"] = self.archive_stale_sessions(idle_days, exclude_pinned=exclude_pinned)
            # Record even a zero-archive run so we don't re-sweep every call.
            self.set_meta("last_auto_archive", str(now))
            if archived > 0:
                logger.info(
                    "state.db auto-archive: archived %d session(s) idle >= %s days", archived, idle_days,
                )
        except Exception as exc:
            logger.warning("state.db auto-archive failed: %s", exc)
            result["error"] = str(exc)
        return result

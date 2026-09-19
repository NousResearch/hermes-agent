"""Crossed-profile durable state in ONE ``state.db`` — finders and fixers for
``hermes sessions repair-profiles`` (#88715 PR-6).

Every ``state.db`` belongs to exactly one profile (``<root>/state.db`` → ``default``,
``<root>/profiles/<name>/state.db`` → ``name``) and every gateway session key encodes the profile
that owns the conversation (``agent:main:…`` for the default, ``agent:<name>:…`` for a named one).
The per-profile store model (#88734) is forward-only: it routes NEW writes to the right store but
never touched rows that had already landed in the wrong one, and the identity fences
(``_INHERIT_PARENT_META_SQL``, ``_recovered_row_allowed_for_active_profile``) only refuse to
*widen* damage that already exists. This module is the backward-looking half: it names each crossed
row and can settle it.

Store-level only. Which profiles exist, which store owns which home, whether a live gateway holds the
routing index in memory, and the JSON files outside ``state.db`` are the orchestrator's concern
(``hermes_cli/sessions_repair_profiles.py``).

Moving a session between two stores is two single-store transactions — copy into the target, then
delete from the source — because a crash between them leaves a *duplicate*, which the next run
settles idempotently (:meth:`import_moved_session` reports ``present``; the delete only proceeds when
the target holds at least as many messages). The reverse order would lose the row.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from hermes_state_common import _id_chunks, _placeholders as _session_ids_placeholders

logger = logging.getLogger(__name__)


def session_key_profile(session_key: Any) -> Optional[str]:
    """Profile encoded in an ``agent:<ns>:…`` gateway key (``default`` for ``main``), or None for a
    keyless row (CLI/subagent lineage) or a key another producer minted (hosted rooms, tests)."""
    if not isinstance(session_key, str):
        return None
    parts = session_key.split(":")
    if len(parts) < 3 or parts[0] != "agent" or not parts[1]:
        return None
    from gateway.session import profile_from_session_key_namespace
    return profile_from_session_key_namespace(parts[1])


def _stored_profile(value: Any) -> Optional[str]:
    """``sessions.profile_name`` as a comparable label; NULL/blank is "unowned", not a crossing."""
    name = str(value or "").strip()
    return name or None


def _table_columns(conn, table: str) -> List[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info('{table}')")]


# Columns a moved message must NOT carry over verbatim: ``id`` is reassigned by the target's
# AUTOINCREMENT and ``display_order`` points at a message id of the SOURCE store — left NULL, the
# ``messages_display_order_insert`` trigger recomputes it from ``display_identity`` (a content hash,
# store-independent) or the new id.
_MESSAGE_MOVE_SKIP = frozenset({"id", "display_order"})


class SessionProfileRepairMixin:
    """Per-store crossed-profile identity: find, relabel, sever, move, and settle routing/topic rows."""

    # ── finders ────────────────────────────────────────────────────────────────

    def find_crossed_profile_sessions(self, owner: str) -> Dict[str, List[Dict[str, Any]]]:
        """Keyed session rows whose identity disagrees with itself or with this store.

        ``mislabelled``: ``profile_name`` names a profile other than the one in the row's own key
        (the key is what routing, the agent cache and ``_db_for_key`` consult; the label is stamped
        after the fact). ``foreign``: the key's profile is not *owner* — the row sits in a store that
        the routed profile never reads. ``crossed_parents``: child and parent are both keyed and name
        different profiles, so the NULL-fill inheritance fence would have been the only thing standing
        between them. A row can appear in more than one list.
        """
        rows = self._read_all(
            "SELECT s.id, s.session_key, s.profile_name, s.parent_session_id, s.message_count, "
            "       p.session_key AS parent_session_key "
            "FROM sessions s LEFT JOIN sessions p ON p.id = s.parent_session_id "
            "WHERE s.session_key LIKE 'agent:%' ORDER BY s.started_at, s.id")
        found: Dict[str, List[Dict[str, Any]]] = {"mislabelled": [], "foreign": [], "crossed_parents": []}
        for row in rows:
            key_profile = session_key_profile(row["session_key"])
            if key_profile is None:
                continue
            label = _stored_profile(row["profile_name"])
            if label is not None and label != key_profile:
                found["mislabelled"].append({
                    "id": row["id"], "session_key": row["session_key"],
                    "profile_name": label, "key_profile": key_profile})
            if key_profile != owner:
                found["foreign"].append({
                    "id": row["id"], "session_key": row["session_key"], "key_profile": key_profile,
                    "message_count": int(row["message_count"] or 0)})
            parent_profile = session_key_profile(row["parent_session_key"])
            if parent_profile is not None and parent_profile != key_profile:
                found["crossed_parents"].append({
                    "id": row["id"], "key_profile": key_profile,
                    "parent_session_id": row["parent_session_id"], "parent_profile": parent_profile})
        return found

    def list_gateway_routing_rows(self) -> List[Dict[str, Any]]:
        return [dict(row) for row in self._read_all(
            "SELECT scope, session_key, entry_json, updated_at FROM gateway_routing "
            "ORDER BY scope, session_key")]

    def find_profile_less_telegram_topic_rows(self) -> List[Dict[str, Any]]:
        """``telegram_dm_topic_bindings`` rows labelled ``default`` whose ``session_key`` names a named
        profile: written by a multiplexer that had not yet learned to stamp the routed profile
        (#76423). Stores whose topic tables predate ``profile_name`` have nothing to relabel."""
        def _read(conn):
            existing = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name = 'telegram_dm_topic_bindings'")}
            if not existing or "profile_name" not in _table_columns(conn, "telegram_dm_topic_bindings"):
                return []
            return [dict(row) for row in conn.execute(
                "SELECT chat_id, thread_id, session_key FROM telegram_dm_topic_bindings "
                "WHERE profile_name = 'default' ORDER BY chat_id, thread_id")]
        out = []
        for row in self._read_retrying_ioerr(_read):
            key_profile = session_key_profile(row["session_key"])
            if key_profile not in (None, "default"):
                out.append({**row, "key_profile": key_profile})
        return out

    def count_messages_all(self, session_id: str) -> int:
        row = self._read_one("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,))
        return int(row[0]) if row else 0

    # ── in-store fixers ───────────────────────────────────────────────────────

    def relabel_sessions_to_key_profile(self, ids: Iterable[str]) -> int:
        """Set ``profile_name`` to the profile in each row's own key. Re-derives the target inside the
        transaction (never trusts a stale report)."""
        wanted = list(ids)
        if not wanted:
            return 0

        def _do(conn) -> int:
            changed = 0
            for chunk in _id_chunks(wanted):
                rows = conn.execute(
                    f"SELECT id, session_key FROM sessions WHERE id IN ({_session_ids_placeholders(chunk)})",
                    chunk).fetchall()
                for session_id, session_key in rows:
                    target = session_key_profile(session_key)
                    if target is None:
                        continue
                    changed += conn.execute(
                        "UPDATE sessions SET profile_name = ? WHERE id = ? AND profile_name IS NOT ?",
                        (target, session_id, target)).rowcount
            return changed
        return self._execute_write(_do)

    def sever_crossed_parents(self, ids: Iterable[str]) -> int:
        """Detach children from a parent keyed under another profile. Only ``parent_session_id`` is
        cleared — ``profile_name`` stays the row's own (relabelled separately when it is wrong) — and
        only while the crossing still holds."""
        wanted = list(ids)
        if not wanted:
            return 0

        def _do(conn) -> int:
            severed = 0
            for chunk in _id_chunks(wanted):
                rows = conn.execute(
                    "SELECT s.id, s.session_key, p.session_key FROM sessions s "
                    "JOIN sessions p ON p.id = s.parent_session_id "
                    f"WHERE s.id IN ({_session_ids_placeholders(chunk)})", chunk).fetchall()
                for session_id, key, parent_key in rows:
                    mine, theirs = session_key_profile(key), session_key_profile(parent_key)
                    if mine is not None and theirs is not None and mine != theirs:
                        severed += conn.execute(
                            "UPDATE sessions SET parent_session_id = NULL WHERE id = ?", (session_id,)).rowcount
            return severed
        return self._execute_write(_do)

    # ── cross-store move ──────────────────────────────────────────────────────

    def export_session_for_move(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Everything the target store needs to hold *session_id* as its own: the row (every column),
        the resolved system prompt, EVERY message row (inactive and compacted generations included —
        a move is not an export) and its usage rows."""
        def _read(conn):
            session = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if session is None:
                return None
            prompt = None
            if session["system_prompt_hash"]:
                row = conn.execute(
                    "SELECT prompt FROM system_prompts WHERE hash = ?", (session["system_prompt_hash"],)).fetchone()
                prompt = row[0] if row else None
            messages = [dict(r) for r in conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id", (session_id,))]
            usage = [dict(r) for r in conn.execute(
                "SELECT * FROM session_model_usage WHERE session_id = ?", (session_id,))]
            return {"session": dict(session), "system_prompt": prompt, "messages": messages, "usage": usage}
        return self._read_retrying_ioerr(_read)

    def import_moved_session(self, payload: Dict[str, Any], *, profile_name: str) -> str:
        """Insert a moved session into THIS store as *profile_name*'s. ``present`` when the id already
        exists (an earlier run copied but did not delete), else ``imported``. The parent link survives
        only when the parent is already here — a moved row must never point across stores or at a
        row of another profile. Columns the target schema lacks are dropped, never invented."""
        session = dict(payload["session"])
        session_id = session["id"]

        def _do(conn) -> str:
            if conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone():
                return "present"
            session["profile_name"] = profile_name
            parent_id = session.get("parent_session_id")
            if parent_id and conn.execute("SELECT 1 FROM sessions WHERE id = ?", (parent_id,)).fetchone() is None:
                session["parent_session_id"] = None
            session["system_prompt_hash"] = self._store_system_prompt(conn, payload.get("system_prompt"))
            self._insert_row(conn, "sessions", session, skip=frozenset())
            for message in payload.get("messages") or []:
                self._insert_row(conn, "messages", {**message, "session_id": session_id}, skip=_MESSAGE_MOVE_SKIP)
            for usage in payload.get("usage") or []:
                self._insert_row(conn, "session_model_usage", {**usage, "session_id": session_id}, skip=frozenset())
            return "imported"
        return self._execute_write(_do)

    @staticmethod
    def _insert_row(conn, table: str, values: Dict[str, Any], *, skip: frozenset) -> None:
        columns = [c for c in _table_columns(conn, table) if c in values and c not in skip]
        conn.execute(
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})",
            [values[c] for c in columns])

    def delete_moved_session(self, session_id: str) -> bool:
        """Remove a session this store no longer owns after the target confirmed it. Children left
        behind are detached (same FK rule as :meth:`delete_session`); topic bindings on the row
        cascade with it."""
        def _do(conn) -> bool:
            if conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone() is None:
                return False
            conn.execute("UPDATE sessions SET parent_session_id = NULL WHERE parent_session_id = ?", (session_id,))
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM session_model_usage WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            self._delete_unreferenced_system_prompts(conn)
            return True
        return bool(self._execute_write(_do))

    # ── routing index ─────────────────────────────────────────────────────────

    def rekey_legacy_main_sessions(self, ids: Iterable[str], profile: str) -> int:
        """``agent:main:…`` → ``agent:<profile>:…`` (+ ``profile_name``) for rows a standalone
        gateway wrote before this store's profile was multiplexed (#113884). Only rows still under
        the legacy namespace change; a ``main``-named profile is written in its marked form."""
        from gateway.session import _session_key_namespace
        wanted = list(ids)
        if not wanted:
            return 0
        new_ns = _session_key_namespace(profile) + ":"

        def _do(conn) -> int:
            changed = 0
            for chunk in _id_chunks(wanted):
                changed += conn.execute(
                    "UPDATE sessions SET session_key = ? || substr(session_key, 12), profile_name = ? "
                    f"WHERE id IN ({_session_ids_placeholders(chunk)}) AND substr(session_key, 1, 11) = 'agent:main:'",
                    (new_ns, profile, *chunk)).rowcount
            return changed
        return self._execute_write(_do)

    def delete_gateway_routing_rows(self, rows: Iterable[Tuple[str, str]]) -> int:
        wanted = list(rows)
        if not wanted:
            return 0

        def _do(conn) -> int:
            return sum(conn.execute(
                "DELETE FROM gateway_routing WHERE scope = ? AND session_key = ?", (scope, key)).rowcount
                for scope, key in wanted)
        return self._execute_write(_do)

    def insert_gateway_routing_rows_if_absent(self, rows: Iterable[Tuple[str, str, str, float]]) -> int:
        """Adopt ``(scope, session_key, entry_json, updated_at)`` rows another store held for this
        one's routing index. An existing key wins — the row the gateway actually loads stays."""
        wanted = list(rows)
        if not wanted:
            return 0

        def _do(conn) -> int:
            return sum(conn.execute(
                "INSERT OR IGNORE INTO gateway_routing (scope, session_key, entry_json, updated_at) "
                "VALUES (?, ?, ?, ?)", row).rowcount for row in wanted)
        return self._execute_write(_do)

    # ── telegram topic tables ─────────────────────────────────────────────────

    def relabel_telegram_topic_rows(self, rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
        """Stamp the key's profile onto ``default``-labelled bindings (and the chat's mode row when
        only the default one exists). A binding that would collide with one the named profile already
        wrote is the stale duplicate and is removed."""
        wanted = list(rows)
        counts = {"bindings_relabelled": 0, "bindings_duplicates_removed": 0, "mode_rows_relabelled": 0}
        if not wanted:
            return counts

        def _do(conn) -> Dict[str, int]:
            mode_has_profile = "profile_name" in _table_columns(conn, "telegram_dm_topic_mode")
            for row in wanted:
                target = session_key_profile(row["session_key"])
                if target in (None, "default"):
                    continue
                chat_id, thread_id = row["chat_id"], row["thread_id"]
                collides = conn.execute(
                    "SELECT 1 FROM telegram_dm_topic_bindings WHERE profile_name = ? AND chat_id = ? "
                    "AND thread_id = ?", (target, chat_id, thread_id)).fetchone()
                if collides:
                    counts["bindings_duplicates_removed"] += conn.execute(
                        "DELETE FROM telegram_dm_topic_bindings WHERE profile_name = 'default' "
                        "AND chat_id = ? AND thread_id = ?", (chat_id, thread_id)).rowcount
                else:
                    counts["bindings_relabelled"] += conn.execute(
                        "UPDATE telegram_dm_topic_bindings SET profile_name = ? WHERE profile_name = 'default' "
                        "AND chat_id = ? AND thread_id = ?", (target, chat_id, thread_id)).rowcount
                if mode_has_profile and conn.execute(
                        "SELECT 1 FROM telegram_dm_topic_mode WHERE profile_name = ? AND chat_id = ?",
                        (target, chat_id)).fetchone() is None:
                    counts["mode_rows_relabelled"] += conn.execute(
                        "UPDATE telegram_dm_topic_mode SET profile_name = ? WHERE profile_name = 'default' "
                        "AND chat_id = ?", (target, chat_id)).rowcount
            return counts
        return self._execute_write(_do)

    # ── evidence for JSON-file repairs ────────────────────────────────────────

    def key_profiles_for_chat(self, platform: str, chat_id: str) -> Set[str]:
        """Profiles whose keyed sessions hold *(platform, chat_id)* in this store — the evidence for
        who owns a profile-less ``gateway_voice_mode.json`` entry."""
        rows = self._read_all(
            "SELECT DISTINCT session_key FROM sessions WHERE source = ? AND chat_id = ? "
            "AND session_key LIKE 'agent:%'", (platform, str(chat_id)))
        return {p for p in (session_key_profile(r["session_key"]) for r in rows) if p is not None}

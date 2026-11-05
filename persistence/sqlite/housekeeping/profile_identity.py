"""``ProfileIdentityRepository`` on SQLite: the profile rekey and purge of the bound ``state.db``, each one write
transaction; a facet whose lazily created table or column the store lacks is skipped and counted ``None``."""

from __future__ import annotations

import json
import time

from persistence.housekeeping.profile_identity import (
    ProfileIdentityCollision, ProfilePurgeCounts, ProfileRekeyCounts,
)

from .._backend import SQLiteStore, register_repository, translated

__all__ = ["SQLiteProfileIdentityRepository"]

_OPTIONAL_TABLE_NAMES = (
    "telegram_dm_topic_mode", "telegram_dm_topic_bindings", "delivery_obligations")


def _optional_table_columns(conn) -> dict[str, set[str]]:
    """Live column sets of the lazily-created tables that exist (``{}`` when none do).

    The telegram topic migration runs only on explicit ``/topic`` opt-in, so a store can hold
    supported v1/v2 tables without ``profile_name`` for its whole life; likewise the delivery
    ledger adds ``delivery_obligations.adapter_profile`` only when a gateway opens it. Identity
    settlement gates its column SQL on the column actually being there rather than on table
    existence, and does not force those migrations.
    """
    existing = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN (?, ?, ?)",
        _OPTIONAL_TABLE_NAMES)}
    return {
        table: {row[1] for row in conn.execute(f"PRAGMA table_info('{table}')")}
        for table in _OPTIONAL_TABLE_NAMES if table in existing
    }


class SQLiteProfileIdentityRepository:
    """Profile identity rewrite and purge inside the bound profile's ``state.db``."""

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    def rekey_profile(self, old_name: str, new_name: str) -> ProfileRekeyCounts:
        old, new = (old_name or "").strip(), (new_name or "").strip()
        counts: dict[str, int] = {}
        if not old or not new or old == new:
            return ProfileRekeyCounts()
        old_ns, new_ns = f"agent:{old}:", f"agent:{new}:"
        ns_len = len(old_ns)

        def _do(conn):
            existing = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            topic_columns = _optional_table_columns(conn)
            collision = conn.execute(
                "SELECT old.scope, ? || substr(old.session_key, ?) "
                "FROM gateway_routing AS old JOIN gateway_routing AS target "
                "ON target.scope = old.scope "
                "AND target.session_key = ? || substr(old.session_key, ?) "
                "WHERE substr(old.session_key, 1, ?) = ? LIMIT 1",
                (new_ns, ns_len + 1, new_ns, ns_len + 1, ns_len, old_ns),
            ).fetchone()
            if collision is not None:
                raise ProfileIdentityCollision(
                    new, f"profile routing collision in scope {collision[0]!r}: {collision[1]!r}")
            for table, columns in (
                ("telegram_dm_topic_mode", ("chat_id",)),
                ("telegram_dm_topic_bindings", ("chat_id", "thread_id")),
            ):
                if "profile_name" not in topic_columns.get(table, set()):
                    continue
                equality = " AND ".join(
                    f"target.{column} = old.{column}" for column in columns)
                collision = conn.execute(
                    f"SELECT 1 FROM {table} AS old JOIN {table} AS target "
                    f"ON target.profile_name = ? AND {equality} "
                    "WHERE old.profile_name = ? LIMIT 1", (new, old)).fetchone()
                if collision is not None:
                    raise ProfileIdentityCollision(new, f"profile identity collision in {table}")

            counts["sessions_profile_name"] = conn.execute(
                "UPDATE sessions SET profile_name = ? WHERE profile_name = ?", (new, old)).rowcount
            counts["gateway_heartbeats_profile"] = conn.execute(
                "UPDATE gateway_heartbeats SET profile = ? WHERE profile = ?", (new, old)).rowcount
            counts["sessions_session_key"] = conn.execute(
                "UPDATE sessions SET session_key = ? || substr(session_key, ?) "
                "WHERE substr(session_key, 1, ?) = ?",
                (new_ns, ns_len + 1, ns_len, old_ns)).rowcount

            origin_count = 0
            for session_id, origin_json in conn.execute(
                    "SELECT id, origin_json FROM sessions WHERE origin_json IS NOT NULL").fetchall():
                try:
                    payload = json.loads(origin_json)
                except (ValueError, TypeError):
                    continue
                if isinstance(payload, dict) and payload.get("profile") == old:
                    payload["profile"] = new
                    conn.execute("UPDATE sessions SET origin_json = ? WHERE id = ?",
                                 (json.dumps(payload, ensure_ascii=False), session_id))
                    origin_count += 1
            counts["sessions_origin_json"] = origin_count

            if "delivery_obligations" in existing:
                if "adapter_profile" in topic_columns.get("delivery_obligations", set()):
                    counts["delivery_obligations_adapter_profile"] = conn.execute(
                        "UPDATE delivery_obligations SET adapter_profile = ? WHERE adapter_profile = ?",
                        (new, old)).rowcount
                counts["delivery_obligations_session_key"] = conn.execute(
                    "UPDATE delivery_obligations SET session_key = ? || substr(session_key, ?) "
                    "WHERE substr(session_key, 1, ?) = ?",
                    (new_ns, ns_len + 1, ns_len, old_ns)).rowcount
            for table in ("telegram_dm_topic_mode", "telegram_dm_topic_bindings"):
                if "profile_name" in topic_columns.get(table, set()):
                    counts[f"{table}_profile_name"] = conn.execute(
                        f"UPDATE {table} SET profile_name = ? WHERE profile_name = ?",
                        (new, old)).rowcount
            if "session_key" in topic_columns.get("telegram_dm_topic_bindings", set()):
                counts["telegram_dm_topic_bindings_session_key"] = conn.execute(
                    "UPDATE telegram_dm_topic_bindings "
                    "SET session_key = ? || substr(session_key, ?) "
                    "WHERE substr(session_key, 1, ?) = ?",
                    (new_ns, ns_len + 1, ns_len, old_ns)).rowcount

            routing = conn.execute(
                "SELECT rowid, session_key, entry_json FROM gateway_routing "
                "WHERE substr(session_key, 1, ?) = ?", (ns_len, old_ns)).fetchall()
            for rowid, session_key, entry_json in routing:
                new_session_key = new_ns + session_key[ns_len:]
                new_json = entry_json
                if entry_json:
                    try:
                        payload = json.loads(entry_json)
                    except (ValueError, TypeError):
                        payload = None
                    if isinstance(payload, dict):
                        if isinstance(payload.get("session_key"), str) and payload["session_key"].startswith(old_ns):
                            payload["session_key"] = new_ns + payload["session_key"][ns_len:]
                        origin = payload.get("origin")
                        if isinstance(origin, dict) and origin.get("profile") == old:
                            origin["profile"] = new
                        new_json = json.dumps(payload, ensure_ascii=False)
                conn.execute(
                    "UPDATE gateway_routing SET session_key = ?, entry_json = ? WHERE rowid = ?",
                    (new_session_key, new_json, rowid))
            counts["gateway_routing"] = len(routing)

        with translated():
            self._store.holder()._execute_write(_do)
        return ProfileRekeyCounts(**counts)

    def purge_profile(self, profile: str) -> ProfilePurgeCounts:
        name = (profile or "").strip()
        counts: dict[str, int] = {}
        if not name:
            return ProfilePurgeCounts()
        ns, ns_len = f"agent:{name}:", len(f"agent:{name}:")

        def _do(conn):
            existing = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            topic_columns = _optional_table_columns(conn)
            if "gateway_routing" in existing:
                counts["gateway_routing"] = conn.execute(
                    "DELETE FROM gateway_routing WHERE substr(session_key, 1, ?) = ?",
                    (ns_len, ns)).rowcount
            if "gateway_heartbeats" in existing:
                counts["gateway_heartbeats"] = conn.execute(
                    "DELETE FROM gateway_heartbeats WHERE profile = ?", (name,)).rowcount
            if "delivery_obligations" in existing:
                # Terminalize, never hard-delete: a pending obligation is delivery state someone may
                # still care about, and the ledger's own retention prunes abandoned rows. Only
                # non-terminal rows are touched; delivered history is left exactly as it was.
                # A ledger created before ``adapter_profile`` existed matches on namespace alone.
                by_profile = ("adapter_profile = ? OR "
                              if "adapter_profile" in topic_columns.get("delivery_obligations", set())
                              else "")
                params = (time.time(), name, ns_len, ns) if by_profile else (time.time(), ns_len, ns)
                counts["delivery_obligations"] = conn.execute(
                    "UPDATE delivery_obligations SET state='abandoned', updated_at=? "
                    f"WHERE ({by_profile}substr(session_key, 1, ?) = ?) "
                    "AND state NOT IN ('delivered', 'abandoned')", params).rowcount
            if "profile_name" in topic_columns.get("telegram_dm_topic_mode", set()):
                counts["telegram_dm_topic_mode"] = conn.execute(
                    "DELETE FROM telegram_dm_topic_mode WHERE profile_name = ?", (name,)).rowcount
            binding_columns = topic_columns.get("telegram_dm_topic_bindings", set())
            if "session_key" in binding_columns:
                # A rename rewrites a binding's session_key namespace as well as its profile_name, so
                # matching on one alone leaves the other behind. Legacy v1/v2 bindings have no
                # profile_name but keep the ``agent:<name>:`` namespace in session_key, so the namespace
                # match alone is the exact cleanup there.
                by_profile = "profile_name = ? OR " if "profile_name" in binding_columns else ""
                params = (name, ns_len, ns) if by_profile else (ns_len, ns)
                counts["telegram_dm_topic_bindings"] = conn.execute(
                    "DELETE FROM telegram_dm_topic_bindings "
                    f"WHERE {by_profile}substr(session_key, 1, ?) = ?", params).rowcount

        with translated():
            self._store.holder()._execute_write(_do)
        return ProfilePurgeCounts(**counts)


register_repository("profile_identity", SQLiteProfileIdentityRepository)

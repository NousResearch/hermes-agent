"""Private SQL of the SQLite ``RoutingRepository``: the route-retirement predicate and fences, the peer, orphan and
topic statements, and the topic-table migration the routing verbs run in their own write."""

from __future__ import annotations

import time

from ..conversation._store.rules import EPOCH_MAX, EPOCH_MIN
from persistence.authority.vocabulary import RETRY_HORIZON_S
from persistence.gateway.routing import route_namespace

from .._schema import _rebuild_table
from .._schema_ddl import _RESET_CHILD_SQL, _RESET_END_REASONS_SQL, _sql_json_extract
from ..conversation._store.retirement import RETIRED_PREFIX, _mark_retired, entry_session_id

# Accidental end reasons recovery treats as resumable: superseded_by_resume = sentinel-parked runtime replaced
# by a fresh session.resume; startup_orphan_reap = dead-gateway sweep, same class as ws_orphan_reap but kept
# distinct for forensics.
_RECOVERABLE_END_REASONS = ("agent_close", "ws_orphan_reap", "superseded_by_resume", "startup_orphan_reap")
_RECOVERABLE_END_REASONS_SQL = ", ".join(f"'{reason}'" for reason in _RECOVERABLE_END_REASONS)

_SQL_IN_WINDOW = f"BETWEEN {EPOCH_MIN!r} AND {EPOCH_MAX!r}"


def _sql_in_window(expr: str) -> str:
    """*expr* when it is inside the ``coerce_epoch`` window, else NULL."""
    return f"(SELECT _win.v FROM (SELECT {expr} AS v) _win WHERE _win.v {_SQL_IN_WINDOW})"


def _sql_freshest_of(activity: str, session_id_expr: str, started: str) -> str:
    """Freshest of *activity* and the latest message timestamp for *session_id_expr*, else *started*.
    Heartbeats are rate-limited (~60s) so ``last_activity_at`` can lag a newer message; never use it alone.
    Cells outside the ``coerce_epoch`` window (garbage doubles salvaged from a damaged page, TEXT) are
    skipped, fallback included, or one bad row pins the session's recency to ``5e+246``; a session with no
    trusted cell at all is NULL. The message index is pinned so the planner never scans every message."""
    msg_max = (f"(SELECT MAX(_act_m.timestamp) FROM messages _act_m INDEXED BY idx_messages_session"
        f" WHERE _act_m.session_id = {session_id_expr}"
        f" AND _act_m.timestamp {_SQL_IN_WINDOW})")
    return (f"COALESCE((SELECT MAX(_act_v.v) FROM (SELECT {activity} AS v UNION ALL SELECT {msg_max}) _act_v"
        f" WHERE _act_v.v {_SQL_IN_WINDOW}), {_sql_in_window(started)})")


def _sql_session_last_active(alias: str = "s") -> str:
    """Session recency expression for a ``sessions {alias}`` row."""
    return _sql_freshest_of(f"{alias}.last_activity_at", f"{alias}.id", f"{alias}.started_at")


# ── route retirement ───────────────────────────────────────────────────────────

# A multiplexed secondary's retired session, in the routing index every served profile shares
# (the launch home's state.db): ``<prefix><namespace>:<id>``, keyed by the ``agent:<namespace>:``
# slot of that profile's routing keys. Only routing saves read it, so the launch profile's own
# creates, binds and generation walks never see another profile's retirement.
ROUTE_FENCE_PREFIX = 'gateway.retired_route.v1.'
_EXPIRED = "json_valid(value) AND json_extract(value, '$.retired_at') < ?"


def route_fence_key(namespace, session_id):
    return f'{ROUTE_FENCE_PREFIX}{namespace}:{session_id}'


# True when a routing save must skip an entry, checked only against the profile the route belongs
# to: this store's own marker for a launch (``main``) route, that profile's fence for a secondary's.
# A launch-keyed route to an id this store holds no session of can only name another profile's
# session, so a secondary's fence covers it too (``fence_shared_routes``). A sweep's fence stops
# fencing at the horizon by itself: only the owner's own store sweeps the owner's marker, and nothing
# guarantees the launch profile ever sweeps this one. A delete's fence (no ``expires``) never does:
# its age test is NULL, which ``coalesce`` reads as not expired. An unscoped marker (no ``own``: an
# earlier release's, written for either) still fences every route. So does a marker or fence that is
# not valid JSON (no writer stores one): ``json_extract`` raises ``malformed JSON`` on it even behind
# ``json_valid(...) AND``, so each JSON read sits in a ``CASE WHEN json_valid(...)`` branch.
# Bound by name (``route_retired_params``): :entry is the route's JSON, :ns its key's
# ``route_namespace``, :now the save's time.
ROUTE_RETIRED_SQL = """(
    EXISTS (SELECT 1 FROM state_meta m
             WHERE m.key = :retired_prefix || json_extract(:entry, '$.session_id')
               AND (:ns = 'main' OR NOT CASE WHEN json_valid(m.value)
                                             THEN json_extract(m.value, '$.own') IS NOT NULL ELSE 0 END))
    OR EXISTS (SELECT 1 FROM state_meta f
                WHERE f.key = :fence_prefix || :ns || ':' || json_extract(:entry, '$.session_id')
                  AND NOT CASE WHEN json_valid(f.value)
                               THEN coalesce(json_extract(f.value, '$.expires')
                                             AND json_extract(f.value, '$.retired_at') < :now - :horizon, 0)
                               ELSE 0 END
                  AND (:ns != 'main' OR NOT EXISTS (
                          SELECT 1 FROM sessions s WHERE s.id = json_extract(:entry, '$.session_id')))))"""


def route_retired_params(session_key, entry_json, now):
    """The named parameters of ``ROUTE_RETIRED_SQL`` for one routing entry saved at *now*."""
    return {'entry': entry_json, 'ns': route_namespace(session_key), 'now': now, 'horizon': RETRY_HORIZON_S,
            'retired_prefix': RETIRED_PREFIX, 'fence_prefix': ROUTE_FENCE_PREFIX}


def fence_shared_routes(conn, namespace, session_ids, *, expiring=()):
    """In the shared routing index (*conn*), fence the routes of *session_ids*, retired in the
    profile whose routing keys use *namespace*, and drop them. The fence is keyed by (profile, id)
    and copies the owner's marker: permanent for a delete, aging out with an *expiring* sweep's.
    Unless this store holds a session of that id, the launch (``main``) namespace is fenced too: a
    launch-keyed route to it names the retired session. The sweep fences that aged out here are
    reclaimed in the same write, so the owner's prunes bound them without the launch profile's sweep."""
    _reclaim_expired(conn, ROUTE_FENCE_PREFIX, time.time() - RETRY_HORIZON_S)
    expiring, owned = set(expiring), {}
    for sid in session_ids:
        names = {namespace}
        if not conn.execute('SELECT 1 FROM sessions WHERE id=?', (sid,)).fetchone():
            names.add('main')
        for name in names:
            _mark_retired(conn, route_fence_key(name, sid), expires=sid in expiring)
        owned[sid] = names
    for row in conn.execute('SELECT scope,session_key,entry_json FROM gateway_routing').fetchall():
        if route_namespace(row['session_key']) in owned.get(entry_session_id(row['entry_json']), ()):
            conn.execute('DELETE FROM gateway_routing WHERE scope=? AND session_key=?',
                         (row['scope'], row['session_key']))


def _reclaim_expired(conn, prefix, cutoff):
    """Drop the sweep (``expires``) markers under *prefix* retired before *cutoff*."""
    conn.execute(f"DELETE FROM state_meta WHERE key GLOB ? AND {_EXPIRED} AND json_extract(value, '$.expires')",
                 (prefix + '*', cutoff))


# ── peers and orphans ─────────────────────────────────────────────────────────

# Recursive CTE naming a session plus its compression ancestors (rows a
# resume must keep on one routing peer); branch/delegate/tool/reset rows stop it
# (same membership rule as get_compression_lineage / _CHAIN_STEP_SQL, #114271).
_COMPRESSION_LINEAGE_CTE = f"""
                    WITH RECURSIVE compression_lineage(id) AS (
                        SELECT ?
                        UNION
                        SELECT parent.id
                        FROM compression_lineage lineage
                        JOIN sessions child ON child.id = lineage.id
                        JOIN sessions parent ON parent.id = child.parent_session_id
                        WHERE parent.end_reason = 'compression'
                          AND {_sql_json_extract('child.model_config', '$._branched_from')} IS NULL
                          AND {_sql_json_extract('child.model_config', '$._delegate_from')} IS NULL
                          AND NOT ({_RESET_CHILD_SQL.format(a='child')})
                          AND COALESCE(child.source, '') != 'tool'
                    )
                """

# Projection shared by both peer-recovery queries (exact key, then peer tuple).
_PEER_SELECT_HEAD = """
                SELECT s.*,
                       COALESCE(sp.prompt, s.system_prompt)
                           AS _system_prompt_resolved,
                       (COALESCE(s.message_count, 0) > 0 OR EXISTS (
                           SELECT 1 FROM messages WHERE messages.session_id = s.id LIMIT 1
                       )) AS _has_messages
                FROM sessions s
                LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash
"""
# A row a *completed handoff* owns stays recoverable whatever end_reason it carries: the source
# interface's teardown runs AFTER ownership moved to the destination and can stamp a terminal reason
# (``cli_close``) before the destination's first reply. A later DELIBERATE close is still fenced the
# ordinary way: a reset-boundary row ended after this row's last activity blocks recovery (the NOT
# EXISTS below). ``pending``/``failed``/NULL handoffs are untouched.
_HANDOFF_OWNED_ROW_SQL = "(s.handoff_state = 'completed')"
_PEER_BY_KEY_SQL = f"""{_PEER_SELECT_HEAD}                WHERE s.session_key = ?
                  AND s.source = ?
                  AND (s.ended_at IS NULL OR s.end_reason IN ({_RECOVERABLE_END_REASONS_SQL})
                       OR {_HANDOFF_OWNED_ROW_SQL})
                  AND NOT EXISTS (
                      SELECT 1 FROM sessions b
                      WHERE b.session_key = s.session_key
                        AND b.source = s.source
                        AND b.ended_at IS NOT NULL
                        AND b.end_reason IN ({_RESET_END_REASONS_SQL})
                        AND b.ended_at
                            > COALESCE(s.last_activity_at, s.started_at)
                  )
                ORDER BY _has_messages DESC,
                         COALESCE(s.last_activity_at, s.started_at) DESC
                LIMIT 1
                """
_PEER_BY_TUPLE_SQL = f"""{_PEER_SELECT_HEAD}                WHERE s.source = ?
                  AND COALESCE(s.user_id, '') = COALESCE(?, '')
                  AND COALESCE(s.chat_id, '') = COALESCE(?, '')
                  AND COALESCE(s.chat_type, '') = COALESCE(?, '')
                  AND COALESCE(s.thread_id, '') = COALESCE(?, '')
                  AND (? IS NULL OR COALESCE(s.profile_name, ?) = ?)
                  AND (s.ended_at IS NULL OR s.end_reason IN ({_RECOVERABLE_END_REASONS_SQL})
                       OR {_HANDOFF_OWNED_ROW_SQL})
                  AND (COALESCE(s.message_count, 0) > 0 OR EXISTS (
                      SELECT 1 FROM messages WHERE messages.session_id = s.id LIMIT 1
                  ))
                  AND NOT EXISTS (
                      SELECT 1 FROM sessions b
                      WHERE b.source = s.source
                        AND COALESCE(b.user_id, '') = COALESCE(s.user_id, '')
                        AND COALESCE(b.chat_id, '') = COALESCE(s.chat_id, '')
                        AND COALESCE(b.chat_type, '') = COALESCE(s.chat_type, '')
                        AND COALESCE(b.thread_id, '') = COALESCE(s.thread_id, '')
                        AND (? IS NULL OR COALESCE(b.profile_name, ?) = ?)
                        AND b.ended_at IS NOT NULL
                        AND b.end_reason IN ({_RESET_END_REASONS_SQL})
                        AND b.ended_at
                            > COALESCE(s.last_activity_at, s.started_at)
                  )
                ORDER BY COALESCE(s.last_activity_at, s.started_at) DESC
                LIMIT 1
                """

_ORPHAN_DONOR_COLUMNS = (
    "d.id, d.session_key, d.chat_id, d.chat_type, d.thread_id, "
    "d.user_id, d.origin_json, d.display_name, d.end_reason"
)
_ORPHANS_SQL = f"""
                SELECT o.id, o.source, o.user_id, o.started_at,
                       o.parent_session_id, o.profile_name,
                       {_sql_session_last_active("o")} AS last_active,
                       (SELECT COUNT(*) FROM messages m
                         WHERE m.session_id = o.id) AS message_count
                FROM sessions o
                WHERE o.session_key IS NULL
                  AND EXISTS (SELECT 1 FROM messages m
                               WHERE m.session_id = o.id)
                  AND COALESCE(o.source, '') != 'tool'
                  AND {_sql_json_extract('o.model_config', '$._branched_from')} IS NULL
                  AND {_sql_json_extract('o.model_config', '$._delegate_from')} IS NULL
                ORDER BY o.started_at ASC
                """
# Donor profile fence: the donor's effective profile (NULL reads as the store's own, the same convention
# the peer-tuple recovery predicate uses) must equal the orphan's. A sibling profile's keyed row in a
# shared legacy store would otherwise stamp its ``agent:<other>:`` routing identity onto this profile's
# orphan. Stores outside the profile tree derive no owner and keep the historical unfenced behavior.
_ORPHAN_DONOR_PROFILE_FENCE_SQL = "AND (? IS NULL OR COALESCE(d.profile_name, ?) = ?)"
_ORPHAN_LINEAGE_DONOR_SQL = f"""
                        SELECT {_ORPHAN_DONOR_COLUMNS}
                        FROM sessions d
                        WHERE d.id = ?
                          AND d.session_key IS NOT NULL
                          AND COALESCE(d.source, '') = COALESCE(?, '')
                          {_ORPHAN_DONOR_PROFILE_FENCE_SQL}
                        """
_ORPHAN_CONTIGUITY_DONORS_SQL = f"""
                        SELECT {_ORPHAN_DONOR_COLUMNS}, {_sql_session_last_active("d")} AS last_active
                        FROM sessions d
                        WHERE d.session_key IS NOT NULL
                          AND d.id != ?
                          AND COALESCE(d.source, '') = COALESCE(?, '')
                          AND (COALESCE(d.user_id, '') = ''
                               OR COALESCE(?, '') = ''
                               OR d.user_id = ?)
                          AND {_sql_session_last_active("d")} BETWEEN ? AND ?
                          AND {_sql_session_last_active("d")} < ?
                          {_ORPHAN_DONOR_PROFILE_FENCE_SQL}
                        ORDER BY last_active DESC
                        LIMIT 2
                        """


# ── Telegram DM topics ────────────────────────────────────────────────────────

# (table, column list, DDL body). profile_name leads the PK: a private chat_id is the
# user id, identical across bots sharing one state.db.
_TOPIC_TABLES = (
    (
        "telegram_dm_topic_mode",
        "profile_name, chat_id, user_id, enabled, activated_at, updated_at, "
        "has_topics_enabled, allows_users_to_create_topics, capability_checked_at, intro_message_id, pinned_message_id",
        """
                    profile_name TEXT NOT NULL DEFAULT 'default',
                    chat_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    activated_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    has_topics_enabled INTEGER,
                    allows_users_to_create_topics INTEGER,
                    capability_checked_at REAL,
                    intro_message_id TEXT,
                    pinned_message_id TEXT,
                    PRIMARY KEY (profile_name, chat_id)
                """,
    ),
    (
        "telegram_dm_topic_bindings",
        "profile_name, chat_id, thread_id, user_id, session_key, session_id, managed_mode, linked_at, updated_at",
        """
                    profile_name TEXT NOT NULL DEFAULT 'default',
                    chat_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_key TEXT NOT NULL,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    managed_mode TEXT NOT NULL DEFAULT 'auto',
                    linked_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (profile_name, chat_id, thread_id)
                """,
    ),
)


def apply_telegram_topic_migration(holder) -> None:
    """Create Telegram DM topic-mode tables on explicit /topic opt-in, never at startup. Also run by
    ``_topic_read`` to heal a pre-v3 table an upgrade left behind. Schema versions: v1 initial; v2
    session_id FK ON DELETE CASCADE (pruning clears bindings); v3 ``profile_name`` on both tables so
    multiplexed gateways sharing one state.db isolate topic state per profile."""
    def _do(conn):
        for table, columns, ddl in _TOPIC_TABLES:
            conn.execute(f"CREATE TABLE IF NOT EXISTS {table} ({ddl})")
            have = {row[1] for row in conn.execute(f"PRAGMA table_info('{table}')")}
            if "profile_name" in have:
                continue
            # v1/v2 → v3. SQLite can't ALTER a PK or FK, so rebuild (also supplies v2's
            # ON DELETE CASCADE); _rebuild_table runs inside the open BEGIN IMMEDIATE so a crash
            # rolls back instead of stranding rows (#42004). A {table}_new left by an older
            # build's executescript crash is dropped first; its legacy table is still intact.
            # v1 bindings had no ON DELETE CASCADE, so pruned sessions left orphan rows that
            # the v3 FK (foreign_keys=ON on the writer) would reject, so copy only live ones.
            legacy_columns = columns.replace("profile_name, ", "", 1)
            live = " WHERE EXISTS (SELECT 1 FROM sessions s WHERE s.id = session_id)" if "session_id" in columns else ""
            conn.execute(f"DROP TABLE IF EXISTS {table}_new")
            _rebuild_table(
                conn.cursor(), table, f"{table}_legacy", f"CREATE TABLE {table} ({ddl})",
                f"INSERT INTO {table} ({columns}) SELECT 'default', {legacy_columns} FROM {table}_legacy{live}",
            )
        # Indexes after any rebuild: the user index needs profile_name.
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_telegram_dm_topic_bindings_session "
            "ON telegram_dm_topic_bindings(session_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_telegram_dm_topic_bindings_user "
            "ON telegram_dm_topic_bindings(profile_name, user_id, chat_id)"
        )
        conn.execute(
            "INSERT INTO state_meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            ("telegram_dm_topic_schema_version", "3"),
        )
    holder._execute_write(_do)

"""Gateway routing index CRUD (sessions.json replacement, #9006 follow-up) for SessionDB.

Mixin contract: this is a plain mixin class consumed by
``hermes_state.SessionDB``. It defines no ``__init__`` and no state of its
own; methods access the host's attributes (``self._conn``, ``self._lock``,
``self._execute_write``) established by ``SessionDB.__init__``. It must
never import hermes_state (cycle) - shared module-level constants live in
hermes_state_common.

Extracted from hermes_state.py slice R2-S1 (lines 3204-3279, banner +
save/replace/load/delete_gateway_routing_entries), epic #78647, issue
#78636, consensus pin 01a1037d1e. Bytes are verbatim; the golden sha of
the moved window is 9fe5c63d445022c9163d1d2117c5a580878d6911a51ec74278391919318e8867.
"""

import logging
import time
from typing import Dict, List

# Moved methods logged under the "hermes_state" logger before the split;
# keep that logger identity so log filtering/capture behavior is unchanged.
logger = logging.getLogger("hermes_state")


class SessionGatewayRoutingMixin:
    """See module docstring - mixin for SessionDB (Gateway routing cluster)."""

    # ── Gateway routing index (replaces sessions.json, #9006 follow-up) ────

    def save_gateway_routing_entry(
        self, session_key: str, entry_json: str, *, scope: str = ""
    ) -> None:
        """Upsert one gateway routing entry (session_key -> SessionEntry JSON).

        The gateway_routing table is the durable replacement for
        sessions.json: one row per routing key, holding the full serialized
        ``SessionEntry`` so the gateway can rehydrate exactly what it wrote.

        ``scope`` namespaces the index the way separate sessions.json files
        did (one per sessions_dir) — callers pass their sessions_dir path so
        two stores with different directories never share routing state.
        """
        if not session_key or not entry_json:
            return

        def _do(conn):
            conn.execute(
                """INSERT INTO gateway_routing (scope, session_key, entry_json, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(scope, session_key) DO UPDATE SET
                       entry_json = excluded.entry_json,
                       updated_at = excluded.updated_at""",
                (scope, session_key, entry_json, time.time()),
            )

        self._execute_write(_do)

    def replace_gateway_routing_entries(
        self, entries: Dict[str, str], *, scope: str = ""
    ) -> None:
        """Atomically replace the routing index for *scope* with *entries*.

        Mirrors the sessions.json full-rewrite semantics: keys absent from
        *entries* are removed (pruned/reset sessions disappear from the
        index).  Runs as a single write transaction.  Other scopes are
        untouched.
        """
        now = time.time()

        def _do(conn):
            conn.execute("DELETE FROM gateway_routing WHERE scope = ?", (scope,))
            if entries:
                conn.executemany(
                    "INSERT INTO gateway_routing (scope, session_key, entry_json, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    [(scope, k, v, now) for k, v in entries.items() if k and v],
                )

        self._execute_write(_do)

    def load_gateway_routing_entries(self, *, scope: str = "") -> Dict[str, str]:
        """Load routing entries for *scope* as {session_key: entry_json}."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT session_key, entry_json FROM gateway_routing WHERE scope = ?",
                (scope,),
            ).fetchall()
        return {r["session_key"]: r["entry_json"] for r in rows}

    def delete_gateway_routing_entries(
        self, session_keys: List[str], *, scope: str = ""
    ) -> None:
        """Remove routing entries for the given session keys in *scope*."""
        if not session_keys:
            return

        def _do(conn):
            conn.executemany(
                "DELETE FROM gateway_routing WHERE scope = ? AND session_key = ?",
                [(scope, k) for k in session_keys],
            )

        self._execute_write(_do)

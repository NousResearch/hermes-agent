"""Durable profile-bound execution journal and conservative recovery handshake."""
from __future__ import annotations

from contextlib import contextmanager
import json
import sqlite3
from pathlib import Path
from typing import Callable

VERSION = "chat-execution/v1"
# Detached work (a background delegate unit) is admitted by its dispatching worker but outlives it.
DETACHED_RESOURCE_KINDS = frozenset({"async_delegation_thread"})
_RESOURCE_KINDS = frozenset({
    "concurrent_tool_thread", "sequential_tool_thread", "prepared_terminal_tool_thread",
    "middleware_tool_continuation", "memory_prefetch_thread", "terminal_local_process",
}) | DETACHED_RESOURCE_KINDS


class ChatExecutionStore:
    """SQLite journal. Unknown tool effects are never eligible for transfer."""

    def __init__(self, profile_home: str | Path):
        self.home = Path(profile_home).resolve()
        self.home.mkdir(parents=True, exist_ok=True)
        self.path = self.home / "chat-execution.sqlite3"
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS executions (
                session_key TEXT NOT NULL, generation INTEGER NOT NULL CHECK(generation > 0),
                session_id TEXT NOT NULL, worker TEXT NOT NULL, worker_started INTEGER NOT NULL,
                effects TEXT NOT NULL, tool_observed INTEGER NOT NULL,
                PRIMARY KEY(session_key, generation))""")
            db.execute("""CREATE TABLE IF NOT EXISTS session_fences (
                session_key TEXT PRIMARY KEY, generation INTEGER NOT NULL,
                recovery_id TEXT NOT NULL, state TEXT NOT NULL, receipt TEXT)""")
            db.execute("""CREATE TABLE IF NOT EXISTS pending_submits (
                session_key TEXT NOT NULL, submit_id TEXT NOT NULL,
                PRIMARY KEY(session_key, submit_id))""")
            db.execute("""CREATE TABLE IF NOT EXISTS session_key_lineage (
                parent_key TEXT NOT NULL, child_key TEXT PRIMARY KEY)""")
            db.execute("""CREATE TABLE IF NOT EXISTS concurrent_tool_resources (
                session_key TEXT NOT NULL, generation INTEGER NOT NULL, resource_id TEXT NOT NULL,
                session_id TEXT NOT NULL, kind TEXT NOT NULL, state TEXT NOT NULL,
                PRIMARY KEY(session_key, generation, resource_id))""")
            db.execute("""CREATE TABLE IF NOT EXISTS terminal_process_lineage (
                session_key TEXT NOT NULL, generation INTEGER NOT NULL, resource_id TEXT NOT NULL,
                session_id TEXT NOT NULL, process_session_id TEXT NOT NULL, pid INTEGER NOT NULL,
                host_start_time INTEGER, PRIMARY KEY(session_key, generation, resource_id),
                UNIQUE(process_session_id))""")

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA busy_timeout=15000")
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _valid_generation(generation):
        return type(generation) is int and generation > 0

    def latest_generation(self, session_key):
        with self._db() as db:
            row = db.execute("SELECT MAX(generation) AS g FROM executions WHERE session_key=?", (session_key,)).fetchone()
            return int(row["g"] or 0)

    @staticmethod
    def _lineage_keys(db, root_key):
        rows = db.execute("""WITH RECURSIVE descendants(key) AS (
            SELECT ? UNION SELECT child_key FROM session_key_lineage
            JOIN descendants ON parent_key=descendants.key)
            SELECT key FROM descendants""", (root_key,)).fetchall()
        return tuple(row["key"] for row in rows)

    def lineage_keys(self, root_key):
        if not root_key:
            return ()
        with self._db() as db:
            return self._lineage_keys(db, root_key)

    def link_session_keys(self, parent_key, child_key):
        """Record an actual compression re-anchor before the runtime adopts its new key."""
        if not parent_key or not child_key or parent_key == child_key:
            return False
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT parent_key FROM session_key_lineage WHERE child_key=?",
                                  (child_key,)).fetchone()
            if existing:
                db.rollback(); return existing["parent_key"] == parent_key
            if parent_key in self._lineage_keys(db, child_key):
                db.rollback(); return False
            parent_fence = db.execute("SELECT * FROM session_fences WHERE session_key=?",
                                      (parent_key,)).fetchone()
            child_fence = db.execute("SELECT 1 FROM session_fences WHERE session_key=?",
                                     (child_key,)).fetchone()
            # A late alias cannot make a previously issued retirement receipt
            # claim that an already-running child was quiescent.
            child_work = (db.execute("SELECT 1 FROM executions WHERE session_key=? LIMIT 1",
                                     (child_key,)).fetchone()
                          or db.execute("SELECT 1 FROM pending_submits WHERE session_key=? LIMIT 1",
                                        (child_key,)).fetchone())
            if child_fence or (parent_fence and parent_fence["receipt"] and child_work):
                db.rollback(); return False
            db.execute("INSERT INTO session_key_lineage VALUES (?, ?)", (parent_key, child_key))
            if parent_fence:
                db.execute("INSERT INTO session_fences VALUES (?, ?, ?, 'fenced', NULL)",
                           (child_key, parent_fence["generation"], parent_fence["recovery_id"]))
            db.commit(); return True

    def begin_submit(self, session_key, submit_id):
        """Reserve a real incoming turn before any pre-worker transcript mutation."""
        if not session_key or not submit_id:
            return False
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            fence = db.execute("SELECT state FROM session_fences WHERE session_key=?", (session_key,)).fetchone()
            if fence and fence["state"] == "fenced":
                db.rollback(); return False
            db.execute("INSERT INTO pending_submits VALUES (?, ?)", (session_key, submit_id))
            db.commit(); return True

    def abandon_submit(self, session_key, submit_id):
        if not session_key or not submit_id:
            return False
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            source = db.execute("SELECT 1 FROM pending_submits WHERE session_key=? AND submit_id=?",
                                (session_key, submit_id)).fetchone()
            if not source:
                db.rollback(); return False
            db.execute("DELETE FROM pending_submits WHERE submit_id=?", (submit_id,))
            db.commit(); return True

    def submit_claims_active(self, session_key, submit_ids, *, source_key=None):
        """Check a staged turn; bridge a real session-key rotation without a fence gap."""
        submit_ids = tuple(submit_ids)
        source_key = source_key or session_key
        if not session_key or not source_key or not submit_ids:
            return False
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            fences = db.execute("SELECT session_key FROM session_fences WHERE state='fenced' AND session_key IN (?, ?)",
                                (session_key, source_key)).fetchall()
            owned = {row["submit_id"] for row in db.execute(
                "SELECT submit_id FROM pending_submits WHERE session_key=?", (source_key,))}
            if fences or not set(submit_ids) <= owned:
                db.rollback(); return False
            if source_key != session_key:
                db.executemany("INSERT OR IGNORE INTO pending_submits VALUES (?, ?)",
                               ((session_key, submit_id) for submit_id in submit_ids))
            db.commit(); return True

    def start(self, session_key, generation, session_id):
        if not self._valid_generation(generation) or not session_key or not session_id:
            return False
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            fence = db.execute("SELECT * FROM session_fences WHERE session_key=?", (session_key,)).fetchone()
            if fence and fence["state"] == "fenced":
                db.rollback(); return False
            latest = db.execute("SELECT MAX(generation) AS g FROM executions WHERE session_key=?", (session_key,)).fetchone()["g"]
            if latest is not None and generation <= latest:
                db.rollback(); return False
            db.execute("INSERT INTO executions VALUES (?, ?, ?, 'pending', 0, 'unknown', 0)", (session_key, generation, session_id))
            db.commit()
            return True

    def start_local_worker(self, session_key, session_id, submit_ids=()):
        """Atomically admit a Desktop/TUI turn with its actual Agent session id."""
        if not session_key or not session_id:
            return None
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            fence = db.execute("SELECT state FROM session_fences WHERE session_key=?", (session_key,)).fetchone()
            pending = db.execute("SELECT 1 FROM executions WHERE session_key=? AND worker!='completed' LIMIT 1",
                                 (session_key,)).fetchone()
            submit_ids = tuple(submit_ids)
            owned = {row["submit_id"] for row in db.execute(
                "SELECT submit_id FROM pending_submits WHERE session_key=?", (session_key,))}
            if ((fence and fence["state"] == "fenced") or pending
                    or any(submit_id not in owned for submit_id in submit_ids)):
                db.rollback(); return None
            latest = db.execute("SELECT MAX(generation) FROM executions WHERE session_key=?",
                                (session_key,)).fetchone()[0] or 0
            generation = latest + 1
            db.execute("INSERT INTO executions VALUES (?, ?, ?, 'pending', 1, 'unknown', 0)",
                       (session_key, generation, session_id))
            for submit_id in submit_ids:
                # Aliases for a rotating session stay fenced until this exact
                # physical worker is admitted, then settle in one transaction.
                db.execute("DELETE FROM pending_submits WHERE submit_id=?", (submit_id,))
            db.commit()
            return generation

    def reserve(self, session_key, memory_floor, caller_floor, *, predecessor=None):
        """Atomically reserve a durable generation before the physical session id exists."""
        if not session_key or type(memory_floor) is not int or type(caller_floor) is not int:
            return None
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            fence = db.execute("SELECT state FROM session_fences WHERE session_key=?", (session_key,)).fetchone()
            if fence and fence["state"] == "fenced":
                db.rollback()
                return None
            latest = db.execute("SELECT MAX(generation) AS g FROM executions WHERE session_key=?", (session_key,)).fetchone()["g"] or 0
            if predecessor is not None:
                previous_generation, previous_session_id = predecessor
                previous = db.execute(
                    "SELECT session_id,worker,worker_started FROM executions WHERE session_key=? AND generation=?",
                    (session_key, previous_generation),
                ).fetchone()
                if (latest != previous_generation or not previous or not previous_session_id
                        or tuple(previous) != (previous_session_id, "completed", 1)):
                    db.rollback()
                    return None
            generation = max(int(latest), memory_floor, caller_floor) + 1
            db.execute("INSERT INTO executions VALUES (?, ?, '', 'pending', 0, 'unknown', 0)", (session_key, generation))
            db.commit()
            return generation

    def bind_worker_started(self, session_key, generation, session_id):
        """Bind one actual session to the latest reservation and admit its physical worker."""
        if not self._valid_generation(generation) or not session_key or not session_id:
            return False
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            fence = db.execute("SELECT state FROM session_fences WHERE session_key=?", (session_key,)).fetchone()
            latest = db.execute("SELECT MAX(generation) AS g FROM executions WHERE session_key=?", (session_key,)).fetchone()["g"]
            if (fence and fence["state"] == "fenced") or latest != generation:
                db.rollback()
                return False
            result = db.execute(
                "UPDATE executions SET session_id=?, worker_started=1 WHERE session_key=? AND generation=? AND session_id='' AND worker_started=0",
                (session_id, session_key, generation),
            )
            if result.rowcount != 1:
                db.rollback()
                return False
            db.commit()
            return True

    @staticmethod
    def _detached_admits(db, session_key, generation, under):
        """A started detached resource ``(resource_id, session_id)`` still admits its own subtree after its
        dispatching worker settles or a successor generation starts; a fence admits nothing."""
        if under is None:
            return False
        fence = db.execute("SELECT state FROM session_fences WHERE session_key=?", (session_key,)).fetchone()
        if fence and fence["state"] == "fenced":
            return False
        row = db.execute("SELECT kind FROM concurrent_tool_resources WHERE session_key=? AND generation=? AND resource_id=? AND session_id=? AND state='started'",
                         (session_key, generation, *under)).fetchone()
        return row is not None and row["kind"] in DETACHED_RESOURCE_KINDS

    def reserve_concurrent_tool_resource(self, session_key, generation, resource_id, kind="concurrent_tool_thread", under=None):
        if kind not in _RESOURCE_KINDS:
            return False
        if not session_key or not self._valid_generation(generation) or not resource_id:
            return False
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            parent = db.execute("SELECT session_id, worker, worker_started FROM executions WHERE session_key=? AND generation=?", (session_key, generation)).fetchone()
            latest = db.execute("SELECT MAX(generation) FROM executions WHERE session_key=?", (session_key,)).fetchone()[0]
            fence = db.execute("SELECT state FROM session_fences WHERE session_key=?", (session_key,)).fetchone()
            live = under is None and parent and parent["worker"] == "pending" and latest == generation
            # Detached authority never mints more detached authority: only a live worker reserves one.
            inherited = kind not in DETACHED_RESOURCE_KINDS and self._detached_admits(db, session_key, generation, under)
            if (not parent or not parent["session_id"] or not parent["worker_started"] or (fence and fence["state"] == "fenced")
                    or not (live or inherited)):
                db.rollback()
                return False
            try:
                db.execute("INSERT INTO concurrent_tool_resources VALUES (?, ?, ?, ?, ?, 'reserved')",
                           (session_key, generation, resource_id, parent["session_id"], kind))
            except sqlite3.IntegrityError:
                db.rollback()
                return False
            db.execute("UPDATE executions SET tool_observed=1, effects='unknown' WHERE session_key=? AND generation=?", (session_key, generation))
            db.commit()
            return True

    def start_concurrent_tool_resource(self, session_key, generation, resource_id, session_id, kind="concurrent_tool_thread", under=None):
        if kind not in _RESOURCE_KINDS:
            return False
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT session_id, state, kind FROM concurrent_tool_resources WHERE session_key=? AND generation=? AND resource_id=?", (session_key, generation, resource_id)).fetchone()
            parent = db.execute("SELECT worker FROM executions WHERE session_key=? AND generation=?", (session_key, generation)).fetchone()
            latest = db.execute("SELECT MAX(generation) FROM executions WHERE session_key=?", (session_key,)).fetchone()[0]
            fence = db.execute("SELECT state FROM session_fences WHERE session_key=?", (session_key,)).fetchone()
            # Detached work was admitted at reservation by its pending, current worker; a queued pool
            # thread starting after that worker settles is not stale.
            live = under is None and parent and ((parent["worker"] == "pending" and latest == generation) or kind in DETACHED_RESOURCE_KINDS)
            inherited = kind not in DETACHED_RESOURCE_KINDS and self._detached_admits(db, session_key, generation, under)
            if (not row or row["session_id"] != session_id or row["kind"] != kind or row["state"] != "reserved" or not parent
                    or (fence and fence["state"] == "fenced") or not (live or inherited)):
                db.rollback()
                return False
            db.execute("UPDATE concurrent_tool_resources SET state='started' WHERE session_key=? AND generation=? AND resource_id=?", (session_key, generation, resource_id))
            db.commit()
            return True

    def finish_concurrent_tool_resource(self, session_key, generation, resource_id, session_id, kind="concurrent_tool_thread"):
        # Process identity/direct-child unwind is not a process-tree settlement witness.
        if kind == "terminal_local_process":
            return False
        if kind not in _RESOURCE_KINDS:
            return False
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            result = db.execute("UPDATE concurrent_tool_resources SET state='finished' WHERE session_key=? AND generation=? AND resource_id=? AND session_id=? AND kind=? AND state='started'", (session_key, generation, resource_id, session_id, kind))
            db.commit()
            return result.rowcount == 1

    def record_terminal_process_lineage(self, session_key, generation, resource_id, session_id, process_session_id, pid, host_start_time):
        """Bind observed post-Popen identity to exactly the already-started resource."""
        if (not session_key or not self._valid_generation(generation) or not resource_id
                or not isinstance(process_session_id, str) or not process_session_id
                or type(pid) is not int or pid <= 0
                or type(host_start_time) is not int or host_start_time <= 0):
            return False
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT session_id, kind, state FROM concurrent_tool_resources WHERE session_key=? AND generation=? AND resource_id=?", (session_key, generation, resource_id)).fetchone()
            if not row or row["session_id"] != session_id or row["kind"] != "terminal_local_process" or row["state"] != "started":
                db.rollback()
                return False
            try:
                db.execute("INSERT INTO terminal_process_lineage VALUES (?, ?, ?, ?, ?, ?, ?)",
                           (session_key, generation, resource_id, session_id, process_session_id, pid, host_start_time))
            except sqlite3.IntegrityError:
                existing = db.execute("SELECT session_id,process_session_id,pid,host_start_time FROM terminal_process_lineage WHERE session_key=? AND generation=? AND resource_id=?", (session_key, generation, resource_id)).fetchone()
                if existing and tuple(existing) == (session_id, process_session_id, pid, host_start_time):
                    db.commit()
                    return True
                db.rollback()
                return False
            db.commit()
            return True

    def inspect_terminal_process_lineage(self, session_key, generation, resource_id):
        with self._db() as db:
            row = db.execute("SELECT * FROM terminal_process_lineage WHERE session_key=? AND generation=? AND resource_id=?", (session_key, generation, resource_id)).fetchone()
        return dict(row) if row else None

    def inspect_resources(self, session_key):
        with self._db() as db:
            rows = db.execute("SELECT session_key,generation,resource_id,session_id,kind,state FROM concurrent_tool_resources WHERE session_key=? ORDER BY generation,resource_id", (session_key,)).fetchall()
        return [dict(row) for row in rows]

    def _mutate(self, session_key, generation, sql):
        if not self._valid_generation(generation):
            return False
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            fence = db.execute("SELECT * FROM session_fences WHERE session_key=?", (session_key,)).fetchone()
            if fence and fence["state"] == "fenced":
                db.rollback(); return False
            result = db.execute(sql, (session_key, generation))
            if result.rowcount == 1 and db.execute("SELECT session_id FROM executions WHERE session_key=? AND generation=?", (session_key, generation)).fetchone()["session_id"] == "":
                db.rollback()
                return False
            db.commit()
            return result.rowcount == 1

    def worker_started(self, session_key, generation):
        return self._mutate(session_key, generation, "UPDATE executions SET worker='pending', worker_started=1 WHERE session_key=? AND generation=? AND worker_started=0")

    def worker_finished(self, session_key, generation):
        if not self._valid_generation(generation):
            return False
        # Completion is an observation, not admission. A Stop fence must not
        # prevent its already-started physical worker from settling.
        with self._db() as db:
            reserved = db.execute("SELECT session_id FROM executions WHERE session_key=? AND generation=?", (session_key, generation)).fetchone()
            if reserved is None or not reserved["session_id"]:
                return False
            result = db.execute("UPDATE executions SET worker='completed' WHERE session_key=? AND generation=? AND worker_started=1", (session_key, generation))
            return result.rowcount == 1

    def observe_tool(self, session_key, generation):
        return self._mutate(session_key, generation, "UPDATE executions SET tool_observed=1, effects='unknown' WHERE session_key=? AND generation=? AND worker_started=1")

    def observe_started_worker_tool(self, session_key, generation, under=None):
        """Durably mark a tool from an already-started worker, even after fencing. With ``under``, mark one
        from the subtree of that started detached resource instead, whatever its worker's state; not fenced."""
        if not session_key or not self._valid_generation(generation):
            return False
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            if under is not None and not self._detached_admits(db, session_key, generation, under):
                db.rollback()
                return False
            result = db.execute(
                "UPDATE executions SET tool_observed=1, effects='unknown' "
                "WHERE session_key=? AND generation=? AND session_id!='' AND worker_started=1"
                + ("" if under is not None else " AND worker='pending'"),
                (session_key, generation),
            )
            if result.rowcount != 1:
                db.rollback()
                return False
            db.commit()
            return True

    def effects_none(self, session_key, generation):
        return self._mutate(session_key, generation, "UPDATE executions SET effects='none' WHERE session_key=? AND generation=? AND tool_observed=0 AND effects='unknown'")

    def inspect(self, session_key, generation):
        if not self._valid_generation(generation):
            return self._unknown(session_key, generation)
        with self._db() as db:
            row = db.execute("SELECT * FROM executions WHERE session_key=? AND generation=?", (session_key, generation)).fetchone()
            unresolved = db.execute("SELECT 1 FROM executions WHERE session_key=? AND (worker!='completed' OR worker_started!=1 OR effects!='none' OR tool_observed!=0) LIMIT 1", (session_key,)).fetchone()
            unresolved_resource = db.execute("SELECT 1 FROM concurrent_tool_resources WHERE session_key=? AND state!='finished' LIMIT 1", (session_key,)).fetchone()
            unresolved_submit = db.execute("SELECT 1 FROM pending_submits WHERE session_key=? LIMIT 1",
                                           (session_key,)).fetchone()
        if row is None:
            return self._unknown(session_key, generation)
        if not row["session_id"]:
            return self._unknown(session_key, generation)
        return {"version": VERSION, "session_id": row["session_id"], "session_key": session_key, "run_generation": generation, "worker": row["worker"], "effects": "unknown" if unresolved or unresolved_resource or unresolved_submit else row["effects"]}

    @staticmethod
    def _unknown(session_key, generation):
        return {"version": VERSION, "session_id": None, "session_key": session_key, "run_generation": generation, "worker": "unknown", "effects": "unknown"}

    def retire_for_handoff(self, session_key, claim_generation, recovery_id):
        """Fence a quiescent thread for control handoff, never authorize replay or execution.

        Historical tool effects remain unknown. The successor must reconcile them;
        this receipt is deliberately not an execution-recovery receipt.
        """
        denied = {"version": VERSION, "status": "blocked", "effects": "unknown",
                  "execution_admitted": False}
        if (not session_key or not self._valid_generation(claim_generation)
                or not isinstance(recovery_id, str) or not recovery_id):
            return {**denied, "status": "stale"}
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            claim = db.execute("SELECT session_id FROM executions WHERE session_key=? AND generation=?",
                               (session_key, claim_generation)).fetchone()
            if not claim or not claim["session_id"]:
                db.rollback(); return denied
            keys = self._lineage_keys(db, session_key)
            ancestor = db.execute("SELECT 1 FROM session_key_lineage WHERE child_key=? LIMIT 1",
                                  (session_key,)).fetchone()
            # Core cannot prove that another Linear issue does not own a later
            # or earlier segment of this physical chat. Refuse before
            # fencing any segment; the plugin must refuse before preparing intent.
            if ancestor or len(keys) != 1:
                db.rollback(); return {**denied, "reason": "compressed_lineage"}
            marks = ",".join("?" for _ in keys)
            fences = {row["session_key"]: row for row in db.execute(
                f"SELECT * FROM session_fences WHERE session_key IN ({marks})", keys)}
            fence = fences.get(session_key)
            if fence:
                if fence["recovery_id"] != recovery_id or fence["state"] != "fenced":
                    db.rollback(); return {**denied, "status": "stale"}
                if fence["receipt"]:
                    receipt = json.loads(fence["receipt"])
                    db.rollback()
                    return receipt if receipt.get("kind") == "chat-retirement/v1" and receipt.get("claim_generation") == claim_generation else {**denied, "status": "stale"}
            if any(row["recovery_id"] != recovery_id or row["state"] != "fenced"
                   for row in fences.values()):
                db.rollback(); return {**denied, "status": "stale"}
            latest = db.execute("SELECT * FROM executions WHERE session_key=? ORDER BY generation DESC LIMIT 1",
                                (session_key,)).fetchone()
            submitting = db.execute(f"SELECT 1 FROM pending_submits WHERE session_key IN ({marks}) LIMIT 1",
                                    keys).fetchone()
            if submitting:
                db.rollback(); return denied
            for key in keys:
                if key not in fences:
                    key_latest = db.execute("SELECT MAX(generation) FROM executions WHERE session_key=?",
                                            (key,)).fetchone()[0]
                    db.execute("INSERT INTO session_fences VALUES (?, ?, ?, 'fenced', NULL)",
                               (key, key_latest or claim_generation, recovery_id))
            pending = db.execute(f"SELECT 1 FROM executions WHERE session_key IN ({marks}) AND "
                                 "(worker!='completed' OR worker_started!=1 OR session_id='') LIMIT 1",
                                 keys).fetchone()
            resources = db.execute(f"SELECT 1 FROM concurrent_tool_resources WHERE session_key IN ({marks}) "
                                   "AND state!='finished' LIMIT 1", keys).fetchone()
            if pending or resources:
                db.commit(); return denied
            unknown = db.execute(f"SELECT 1 FROM executions WHERE session_key IN ({marks}) AND "
                                 "(effects!='none' OR tool_observed!=0) LIMIT 1", keys).fetchone()
            receipt = {"version": VERSION, "kind": "chat-retirement/v1", "status": "retired",
                       "profile_home": str(self.home), "session_key": session_key,
                       "claim_generation": claim_generation, "run_generation": latest["generation"],
                       "session_id": latest["session_id"], "recovery_id": recovery_id,
                       "lineage_keys": list(keys),
                       "worker": "completed", "effects": "unknown" if unknown else "none",
                       "execution_admitted": False}
            db.execute("UPDATE session_fences SET receipt=? WHERE session_key=?", (json.dumps(receipt), session_key))
            db.commit()
            return receipt

    def begin_recovery(self, session_key, generation, recovery_id):
        if not self._valid_generation(generation) or not recovery_id:
            return False
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM executions WHERE session_key=? AND generation=?", (session_key, generation)).fetchone()
            if row is None or not row["session_id"]:
                db.rollback(); return False
            current = db.execute("SELECT * FROM session_fences WHERE session_key=?", (session_key,)).fetchone()
            if current:
                same = (current["generation"] == generation and current["recovery_id"] == recovery_id
                        and not current["receipt"])
                db.rollback()
                return same
            latest_active = db.execute("SELECT generation FROM executions WHERE session_key=? AND worker!='completed' ORDER BY generation DESC LIMIT 1", (session_key,)).fetchone()
            if latest_active and latest_active["generation"] != generation:
                db.rollback(); return False
            db.execute("INSERT INTO session_fences VALUES (?, ?, ?, 'fenced', NULL)", (session_key, generation, recovery_id))
            db.commit(); return True

    def recover(self, session_key, generation, recovery_id, transfer: Callable):
        if self._valid_generation(generation):
            with self._db() as db:
                saved = db.execute("SELECT generation, recovery_id, state, receipt FROM session_fences WHERE session_key=?", (session_key,)).fetchone()
            if saved and saved["generation"] == generation and saved["recovery_id"] == recovery_id and saved["state"] == "committed":
                return json.loads(saved["receipt"])
        if not self.begin_recovery(session_key, generation, recovery_id):
            return {"version": VERSION, "status": "stale", "effects": "unknown"}
        proof = self.inspect(session_key, generation)
        if proof["worker"] != "completed" or proof["effects"] != "none":
            return {**proof, "status": "blocked", "recovery_id": recovery_id}
        transfer_proof = {**proof, "recovery_id": recovery_id}
        result = transfer(transfer_proof)
        if result is False:
            return {**transfer_proof, "status": "blocked"}
        receipt = {**transfer_proof, "status": "committed", "transfer": result}
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            fence = db.execute("SELECT * FROM session_fences WHERE session_key=?", (session_key,)).fetchone()
            row = db.execute("SELECT * FROM executions WHERE session_key=? AND generation=?", (session_key, generation)).fetchone()
            pending_resource = db.execute("SELECT 1 FROM concurrent_tool_resources WHERE session_key=? AND state!='finished' LIMIT 1", (session_key,)).fetchone()
            pending_submit = db.execute("SELECT 1 FROM pending_submits WHERE session_key=? LIMIT 1", (session_key,)).fetchone()
            if not fence or fence["generation"] != generation or fence["recovery_id"] != recovery_id or fence["state"] != "fenced" or not row or row["worker"] != "completed" or row["effects"] != "none" or row["tool_observed"] or pending_resource or pending_submit:
                db.rollback(); return {**transfer_proof, "status": "blocked"}
            db.execute("UPDATE session_fences SET state='committed', receipt=? WHERE session_key=?", (json.dumps(receipt), session_key))
            db.commit()
        return receipt

    def recover_chat_execution(self, session_key, expected_run_generation, profile_home, recovery_id, transfer):
        if Path(profile_home).resolve() != self.home:
            return {"version": VERSION, "status": "stale", "effects": "unknown"}
        return self.recover(session_key, expected_run_generation, recovery_id, transfer)

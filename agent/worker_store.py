"""Durable delegation records in the existing profile-scoped SessionDB.

The service owns capability policy and tool execution. This module owns atomic
state transitions, lease fencing and idempotency; it never resolves credentials.
"""

from __future__ import annotations

import json
import math
import time
import uuid


_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS orchestration_workers (
        worker_id TEXT PRIMARY KEY, owner_session_id TEXT NOT NULL,
        parent_worker_id TEXT, root_worker_id TEXT NOT NULL, depth INTEGER NOT NULL,
        profile TEXT, config_revision TEXT NOT NULL, policy TEXT NOT NULL,
        frozen_prompt TEXT NOT NULL, frozen_prompt_hash TEXT NOT NULL,
        history TEXT NOT NULL DEFAULT '[]', uncertain_side_effect INTEGER NOT NULL DEFAULT 0,
        created_at REAL NOT NULL, updated_at REAL NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS orchestration_runs (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT UNIQUE NOT NULL,
        worker_id TEXT NOT NULL REFERENCES orchestration_workers(worker_id),
        request_id TEXT NOT NULL, previous_run_id TEXT,
        goal TEXT NOT NULL, context TEXT NOT NULL, status TEXT NOT NULL,
        capability_digest TEXT NOT NULL DEFAULT '',
        budget_epoch_id TEXT,
        lease_token TEXT, lease_expires_at REAL, tool_inflight INTEGER NOT NULL DEFAULT 0,
        tool_inflight_count INTEGER NOT NULL DEFAULT 0,
        uncertain_side_effect INTEGER NOT NULL DEFAULT 0, result TEXT,
        completion_ack INTEGER NOT NULL DEFAULT 0,
        created_at REAL NOT NULL, updated_at REAL NOT NULL,
        UNIQUE(worker_id, request_id))""",
    """CREATE UNIQUE INDEX IF NOT EXISTS orchestration_one_active
        ON orchestration_runs(worker_id) WHERE status = 'RUNNING'""",
    """CREATE TABLE IF NOT EXISTS orchestration_messages (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT, message_id TEXT UNIQUE NOT NULL,
        worker_id TEXT NOT NULL REFERENCES orchestration_workers(worker_id),
        sender_id TEXT, content TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'PENDING',
        delivered_run_id TEXT, created_at REAL NOT NULL, delivered_at REAL)""",
    """CREATE TABLE IF NOT EXISTS orchestration_tool_effects (
        run_id TEXT NOT NULL REFERENCES orchestration_runs(run_id),
        tool_call_id TEXT NOT NULL, status TEXT NOT NULL,
        admitted_at REAL NOT NULL, settled_at REAL,
        PRIMARY KEY(run_id, tool_call_id))""",
    """CREATE TABLE IF NOT EXISTS orchestration_budget_epochs (
        budget_epoch_id TEXT PRIMARY KEY, owner_session_id TEXT NOT NULL,
        root_worker_id TEXT NOT NULL, max_iterations INTEGER, used_iterations INTEGER NOT NULL DEFAULT 0,
        max_tool_calls INTEGER, used_tool_calls INTEGER NOT NULL DEFAULT 0,
        deadline_at REAL, created_at REAL NOT NULL, updated_at REAL NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS orchestration_budget_scopes (
        budget_epoch_id TEXT NOT NULL REFERENCES orchestration_budget_epochs(budget_epoch_id),
        worker_id TEXT NOT NULL REFERENCES orchestration_workers(worker_id),
        max_iterations INTEGER, used_iterations INTEGER NOT NULL DEFAULT 0,
        max_tool_calls INTEGER, used_tool_calls INTEGER NOT NULL DEFAULT 0,
        deadline_at REAL, PRIMARY KEY(budget_epoch_id,worker_id))""",
    """CREATE TABLE IF NOT EXISTS orchestration_parent_messages (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT, message_id TEXT UNIQUE NOT NULL,
        owner_session_id TEXT NOT NULL, worker_id TEXT NOT NULL REFERENCES orchestration_workers(worker_id),
        run_id TEXT NOT NULL REFERENCES orchestration_runs(run_id), content TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'QUEUED', created_at REAL NOT NULL,
        published_at REAL, acknowledged_at REAL)""",
    "CREATE INDEX IF NOT EXISTS orchestration_owner ON orchestration_workers(owner_session_id)",
    "CREATE INDEX IF NOT EXISTS orchestration_queue ON orchestration_runs(worker_id, status, sequence)",
    "CREATE INDEX IF NOT EXISTS orchestration_mailbox ON orchestration_messages(worker_id, status, sequence)",
    "CREATE INDEX IF NOT EXISTS orchestration_tool_effect_state ON orchestration_tool_effects(run_id, status)",
    "CREATE INDEX IF NOT EXISTS orchestration_parent_outbox ON orchestration_parent_messages(owner_session_id,status,sequence)",
)
_JSON_FIELDS = {"policy", "history", "result"}
_SECRET_KEYS = {"api_key", "access_token", "refresh_token", "token", "password", "authorization", "cookie", "credentials"}
_TERMINAL = {"SUCCEEDED", "FAILED", "INTERRUPTED", "CANCELLED"}


class WorkerBudgetExceeded(ValueError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason.replace("_", " "))


def _row(row):
    if row is None:
        return None
    value = dict(row)
    for key in _JSON_FIELDS.intersection(value):
        value[key] = json.loads(value[key]) if value[key] is not None else None
    for key in {"uncertain_side_effect", "tool_inflight", "completion_ack"}.intersection(value):
        value[key] = bool(value[key])
    return value


def _policy_json(value):
    def check(item):
        if isinstance(item, dict):
            if any(str(key).lower() in _SECRET_KEYS for key in item):
                raise ValueError("Worker policy must not contain credentials")
            for child in item.values():
                check(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check(child)
    check(value)
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


class WorkerStore:
    def __init__(self, session_db):
        self.db = session_db

    def ensure_schema(self):
        def migrate(conn):
            for statement in _SCHEMA:
                conn.execute(statement)
            run_columns = {row[1] for row in conn.execute("PRAGMA table_info(orchestration_runs)")}
            if "tool_inflight_count" not in run_columns:
                conn.execute(
                    "ALTER TABLE orchestration_runs ADD COLUMN tool_inflight_count INTEGER NOT NULL DEFAULT 0")
            if "capability_digest" not in run_columns:
                conn.execute(
                    "ALTER TABLE orchestration_runs ADD COLUMN capability_digest TEXT NOT NULL DEFAULT ''")
            if "budget_epoch_id" not in run_columns:
                conn.execute("ALTER TABLE orchestration_runs ADD COLUMN budget_epoch_id TEXT")
            # Older boolean/count checkpoints have no call identity. Preserve
            # them as explicit unknown effects so recovery stays fail-closed.
            legacy = conn.execute("""SELECT run_id,tool_inflight_count FROM orchestration_runs r
                WHERE tool_inflight_count>0 AND NOT EXISTS (
                    SELECT 1 FROM orchestration_tool_effects e WHERE e.run_id=r.run_id)""").fetchall()
            for row in legacy:
                for index in range(int(row[1])):
                    conn.execute("""INSERT INTO orchestration_tool_effects
                        (run_id,tool_call_id,status,admitted_at) VALUES (?,?,?,?)""",
                        (row[0], f"legacy-unknown-{index}", "UNKNOWN", time.time()))
        self.db._execute_write(migrate)

    def schema_present(self) -> bool:
        """Return whether the durable worker schema already exists, without creating it."""
        required = {
            "orchestration_workers", "orchestration_runs", "orchestration_messages",
            "orchestration_parent_messages", "orchestration_tool_effects",
        }
        with self.db._read_ctx() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'orchestration_%'"
            ).fetchall()
        return required.issubset({str(row[0]) for row in rows})

    def list_workers_existing(self, owner_session_id):
        """List owner-scoped workers only when the current schema is already present."""
        if not self.schema_present():
            return []
        return self.list_workers(owner_session_id)

    def get_worker_existing(self, worker_id, owner_session_id):
        if not self.schema_present():
            raise PermissionError("Unknown reference or unavailable discovery store")
        return self.get_worker(worker_id, owner_session_id)

    def list_runs_existing(self, worker_id, owner_session_id):
        if not self.schema_present():
            return []
        return self.list_runs(worker_id, owner_session_id)

    def get_run_existing(self, run_id, owner_session_id):
        if not self.schema_present():
            raise PermissionError("Unknown reference or unavailable discovery store")
        return self.get_run(run_id, owner_session_id)

    @staticmethod
    def _worker(conn, worker_id, owner_session_id):
        worker = _row(conn.execute("SELECT * FROM orchestration_workers WHERE worker_id=?", (worker_id,)).fetchone())
        if worker is None or not owner_session_id or worker["owner_session_id"] != owner_session_id:
            raise PermissionError("Unknown worker or foreign owner")
        return worker

    @classmethod
    def _run(cls, conn, run_id, owner_session_id):
        run = _row(conn.execute("SELECT * FROM orchestration_runs WHERE run_id=?", (run_id,)).fetchone())
        if run is None:
            raise PermissionError("Unknown run or foreign owner")
        cls._worker(conn, run["worker_id"], owner_session_id)
        return run

    @classmethod
    def _lease(cls, conn, run_id, owner_session_id, lease_token):
        run = cls._run(conn, run_id, owner_session_id)
        if (not lease_token or run["status"] != "RUNNING" or run["lease_token"] != lease_token
                or run["lease_expires_at"] <= time.time()):
            raise PermissionError("Worker execution lease is no longer valid")
        return run

    def create_worker(self, owner_session_id, *, profile=None, config_revision="", policy=None,
                      frozen_prompt="", parent_worker_id=None, worker_id=None):
        import hashlib

        if not isinstance(owner_session_id, str) or not owner_session_id.strip():
            raise ValueError("A stable owner session is required")
        worker_id = worker_id or "worker-" + uuid.uuid4().hex
        policy_text = _policy_json(policy or {})
        prompt_hash = hashlib.sha256(frozen_prompt.encode()).hexdigest()
        def create(conn):
            parent = self._worker(conn, parent_worker_id, owner_session_id) if parent_worker_id else None
            now = time.time()
            conn.execute("""INSERT INTO orchestration_workers
                (worker_id,owner_session_id,parent_worker_id,root_worker_id,depth,profile,
                 config_revision,policy,frozen_prompt,frozen_prompt_hash,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (worker_id, owner_session_id, parent_worker_id,
                 parent["root_worker_id"] if parent else worker_id, parent["depth"] + 1 if parent else 1,
                 profile, config_revision, policy_text, frozen_prompt, prompt_hash, now, now))
            return self._worker(conn, worker_id, owner_session_id)
        return self.db._execute_write(create)

    def get_worker(self, worker_id, owner_session_id):
        with self.db._read_ctx() as conn:
            return self._worker(conn, worker_id, owner_session_id)

    def list_workers(self, owner_session_id):
        with self.db._read_ctx() as conn:
            return [_row(r) for r in conn.execute(
                "SELECT * FROM orchestration_workers WHERE owner_session_id=? ORDER BY created_at,worker_id",
                (owner_session_id,))]

    def active_run_count(self, owner_session_id):
        with self.db._read_ctx() as conn:
            return int(conn.execute("""SELECT count(*) FROM orchestration_runs r
                JOIN orchestration_workers w ON r.worker_id=w.worker_id
                WHERE w.owner_session_id=? AND r.status='RUNNING'""", (owner_session_id,)).fetchone()[0])

    def get_run(self, run_id, owner_session_id):
        with self.db._read_ctx() as conn:
            return self._run(conn, run_id, owner_session_id)

    def list_runs(self, worker_id, owner_session_id):
        with self.db._read_ctx() as conn:
            self._worker(conn, worker_id, owner_session_id)
            return [_row(row) for row in conn.execute(
                "SELECT * FROM orchestration_runs WHERE worker_id=? ORDER BY sequence",
                (worker_id,),
            )]

    def pending_runs(self, owner_session_id):
        """Owner-scoped durable FIFO queue, including runs with no process record."""
        with self.db._read_ctx() as conn:
            return [_row(row) for row in conn.execute("""SELECT r.* FROM orchestration_runs r
                JOIN orchestration_workers w ON r.worker_id=w.worker_id
                LEFT JOIN orchestration_runs previous ON previous.run_id=r.previous_run_id
                WHERE w.owner_session_id=? AND r.status='PENDING'
                AND (r.previous_run_id IS NULL OR previous.status IN ('SUCCEEDED','FAILED','INTERRUPTED','CANCELLED'))
                ORDER BY r.sequence""", (owner_session_id,))]

    def enqueue_run(
        self, worker_id, owner_session_id, *, goal, context="", request_id=None,
        previous_run_id=None, capability_digest="", budget_epoch_id=None, budget_limits=None,
    ):
        if not isinstance(goal, str) or not goal.strip() or not isinstance(context, str):
            raise ValueError("A run needs a nonempty goal and text context")
        if not isinstance(capability_digest, str):
            raise ValueError("capability_digest must be text")
        run_id, request_id = "run-" + uuid.uuid4().hex, request_id or uuid.uuid4().hex
        def enqueue(conn):
            worker = self._worker(conn, worker_id, owner_session_id)
            existing = _row(conn.execute("SELECT * FROM orchestration_runs WHERE worker_id=? AND request_id=?",
                                         (worker_id, request_id)).fetchone())
            if existing:
                if (existing["goal"], existing["context"], existing["previous_run_id"]) != (goal, context, previous_run_id):
                    raise ValueError("Run request ID was already used for different content")
                return existing
            if worker["uncertain_side_effect"]:
                raise ValueError("Reconcile the interrupted tool outcome before resuming this worker")
            if previous_run_id:
                previous = self._run(conn, previous_run_id, owner_session_id)
                latest = conn.execute(
                    "SELECT run_id FROM orchestration_runs WHERE worker_id=? ORDER BY sequence DESC LIMIT 1",
                    (worker_id,),
                ).fetchone()
                if previous["worker_id"] != worker_id or not latest or latest[0] != previous_run_id:
                    raise ValueError("Followup must link to this worker's latest run")
            now = time.time()
            epoch_id = budget_epoch_id
            if budget_limits is not None:
                limits = self._normalize_budget_limits(budget_limits)
                if epoch_id is None:
                    if worker["parent_worker_id"] is not None:
                        raise ValueError("A nested worker must inherit its parent budget epoch")
                    epoch_id = "budget-" + uuid.uuid4().hex
                    deadline = now + limits["timeout_seconds"] if limits["timeout_seconds"] else None
                    conn.execute("""INSERT INTO orchestration_budget_epochs
                        (budget_epoch_id,owner_session_id,root_worker_id,max_iterations,max_tool_calls,
                         deadline_at,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?)""",
                        (epoch_id, owner_session_id, worker["root_worker_id"], limits["max_iterations"],
                         limits["max_tool_calls"], deadline, now, now))
                epoch = self._budget_epoch(conn, epoch_id, owner_session_id)
                if epoch["root_worker_id"] != worker["root_worker_id"]:
                    raise PermissionError("Budget epoch belongs to another worker tree")
                parent_deadline = epoch["deadline_at"]
                if worker["parent_worker_id"]:
                    parent_scope = conn.execute("""SELECT deadline_at FROM orchestration_budget_scopes
                        WHERE budget_epoch_id=? AND worker_id=?""",
                        (epoch_id, worker["parent_worker_id"])).fetchone()
                    if parent_scope is None:
                        raise ValueError("Parent budget scope is unavailable")
                    parent_deadline = parent_scope[0]
                local_deadline = now + limits["timeout_seconds"] if limits["timeout_seconds"] else None
                deadline = min(v for v in (parent_deadline, local_deadline) if v is not None) \
                    if parent_deadline is not None or local_deadline is not None else None
                conn.execute("""INSERT OR IGNORE INTO orchestration_budget_scopes
                    (budget_epoch_id,worker_id,max_iterations,max_tool_calls,deadline_at)
                    VALUES (?,?,?,?,?)""", (epoch_id, worker_id, limits["max_iterations"],
                                              limits["max_tool_calls"], deadline))
            conn.execute("""INSERT INTO orchestration_runs
                (run_id,worker_id,request_id,previous_run_id,goal,context,status,
                 capability_digest,budget_epoch_id,created_at,updated_at)
                VALUES (?,?,?,?,?,?,'PENDING',?,?,?,?)""",
                (run_id, worker_id, request_id, previous_run_id, goal, context,
                 capability_digest, epoch_id, now, now))
            return self._run(conn, run_id, owner_session_id)
        return self.db._execute_write(enqueue)

    @staticmethod
    def _normalize_budget_limits(value):
        if not isinstance(value, dict):
            raise ValueError("budget_limits must be a mapping")
        result = {}
        for name in ("max_iterations", "max_tool_calls"):
            item = value.get(name)
            if item is not None and (isinstance(item, bool) or not isinstance(item, int) or item < 0):
                raise ValueError(f"{name} must be a nonnegative integer or null")
            result[name] = item
        timeout = value.get("timeout_seconds")
        if timeout is not None:
            _positive(timeout, "timeout_seconds")
        result["timeout_seconds"] = timeout
        return result

    @staticmethod
    def _budget_epoch(conn, budget_epoch_id, owner_session_id):
        row = conn.execute("SELECT * FROM orchestration_budget_epochs WHERE budget_epoch_id=?",
                           (budget_epoch_id,)).fetchone()
        if row is None or row["owner_session_id"] != owner_session_id:
            raise PermissionError("Unknown budget epoch or foreign owner")
        return dict(row)

    @classmethod
    def _reserve_budget(cls, conn, run, owner_session_id, kind):
        epoch_id = run.get("budget_epoch_id")
        if not epoch_id:
            return
        epoch = cls._budget_epoch(conn, epoch_id, owner_session_id)
        worker = cls._worker(conn, run["worker_id"], owner_session_id)
        scopes = []
        cursor = worker
        while cursor is not None:
            scope = conn.execute("""SELECT * FROM orchestration_budget_scopes
                WHERE budget_epoch_id=? AND worker_id=?""", (epoch_id, cursor["worker_id"])).fetchone()
            if scope is None:
                raise PermissionError("Worker budget scope is unavailable")
            scopes.append(dict(scope))
            cursor = cls._worker(conn, cursor["parent_worker_id"], owner_session_id) \
                if cursor["parent_worker_id"] else None
        now = time.time()
        deadlines = [item["deadline_at"] for item in [epoch, *scopes] if item["deadline_at"] is not None]
        if deadlines and now >= min(deadlines):
            raise WorkerBudgetExceeded("tree_deadline_exhausted")
        max_key, used_key = f"max_{kind}s", f"used_{kind}s"
        for item in [epoch, *scopes]:
            if item[max_key] is not None and item[used_key] >= item[max_key]:
                raise WorkerBudgetExceeded(f"tree_{kind}_budget_exhausted")
        conn.execute(f"UPDATE orchestration_budget_epochs SET {used_key}={used_key}+1,updated_at=? WHERE budget_epoch_id=?",
                     (now, epoch_id))
        for scope in scopes:
            conn.execute(f"UPDATE orchestration_budget_scopes SET {used_key}={used_key}+1 WHERE budget_epoch_id=? AND worker_id=?",
                         (epoch_id, scope["worker_id"]))

    def reserve_iteration(self, run_id, owner_session_id, lease_token):
        def reserve(conn):
            run = self._lease(conn, run_id, owner_session_id, lease_token)
            self._reserve_budget(conn, run, owner_session_id, "iteration")
        self.db._execute_write(reserve)

    def budget_snapshot(self, run_id, owner_session_id):
        with self.db._read_ctx() as conn:
            run = self._run(conn, run_id, owner_session_id)
            if not run.get("budget_epoch_id"):
                return None
            epoch = self._budget_epoch(conn, run["budget_epoch_id"], owner_session_id)
            return {key: epoch[key] for key in (
                "budget_epoch_id", "root_worker_id", "max_iterations", "used_iterations",
                "max_tool_calls", "used_tool_calls", "deadline_at")}

    def claim_next_run(self, worker_id, owner_session_id, *, lease_seconds=60, max_concurrent=10):
        _positive(lease_seconds, "lease_seconds")
        if isinstance(max_concurrent, bool) or not isinstance(max_concurrent, int) or max_concurrent < 1:
            raise ValueError("max_concurrent must be a positive integer")
        token = uuid.uuid4().hex
        def claim(conn):
            worker = self._worker(conn, worker_id, owner_session_id)
            if worker["uncertain_side_effect"]:
                return None
            # Expired RUNNING rows still count until explicit recovery fences them.
            active = conn.execute("SELECT 1 FROM orchestration_runs WHERE worker_id=? AND status='RUNNING'", (worker_id,)).fetchone()
            count = conn.execute("""SELECT count(*) FROM orchestration_runs r JOIN orchestration_workers w
                ON r.worker_id=w.worker_id WHERE w.owner_session_id=? AND r.status='RUNNING'""",
                (owner_session_id,)).fetchone()[0]
            if active or count >= max_concurrent:
                return None
            run = conn.execute("""SELECT pending.run_id FROM orchestration_runs pending
                LEFT JOIN orchestration_runs previous ON previous.run_id=pending.previous_run_id
                WHERE pending.worker_id=? AND pending.status='PENDING'
                AND (pending.previous_run_id IS NULL OR previous.status IN ('SUCCEEDED','FAILED','INTERRUPTED','CANCELLED'))
                ORDER BY pending.sequence LIMIT 1""", (worker_id,)).fetchone()
            if run is None:
                return None
            now = time.time()
            conn.execute("UPDATE orchestration_runs SET status='RUNNING',lease_token=?,lease_expires_at=?,updated_at=? WHERE run_id=?",
                         (token, now + lease_seconds, now, run[0]))
            return self._run(conn, run[0], owner_session_id)
        return self.db._execute_write(claim)

    def claim_run(self, run_id, owner_session_id, *, lease_seconds=60, max_concurrent=10):
        """Claim this exact FIFO-eligible run or return None without leasing a peer."""
        _positive(lease_seconds, "lease_seconds")
        if isinstance(max_concurrent, bool) or not isinstance(max_concurrent, int) or max_concurrent < 1:
            raise ValueError("max_concurrent must be a positive integer")
        token = uuid.uuid4().hex
        def claim(conn):
            run = self._run(conn, run_id, owner_session_id)
            worker = self._worker(conn, run["worker_id"], owner_session_id)
            if run["status"] != "PENDING" or worker["uncertain_side_effect"]:
                return None
            active = conn.execute(
                "SELECT 1 FROM orchestration_runs WHERE worker_id=? AND status='RUNNING'",
                (run["worker_id"],),
            ).fetchone()
            count = conn.execute("""SELECT count(*) FROM orchestration_runs r
                JOIN orchestration_workers w ON r.worker_id=w.worker_id
                WHERE w.owner_session_id=? AND r.status='RUNNING'""", (owner_session_id,)).fetchone()[0]
            eligible = conn.execute("""SELECT pending.run_id FROM orchestration_runs pending
                LEFT JOIN orchestration_runs previous ON previous.run_id=pending.previous_run_id
                WHERE pending.worker_id=? AND pending.status='PENDING'
                AND (pending.previous_run_id IS NULL OR previous.status IN ('SUCCEEDED','FAILED','INTERRUPTED','CANCELLED'))
                ORDER BY pending.sequence LIMIT 1""", (run["worker_id"],)).fetchone()
            if active or count >= max_concurrent or not eligible or eligible[0] != run_id:
                return None
            now = time.time()
            conn.execute("""UPDATE orchestration_runs SET status='RUNNING',lease_token=?,lease_expires_at=?,updated_at=?
                WHERE run_id=? AND status='PENDING'""", (token, now + lease_seconds, now, run_id))
            return self._run(conn, run_id, owner_session_id)
        return self.db._execute_write(claim)

    def heartbeat_run(self, run_id, owner_session_id, lease_token, *, lease_seconds=60):
        _positive(lease_seconds, "lease_seconds")
        def heartbeat(conn):
            self._lease(conn, run_id, owner_session_id, lease_token)
            now = time.time()
            conn.execute("UPDATE orchestration_runs SET lease_expires_at=?,updated_at=? WHERE run_id=?", (now + lease_seconds, now, run_id))
        self.db._execute_write(heartbeat)

    def checkpoint_run(self, run_id, owner_session_id, lease_token, *, history, tool_inflight=None, delivered_message_ids=()):
        history_json = json.dumps(history, allow_nan=False)
        def checkpoint(conn):
            run = self._lease(conn, run_id, owner_session_id, lease_token)
            now = time.time()
            conn.execute("UPDATE orchestration_workers SET history=?,updated_at=? WHERE worker_id=?", (history_json, now, run["worker_id"]))
            if tool_inflight is True:
                call_id = "legacy-checkpoint-" + uuid.uuid4().hex
                conn.execute("""INSERT INTO orchestration_tool_effects
                    (run_id,tool_call_id,status,admitted_at) VALUES (?,?,'UNKNOWN',?)""",
                    (run_id, call_id, now))
            count = conn.execute("""SELECT count(*) FROM orchestration_tool_effects
                WHERE run_id=? AND status IN ('INFLIGHT','UNKNOWN')""", (run_id,)).fetchone()[0]
            conn.execute("""UPDATE orchestration_runs SET tool_inflight=?,tool_inflight_count=?,updated_at=?
                WHERE run_id=?""", (int(count > 0), count, now, run_id))
            for message_id in delivered_message_ids:
                self._ack_message(conn, run, message_id)
        self.db._execute_write(checkpoint)

    def mark_tool_boundary(self, run_id, owner_session_id, lease_token, *, tool_call_id=None, tool_inflight=True):
        """Admit one identified effect before dispatch; anonymous clearing is forbidden."""
        if tool_inflight is not True:
            raise ValueError("Tool effects clear only through their matching result checkpoint")
        tool_call_id = tool_call_id or "legacy-tool-" + uuid.uuid4().hex
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise ValueError("tool_call_id must be nonempty text")
        def mark(conn):
            run = self._lease(conn, run_id, owner_session_id, lease_token)
            existing = conn.execute("""SELECT status FROM orchestration_tool_effects
                WHERE run_id=? AND tool_call_id=?""", (run_id, tool_call_id)).fetchone()
            if existing is not None:
                raise ValueError("Tool call was already admitted; refusing duplicate execution")
            self._reserve_budget(conn, run, owner_session_id, "tool_call")
            now = time.time()
            conn.execute("""INSERT INTO orchestration_tool_effects
                (run_id,tool_call_id,status,admitted_at) VALUES (?,?,'INFLIGHT',?)""",
                (run_id, tool_call_id, now))
            count = conn.execute("""SELECT count(*) FROM orchestration_tool_effects
                WHERE run_id=? AND status IN ('INFLIGHT','UNKNOWN')""", (run_id,)).fetchone()[0]
            conn.execute("""UPDATE orchestration_runs SET tool_inflight=?,tool_inflight_count=?,updated_at=?
                WHERE run_id=?""", (int(count > 0), count, now, run_id))
            return tool_call_id
        return self.db._execute_write(mark)

    def checkpoint_tool_result(
        self, run_id, owner_session_id, lease_token, *, history,
        tool_call_id=None, admitted=None, settled=True, delivered_message_ids=(),
    ):
        """Persist one outcome and settle only its matching admitted effect."""
        history_json = json.dumps(history, allow_nan=False)
        def checkpoint(conn):
            run = self._lease(conn, run_id, owner_session_id, lease_token)
            now = time.time()
            effect = None
            if isinstance(tool_call_id, str) and tool_call_id:
                effect = conn.execute("""SELECT status FROM orchestration_tool_effects
                    WHERE run_id=? AND tool_call_id=?""", (run_id, tool_call_id)).fetchone()
            admitted_here = effect is not None if admitted is None else bool(admitted)
            if admitted_here:
                if not isinstance(tool_call_id, str) or not tool_call_id:
                    raise ValueError("An admitted tool result requires tool_call_id")
                if effect is None or effect[0] != "INFLIGHT":
                    raise ValueError("Tool result does not match an inflight admitted call")
                conn.execute("""UPDATE orchestration_tool_effects SET status=?,settled_at=?
                    WHERE run_id=? AND tool_call_id=?""",
                    ("SETTLED" if settled else "UNKNOWN", now, run_id, tool_call_id))
            conn.execute(
                "UPDATE orchestration_workers SET history=?,updated_at=? WHERE worker_id=?",
                (history_json, now, run["worker_id"]),
            )
            count = conn.execute("""SELECT count(*) FROM orchestration_tool_effects
                WHERE run_id=? AND status IN ('INFLIGHT','UNKNOWN')""", (run_id,)).fetchone()[0]
            conn.execute("""UPDATE orchestration_runs SET tool_inflight=?,tool_inflight_count=?,updated_at=?
                WHERE run_id=?""", (int(count > 0), count, now, run_id))
            for message_id in delivered_message_ids:
                self._ack_message(conn, run, message_id)
        self.db._execute_write(checkpoint)

    def finish_run(
        self, run_id, owner_session_id, lease_token, *, status, result, history=None,
        delivered_message_ids=(),
    ):
        if status not in _TERMINAL:
            raise ValueError("Invalid terminal run status")
        def finish(conn):
            run = self._lease(conn, run_id, owner_session_id, lease_token)
            now = time.time()
            uncertain = bool(run["tool_inflight"])
            parent_messages = [dict(row) for row in conn.execute("""SELECT message_id,content
                FROM orchestration_parent_messages WHERE run_id=? AND status='QUEUED' ORDER BY sequence""",
                (run_id,))]
            stored_result = dict(result)
            if parent_messages:
                stored_result["messages_to_parent"] = [
                    {**item, "status": "PUBLISHED"} for item in parent_messages]
            result_json = _policy_json(stored_result)
            conn.execute("""UPDATE orchestration_runs SET status=?,result=?,uncertain_side_effect=?,
                lease_token=NULL,lease_expires_at=NULL,updated_at=? WHERE run_id=?""",
                (status, result_json, int(uncertain), now, run_id))
            conn.execute("UPDATE orchestration_workers SET uncertain_side_effect=?,updated_at=? WHERE worker_id=?", (int(uncertain), now, run["worker_id"]))
            if history is not None:
                conn.execute("UPDATE orchestration_workers SET history=? WHERE worker_id=?", (json.dumps(history, allow_nan=False), run["worker_id"]))
            for message_id in delivered_message_ids:
                self._ack_message(conn, run, message_id)
            conn.execute("""UPDATE orchestration_parent_messages SET status='PUBLISHED',published_at=?
                WHERE run_id=? AND status='QUEUED'""", (now, run_id))
            return self._run(conn, run_id, owner_session_id)
        return self.db._execute_write(finish)

    def enqueue_parent_message(self, run_id, owner_session_id, content, *, message_id=None):
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Message must be nonempty text")
        message_id = message_id or "message-" + uuid.uuid4().hex
        def enqueue(conn):
            run = self._run(conn, run_id, owner_session_id)
            if run["status"] != "RUNNING":
                raise ValueError("Parent messages require an active worker run")
            conn.execute("""INSERT INTO orchestration_parent_messages
                (message_id,owner_session_id,worker_id,run_id,content,created_at)
                VALUES (?,?,?,?,?,?)""",
                (message_id, owner_session_id, run["worker_id"], run_id, content, time.time()))
            return dict(conn.execute("SELECT * FROM orchestration_parent_messages WHERE message_id=?",
                                     (message_id,)).fetchone())
        return self.db._execute_write(enqueue)

    def list_parent_messages(self, run_id, owner_session_id):
        with self.db._read_ctx() as conn:
            self._run(conn, run_id, owner_session_id)
            return [dict(row) for row in conn.execute("""SELECT * FROM orchestration_parent_messages
                WHERE run_id=? ORDER BY sequence""", (run_id,))]

    def enqueue_message(self, worker_id, owner_session_id, content, *, message_id=None, sender_id=None):
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Message must be nonempty text")
        message_id = message_id or "message-" + uuid.uuid4().hex
        def enqueue(conn):
            self._worker(conn, worker_id, owner_session_id)
            existing = _row(conn.execute("SELECT * FROM orchestration_messages WHERE message_id=?", (message_id,)).fetchone())
            if existing:
                if (existing["worker_id"], existing["content"], existing["sender_id"]) != (worker_id, content, sender_id):
                    raise ValueError("Message ID was already used for different content")
                return existing
            conn.execute("INSERT INTO orchestration_messages(message_id,worker_id,sender_id,content,created_at) VALUES (?,?,?,?,?)",
                         (message_id, worker_id, sender_id, content, time.time()))
            return _row(conn.execute("SELECT * FROM orchestration_messages WHERE message_id=?", (message_id,)).fetchone())
        return self.db._execute_write(enqueue)

    def list_messages(self, worker_id, owner_session_id):
        with self.db._read_ctx() as conn:
            self._worker(conn, worker_id, owner_session_id)
            return [_row(row) for row in conn.execute(
                "SELECT * FROM orchestration_messages WHERE worker_id=? ORDER BY sequence", (worker_id,),
            )]

    def claim_messages(self, run_id, owner_session_id, lease_token):
        # Delivery becomes final only alongside a persisted conversation checkpoint.
        with self.db._read_ctx() as conn:
            run = self._lease(conn, run_id, owner_session_id, lease_token)
            return [_row(r) for r in conn.execute("SELECT * FROM orchestration_messages WHERE worker_id=? AND status='PENDING' ORDER BY sequence", (run["worker_id"],))]

    @staticmethod
    def _ack_message(conn, run, message_id):
        message = conn.execute("SELECT worker_id FROM orchestration_messages WHERE message_id=?", (message_id,)).fetchone()
        if not message or message[0] != run["worker_id"]:
            raise PermissionError("Unknown message or foreign worker")
        conn.execute("""UPDATE orchestration_messages SET status='DELIVERED',delivered_run_id=?,delivered_at=?
            WHERE message_id=? AND status='PENDING'""", (run["run_id"], time.time(), message_id))

    def ack_message(self, run_id, owner_session_id, lease_token, message_id):
        def ack(conn):
            self._ack_message(conn, self._lease(conn, run_id, owner_session_id, lease_token), message_id)
        self.db._execute_write(ack)

    def recover_expired_runs(self, owner_session_id, worker_ids=None):
        worker_ids = tuple(dict.fromkeys(worker_ids or ()))
        def recover(conn):
            params = [owner_session_id, time.time()]
            scope = ""
            if worker_ids:
                for worker_id in worker_ids:
                    self._worker(conn, worker_id, owner_session_id)
                scope = f" AND r.worker_id IN ({','.join('?' for _ in worker_ids)})"
                params.extend(worker_ids)
            expired = conn.execute("""SELECT r.run_id FROM orchestration_runs r JOIN orchestration_workers w
                ON r.worker_id=w.worker_id WHERE w.owner_session_id=? AND r.status='RUNNING'
                AND r.lease_expires_at<=?""" + scope, params).fetchall()
            for row in expired:
                run = self._run(conn, row[0], owner_session_id)
                uncertain = int(run["tool_inflight"])
                parent_messages = [dict(item) for item in conn.execute("""SELECT message_id,content
                    FROM orchestration_parent_messages WHERE run_id=? AND status='QUEUED' ORDER BY sequence""",
                    (row[0],))]
                result = {"reason": "execution_lease_expired", "uncertain_side_effect": bool(uncertain)}
                if parent_messages:
                    result["messages_to_parent"] = [
                        {**item, "status": "PUBLISHED"} for item in parent_messages]
                conn.execute("""UPDATE orchestration_runs SET status='INTERRUPTED',uncertain_side_effect=?,
                    lease_token=NULL,lease_expires_at=NULL,result=?,updated_at=? WHERE run_id=?""",
                    (uncertain, _policy_json(result), time.time(), row[0]))
                conn.execute("""UPDATE orchestration_parent_messages SET status='PUBLISHED',published_at=?
                    WHERE run_id=? AND status='QUEUED'""", (time.time(), row[0]))
                conn.execute("UPDATE orchestration_workers SET uncertain_side_effect=?,updated_at=? WHERE worker_id=?", (uncertain, time.time(), run["worker_id"]))
            return [self._run(conn, r[0], owner_session_id) for r in expired]
        return self.db._execute_write(recover)

    def cancel_pending_run(self, run_id, owner_session_id):
        """Interrupt one exact queued run without cancelling its durable worker."""
        def cancel(conn):
            run = self._run(conn, run_id, owner_session_id)
            if run["status"] != "PENDING":
                return None
            now = time.time()
            conn.execute(
                "UPDATE orchestration_runs SET status='INTERRUPTED',result=?,updated_at=? WHERE run_id=?",
                (json.dumps({"reason": "worker_run_interrupted_before_start"}), now, run_id),
            )
            return self._run(conn, run_id, owner_session_id)
        return self.db._execute_write(cancel)

    def cancel_pending_runs(self, worker_ids, owner_session_id):
        """Cancel queued work for an owner-verified subtree; live runs remain lease-fenced."""
        worker_ids = tuple(dict.fromkeys(worker_ids))
        if not worker_ids:
            return []
        def cancel(conn):
            for worker_id in worker_ids:
                self._worker(conn, worker_id, owner_session_id)
            placeholders = ",".join("?" for _ in worker_ids)
            now = time.time()
            rows = conn.execute(
                f"SELECT run_id FROM orchestration_runs WHERE worker_id IN ({placeholders}) AND status='PENDING'",
                worker_ids,
            ).fetchall()
            for row in rows:
                conn.execute(
                    "UPDATE orchestration_runs SET status='CANCELLED',result=?,updated_at=? WHERE run_id=?",
                    (json.dumps({"reason": "worker_tree_cancelled"}), now, row[0]),
                )
            return [self._run(conn, row[0], owner_session_id) for row in rows]
        return self.db._execute_write(cancel)

    def reconcile_run(self, run_id, owner_session_id, *, disposition, note):
        allowed = {"confirmed_applied", "confirmed_not_applied", "accepted_unknown_no_replay"}
        if disposition not in allowed:
            raise ValueError(f"Invalid reconciliation disposition; choose one of {sorted(allowed)}")
        if not isinstance(note, str) or not note.strip():
            raise ValueError("Reconciliation requires a nonempty decision note")
        def reconcile(conn):
            run = self._run(conn, run_id, owner_session_id)
            if run["status"] not in _TERMINAL:
                raise ValueError("Only a terminal run can be reconciled")
            latest = conn.execute("SELECT run_id FROM orchestration_runs WHERE worker_id=? AND status IN ('INTERRUPTED','FAILED','CANCELLED','SUCCEEDED') ORDER BY sequence DESC LIMIT 1", (run["worker_id"],)).fetchone()
            if not latest or latest[0] != run_id:
                raise ValueError("Reconcile the latest terminal run")
            now = time.time()
            result = dict(run.get("result") or {})
            effects = [
                {"tool_call_id": row[0], "prior_status": row[1]}
                for row in conn.execute("""SELECT tool_call_id,status FROM orchestration_tool_effects
                    WHERE run_id=? AND status IN ('INFLIGHT','UNKNOWN') ORDER BY tool_call_id""", (run_id,))
            ]
            result["reconciliation"] = {
                "disposition": disposition,
                "note": note.strip(),
                "affected_tool_calls": effects,
                "reconciled_at": now,
            }
            conn.execute("""UPDATE orchestration_tool_effects SET status='RECONCILED',settled_at=?
                WHERE run_id=? AND status IN ('INFLIGHT','UNKNOWN')""", (now, run_id))
            conn.execute("""UPDATE orchestration_runs SET uncertain_side_effect=0,tool_inflight=0,
                tool_inflight_count=0,result=?,updated_at=? WHERE run_id=?""",
                (_policy_json(result), now, run_id))
            conn.execute("UPDATE orchestration_workers SET uncertain_side_effect=0,updated_at=? WHERE worker_id=?", (now, run["worker_id"]))
        self.db._execute_write(reconcile)

    def pending_completions(self, owner_session_id):
        with self.db._read_ctx() as conn:
            return [_row(r) for r in conn.execute("""SELECT r.* FROM orchestration_runs r JOIN orchestration_workers w
                ON r.worker_id=w.worker_id WHERE w.owner_session_id=? AND r.completion_ack=0
                AND r.status IN ('SUCCEEDED','FAILED','INTERRUPTED','CANCELLED') ORDER BY r.sequence""", (owner_session_id,))]

    def ack_completion(self, run_id, owner_session_id):
        def ack(conn):
            run = self._run(conn, run_id, owner_session_id)
            if run["status"] not in _TERMINAL:
                raise ValueError("A running task has no completion to acknowledge")
            now = time.time()
            conn.execute("UPDATE orchestration_runs SET completion_ack=1 WHERE run_id=?", (run_id,))
            conn.execute("""UPDATE orchestration_parent_messages SET status='ACKNOWLEDGED',acknowledged_at=?
                WHERE run_id=? AND status='PUBLISHED'""", (now, run_id))
        self.db._execute_write(ack)

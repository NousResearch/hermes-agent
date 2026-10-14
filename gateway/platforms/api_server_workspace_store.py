"""Durable original-key reservations for workspace side effects."""
import hashlib
import json
import sqlite3
import threading
from functools import wraps
import uuid
from pathlib import Path

from gateway.platforms.api_server_workspace_models import OperationReceipt


class WorkspaceConflict(ValueError):
    pass


class WorkspaceKeyExpired(ValueError):
    pass


def serialized(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapped


class WorkspaceStore:
    """Bodies and environment values never enter the ledger; output is explicitly retained."""

    MAX_UNACKNOWLEDGED = 4096
    OPERATIONS_PER_EPOCH = 256

    def __init__(self, path: Path):
        import fcntl

        self._lock = threading.RLock()
        self._lease = path.with_suffix(".lock").open("a+b")
        path.with_suffix(".lock").chmod(0o600)
        fcntl.flock(self._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        path.chmod(0o600)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS operations (owner TEXT, key TEXT, fingerprint TEXT, receipt TEXT, "
            "epoch TEXT NOT NULL, stdout BLOB NOT NULL DEFAULT X'', stderr BLOB NOT NULL DEFAULT X'', PRIMARY KEY(owner,key))")
        self.connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS operation_ids ON operations(json_extract(receipt,'$.operation_id'))")
        self.connection.execute("CREATE TABLE IF NOT EXISTS admission (singleton INTEGER PRIMARY KEY CHECK(singleton=1), epoch TEXT, count INTEGER)")
        self.connection.execute("INSERT OR IGNORE INTO admission VALUES (1,?,0)", (uuid.uuid4().hex,))
        self.connection.commit()
        for owner, raw in self.connection.execute("SELECT owner,receipt FROM operations").fetchall():
            receipt = OperationReceipt.model_validate_json(raw)
            if receipt.status in {"reserved", "running", "stopping"}:
                receipt.status = "interrupted"
                receipt.reason = "gateway_restarted"
                receipt.termination_confirmed = receipt.kind != "process"
                self.save(owner, receipt)

    @staticmethod
    def key(value: str) -> str:
        if not value or len(value) > 256:
            raise ValueError("Idempotency-Key must contain 1–256 characters")
        return hashlib.sha256(value.encode()).hexdigest()

    @property
    @serialized
    def admission_epoch(self):
        epoch, count = self.connection.execute("SELECT epoch,count FROM admission WHERE singleton=1").fetchone()
        if count >= self.OPERATIONS_PER_EPOCH:
            epoch = uuid.uuid4().hex
            with self.connection:
                self.connection.execute("UPDATE admission SET epoch=?,count=0 WHERE singleton=1", (epoch,))
                self.connection.execute("DELETE FROM operations WHERE json_extract(receipt,'$.acknowledged')=1")
        return epoch

    @serialized
    def acknowledge(self, owner: str, key: str, operation_id: str):
        record = self.by_key(owner, key)
        if record is None:
            if key.partition(":")[0] != self.admission_epoch:
                raise WorkspaceKeyExpired("Acknowledged receipt was compacted or the admission epoch retired")
            raise WorkspaceConflict("Unknown original operation key")
        _, receipt = record
        if receipt.operation_id != operation_id:
            raise WorkspaceConflict("Original key does not identify this operation")
        if receipt.status not in {"completed", "failed", "cancelled", "interrupted"} or not receipt.termination_confirmed:
            raise WorkspaceConflict("Operation has not confirmed termination")
        receipt.acknowledged, receipt.output_available = True, False
        with self.connection:
            self.connection.execute(
                "UPDATE operations SET receipt=?,stdout=X'',stderr=X'' WHERE owner=? AND key=?",
                (receipt.model_dump_json(), owner, self.key(key)))
            self.connection.execute(
                "DELETE FROM operations WHERE epoch!=? AND json_extract(receipt,'$.acknowledged')=1",
                (self.admission_epoch,))
        return receipt

    @serialized
    def by_key(self, owner: str, key: str):
        row = self.connection.execute(
            "SELECT fingerprint,receipt FROM operations WHERE owner=? AND key=?", (owner, self.key(key))).fetchone()
        return None if row is None else (row[0], OperationReceipt.model_validate_json(row[1]))

    @serialized
    def get(self, owner: str, operation_id: str) -> OperationReceipt | None:
        row = self.connection.execute(
            "SELECT receipt FROM operations WHERE owner=? AND json_extract(receipt,'$.operation_id')=?",
            (owner, operation_id)).fetchone()
        return None if row is None else OperationReceipt.model_validate_json(row[0])

    @serialized
    def reserve(self, owner: str, key: str, kind: str, body, *, cancel=False):
        fingerprint = "" if cancel else hashlib.sha256(
            json.dumps([kind, body.model_dump()], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            existing = self.by_key(owner, key)
            if existing is not None:
                previous, receipt = existing
                if receipt.kind != kind or (not cancel and previous and previous != fingerprint):
                    raise WorkspaceConflict("Idempotency-Key belongs to a different request")
                return receipt, False
            epoch = self.admission_epoch
            prefix, separator, suffix = key.partition(":")
            if not separator or not suffix or len(prefix) != 32 or any(c not in "0123456789abcdef" for c in prefix):
                raise ValueError("Idempotency-Key must be <admission_epoch>:<opaque suffix>")
            if prefix != epoch:
                raise WorkspaceKeyExpired("Unknown key from a retired admission epoch")
            count = self.connection.execute(
                "SELECT COUNT(*) FROM operations WHERE json_extract(receipt,'$.acknowledged')=0").fetchone()[0]
            if count >= self.MAX_UNACKNOWLEDGED:
                raise WorkspaceConflict("Unacknowledged workspace operation capacity reached")
            receipt = OperationReceipt(
                operation_id=f"wop_{uuid.uuid4().hex}", session_id=body.session_id,
                kind=kind, status="cancelled" if cancel else "reserved",
                reason="cancelled_before_admission" if cancel else "",
                termination_confirmed=cancel or kind != "process")
            self.connection.execute("INSERT INTO operations(owner,key,fingerprint,receipt,epoch) VALUES (?,?,?,?,?)", (
                owner, self.key(key), fingerprint, receipt.model_dump_json(), epoch))
            self.connection.execute("UPDATE admission SET count=count+1 WHERE singleton=1")
        return receipt, True

    @serialized
    def save(self, owner: str, receipt: OperationReceipt):
        with self.connection:
            self.connection.execute(
                "UPDATE operations SET receipt=? WHERE owner=? AND json_extract(receipt,'$.operation_id')=?",
                (receipt.model_dump_json(), owner, receipt.operation_id))

    @serialized
    def capture(self, owner: str, receipt: OperationReceipt, capture):
        with capture.lock:
            stdout, stderr = (bytes(capture.streams[name]) for name in ("stdout", "stderr"))
            receipt.stdout_truncated = capture.truncated["stdout"]
            receipt.stderr_truncated = capture.truncated["stderr"]
        receipt.stdout_bytes, receipt.stderr_bytes = len(stdout), len(stderr)
        receipt.output_available = True
        with self.connection:
            self.connection.execute(
                "UPDATE operations SET receipt=?,stdout=?,stderr=? WHERE owner=? "
                "AND json_extract(receipt,'$.operation_id')=?",
                (receipt.model_dump_json(), stdout, stderr, owner, receipt.operation_id))

    @serialized
    def output(self, owner: str, operation_id: str):
        row = self.connection.execute(
            "SELECT stdout,stderr FROM operations WHERE owner=? AND json_extract(receipt,'$.operation_id')=?",
            (owner, operation_id)).fetchone()
        return {"stdout": row[0], "stderr": row[1]}

    @serialized
    def unresolved(self):
        rows = self.connection.execute(
            "SELECT owner,receipt FROM operations WHERE json_extract(receipt,'$.kind')='process' "
            "AND json_extract(receipt,'$.status')='interrupted' "
            "AND json_extract(receipt,'$.termination_confirmed')=0").fetchall()
        return [(owner, OperationReceipt.model_validate_json(raw)) for owner, raw in rows]

    @serialized
    def close(self):
        self.connection.close()
        self._lease.close()

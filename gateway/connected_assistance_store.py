"""Profile-local inbox for grounded connected-source assistance.

No connector clients or delivery side effects live here. Callers own source-access
revalidation and scan serialization; persisted source text remains untrusted data.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from hermes_constants import assert_named_profile_home_live
from hermes_dir_acl import make_user_dir

MAX_EVIDENCE_RECORDS = 200
MAX_EVIDENCE_BYTES = 16_384


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _digest(value) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _text(record: dict, key: str, limit=4096) -> None:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"invalid finding/evidence {key}")


def _validate_evidence(record: dict) -> None:
    if not isinstance(record, dict):
        raise ValueError("invalid evidence record")
    for key in ("source_id", "record_id"):
        _text(record, key, limit=512)
    if len(_json(record).encode("utf-8")) > MAX_EVIDENCE_BYTES:
        raise ValueError("evidence record exceeds retained size limit")


def _validate_finding(finding: dict) -> None:
    if not isinstance(finding, dict):
        raise ValueError("invalid finding")
    for key in ("kind", "entity_key", "title", "summary", "why_now", "suggested_action"):
        _text(finding, key)
    evidence = finding.get("evidence")
    if not isinstance(evidence, list) or not 1 <= len(evidence) <= 20:
        raise ValueError("finding must contain 1 to 20 evidence records")
    for record in evidence:
        _validate_evidence(record)
        _text(record, "excerpt")
    if len(_json(finding).encode("utf-8")) > 65_536:
        raise ValueError("finding exceeds retained size limit")


class AssistanceStore:
    """A store bound to one explicit profile home, never the launch profile."""

    def __init__(self, home: Path):
        self.home = Path(home).resolve()
        self._require_home()
        directory = self.home / "connected_assistance"
        if directory.is_symlink():
            raise OSError("linked assistance directory refused")
        make_user_dir(directory, mode=0o700, exist_ok=True)
        if os.name == "posix":
            directory.chmod(0o700)
        self.path = directory / "state.sqlite3"
        with closing(self._connect()) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS findings "
                       "(id TEXT PRIMARY KEY, identity TEXT UNIQUE NOT NULL, "
                       "status TEXT NOT NULL, payload TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS fingerprints "
                       "(finding_id TEXT NOT NULL, fingerprint TEXT NOT NULL, "
                       "PRIMARY KEY(finding_id, fingerprint))")
            db.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS evidence "
                       "(sequence INTEGER PRIMARY KEY AUTOINCREMENT, identity TEXT UNIQUE NOT NULL, "
                       "payload TEXT NOT NULL)")

    def _require_home(self):
        assert_named_profile_home_live(self.home)
        if not self.home.is_dir():
            raise FileNotFoundError("assistance profile home does not exist")

    def _connect(self):
        self._require_home()
        if self.path.parent.is_symlink():
            raise OSError("linked assistance directory refused")
        for suffix in ("", "-journal", "-wal", "-shm"):
            candidate = self.path.with_name(self.path.name + suffix)
            if candidate.is_symlink():
                raise OSError("linked assistance file refused")
            if candidate.exists() and (not candidate.is_file() or candidate.stat().st_nlink != 1):
                raise OSError("non-private assistance file refused")
        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise OSError("non-private assistance file refused")
            if os.name == "posix":
                os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        return db

    @staticmethod
    def _finding(row):
        return json.loads(row["payload"]) | {"id": row["id"], "status": row["status"]}

    def is_paused(self) -> bool:
        with closing(self._connect()) as db:
            row = db.execute("SELECT value FROM settings WHERE key = 'paused'").fetchone()
        return bool(row and row["value"] == "true")

    def set_paused(self, paused: bool) -> None:
        if not isinstance(paused, bool):
            raise ValueError("paused must be a boolean")
        with closing(self._connect()) as db, db:
            db.execute("INSERT INTO settings VALUES ('paused', ?) ON CONFLICT(key) "
                       "DO UPDATE SET value = excluded.value", ("true" if paused else "false",))

    def ingest_scan(self, result: dict, *, check_current=None) -> list:
        if not isinstance(result, dict):
            raise ValueError("invalid scan result")
        if result.get("connected") is not True:
            return []
        findings, evidence = result.get("findings", []), result.get("evidence", [])
        if not isinstance(findings, list) or len(findings) > 50:
            raise ValueError("scan must contain at most 50 findings")
        if not isinstance(evidence, list) or len(evidence) > MAX_EVIDENCE_RECORDS:
            raise ValueError("scan evidence exceeds record limit")
        for record in evidence:
            _validate_evidence(record)
        for finding in findings:
            _validate_finding(finding)
        changed = []
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            if check_current is not None:
                check_current()
            for record in evidence:
                self._retain_evidence(db, record)
            db.execute("DELETE FROM evidence WHERE sequence NOT IN "
                       "(SELECT sequence FROM evidence ORDER BY sequence DESC LIMIT ?)",
                       (MAX_EVIDENCE_RECORDS,))
            for finding in findings:
                row = self._ingest_finding(db, finding)
                if row is not None:
                    changed.append(row)
            if check_current is not None:
                check_current()
        return changed

    @staticmethod
    def _retain_evidence(db, record):
        payload = _json(record)
        if len(payload.encode("utf-8")) > MAX_EVIDENCE_BYTES:
            raise ValueError("evidence record exceeds retained size limit")
        identity = _digest([record["source_id"], record["record_id"], record.get("capability_id"),
                            record.get("json_pointer", record.get("pointer"))])
        db.execute("DELETE FROM evidence WHERE identity = ?", (identity,))
        db.execute("INSERT INTO evidence (identity, payload) VALUES (?, ?)", (identity, payload))

    def prior_evidence(self) -> list:
        """Newest observed records first, preserving complete data-use constraints.

        This cache is not proof of continued access. The runtime must revalidate
        sources before passing any retained record to a model or user surface.
        """
        with closing(self._connect()) as db:
            rows = db.execute("SELECT payload FROM evidence ORDER BY sequence DESC LIMIT ?",
                              (MAX_EVIDENCE_RECORDS,)).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def _ingest_finding(self, db, finding):
        sources = sorted({item["source_id"] for item in finding["evidence"]})
        identity = _digest([finding["kind"], finding["entity_key"], sources])
        fingerprint = _digest(sorted({_json(item) for item in finding["evidence"]}))
        existing = db.execute("SELECT * FROM findings WHERE identity = ?", (identity,)).fetchone()
        identifier = existing["id"] if existing else uuid4().hex
        if db.execute("SELECT 1 FROM fingerprints p JOIN findings f ON f.id = p.finding_id "
                      "WHERE p.fingerprint = ? AND json_extract(f.payload, '$.kind') = ?",
                      (fingerprint, finding["kind"])).fetchone():
            return None
        status = existing["status"] if existing else "open"
        db.execute("INSERT INTO findings VALUES (?, ?, ?, ?) ON CONFLICT(identity) "
                   "DO UPDATE SET payload = excluded.payload", (identifier, identity, status, _json(finding)))
        db.execute("INSERT INTO fingerprints VALUES (?, ?)", (identifier, fingerprint))
        return finding | {"id": identifier, "status": status} if status == "open" else None

    def list_findings(self, status=None, limit=20) -> list:
        if status not in {None, "open", "dismissed", "resolved"}:
            raise ValueError("invalid finding status")
        if type(limit) is not int or limit < 0:
            raise ValueError("limit must be a nonnegative integer")
        with closing(self._connect()) as db:
            rows = db.execute("SELECT * FROM findings WHERE (? IS NULL OR status = ?) "
                              "ORDER BY rowid DESC LIMIT ?", (status, status, min(limit, 100))).fetchall()
        return [self._finding(row) for row in rows]

    def get_finding(self, identifier) -> dict | None:
        with closing(self._connect()) as db:
            row = db.execute("SELECT * FROM findings WHERE id = ?", (identifier,)).fetchone()
        return self._finding(row) if row else None

    def transition(self, identifier, status: str) -> dict | None:
        """Terminal decisions are sticky across scans; only the user changes them."""
        if status not in {"dismissed", "resolved"}:
            raise ValueError("status must be dismissed or resolved")
        with closing(self._connect()) as db, db:
            db.execute("UPDATE findings SET status = ? WHERE id = ?", (status, identifier))
            row = db.execute("SELECT * FROM findings WHERE id = ?", (identifier,)).fetchone()
        return self._finding(row) if row else None

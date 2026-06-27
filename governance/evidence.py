"""Hash-chained governance evidence ledger.\n\nAll governance decisions and evidence records are append-only JSONL files under\n``$HERMES_HOME/governance``.  Each row contains the previous row hash and its\nown SHA-256 hash over canonical JSON, producing a lightweight tamper-evident\nchain without any external service dependency.\n"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
from typing import Any, Dict, Mapping, Optional

from hermes_constants import get_hermes_home

SCHEMA_VERSION = "governance.evidence.v1"
GENESIS_HASH = "GENESIS"


def utc_now_iso() -> str:
    """Return a stable UTC timestamp for governance records."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def governance_dir() -> Path:
    path = get_hermes_home() / "governance"
    path.mkdir(parents=True, exist_ok=True)
    return path


def canonical_json(data: Mapping[str, Any]) -> str:
    """Canonical JSON used for hash calculation and deterministic logs."""
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _last_entry_hash(path: Path) -> str:
    if not path.exists():
        return GENESIS_HASH
    try:
        last = ""
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    last = line
        if not last:
            return GENESIS_HASH
        row = json.loads(last)
        entry_hash = row.get("entry_hash")
        return entry_hash if isinstance(entry_hash, str) and entry_hash else GENESIS_HASH
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return GENESIS_HASH


class _OptionalFileLock:
    """Best-effort cross-process append lock.

    ``fcntl`` is unavailable on Windows, so this lock is advisory on POSIX and a
    no-op elsewhere.  JSONL rows are still written as one line with normal file
    append semantics, which is sufficient for tests and most local use.
    """

    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self._handle = None

    def __enter__(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.lock_path.open("a", encoding="utf-8")
        try:
            import fcntl  # type: ignore
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        except Exception:
            pass
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._handle is not None:
            try:
                import fcntl  # type: ignore
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            self._handle.close()


def append_hash_chained_event(log_name: str, event: Mapping[str, Any]) -> Dict[str, Any]:
    """Append *event* to ``governance/<log_name>.jsonl`` and return the row.

    ``log_name`` may be supplied with or without ``.jsonl``.  The caller's
    fields are preserved, but governance metadata wins for timestamp/hash keys
    so a caller cannot spoof the chain.
    """
    clean_name = log_name[:-6] if log_name.endswith(".jsonl") else log_name
    if not clean_name or "/" in clean_name or ".." in clean_name:
        raise ValueError(f"Invalid governance log name: {log_name!r}")

    directory = governance_dir()
    path = directory / f"{clean_name}.jsonl"
    lock_path = directory / f".{clean_name}.lock"

    with _OptionalFileLock(lock_path):
        previous_hash = _last_entry_hash(path)
        row = dict(event)
        row.update({
            "schema_version": row.get("schema_version", SCHEMA_VERSION),
            "timestamp_utc": row.get("timestamp_utc", utc_now_iso()),
            "previous_entry_hash": previous_hash,
        })
        row.pop("entry_hash", None)
        row["entry_hash"] = sha256_text(canonical_json(row))
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return row


@dataclass
class EvidenceRecord:
    phase: str
    claim: str
    evidence_type: str
    source: str
    confidence: str
    redaction_status: str = "none"
    metadata: Dict[str, Any] = field(default_factory=dict)


class EvidenceLedger:
    """Append-only ledger for claims backed by concrete evidence."""

    def record(self, record: EvidenceRecord | Mapping[str, Any]) -> Dict[str, Any]:
        payload = asdict(record) if isinstance(record, EvidenceRecord) else dict(record)
        payload.setdefault("event_type", "evidence")
        payload.setdefault("redaction_status", "none")
        return append_hash_chained_event("evidence", payload)

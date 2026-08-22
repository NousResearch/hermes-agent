"""Profile-scoped durable outbox for external memory-provider mirrors."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict


class MemoryWriteOutbox:
    """Persist failed provider writes and replay them in FIFO order."""

    def __init__(
        self,
        hermes_home: str | Path,
        *,
        max_entries_per_provider: int = 1000,
        claim_lease_seconds: float = 300.0,
    ) -> None:
        self._path = Path(hermes_home) / "memories" / "provider_write_outbox.sqlite3"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._max_entries = max(1, int(max_entries_per_provider))
        self._claim_lease_seconds = max(1.0, float(claim_lease_seconds))
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_writes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    action TEXT NOT NULL,
                    target TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT NOT NULL DEFAULT '',
                    claim_token TEXT NOT NULL DEFAULT '',
                    claim_until REAL NOT NULL DEFAULT 0,
                    UNIQUE(provider, fingerprint)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS provider_alerts (
                    provider TEXT PRIMARY KEY,
                    last_alert_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pending_provider_fifo "
                "ON pending_writes(provider, id)"
            )
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(pending_writes)")
            }
            if "claim_token" not in columns:
                conn.execute(
                    "ALTER TABLE pending_writes "
                    "ADD COLUMN claim_token TEXT NOT NULL DEFAULT ''"
                )
            if "claim_until" not in columns:
                conn.execute(
                    "ALTER TABLE pending_writes "
                    "ADD COLUMN claim_until REAL NOT NULL DEFAULT 0"
                )

    @staticmethod
    def _fingerprint(
        action: str,
        target: str,
        content: str,
        metadata: Dict[str, Any],
    ) -> str:
        # Provenance (session/tool-call ids) changes when the agent retries the
        # same intent. Only old_text affects replace/remove write semantics.
        payload = json.dumps(
            [action, target, content, metadata.get("old_text", "")],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def enqueue(
        self,
        provider: str,
        action: str,
        target: str,
        content: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Insert one write intent, deduplicate it, and enforce the bound."""
        metadata = dict(metadata or {})
        metadata_json = json.dumps(metadata, sort_keys=True, ensure_ascii=False, default=str)
        fingerprint = self._fingerprint(action, target, content, metadata)
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO pending_writes
                    (provider, fingerprint, action, target, content, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (provider, fingerprint, action, target, content, metadata_json, time.time()),
            )
            inserted = cursor.rowcount == 1
            before = conn.execute(
                "SELECT COUNT(*) FROM pending_writes WHERE provider = ?", (provider,)
            ).fetchone()[0]
            overflow = max(0, before - self._max_entries)
            dropped = 0
            if overflow:
                # A leased row may still be inside a provider callback. Never
                # evict it: doing so makes a later callback failure impossible
                # to release/replay. Prefer the oldest unclaimed rows, which
                # also causes a newly inserted row to be rejected when every
                # older row is in flight.
                eligible = conn.execute(
                    """
                    SELECT id FROM pending_writes
                    WHERE provider = ? AND claim_token = ''
                    ORDER BY id LIMIT ?
                    """,
                    (provider, overflow),
                ).fetchall()
                if eligible:
                    placeholders = ",".join("?" for _ in eligible)
                    dropped = conn.execute(
                        f"DELETE FROM pending_writes WHERE id IN ({placeholders})",
                        tuple(row["id"] for row in eligible),
                    ).rowcount
            after = conn.execute(
                "SELECT COUNT(*) FROM pending_writes WHERE provider = ?", (provider,)
            ).fetchone()[0]
            queued = conn.execute(
                "SELECT 1 FROM pending_writes WHERE provider = ? AND fingerprint = ?",
                (provider, fingerprint),
            ).fetchone() is not None
        return {
            "queued": queued,
            "deduplicated": not inserted,
            "dropped": dropped,
            "overflow": max(0, after - self._max_entries),
        }

    def pending_count(self, provider: str) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM pending_writes WHERE provider = ?", (provider,)
            ).fetchone()
        return int(row[0])

    def replay(
        self,
        provider: str,
        deliver: Callable[[str, str, str, Dict[str, Any]], None],
    ) -> Dict[str, Any]:
        """Replay queued writes until the provider fails, preserving FIFO order."""
        replayed = 0
        with self._lock:
            while True:
                claim_token = uuid.uuid4().hex
                now = time.time()
                blocked = False
                with self._connect() as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    row = conn.execute(
                        """
                        SELECT id, action, target, content, metadata_json,
                               claim_token, claim_until
                        FROM pending_writes WHERE provider = ? ORDER BY id LIMIT 1
                        """,
                        (provider,),
                    ).fetchone()
                    if row is not None:
                        blocked = bool(
                            row["claim_token"] and float(row["claim_until"]) > now
                        )
                        if not blocked:
                            claimed = conn.execute(
                                """
                                UPDATE pending_writes
                                SET claim_token = ?, claim_until = ?
                                WHERE id = ? AND (claim_token = '' OR claim_until <= ?)
                                """,
                                (
                                    claim_token,
                                    now + self._claim_lease_seconds,
                                    row["id"],
                                    now,
                                ),
                            ).rowcount
                            blocked = claimed != 1
                if row is None:
                    self.clear_alert(provider)
                    return {
                        "replayed": replayed,
                        "remaining": 0,
                        "error": "",
                        "blocked": False,
                    }
                if blocked:
                    return {
                        "replayed": replayed,
                        "remaining": self.pending_count(provider),
                        "error": "",
                        "blocked": True,
                    }
                stop_renewal = threading.Event()
                ownership_lost = threading.Event()
                renewal_interval = min(5.0, max(0.05, self._claim_lease_seconds / 3.0))

                def _renew_claim() -> None:
                    while not stop_renewal.wait(renewal_interval):
                        try:
                            with self._connect() as renewal_conn:
                                renewed = renewal_conn.execute(
                                    """
                                    UPDATE pending_writes SET claim_until = ?
                                    WHERE id = ? AND claim_token = ?
                                    """,
                                    (
                                        time.time() + self._claim_lease_seconds,
                                        row["id"],
                                        claim_token,
                                    ),
                                ).rowcount
                            if renewed != 1:
                                ownership_lost.set()
                                return
                        except sqlite3.Error:
                            # A transient SQLite writer can consume one renewal
                            # interval. Keep trying; the guarded final mutation
                            # below is the authority on ownership.
                            continue

                renewer = threading.Thread(
                    target=_renew_claim,
                    daemon=True,
                    name=f"memory-outbox-lease-{provider}",
                )
                renewer.start()
                try:
                    metadata = json.loads(row["metadata_json"])
                    deliver(row["action"], row["target"], row["content"], metadata)
                except Exception as exc:
                    stop_renewal.set()
                    renewer.join()
                    error = str(exc)
                    with self._connect() as conn:
                        released = conn.execute(
                            """
                            UPDATE pending_writes
                            SET attempts = attempts + 1, last_error = ?,
                                claim_token = '', claim_until = 0
                            WHERE id = ? AND claim_token = ?
                            """,
                            (error, row["id"], claim_token),
                        ).rowcount
                    return {
                        "replayed": replayed,
                        "remaining": self.pending_count(provider),
                        "error": error,
                        "blocked": released != 1 or ownership_lost.is_set(),
                    }
                stop_renewal.set()
                renewer.join()
                with self._connect() as conn:
                    deleted = conn.execute(
                        "DELETE FROM pending_writes WHERE id = ? AND claim_token = ?",
                        (row["id"], claim_token),
                    ).rowcount
                if deleted != 1 or ownership_lost.is_set():
                    return {
                        "replayed": replayed,
                        "remaining": self.pending_count(provider),
                        "error": "provider write claim ownership was lost during delivery",
                        "blocked": True,
                    }
                replayed += 1

    def should_alert(self, provider: str, cooldown_seconds: float) -> bool:
        """Persistently rate-limit alerts across short-lived agent instances."""
        now = time.time()
        cooldown = max(0.0, float(cooldown_seconds))
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT last_alert_at FROM provider_alerts WHERE provider = ?", (provider,)
            ).fetchone()
            if row is not None and now - float(row[0]) < cooldown:
                return False
            conn.execute(
                """
                INSERT INTO provider_alerts(provider, last_alert_at) VALUES (?, ?)
                ON CONFLICT(provider) DO UPDATE SET last_alert_at = excluded.last_alert_at
                """,
                (provider, now),
            )
        return True

    def clear_alert(self, provider: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM provider_alerts WHERE provider = ?", (provider,))

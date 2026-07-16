from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from .models import CampaignPlan, CampaignRecord, WorkerRecord, WorkerState


class ConductorStore:
    """Durable campaign state with SQLite-serialized tick ownership."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, timeout=5)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS campaigns (
              campaign_id TEXT PRIMARY KEY, plan_json TEXT NOT NULL,
              state TEXT NOT NULL, step_index INTEGER NOT NULL DEFAULT 0,
              conductor_turns INTEGER NOT NULL DEFAULT 0,
              retries INTEGER NOT NULL DEFAULT 0,
              next_retry_at REAL NOT NULL DEFAULT 0,
              started_at REAL NOT NULL, blocker_key TEXT
            );
            CREATE TABLE IF NOT EXISTS workers (
              worker_id TEXT PRIMARY KEY, campaign_id TEXT NOT NULL,
              step_index INTEGER NOT NULL, role TEXT NOT NULL, cwd TEXT NOT NULL,
              tmux_session TEXT NOT NULL UNIQUE, pid INTEGER, start_marker TEXT,
              provider TEXT NOT NULL, model TEXT NOT NULL, prompt_hash TEXT NOT NULL,
              manifest_json TEXT NOT NULL, launched_at REAL NOT NULL,
              heartbeat_at REAL, progress_evidence TEXT, state TEXT NOT NULL,
              output_path TEXT NOT NULL, receipt_path TEXT NOT NULL,
              receipt_hash TEXT, nonce TEXT NOT NULL, read_only INTEGER NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS one_active_worker
              ON workers(campaign_id) WHERE state = 'RUNNING';
            CREATE TABLE IF NOT EXISTS daily_usage (
              campaign_id TEXT NOT NULL, day TEXT NOT NULL,
              input_tokens INTEGER NOT NULL DEFAULT 0,
              output_tokens INTEGER NOT NULL DEFAULT 0,
              reasoning_tokens INTEGER NOT NULL DEFAULT 0,
              cache_read_tokens INTEGER NOT NULL DEFAULT 0,
              cache_write_tokens INTEGER NOT NULL DEFAULT 0,
              processed_tokens INTEGER NOT NULL DEFAULT 0,
              runs INTEGER NOT NULL DEFAULT 0,
              PRIMARY KEY(campaign_id, day)
            );
            CREATE TABLE IF NOT EXISTS tick_owners (
              campaign_id TEXT PRIMARY KEY, owner TEXT NOT NULL, acquired_at REAL NOT NULL
            );
            """
        )
        self._db.commit()

    def create_campaign(self, plan: CampaignPlan, now: float | None = None) -> None:
        with self._lock:
            serialized = json.dumps(plan.to_dict(), sort_keys=True)
            inserted = self._db.execute(
                "INSERT OR IGNORE INTO campaigns(campaign_id,plan_json,state,started_at) VALUES(?,?,?,?)",
                (
                    plan.campaign_id,
                    serialized,
                    "READY",
                    now if now is not None else time.time(),
                ),
            ).rowcount
            if not inserted:
                existing = self._db.execute(
                    "SELECT plan_json FROM campaigns WHERE campaign_id=?",
                    (plan.campaign_id,),
                ).fetchone()
                if existing is None or existing["plan_json"] != serialized:
                    self._db.rollback()
                    raise ValueError(
                        "campaign_id already exists with a different definition"
                    )
            self._db.commit()

    def get_campaign(self, campaign_id: str) -> CampaignRecord:
        row = self._db.execute(
            "SELECT * FROM campaigns WHERE campaign_id=?", (campaign_id,)
        ).fetchone()
        if row is None:
            raise KeyError(campaign_id)
        return CampaignRecord(
            campaign_id=row["campaign_id"],
            plan=CampaignPlan.from_dict(json.loads(row["plan_json"])),
            state=row["state"],
            step_index=row["step_index"],
            conductor_turns=row["conductor_turns"],
            retries=row["retries"],
            next_retry_at=row["next_retry_at"],
            started_at=row["started_at"],
            blocker_key=row["blocker_key"],
        )

    def update_campaign(self, campaign_id: str, **fields) -> None:
        if not fields:
            return
        allowed = {
            "state",
            "step_index",
            "conductor_turns",
            "retries",
            "next_retry_at",
            "blocker_key",
        }
        if not set(fields) <= allowed:
            raise ValueError("invalid campaign field")
        sql = ",".join(f"{key}=?" for key in fields)
        with self._lock:
            self._db.execute(
                f"UPDATE campaigns SET {sql} WHERE campaign_id=?",
                (*fields.values(), campaign_id),
            )
            self._db.commit()

    def acquire_tick(
        self,
        campaign_id: str,
        owner: str,
        *,
        lease_seconds: float,
        now: float | None = None,
    ) -> bool:
        if lease_seconds <= 0:
            raise ValueError("tick lease must be positive")
        acquired_at = time.time() if now is None else float(now)
        with self._lock:
            try:
                self._db.execute("BEGIN IMMEDIATE")
                changed = self._db.execute(
                    """INSERT INTO tick_owners(campaign_id,owner,acquired_at) VALUES(?,?,?)
                       ON CONFLICT(campaign_id) DO UPDATE SET
                         owner=excluded.owner, acquired_at=excluded.acquired_at
                       WHERE tick_owners.acquired_at <= ?""",
                    (
                        campaign_id,
                        owner,
                        acquired_at,
                        acquired_at - float(lease_seconds),
                    ),
                )
                self._db.commit()
                return changed.rowcount == 1
            except sqlite3.IntegrityError:
                self._db.rollback()
                return False

    def release_tick(self, campaign_id: str, owner: str) -> None:
        with self._lock:
            self._db.execute(
                "DELETE FROM tick_owners WHERE campaign_id=? AND owner=?",
                (campaign_id, owner),
            )
            self._db.commit()

    def insert_worker(self, values: dict) -> None:
        columns = ",".join(values)
        placeholders = ",".join("?" for _ in values)
        with self._lock:
            self._db.execute(
                f"INSERT INTO workers({columns}) VALUES({placeholders})",
                tuple(values.values()),
            )
            self._db.commit()

    def _worker(self, row) -> WorkerRecord | None:
        if row is None:
            return None
        return WorkerRecord(
            worker_id=row["worker_id"],
            campaign_id=row["campaign_id"],
            step_index=row["step_index"],
            role=row["role"],
            cwd=row["cwd"],
            tmux_session=row["tmux_session"],
            pid=row["pid"],
            start_marker=row["start_marker"],
            provider=row["provider"],
            model=row["model"],
            prompt_hash=row["prompt_hash"],
            mutable_manifest=json.loads(row["manifest_json"]),
            launched_at=row["launched_at"],
            heartbeat_at=row["heartbeat_at"],
            progress_evidence=row["progress_evidence"],
            state=WorkerState(row["state"]),
            output_path=row["output_path"],
            receipt_path=row["receipt_path"],
            receipt_hash=row["receipt_hash"],
            nonce=row["nonce"],
            read_only=bool(row["read_only"]),
        )

    def active_worker(self, campaign_id: str) -> WorkerRecord | None:
        row = self._db.execute(
            "SELECT * FROM workers WHERE campaign_id=? ORDER BY launched_at DESC LIMIT 1",
            (campaign_id,),
        ).fetchone()
        return self._worker(row)

    def update_worker(self, worker_id: str, **fields) -> None:
        allowed = {
            "pid",
            "start_marker",
            "heartbeat_at",
            "progress_evidence",
            "state",
            "receipt_hash",
        }
        if not set(fields) <= allowed:
            raise ValueError("invalid worker field")
        sql = ",".join(f"{key}=?" for key in fields)
        values = [
            value.value if isinstance(value, WorkerState) else value
            for value in fields.values()
        ]
        with self._lock:
            self._db.execute(
                f"UPDATE workers SET {sql} WHERE worker_id=?", (*values, worker_id)
            )
            self._db.commit()

    def add_usage(self, campaign_id: str, usage: dict, day: str) -> None:
        fields = (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
        )
        values = [max(0, int(usage.get(field, 0) or 0)) for field in fields]
        processed = sum(values[:4])
        with self._lock:
            self._db.execute(
                """INSERT INTO daily_usage(campaign_id,day,input_tokens,output_tokens,reasoning_tokens,
                   cache_read_tokens,cache_write_tokens,processed_tokens,runs) VALUES(?,?,?,?,?,?,?,?,1)
                   ON CONFLICT(campaign_id,day) DO UPDATE SET
                   input_tokens=input_tokens+excluded.input_tokens,
                   output_tokens=output_tokens+excluded.output_tokens,
                   reasoning_tokens=reasoning_tokens+excluded.reasoning_tokens,
                   cache_read_tokens=cache_read_tokens+excluded.cache_read_tokens,
                   cache_write_tokens=cache_write_tokens+excluded.cache_write_tokens,
                   processed_tokens=processed_tokens+excluded.processed_tokens,runs=runs+1""",
                (campaign_id, day, *values, processed),
            )
            self._db.commit()

    def daily_usage(self, campaign_id: str, day: str | None = None) -> dict:
        day = day or time.strftime("%Y-%m-%d", time.gmtime())
        row = self._db.execute(
            "SELECT * FROM daily_usage WHERE campaign_id=? AND day=?",
            (campaign_id, day),
        ).fetchone()
        keys = (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "processed_tokens",
            "runs",
        )
        return {key: int(row[key]) if row else 0 for key in keys}

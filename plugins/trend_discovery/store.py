"""SQLite persistence for Trend Discovery Center."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from hermes_constants import get_hermes_home

from .plan import DEFAULT_SCOPES, ISSUES, PHASES


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_store_path() -> Path:
    return get_hermes_home() / "trend-discovery" / "trend_discovery.db"


class TrendDiscoveryStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else default_store_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS phases (
                    phase_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    objective TEXT NOT NULL,
                    duration_days INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    percent_complete INTEGER NOT NULL DEFAULT 0,
                    started_at TEXT,
                    due_at TEXT,
                    completed_at TEXT,
                    evidence TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS issues (
                    issue_id TEXT PRIMARY KEY,
                    phase_id TEXT NOT NULL REFERENCES phases(phase_id),
                    title TEXT NOT NULL,
                    owner_role TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    percent_complete INTEGER NOT NULL DEFAULT 0,
                    remaining_percent INTEGER NOT NULL DEFAULT 100,
                    blocker TEXT NOT NULL DEFAULT '',
                    evidence TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    run_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    source_name TEXT,
                    error TEXT NOT NULL DEFAULT '',
                    evidence TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS reminders (
                    reminder_id TEXT PRIMARY KEY,
                    phase_id TEXT NOT NULL REFERENCES phases(phase_id),
                    due_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    sent_at TEXT,
                    evidence TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS notifications (
                    notification_id TEXT PRIMARY KEY,
                    target TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    sent_at TEXT,
                    error TEXT NOT NULL DEFAULT '',
                    evidence TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS sources (
                    name TEXT PRIMARY KEY,
                    adapter TEXT NOT NULL,
                    url TEXT NOT NULL DEFAULT '',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    priority INTEGER NOT NULL DEFAULT 50,
                    timeout_seconds INTEGER NOT NULL DEFAULT 20,
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    success_count INTEGER NOT NULL DEFAULT 0,
                    circuit_open_until TEXT,
                    last_status TEXT,
                    last_error TEXT NOT NULL DEFAULT '',
                    metadata TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS findings (
                    finding_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL,
                    domain TEXT NOT NULL DEFAULT '',
                    summary TEXT NOT NULL DEFAULT '',
                    source_name TEXT NOT NULL,
                    discovered_at TEXT NOT NULL,
                    relevance_score INTEGER NOT NULL,
                    novelty_score INTEGER NOT NULL,
                    tags TEXT NOT NULL DEFAULT '[]',
                    entity_name TEXT NOT NULL DEFAULT '',
                    provenance TEXT NOT NULL DEFAULT '{}',
                    UNIQUE(url)
                );
                CREATE TABLE IF NOT EXISTS config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            self._seed_plan(conn)
            self._seed_sources(conn)
            self._set_default_config(conn)

    def _seed_plan(self, conn: sqlite3.Connection) -> None:
        start = datetime.now(timezone.utc).replace(microsecond=0)
        offset_days = 0
        for phase in PHASES:
            due = start + timedelta(days=offset_days + phase.duration_days)
            conn.execute(
                """
                INSERT OR IGNORE INTO phases
                    (phase_id, name, objective, duration_days, due_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (phase.phase_id, phase.name, phase.objective, phase.duration_days, due.isoformat()),
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO reminders
                    (reminder_id, phase_id, due_at)
                VALUES (?, ?, ?)
                """,
                (f"reminder-{phase.phase_id}", phase.phase_id, due.isoformat()),
            )
            offset_days += phase.duration_days
        now = utc_now()
        for issue in ISSUES:
            conn.execute(
                """
                INSERT OR IGNORE INTO issues
                    (issue_id, phase_id, title, owner_role, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (issue.issue_id, issue.phase_id, issue.title, issue.owner_role, now),
            )

    def _seed_sources(self, conn: sqlite3.Connection) -> None:
        defaults = (
            ("hn-frontpage", "rss", "https://hnrss.org/frontpage", 10, {"scopes": list(DEFAULT_SCOPES)}),
            ("techcrunch-startups", "rss", "https://techcrunch.com/category/startups/feed/", 20, {"scopes": list(DEFAULT_SCOPES)}),
            ("direct-sample", "webpage", "https://example.com", 90, {"scopes": ["healthcheck"]}),
            ("open-crawl-optional", "open_crawl", "", 80, {"optional": True}),
            ("n8n-optional", "n8n", "", 85, {"optional": True}),
        )
        for name, adapter, url, priority, metadata in defaults:
            conn.execute(
                """
                INSERT OR IGNORE INTO sources
                    (name, adapter, url, priority, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (name, adapter, url, priority, json.dumps(metadata, sort_keys=True)),
            )

    def _set_default_config(self, conn: sqlite3.Connection) -> None:
        defaults = {
            "notification.primary": "local",
            "notification.fallback": "local",
            "knowledge.review_queue": str(get_hermes_home() / "review-queue" / "trend-discovery"),
            "project.slug": "trend-discovery",
            "health.localhost_url": "http://127.0.0.1:9119",
        }
        for key, value in defaults.items():
            conn.execute(
                "INSERT OR IGNORE INTO config (key, value) VALUES (?, ?)",
                (key, value),
            )

    def set_issue_complete(self, issue_id: str, evidence: dict[str, Any] | None = None) -> None:
        self.update_issue(issue_id, 100, evidence=evidence)

    def update_issue(
        self,
        issue_id: str,
        percent: int,
        *,
        status: str | None = None,
        blocker: str = "",
        evidence: dict[str, Any] | None = None,
    ) -> None:
        percent = max(0, min(100, int(percent)))
        status = status or ("complete" if percent == 100 else "in_progress")
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE issues
                SET percent_complete=?, remaining_percent=?, status=?, blocker=?,
                    evidence=?, updated_at=?
                WHERE issue_id=?
                """,
                (
                    percent,
                    100 - percent,
                    status,
                    blocker,
                    json.dumps(evidence or {}, sort_keys=True),
                    utc_now(),
                    issue_id,
                ),
            )
            self._refresh_phase_percent(conn)

    def _refresh_phase_percent(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            "SELECT phase_id, AVG(percent_complete) AS pct FROM issues GROUP BY phase_id"
        ).fetchall()
        for row in rows:
            pct = int(round(row["pct"] or 0))
            status = "complete" if pct == 100 else ("in_progress" if pct else "pending")
            conn.execute(
                "UPDATE phases SET percent_complete=?, status=?, completed_at=CASE WHEN ?=100 THEN COALESCE(completed_at, ?) ELSE completed_at END WHERE phase_id=?",
                (pct, status, pct, utc_now(), row["phase_id"]),
            )

    def compliance_rows(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT issue_id, phase_id, percent_complete, remaining_percent
                FROM issues
                ORDER BY phase_id, issue_id
                """
            ).fetchall()
            phases = conn.execute(
                """
                SELECT phase_id, percent_complete, 100 - percent_complete AS remaining_percent
                FROM phases ORDER BY phase_id
                """
            ).fetchall()
        out = [dict(row) for row in rows]
        out.extend(
            {
                "issue_id": f"{dict(row)['phase_id']}_TOTAL",
                "phase_id": dict(row)["phase_id"],
                "percent_complete": dict(row)["percent_complete"],
                "remaining_percent": dict(row)["remaining_percent"],
            }
            for row in phases
        )
        total = int(round(sum(row["percent_complete"] for row in phases) / max(len(phases), 1)))
        out.append(
            {
                "issue_id": "PROJECT_TOTAL",
                "phase_id": "ALL",
                "percent_complete": total,
                "remaining_percent": 100 - total,
            }
        )
        return out

    def status_snapshot(self) -> dict[str, Any]:
        with self.connect() as conn:
            phases = [dict(r) for r in conn.execute("SELECT * FROM phases ORDER BY phase_id")]
            issues = [dict(r) for r in conn.execute("SELECT * FROM issues ORDER BY phase_id, issue_id")]
            sources = [dict(r) for r in conn.execute("SELECT * FROM sources ORDER BY priority, name")]
            latest_runs = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM runs ORDER BY started_at DESC LIMIT 10"
                )
            ]
            findings_count = conn.execute("SELECT COUNT(*) AS c FROM findings").fetchone()["c"]
        return {
            "store_path": str(self.path),
            "phases": phases,
            "issues": issues,
            "sources": sources,
            "latest_runs": latest_runs,
            "findings_count": findings_count,
        }

    def get_config(self, key: str, default: str = "") -> str:
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM config WHERE key=?", (key,)).fetchone()
        return str(row["value"]) if row else default

    def set_config(self, key: str, value: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO config (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )


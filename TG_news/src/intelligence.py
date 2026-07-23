"""Explainable Telegram News intelligence and editorial state.

This module stores public-news data only.  It has no import from Communication
Core and no publication/send path.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _id(prefix: str, *parts: str) -> str:
    value = "\x1f".join(parts)
    return f"{prefix}_{uuid.uuid5(uuid.NAMESPACE_URL, value).hex}"


_SCHEMA = r"""
CREATE TABLE IF NOT EXISTS news_stories (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'developing'
        CHECK(status IN ('developing', 'stable', 'closed')),
    topics_json TEXT NOT NULL DEFAULT '[]',
    entities_json TEXT NOT NULL DEFAULT '[]',
    geography_json TEXT NOT NULL DEFAULT '[]',
    first_seen_at TEXT NOT NULL,
    last_updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS news_story_articles (
    story_id TEXT NOT NULL REFERENCES news_stories(id) ON DELETE CASCADE,
    article_id TEXT NOT NULL REFERENCES articles(id),
    source_id TEXT NOT NULL REFERENCES sources(id),
    added_at TEXT NOT NULL,
    PRIMARY KEY(story_id, article_id)
);
CREATE TABLE IF NOT EXISTS news_story_updates (
    id TEXT PRIMARY KEY,
    story_id TEXT NOT NULL REFERENCES news_stories(id) ON DELETE CASCADE,
    article_id TEXT REFERENCES articles(id),
    update_type TEXT NOT NULL CHECK(update_type IN ('created','development','correction','confirmation','closure')),
    summary TEXT NOT NULL,
    happened_at TEXT NOT NULL,
    provenance_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS news_claims (
    id TEXT PRIMARY KEY,
    story_id TEXT NOT NULL REFERENCES news_stories(id) ON DELETE CASCADE,
    statement TEXT NOT NULL,
    normalized_statement TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(story_id, normalized_statement)
);
CREATE TABLE IF NOT EXISTS news_claim_evidence (
    claim_id TEXT NOT NULL REFERENCES news_claims(id) ON DELETE CASCADE,
    article_id TEXT NOT NULL REFERENCES articles(id),
    source_id TEXT NOT NULL REFERENCES sources(id),
    stance TEXT NOT NULL CHECK(stance IN ('supports','disputes','mentions')),
    quote TEXT,
    observed_at TEXT NOT NULL,
    PRIMARY KEY(claim_id, article_id)
);
CREATE TABLE IF NOT EXISTS news_source_reliability_history (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(id),
    score REAL NOT NULL CHECK(score BETWEEN 0 AND 1),
    outcome TEXT NOT NULL CHECK(outcome IN ('confirmed','contradicted','mixed')),
    evidence_ref TEXT NOT NULL,
    explanation TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS news_watchlists (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    topics_json TEXT NOT NULL DEFAULT '[]',
    entities_json TEXT NOT NULL DEFAULT '[]',
    geography_json TEXT NOT NULL DEFAULT '[]',
    enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0,1)),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS news_breaking_alerts (
    id TEXT PRIMARY KEY,
    story_id TEXT NOT NULL UNIQUE REFERENCES news_stories(id),
    min_sources INTEGER NOT NULL DEFAULT 2 CHECK(min_sources >= 2),
    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','confirmed','rejected')),
    explanation TEXT NOT NULL DEFAULT 'awaiting independent confirmation',
    created_at TEXT NOT NULL,
    confirmed_at TEXT
);
CREATE TABLE IF NOT EXISTS news_breaking_confirmations (
    alert_id TEXT NOT NULL REFERENCES news_breaking_alerts(id) ON DELETE CASCADE,
    source_id TEXT NOT NULL REFERENCES sources(id),
    article_id TEXT NOT NULL REFERENCES articles(id),
    observed_at TEXT NOT NULL,
    PRIMARY KEY(alert_id, source_id)
);
CREATE TABLE IF NOT EXISTS news_article_normalizations (
    article_id TEXT PRIMARY KEY REFERENCES articles(id) ON DELETE CASCADE,
    source_language TEXT NOT NULL,
    normalized_language TEXT NOT NULL,
    normalized_title TEXT NOT NULL,
    normalized_summary TEXT NOT NULL,
    translator TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS news_editorial_decisions (
    id TEXT PRIMARY KEY,
    article_id TEXT NOT NULL REFERENCES articles(id),
    decision TEXT NOT NULL CHECK(decision IN ('selected','deduplicated','rejected','held')),
    reason_codes_json TEXT NOT NULL,
    explanation TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS news_source_health_state (
    source_id TEXT PRIMARY KEY REFERENCES sources(id),
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL DEFAULT 'healthy' CHECK(state IN ('healthy','degraded','quarantined')),
    quarantined_at TEXT,
    recovered_at TEXT,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS news_source_health_events (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(id),
    outcome TEXT NOT NULL CHECK(outcome IN ('success','failure','quarantined','recovered')),
    detail_redacted TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS news_digests (
    id TEXT PRIMARY KEY,
    cadence TEXT NOT NULL CHECK(cadence IN ('daily','weekly')),
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL,
    summary TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(cadence, period_start, period_end)
);
CREATE TABLE IF NOT EXISTS news_digest_items (
    digest_id TEXT NOT NULL REFERENCES news_digests(id) ON DELETE CASCADE,
    story_id TEXT NOT NULL REFERENCES news_stories(id),
    explanation TEXT NOT NULL,
    PRIMARY KEY(digest_id, story_id)
);
CREATE TABLE IF NOT EXISTS news_archive (
    id TEXT PRIMARY KEY,
    digest_id TEXT NOT NULL UNIQUE REFERENCES news_digests(id),
    snapshot_json TEXT NOT NULL,
    archived_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS news_updates_story_time ON news_story_updates(story_id, happened_at);
CREATE INDEX IF NOT EXISTS news_claim_story ON news_claims(story_id);
CREATE INDEX IF NOT EXISTS news_health_source_time ON news_source_health_events(source_id, created_at);
"""


class NewsIntelligenceRepository:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        connection = self._connect()
        try:
            connection.executescript(f"BEGIN IMMEDIATE;\n{_SCHEMA}\nCOMMIT;")
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def upsert_story(
        self,
        *,
        story_id: str,
        title: str,
        article_id: str,
        source_id: str,
        topics: Iterable[str] = (),
        entities: Iterable[str] = (),
        geography: Iterable[str] = (),
        summary: str,
        update_type: str = "development",
        happened_at: str | None = None,
    ) -> dict[str, Any]:
        happened_at = happened_at or _now()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            exists = connection.execute("SELECT 1 FROM news_stories WHERE id = ?", (story_id,)).fetchone()
            connection.execute(
                """INSERT INTO news_stories VALUES (?, ?, 'developing', ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET title = excluded.title,
                       topics_json = excluded.topics_json,
                       entities_json = excluded.entities_json,
                       geography_json = excluded.geography_json,
                       last_updated_at = excluded.last_updated_at""",
                (story_id, title, _json(sorted(set(topics))), _json(sorted(set(entities))), _json(sorted(set(geography))), happened_at, happened_at),
            )
            connection.execute(
                "INSERT OR IGNORE INTO news_story_articles VALUES (?, ?, ?, ?)",
                (story_id, article_id, source_id, happened_at),
            )
            update_id = _id("storyupdate", story_id, article_id, update_type, summary)
            connection.execute(
                """INSERT OR IGNORE INTO news_story_updates VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (update_id, story_id, article_id, "created" if not exists else update_type, summary, happened_at, _json({"article_id": article_id, "source_id": source_id})),
            )
            connection.commit()
            row = connection.execute("SELECT * FROM news_stories WHERE id = ?", (story_id,)).fetchone()
            return dict(row)
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def story_timeline(self, story_id: str) -> dict[str, Any]:
        connection = self._connect()
        try:
            story = connection.execute("SELECT * FROM news_stories WHERE id = ?", (story_id,)).fetchone()
            if story is None:
                raise KeyError(story_id)
            updates = connection.execute(
                "SELECT * FROM news_story_updates WHERE story_id = ? ORDER BY happened_at, id",
                (story_id,),
            ).fetchall()
            return {"story": dict(story), "updates": [dict(row) for row in updates]}
        finally:
            connection.close()

    def add_claim_evidence(
        self,
        *,
        story_id: str,
        statement: str,
        article_id: str,
        source_id: str,
        stance: str,
        quote: str | None = None,
    ) -> str:
        normalized = " ".join(statement.casefold().split())
        claim_id = _id("claim", story_id, normalized)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT OR IGNORE INTO news_claims VALUES (?, ?, ?, ?, ?)",
                (claim_id, story_id, statement.strip(), normalized, _now()),
            )
            connection.execute(
                """INSERT INTO news_claim_evidence VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(claim_id, article_id) DO UPDATE SET
                       stance = excluded.stance, quote = excluded.quote,
                       observed_at = excluded.observed_at""",
                (claim_id, article_id, source_id, stance, quote, _now()),
            )
            connection.commit()
            return claim_id
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def contradiction_matrix(self, story_id: str) -> list[dict[str, Any]]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """SELECT c.id, c.statement, e.source_id, e.article_id, e.stance, e.quote
                   FROM news_claims c JOIN news_claim_evidence e ON e.claim_id = c.id
                   WHERE c.story_id = ? ORDER BY c.id, e.source_id""",
                (story_id,),
            ).fetchall()
        finally:
            connection.close()
        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            item = grouped.setdefault(row["id"], {"claim_id": row["id"], "statement": row["statement"], "sources": [], "contradiction": False})
            item["sources"].append({"source_id": row["source_id"], "article_id": row["article_id"], "stance": row["stance"], "quote": row["quote"]})
        for item in grouped.values():
            stances = {source["stance"] for source in item["sources"]}
            item["contradiction"] = "supports" in stances and "disputes" in stances
        return list(grouped.values())

    def record_reliability(
        self, source_id: str, *, outcome: str, evidence_ref: str, explanation: str
    ) -> dict[str, Any]:
        outcome_score = {"confirmed": 1.0, "contradicted": 0.0, "mixed": 0.5}[outcome]
        connection = self._connect()
        try:
            previous = connection.execute(
                """SELECT score FROM news_source_reliability_history
                   WHERE source_id = ? ORDER BY created_at DESC, id DESC LIMIT 1""",
                (source_id,),
            ).fetchone()
            score = outcome_score if previous is None else round(float(previous[0]) * 0.8 + outcome_score * 0.2, 6)
            row_id = _id("reliability", source_id, evidence_ref)
            connection.execute(
                "INSERT OR REPLACE INTO news_source_reliability_history VALUES (?, ?, ?, ?, ?, ?, ?)",
                (row_id, source_id, score, outcome, evidence_ref, explanation, _now()),
            )
            connection.commit()
            return {"source_id": source_id, "score": score, "outcome": outcome, "evidence_ref": evidence_ref, "explanation": explanation}
        finally:
            connection.close()

    def create_watchlist(
        self, name: str, *, topics: Iterable[str] = (), entities: Iterable[str] = (), geography: Iterable[str] = ()
    ) -> dict[str, Any]:
        watchlist_id = _id("watchlist", name.casefold())
        now = _now()
        connection = self._connect()
        try:
            connection.execute(
                "INSERT INTO news_watchlists VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
                (watchlist_id, name, _json(sorted(set(topics))), _json(sorted(set(entities))), _json(sorted(set(geography))), now, now),
            )
            connection.commit()
            return {"id": watchlist_id, "name": name}
        finally:
            connection.close()

    def preview_watchlist(self, watchlist_id: str) -> list[dict[str, Any]]:
        connection = self._connect()
        try:
            watchlist = connection.execute("SELECT * FROM news_watchlists WHERE id = ? AND enabled = 1", (watchlist_id,)).fetchone()
            if watchlist is None:
                raise KeyError(watchlist_id)
            stories = connection.execute("SELECT * FROM news_stories ORDER BY last_updated_at DESC, id").fetchall()
        finally:
            connection.close()
        wanted = {key: set(json.loads(watchlist[f"{key}_json"])) for key in ("topics", "entities", "geography")}
        result = []
        for story in stories:
            reasons = []
            for key in wanted:
                matches = sorted(wanted[key] & set(json.loads(story[f"{key}_json"])))
                if matches:
                    reasons.append({key: matches})
            if reasons:
                result.append({"story_id": story["id"], "title": story["title"], "reasons": reasons})
        return result

    def confirm_breaking(
        self, story_id: str, *, source_id: str, article_id: str, min_sources: int = 2
    ) -> dict[str, Any]:
        alert_id = _id("breaking", story_id)
        now = _now()
        connection = self._connect()
        try:
            health = connection.execute("SELECT state FROM news_source_health_state WHERE source_id = ?", (source_id,)).fetchone()
            if health and health[0] == "quarantined":
                raise ValueError("quarantined sources cannot confirm breaking alerts")
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """INSERT OR IGNORE INTO news_breaking_alerts(
                       id, story_id, min_sources, created_at
                   ) VALUES (?, ?, ?, ?)""",
                (alert_id, story_id, max(2, min_sources), now),
            )
            connection.execute(
                "INSERT OR IGNORE INTO news_breaking_confirmations VALUES (?, ?, ?, ?)",
                (alert_id, source_id, article_id, now),
            )
            count = connection.execute(
                "SELECT COUNT(DISTINCT source_id) FROM news_breaking_confirmations WHERE alert_id = ?",
                (alert_id,),
            ).fetchone()[0]
            required = connection.execute("SELECT min_sources FROM news_breaking_alerts WHERE id = ?", (alert_id,)).fetchone()[0]
            if count >= required:
                connection.execute(
                    """UPDATE news_breaking_alerts SET status = 'confirmed', confirmed_at = ?,
                           explanation = ? WHERE id = ?""",
                    (now, f"confirmed by {count} distinct non-quarantined sources", alert_id),
                )
            connection.commit()
            row = connection.execute("SELECT * FROM news_breaking_alerts WHERE id = ?", (alert_id,)).fetchone()
            return {**dict(row), "distinct_sources": count}
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def normalize_article(
        self,
        article_id: str,
        *,
        source_language: str,
        normalized_language: str,
        title: str,
        summary: str,
        translator: str,
    ) -> dict[str, Any]:
        now = _now()
        connection = self._connect()
        try:
            connection.execute(
                """INSERT INTO news_article_normalizations VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(article_id) DO UPDATE SET
                       source_language = excluded.source_language,
                       normalized_language = excluded.normalized_language,
                       normalized_title = excluded.normalized_title,
                       normalized_summary = excluded.normalized_summary,
                       translator = excluded.translator, updated_at = excluded.updated_at""",
                (article_id, source_language, normalized_language, title, summary, translator, now, now),
            )
            connection.commit()
            row = connection.execute("SELECT * FROM news_article_normalizations WHERE article_id = ?", (article_id,)).fetchone()
            return dict(row)
        finally:
            connection.close()

    def explain_decision(
        self, article_id: str, *, decision: str, reason_codes: Iterable[str], explanation: str, evidence: dict[str, Any]
    ) -> dict[str, Any]:
        decision_id = _id("decision", article_id, decision, hashlib.sha256(_json(evidence).encode()).hexdigest())
        connection = self._connect()
        try:
            connection.execute(
                "INSERT OR REPLACE INTO news_editorial_decisions VALUES (?, ?, ?, ?, ?, ?, ?)",
                (decision_id, article_id, decision, _json(sorted(set(reason_codes))), explanation, _json(evidence), _now()),
            )
            connection.commit()
            return {"id": decision_id, "article_id": article_id, "decision": decision, "reason_codes": sorted(set(reason_codes)), "explanation": explanation, "evidence": evidence}
        finally:
            connection.close()

    def record_source_health(
        self, source_id: str, *, success: bool, detail_redacted: str, failure_threshold: int = 3
    ) -> dict[str, Any]:
        now = _now()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute("SELECT * FROM news_source_health_state WHERE source_id = ?", (source_id,)).fetchone()
            failures = 0 if success else (int(previous["consecutive_failures"]) if previous else 0) + 1
            previous_state = previous["state"] if previous else "healthy"
            state = "healthy" if success else "quarantined" if failures >= failure_threshold else "degraded"
            quarantined_at = now if state == "quarantined" and previous_state != "quarantined" else (previous["quarantined_at"] if previous else None)
            recovered_at = now if success and previous_state == "quarantined" else (previous["recovered_at"] if previous else None)
            connection.execute(
                """INSERT INTO news_source_health_state VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(source_id) DO UPDATE SET
                       consecutive_failures = excluded.consecutive_failures,
                       state = excluded.state, quarantined_at = excluded.quarantined_at,
                       recovered_at = excluded.recovered_at, updated_at = excluded.updated_at""",
                (source_id, failures, state, quarantined_at, recovered_at, now),
            )
            outcome = "recovered" if success and previous_state == "quarantined" else "success" if success else "quarantined" if state == "quarantined" else "failure"
            connection.execute(
                "INSERT INTO news_source_health_events VALUES (?, ?, ?, ?, ?)",
                (f"health_{uuid.uuid4().hex}", source_id, outcome, detail_redacted, now),
            )
            connection.commit()
            return {"source_id": source_id, "state": state, "consecutive_failures": failures, "outcome": outcome}
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def build_digest(self, *, cadence: str, period_start: str, period_end: str) -> dict[str, Any]:
        if cadence not in {"daily", "weekly"}:
            raise ValueError("digest cadence must be daily or weekly")
        digest_id = _id("digest", cadence, period_start, period_end)
        connection = self._connect()
        try:
            stories = connection.execute(
                """SELECT DISTINCT s.id, s.title FROM news_stories s
                   JOIN news_story_updates u ON u.story_id = s.id
                   WHERE u.happened_at >= ? AND u.happened_at < ?
                   ORDER BY s.last_updated_at DESC, s.id""",
                (period_start, period_end),
            ).fetchall()
            summary = f"{len(stories)} public stories updated in this {cadence} period"
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT OR IGNORE INTO news_digests VALUES (?, ?, ?, ?, ?, ?)",
                (digest_id, cadence, period_start, period_end, summary, _now()),
            )
            for story in stories:
                connection.execute(
                    "INSERT OR IGNORE INTO news_digest_items VALUES (?, ?, ?)",
                    (digest_id, story["id"], "included because its public timeline changed in the period"),
                )
            snapshot = {"digest_id": digest_id, "cadence": cadence, "period_start": period_start, "period_end": period_end, "stories": [dict(row) for row in stories]}
            archive_id = _id("archive", digest_id)
            connection.execute(
                "INSERT OR REPLACE INTO news_archive VALUES (?, ?, ?, ?)",
                (archive_id, digest_id, _json(snapshot), _now()),
            )
            connection.commit()
            return {**snapshot, "summary": summary, "archive_id": archive_id}
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

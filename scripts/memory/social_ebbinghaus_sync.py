#!/usr/bin/env python
"""Synchronize social-platform session traces into Ebbinghaus memory.

This utility reads Hermes' profile-aware session database plus the local
LM-twitterer activity log, converts bounded recent social interactions into
compact durable memory traces, and stores them in the Ebbinghaus SQLite store.
It intentionally stores summaries/snippets, not full private chat dumps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:  # pragma: no cover - exercised in real Hermes runtime, not unit fixtures
    from hermes_constants import get_hermes_home
except Exception:  # pragma: no cover
    get_hermes_home = None  # type: ignore[assignment]

# Gateway session `source` values for social channels.  ``line-personal`` is kept
# as an alias for forks that tag the LINE AI bot separately from Messaging API.
DEFAULT_SOURCES = ("line", "line-personal", "discord", "telegram")
SECRET_PATTERNS = (
    re.compile(r"(?i)\b(api[_-]?key|secret[_-]?key|token|secret|password|passwd|auth[_-]?token|ct0)\s*[:=]\s*[^\s,;]+"),
    re.compile(r"(?i)([?&](code|token|auth|key|secret)=)[^\s&#]+"),
    re.compile(r"\b[A-Za-z0-9_-]{32,}\b"),
)
SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[\w][\w.+#:/-]{1,}", re.UNICODE)
CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]+")


@dataclass(frozen=True)
class MemoryCandidate:
    content: str
    tags: tuple[str, ...]
    source: str
    session_id: str = ""
    salience: float = 0.55
    valence: float = 0.0


def hermes_home() -> Path:
    if get_hermes_home is not None:
        return Path(get_hermes_home())
    return Path.home() / ".hermes"


def redact_sensitive_text(text: str) -> str:
    cleaned = text or ""
    for pattern in SECRET_PATTERNS:
        cleaned = pattern.sub(lambda m: (m.group(1) + "[REDACTED]") if m.lastindex else "[REDACTED]", cleaned)
    return SPACE_RE.sub(" ", cleaned).strip()


def _snippet(text: str, limit: int) -> str:
    return redact_sensitive_text(text)[:limit].strip()


def _tokenize(text: str) -> list[str]:
    lowered = (text or "").lower()
    tokens = [m.group(0).strip("._-/#:") for m in TOKEN_RE.finditer(lowered)]
    compact_cjk = "".join(CJK_RE.findall(lowered))
    tokens.extend(compact_cjk[i : i + 2] for i in range(max(0, len(compact_cjk) - 1)))
    tokens.extend(compact_cjk[i : i + 3] for i in range(max(0, len(compact_cjk) - 2)))
    return [t for t in tokens if len(t) >= 2]


def _encoding(content: str, tags: Sequence[str]) -> tuple[str, str]:
    counts: dict[str, int] = {}
    for token in _tokenize(" ".join([content, *tags])):
        counts[token] = counts.get(token, 0) + 1
    cues = sorted(counts, key=lambda t: (-counts[t], t))[:64]
    encoded = {
        "version": 1,
        "kind": "social_memory_sync",
        "summary": content[:280],
        "cue_vector": {cue: counts[cue] for cue in cues},
        "cues": cues,
        "length": len(content),
    }
    return json.dumps(encoded, ensure_ascii=False), " ".join(cues)


def _ensure_memory_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS memories (
            memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL UNIQUE,
            encoded TEXT NOT NULL,
            cues TEXT DEFAULT '',
            tags TEXT DEFAULT '',
            salience REAL DEFAULT 0.6,
            valence REAL DEFAULT 0.0,
            strength REAL DEFAULT 1.0,
            rehearsal_count INTEGER DEFAULT 0,
            retrieval_count INTEGER DEFAULT 0,
            source TEXT DEFAULT '',
            session_id TEXT DEFAULT '',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            last_rehearsed_at REAL,
            last_retrieved_at REAL
        );
        CREATE INDEX IF NOT EXISTS idx_ebbinghaus_tags ON memories(tags);
        CREATE INDEX IF NOT EXISTS idx_ebbinghaus_updated ON memories(updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_ebbinghaus_salience ON memories(salience DESC);
        """
    )


def _remember(conn: sqlite3.Connection, candidate: MemoryCandidate) -> bool:
    content = redact_sensitive_text(candidate.content)
    if len(content) < 12:
        return False
    tags = tuple(dict.fromkeys(t.strip().lower() for t in candidate.tags if t.strip()))
    encoded, cues = _encoding(content, tags)
    now = time.time()
    before = conn.total_changes
    conn.execute(
        """
        INSERT INTO memories
            (content, encoded, cues, tags, salience, valence, strength,
             source, session_id, created_at, updated_at, last_rehearsed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(content) DO UPDATE SET
            encoded=excluded.encoded,
            cues=excluded.cues,
            tags=excluded.tags,
            salience=max(memories.salience, excluded.salience),
            strength=min(6.0, memories.strength + 0.05),
            updated_at=excluded.updated_at,
            last_rehearsed_at=excluded.last_rehearsed_at,
            source=excluded.source,
            session_id=excluded.session_id
        """,
        (
            content,
            encoded,
            cues,
            ",".join(tags),
            min(1.0, max(0.05, candidate.salience)),
            min(1.0, max(-1.0, candidate.valence)),
            1.0 + min(1.0, max(0.05, candidate.salience)),
            candidate.source,
            candidate.session_id,
            now,
            now,
            now,
        ),
    )
    return conn.total_changes > before


class SocialMemorySync:
    def __init__(
        self,
        *,
        state_db: Path | None = None,
        memory_db: Path | None = None,
        x_activity_log: Path | None = None,
        sources: Sequence[str] = DEFAULT_SOURCES,
        min_started_at: float | None = None,
    ) -> None:
        home = hermes_home()
        self.state_db = Path(state_db or home / "state.db").expanduser()
        self.memory_db = Path(memory_db or home / "ebbinghaus_memory.db").expanduser()
        self.x_activity_log = Path(x_activity_log or home / "lm-twitterer" / "activity.jsonl").expanduser()
        self.sources = tuple(dict.fromkeys(s.strip().lower() for s in sources if s.strip()))
        self.min_started_at = min_started_at

    def run(self, *, max_sessions: int, max_x_events: int, sleep: bool) -> dict[str, Any]:
        self.memory_db.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "sources": list(self.sources),
            "state_db": str(self.state_db),
            "memory_db": str(self.memory_db),
            "x_activity_log": str(self.x_activity_log),
            "sessions_seen": 0,
            "x_events_seen": 0,
            "memories_written": 0,
            "min_started_at": self.min_started_at,
            "sleep": None,
        }
        with sqlite3.connect(self.memory_db) as memory_conn:
            _ensure_memory_schema(memory_conn)
            for candidate in self._session_candidates(max_sessions):
                result["sessions_seen"] += 1
                if _remember(memory_conn, candidate):
                    result["memories_written"] += 1
            for candidate in self._x_candidates(max_x_events):
                result["x_events_seen"] += 1
                if _remember(memory_conn, candidate):
                    result["memories_written"] += 1
            if sleep:
                result["sleep"] = self._sleep(memory_conn)
            memory_conn.commit()
        return result

    def _session_candidates(self, max_sessions: int) -> Iterable[MemoryCandidate]:
        if max_sessions <= 0 or not self.state_db.exists() or not self.sources:
            return []
        placeholders = ",".join("?" for _ in self.sources)
        filters = [f"lower(source) IN ({placeholders})"]
        params: list[Any] = list(self.sources)
        if self.min_started_at is not None:
            filters.append("started_at > ?")
            params.append(float(self.min_started_at))
        query = f"""
            SELECT id, source, COALESCE(title, '') AS title, started_at
            FROM sessions
            WHERE {' AND '.join(filters)}
            ORDER BY started_at DESC
            LIMIT ?
        """
        params.append(max_sessions)
        candidates: list[MemoryCandidate] = []
        try:
            with sqlite3.connect(self.state_db) as con:
                con.row_factory = sqlite3.Row
                sessions = con.execute(query, tuple(params)).fetchall()
                for session in sessions:
                    messages = con.execute(
                        """
                        SELECT role, COALESCE(content, '') AS content
                        FROM messages
                        WHERE session_id = ? AND active = 1 AND role IN ('user', 'assistant')
                        ORDER BY timestamp ASC, id ASC
                        LIMIT 12
                        """,
                        (session["id"],),
                    ).fetchall()
                    turns = []
                    for msg in messages:
                        text = _snippet(str(msg["content"]), 260)
                        if text:
                            turns.append(f"{msg['role']}: {text}")
                    if not turns:
                        continue
                    title = _snippet(str(session["title"]), 100)
                    content = (
                        f"Social memory from {session['source']} session"
                        f" {session['id']}"
                        f"{f' ({title})' if title else ''}: "
                        + " | ".join(turns)
                    )
                    candidates.append(
                        MemoryCandidate(
                            content=content,
                            tags=("social-memory", str(session["source"]), "session", "gateway"),
                            source="social-memory-sync",
                            session_id=str(session["id"]),
                            salience=0.58,
                        )
                    )
        except sqlite3.Error:
            return []
        return candidates

    def _x_candidates(self, max_events: int) -> Iterable[MemoryCandidate]:
        if max_events <= 0 or not self.x_activity_log.exists():
            return []
        lines = self.x_activity_log.read_text(encoding="utf-8", errors="ignore").splitlines()[-max_events:]
        candidates: list[MemoryCandidate] = []
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = event.get("tweet_text") or event.get("reply_text") or event.get("text") or ""
            text = _snippet(str(text), 360)
            if not text:
                continue
            action = str(event.get("action") or "x-event")
            dry_run = bool(event.get("dry_run"))
            ok = event.get("ok")
            state = "draft" if dry_run else "published"
            content = f"X memory from lm-twitterer {action} ({state}, ok={ok}): {text}"
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
            candidates.append(
                MemoryCandidate(
                    content=content,
                    tags=("social-memory", "x", "twitter", "lm-twitterer", action, state),
                    source="social-memory-sync",
                    session_id=f"lm-twitterer:{digest}",
                    salience=0.45 if dry_run else 0.55,
                    valence=0.05,
                )
            )
        return candidates

    def _sleep(self, conn: sqlite3.Connection) -> dict[str, int]:
        now = time.time()
        rows = conn.execute(
            "SELECT memory_id, salience, strength, COALESCE(last_rehearsed_at, updated_at, created_at) FROM memories"
        ).fetchall()
        rehearsed = forgotten = 0
        for memory_id, salience, strength, anchor in rows:
            elapsed_days = max(0.0, (now - float(anchor or now)) / 86400.0)
            stability = 3.0 * max(0.1, float(strength or 1.0))
            retention = math.exp(-elapsed_days / stability)
            if retention < 0.45 and float(salience or 0.0) >= 0.7:
                conn.execute(
                    "UPDATE memories SET rehearsal_count=rehearsal_count+1, strength=min(6.0, strength+0.2), last_rehearsed_at=?, updated_at=? WHERE memory_id=?",
                    (now, now, memory_id),
                )
                rehearsed += 1
            elif retention < 0.08 and float(salience or 0.0) < 0.35:
                conn.execute("DELETE FROM memories WHERE memory_id=?", (memory_id,))
                forgotten += 1
        return {"mode": "sleep_cycle", "rehearsed": rehearsed, "forgotten": forgotten}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-db", type=Path, default=None)
    parser.add_argument("--memory-db", type=Path, default=None)
    parser.add_argument("--x-activity-log", type=Path, default=None)
    parser.add_argument("--sources", default=",".join(DEFAULT_SOURCES), help="Comma-separated session sources to import")
    parser.add_argument("--max-sessions", type=int, default=80)
    parser.add_argument("--max-x-events", type=int, default=80)
    parser.add_argument("--no-sleep", action="store_true", help="Skip Ebbinghaus rehearsal/forgetting pass")
    parser.add_argument(
        "--min-started-at",
        type=float,
        default=None,
        help="Only import sessions with started_at greater than this Unix timestamp",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    sync = SocialMemorySync(
        state_db=args.state_db,
        memory_db=args.memory_db,
        x_activity_log=args.x_activity_log,
        sources=tuple(part.strip() for part in args.sources.split(",") if part.strip()),
        min_started_at=args.min_started_at,
    )
    result = sync.run(max_sessions=args.max_sessions, max_x_events=args.max_x_events, sleep=not args.no_sleep)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

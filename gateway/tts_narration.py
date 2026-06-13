"""Long-form TTS narration jobs for gateway voice delivery.

This module intentionally keeps the first implementation boring and auditable:
SQLite for job/chunk state, deterministic chunk ordering, and metadata-first
telemetry.  Full private text lives only in chunk rows while a job is retryable;
job rows store hashes/lengths rather than full reply text.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from hermes_constants import get_hermes_home
from tools.tts_tool import text_to_speech_tool

DEFAULT_TARGET_CHARS = 1000
DEFAULT_MAX_CHARS = 1200

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?…])\s+")
_CLAUSE_BOUNDARY_RE = re.compile(r"(?<=[,;:])\s+")


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _collapse_ws(text: str) -> str:
    # Preserve paragraph boundaries for splitting, but avoid accidental giant
    # whitespace runs inside a chunk.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_once(text: str, max_chars: int) -> tuple[str, str]:
    """Split *text* into a non-empty head/tail, preferring natural boundaries."""
    if len(text) <= max_chars:
        return text, ""

    window = text[: max_chars + 1]
    boundary_patterns = ["\n\n", "\n", ". ", "! ", "? ", "… ", ", ", "; ", ": ", " "]
    best = -1
    for marker in boundary_patterns:
        idx = window.rfind(marker, 0, max_chars + 1)
        if idx > best:
            best = idx + len(marker.rstrip())
    if best <= 0:
        best = max_chars
    head = text[:best].strip()
    tail = text[best:].strip()
    if not head:  # defensive: never emit empty chunks
        head = text[:max_chars].strip()
        tail = text[max_chars:].strip()
    return head, tail


def _paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _sentences(text: str) -> List[str]:
    return [p.strip() for p in _SENTENCE_BOUNDARY_RE.split(text) if p.strip()]


def _clauses(text: str) -> List[str]:
    return [p.strip() for p in _CLAUSE_BOUNDARY_RE.split(text) if p.strip()]


def _units_for(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    for splitter in (_paragraphs, _sentences, _clauses, lambda s: s.split()):
        units = splitter(text)
        if len(units) > 1 and all(len(u) <= max_chars for u in units):
            return units
    pieces: List[str] = []
    rest = text
    while rest:
        head, rest = _split_once(rest, max_chars)
        if head:
            pieces.append(head)
    return pieces


def chunk_narration_text(
    text: str,
    *,
    target_chars: int = DEFAULT_TARGET_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> List[str]:
    """Split narration text into ordered provider-safe chunks.

    Preference order: paragraph → sentence → clause → word → hard cut.  The
    function never emits empty chunks and never adds labels or commentary.
    """
    clean = _collapse_ws(text or "")
    if not clean:
        return []
    target_chars = max(1, min(int(target_chars), int(max_chars)))
    max_chars = max(1, int(max_chars))

    chunks: List[str] = []
    current = ""
    for unit in _units_for(clean, max_chars):
        if not unit:
            continue
        if len(unit) > max_chars:
            # Last-resort hard splitting for a pathological single token.
            while unit:
                head, unit = _split_once(unit, max_chars)
                if head:
                    chunks.append(head)
            continue
        candidate = f"{current} {unit}".strip() if current else unit
        if current and len(candidate) > target_chars:
            chunks.append(current.strip())
            current = unit
        else:
            current = candidate
        while len(current) > max_chars:
            head, current = _split_once(current, max_chars)
            chunks.append(head)
    if current.strip():
        chunks.append(current.strip())
    # Defensive final pass: no empty chunks, no over-limit chunks.
    final: List[str] = []
    for chunk in chunks:
        rest = chunk.strip()
        while len(rest) > max_chars:
            head, rest = _split_once(rest, max_chars)
            if head:
                final.append(head)
        if rest:
            final.append(rest)
    return final


@dataclass(frozen=True)
class NarrationJob:
    job_id: str
    idempotency_key: str
    status: str


class NarrationJobStore:
    """SQLite store for long-form narration jobs and ordered chunks."""

    def __init__(self, path: Optional[os.PathLike[str] | str] = None):
        self.path = Path(path) if path is not None else get_hermes_home() / "tts_narration.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tts_narration_jobs (
                    job_id TEXT PRIMARY KEY,
                    idempotency_key TEXT UNIQUE NOT NULL,
                    platform TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    thread_id TEXT,
                    reply_to_message_id TEXT,
                    provider TEXT,
                    model TEXT,
                    voice TEXT,
                    scope_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    text_sha256 TEXT NOT NULL,
                    text_chars INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    policy_json TEXT NOT NULL,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    last_error TEXT
                )
                """
            )
            table_info = conn.execute("PRAGMA table_info(tts_narration_jobs)").fetchall()
            columns = {row[1] for row in table_info}
            if "metadata_json" not in columns:
                conn.execute("ALTER TABLE tts_narration_jobs ADD COLUMN metadata_json TEXT")
                table_info = conn.execute("PRAGMA table_info(tts_narration_jobs)").fetchall()
                columns = {row[1] for row in table_info}
            if any(row[1] == "provider" and row[3] for row in table_info):
                self._migrate_nullable_provider_metadata(conn)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tts_narration_chunks (
                    job_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_total INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    text_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    audio_path TEXT,
                    sent_message_id TEXT,
                    telegram_file_id TEXT,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, chunk_index),
                    FOREIGN KEY (job_id) REFERENCES tts_narration_jobs(job_id)
                )
                """
            )
            conn.commit()

    @staticmethod
    def _migrate_nullable_provider_metadata(conn: sqlite3.Connection) -> None:
        """Relax old prototype DBs where provider was NOT NULL.

        Early local builds always stored an explicit narration provider.  The
        upstream-ready path allows NULL provider/model/voice so processing can
        inherit the regular text_to_speech provider at execution time. SQLite
        cannot drop a NOT NULL constraint in place, so rebuild the small jobs
        table while preserving existing rows.
        """
        temp_name = "tts_narration_jobs_old_provider_not_null"
        conn.execute(f"DROP TABLE IF EXISTS {temp_name}")
        conn.execute("PRAGMA legacy_alter_table=ON")
        conn.execute("ALTER TABLE tts_narration_jobs RENAME TO tts_narration_jobs_old_provider_not_null")
        conn.execute("PRAGMA legacy_alter_table=OFF")
        conn.execute(
            """
            CREATE TABLE tts_narration_jobs (
                job_id TEXT PRIMARY KEY,
                idempotency_key TEXT UNIQUE NOT NULL,
                platform TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                thread_id TEXT,
                reply_to_message_id TEXT,
                provider TEXT,
                model TEXT,
                voice TEXT,
                scope_key TEXT NOT NULL,
                status TEXT NOT NULL,
                text_sha256 TEXT NOT NULL,
                text_chars INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL,
                policy_json TEXT NOT NULL,
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                last_error TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO tts_narration_jobs (
                job_id, idempotency_key, platform, chat_id, thread_id,
                reply_to_message_id, provider, model, voice, scope_key, status,
                text_sha256, text_chars, chunk_count, policy_json,
                metadata_json, created_at, started_at, completed_at, last_error
            )
            SELECT
                job_id, idempotency_key, platform, chat_id, thread_id,
                reply_to_message_id, provider, model, voice, scope_key, status,
                text_sha256, text_chars, chunk_count, policy_json,
                metadata_json, created_at, started_at, completed_at, last_error
            FROM tts_narration_jobs_old_provider_not_null
            """
        )
        conn.execute("DROP TABLE tts_narration_jobs_old_provider_not_null")

    def enqueue_job(
        self,
        *,
        platform: str,
        chat_id: str,
        thread_id: Optional[str],
        reply_to_message_id: Optional[str],
        idempotency_key: str,
        text: str,
        chunks: Iterable[str],
        provider: Optional[str],
        model: Optional[str],
        voice: Optional[str],
        scope_key: str,
        policy: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> NarrationJob:
        chunk_list = [c for c in chunks if c and c.strip()]
        if not chunk_list:
            raise ValueError("Cannot enqueue narration job with no chunks")
        now = _utc_now()
        job_id = f"tts-{uuid.uuid4().hex}"
        policy_json = json.dumps(policy or {}, sort_keys=True)
        metadata_json = json.dumps(metadata or {}, sort_keys=True)
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT job_id, idempotency_key, status FROM tts_narration_jobs WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if existing:
                return NarrationJob(existing["job_id"], existing["idempotency_key"], existing["status"])
            try:
                conn.execute(
                    """
                    INSERT INTO tts_narration_jobs (
                    job_id, idempotency_key, platform, chat_id, thread_id,
                    reply_to_message_id, provider, model, voice, scope_key,
                    status, text_sha256, text_chars, chunk_count, policy_json,
                    metadata_json,
                    created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id, idempotency_key, platform, chat_id, thread_id,
                        reply_to_message_id, provider, model, voice, scope_key,
                        _sha256(text), len(text), len(chunk_list), policy_json, metadata_json, now,
                    ),
                )
            except sqlite3.IntegrityError:
                existing = conn.execute(
                    "SELECT job_id, idempotency_key, status FROM tts_narration_jobs WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                if existing:
                    return NarrationJob(existing["job_id"], existing["idempotency_key"], existing["status"])
                raise
            for index, chunk in enumerate(chunk_list, start=1):
                conn.execute(
                    """
                    INSERT INTO tts_narration_chunks (
                        job_id, chunk_index, chunk_total, text, text_sha256,
                        status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'queued', ?, ?)
                    """,
                    (job_id, index, len(chunk_list), chunk, _sha256(chunk), now, now),
                )
            conn.commit()
        return NarrationJob(job_id, idempotency_key, "queued")

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM tts_narration_jobs WHERE job_id = ?", (job_id,)).fetchone()
            return dict(row) if row else None

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM tts_narration_jobs ORDER BY created_at, job_id").fetchall()
            return [dict(r) for r in rows]

    def list_chunks(self, job_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tts_narration_chunks WHERE job_id = ? ORDER BY chunk_index",
                (job_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def update_job_status(self, job_id: str, status: str, *, last_error: Optional[str] = None) -> None:
        now = _utc_now()
        completed = now if status in {"complete", "failed", "cancelled"} else None
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE tts_narration_jobs
                   SET status = ?, last_error = COALESCE(?, last_error),
                       started_at = COALESCE(started_at, ?),
                       completed_at = COALESCE(?, completed_at)
                 WHERE job_id = ?
                """,
                (status, last_error, now, completed, job_id),
            )
            if status == "complete":
                conn.execute(
                    """
                    UPDATE tts_narration_chunks
                       SET text = '', audio_path = NULL, updated_at = ?
                     WHERE job_id = ? AND status IN ('sent', 'skipped')
                    """,
                    (now, job_id),
                )
            conn.commit()

    def recover_stale_processing(self, *, older_than_seconds: int = 900) -> int:
        """Move stale in-flight jobs/chunks back to retryable failed state.

        A narration worker can be cancelled after claiming a job/chunk, for
        example when a Telegram session is interrupted by a new inbound message
        or the gateway is draining.  ``asyncio.CancelledError`` does not inherit
        from ``Exception`` on modern Python, so defensive exception handlers may
        never run.  Without a recovery sweep those rows remain ``processing``
        forever and strict chunk ordering prevents later chunks from sending.
        """
        seconds = max(1, int(older_than_seconds))
        cutoff = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(seconds=seconds)).isoformat()
        now = _utc_now()
        message = "stale narration processing recovered for retry"
        with self._connect() as conn:
            chunk_cur = conn.execute(
                """
                UPDATE tts_narration_chunks
                   SET status = 'failed',
                       last_error = COALESCE(last_error, ?),
                       updated_at = ?
                 WHERE status = 'processing' AND updated_at < ?
                """,
                (message, now, cutoff),
            )
            job_cur = conn.execute(
                """
                UPDATE tts_narration_jobs
                   SET status = 'failed',
                       last_error = COALESCE(last_error, ?)
                 WHERE status = 'processing'
                   AND COALESCE(started_at, created_at) < ?
                """,
                (message, cutoff),
            )
            conn.commit()
            return int(chunk_cur.rowcount or 0) + int(job_cur.rowcount or 0)

    def claim_job(self, job_id: str) -> bool:
        """Atomically claim a queued/failed job for one processor."""
        now = _utc_now()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE tts_narration_jobs
                   SET status = 'processing', started_at = COALESCE(started_at, ?)
                 WHERE job_id = ? AND status IN ('queued', 'failed')
                """,
                (now, job_id),
            )
            conn.commit()
            return cur.rowcount == 1

    def claim_chunk(self, job_id: str, chunk_index: int) -> bool:
        """Atomically claim a queued or failed chunk before retrying it."""
        now = _utc_now()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE tts_narration_chunks
                   SET status = 'processing', updated_at = ?
                 WHERE job_id = ? AND chunk_index = ? AND status IN ('queued', 'failed')
                """,
                (now, job_id, chunk_index),
            )
            conn.commit()
            return cur.rowcount == 1

    def update_chunk(
        self,
        job_id: str,
        chunk_index: int,
        *,
        status: str,
        audio_path: Optional[str] = None,
        sent_message_id: Optional[str] = None,
        last_error: Optional[str] = None,
        increment_attempts: bool = False,
    ) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE tts_narration_chunks
                   SET status = ?,
                       audio_path = COALESCE(?, audio_path),
                       sent_message_id = COALESCE(?, sent_message_id),
                       last_error = COALESCE(?, last_error),
                       attempt_count = attempt_count + ?,
                       updated_at = ?
                 WHERE job_id = ? AND chunk_index = ?
                """,
                (status, audio_path, sent_message_id, last_error, 1 if increment_attempts else 0, now, job_id, chunk_index),
            )
            conn.commit()


def provider_metadata_from_config() -> Dict[str, Optional[str]]:
    """Return optional long-form narration provider metadata.

    Narration defaults to the same provider resolution path as the regular
    ``text_to_speech`` tool: when no narration-specific override is configured,
    ``provider`` is left as ``None`` so ``text_to_speech_tool`` uses
    ``tts.provider`` and the matching provider config.  Users may override only
    narration via either of these config shapes::

        tts:
          narration:
            provider: openai
            model: gpt-4o-mini-tts
            voice: coral

        voice:
          long_form_tts:   # legacy/local prototype spelling
            primary_provider: openai
            model: gpt-4o-mini-tts
            voice: coral
    """
    provider = None
    model = None
    voice = None
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        tts_cfg = cfg.get("tts") or {}
        narration_cfg = (tts_cfg.get("narration") or {}) if isinstance(tts_cfg, dict) else {}
        legacy_cfg = ((cfg.get("voice") or {}).get("long_form_tts") or {})
        provider = (
            narration_cfg.get("provider")
            or narration_cfg.get("primary_provider")
            or legacy_cfg.get("primary_provider")
            or legacy_cfg.get("provider")
        )
        model = narration_cfg.get("model") or legacy_cfg.get("model")
        voice = narration_cfg.get("voice") or legacy_cfg.get("voice")
    except Exception:
        pass

    def clean(value):
        if value is None:
            return None
        value = str(value).strip()
        return value or None

    return {"provider": clean(provider), "model": clean(model), "voice": clean(voice)}


def sanitize_error(error: Any) -> str:
    """Keep failure state useful without leaking local filesystem paths."""
    text = str(error or "unknown error")[:500]
    text = re.sub(r"/[^\s]+", "[path]", text)
    return text


def narration_audio_path(job_id: str, chunk_index: int) -> str:
    out_dir = Path(tempfile.gettempdir()) / "hermes_voice" / "narration"
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"{job_id}_{chunk_index:03d}.ogg")

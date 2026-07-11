"""SQLite persistence for the local video material library."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import time
from typing import Any, Iterator
import uuid

from hermes_constants import get_hermes_home


SUPPORTED_VIDEO_SUFFIXES = {".avi", ".flv", ".mkv", ".mov", ".mp4", ".webm"}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def _clip_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    item = _row(row)
    if item is None:
        return None
    raw_semantic = item.get("semantic_json") or "{}"
    try:
        item["semantic_json"] = json.loads(raw_semantic)
    except (TypeError, ValueError):
        item["semantic_json"] = {}
    item["materialized"] = bool(item.get("materialized", 1))
    return item


def _safe_filename(name: str) -> str:
    filename = Path(str(name or "video.mp4").replace("\\", "/")).name
    filename = re.sub(r"[\x00-\x1f\x7f/:*?\"<>|]+", "_", filename).strip(" .")
    return filename[:180] or "video.mp4"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class VideoLibraryStore:
    def __init__(
        self,
        root: Path | str | None = None,
        db_path: Path | str | None = None,
        *,
        assets_dir: Path | str | None = None,
        clips_dir: Path | str | None = None,
        keyframes_dir: Path | str | None = None,
    ):
        self.root = Path(root) if root is not None else get_hermes_home() / "video-library"
        self.assets_dir = Path(assets_dir) if assets_dir is not None else self.root / "assets"
        self.clips_dir = Path(clips_dir) if clips_dir is not None else self.root / "clips"
        self.keyframes_dir = Path(keyframes_dir) if keyframes_dir is not None else self.root / "keyframes"
        self.db_path = Path(db_path) if db_path is not None else self.root / "video_library.db"
        for directory in (self.root, self.assets_dir, self.clips_dir, self.keyframes_dir, self.db_path.parent):
            directory.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self.connect() as conn:
            try:
                conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.OperationalError:
                conn.execute("PRAGMA journal_mode=DELETE")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS assets (
                    id TEXT PRIMARY KEY,
                    sha256 TEXT NOT NULL UNIQUE,
                    original_name TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    managed_path TEXT NOT NULL UNIQUE,
                    size_bytes INTEGER NOT NULL,
                    duration_seconds REAL,
                    width INTEGER,
                    height INTEGER,
                    fps REAL,
                    status TEXT NOT NULL,
                    library_id TEXT NOT NULL DEFAULT 'default',
                    source_mode TEXT NOT NULL DEFAULT 'managed',
                    source_mtime_ns INTEGER NOT NULL DEFAULT 0,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS clips (
                    id TEXT PRIMARY KEY,
                    asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
                    clip_index INTEGER NOT NULL,
                    start_seconds REAL NOT NULL,
                    end_seconds REAL NOT NULL,
                    duration_seconds REAL NOT NULL,
                    file_path TEXT NOT NULL,
                    keyframe_path TEXT,
                    description TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'ready',
                    source_file_path TEXT NOT NULL DEFAULT '',
                    semantic_json TEXT NOT NULL DEFAULT '{}',
                    quality_score REAL NOT NULL DEFAULT 0,
                    confidence REAL NOT NULL DEFAULT 0,
                    materialized INTEGER NOT NULL DEFAULT 1,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    UNIQUE(asset_id, clip_index)
                );

                CREATE TABLE IF NOT EXISTS tags (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    created_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS clip_tags (
                    clip_id TEXT NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
                    tag_id TEXT NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
                    confidence REAL NOT NULL DEFAULT 1,
                    source TEXT NOT NULL DEFAULT 'manual',
                    PRIMARY KEY (clip_id, tag_id)
                );

                CREATE TABLE IF NOT EXISTS analysis_jobs (
                    id TEXT PRIMARY KEY,
                    asset_id TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
                    analyzer_version TEXT NOT NULL,
                    state TEXT NOT NULL,
                    progress INTEGER NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT '',
                    stage TEXT NOT NULL DEFAULT 'discovered',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_clips_asset ON clips(asset_id, clip_index);
                CREATE INDEX IF NOT EXISTS idx_clip_tags_tag ON clip_tags(tag_id, clip_id);
                CREATE INDEX IF NOT EXISTS idx_analysis_jobs_asset ON analysis_jobs(asset_id, created_at DESC);
                """
            )

            self._ensure_column(conn, "assets", "library_id", "library_id TEXT NOT NULL DEFAULT 'default'")
            self._ensure_column(conn, "assets", "source_mode", "source_mode TEXT NOT NULL DEFAULT 'managed'")
            self._ensure_column(conn, "assets", "source_mtime_ns", "source_mtime_ns INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "clips", "source_file_path", "source_file_path TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "clips", "semantic_json", "semantic_json TEXT NOT NULL DEFAULT '{}'")
            self._ensure_column(conn, "clips", "quality_score", "quality_score REAL NOT NULL DEFAULT 0")
            self._ensure_column(conn, "clips", "confidence", "confidence REAL NOT NULL DEFAULT 0")
            self._ensure_column(conn, "clips", "materialized", "materialized INTEGER NOT NULL DEFAULT 1")
            self._ensure_column(conn, "analysis_jobs", "stage", "stage TEXT NOT NULL DEFAULT 'discovered'")
            self._ensure_column(conn, "analysis_jobs", "attempts", "attempts INTEGER NOT NULL DEFAULT 0")

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
        columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")

    def import_asset(
        self,
        source_path: Path | str,
        *,
        source_mode: str = "managed",
        library_id: str = "default",
    ) -> dict[str, Any]:
        source = Path(source_path).expanduser().resolve(strict=True)
        if not source.is_file():
            raise ValueError("video asset must be a file")
        if source.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
            raise ValueError(f"unsupported video asset type: {source.suffix or '<none>'}")

        normalized_mode = str(source_mode or "managed").strip().lower()
        if normalized_mode not in {"managed", "linked"}:
            raise ValueError(f"unsupported video asset source mode: {source_mode}")

        content_hash = _sha256(source)
        asset_id = f"asset_{content_hash[:24]}"
        managed = source if normalized_mode == "linked" else self.assets_dir / f"{content_hash[:12]}-{_safe_filename(source.name)}"
        source_stat = source.stat()
        now = _now_ms()
        with self.connect() as conn:
            existing = conn.execute("SELECT * FROM assets WHERE sha256 = ?", (content_hash,)).fetchone()
            if existing is not None:
                if normalized_mode == "linked":
                    conn.execute(
                        """
                        UPDATE assets
                        SET original_name = ?, source_path = ?, managed_path = ?, size_bytes = ?,
                            library_id = ?, source_mode = 'linked', source_mtime_ns = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (
                            source.name,
                            str(source),
                            str(source),
                            source_stat.st_size,
                            str(library_id or "default"),
                            source_stat.st_mtime_ns,
                            now,
                            existing["id"],
                        ),
                    )
                    existing = conn.execute("SELECT * FROM assets WHERE id = ?", (existing["id"],)).fetchone()
                return dict(existing)
            if normalized_mode == "managed" and source != managed:
                shutil.copy2(source, managed)
            conn.execute(
                """
                INSERT INTO assets (
                    id, sha256, original_name, source_path, managed_path,
                    size_bytes, status, library_id, source_mode, source_mtime_ns,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'imported', ?, ?, ?, ?, ?)
                """,
                (
                    asset_id,
                    content_hash,
                    source.name,
                    str(source),
                    str(managed),
                    source_stat.st_size,
                    str(library_id or "default"),
                    normalized_mode,
                    source_stat.st_mtime_ns,
                    now,
                    now,
                ),
            )
            return dict(conn.execute("SELECT * FROM assets WHERE id = ?", (asset_id,)).fetchone())

    def get_asset(self, asset_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            return _row(conn.execute("SELECT * FROM assets WHERE id = ?", (asset_id,)).fetchone())

    def get_clip(self, clip_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            item = _clip_row(conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone())
            if item is not None:
                item["tags"] = self._clip_tags(conn, clip_id)
            return item

    def list_assets(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM assets ORDER BY created_at DESC, id").fetchall()]

    def update_asset_metadata(self, asset_id: str, metadata: dict[str, Any], *, status: str = "analyzed") -> dict[str, Any]:
        allowed = {"duration_seconds", "width", "height", "fps"}
        values = {key: metadata.get(key) for key in allowed if key in metadata}
        assignments = [f"{key} = ?" for key in values]
        assignments.extend(["status = ?", "updated_at = ?"])
        params = [*values.values(), status, _now_ms(), asset_id]
        with self.connect() as conn:
            cursor = conn.execute(f"UPDATE assets SET {', '.join(assignments)} WHERE id = ?", params)
            if cursor.rowcount != 1:
                raise KeyError(f"unknown video asset: {asset_id}")
            return dict(conn.execute("SELECT * FROM assets WHERE id = ?", (asset_id,)).fetchone())

    def replace_clips(self, asset_id: str, clips: list[dict[str, Any]]) -> list[dict[str, Any]]:
        now = _now_ms()
        with self.connect() as conn:
            if conn.execute("SELECT 1 FROM assets WHERE id = ?", (asset_id,)).fetchone() is None:
                raise KeyError(f"unknown video asset: {asset_id}")
            conn.execute("DELETE FROM clips WHERE asset_id = ?", (asset_id,))
            for index, raw in enumerate(clips):
                start = float(raw["start_seconds"])
                end = float(raw["end_seconds"])
                if start < 0 or end <= start:
                    raise ValueError("clip boundaries must have 0 <= start < end")
                clip_id = f"clip_{uuid.uuid4().hex}"
                conn.execute(
                    """
                    INSERT INTO clips (
                        id, asset_id, clip_index, start_seconds, end_seconds,
                        duration_seconds, file_path, keyframe_path, description,
                        status, source_file_path, semantic_json, quality_score,
                        confidence, materialized, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        clip_id,
                        asset_id,
                        index,
                        start,
                        end,
                        round(end - start, 6),
                        str(raw.get("file_path") or ""),
                        str(raw.get("keyframe_path") or ""),
                        str(raw.get("description") or ""),
                        str(raw.get("status") or "ready"),
                        str(raw.get("source_file_path") or ""),
                        json.dumps(raw.get("semantic_json") or {}, ensure_ascii=False),
                        max(0.0, min(1.0, float(raw.get("quality_score", 0.0)))),
                        max(0.0, min(1.0, float(raw.get("confidence", 0.0)))),
                        int(bool(raw.get("materialized", bool(raw.get("file_path"))))),
                        now,
                        now,
                    ),
                )
            return [
                _clip_row(row)
                for row in conn.execute(
                    "SELECT * FROM clips WHERE asset_id = ? ORDER BY clip_index", (asset_id,)
                ).fetchall()
            ]

    def replace_clip_tags(self, clip_id: str, tags: list[dict[str, Any]]) -> list[dict[str, Any]]:
        with self.connect() as conn:
            if conn.execute("SELECT 1 FROM clips WHERE id = ?", (clip_id,)).fetchone() is None:
                raise KeyError(f"unknown video clip: {clip_id}")
            self._replace_clip_tags_in_conn(conn, clip_id, tags, now=_now_ms())
            return self._clip_tags(conn, clip_id)

    def update_clip_materialization(self, clip_id: str, file_path: Path | str) -> dict[str, Any]:
        resolved = Path(file_path).expanduser().resolve(strict=True)
        with self.connect() as conn:
            cursor = conn.execute(
                """
                UPDATE clips
                SET file_path = ?, materialized = 1, updated_at = ?
                WHERE id = ?
                """,
                (str(resolved), _now_ms(), clip_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown video clip: {clip_id}")
            item = _clip_row(conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone())
            assert item is not None
            item["tags"] = self._clip_tags(conn, clip_id)
            return item

    def update_clip_semantic(
        self,
        clip_id: str,
        *,
        confidence: float,
        description: str,
        quality_score: float,
        semantic_json: dict[str, Any],
        status: str,
        tags: list[dict[str, Any]],
    ) -> dict[str, Any]:
        now = _now_ms()
        with self.connect() as conn:
            if conn.execute("SELECT 1 FROM clips WHERE id = ?", (clip_id,)).fetchone() is None:
                raise KeyError(f"unknown video clip: {clip_id}")
            preserved = [tag for tag in self._clip_tags(conn, clip_id) if not str(tag["source"]).startswith("semantic")]
            self._replace_clip_tags_in_conn(conn, clip_id, [*preserved, *tags], now=now)
            conn.execute(
                """
                UPDATE clips
                SET description = ?, semantic_json = ?, quality_score = ?, confidence = ?,
                    status = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    str(description or ""),
                    json.dumps(semantic_json or {}, ensure_ascii=False),
                    max(0.0, min(1.0, float(quality_score))),
                    max(0.0, min(1.0, float(confidence))),
                    str(status or "ready"),
                    now,
                    clip_id,
                ),
            )
            item = _clip_row(conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone())
            assert item is not None
            item["tags"] = self._clip_tags(conn, clip_id)
            return item

    def update_clip_status(self, clip_id: str, status: str) -> dict[str, Any]:
        with self.connect() as conn:
            cursor = conn.execute(
                "UPDATE clips SET status = ?, updated_at = ? WHERE id = ?",
                (str(status), _now_ms(), clip_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown video clip: {clip_id}")
            item = _clip_row(conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone())
            assert item is not None
            item["tags"] = self._clip_tags(conn, clip_id)
            return item

    def _replace_clip_tags_in_conn(
        self,
        conn: sqlite3.Connection,
        clip_id: str,
        tags: list[dict[str, Any]],
        *,
        now: int,
    ) -> None:
        conn.execute("DELETE FROM clip_tags WHERE clip_id = ?", (clip_id,))
        for raw in tags:
            name = re.sub(r"\s+", " ", str(raw.get("name") or "")).strip()
            if not name:
                continue
            tag_id = "tag_" + hashlib.sha256(name.casefold().encode("utf-8")).hexdigest()[:24]
            conn.execute(
                "INSERT OR IGNORE INTO tags (id, name, created_at) VALUES (?, ?, ?)",
                (tag_id, name, now),
            )
            conn.execute(
                """
                INSERT INTO clip_tags (clip_id, tag_id, confidence, source)
                VALUES (?, ?, ?, ?)
                """,
                (
                    clip_id,
                    tag_id,
                    max(0.0, min(1.0, float(raw.get("confidence", 1.0)))),
                    str(raw.get("source") or "manual"),
                ),
            )

    def commit_analysis(
        self,
        asset_id: str,
        job_id: str,
        metadata: dict[str, Any],
        clips: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Commit generated clips, tags, asset metadata, and job state together."""
        now = _now_ms()
        with self.connect() as conn:
            if conn.execute("SELECT 1 FROM assets WHERE id = ?", (asset_id,)).fetchone() is None:
                raise KeyError(f"unknown video asset: {asset_id}")
            if (
                conn.execute(
                    "SELECT 1 FROM analysis_jobs WHERE id = ? AND asset_id = ?",
                    (job_id, asset_id),
                ).fetchone()
                is None
            ):
                raise KeyError(f"unknown analysis job: {job_id}")

            conn.execute("DELETE FROM clips WHERE asset_id = ?", (asset_id,))
            for index, raw in enumerate(clips):
                start = float(raw["start_seconds"])
                end = float(raw["end_seconds"])
                if start < 0 or end <= start:
                    raise ValueError("clip boundaries must have 0 <= start < end")
                clip_id = f"clip_{uuid.uuid4().hex}"
                conn.execute(
                    """
                    INSERT INTO clips (
                        id, asset_id, clip_index, start_seconds, end_seconds,
                        duration_seconds, file_path, keyframe_path, description,
                        status, source_file_path, semantic_json, quality_score,
                        confidence, materialized, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        clip_id,
                        asset_id,
                        index,
                        start,
                        end,
                        round(end - start, 6),
                        str(raw.get("file_path") or ""),
                        str(raw.get("keyframe_path") or ""),
                        str(raw.get("description") or ""),
                        str(raw.get("status") or "ready"),
                        str(raw.get("source_file_path") or ""),
                        json.dumps(raw.get("semantic_json") or {}, ensure_ascii=False),
                        max(0.0, min(1.0, float(raw.get("quality_score", 0.0)))),
                        max(0.0, min(1.0, float(raw.get("confidence", 0.0)))),
                        int(bool(raw.get("materialized", bool(raw.get("file_path"))))),
                        now,
                        now,
                    ),
                )
                self._replace_clip_tags_in_conn(conn, clip_id, list(raw.get("tags") or []), now=now)

            conn.execute(
                """
                UPDATE assets
                SET duration_seconds = ?, width = ?, height = ?, fps = ?,
                    status = 'analyzed', updated_at = ?
                WHERE id = ?
                """,
                (
                    metadata.get("duration_seconds"),
                    metadata.get("width"),
                    metadata.get("height"),
                    metadata.get("fps"),
                    now,
                    asset_id,
                ),
            )
            conn.execute(
                """
                UPDATE analysis_jobs
                SET state = 'complete', progress = 100, error = '', updated_at = ?
                WHERE id = ?
                """,
                (now, job_id),
            )
            result_clips = []
            for row in conn.execute(
                "SELECT * FROM clips WHERE asset_id = ? ORDER BY clip_index",
                (asset_id,),
            ).fetchall():
                item = _clip_row(row)
                assert item is not None
                item["tags"] = self._clip_tags(conn, item["id"])
                result_clips.append(item)
            return {
                "asset": dict(conn.execute("SELECT * FROM assets WHERE id = ?", (asset_id,)).fetchone()),
                "clips": result_clips,
                "job": dict(conn.execute("SELECT * FROM analysis_jobs WHERE id = ?", (job_id,)).fetchone()),
            }

    def _clip_tags(self, conn: sqlite3.Connection, clip_id: str) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in conn.execute(
                """
                SELECT tags.id, tags.name, clip_tags.confidence, clip_tags.source
                FROM clip_tags JOIN tags ON tags.id = clip_tags.tag_id
                WHERE clip_tags.clip_id = ?
                ORDER BY tags.name COLLATE NOCASE
                """,
                (clip_id,),
            ).fetchall()
        ]

    def list_clips(self, *, asset_id: str | None = None, tag: str | None = None) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if asset_id:
            clauses.append("clips.asset_id = ?")
            params.append(asset_id)
        if tag:
            clauses.append(
                "EXISTS (SELECT 1 FROM clip_tags ct JOIN tags t ON t.id = ct.tag_id WHERE ct.clip_id = clips.id AND t.name = ? COLLATE NOCASE)"
            )
            params.append(tag)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT clips.* FROM clips{where} ORDER BY clips.created_at DESC, clips.clip_index",
                params,
            ).fetchall()
            result = []
            for row in rows:
                item = _clip_row(row)
                assert item is not None
                item["tags"] = self._clip_tags(conn, item["id"])
                result.append(item)
            return result

    def create_analysis_job(self, asset_id: str, *, analyzer_version: str) -> dict[str, Any]:
        job_id = f"analysis_{uuid.uuid4().hex}"
        now = _now_ms()
        with self.connect() as conn:
            if conn.execute("SELECT 1 FROM assets WHERE id = ?", (asset_id,)).fetchone() is None:
                raise KeyError(f"unknown video asset: {asset_id}")
            conn.execute(
                """
                INSERT INTO analysis_jobs (
                    id, asset_id, analyzer_version, state, progress, error, created_at, updated_at
                ) VALUES (?, ?, ?, 'queued', 0, '', ?, ?)
                """,
                (job_id, asset_id, analyzer_version, now, now),
            )
            return dict(conn.execute("SELECT * FROM analysis_jobs WHERE id = ?", (job_id,)).fetchone())

    def update_analysis_job(
        self,
        job_id: str,
        *,
        state: str,
        progress: int,
        error: str = "",
        stage: str | None = None,
    ) -> dict[str, Any]:
        with self.connect() as conn:
            if stage is None:
                cursor = conn.execute(
                    """
                    UPDATE analysis_jobs
                    SET state = ?, progress = ?, error = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (state, max(0, min(100, int(progress))), error[:2000], _now_ms(), job_id),
                )
            else:
                cursor = conn.execute(
                    """
                    UPDATE analysis_jobs
                    SET state = ?, progress = ?, error = ?, stage = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (state, max(0, min(100, int(progress))), error[:2000], stage, _now_ms(), job_id),
                )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown analysis job: {job_id}")
            return dict(conn.execute("SELECT * FROM analysis_jobs WHERE id = ?", (job_id,)).fetchone())

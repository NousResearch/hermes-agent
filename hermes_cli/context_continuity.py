"""Local context-continuity ledger and recall index.

The continuation path must be deterministic and local-first: writing a record
or searching old records is useful, but chat recovery must never depend on an
external app or an embedding service.  This module therefore keeps a JSONL
ledger for auditability and a small SQLite FTS5 index for bounded recall.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_hermes_home

MAX_SNIPPET_CHARS = 700
MAX_MARKDOWN_BYTES = 512_000
MAX_INDEX_SECONDS = 2.0
SECRET_PARTS = {
    ".env",
    ".git",
    ".obsidian",
    ".ssh",
    "credentials",
    "node_modules",
    "private",
    "secret",
    "secrets",
}


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_text(value: Any, limit: int = 4_000) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            text = str(value)
    text = text.strip()
    return text if len(text) <= limit else f"{text[:limit].rstrip()}..."


def prompt_hash(prompt: Any) -> str:
    return hashlib.sha256(_safe_text(prompt, 200_000).encode("utf-8")).hexdigest()


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_content_text(part) for part in content)
    if isinstance(content, dict):
        if "text" in content:
            return _safe_text(content.get("text"), 20_000)
        if "content" in content:
            return _safe_text(content.get("content"), 20_000)
    return _safe_text(content, 20_000)


def summarize_recent(messages: Iterable[dict[str, Any]], limit: int = 6) -> str:
    lines: list[str] = []
    for msg in list(messages)[-limit:]:
        role = str(msg.get("role") or "unknown")
        text = _safe_text(_content_text(msg.get("content")), 500)
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


def referenced_files(messages: Iterable[dict[str, Any]], cwd: str = "") -> list[str]:
    seen: dict[str, None] = {}
    pattern = re.compile(r"@file:([^\s)>\]]+)|(?<![\w.-])((?:[./~][^\s:<>|?*]+|/[^\s:<>|?*]+))")
    root = Path(cwd).expanduser() if cwd else None
    for msg in messages:
        text = _content_text(msg.get("content"))
        for match in pattern.finditer(text):
            candidate = (match.group(1) or match.group(2) or "").strip("`'\".,")
            if not candidate:
                continue
            if root and candidate.startswith("./"):
                candidate = str((root / candidate).resolve())
            seen.setdefault(candidate)
            if len(seen) >= 24:
                return list(seen)
    return list(seen)


def _default_paths() -> tuple[Path, Path, Path]:
    base = get_hermes_home() / "continuity"
    return base / "continuity.db", base / "dropoffs.jsonl", base / "obsidian"


@dataclass
class ContinuitySettings:
    obsidian_vault_path: str = ""
    obsidian_allowlisted_folders: list[str] | None = None
    obsidian_read_enabled: bool = False
    obsidian_mirror_enabled: bool = False
    obsidian_last_indexed_at: str = ""

    def normalized_allowlist(self) -> list[str]:
        values = self.obsidian_allowlisted_folders or []
        return [v.strip().strip("/\\") for v in values if isinstance(v, str) and v.strip()]


class ContinuityStore:
    def __init__(self, db_path: Path | None = None, ledger_path: Path | None = None, mirror_dir: Path | None = None):
        default_db, default_ledger, default_mirror = _default_paths()
        self.db_path = db_path or default_db
        self.ledger_path = ledger_path or default_ledger
        self.mirror_dir = mirror_dir or default_mirror
        self.settings_path = self.db_path.parent / "settings.json"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS continuity_records (
                    id TEXT PRIMARY KEY,
                    ts REAL NOT NULL,
                    kind TEXT NOT NULL,
                    source_path TEXT,
                    parent_session_id TEXT,
                    child_session_id TEXT,
                    cwd TEXT,
                    trigger TEXT,
                    prompt_hash TEXT,
                    summary TEXT,
                    payload_json TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS continuity_fts USING fts5(
                    id UNINDEXED,
                    kind UNINDEXED,
                    source_path UNINDEXED,
                    cwd UNINDEXED,
                    body
                );
                """
            )

    def load_settings(self) -> ContinuitySettings:
        try:
            data = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except Exception:
            return ContinuitySettings(obsidian_allowlisted_folders=[])
        return ContinuitySettings(
            obsidian_vault_path=str(data.get("obsidian_vault_path") or ""),
            obsidian_allowlisted_folders=list(data.get("obsidian_allowlisted_folders") or []),
            obsidian_last_indexed_at=str(data.get("obsidian_last_indexed_at") or ""),
            obsidian_read_enabled=bool(data.get("obsidian_read_enabled")),
            obsidian_mirror_enabled=bool(data.get("obsidian_mirror_enabled")),
        )

    def save_settings(self, patch: dict[str, Any]) -> ContinuitySettings:
        current = self.load_settings()
        data = {
            "obsidian_vault_path": current.obsidian_vault_path,
            "obsidian_allowlisted_folders": current.normalized_allowlist(),
            "obsidian_last_indexed_at": current.obsidian_last_indexed_at,
            "obsidian_read_enabled": current.obsidian_read_enabled,
            "obsidian_mirror_enabled": current.obsidian_mirror_enabled,
        }
        data.update({k: v for k, v in patch.items() if k in data})
        if not isinstance(data.get("obsidian_allowlisted_folders"), list):
            data["obsidian_allowlisted_folders"] = []
        self.settings_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        return self.load_settings()

    def record_dropoff(self, record: dict[str, Any]) -> dict[str, Any]:
        record = dict(record)
        record.setdefault("id", f"dropoff-{int(time.time() * 1000)}-{os.getpid()}")
        record.setdefault("timestamp", _now_iso())
        record.setdefault("kind", "dropoff")
        record.setdefault("prompt_hash", prompt_hash(record.get("pending_prompt", "")))
        record.setdefault("summary", _safe_text(record.get("recent_summary"), 2_000))

        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._upsert_record(record)

        if self.load_settings().obsidian_mirror_enabled:
            self._mirror_dropoff(record)

        return record

    def _upsert_record(self, record: dict[str, Any]) -> None:
        rid = str(record.get("id") or "")
        body = "\n".join(
            _safe_text(record.get(key), 4_000)
            for key in (
                "trigger",
                "error",
                "summary",
                "recent_summary",
                "open_tasks",
                "files",
                "decisions",
                "pending_prompt",
            )
        )
        payload = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO continuity_records
                   (id, ts, kind, source_path, parent_session_id, child_session_id,
                    cwd, trigger, prompt_hash, summary, payload_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    rid,
                    time.time(),
                    str(record.get("kind") or "dropoff"),
                    str(record.get("source_path") or ""),
                    str(record.get("parent_session_id") or ""),
                    str(record.get("child_session_id") or ""),
                    str(record.get("cwd") or ""),
                    str(record.get("trigger") or record.get("error") or ""),
                    str(record.get("prompt_hash") or ""),
                    str(record.get("summary") or record.get("recent_summary") or ""),
                    payload,
                ),
            )
            conn.execute("DELETE FROM continuity_fts WHERE id = ?", (rid,))
            conn.execute(
                "INSERT INTO continuity_fts (id, kind, source_path, cwd, body) VALUES (?, ?, ?, ?, ?)",
                (
                    rid,
                    str(record.get("kind") or "dropoff"),
                    str(record.get("source_path") or ""),
                    str(record.get("cwd") or ""),
                    body,
                ),
            )

    def _mirror_dropoff(self, record: dict[str, Any]) -> None:
        self.mirror_dir.mkdir(parents=True, exist_ok=True)
        name = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(record.get("id") or "dropoff")).strip("-")
        path = self.mirror_dir / f"{name}.md"
        body = [
            "---",
            "type: hermes-continuity",
            f"id: {record.get('id')}",
            f"timestamp: {record.get('timestamp')}",
            f"parent_session_id: {record.get('parent_session_id')}",
            f"child_session_id: {record.get('child_session_id')}",
            "---",
            "",
            "# Hermes Continuity Dropoff",
            "",
            f"Trigger: {_safe_text(record.get('trigger') or record.get('error'), 500)}",
            "",
            "## Summary",
            _safe_text(record.get("recent_summary") or record.get("summary"), 2_000),
        ]
        path.write_text("\n".join(body), encoding="utf-8")

    def search(self, query: str, *, cwd: str = "", limit: int = 5, timeout_s: float = 0.5) -> list[dict[str, Any]]:
        q = _fts_query(query)
        if not q:
            return []
        deadline = time.monotonic() + max(0.05, timeout_s)
        try:
            with self._connect() as conn:
                params: list[Any] = [q]
                cwd_clause = ""
                if cwd:
                    cwd_clause = " AND (r.cwd = ? OR r.cwd LIKE ? OR f.cwd = ? OR f.cwd LIKE ?)"
                    params.extend([cwd, f"{cwd}%", cwd, f"{cwd}%"])
                params.append(max(1, min(20, int(limit))))
                rows = conn.execute(
                    f"""SELECT r.id, r.kind, r.source_path, r.cwd, r.payload_json,
                               snippet(continuity_fts, 4, '[', ']', '...', 18) AS snippet,
                               bm25(continuity_fts) AS rank
                        FROM continuity_fts f
                        JOIN continuity_records r ON r.id = f.id
                        WHERE continuity_fts MATCH ?{cwd_clause}
                        ORDER BY rank, r.ts DESC
                        LIMIT ?""",
                    tuple(params),
                ).fetchall()
                if time.monotonic() > deadline:
                    return []
                return [_row_to_hit(row) for row in rows]
        except sqlite3.Error:
            return []

    def status(self) -> dict[str, Any]:
        settings = self.load_settings()
        count = 0
        try:
            with self._connect() as conn:
                count = int(conn.execute("SELECT COUNT(*) FROM continuity_records").fetchone()[0])
        except sqlite3.Error:
            count = 0
        return {
            "enabled": True,
            "ledger_path": str(self.ledger_path),
            "record_count": count,
            "settings": {
                "obsidian_vault_path": settings.obsidian_vault_path,
                "obsidian_allowlisted_folders": settings.normalized_allowlist(),
                "obsidian_last_indexed_at": settings.obsidian_last_indexed_at,
                "obsidian_read_enabled": settings.obsidian_read_enabled,
                "obsidian_mirror_enabled": settings.obsidian_mirror_enabled,
            },
        }

    def index_obsidian(self, settings: ContinuitySettings | None = None, *, timeout_s: float = MAX_INDEX_SECONDS) -> dict[str, Any]:
        settings = settings or self.load_settings()
        if not settings.obsidian_read_enabled:
            return {"enabled": False, "indexed": 0, "skipped": 0}
        vault = Path(settings.obsidian_vault_path).expanduser()
        allow = settings.normalized_allowlist()
        if not vault.is_dir() or not allow:
            return {"enabled": True, "indexed": 0, "skipped": 0, "error": "vault path or allowlist missing"}
        deadline = time.monotonic() + timeout_s
        indexed = 0
        skipped = 0
        for folder in allow:
            root = (vault / folder).resolve()
            try:
                root.relative_to(vault.resolve())
            except ValueError:
                skipped += 1
                continue
            for path in root.rglob("*.md"):
                if time.monotonic() > deadline:
                    return {"enabled": True, "indexed": indexed, "skipped": skipped, "timed_out": True}
                if not _obsidian_path_allowed(path, vault):
                    skipped += 1
                    continue
                try:
                    if path.stat().st_size > MAX_MARKDOWN_BYTES:
                        skipped += 1
                        continue
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    skipped += 1
                    continue
                for block in _markdown_blocks(text):
                    rid = f"obsidian:{hashlib.sha256(str(path).encode()).hexdigest()[:16]}:{block['index']}"
                    record = {
                        "id": rid,
                        "kind": "obsidian",
                        "source_path": str(path),
                        "cwd": "",
                        "trigger": block["heading"],
                        "summary": block["text"][:MAX_SNIPPET_CHARS],
                        "heading": block["heading"],
                        "mtime": path.stat().st_mtime,
                        "checksum": hashlib.sha256(block["text"].encode("utf-8")).hexdigest(),
                    }
                    self._upsert_record(record)
                    indexed += 1
        last_indexed_at = _now_iso()
        self.save_settings({"obsidian_last_indexed_at": last_indexed_at})
        return {"enabled": True, "indexed": indexed, "skipped": skipped, "last_indexed_at": last_indexed_at}


def _row_to_hit(row: sqlite3.Row) -> dict[str, Any]:
    try:
        payload = json.loads(row["payload_json"] or "{}")
    except Exception:
        payload = {}
    return {
        "id": row["id"],
        "kind": row["kind"],
        "source_path": row["source_path"],
        "cwd": row["cwd"],
        "snippet": _safe_text(row["snippet"], MAX_SNIPPET_CHARS),
        "payload": payload,
    }


def _fts_query(query: str) -> str:
    terms = re.findall(r"[\w./:-]+", query or "", flags=re.UNICODE)
    cleaned = [term.replace('"', "").strip() for term in terms[:12]]
    return " OR ".join(f'"{term}"' for term in cleaned if term)


def _obsidian_path_allowed(path: Path, vault: Path) -> bool:
    try:
        rel = path.resolve().relative_to(vault.resolve())
    except ValueError:
        return False
    parts = {p.lower() for p in rel.parts}
    if parts & SECRET_PARTS:
        return False
    return path.suffix.lower() == ".md"


def _markdown_blocks(text: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    heading = "Document"
    buf: list[str] = []
    index = 0

    def flush() -> None:
        nonlocal index, buf
        body = "\n".join(buf).strip()
        if body:
            blocks.append({"heading": heading, "index": index, "text": body[:4_000]})
            index += 1
        buf = []

    for line in text.splitlines():
        if line.startswith("#"):
            flush()
            heading = line.lstrip("#").strip() or "Untitled"
        else:
            buf.append(line)
    flush()
    return blocks[:100]

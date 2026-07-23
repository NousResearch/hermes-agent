"""Durable file-intake manifest and lightweight extractors.

This module sits below platform adapters: adapters cache bytes first, then record
what arrived here.  The manifest is intentionally profile-local and boring
(SQLite + files under HERMES_HOME/file_intake) so uploads survive context
compression/restarts without requiring another service.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import sqlite3
import time
import zipfile
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as _package_version
from pathlib import Path
from typing import Any, Mapping

from packaging.version import InvalidVersion, Version

from archive_inspector import (
    ensure_zip_crc,
    file_intake_zip_limits_from_config,
    inspect_zip,
    read_zip_member,
)
from hermes_constants import get_hermes_dir


SCHEMA_VERSION = 1
_MIN_SAFE_PYPDF_VERSION = Version("6.13.3")
_MAX_SAFE_PYPDF_VERSION = Version("7")


@dataclass(frozen=True)
class IntakeRecord:
    sha256: str
    source_id: int
    duplicate: bool
    cache_path: str
    artifact_dir: str
    status: str
    extracted_text_path: str | None = None
    structure_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "source_id": self.source_id,
            "duplicate": self.duplicate,
            "cache_path": self.cache_path,
            "artifact_dir": self.artifact_dir,
            "status": self.status,
            "extracted_text_path": self.extracted_text_path,
            "structure_path": self.structure_path,
        }


def get_file_intake_dir() -> Path:
    root = get_hermes_dir("file_intake", "file_intake")
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_artifacts_dir() -> Path:
    root = get_file_intake_dir() / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_intake_db_path() -> Path:
    return get_file_intake_dir() / "intake.sqlite"


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or get_intake_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS file_intake_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS intake_files (
            sha256 TEXT PRIMARY KEY,
            original_filename TEXT NOT NULL,
            mime_type TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            first_cache_path TEXT NOT NULL,
            artifact_dir TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'received',
            retention_policy TEXT NOT NULL DEFAULT 'transient',
            first_seen_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            extracted_text_path TEXT,
            structure_path TEXT,
            error TEXT
        );

        CREATE TABLE IF NOT EXISTS intake_sources (
            source_id INTEGER PRIMARY KEY AUTOINCREMENT,
            sha256 TEXT NOT NULL REFERENCES intake_files(sha256) ON DELETE CASCADE,
            source_platform TEXT NOT NULL,
            chat_id TEXT,
            thread_id TEXT,
            message_id TEXT,
            user_id TEXT,
            username TEXT,
            platform_file_id TEXT,
            platform_unique_id TEXT,
            cache_path TEXT NOT NULL,
            received_at REAL NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE INDEX IF NOT EXISTS idx_intake_sources_sha256 ON intake_sources(sha256);
        CREATE INDEX IF NOT EXISTS idx_intake_sources_message ON intake_sources(source_platform, chat_id, thread_id, message_id);
        CREATE INDEX IF NOT EXISTS idx_intake_files_status ON intake_files(status);
        """
    )
    conn.execute(
        "INSERT OR REPLACE INTO file_intake_meta(key, value) VALUES('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()


def sha256_file(path: str | Path) -> tuple[str, int]:
    h = hashlib.sha256()
    total = 0
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
            total += len(chunk)
    return h.hexdigest(), total


def _artifact_dir_for(sha256: str) -> Path:
    # Two-level prefix keeps future directories from becoming huge.
    return get_artifacts_dir() / sha256[:2] / sha256


def record_incoming_file(
    *,
    cache_path: str | Path,
    original_filename: str | None = None,
    mime_type: str | None = None,
    size_bytes: int | None = None,
    source_platform: str,
    chat_id: str | int | None = None,
    thread_id: str | int | None = None,
    message_id: str | int | None = None,
    user_id: str | int | None = None,
    username: str | None = None,
    platform_file_id: str | None = None,
    platform_unique_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> IntakeRecord:
    """Record one inbound cached file and its platform provenance.

    Content identity is keyed by sha256.  Every Telegram/Slack/etc. message gets
    a provenance row even when the bytes are a duplicate, so follow-up prompts can
    resolve "the file I just sent" without re-extracting the same content.
    """

    path = Path(cache_path)
    digest, actual_size = sha256_file(path)
    size = int(size_bytes) if size_bytes is not None else actual_size
    now = time.time()
    artifact_dir = _artifact_dir_for(digest)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(original_filename or path.name or "document").name
    mime = (mime_type or "application/octet-stream").strip() or "application/octet-stream"
    metadata_json = json.dumps(dict(metadata or {}), ensure_ascii=False, sort_keys=True)

    with _connect() as conn:
        existing = conn.execute(
            "SELECT sha256, status, extracted_text_path, structure_path FROM intake_files WHERE sha256=?",
            (digest,),
        ).fetchone()
        duplicate = existing is not None
        if duplicate:
            conn.execute(
                "UPDATE intake_files SET updated_at=? WHERE sha256=?",
                (now, digest),
            )
        else:
            conn.execute(
                """
                INSERT INTO intake_files(
                    sha256, original_filename, mime_type, size_bytes,
                    first_cache_path, artifact_dir, status, first_seen_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'received', ?, ?)
                """,
                (digest, filename, mime, size, str(path), str(artifact_dir), now, now),
            )
        cur = conn.execute(
            """
            INSERT INTO intake_sources(
                sha256, source_platform, chat_id, thread_id, message_id, user_id,
                username, platform_file_id, platform_unique_id, cache_path,
                received_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                digest,
                source_platform,
                _str_or_none(chat_id),
                _str_or_none(thread_id),
                _str_or_none(message_id),
                _str_or_none(user_id),
                username,
                platform_file_id,
                platform_unique_id,
                str(path),
                now,
                metadata_json,
            ),
        )
        conn.commit()
        row = conn.execute(
            "SELECT status, extracted_text_path, structure_path FROM intake_files WHERE sha256=?",
            (digest,),
        ).fetchone()
        return IntakeRecord(
            sha256=digest,
            source_id=int(cur.lastrowid),
            duplicate=duplicate,
            cache_path=str(path),
            artifact_dir=str(artifact_dir),
            status=row["status"],
            extracted_text_path=row["extracted_text_path"],
            structure_path=row["structure_path"],
        )


def _str_or_none(value: str | int | None) -> str | None:
    if value is None:
        return None
    return str(value)




def should_keep_cache_path(path: str | Path) -> bool:
    """Return True when manifest retention says cleanup must keep this blob."""
    raw = str(Path(path))
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT f.status, f.retention_policy
            FROM intake_sources s
            JOIN intake_files f ON f.sha256 = s.sha256
            WHERE s.cache_path=? OR f.first_cache_path=?
            """,
            (raw, raw),
        ).fetchall()
    for row in rows:
        if row["retention_policy"] in {"keep", "linked", "project", "current_session"}:
            return True
        if row["status"] in {"extracted", "routed", "archived"}:
            return True
    return False


def update_file_status(
    sha256: str,
    status: str,
    *,
    extracted_text_path: str | None = None,
    structure_path: str | None = None,
    error: str | None = None,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            UPDATE intake_files
            SET status=?, updated_at=?,
                extracted_text_path=COALESCE(?, extracted_text_path),
                structure_path=COALESCE(?, structure_path),
                error=?
            WHERE sha256=?
            """,
            (status, time.time(), extracted_text_path, structure_path, error, sha256),
        )
        conn.commit()


def get_file_record(sha256: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM intake_files WHERE sha256=?", (sha256,)).fetchone()
        return dict(row) if row else None




def get_file_record_by_cache_path(path: str | Path) -> dict[str, Any] | None:
    raw = str(Path(path))
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT f.*
            FROM intake_sources s
            JOIN intake_files f ON f.sha256 = s.sha256
            WHERE s.cache_path=? OR f.first_cache_path=?
            ORDER BY s.source_id DESC
            LIMIT 1
            """,
            (raw, raw),
        ).fetchone()
        return dict(row) if row else None


def list_recent_files(
    *,
    limit: int = 10,
    source_platform: str | None = None,
    chat_id: str | int | None = None,
    thread_id: str | int | None = None,
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if source_platform is not None:
        clauses.append("s.source_platform=?")
        params.append(source_platform)
    if chat_id is not None:
        clauses.append("s.chat_id=?")
        params.append(str(chat_id))
    if thread_id is not None:
        clauses.append("s.thread_id=?")
        params.append(str(thread_id))
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    sql = f"""
        SELECT
            s.source_id, s.source_platform, s.chat_id, s.thread_id, s.message_id,
            s.received_at, s.cache_path,
            f.sha256, f.original_filename, f.mime_type, f.size_bytes, f.status,
            f.extracted_text_path, f.structure_path, f.artifact_dir, f.retention_policy
        FROM intake_sources s
        JOIN intake_files f ON f.sha256 = s.sha256
        {where}
        ORDER BY s.received_at DESC, s.source_id DESC
        LIMIT ?
    """
    params.append(max(1, min(int(limit), 100)))
    with _connect() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def build_intake_context_note(path: str | Path) -> str | None:
    row = get_file_record_by_cache_path(path)
    if not row:
        return None
    parts = [
        f"intake sha256={row['sha256']}",
        f"status={row['status']}",
        f"size={row['size_bytes']} bytes",
        f"artifact_dir={row['artifact_dir']}",
    ]
    if row.get("extracted_text_path"):
        parts.append(f"extracted_text={row['extracted_text_path']}")
    if row.get("structure_path"):
        parts.append(f"structure={row['structure_path']}")
    return "; ".join(parts)


def build_semantic_file_card(sha256: str, *, summary: str | None = None) -> str:
    row = get_file_record(sha256)
    if not row:
        raise KeyError(f"unknown intake file sha256: {sha256}")
    sources = list_sources(sha256)
    lines = [
        f"# File intake: {row['original_filename']}",
        "",
        "## Identity",
        f"- sha256: `{row['sha256']}`",
        f"- mime_type: `{row['mime_type']}`",
        f"- size_bytes: `{row['size_bytes']}`",
        f"- status: `{row['status']}`",
        f"- retention_policy: `{row['retention_policy']}`",
        f"- artifact_dir: `{row['artifact_dir']}`",
    ]
    if row.get("extracted_text_path"):
        lines.append(f"- extracted_text_path: `{row['extracted_text_path']}`")
    if row.get("structure_path"):
        lines.append(f"- structure_path: `{row['structure_path']}`")
    if summary:
        lines.extend(["", "## Summary", summary.strip()])
    lines.extend(["", "## Sources"])
    for src in sources[:20]:
        lines.append(
            f"- {src['source_platform']} chat={src.get('chat_id')} thread={src.get('thread_id')} "
            f"message={src.get('message_id')} cache_path=`{src.get('cache_path')}`"
        )
    lines.extend(["", "## Routing note", "No automatic Gbrain/Notion writeback was performed. Use this card only when the user explicitly routes the file."])
    return "\n".join(lines)


def list_sources(sha256: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM intake_sources WHERE sha256=? ORDER BY source_id",
            (sha256,),
        ).fetchall()
        return [dict(r) for r in rows]


def extract_if_supported(record: IntakeRecord, *, filename: str = "", mime_type: str = "") -> IntakeRecord:
    """Run the v1 extractor when the file type is supported.

    Currently full extraction is implemented for PPTX because it directly
    supports Tim's current presentation workflow. Unsupported files stay in the
    manifest as `received` rather than failing noisy.
    """

    ext = Path(filename or record.cache_path).suffix.lower()
    try:
        extractor = None
        if ext == ".pptx" or mime_type == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
            extractor = extract_pptx
        elif ext == ".docx" or mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            extractor = extract_docx
        elif ext == ".xlsx" or mime_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            extractor = extract_xlsx
        elif ext == ".zip" or mime_type in {"application/zip", "application/x-zip-compressed"}:
            extractor = inspect_zip_archive
        elif ext == ".pdf" or mime_type == "application/pdf":
            extractor = extract_pdf

        if extractor is not None:
            text_path, structure_path = extractor(record.cache_path, record.artifact_dir)
            update_file_status(
                record.sha256,
                "extracted",
                extracted_text_path=str(text_path),
                structure_path=str(structure_path),
                error=None,
            )
            return IntakeRecord(
                sha256=record.sha256,
                source_id=record.source_id,
                duplicate=record.duplicate,
                cache_path=record.cache_path,
                artifact_dir=record.artifact_dir,
                status="extracted",
                extracted_text_path=str(text_path),
                structure_path=str(structure_path),
            )
    except Exception as exc:
        update_file_status(record.sha256, "failed_extract", error=str(exc))
    return record


def extract_pptx(path: str | Path, artifact_dir: str | Path) -> tuple[Path, Path]:
    """Extract slide text from a PPTX without depending on python-pptx."""

    pptx_path = Path(path)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slides: list[dict[str, Any]] = []
    limits = file_intake_zip_limits_from_config()
    with zipfile.ZipFile(pptx_path) as zf:
        report = inspect_zip(zf, limits)
        slide_names = sorted(
            (n for n in report.names if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
            key=lambda n: int(re.search(r"slide(\d+)\.xml", n).group(1)),
        )
        for idx, name in enumerate(slide_names, start=1):
            xml = read_zip_member(zf, name, max_bytes=limits.max_xml_member_bytes).decode("utf-8", errors="ignore")
            parts = [html.unescape(x) for x in re.findall(r"<a:t>(.*?)</a:t>", xml, flags=re.S)]
            text = " ".join(_clean_xml_text(p) for p in parts if p).strip()
            slides.append({"index": idx, "source": name, "text": text})

    structure = {
        "type": "pptx",
        "source_path": str(pptx_path),
        "slide_count": len(slides),
        "slides": slides,
    }
    structure_path = out_dir / "structure.json"
    text_path = out_dir / "extracted.md"
    structure_path.write_text(json.dumps(structure, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"# Extracted PPTX: {pptx_path.name}", "", f"Slides: {len(slides)}", ""]
    for slide in slides:
        lines.append(f"## Slide {slide['index']}")
        lines.append(slide["text"] or "[no text extracted]")
        lines.append("")
    text_path.write_text("\n".join(lines), encoding="utf-8")
    return text_path, structure_path


def extract_docx(path: str | Path, artifact_dir: str | Path) -> tuple[Path, Path]:
    docx_path = Path(path)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    limits = file_intake_zip_limits_from_config()
    with zipfile.ZipFile(docx_path) as zf:
        inspect_zip(zf, limits)
        xml = read_zip_member(zf, "word/document.xml", max_bytes=limits.max_xml_member_bytes).decode("utf-8", errors="ignore")
    paragraphs = []
    for para_xml in re.findall(r"<w:p[\s>].*?</w:p>", xml, flags=re.S):
        parts = [html.unescape(x) for x in re.findall(r"<w:t[^>]*>(.*?)</w:t>", para_xml, flags=re.S)]
        text = "".join(_clean_xml_text(p) for p in parts).strip()
        if text:
            paragraphs.append(text)
    structure = {"type": "docx", "source_path": str(docx_path), "paragraph_count": len(paragraphs), "paragraphs": paragraphs}
    return _write_text_structure(out_dir, f"Extracted DOCX: {docx_path.name}", paragraphs, structure)


def extract_xlsx(path: str | Path, artifact_dir: str | Path) -> tuple[Path, Path]:
    xlsx_path = Path(path)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    limits = file_intake_zip_limits_from_config()
    with zipfile.ZipFile(xlsx_path) as zf:
        report = inspect_zip(zf, limits)
        shared_strings = []
        if "xl/sharedStrings.xml" in report.names:
            ss = read_zip_member(zf, "xl/sharedStrings.xml", max_bytes=limits.max_xml_member_bytes).decode("utf-8", errors="ignore")
            for item in re.findall(r"<si[\s>].*?</si>", ss, flags=re.S):
                parts = [html.unescape(x) for x in re.findall(r"<t[^>]*>(.*?)</t>", item, flags=re.S)]
                shared_strings.append("".join(_clean_xml_text(p) for p in parts))
        sheet_names = sorted(n for n in report.names if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n))
        sheets = []
        lines = []
        for idx, name in enumerate(sheet_names, start=1):
            xml = read_zip_member(zf, name, max_bytes=limits.max_xml_member_bytes).decode("utf-8", errors="ignore")
            cells = re.findall(r"<c[^>]*(?:t=\"s\")?[^>]*>.*?<v>(.*?)</v>.*?</c>", xml, flags=re.S)
            values = []
            for raw in cells[:200]:
                value = html.unescape(raw.strip())
                try:
                    si = int(value)
                    if 0 <= si < len(shared_strings):
                        value = shared_strings[si]
                except ValueError:
                    pass
                if value:
                    values.append(value)
            sheets.append({"index": idx, "source": name, "sample_values": values[:50]})
            lines.append(f"Sheet {idx}: " + (", ".join(values[:20]) if values else "[no text extracted]"))
    structure = {"type": "xlsx", "source_path": str(xlsx_path), "sheet_count": len(sheets), "sheets": sheets}
    return _write_text_structure(out_dir, f"Extracted XLSX: {xlsx_path.name}", lines, structure)


def inspect_zip_archive(path: str | Path, artifact_dir: str | Path) -> tuple[Path, Path]:
    zip_path = Path(path)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    limits = file_intake_zip_limits_from_config()
    entries = []
    with zipfile.ZipFile(zip_path) as zf:
        report = inspect_zip(zf, limits)
        ensure_zip_crc(zf, report)
        for info in report.members:
            entries.append({"name": info.name, "size": info.file_size, "compressed_size": info.compress_size, "is_dir": info.is_dir})
    structure = {"type": "zip", "source_path": str(zip_path), "entry_count": len(entries), "total_compressed_bytes": report.total_compressed_bytes, "total_uncompressed_bytes": report.total_uncompressed_bytes, "entries": entries}
    lines = [f"{e['name']} ({e['size']} bytes)" for e in entries[:500]]
    return _write_text_structure(out_dir, f"Inspected ZIP: {zip_path.name}", lines, structure)


def extract_pdf(path: str | Path, artifact_dir: str | Path) -> tuple[Path, Path]:
    pdf_path = Path(path)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    error = None
    _ensure_safe_pypdf_version()
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(str(pdf_path))
        for idx, page in enumerate(reader.pages, start=1):
            try:
                pages.append({"index": idx, "text": (page.extract_text() or "").strip()})
            except Exception as exc:  # pragma: no cover - depends on PDF internals
                pages.append({"index": idx, "text": "", "error": str(exc)})
    except Exception as exc:
        error = f"PDF text extraction unavailable or failed: {exc}"
    lines = [p.get("text", "") or "[no text extracted]" for p in pages] if pages else [error or "[no text extracted]"]
    structure = {"type": "pdf", "source_path": str(pdf_path), "page_count": len(pages), "pages": pages, "error": error}
    return _write_text_structure(out_dir, f"Extracted PDF: {pdf_path.name}", lines, structure)


def _ensure_safe_pypdf_version() -> None:
    try:
        installed = _package_version("pypdf")
    except PackageNotFoundError as exc:
        raise RuntimeError("PDF text extraction requires pypdf>=6.13.3,<7") from exc
    try:
        parsed = Version(installed)
    except InvalidVersion as exc:
        raise RuntimeError(f"PDF text extraction disabled: invalid pypdf version {installed!r}") from exc
    if not (_MIN_SAFE_PYPDF_VERSION <= parsed < _MAX_SAFE_PYPDF_VERSION):
        raise RuntimeError(
            f"PDF text extraction disabled: pypdf {installed} is outside the supported security window >=6.13.3,<7"
        )


def _write_text_structure(out_dir: Path, title: str, blocks: list[str], structure: dict[str, Any]) -> tuple[Path, Path]:
    structure_path = out_dir / "structure.json"
    text_path = out_dir / "extracted.md"
    structure_path.write_text(json.dumps(structure, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"# {title}", ""]
    for i, block in enumerate(blocks, start=1):
        lines.append(f"## Item {i}")
        lines.append(block or "[no text extracted]")
        lines.append("")
    text_path.write_text("\n".join(lines), encoding="utf-8")
    return text_path, structure_path


def _clean_xml_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

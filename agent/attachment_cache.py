from __future__ import annotations

import re
import shutil
import uuid
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_dir

IMAGE_CACHE_DIR = get_hermes_dir("cache/images", "image_cache")
DOCUMENT_CACHE_DIR = get_hermes_dir("cache/documents", "document_cache")

TEXT_INJECTABLE_DOCUMENT_EXTENSIONS = {".md", ".txt", ".csv", ".tsv"}
MAX_TEXT_INJECT_BYTES = 100 * 1024
CSV_PREVIEW_LINE_LIMIT = 120
CSV_PREVIEW_CHAR_LIMIT = 16 * 1024


def get_image_cache_dir() -> Path:
    IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return IMAGE_CACHE_DIR


def get_document_cache_dir() -> Path:
    DOCUMENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return DOCUMENT_CACHE_DIR


def cache_image_from_path(source_path: str | Path, ext: str | None = None) -> str:
    source = Path(source_path)
    cache_dir = get_image_cache_dir()
    suffix = (ext or source.suffix or ".jpg").lower()
    filepath = cache_dir / f"img_{uuid.uuid4().hex[:12]}{suffix}"
    with source.open("rb") as src, filepath.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return str(filepath)


def cache_document_from_path(source_path: str | Path, filename: str | None = None) -> str:
    source = Path(source_path)
    cache_dir = get_document_cache_dir()
    safe_name = Path(filename or source.name).name if (filename or source.name) else "document"
    safe_name = safe_name.replace("\x00", "").strip()
    if not safe_name or safe_name in (".", ".."):
        safe_name = "document"
    cached_name = f"doc_{uuid.uuid4().hex[:12]}_{safe_name}"
    filepath = cache_dir / cached_name
    if not filepath.resolve().is_relative_to(cache_dir.resolve()):
        raise ValueError(f"Path traversal rejected: {filename!r}")
    with source.open("rb") as src, filepath.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return str(filepath)


def _sanitize_attachment_display_name(display_name: str, ext: str) -> str:
    return re.sub(r"[^\w.\- ]", "_", display_name or f"document{ext}")


def _build_csv_preview(text_content: str) -> str:
    preview_lines = []
    char_count = 0
    for line in text_content.splitlines():
        if len(preview_lines) >= CSV_PREVIEW_LINE_LIMIT or char_count >= CSV_PREVIEW_CHAR_LIMIT:
            break
        remaining = CSV_PREVIEW_CHAR_LIMIT - char_count
        if remaining <= 0:
            break
        line = line[:remaining]
        preview_lines.append(line)
        char_count += len(line) + 1
    return "\n".join(preview_lines).strip()


def build_text_attachment_injection(raw_bytes: bytes, display_name: str, ext: str) -> Optional[str]:
    ext = (ext or "").lower()
    if ext not in TEXT_INJECTABLE_DOCUMENT_EXTENSIONS:
        return None
    try:
        text_content = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return None
    safe_name = _sanitize_attachment_display_name(display_name, ext)
    should_preview_csv = ext in {".csv", ".tsv"} and len(raw_bytes) > CSV_PREVIEW_CHAR_LIMIT
    if len(raw_bytes) <= MAX_TEXT_INJECT_BYTES and not should_preview_csv:
        return f"[Content of {safe_name}]:\n{text_content}"
    if ext not in {".csv", ".tsv"}:
        return None
    preview = _build_csv_preview(text_content)
    if not preview:
        return None
    return f"[Preview of {safe_name}]:\n{preview}\n\n[The preview is truncated because the original attachment is large.]"


def build_text_attachment_injection_from_path(path: str | Path, *, display_name: str | None = None) -> Optional[str]:
    file_path = Path(path)
    ext = file_path.suffix.lower()
    if ext not in TEXT_INJECTABLE_DOCUMENT_EXTENSIONS:
        return None
    safe_name = _sanitize_attachment_display_name(display_name or file_path.name, ext)
    try:
        file_size = file_path.stat().st_size
    except OSError:
        return None
    should_preview_csv = ext in {".csv", ".tsv"} and file_size > CSV_PREVIEW_CHAR_LIMIT
    if file_size <= MAX_TEXT_INJECT_BYTES and not should_preview_csv:
        try:
            text_content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return None
        return f"[Content of {safe_name}]:\n{text_content}"
    if ext not in {".csv", ".tsv"}:
        return None
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            preview_lines = []
            char_count = 0
            for raw_line in handle:
                if len(preview_lines) >= CSV_PREVIEW_LINE_LIMIT or char_count >= CSV_PREVIEW_CHAR_LIMIT:
                    break
                remaining = CSV_PREVIEW_CHAR_LIMIT - char_count
                if remaining <= 0:
                    break
                line = raw_line.rstrip("\n")[:remaining]
                preview_lines.append(line)
                char_count += len(line) + 1
    except UnicodeDecodeError:
        return None
    preview = "\n".join(preview_lines).strip()
    if not preview:
        return None
    return f"[Preview of {safe_name}]:\n{preview}\n\n[The preview is truncated because the original attachment is large.]"

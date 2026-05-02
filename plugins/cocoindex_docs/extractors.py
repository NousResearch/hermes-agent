"""Document extraction and context helpers for the Hermes knowledge pipeline."""

from __future__ import annotations

import io
import logging
from pathlib import Path, PurePath

from cocoindex.ops.text import RecursiveSplitter, detect_code_language
from pypdf import PdfReader

from config import DOCUMENTS_DIR, LOGGER, MAX_FILE_BYTES


PDF_EXTENSIONS = {".pdf"}
MARKDOWN_EXTENSIONS = {".md", ".mdx"}
TEXT_EXTENSIONS = {".txt"}
CODE_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx"}
JSON_EXTENSIONS = {".json"}


def resolve_paths(path: PurePath) -> tuple[Path, PurePath]:
    """Return absolute and document-relative paths for a LocalFS item."""
    candidate = Path(path)
    if candidate.is_absolute():
        absolute_path = candidate
        try:
            relative_path = absolute_path.relative_to(DOCUMENTS_DIR)
        except ValueError:
            relative_path = PurePath(absolute_path.name)
    else:
        relative_path = PurePath(candidate)
        absolute_path = DOCUMENTS_DIR / candidate
    return absolute_path, relative_path


def determine_file_type(path: PurePath) -> str:
    """Map a path to a stable file type label used in metadata."""
    suffix = path.suffix.lower()
    if suffix in PDF_EXTENSIONS:
        return "pdf"
    if suffix in MARKDOWN_EXTENSIONS:
        return "markdown"
    if suffix in TEXT_EXTENSIONS:
        return "text"
    if suffix in CODE_EXTENSIONS:
        return detect_code_language(filename=path.name) or "code"
    if suffix in JSON_EXTENSIONS:
        return "json"
    return "unknown"


def topic_from_relative_path(relative_path: PurePath) -> str:
    """Extract the top-level folder as topic metadata."""
    parts = relative_path.parts
    if len(parts) > 1:
        return parts[0]
    return "Hauptverzeichnis"


def folder_context_from_relative_path(relative_path: PurePath) -> str:
    """Return the relative folder path for richer retrieval context."""
    parent = relative_path.parent
    if str(parent) in {"", "."}:
        return "Hauptverzeichnis"
    return str(parent)


def build_prefixed_text(
    *,
    relative_path: PurePath,
    topic: str,
    folder_context: str,
    file_type: str,
    chunk_text: str,
) -> str:
    """Inject path and topic metadata into the text sent to the embedder."""
    return (
        f"Quelle: {relative_path}\n"
        f"Thema: {topic}\n"
        f"Ordnerkontext: {folder_context}\n"
        f"Dateityp: {file_type}\n"
        "---\n"
        f"{chunk_text}"
    )


def should_skip_file_size(size_bytes: int) -> bool:
    """Return whether a file is too large for the first-pass indexer."""
    return size_bytes > MAX_FILE_BYTES


def extract_pdf_text(content: bytes) -> str:
    """Extract text from a PDF byte stream using pypdf."""
    reader = PdfReader(io.BytesIO(content))
    pages: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            logging.getLogger(LOGGER.name).warning(
                "Failed to extract PDF page %s: %s", page_number, exc
            )
            text = ""
        if text.strip():
            pages.append(f"[Seite {page_number}]\n{text}")
    return "\n\n".join(pages)


def split_text(
    splitter: RecursiveSplitter,
    *,
    text: str,
    file_type: str,
    filename: str,
) -> list:
    """Split text with syntax-aware settings where CocoIndex supports it."""
    if file_type == "markdown":
        language = "markdown"
        chunk_size = 1800
        min_chunk_size = 300
        chunk_overlap = 250
    elif file_type in {"python", "javascript", "typescript", "tsx", "jsx", "code"}:
        language = detect_code_language(filename=filename)
        chunk_size = 1200
        min_chunk_size = 250
        chunk_overlap = 220
    elif file_type == "json":
        language = "json"
        chunk_size = 1600
        min_chunk_size = 250
        chunk_overlap = 200
    else:
        language = None
        chunk_size = 1800
        min_chunk_size = 300
        chunk_overlap = 250

    try:
        return splitter.split(
            text,
            chunk_size=chunk_size,
            min_chunk_size=min_chunk_size,
            chunk_overlap=chunk_overlap,
            language=language,
        )
    except Exception as exc:
        LOGGER.warning(
            "Syntax-aware splitting failed for %s, falling back to plain text: %s",
            filename,
            exc,
        )
        return splitter.split(
            text,
            chunk_size=1800,
            min_chunk_size=300,
            chunk_overlap=250,
            language=None,
        )

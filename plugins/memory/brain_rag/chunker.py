"""Text chunking for Brain RAG document ingestion."""

from __future__ import annotations

import re
from typing import Iterator, List


def chunk_text(
    text: str,
    *,
    chunk_size: int = 800,
    overlap: int = 120,
) -> List[str]:
    """Split *text* into overlapping chunks suitable for retrieval."""
    text = (text or "").strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    paragraphs = re.split(r"\n\s*\n", text)
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.extend(_split_long(current, chunk_size=chunk_size, overlap=overlap))
        current = para

    if current:
        chunks.extend(_split_long(current, chunk_size=chunk_size, overlap=overlap))

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: List[str] = []
    for c in chunks:
        c = c.strip()
        if c and c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def _split_long(text: str, *, chunk_size: int, overlap: int) -> Iterator[str]:
    """Split a long block with character overlap."""
    start = 0
    n = len(text)
    step = max(1, chunk_size - overlap)
    while start < n:
        end = min(n, start + chunk_size)
        piece = text[start:end].strip()
        if piece:
            yield piece
        if end >= n:
            break
        start += step

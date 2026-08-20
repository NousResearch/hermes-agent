"""Chunking + lightweight embeddings (stdlib only).

Used by local/offline providers and by the sync pipeline. Deliberately
dependency-free: a hashed bag-of-words vector with cosine similarity is
good enough for local semantic-ish retrieval and keeps the subsystem
installable everywhere. Real providers use their own embedding models.
"""
from __future__ import annotations

import math
import re
import zlib
from typing import List

_WORD = re.compile(r"[A-Za-z0-9_]+")
_DIM = 512

_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "it", "for", "on",
    "with", "that", "this", "as", "are", "be", "by", "at", "from", "was",
    "what", "do", "i", "you", "my", "about", "have", "has",
}


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD.findall(text or "") if t.lower() not in _STOP]


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> List[str]:
    """Split text into overlapping chunks on paragraph-ish boundaries."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text.rfind("\n\n", start + int(size * 0.5), end)
            if window == -1:
                window = text.rfind("\n", start + int(size * 0.5), end)
            if window == -1:
                window = text.rfind(". ", start + int(size * 0.5), end)
            if window != -1:
                end = window + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def embed(text: str, dim: int = _DIM) -> List[float]:
    """Hashed TF vector, L2-normalised."""
    vec = [0.0] * dim
    toks = tokenize(text)
    if not toks:
        return vec
    for t in toks:
        # zlib.crc32 is stable across processes (unlike hash(), which is
        # PYTHONHASHSEED-salted) — required because vectors are persisted.
        vec[zlib.crc32(t.encode("utf-8")) % dim] += 1.0
    # sublinear scaling damps repeated terms
    vec = [math.log1p(v) for v in vec]
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def keyword_overlap(query: str, text: str) -> float:
    q = set(tokenize(query))
    if not q:
        return 0.0
    t = set(tokenize(text))
    return len(q & t) / len(q)

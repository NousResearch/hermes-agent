"""Lightweight local embeddings for Brain RAG (no external API required)."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Dict, Iterable, List


_TOKEN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def hash_embed(text: str, dim: int = 256) -> List[float]:
    """Deterministic sparse hash embedding — fast and fully local."""
    vec = [0.0] * dim
    tokens = tokenize(text)
    if not tokens:
        return vec
    for tok in tokens:
        digest = hashlib.blake2b(tok.encode(), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    return _l2_normalize(vec)


def tfidf_embed(text: str, idf: Dict[str, float], dim: int = 256) -> List[float]:
    """TF-IDF weighted hash projection for richer semantic-ish signals."""
    tokens = tokenize(text)
    if not tokens:
        return [0.0] * dim
    tf: Dict[str, float] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0.0) + 1.0
    n = float(len(tokens))
    vec = [0.0] * dim
    for tok, count in tf.items():
        weight = (count / n) * idf.get(tok, 1.0)
        digest = hashlib.blake2b(tok.encode(), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign * weight
    return _l2_normalize(vec)


def build_idf(corpus: Iterable[str]) -> Dict[str, float]:
    """Build IDF table from a corpus of text chunks."""
    doc_freq: Dict[str, int] = {}
    n_docs = 0
    for doc in corpus:
        toks = set(tokenize(doc))
        if not toks:
            continue
        n_docs += 1
        for t in toks:
            doc_freq[t] = doc_freq.get(t, 0) + 1
    if n_docs == 0:
        return {}
    return {t: math.log((1.0 + n_docs) / (1.0 + df)) + 1.0 for t, df in doc_freq.items()}


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return max(0.0, min(1.0, dot))  # vectors are L2-normalized


def _l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm <= 1e-12:
        return vec
    return [x / norm for x in vec]

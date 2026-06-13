"""BM25 local index + RRF fusion for mem0 plugin.

Adds keyword-based retrieval as a complement to mem0's vector search.
When both return results, Reciprocal Rank Fusion (RRF) merges them.
When vector search fails, BM25 acts as a fallback.

Design: agentmemory-inspired, adapted for self-hosted mem0.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_CACHE_DIR = Path.home() / ".hermes" / "state"
_CACHE_FILE = _CACHE_DIR / "mem0_bm25_cache.json"
_CACHE_MAX_AGE_HOURS = 24

# CJK character range for tokenization
_CJK_RE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')


def _tokenize(text: str) -> List[str]:
    """Tokenize text for BM25.

    - English: split on non-alphanumeric, lowercase, remove stopwords
    - Chinese: character-level bigrams (no jieba dependency)
    """
    if not text:
        return []

    tokens = []

    # CJK bigrams
    cjk_chars = _CJK_RE.findall(text)
    if cjk_chars:
        cjk_str = ''.join(cjk_chars)
        # Unigrams + bigrams for better recall
        tokens.extend(cjk_str)
        for i in range(len(cjk_str) - 1):
            tokens.append(cjk_str[i:i+2])

    # English/numeric tokens
    en_tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
    # Remove very short tokens and common stopwords
    stopwords = {'the', 'is', 'at', 'of', 'on', 'a', 'an', 'and', 'or', 'but',
                 'in', 'with', 'to', 'for', 'not', 'this', 'that', 'it', 'was'}
    tokens.extend(t for t in en_tokens if len(t) > 1 and t not in stopwords)

    return tokens


def _fingerprint(text: str) -> str:
    """SHA-256 fingerprint for dedup."""
    return hashlib.sha256(text.encode('utf-8', errors='replace')).hexdigest()[:16]


class BM25Index:
    """Local BM25 index backed by a disk cache.

    Thread-safe via the caller's locks. Rebuilds lazily.
    """

    def __init__(self):
        self._index = None  # rank_bm25.BM25Okapi instance
        self._docs: List[Dict[str, Any]] = []  # [{id, memory, score, ...}]
        self._fingerprints: set = set()
        self._built_at: float = 0
        self._available = False
        try:
            from rank_bm25 import BM25Okapi  # noqa: F401
            self._available = True
        except ImportError:
            logger.warning("rank_bm25 not installed — BM25 search disabled. pip install rank_bm25")

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def doc_count(self) -> int:
        return len(self._docs)

    def load_cache(self) -> bool:
        """Load BM25 cache from disk. Returns True if loaded successfully."""
        if not self._available:
            return False
        if not _CACHE_FILE.exists():
            return False
        try:
            data = json.loads(_CACHE_FILE.read_text(encoding='utf-8'))
            if time.time() - data.get('built_at', 0) > _CACHE_MAX_AGE_HOURS * 3600:
                logger.info("BM25 cache expired (age: %.1fh)", 
                           (time.time() - data.get('built_at', 0)) / 3600)
                return False
            self._docs = data.get('documents', [])
            self._fingerprints = set(data.get('fingerprints', []))
            self._rebuild_index()
            logger.info("BM25 cache loaded: %d docs", len(self._docs))
            return True
        except Exception as e:
            logger.debug("BM25 cache load failed: %s", e)
            return False

    def save_cache(self):
        """Persist BM25 index to disk."""
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            'built_at': self._built_at,
            'documents': self._docs,
            'fingerprints': list(self._fingerprints),
        }
        try:
            _CACHE_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
            logger.debug("BM25 cache saved: %d docs", len(self._docs))
        except Exception as e:
            logger.debug("BM25 cache save failed: %s", e)

    def build_from_memories(self, memories: List[Dict[str, Any]]):
        """Build BM25 index from a list of memory dicts (from mem0 get_all)."""
        if not self._available:
            return
        self._docs = []
        self._fingerprints = set()
        for m in memories:
            text = m.get('memory', m.get('data', ''))
            if not text:
                continue
            fp = _fingerprint(text)
            if fp in self._fingerprints:
                continue
            self._fingerprints.add(fp)
            self._docs.append({
                'id': m.get('id', ''),
                'memory': text,
                'created_at': m.get('created_at', ''),
                'score': 0,
            })
        self._rebuild_index()
        self._built_at = time.time()
        self.save_cache()
        logger.info("BM25 index built: %d docs from %d memories",
                    len(self._docs), len(memories))

    def add_document(self, doc_id: str, text: str, created_at: str = ''):
        """Add a single document to the index (for incremental updates)."""
        if not self._available:
            return
        fp = _fingerprint(text)
        if fp in self._fingerprints:
            return
        self._fingerprints.add(fp)
        self._docs.append({
            'id': doc_id,
            'memory': text,
            'created_at': created_at,
            'score': 0,
        })
        self._rebuild_index()

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """BM25 search. Returns list of {id, memory, score} dicts."""
        if not self._available or self._index is None or not self._docs:
            return []
        from rank_bm25 import BM25Okapi  # already checked in __init__
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        scores = self._index.get_scores(query_tokens)
        # Get top-k indices sorted by score descending
        scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for idx, score in scored:
            if score <= 0:
                break
            doc = self._docs[idx]
            results.append({
                'id': doc['id'],
                'memory': doc['memory'],
                'score': round(float(score), 4),
                'created_at': doc.get('created_at', ''),
                'source': 'bm25',
            })
        return results

    def _rebuild_index(self):
        """Rebuild the BM25Okapi index from self._docs."""
        if not self._available or not self._docs:
            self._index = None
            return
        from rank_bm25 import BM25Okapi
        corpus = [_tokenize(d['memory']) for d in self._docs]
        # Filter out empty token lists
        valid = [(i, tokens) for i, tokens in enumerate(corpus) if tokens]
        if not valid:
            self._index = None
            return
        valid_corpus = [tokens for _, tokens in valid]
        self._index = BM25Okapi(valid_corpus, k1=1.5, b=0.75)
        # Remap _docs to only valid entries
        self._docs = [self._docs[i] for i, _ in valid]


def rrf_fuse(
    vector_results: List[Dict],
    bm25_results: List[Dict],
    top_k: int = 10,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> List[Dict]:
    """Reciprocal Rank Fusion of vector and BM25 results.

    Formula: score(d) = sum( weight_i / (k + rank_i(d)) )
    where k=60 (standard RRF constant).

    Deduplicates by memory text fingerprint.
    """
    k = 60
    scores: Dict[str, Tuple[float, Dict]] = {}  # fingerprint -> (score, best_result)

    for rank, r in enumerate(vector_results):
        text = r.get('memory', r.get('data', ''))
        if not text:
            continue
        fp = _fingerprint(text)
        rrf_score = vector_weight / (k + rank + 1)
        if fp in scores:
            scores[fp] = (scores[fp][0] + rrf_score, scores[fp][1])
        else:
            item = dict(r)
            item['source'] = 'vector'
            scores[fp] = (rrf_score, item)

    for rank, r in enumerate(bm25_results):
        text = r.get('memory', r.get('data', ''))
        if not text:
            continue
        fp = _fingerprint(text)
        rrf_score = bm25_weight / (k + rank + 1)
        if fp in scores:
            existing_score, existing_item = scores[fp]
            existing_item['source'] = 'hybrid'  # Found by both
            scores[fp] = (existing_score + rrf_score, existing_item)
        else:
            item = dict(r)
            item['source'] = 'bm25'
            scores[fp] = (rrf_score, item)

    # Sort by RRF score descending
    sorted_results = sorted(scores.values(), key=lambda x: x[0], reverse=True)
    fused = []
    for rrf_score, item in sorted_results[:top_k]:
        item['rrf_score'] = round(rrf_score, 6)
        fused.append(item)

    return fused

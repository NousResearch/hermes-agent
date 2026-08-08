"""CJK-aware tokenization shared by every text path in the holographic store.

SQLite FTS5's default ``unicode61`` tokenizer treats a contiguous CJK run as
a single token — it never segments Chinese/Japanese words. A store whose
facts are written in Chinese therefore gets ~zero recall: the indexed token
is the whole clause, and a query only matches if it contains the identical
clause. (Even embedded ASCII words like "mihomo" can't match when they sit
inside an unspaced CJK run.)

Fix: pre-segment CJK text with jieba on BOTH sides of the index —
``segment_for_index()`` at write time (the FTS table stores space-joined
tokens) and ``tokenize()`` at query time (sanitizer, Jaccard rerank, HRR
encoding). unicode61 then sees space-delimited words and behaves exactly as
it does for English.

jieba is optional. Without it, everything degrades to the historical
whitespace behavior — and the store records which mode built the index
(``PRAGMA user_version``), so installing/removing jieba later triggers a
transparent index rebuild on next open (see MemoryStore._ensure_fts_schema).
"""

from __future__ import annotations

import re

try:
    import jieba

    # 60 = above CRITICAL: silence "Building prefix dict" chatter on stderr.
    jieba.setLogLevel(60)
    _HAS_JIEBA = True
except ImportError:
    jieba = None  # type: ignore[assignment]
    _HAS_JIEBA = False

# CJK Unified Ideographs + Extension A, Hiragana/Katakana, Hangul.
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")

# Tokens that are pure punctuation/whitespace (jieba emits punctuation as
# standalone tokens). \w is unicode-aware in py3, so CJK chars are \w and
# never match this.
_PUNCT_ONLY_RE = re.compile(r"^[^\w]+$")

# FTS index modes recorded in PRAGMA user_version. 0 = legacy schema
# (external-content FTS + triggers, unsegmented).
FTS_MODE_LEGACY = 0
FTS_MODE_PLAIN = 2      # standalone FTS table, whitespace tokens (no jieba)
FTS_MODE_SEGMENTED = 3  # standalone FTS table, jieba-segmented tokens


def has_cjk(text: str) -> bool:
    """True when *text* contains at least one CJK character."""
    return bool(text) and _CJK_RE.search(text) is not None


def current_fts_mode() -> int:
    """The index mode this process would build right now."""
    return FTS_MODE_SEGMENTED if _HAS_JIEBA else FTS_MODE_PLAIN


def tokenize(text: str) -> list[str]:
    """Split *text* into retrieval tokens.

    Non-CJK text (or any text when jieba is unavailable) uses plain
    whitespace splitting — byte-for-byte the historical behavior, so
    English-only stores are unaffected. CJK text goes through
    ``jieba.cut_for_search`` (fine-grained mode: emits both words and
    their sub-words, maximizing recall), which also yields embedded
    ASCII words ("mihomo", "RustDesk") as standalone tokens.
    """
    if not text:
        return []
    if not (_HAS_JIEBA and has_cjk(text)):
        return text.split()
    tokens: list[str] = []
    for tok in jieba.cut_for_search(text):
        tok = tok.strip()
        if not tok or _PUNCT_ONLY_RE.match(tok):
            continue
        tokens.append(tok)
    return tokens


def segment_for_index(text: str) -> str:
    """Space-join tokens for FTS indexing.

    Pure-ASCII text is returned unchanged (unicode61 already handles it);
    CJK text becomes a space-delimited token stream so unicode61 indexes
    real words instead of whole clauses.
    """
    if not text:
        return text
    if not (_HAS_JIEBA and has_cjk(text)):
        return text
    return " ".join(tokenize(text))

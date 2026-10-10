"""Shared CJK-aware tokenizer for the holographic fact store.

SQLite's default ``unicode61`` FTS5 tokenizer treats a run of CJK codepoints (no spaces)
as ONE token, so both the indexed content and a natural-language Chinese/Korean/Thai
query collapse to a single long token that never matches (#85524, #83593, #73868).
This module expands CJK runs into overlapping 2-grams on BOTH the index side
(``facts.search_text``, kept in sync by triggers) and the query side
(``FactRetriever._sanitize_fts_query``), so a 2+ char CJK substring anywhere in the
query matches the same bigrams stored with the fact. Latin/digit runs pass through
unchanged, keeping English behavior byte-identical.
"""

from __future__ import annotations

import re

# A "run" is either a maximal CJK sequence or a maximal latin/alnum/._ sequence.
_RUN_RE = re.compile(r"[\u2e80-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]+|[A-Za-z0-9_.]+")
_CJK_RANGES = (("\u2e80", "\u9fff"), ("\uf900", "\ufaff"), ("\u3040", "\u30ff"), ("\uac00", "\ud7af"))


def _is_cjk_char(ch: str) -> bool:
    return any(lo <= ch <= hi for lo, hi in _CJK_RANGES)


def tokenize_for_index(text: str) -> list[str]:
    """Text -> indexable tokens: CJK runs become overlapping 2-grams (single CJK chars
    pass through), latin runs stay whole. Use for both ``search_text`` generation and
    query expansion — the two MUST share this function or bigrams never line up."""
    if not text:
        return []
    tokens: list[str] = []
    for run in _RUN_RE.findall(text.lower()):
        if run and _is_cjk_char(run[0]):
            if len(run) == 1:
                tokens.append(run)
            else:
                tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
        else:
            tokens.append(run)
    return tokens


def build_search_text(content: str, tags: str) -> str:
    """Space-joined index token soup for the ``search_text`` FTS column."""
    return " ".join(tokenize_for_index(content) + tokenize_for_index(tags or ""))

# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------------
# Portions of this file are adapted from BaiLongma
#   Upstream: https://github.com/xiaoyuanda666-ship-it/BaiLongma
#   Original: src/memory/keywords.js
#   Copyright (c) 2026 xiaoyuanda666-ship-it — Licensed under MIT
#   License text: see LICENSES/BaiLongma-MIT.txt
# ---------------------------------------------------------------------------
"""Zero-dependency keyword extraction for Chinese + English text.

Used by :mod:`agent.threads` to score message-to-thread overlap without
pulling in a heavy NLP stack. Pure function; no DB, no network, no
runtime state.

The extractor emits Chinese n-grams (length 2..4) plus English tokens
of length ≥ 3, with a small stop-word/stop-char curriculum that culls
the recurring false positives observed in production: single-syllable
particles, quantifier-lead n-grams, and boundary-straddling fragments.
Length weighting biases the ranking toward two-character words because
they hit real-content FTS/LIKE recalls more reliably than 4-grams.
"""
from __future__ import annotations

import re
from typing import List


# Stop-words: high-frequency, low-information tokens.
STOP_WORDS = frozenset(
    [
        "的", "了", "是", "在", "我", "你", "他", "她", "它",
        "我们", "你们", "他们", "这", "那", "有", "没有",
        "和", "与", "把", "被", "因为", "所以", "如果",
        "一个", "一些", "什么", "怎么", "为什么",
        "帮我", "请", "好的", "明白", "告诉", "让", "做",
        "去", "来", "说", "给",
        # Relative-time words: temporal recall handles date windows
        # elsewhere; treating them as literal FTS keywords would recall
        # every past record that happened to contain the string
        # "yesterday", which is not what the user meant.
        "今天", "昨天", "前天", "大前天", "今早", "今晨", "今夜",
        "今晚", "昨晚", "昨夜", "昨日", "今日",
    ]
)

# Chars that indicate an n-gram straddled a word boundary — such an
# n-gram is a boundary fragment, not a real word, so drop it.
STOP_CHARS = frozenset(
    [
        "的", "了",
        "着", "过", "起", "来", "去",
        "吗", "呢", "吧", "啊", "呀", "嘛", "哦",
        "和", "与", "跟", "或", "及", "并",
        "很", "太", "再", "又", "也", "都", "还", "只", "就", "才",
    ]
)

# Forbidden as the first char of an n-gram: single-char classifiers
# should never open a word (otherwise we cut pseudo-words like "个项目").
STOP_HEAD_CHARS = frozenset(["们", "个", "些", "点", "次", "件", "种", "样"])

# Forbidden as the last char of an n-gram: demonstratives / time
# prefixes never *end* a real word (otherwise "成今/项目这" leak in).
STOP_TAIL_CHARS = frozenset(["一", "几", "某", "每", "这", "那", "今"])

_PUNCTUATION_RE = re.compile(r"[，。！？、；：”””’’’【】\[\]()（）\d]")
_WHITESPACE_RE = re.compile(r"\s+")
_ALPHA_RE = re.compile(r"[a-zA-Z]+")
_ENGLISH_WORD_RE = re.compile(r"[a-zA-Z]{3,}")


def _has_invalid_duplicate(word: str) -> bool:
    """N-gram is bad if it contains a repeated char *except* legit
    two-char reduplications like 天天/常常.
    """
    if len(word) == 2:
        return False
    seen: set[str] = set()
    for ch in word:
        if ch in seen:
            return True
        seen.add(ch)
    return False


def _is_valid_ngram(word: str) -> bool:
    if not word or len(word) < 2 or word in STOP_WORDS:
        return False
    for ch in word:
        if ch in STOP_CHARS:
            return False
    if word[0] in STOP_HEAD_CHARS:
        return False
    if word[-1] in STOP_TAIL_CHARS:
        return False
    if _has_invalid_duplicate(word):
        return False
    return True


def _length_weight(length: int) -> float:
    """Shorter words hit real content more often — up-weight 2-grams,
    down-weight 4-grams. Longer n-grams are more likely to straddle a
    word boundary.
    """
    if length == 2:
        return 1.5
    if length == 4:
        return 0.8
    return 1.0


def _extract_core(text: str) -> tuple[dict[str, int], list[str]]:
    if not text:
        return {}, []
    cleaned = _PUNCTUATION_RE.sub(" ", text)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()

    freq: dict[str, int] = {}
    raw_ngrams: list[str] = []

    def bump_chinese(word: str) -> None:
        if not word:
            return
        raw_ngrams.append(word)
        if not _is_valid_ngram(word):
            return
        freq[word] = freq.get(word, 0) + 1

    def bump_english(word: str) -> None:
        if not word or len(word) < 2 or word in STOP_WORDS:
            return
        freq[word] = freq.get(word, 0) + 1

    chinese = _ALPHA_RE.sub(" ", cleaned)
    n = len(chinese)
    for i in range(n - 1):
        for length in range(2, 5):
            if i + length > n:
                break
            bump_chinese(chinese[i : i + length].strip())

    for match in _ENGLISH_WORD_RE.findall(text):
        normalized = match.lower()
        if normalized not in STOP_WORDS:
            bump_english(match)

    return freq, raw_ngrams


def extract_keywords(text: str, max_keywords: int = 8) -> List[str]:
    """Rank tokens by ``freq * length_weight`` then length, return top-N.

    No substring dedup — historically it looked appealing to drop
    shorter tokens covered by longer n-grams, but in production it cut
    exactly the two-char keywords ("业余") that hit real records more
    reliably than four-grams like "业余写什".
    """
    freq, _ = _extract_core(text)
    ranked = sorted(
        ((word, count * _length_weight(len(word)), count) for word, count in freq.items()),
        key=lambda x: (-x[1], -len(x[0])),
    )
    return [word for word, _weighted, _raw in ranked[:max_keywords]]


def extract_keywords_debug(text: str, max_keywords: int = 8) -> dict:
    """Diagnostic helper: raw ngrams, kept-after-filter, and final.

    Useful in tests to assert "pseudo-word X was dropped at the
    filter stage" without depending on the ranked output.
    """
    freq, raw = _extract_core(text)
    return {
        "raw": raw,
        "filtered": list(freq.keys()),
        "final": extract_keywords(text, max_keywords),
    }


__all__ = ["extract_keywords", "extract_keywords_debug"]

"""Feature extraction for learned model routing.

Pure functions — no ML dependencies. Extracts numeric features from
a user message for use by the routing classifier.
"""

from __future__ import annotations

import re
from typing import Dict

_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_CODE_BLOCK_RE = re.compile(r"```")
_BACKTICK_RE = re.compile(r"`[^`]+`")

_TOOL_KEYWORDS = {
    "web_search", "web_extract", "file_read", "file_write",
    "terminal", "shell", "python", "execute", "run",
    "search", "browse", "fetch", "download",
    "yfinance", "web_extract",
}

_TICKER_RE = re.compile(
    r"\b[A-Z]{2,}\.(?:NS|BO)\b|"
    r"\^(?:NSEI|BSESN|NSEBANK|CNXIT|CNXPHARMA|INDIAVIX)\b|"
    r"\b(?:NIFTY|SENSEX|BANK\s*NIFTY)\b",
    re.IGNORECASE,
)

_FINANCE_TERMS = {
    "nifty", "sensex", "nse", "bse", "fii", "dii", "rbi", "sebi",
    "pe", "pb", "eps", "roe", "rsi", "macd", "sma", "ema",
    "cpi", "wpi", "gdp", "repo", "inflation", "crude",
    "bullish", "bearish", "support", "resistance",
}

_COMPLEX_KEYWORDS = {
    "debug", "debugging", "implement", "implementation", "refactor",
    "patch", "traceback", "stacktrace", "exception", "error",
    "analyze", "analysis", "investigate", "architecture", "design",
    "compare", "benchmark", "optimize", "optimise", "review",
    "terminal", "shell", "tool", "tools", "pytest", "test", "tests",
    "plan", "planning", "delegate", "subagent", "cron", "docker",
    "kubernetes",
}


def extract_features(message: str, conversation_depth: int = 0) -> Dict[str, float]:
    """Extract a feature vector from a user message for routing classification.

    Returns a dict of numeric features suitable for sklearn or similar.
    All values are numeric (int/float). Feature names are stable across versions
    so that trained models remain compatible.
    """
    text = (message or "").strip()
    if not text:
        return _empty_features()

    words = text.split()
    lowered = text.lower()
    word_set = {token.strip(".,:;!?()[]{}\"'`") for token in lowered.split()}

    char_count = len(text)
    word_count = len(words)
    line_count = text.count("\n") + 1

    has_code_block = 1.0 if _CODE_BLOCK_RE.search(text) else 0.0
    has_url = 1.0 if _URL_RE.search(text) else 0.0
    has_backtick = 1.0 if _BACKTICK_RE.search(text) else 0.0

    complex_keyword_count = len(word_set & _COMPLEX_KEYWORDS)
    tool_mention_count = len(word_set & _TOOL_KEYWORDS)

    question_mark_present = 1.0 if "?" in text else 0.0

    total_word_len = sum(len(w) for w in words)
    avg_word_length = total_word_len / word_count if word_count > 0 else 0.0

    ticker_mention_count = float(len(_TICKER_RE.findall(text)))
    finance_term_count = float(len(word_set & _FINANCE_TERMS))

    return {
        "char_count": float(char_count),
        "word_count": float(word_count),
        "line_count": float(line_count),
        "has_code_block": has_code_block,
        "has_url": has_url,
        "has_backtick": has_backtick,
        "complex_keyword_count": float(complex_keyword_count),
        "tool_mention_count": float(tool_mention_count),
        "question_mark_present": question_mark_present,
        "avg_word_length": avg_word_length,
        "conversation_depth": float(conversation_depth),
        "ticker_mention_count": ticker_mention_count,
        "finance_term_count": finance_term_count,
    }


def feature_names() -> list[str]:
    """Return ordered list of feature names (matches extract_features keys)."""
    return [
        "char_count", "word_count", "line_count",
        "has_code_block", "has_url", "has_backtick",
        "complex_keyword_count", "tool_mention_count",
        "question_mark_present", "avg_word_length",
        "conversation_depth",
        "ticker_mention_count", "finance_term_count",
    ]


def features_to_array(features: Dict[str, float]) -> list[float]:
    """Convert feature dict to ordered array for sklearn."""
    return [features[name] for name in feature_names()]


def _empty_features() -> Dict[str, float]:
    return {name: 0.0 for name in feature_names()}

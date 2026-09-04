"""Deterministic triage engine — port of pi-smart-router triage/.

Pipeline: sanitize (adversarial-inflation stripping) → entropy tail check →
Aho-Corasick keyword scan → cyclomatic scan → verdict.

Verdicts: obvious-trivial → economical fast path, obvious-complex → frontier
fast path, otherwise ambiguous for deeper pipeline stages.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

VERDICT_TRIVIAL = "trivial"
VERDICT_COMPLEX = "complex"
VERDICT_AMBIGUOUS = "ambiguous"

CYCLOMATIC_THRESHOLD = 15


@dataclass(frozen=True)
class TriageResult:
    verdict: str
    reason_code: str
    trivial_hits: int = 0
    complex_hits: int = 0
    cyclomatic_score: int = 0
    sanitized_length_delta: int = 0
    entropy_score: float = 0.0
    entropy_tail_delta: float = 0.0
    entropy_tail_stripped_length: int = 0


# ─── Adversarial sanitization ─────────────────────────────────────────────────

_RE_BASE64_BLOCK = re.compile(r"[A-Za-z0-9+/=]{64,}")
_RE_HEX_BLOCK = re.compile(r"(?:0x)?[0-9a-fA-F]{32,}")
_RE_HTML_TAGS = re.compile(r"</?[a-z][^>]*>", re.I)
_RE_HTML_COMMENT = re.compile(r"<!--[\s\S]*?-->")
_RE_URL_ENCODED = re.compile(r"%[0-9A-Fa-f]{2}")
_RE_REPEATED_CHARS = re.compile(r"(.)\1{4,}")
_RE_MULTI_HSPACE = re.compile(r"[^\S\r\n]{2,}")
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")


def sanitize(raw: str) -> str:
    """Strip adversarial complexity-inflation patterns from prompt text."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = _RE_BASE64_BLOCK.sub(" ", text)
    text = _RE_HEX_BLOCK.sub(" ", text)
    text = _RE_HTML_COMMENT.sub(" ", text)
    text = _RE_HTML_TAGS.sub(" ", text)
    text = _RE_URL_ENCODED.sub("", text)
    text = _RE_REPEATED_CHARS.sub(r"\1\1", text)
    text = _RE_MULTI_HSPACE.sub(" ", text)
    text = _RE_MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


# ─── Entropy tail check (port of triage/entropy-check.ts) ────────────────────

DEFAULT_TAIL_WINDOW_TOKENS = 32
MIN_TAIL_TOKENS = 8
ENTROPY_DELTA_THRESHOLD = 0.35
ABSOLUTE_TAIL_ENTROPY_THRESHOLD = 0.55
GIBBERISH_TAIL_RATIO_THRESHOLD = 0.45
MIN_PREFIX_TOKENS = 4
MIN_PROMPT_TOKENS = MIN_PREFIX_TOKENS + MIN_TAIL_TOKENS

_COMMON_SHORT_TOKENS = frozenset(
    "a i an as at be by do go if in is it me my no of on or so to up us we".split()
)
_RE_DIGIT = re.compile(r"\d")
_RE_SYMBOL_ONLY = re.compile(r"^[^\w]+$", re.U)


def _tokenize(text: str) -> list:
    return text.split()


def _normalized_entropy(tokens) -> float:
    n = len(tokens)
    if n <= 1:
        return 0.0
    counts = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1
    entropy = 0.0
    for count in counts.values():
        p = count / n
        entropy -= p * math.log2(p)
    max_entropy = math.log2(n)
    if max_entropy <= 0:
        return 0.0
    return min(1.0, entropy / max_entropy)


def _gibberish_ratio(tokens) -> float:
    if not tokens:
        return 0.0
    gibberish = 0
    for token in tokens:
        if _RE_DIGIT.search(token):
            gibberish += 1
        elif _RE_SYMBOL_ONLY.match(token):
            gibberish += 1
        elif len(token) <= 2 and token.lower() not in _COMMON_SHORT_TOKENS:
            gibberish += 1
    return gibberish / len(tokens)


def _segment_anomaly(tokens) -> float:
    if not tokens:
        return 0.0
    return _normalized_entropy(tokens) * 0.5 + _gibberish_ratio(tokens) * 0.5


def _find_tail_strip_index(text: str, strip_token_count: int) -> int:
    if strip_token_count <= 0:
        return len(text)
    remaining = strip_token_count
    i = len(text)
    while i > 0 and remaining > 0:
        while i > 0 and text[i - 1].isspace():
            i -= 1
        if i == 0:
            break
        while i > 0 and not text[i - 1].isspace():
            i -= 1
        remaining -= 1
    while i > 0 and text[i - 1].isspace():
        i -= 1
    return i


@dataclass(frozen=True)
class EntropyResult:
    text: str
    entropy_score: float = 0.0
    tail_delta: float = 0.0
    anomaly_detected: bool = False
    tail_stripped_length: int = 0


def check_entropy_tail(text: str, *, tail_window: int = DEFAULT_TAIL_WINDOW_TOKENS, strip: bool = True) -> EntropyResult:
    """Detect and optionally strip high-entropy adversarial suffixes (R2A-style)."""
    tokens = _tokenize(text)
    if len(tokens) < MIN_PROMPT_TOKENS:
        return EntropyResult(text)

    max_tail = len(tokens) - MIN_PREFIX_TOKENS
    if max_tail < MIN_TAIL_TOKENS:
        return EntropyResult(text)

    tail_size = min(tail_window, max_tail)
    tail_tokens = tokens[len(tokens) - tail_size:]
    prefix_tokens = tokens[: len(tokens) - tail_size]

    tail_entropy = _normalized_entropy(tail_tokens)
    tail_score = _segment_anomaly(tail_tokens)
    prefix_score = _segment_anomaly(prefix_tokens)
    tail_delta = tail_score - prefix_score
    tail_gibberish = _gibberish_ratio(tail_tokens)

    anomaly = (
        len(tail_tokens) >= MIN_TAIL_TOKENS
        and tail_delta >= ENTROPY_DELTA_THRESHOLD
        and tail_score >= ABSOLUTE_TAIL_ENTROPY_THRESHOLD
        and tail_gibberish >= GIBBERISH_TAIL_RATIO_THRESHOLD
    )
    if not anomaly or not strip:
        return EntropyResult(text, tail_entropy, tail_delta, anomaly, 0)

    strip_index = _find_tail_strip_index(text, len(tail_tokens))
    stripped = text[:strip_index].rstrip()
    return EntropyResult(stripped, tail_entropy, tail_delta, True, len(text) - len(stripped))


# ─── Aho-Corasick multi-pattern matcher ───────────────────────────────────────

_WORD_CHAR = re.compile(r"[0-9A-Za-z_]")


class AhoCorasick:
    """Aho-Corasick automaton for simultaneous multi-pattern matching.

    Word-boundary checks prevent substring false positives
    (e.g. "format" inside "information").
    """

    def __init__(self, patterns):
        # patterns: iterable of (text, set_name)
        self._states = [{"children": {}, "fail": 0, "outputs": []}]
        for text, set_name in patterns:
            self._insert(text.lower(), set_name)
        self._build_failure()

    def _insert(self, text: str, set_name: str) -> None:
        current = 0
        for ch in text:
            nxt = self._states[current]["children"].get(ch)
            if nxt is None:
                nxt = len(self._states)
                self._states.append({"children": {}, "fail": 0, "outputs": []})
                self._states[current]["children"][ch] = nxt
            current = nxt
        self._states[current]["outputs"].append((len(text), set_name))

    def _build_failure(self) -> None:
        queue = []
        for child in self._states[0]["children"].values():
            self._states[child]["fail"] = 0
            queue.append(child)
        head = 0
        while head < len(queue):
            r = queue[head]
            head += 1
            for ch, s in self._states[r]["children"].items():
                queue.append(s)
                f = self._states[r]["fail"]
                while f != 0 and ch not in self._states[f]["children"]:
                    f = self._states[f]["fail"]
                target = self._states[f]["children"].get(ch)
                fail_target = target if target is not None and target != s else 0
                self._states[s]["fail"] = fail_target
                if self._states[fail_target]["outputs"]:
                    self._states[s]["outputs"].extend(self._states[fail_target]["outputs"])

    def search(self, text: str):
        """Return (trivial_hits, complex_hits) with word-boundary checks."""
        lower = text.lower()
        current = 0
        trivial_hits = 0
        complex_hits = 0
        seen = set()
        n = len(lower)
        for i, ch in enumerate(lower):
            while current != 0 and ch not in self._states[current]["children"]:
                current = self._states[current]["fail"]
            current = self._states[current]["children"].get(ch, 0)
            for length, set_name in self._states[current]["outputs"]:
                start = i - length + 1
                if start > 0 and _WORD_CHAR.match(lower[start - 1]):
                    continue
                if i < n - 1 and _WORD_CHAR.match(lower[i + 1]):
                    continue
                key = (start, length)
                if key in seen:
                    continue
                seen.add(key)
                if set_name == VERDICT_TRIVIAL:
                    trivial_hits += 1
                else:
                    complex_hits += 1
        return trivial_hits, complex_hits


# ─── Keyword dictionaries ─────────────────────────────────────────────────────

TRIVIAL_KEYWORDS = (
    "format", "formatting", "lint", "linting", "rename", "indent", "indentation",
    "prettier", "eslint", "semicolon", "whitespace", "spacing", "typo",
    "boilerplate", "template", "uncomment", "sort imports", "fix import",
    "fix imports", "add export", "remove unused", "unused import",
    "unused variable", "fix spacing", "fix whitespace", "simple test", "move file",
)

COMPLEX_KEYWORDS = (
    "architect", "architecture", "refactor", "refactoring", "debug", "debugging",
    "distributed", "microservice", "microservices", "concurrency", "concurrent",
    "deadlock", "race condition", "migration", "migrate", "scalability",
    "infrastructure", "optimize", "optimization", "performance tuning",
    "security audit", "vulnerability", "exploit", "algorithm", "algorithmic",
    "system design", "design pattern", "design patterns", "memory leak",
    "memory management", "state machine", "error handling strategy", "api design",
    "schema design",
    # Repo-hygiene / destructive-intent (SP-176) — never cheap-route on turn 1
    "clean up the repo", "cleanup the repo", "repo cleanup", "clean up", "cleanup",
    "mistakenly added", "accidentally added", "accidental add", "unstage",
    "git rm", "rm -rf", "force push", "git reset --hard", "destructive",
)

_MATCHER = AhoCorasick(
    [(kw, VERDICT_TRIVIAL) for kw in TRIVIAL_KEYWORDS]
    + [(kw, VERDICT_COMPLEX) for kw in COMPLEX_KEYWORDS]
)


# ─── Cyclomatic scan ──────────────────────────────────────────────────────────

_RE_CODE_FENCE = re.compile(r"```\w*\n([\s\S]*?)```")
_RE_INDENTED_BLOCK = re.compile(r"(?:^|\n)((?:(?: {4}|\t).+(?:\n|$))+)")
_DECISION_PATTERNS = tuple(
    re.compile(p)
    for p in (r"\bif\b", r"\belif\b", r"\bfor\b", r"\bwhile\b", r"\bcase\b", r"\bcatch\b", r"&&", r"\|\|", r"\?\?")
)


def _extract_code(text: str) -> str:
    blocks = _RE_CODE_FENCE.findall(text)
    if not blocks:
        blocks = _RE_INDENTED_BLOCK.findall(text)
    return "\n".join(blocks)


def cyclomatic_scan(text: str) -> int:
    """Estimate cyclomatic complexity of code embedded in the prompt."""
    code = _extract_code(text)
    if not code:
        return 1
    score = 1
    for pattern in _DECISION_PATTERNS:
        score += len(pattern.findall(code))
    return score


# ─── Triage entry point ───────────────────────────────────────────────────────


def triage(prompt_text: str) -> TriageResult:
    """Classify a prompt as trivial, complex, or ambiguous for fast-path routing."""
    if not prompt_text or not prompt_text.strip():
        return TriageResult(VERDICT_AMBIGUOUS, "empty_prompt")

    sanitized = sanitize(prompt_text)
    entropy = check_entropy_tail(sanitized)
    scored_text = entropy.text
    delta = len(prompt_text) - len(scored_text)

    if not scored_text or not scored_text.strip():
        return TriageResult(
            VERDICT_AMBIGUOUS,
            "empty_prompt",
            sanitized_length_delta=delta,
            entropy_score=entropy.entropy_score,
            entropy_tail_delta=entropy.tail_delta,
            entropy_tail_stripped_length=entropy.tail_stripped_length,
        )

    trivial_hits, complex_hits = _MATCHER.search(scored_text)
    cyclomatic = cyclomatic_scan(scored_text)

    if cyclomatic >= CYCLOMATIC_THRESHOLD:
        verdict, reason = VERDICT_COMPLEX, "cyclomatic_high"
    elif complex_hits > 0 and trivial_hits == 0:
        verdict, reason = VERDICT_COMPLEX, "keyword_frontier"
    elif trivial_hits > 0 and complex_hits == 0:
        verdict, reason = VERDICT_TRIVIAL, "keyword_economical"
    elif complex_hits > trivial_hits:
        verdict, reason = VERDICT_COMPLEX, "keyword_frontier"
    elif trivial_hits > complex_hits:
        verdict, reason = VERDICT_TRIVIAL, "keyword_economical"
    else:
        verdict, reason = VERDICT_AMBIGUOUS, "no_fast_path"

    return TriageResult(
        verdict,
        reason,
        trivial_hits=trivial_hits,
        complex_hits=complex_hits,
        cyclomatic_score=cyclomatic,
        sanitized_length_delta=delta,
        entropy_score=entropy.entropy_score,
        entropy_tail_delta=entropy.tail_delta,
        entropy_tail_stripped_length=entropy.tail_stripped_length,
    )

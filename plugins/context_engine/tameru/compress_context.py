"""Local extractive, query-aware context compression.

Independent implementation. No Baseline RAG API, no their weights, no
off-box calls. Keep original wording; drop whole blocks; fail open when
the query signal is too weak to justify a cut.
"""

from __future__ import annotations

import hashlib
import bisect
import json
import math
import os
import re
import stat
import tempfile
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .contract_gates import (
    GENERIC_WORDS,
    distinctive_query_terms,
    query_has_distinctive_selectors,
)
from .industrial import IndustrialLimits, IndustrialResult, industrial_preprocess
from .supersession import apply_supersession
from .unicode_profile import script_of, search_units, token_units

try:  # optional semantic tier — never required
    from .semantic import resolve_tier  # noqa: F401
except ImportError:  # direct-script use without package context
    def resolve_tier(tier):
        return tier

try:
    DEFAULT_CCR_DIR = Path.home() / ".tameru" / "cache" / "context-compress-ccr"
except Exception:
    import tempfile
    DEFAULT_CCR_DIR = Path(tempfile.gettempdir()) / ".tameru" / "cache" / "context-compress-ccr"
CCR_TTL_SECONDS = 6 * 60 * 60
CCR_MAX_CLOCK_SKEW_SECONDS = 5 * 60
FREEZE_MAX_DECISIONS = 4096
MAX_QUERY_TEMPLATE_MATCHES = 64
CCR_SWEEP_MAX_RECORDS = 256
MAX_SUMMARY_RESPONSE_BYTES = 1_000_000
DEFAULT_SUMMARY_ENDPOINT = "http://127.0.0.1:18000/v1/chat/completions"
DEFAULT_SUMMARY_MODELS = (
    "Qwen3.8-27B-NVFP4",
    "Qwen3.6 APEX MTP Compact",
    "DeepSeek V4 Flash 0713",
)
DEFAULT_SUMMARY_TIMEOUT = 30.0
_CCR_HASH_RE = re.compile(r"^[0-9a-f]{24}$")
_CCR_SWEEP_CURSOR = 0

_STOP = {
    "what", "how", "does", "do", "the", "is", "are", "was", "were", "why",
    "when", "where", "which", "who", "whom", "whose", "this", "that", "with",
    "into", "about", "for", "and", "or", "of", "to", "a", "an", "in", "on",
    "it", "be", "as", "by", "from", "at", "if", "not", "can", "you", "we",
    "they", "them", "their", "our", "your", "please", "tell", "give", "show",
    "explain", "summarize", "summary", "function", "return", "class", "def",
    "import", "hmm",
}
_KANA_STOP = frozenset("のはがをにでともへやかなねよわさぞぜしてただす")

_TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|"
    r"[\u4e00-\u9fff]|"
    r"[\u3040-\u30ff]{2,}|[\uac00-\ud7a3]{2,}|"
    r"[\u0600-\u06ff]{2,}|[\u0e00-\u0e7f]{2,}"
)
_CAMEL_RE = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b")
# Slash is required between segments. `(?:/?[\w.-]+){2,}` is catastrophic
# on long hyphenated ids (acmecorp-genesis-v14-focused-node22-raw).
_PATH_RE = re.compile(r"(?:[\w.-]+/){2,}[\w.-]+")
_DOTTED_RE = re.compile(r"\b[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+\b")
_QUOTED_RE = re.compile(r"[\"']([^\"']{2,64})[\"']")
_CAPS_RE = re.compile(r"\b[A-Z]{2,24}\b")
_URI_RE = re.compile(r"\b[a-z][a-z0-9+.-]*://\S+")
_FAST_ACCOUNTING_PUNCT = frozenset("§…×–—“”‘’")


@dataclass
class CompressResult:
    compressed_text: str
    original_tokens: int = 0
    kept_tokens: int = 0
    tokens_saved_pct: float = 0.0
    policy_name: str = "local-extractive"
    mode: str = "adaptive"
    keep_ratio: float = 1.0
    tokens_saved: int = 0
    kept_line_ratio: float = 1.0
    cache_prefix_applied: bool = False
    compression_risk: Optional[str] = None
    confidence: Optional[float] = None
    ccr: Optional[dict[str, Any]] = None
    content_type: str = "text"
    fail_open: bool = False
    frozen_blocks: int = 0
    reasons: list[str] = field(default_factory=list)
    verifier: Optional[dict[str, Any]] = None
    receipt: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.tokens_saved = max(0, self.original_tokens - self.kept_tokens)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8", errors="surrogatepass")).hexdigest()


def _bounded_id_manifest(ids: Iterable[int], limit: int) -> list[int] | dict[str, Any]:
    ordered = sorted(int(item) for item in ids)
    resolved = max(0, int(limit))
    if len(ordered) <= resolved:
        return ordered
    head_count = (resolved + 1) // 2
    tail_count = resolved - head_count
    encoded = json.dumps(ordered, separators=(",", ":")).encode("ascii")
    return {
        "count": len(ordered),
        "head": ordered[:head_count],
        "tail": ordered[-tail_count:] if tail_count else [],
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _receipt_hashes(
    source: str,
    output: str,
    industrial: IndustrialResult,
    *,
    mode: str,
    strategy: str,
    budget_ratio: Optional[float],
    citations: bool,
) -> dict[str, str]:
    config = {
        "budget_ratio": budget_ratio,
        "citations": bool(citations),
        "limits": asdict(industrial.limits),
        "mode": mode,
        "strategy": strategy,
    }
    config_bytes = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "source_sha256": _sha256_text(source),
        "output_sha256": _sha256_text(output),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    if text.isascii() or all(
        ord(char) < 128 or char in _FAST_ACCOUNTING_PUNCT for char in text
    ):
        return max(1, len(_TOKEN_RE.findall(text)))
    return max(1, len(token_units(text)))


def _norm_newlines(text: str) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n")


def classify_line(line: str) -> str:
    t = line.strip()
    if not t:
        return "blank"
    if re.match(r"^#{1,6}\s+", t):
        return "heading"
    if re.match(r"^(```|~~~)", t):
        return "fence"
    if re.match(
        r"^(Traceback|Caused by:|Error:|Exception:|[A-Za-z]+Error\b|at\s+\S+\(|\s*File\s+\".+\", line \d+)",
        t,
    ):
        return "trace"
    if re.match(
        r"^\[?(20\d\d-\d\d-\d\d|\d\d:\d\d:\d\d|INFO|WARN|WARNING|ERROR|DEBUG|TRACE|ERR|FAIL|FATAL|OK)\]?",
        t,
    ):
        return "log"
    if re.match(r"^(import|from)\s+|^#include\s+|^using\s+", t):
        return "import"
    if re.match(r"^(def|class|async\s+def|function|const|let|var|public|private|protected|static|fn|func)\b", t):
        return "definition"
    if re.match(r"^[-*+]\s+|\d+\.\s+", t):
        return "list"
    if re.match(r"^\|.*\|$", t):
        return "table"
    if re.match(r"^(User|Assistant|System|Tool|Developer):", t) or re.match(
        r"^\[(user|assistant|system|tool|developer)\]", t, re.I
    ):
        return "chat"
    if t[:1] in "{[" or re.match(r'^"[A-Za-z0-9_]+"\s*:', t):
        return "json"
    if re.match(r"^(\/\/|#|\/\*|\*)", t):
        return "comment"
    return "text"


def route_content_type(lines: list[str]) -> str:
    counts: dict[str, int] = {}
    sample = lines[:50]
    for line in sample:
        t = classify_line(line)
        counts[t] = counts.get(t, 0) + 1
    nonblank = [(k, v) for k, v in counts.items() if k != "blank"]
    if not nonblank:
        return "text"
    top = max(nonblank, key=lambda kv: kv[1])[0]
    importish = counts.get("import", 0) + counts.get("definition", 0) + counts.get("fence", 0)
    if top in {"import", "definition", "fence", "comment"} or importish >= 3:
        return "code"
    if top in {"log", "trace"} or counts.get("log", 0) + counts.get("trace", 0) > 3:
        return "log"
    if top == "chat":
        return "chat"
    joined = "\n".join(sample).strip()
    if joined[:1] in "{[" and re.search(r'"[A-Za-z0-9_]+"\s*:', joined) and importish == 0:
        return "json"
    if top in {"json", "table"} and importish == 0:
        return "json"
    return "text"


def _collapse_blanks(lines: list[str]) -> list[str]:
    out: list[str] = []
    blanks = 0
    for line in lines:
        if not line.strip():
            blanks += 1
            if blanks <= 1:
                out.append(line)
        else:
            blanks = 0
            out.append(line)
    return out


_PROGRESS_RE = re.compile(
    r"^\[?[=#_>\-*\s]{3,}\d{1,3}%[^\n]*$"
    r"|^Compiling [\w-]+ v[\d.]+"
    r"|^Downloading\b"
    r"|^Downloading crates"
    r"|^Pulling fs layer"
    r"|^Step \d+/\d+",
)

_ERROR_RE = re.compile(
    r"error|ERROR|panic:|Caused by|Traceback|Exception|FAILED|fatal|FATAL"
    r"|warning:|WARN|E\d{4}:",
)


def _log_fingerprint(line: str) -> str | None:
    """Template form of a log line: volatile fields -> placeholders.

    Returns None for marker lines ([\u00d7N] ...) so re-running preprocessing
    never re-collapses its own output markers (NTK idempotency invariant).
    """
    if line.lstrip().startswith("[\u00d7"):
        return None
    stripped = _PROGRESS_RE.match(line.strip())
    if stripped:
        return None
    s = re.sub(r"\d+", "#", line.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s[:240] or None


def _is_progress_bar(line: str) -> bool:
    t = line.strip()
    if not t:
        return False
    if re.search(r"(\[[=#>\-. ]*\]\s*)?\d{1,3}\s*(%|/\s*\d+)", t):
        return True
    return bool(re.match(r"^Compiling [\w-]+ ", t))


def preprocess_logs(lines: list[str], query: str = "") -> list[str]:
    """v0.9.0 log preprocessing (NTK layer1 filter adoption):

    - progress bars / build spam deleted (N4)
    - consecutive same-template repeats collapse to one exemplar plus a
      count marker "[\u00d7N] ..." — counts are decision-relevant (N1)
    - markers are fingerprint-inert so preprocessing is idempotent (N2)
    - error-signal invariant: if a transform would remove ALL error lines,
      the errors are restored (N6)
    - Go/Java-style indented frame runs collapse like Python traces,
      preserving first user frame (N3)
    """
    out: list[str] = []
    trace: list[str] = []
    query_match_indices: set[int] = set()
    query_selectors = distinctive_query_terms(query or "")

    def flush_trace() -> None:
        nonlocal trace
        if not trace:
            return
        if len(trace) <= 4:
            out.extend(trace)
        else:
            out.append(trace[0])
            out.append(f"  ... {len(trace) - 2} frames omitted")
            out.append(trace[-1])
        trace = []

    # First pass: drop progress bars but remember whether we saw any.
    kept_lines = [ln for ln in lines if not _is_progress_bar(ln)]

    query_match_source_indices: set[int] = set()
    if query_selectors:
        candidates = [
            idx
            for idx, line in enumerate(kept_lines)
            if any(
                _term_in_text(selector, line.casefold())
                for selector in query_selectors
            )
        ]
        if len(candidates) <= MAX_QUERY_TEMPLATE_MATCHES:
            query_match_source_indices = set(candidates)
        else:
            # Evenly sample the full match span. This retains head/tail and
            # representative middle records without turning one common
            # selector (for example "stage 3") into thousands of blocks.
            last = len(candidates) - 1
            query_match_source_indices = {
                candidates[(slot * last) // (MAX_QUERY_TEMPLATE_MATCHES - 1)]
                for slot in range(MAX_QUERY_TEMPLATE_MATCHES)
            }

    seen: dict[str, int] = {}
    exemplar_idx: dict[str, int] = {}

    i = 0
    n = len(kept_lines)
    while i < n:
        line = kept_lines[i]
        kind = classify_line(line)
        is_trace_like = kind == "trace" or bool(
            (trace or (out and _looks_indented_frame(line))) and re.match(r"^\s+", line)
        )
        if is_trace_like:
            trace.append(line)
            i += 1
            continue
        flush_trace()
        if i in query_match_source_indices:
            # Volatile values can be the selector (for example "policy 17").
            # Keep the exact matching record outside numeric-template collapse.
            query_match_indices.add(len(out))
            out.append(line)
            i += 1
            continue
        fp = _log_fingerprint(line)
        if fp and len(fp) > 12:
            if fp in exemplar_idx:
                seen[fp] += 1
                i += 1
                continue
            exemplar_idx[fp] = len(out)
            seen[fp] = 1
        out.append(line)
        i += 1
    flush_trace()

    # Attach counts: replace exemplar line with counted variant.
    counted_out: list[str] = []
    for idx, ln in enumerate(out):
        if idx in query_match_indices:
            counted_out.append(ln)
            continue
        fp = _log_fingerprint(ln)
        cnt = seen.get(fp, 1) if fp else 1
        if fp and cnt > 1 and not ln.lstrip().startswith("[\u00d7"):
            counted_out.append(f"[\u00d7{cnt}] {ln}")
        else:
            counted_out.append(ln)
    result = counted_out

    # N6 error-signal invariant: original error lines that are missing from
    # the output must be restored.
    def has_error_signal(text: str) -> bool:
        return bool(_ERROR_RE.search(text))

    orig_errors = [ln for ln in lines if has_error_signal(ln)]
    if orig_errors:
        out_text = "\n".join(result)
        missing = [ln for ln in orig_errors if ln not in result and ln not in out_text]
        if missing:
            # Restore missing error lines right after the head (or at top).
            insert_at = min(1, len(result))
            result = result[:insert_at] + missing + result[insert_at:]

    return _collapse_blanks(result)


def _looks_indented_frame(line: str) -> bool:
    """Go/Java-style stack frame shapes (NTK classifier table, subset)."""
    t = line.strip()
    if not t.startswith("\t") and not line.startswith("    "):
        # also treat single-tab-ish indents via leading whitespace check below
        pass
    if re.match(r"^\s+goroutine \d+", t) or "goroutine" in t[:40]:
        return True
    if re.match(r"^\s*at\s+[\w$.]+\(", t):          # Java: at org.x.Y.f(Foo.java:12)
        return True
    if re.match(r"^\s*[\w().*/]+\([\w/.$-]+:\d+\)", t):  # Go: pkg.Fn(/path/file.go:88)
        return True
    if re.match(r"^\s+[\w./$-]+\([^)]*:\d+\)", t):
        return True
    return False


def _crush_value(
    val: Any,
    depth: int,
    query_needles: list[str],
    query_selectors: list[re.Pattern[str]] | None = None,
) -> Any:
    if query_selectors is None:
        query_selectors = _selector_patterns(query_needles)
    if depth > 10:
        return val
    if val is None:
        return None
    if isinstance(val, list):
        if not val:
            return []
        matching = []
        rest = []
        for item in val:
            blob = json.dumps(item, ensure_ascii=False) if not isinstance(item, str) else item
            if query_selectors and any(selector.search(blob) for selector in query_selectors):
                matching.append(
                    _crush_value(item, depth + 1, query_needles, query_selectors)
                )
            else:
                rest.append(item)
        # No Baseline RAG first-3 cliff: if nothing matches, keep the
        # whole array. Destructive crush only happens when needles hit.
        if matching:
            leftover = len(val) - len(matching)
            if leftover > 0:
                matching.append(f"... {leftover} more items")
            return matching
        return [
            _crush_value(x, depth + 1, query_needles, query_selectors)
            for x in val
        ]
    if isinstance(val, dict):
        out: dict[str, Any] = {}
        for k, v in val.items():
            if v is None:
                continue
            if isinstance(v, list) and not v:
                continue
            if isinstance(v, str) and not re.search(r"\s", v):
                # Opaque values are commonly hashes, tokens or identifiers;
                # shortening or dropping them destroys their exact meaning.
                out[k] = v
                continue
            if isinstance(v, str) and len(v) > 200:
                if query_selectors and any(
                    selector.search(v) for selector in query_selectors
                ):
                    out[k] = v
                    continue
                out[k] = v[:100] + f"... [{len(v) - 100} more chars]"
                continue
            out[k] = _crush_value(v, depth + 1, query_needles, query_selectors)
        return out
    return val


def _json_query_needles(query: str) -> list[str]:
    terms = set(distinctive_query_terms(query))
    terms.update(_extract_entities(query))
    strong = {
        term
        for term in terms
        if any(ch.isdigit() for ch in term)
        or "-" in term
        or "_" in term
        or len(term) >= 8
    }
    return sorted(strong or terms)


def preprocess_json(text: str, query: str) -> str:
    stripped = text.strip()
    if not query_has_distinctive_selectors(query):
        return text
    needles = _json_query_needles(query)
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, RecursionError):
        return text
    try:
        crushed = _crush_value(parsed, 0, needles)
    except RecursionError:
        return text
    try:
        dumped = json.dumps(crushed, ensure_ascii=False, indent=2)
    except (TypeError, ValueError, RecursionError):
        return text
    if len(dumped) < len(stripped) * 0.92:
        return dumped
    return text


def _looks_like_csv(text: str) -> bool:
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) < 6 or lines[0].count(",") < 2:
        return False
    similar = sum(1 for ln in lines[1:40] if ln.count(",") >= 2)
    return similar >= 4


def _selector_patterns(terms: Iterable[str]) -> list[re.Pattern[str]]:
    return [
        re.compile(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])", re.I)
        for term in terms
    ]


def preprocess_csv(text: str, query: str) -> str:
    """Keep header + distinctive-matching rows. No first-N cliff."""
    if not query_has_distinctive_selectors(query) or not _looks_like_csv(text):
        return text
    terms = [t for t in distinctive_query_terms(query) if not t.startswith("script:")]
    if not terms:
        return text
    selectors = _selector_patterns(terms)
    lines = text.split("\n")
    kept = [lines[0]]
    for line in lines[1:]:
        if any(selector.search(line) for selector in selectors):
            kept.append(line)
    if len(kept) == 1 or len(kept) == len(lines):
        return text
    return "\n".join(kept)


_FLAT_RECORD_RE = re.compile(r"^[^\s:#][^:\n]{0,200}:\s+\S.*$")


def preprocess_flat_records(text: str, query: str) -> str:
    """Select exact rows from a flat key/value record dump."""
    if not query_has_distinctive_selectors(query):
        return text
    lines = text.split("\n")
    if any(not line.strip() for line in lines):
        return text
    if len(lines) < 6 or not all(_FLAT_RECORD_RE.match(line) for line in lines):
        return text
    terms = [
        term
        for term in distinctive_query_terms(query)
        if not term.startswith("script:")
    ]
    selectors = _selector_patterns(terms)
    kept = [line for line in lines if any(rx.search(line) for rx in selectors)]
    if not kept or len(kept) == len(lines):
        return text
    return "\n".join(kept)


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_TEST_SUMMARY_RE = re.compile(r"Tests\s+\d+\s+passed", re.I)
_TEST_FILE_NOISE_RE = re.compile(
    r"ExperimentalWarning|Use `node --trace-warnings|^\s*[✓✔]\s"
)


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def preprocess_test_runner(text: str, query: str) -> str:
    """Keep suite summary; drop per-file ticks and node warnings.

    Super does this on vitest dumps. Skip if the query names a test file.
    """
    plain = strip_ansi(text)
    if not _TEST_SUMMARY_RE.search(plain):
        return text
    q = (query or "").lower()
    if re.search(r"\.(test|spec)\.\w+|__tests__/", q):
        return text
    kept = []
    for ln in text.split("\n"):
        p = strip_ansi(ln)
        if _TEST_FILE_NOISE_RE.search(p):
            continue
        kept.append(p)
    if len(kept) >= len(text.split("\n")):
        return text
    return "\n".join(kept)


_FILLER_COMMENT_RE = re.compile(r"^(--|#)\s*filler\b", re.I)
_TRUST_RISK_RE = re.compile(
    r"\b(?:untrusted|not\s+operational)\b"
    r"|\b(?:obsolete|unrelated)\b.{0,24}"
    r"\b(?:glossary|example|sample|entry|data)\b"
    r"|\b(?:ignore|disregard|override|forget)\b.{0,48}"
    r"\b(?:instructions?|prompt|system)\b",
    re.I,
)
_TRUST_QUERY_RE = re.compile(
    r"\b(?:injection|untrusted|malicious|prompt\s+attack|obsolete|unrelated)\b"
    r"|\bignore\s+(?:prior|previous)\b",
    re.I,
)
# v0.6.0: a query that says "trust nothing from untrusted sources",
# "exclude untrusted …" etc. is asking for the SAFE behaviour — it must
# NOT suppress the trust-risk flag. Only explicit include-phrasing
# ("include the untrusted block verbatim") counts as an override.
_TRUST_EXCLUDE_CUE_RE = re.compile(
    r"\b(?:trust\s+nothing|don't\s+trust|do\s+not\s+trust|exclude|filter\s+out|omit)\b"
    r".{0,40}\b(?:untrusted|unverified|malicious)\b"
    r"|\b(?:exclude|drop|skip|omit)\s+(?:any\s+)?(?:untrusted|unverified|malicious)\b",
    re.I,
)


def _normalise_instruction_text(text: str) -> str:
    """Normalise width and remove invisible formatting before classification."""
    normalised = unicodedata.normalize("NFKC", str(text or ""))
    return "".join(
        char for char in normalised if unicodedata.category(char) != "Cf"
    )


def preprocess_filler_comments(text: str, query: str) -> str:
    """Drop `-- filler` / `# filler` comments unless the query names them."""
    if "filler" in (query or "").casefold():
        return text
    if not query_has_distinctive_selectors(query or ""):
        return text
    lines = text.split("\n")
    kept = [ln for ln in lines if not _FILLER_COMMENT_RE.match(ln.strip())]
    if len(kept) == len(lines):
        return text
    return "\n".join(kept)


def unwrap_hermes_tool(text: str) -> str:
    """Score the inner payload of a Hermes tool JSON wrapper."""
    stripped = text.strip()
    if not stripped.startswith("{"):
        return text
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, RecursionError):
        return text
    if not isinstance(parsed, dict):
        return text
    for key in ("content", "output"):
        val = parsed.get(key)
        if isinstance(val, str) and len(val) > 200:
            return val
    return text


def preprocess(text: str, query: str) -> tuple[list[str], str]:
    lines = _norm_newlines(text).split("\n")
    kind = route_content_type(lines)
    if kind == "log":
        return preprocess_logs(lines, query), kind
    if kind == "code":
        return _collapse_blanks(lines), kind
    if kind == "json":
        # Decode recognised Hermes content/output wrappers structurally before
        # JSON crushing. json.loads distinguishes escaped newlines from literal
        # backslash-n bytes; a blanket text replacement cannot.
        unwrapped = unwrap_hermes_tool(text)
        if unwrapped != text:
            return preprocess(unwrapped, query)
        processed = preprocess_json(text, query)
        return processed.split("\n"), kind
    if kind == "text":
        processed = preprocess_flat_records(text, query)
        if processed != text:
            return processed.split("\n"), kind
    return _collapse_blanks(lines), kind


def _extract_terms(query: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for raw in _TOKEN_RE.findall(query or ""):
        term = raw.lower().strip("؟?!.،,;:…\"'()[]")
        if not term:
            continue
        # CJK characters are single tokens (no word boundaries in CJK).
        # Accept them even though they're 1 char — they carry meaning.
        o = ord(raw[0]) if raw else 0
        is_cjk = (
            0x4E00 <= o <= 0x9FFF
            or 0x3040 <= o <= 0x30FF
            or 0xAC00 <= o <= 0xD7A3
            or 0x0600 <= o <= 0x06FF
            or 0x0E00 <= o <= 0x0E7F
        )
        if (len(term) <= 2 and not is_cjk) or term in _STOP or term in _KANA_STOP or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    for raw in search_units(query or "")[:128]:
        term = raw.strip("؟?!.،,;:…\"'()[]")
        if not term or term in seen or term in _STOP or term in _KANA_STOP:
            continue
        representative = next((char for char in term if char.isalnum()), term[0])
        script = script_of(representative)
        if script in {"han", "kana", "hangul", "arabic"}:
            continue
        if len(term) <= 2 and script in {None, "latin", "greek", "cyrillic"}:
            continue
        seen.add(term)
        terms.append(term)
    # Hyphenated compound names in the query ("titanium-torsion-rod") are
    # single search terms too — the tokenizer splits them on "-".
    for m in re.findall(r"\b\w+(?:-\w+)+\b", query or ""):
        m = m.lower()
        if m not in _STOP and m not in seen:
            seen.add(m)
            terms.append(m)
    return terms


_WEAK_NEXT_CHAT = frozenset(
    {
        "keep",
        "track",
        "every",
        "detail",
        "details",
        "next",
        "chat",
        "start",
        "work",
        "please",
        "note",
        "final",
        "begin",
    }
)


def _topic_terms(query: str) -> list[str]:
    """English topic words that are not stop/generic/next-chat filler."""
    return [
        t
        for t in _extract_terms(query)
        if len(t) >= 4
        and t not in _STOP
        and t not in GENERIC_WORDS
        and t not in _WEAK_NEXT_CHAT
    ]


def _extract_entities(query: str) -> list[str]:
    found: list[str] = []
    for rx in (_URI_RE, _PATH_RE, _DOTTED_RE, _CAMEL_RE, _QUOTED_RE, _CAPS_RE):
        for m in rx.finditer(query or ""):
            val = m.group(1) if m.lastindex else m.group(0)
            if val and val not in found and len(val) >= 2:
                found.append(val)
    return found


_GIT_ONELINE_RE = re.compile(r"^[0-9a-f]{7,40}\s+\d{4}-\d{2}-\d{2}\s")
_NPM_VERBOSE_RE = re.compile(
    r"^\d+\s+(verbose|info|silly|http|warn|error|timing)\b"
)
_HEX_ID_RE = re.compile(r"^[0-9a-f]{7,40}$")


def _looks_like_line_records(lines: list[str]) -> bool:
    """Git oneline / npm verbose: one record per line, not 10-line chunks."""
    sample = [ln for ln in lines[:80] if ln.strip()]
    if len(sample) < 8:
        return False
    git_n = sum(1 for ln in sample if _GIT_ONELINE_RE.match(ln))
    npm_n = sum(1 for ln in sample if _NPM_VERBOSE_RE.match(ln))
    return git_n >= 6 or npm_n >= 6


def segment_blocks(lines: list[str]) -> list[dict[str, Any]]:
    if _looks_like_line_records(lines):
        blocks: list[dict[str, Any]] = []
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            blocks.append(
                {
                    "id": len(blocks),
                    "start": i,
                    "end": i,
                    "type": "log" if _NPM_VERBOSE_RE.match(line) else "text",
                    "text": line,
                    "tokens": max(1, estimate_tokens(line)),
                }
            )
        return blocks
    return _segment_blocks_chunked(lines)


def _segment_blocks_chunked(lines: list[str]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    start: Optional[int] = None
    btype = "text"
    tokens = 0

    def close(end: int) -> None:
        nonlocal start, tokens
        if start is None:
            return
        while end >= start and not lines[end].strip():
            end -= 1
        if end < start:
            start = None
            tokens = 0
            return
        text = "\n".join(lines[start : end + 1])
        if text.strip():
            blocks.append(
                {
                    "id": len(blocks),
                    "start": start,
                    "end": end,
                    "type": btype,
                    "text": text,
                    "tokens": max(1, tokens or estimate_tokens(text)),
                }
            )
        start = None
        tokens = 0

    prev_type = None
    in_fence = False
    for i, line in enumerate(lines):
        kind = classify_line(line)
        line_toks = estimate_tokens(line)
        if in_fence:
            tokens += line_toks
            if kind == "fence":
                close(i)
                in_fence = False
            prev_type = kind
            continue
        if kind == "fence":
            close(i - 1)
            start = i
            btype = "fence"
            tokens = line_toks
            in_fence = True
            prev_type = kind
            continue
        new = False
        if start is None:
            new = True
        elif kind in {"heading", "fence", "definition"}:
            new = True
        elif kind == "list":
            new = True
        elif prev_type == "blank" and kind != "blank":
            new = True
        elif kind != prev_type and kind in {"log", "chat", "import", "list", "table", "json"}:
            new = True
        elif btype == "text" and kind in {"definition", "code"}:
            # A code/definition region starting inside a text block: split so
            # the code has its own block. Otherwise one "fn name(" line
            # classifies the whole block as code and the impl body ends up
            # scored as text.
            new = True
        elif start is not None and (i - start >= 10 or tokens + line_toks > 220):
            new = True
        if new:
            close(i - 1)
            start = i
            btype = kind if kind != "blank" else "text"
            tokens = line_toks
        else:
            tokens += line_toks
            if btype == "text" and kind != "blank":
                btype = kind
        prev_type = kind
    close(len(lines) - 1)
    return blocks


def _soft_has(text: str, needle: str) -> bool:
    if not needle:
        return False
    if needle in text:
        return True
    return needle.lower() in text.lower()


def _term_in_text(term: str, lower: str) -> bool:
    """Substring for long / syllabic terms; word-boundary for short Latin."""
    if not term:
        return False
    o = ord(term[0])
    syllabic = (
        0x4E00 <= o <= 0x9FFF
        or 0x3040 <= o <= 0x30FF
        or 0xAC00 <= o <= 0xD7A3
        or 0x0600 <= o <= 0x06FF
        or 0x0E00 <= o <= 0x0E7F
    )
    if syllabic or len(term) > 5:
        return term in lower
    return re.search(rf"\b{re.escape(term)}\b", lower) is not None


def score_blocks(blocks: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    terms = _extract_terms(query)
    entities = _extract_entities(query)
    n = max(1, len(blocks))
    entity_df: dict[str, int] = {}
    term_df: dict[str, int] = {}
    fp_counts: dict[str, int] = {}
    # Bridge-entity detection: find entities (proper nouns, identifiers) that
    # appear in multiple blocks. These are the links in a multi-hop chain —
    # dropping the block that holds a bridge entity severs the chain.
    # Only boost entities that look like proper nouns or code identifiers,
    # not common filler words.
    bridge_entities: dict[str, int] = {}
    # Perf (v0.6.0): lowercase every block text exactly once. The previous
    # code re-lowered all N texts for every candidate of every block
    # (O(n²·c) .lower() calls) — the dominant cost on large documents.
    # Results are byte-identical; only the redundant work is removed.
    lower_texts = [b["text"].lower() for b in blocks]
    # Perf (v0.6.0): deduplicate candidates globally and resolve df via a
    # token Counter for single-token candidates (the vast majority). Only
    # multi-token phrases pay the exact substring scan. df for a single
    # \w-delimited token equals its token frequency except when it is a
    # substring of a LONGER token (e.g. "module_9" inside "module_90");
    # such boundary cases shift the boost amount marginally but never flip
    # the 2 <= df <= n//2 gate's intent.
    token_counter: dict[str, int] = {}
    # Perf (v0.6.0): one shared \w+ tokenisation per block feeds the counter,
    # the candidate scan AND tok_df below (previously 3 separate passes).
    block_token_lists: list[list[str]] = []
    for lt in lower_texts:
        toks = re.findall(r"\w+", lt)
        block_token_lists.append(toks)
        for w in toks:
            token_counter[w] = token_counter.get(w, 0) + 1
    seen_candidates: set[str] = set()
    candidate_blocks: dict[str, int] = {}
    _PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b")
    for b_idx, b in enumerate(blocks):
        toks = block_token_lists[b_idx]
        # Extract candidate bridge entities: CamelCase, UPPER_SNAKE, dotted
        # paths, quoted strings, and 6+ char tokens with digits (identifiers).
        candidates = set()
        candidates.update(_CAMEL_RE.findall(b["text"]))
        candidates.update(_PROPER_NOUN_RE.findall(b["text"]))  # Proper nouns
        candidates.update(re.findall(r"\b[A-Z][A-Z_][A-Z_0-9]{3,}\b", b["text"]))  # UPPER_SNAKE
        candidates.update(_DOTTED_RE.findall(b["text"]))
        candidates.update(_QUOTED_RE.findall(b["text"]))
        # 6+ char identifier tokens with digits — from the shared token list
        # instead of a second regex sweep over the raw text.
        # Perf (v0.6.0): the any(ch.isdigit()) genexpr over every token was
        # the single hottest line in score_blocks (266k calls). isdigit()
        # on the whole token is equivalent for this gate (we only care
        # whether ANY digit exists, and str.isdigit() short-circuits in C).
        for tok in toks:
            if len(tok) >= 6 and not tok.isdigit() and not tok.isalpha() and tok not in _STOP:
                candidates.add(tok)
        for cand in candidates:
            if len(cand) < 4 or cand in _STOP:
                continue
            is_identifier = (
                cand in cand.upper()  # ALL CAPS (UPPER_SNAKE, acronyms)
                or cand[0].isupper()  # Proper noun or CamelCase
                or re.search(r"\d", cand)  # Contains a digit
                or "." in cand  # Dotted path
            )
            if is_identifier and cand not in seen_candidates:
                seen_candidates.add(cand)
                candidate_blocks[cand] = 0
    for cand in seen_candidates:
        c_lower = cand.lower()
        if re.fullmatch(r"\w{4,}", c_lower):
            df = token_counter.get(c_lower, 0)
        else:
            df = sum(1 for lt in lower_texts if c_lower in lt)
        # Require the entity to be in at least 2 but at most n/2 blocks
        # (if it's in every block, it's a filler word, not a bridge)
        if 2 <= df <= n // 2:
            bridge_entities[cand] = df
    for b in blocks:
        fp = re.sub(r"\s+", " ", b["text"].lower())[:240]
        if len(fp) > 20:
            fp_counts[fp] = fp_counts.get(fp, 0) + 1
        for e in entities:
            if _soft_has(b["text"], e):
                entity_df[e] = entity_df.get(e, 0) + 1
        lower = b["text"].lower()
        for t in terms:
            if _term_in_text(t, lower):
                term_df[t] = term_df.get(t, 0) + 1
    # Pre-compute token document frequency for novelty + fact-density.
    # A token's DF is the number of blocks it appears in. Low DF (1-2)
    # means the token is rare and information-dense.
    tok_df: dict[str, int] = {}
    # Perf (v0.6.0): reuse the shared per-block token lists (built above for
    # the bridge-candidate scan) — no third tokenisation pass.
    block_token_sets: list[set[str]] = []
    for toks in block_token_lists:
        seen_in_block: set[str] = set()
        block_tokens: set[str] = set(toks)
        for tok in toks:
            if len(tok) >= 4 and tok not in _STOP and tok not in seen_in_block:
                seen_in_block.add(tok)
                tok_df[tok] = tok_df.get(tok, 0) + 1
        block_token_sets.append(block_tokens)

    # Log-type detection (Baseline RAG "Log" policy): if > 60% of blocks
    # look like timestamped log lines, apply log-specific scoring weights.
    # Error/traceback blocks get a bonus; pure INFO/WARN noise gets penalized.
    _LOG_LINE_RE = re.compile(
        r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}"    # timestamp prefix
        r"|^\[(?:DEBUG|INFO|WARN|ERROR|FATAL)\]"      # bracketed level
        r"|^(?:DEBUG|INFO|WARN|ERROR|FATAL)\s"       # bare level prefix
    )
    log_block_count = sum(
        1 for b in blocks
        if any(_LOG_LINE_RE.match(line) for line in b["text"].splitlines()[:3])
    )
    is_log_context = log_block_count >= 2 and log_block_count / n > 0.4
    if is_log_context:
        _LOG_ERR_RE = re.compile(
            r"\b(?:ERROR|FATAL|WARN|WARNINGS?)\b|Traceback|Exception",
            re.IGNORECASE,
        )
        _LOG_NOISE_RE = re.compile(
            r"^\s*(?:INFO|DEBUG)\b", re.MULTILINE,
        )
    # Recency + novelty prior (rate-distortion signal): even without query
    # overlap, recent blocks and blocks holding rare, high-information
    # identifiers deserve a base floor so a "keep track of every detail"
    # style query does not evict scattered facts.
    # Critical-line pre-computation (Baseline RAG-style):
    # Pre-identify lines that carry operational facts — errors, exceptions,
    # config assignments, key-value pairs, dates, versions, large numbers.
    # Blocks containing these lines get a floor boost so they survive
    # compression even without query overlap.
    _CRIT_LINE_RE = re.compile(
        r"\b(?:error|exception|traceback|fatal|critical)\b"   # error keywords
        r"|\bconfig(?:uration)?\s*[=:]"                       # config assignments
        r"|\b(?:port|host|user|password|url|token|key)\s*[=:]"  # key-value pairs
        r"|\b20[2-3]\d-\d{2}-\d{2}\b"                         # dates
        r"|\b\d+\.\d+\.\d+(?:\.\d+)?\b"                       # version numbers
    )
    crit_lines_per_block: list[int] = []
    for b in blocks:
        cnt = 0
        for line in b["text"].splitlines():
            if _CRIT_LINE_RE.search(line, re.IGNORECASE):
                cnt += 1
        crit_lines_per_block.append(cnt)

    # Weak-query adaptive floor (rate-distortion H(Q) theory):
    # When the query has low information content, raise the novelty floor
    # and recency ramp so middle blocks survive. A vague "keep track of
    # every detail" query has H(Q) ≈ 0 — it tells us nothing about what
    # to keep, so we should be more conservative about dropping.
    #
    # A query is "weak" if NONE of its terms are distinctive identifiers
    # (digits, separators, uppercase). "database host port" is weak
    # (all generic words, no identifiers). "REL-2026 user_id 99113" is
    # strong (has identifiers). This matches the rate-distortion theory:
    # H(Q) ≈ 0 when the query carries no specific information.
    #
    # Guard: only apply the weak-query boost when there are enough
    # blocks (≥ 6) to make compression meaningful. With ≤ 5 blocks,
    # the keep_ratio will be ~1.0 and we'd fail-open anyway.
    weak_boost = 0.0
    if len(blocks) >= 6 and terms:
        distinctive = 0
        for t in terms:
            if len(t) < 4 or t in _STOP:
                continue
            if re.search(r"[\d_./-]", t) or t[0].isupper():
                distinctive += 1
        topic = _topic_terms(query)
        # English topic words (alena, cinematography) are not "weak" just
        # because they lack digits. Vague next-chat ("keep every detail")
        # still gets the uniqueness floor.
        weak_boost = 0.0 if distinctive > 0 or len(topic) >= 2 else 1.0

    scored = []
    for idx, b in enumerate(blocks):
        score = 0.0
        reason = "context"
        entity_hits = 0
        term_hits = 0
        rare_term_hits = 0
        for e in entities:
            if _soft_has(b["text"], e):
                entity_hits += 1
                df = entity_df.get(e, 1)
                score += 1.4 + 4.6 * math.log((n + 1) / (df + 1))
                reason = "query entity"
        lower = b["text"].lower()
        # Classification belongs to the block, never to question wording.
        # An intentional override must be explicit through pin_patterns.
        trust_risk = bool(
            _TRUST_RISK_RE.search(_normalise_instruction_text(b["text"]))
        )
        for t in terms:
            if _term_in_text(t, lower):
                term_hits += 1
                df = term_df.get(t, 1)
                if df <= max(2, int(0.2 * n)):
                    rare_term_hits += 1
                score += 0.9 + 3.2 * math.log((n + 1) / (df + 1))
                if reason == "context":
                    reason = "query term"
        if b["type"] == "definition":
            score += 2.0
        if b["type"] == "trace":
            score += 2.8
        if b["type"] == "heading":
            score += 0.8
        if b["type"] == "log" and re.search(r"error|fail|exception|timeout|denied|invalid", lower):
            score += 1.2
        # Old-error purge (leanctx-inspired): errored dumps with ZERO query
        # hits deep inside a large document are handled history — the fact of
        # the error survives via citations; the bulk shouldn't. Only applies
        # when there are enough blocks that recency is meaningful.
        if (
            n >= 12
            and idx < n - 6
            and entity_hits == 0
            and term_hits == 0
            and (
                b["type"] == "trace"
                or (
                    b["type"] == "log"
                    and re.search(r"error|fail|exception|denied", lower)
                )
                or "Traceback (most recent call last)" in b["text"]
            )
        ):
            score -= 4.5
            if reason == "context":
                reason = "stale error"

        # Log-context policy (Baseline RAG "Log" policy): in log-heavy
        # contexts, boost error/traceback blocks and penalize pure INFO
        # noise so the signal stands out from the chatter.
        if is_log_context:
            if _LOG_ERR_RE.search(b["text"]):
                score += 3.0
                if reason == "context":
                    reason = "log error"
            else:
                lines = b["text"].splitlines()
                noise_lines = sum(
                    1 for block_line in lines if _LOG_NOISE_RE.match(block_line)
                )
                if noise_lines >= len(lines) * 0.8 and len(lines) >= 2:
                    score -= 1.5
                    if reason == "context":
                        reason = "log noise"
        # Critical-line floor: blocks containing operational facts (errors,
        # config, key-value pairs, dates, versions) get a floor boost so
        # they survive even without query overlap. This is the Baseline RAG
        # "important_kept_pct" signal — their verifier tracks critical lines
        # and flags risk when they're dropped. We make it a scoring input
        # instead, so the critical lines are kept in the first place.
        crit = crit_lines_per_block[idx]
        if crit > 0:
            score += min(3.0, 0.8 * crit)
            if reason == "context":
                reason = "critical lines"
        # Bridge-entity boost: blocks that contain cross-block entities (the
        # links in a multi-hop chain) get a floor boost so the chain doesn't
        # get severed. The boost scales with how many blocks the entity spans.
        # Perf (v0.6.0): iterate only over entities present in this block
        # via the precomputed token set instead of scanning every bridge
        # entity for every block (O(n·e) substring scans → O(n + e)).
        bridge_boost = 0.0
        block_tokens = block_token_sets[idx]
        for ent, span in bridge_entities.items():
            if ent.lower() in block_tokens:
                bridge_boost += 1.5 + 0.5 * min(span, 5)
        if bridge_boost > 0:
            score += bridge_boost
            if reason == "context":
                reason = "bridge entity"
        # Uniqueness floor (ACON insight): if a block contains a
        # high-information identifier that appears in NO OTHER block,
        # it's uniquely valuable. Drop it and the fact is gone forever
        # (unless CCR is enabled). Give it a floor boost so scattered
        # facts survive even when the query doesn't mention them.
        # Only applies when the query is weak (low entropy) — a vague
        # "keep track of every detail" query deserves a higher keep
        # threshold. A strong, specific query ("who won the 1998 world
        # cup") should still evict unrelated blocks even if they're
        # unique.
        unique_id_count = 0
        list_noise = b["type"] == "list" and entity_hits == 0 and term_hits == 0
        if weak_boost > 0 and not list_noise:
            # Perf (v0.6.0): reuse the shared token set — no extra regex pass.
            for tl in block_tokens:
                if len(tl) < 6 or tl in _STOP:
                    continue
                if _HEX_ID_RE.match(tl):
                    continue
                df = tok_df.get(tl, 0)
                if df == 1:
                    unique_id_count += 1
            if unique_id_count >= 2:
                score += min(4.0, 1.5 + 0.8 * unique_id_count)
                if reason in ("context",):
                    reason = "unique facts"
        # Recency + novelty prior: recent blocks get a ramp up to ~2.0;
        # rare high-information identifiers (short tokens, dotted paths,
        # UPPER_SNAKE, REL-xxx, 8+ hex) give a small floor so scattered
        # facts survive a vague "keep everything" query.
        # Weak-query boost: when the query has low information content,
        # raise both the recency ramp and the novelty floor so middle
        # blocks survive. This is the rate-distortion H(Q) theory: a
        # vague query deserves a higher keep threshold.
        recency_max = 2.0 + weak_boost
        recency = recency_max * (idx + 1) / n
        score += recency
        novelty = 0.0
        rare_count = 0
        # Perf (v0.6.0): reuse block_token_sets instead of re-tokenising
        # and re-lowering every block here (third full token pass removed).
        for tl in block_token_sets[idx]:
            if len(tl) < 4 or tl in _STOP:
                continue
            if _HEX_ID_RE.match(tl):
                continue
            df = tok_df.get(tl, 0)
            if df <= 2 and not re.match(r"^[a-z]{4}$", tl):
                novelty += 0.6
                rare_count += 1
        # Weak-query boost on novelty floor: raise the cap from 2.4 to
        # 2.4 + weak_boost (up to 3.4) when the query is vague.
        score += min(2.4 + weak_boost, novelty)
        # Fact density: blocks with many distinct rare identifiers are
        # information-dense. Even without query overlap, a block packed
        # with rare facts is worth keeping. This rescues scattered facts
        # in large contexts where the query only mentions a few entities.
        # Threshold: 5+ rare tokens → +1.0, 10+ → +2.0, 15+ → +3.0 (cap).
        if rare_count >= 5 and not list_noise:
            fact_density = min(3.0, 0.2 * rare_count)
            score += fact_density
        # Recency / sinks
        if idx == 0:
            score += 0.35
        if idx == n - 1:
            score += 0.45
        fp = re.sub(r"\s+", " ", b["text"].lower())[:240]
        dup = fp_counts.get(fp, 0)
        if dup > 1:
            score -= min(3.0, dup * 0.8)
        if trust_risk:
            score -= 50.0
            reason = "untrusted instruction"
        # Final stale-error cap: after every boost (recency, novelty,
        # fact-density), a handled error dump must stay below the adaptive
        # floor minimum so it drops to citations instead of re-inflating.
        if reason == "stale error":
            score = min(score, 1.8)
        scored.append(
            {
                **b,
                "score": max(0.0, score),
                "reason": reason,
                "entity_hits": entity_hits,
                "term_hits": term_hits,
                "rare_term_hits": rare_term_hits,
                "trust_risk": trust_risk,
            }
        )
    return scored


def _is_sink_noise(text: str) -> bool:
    """Trailing `-- filler` / `# filler` comments are not a recency sink."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    return all(re.match(r"^(--|#)\s*filler\b", ln, re.I) for ln in lines)


def _query_signal(blocks: list[dict[str, Any]]) -> float:
    hits = sum(1 for b in blocks if b["entity_hits"] or b["term_hits"])
    return hits / max(1, len(blocks))


def _important(blocks: list[dict[str, Any]]) -> set[int]:
    n = max(1, len(blocks))
    rare = []
    for b in blocks:
        if b.get("trust_risk"):
            continue
        if b["entity_hits"] and b["score"] >= 4:
            rare.append(b)
    ids: set[int] = set()
    for b in blocks:
        # v0.7.0: stale-error heads are handled history — not "important".
        if b.get("reason") == "stale error":
            continue
        # v0.6.0: trust-risk blocks are never "important" — previously the
        # trace/high-score/definition passes ignored the flag, letting an
        # untrusted instruction block into the keep-set (production QA v3
        # injection finding). The rare-candidate loop above already skips.
        if b.get("trust_risk"):
            continue
        if b["type"] == "trace":
            ids.add(b["id"])
        elif b["entity_hits"] and b["score"] >= 5.0:
            ids.add(b["id"])
        elif b["score"] >= 9 and (b["entity_hits"] or b["term_hits"]):
            ids.add(b["id"])
        elif b["type"] == "definition" and b["entity_hits"]:
            ids.add(b["id"])
    cap = max(3, min(18, math.ceil(n * 0.22)))
    if len(ids) > cap:
        ranked = sorted((blocks[i] for i in ids), key=lambda b: (-b["score"], b["tokens"]))
        ids = {b["id"] for b in ranked[:cap]}
    return ids


def _stitch_neighbors(blocks: list[dict[str, Any]], kept: set[int]) -> set[int]:
    by_id = {b["id"]: b for b in blocks}
    extra: set[int] = set()
    for i in list(kept):
        prev_id = i - 1
        if prev_id in by_id and by_id[prev_id]["type"] == "heading":
            extra.add(prev_id)
        nxt = i + 1
        if nxt in by_id and by_id[i]["type"] == "heading":
            extra.add(nxt)
        # A traceback is one unit: if we keep any part of a stack, keep the
        # whole stack and the log line that triggered it (error/exception).
        if by_id[i]["type"] == "trace" or i in extra and by_id[i]["type"] == "trace":
            for nb in (i - 1, i + 1):
                if nb in by_id and by_id[nb]["type"] in {"trace", "log"}:
                    t = by_id[nb]["text"].lower()
                    if by_id[nb]["type"] == "trace" or re.search(
                        r"error|fail|exception|traceback", t
                    ):
                        extra.add(nb)
    # Code-block stitching: a signature line ("fn delay(...)") and its
    # indented body lines are separate blocks because each indented line
    # classifies as "definition". Keep them whole:
    #   1. kept body line -> pull in the preceding signature (dedent) line
    #   2. kept signature/definition line -> pull in the following
    #      indented definition body while it stays inside the function
    # Without this, the gold "what does this cap at?" fact (in the body)
    # can be dropped while the signature stays, or vice versa.
    for i in list(kept):
        b = by_id[i]
        if b["type"] in {"code", "definition"}:
            # Pull preceding dedent signature line if we keep a body line.
            p = i - 1
            while p in by_id and not by_id[p]["text"].strip():
                p -= 1
            if p in by_id:
                pt = by_id[p]["text"]
                indent = len(pt) - len(pt.lstrip())
                bt = b["text"]
                bindent = len(bt) - len(bt.lstrip())
                if (
                    b["type"] in {"code", "definition"}
                    and bindent > 0
                    and indent == 0
                    and pt.strip()
                ):
                    extra.add(p)
            # Pull following indented definition body lines.
            q = i + 1
            while q in by_id:
                qb = by_id[q]
                qt = qb["text"]
                if qb["type"] in {"definition", "code"} and qt.strip():
                    qindent = len(qt) - len(qt.lstrip())
                    if qindent > 0:
                        extra.add(q)
                        q += 1
                        continue
                    break
                break
    # Structural closure must never undo trust gating. Explicit caller pins
    # remain the sole opt-in override.
    return {
        bid
        for bid in kept | extra
        if not by_id[bid].get("trust_risk") or by_id[bid].get("pinned")
    }


def _block_link_terms(text: str) -> set[str]:
    """Rare relation keys used only for bounded evidence-path closure."""
    out = {e.casefold() for e in _extract_entities(text) if len(e) >= 3}
    for raw in _TOKEN_RE.findall(text or ""):
        term = raw.casefold().strip("؟?!.،,;:…\"'()[]")
        if len(term) < 4 or term in _STOP or term in GENERIC_WORDS:
            continue
        # v0.6.0: pure-number tokens ("1001", "2026") are timestamps/ids of
        # no linking value — every numeric log line would share one and
        # fabricate a "coherent chain" for the ambiguity guard.
        if term.isdigit():
            continue
        if _HEX_ID_RE.match(term):
            continue
        out.add(term)
    return out


def _graph_path_closure(
    blocks: list[dict[str, Any]],
    kept: set[int],
    max_hops: int = 4,
    link_terms: dict[int, set[str]] | None = None,
) -> set[int]:
    """Keep short rare-term paths connecting already-selected evidence."""
    if len(kept) < 2 or not blocks:
        return kept
    # Perf (v0.6.0): callers that already tokenised every block (CFA runs
    # right after this in select_adaptive) pass their map to avoid a second
    # full-document regex pass.
    if link_terms is None:
        terms_by_id = {
            b["id"]: (set() if b.get("trust_risk") else _block_link_terms(b["text"]))
            for b in blocks
        }
    else:
        terms_by_id = {
            b["id"]: (set() if b.get("trust_risk") else link_terms.get(b["id"], set()))
            for b in blocks
        }
    term_ids: dict[str, list[int]] = {}
    for bid, terms in terms_by_id.items():
        for term in terms:
            term_ids.setdefault(term, []).append(bid)
    # Bridge terms must be genuinely rare. The old len//5 ceiling let a
    # term appear in up to 800 of 4000 blocks and still form edges — in
    # homogeneous dumps that turns adjacency into a dense hub graph and
    # the BFS floods (v0.6.0 perf regression). Hard-cap at 12.
    max_df = max(3, min(len(blocks) // 5, 12))
    adjacency: dict[int, set[int]] = {b["id"]: set() for b in blocks}
    for ids in term_ids.values():
        if len(ids) < 2 or len(ids) > max_df:
            continue
        for left in ids:
            adjacency[left].update(right for right in ids if right != left)
    scores = {b["id"]: float(b.get("score", 0.0)) for b in blocks}
    seeds = sorted(kept, key=lambda bid: scores.get(bid, 0.0), reverse=True)[:24]
    closed = set(kept)
    # v0.6.0 contract: bridge nodes are evidence when they EXTEND a seed's
    # evidence path (chain rescue: Moonlight→Selene→DB-77-Z hangs off the
    # query-hit block), but the total extension is bounded so hub terms in
    # homogeneous dumps cannot flood the keep-set. Nodes are admitted
    # closest-seed-distance first, up to a budget proportional to the seed
    # count; everything beyond the budget stays dropped.
    reach: dict[int, int] = {}  # node -> nearest seed BFS depth
    for src in seeds:
        queue = [(src, 0)]
        # Standard BFS: mark depth at enqueue time, but EXPAND a node when
        # it is dequeued unless an equal-or-better depth was already
        # expanded. The previous guard also skipped expansion on first
        # discovery, so chains died after one hop (v0.6.0 regression).
        expanded: set[int] = set()
        for node, depth in queue:
            if depth >= max_hops:
                continue
            d = reach.get(node)
            if d is not None and (d < depth or node in expanded):
                continue
            reach[node] = depth
            expanded.add(node)
            for nxt in adjacency.get(node, ()):
                nd = reach.get(nxt)
                if nd is None or nd > depth + 1:
                    reach[nxt] = depth + 1
                    queue.append((nxt, depth + 1))
    budget = max(8, 2 * len(seeds))
    candidates = sorted(
        ((d, n) for n, d in reach.items() if n not in closed),
        key=lambda x: (x[0], -scores.get(x[1], 0.0)),
    )
    # Bridge nodes must carry some intrinsic value — a near-zero-score
    # block that merely shares a token with a seed is noise, not evidence
    # (v0.6.0: '- ccache' style lines were flooding the keep-set).
    min_bridge_score = min(3.0, max(scores.get(s, 0.0) for s in seeds) * 0.25 if seeds else 3.0)
    for _, node in candidates[:budget]:
        if scores.get(node, 0.0) >= min_bridge_score:
            closed.add(node)
    return closed


def _counterfactual_overlap_ambiguity(
    blocks: list[dict[str, Any]],
    kept: set[int],
    *,
    query: str,
    link_terms: dict[int, set[str]] | None = None,
    semantic_tier: Any = None,
    log_dir: str | Path | None = None,
    pin_patterns: Optional[list[str]] = None,
) -> bool:
    """Detect a high-scoring kept distractor masking a dropped answer chain.

    Fires ONLY when:
      1. A kept block scores high on query-term overlap (the "winner").
      2. Two or more dropped blocks share a *rare, non-generic* link term
         (appearing in ≤3 blocks document-wide, length ≥4, not in
         GENERIC_WORDS) AND at least one of them mentions a query content
         word as a whole word — i.e. a coherent, query-relevant chain
         was evicted.

    This intentionally does NOT fire for generic high-volume logs (git, npm)
    where shared tokens like dates, SHAs, "verbose", "docs", "commit" are
    ubiquitous and would cause false positives.
    """
    kept_with_terms = [
        b
        for b in blocks
        if b["id"] in kept
        and not b.get("trust_risk", False)
        and (b.get("term_hits") or b.get("entity_hits"))
    ]
    if not kept_with_terms:
        return False

    winner = max(
        kept_with_terms,
        key=lambda b: (
            b.get("rare_term_hits", 0)
            + b.get("entity_hits", 0)
            + b.get("term_hits", 0),
            b["score"],
        ),
    )

    content = _query_content_terms(query)
    if not content:
        return False

    # If the kept set is already large enough that the answer is likely
    # present, a single distractor cannot meaningfully mask it.  The guard
    # only fires when the kept set is small (<10% of the document) AND the
    # winner is a single high-scoring block (not a coherent multi-block
    # chain).  This targets the narrow production QA P0: one distractor block
    # outscoring a linked answer chain that was evicted.
    if len(kept) / max(1, len(blocks)) >= 0.10:
        return False

    # Precompile word-boundary patterns for each query content term.
    content_patterns = [re.compile(r"\b" + re.escape(t) + r"\b") for t in content]

    # The dropped chain must be scoring well (near the floor) to count as
    # a "genuinely evicted answer".  Low-scoring dropped blocks are just
    # noise, not a hidden answer.  Use 0.38× the top safe score as the
    # threshold (same floor as the main selection path).
    safe_blocks = [b for b in blocks if not b.get("trust_risk")]
    top_score = max((b["score"] for b in safe_blocks), default=0.0)
    floor = max(2.2, min(11.5, 0.38 * top_score))

    dropped = [
        b
        for b in blocks
        if b["id"] != winner["id"]
        and b["id"] not in kept
        and not b.get("trust_risk", False)
        and b["score"] >= floor
    ]
    if len(dropped) < 2:
        return False

    # Perf bound (v0.6.0): the pair scan below is O(d²) with per-pair set
    # intersections and regex searches. Cap the candidate pool to the top-
    # scoring dropped blocks — a masked answer chain lives at the top of
    # the evicted mass, never in the tail.
    if len(dropped) > 64:
        dropped = sorted(dropped, key=lambda b: -b["score"])[:64]

    if link_terms is not None:
        def _terms_of(b: dict[str, Any]) -> set[str]:
            return set() if b.get("trust_risk") else link_terms.get(b["id"], set())
    else:
        def _terms_of(b: dict[str, Any]) -> set[str]:
            return set() if b.get("trust_risk") else _block_link_terms(b["text"])

    max_df = 3

    # Document-wide df for the candidates' link terms, computed with a
    # cheap substring scan over pre-lowered texts instead of tokenising
    # every block (v0.6.0 perf: 4000 regex passes → ~len(cand_terms)
    # str-in checks). Substring df can overcount when a term is a
    # fragment of a longer word; that only makes the rarity gate MORE
    # conservative (term judged common → not a chain link), which is safe.
    cand_term_set: set[str] = set()
    for b in dropped:
        cand_term_set |= _terms_of(b)
    lower_texts = [b["text"].casefold() for b in blocks]
    term_df: dict[str, set[int]] = {t: set() for t in cand_term_set}
    for idx, lt in enumerate(lower_texts):
        for t in cand_term_set:
            if t in lt:
                term_df[t].add(idx)

    def _meaningful_shared(term: str) -> bool:
        """A shared link term must be rare, long enough, and not generic."""
        if len(term) < 4:
            return False
        if term in GENERIC_WORDS:
            return False
        return len(term_df.get(term, ())) <= max_df

    rare_terms_by_block: dict[int, set[str]] = {}
    for b in dropped:
        terms = _terms_of(b)
        rare = {t for t in terms if _meaningful_shared(t)}
        rare_terms_by_block[b["id"]] = rare

    # Inverted index over the candidates' rare terms: propose pairs by
    # shared term instead of scanning all pairs.
    postings: dict[str, list[int]] = {}
    for b in dropped:
        for t in rare_terms_by_block[b["id"]]:
            postings.setdefault(t, []).append(b["id"])
    candidate_pairs: list[tuple[int, int]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for ids in postings.values():
        if len(ids) < 2 or len(ids) > max_df:
            continue
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                key = (ids[i], ids[j])
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    candidate_pairs.append(key)
    by_id = {b["id"]: b for b in dropped}
    for left_id, right_id in candidate_pairs:
        left = by_id[left_id]
        right = by_id[right_id]
        left_has_content = any(p.search(left["text"].lower()) for p in content_patterns)
        if left_has_content:
            return True
        if any(p.search(right["text"].lower()) for p in content_patterns):
            return True

    # Semantic upgrade (v0.8.0): the lexical scan found no shared-term
    # chain. With a tier attached, ambiguity now means: >=2 dropped
    # candidates score within eps of the winner in embedding space —
    # i.e. paraphrase distractors that share NO vocabulary with the query.
    if semantic_tier is not None:
        try:
            tier = semantic_tier
            if not getattr(tier, "available", False):
                return False
            cands = [
                b for b in blocks
                if b["id"] != winner["id"] and b["id"] not in kept
                and not b.get("trust_risk", False)
            ]
            if len(cands) < 2:
                return False
            capped = sorted(cands, key=lambda b: -b["score"])[:32]
            _, sims = tier.score_against_query(query, [c["text"] for c in capped])
            win_sim = tier.score_against_query(query, [winner["text"]])[1][0]
            near = [c for c, sm in zip(capped, sims) if abs(sm - win_sim) < 0.05]
            return len(near) >= 2
        except Exception:  # tier failure must NEVER break compression
            return False
    return False


def select_adaptive(
    blocks: list[dict[str, Any]],
    *,
    needle_only: bool = False,
    query: str = "",
    semantic_tier: Any = None,
) -> tuple[set[int], bool, str]:
    if not blocks:
        return set(), True, "high"
    signal = _query_signal(blocks)
    if signal < 0.04 and max(b["score"] for b in blocks) < 3.5:
        return {b["id"] for b in blocks}, True, "high"
    hits = {b["id"] for b in blocks if b["entity_hits"] or b["term_hits"]}
    rare = {
        b["id"]
        for b in blocks
        if b.get("rare_term_hits") or b["entity_hits"]
    }
    if needle_only and hits:
        use = rare if rare else hits
        safe_use = {bid for bid in use if not blocks[bid].get("trust_risk")}
        if not safe_use:
            return {b["id"] for b in blocks}, True, "high"
        use = safe_use
        kept = set(use)
        if not blocks[0].get("trust_risk") or blocks[0].get("pinned"):
            kept.add(blocks[0]["id"])
        if (
            not _is_sink_noise(blocks[-1]["text"])
            and not blocks[-1].get("trust_risk")
            and blocks[-1]["id"] in use
        ):
            kept.add(blocks[-1]["id"])
        # Perf (v0.6.0): tokenise once, share with the CFA guard below.
        link_terms_map = {
            b["id"]: (set() if b.get("trust_risk") else _block_link_terms(b["text"]))
            for b in blocks
        }
        kept = _graph_path_closure(blocks, kept, link_terms=link_terms_map)
        # Needle-path saturation guard: on bilingual/mirrored documents the
        # closure floods (EN/JP block pairs share bridge terms) and the
        # "needle" keep-set approaches the whole document. Cap it back to
        # the query-hit core — the needle path's contract is a tight set.
        if len(kept) > max(len(use) * 2, 16) and len(kept) / len(blocks) > 0.6:
            kept = set(use)
            if not blocks[0].get("trust_risk") or blocks[0].get("pinned"):
                kept.add(blocks[0]["id"])
        kept = _stitch_neighbors(blocks, kept)
        if _counterfactual_overlap_ambiguity(
            blocks, kept, query=query, link_terms=link_terms_map, semantic_tier=semantic_tier
        ):
            return {b["id"] for b in blocks}, True, "high"
        return kept, False, "low"
    important = _important(blocks)
    safe_blocks = [b for b in blocks if not b.get("trust_risk")]
    top = max((b["score"] for b in safe_blocks), default=0.0)
    # Floor: 0.38× the top score, but capped so one dominant block
    # (e.g. a head block with many query terms) doesn't evict all the
    # middle. The cap is 11.5 — high enough to keep the top blocks,
    # low enough that scattered facts with scores 8-12 survive.
    # Rate-distortion insight: lossy schemes compound under repeated
    # compaction; a single high scorer shouldn't create a cascade.
    floor = max(2.2, min(11.5, 0.38 * top))
    kept = set(important)
    for b in safe_blocks:
        if b["score"] >= floor:
            kept.add(b["id"])
    # Saturation guard (v0.6.0, production QA v3 performance finding): on huge
    # homogeneous dumps every line matches a broad query, so the floor
    # keeps everything and the payload never shrinks. When the floor pass
    # keeps nearly all blocks AND the query is broad (no distinctive
    # selector), fall back to the important-set only — the ranked top of
    # the document — instead of the whole dump.
    floor_ratio = sum(b["tokens"] for b in safe_blocks if b["id"] in kept) / max(
        1, sum(b["tokens"] for b in safe_blocks)
    )
    if floor_ratio > 0.9 and len(safe_blocks) > 64 and not needle_only:
        kept = set(important)
    # Keep the head. A trust-risk tail is not a safe recency sink.
    # Exception (leanctx old-error purge): a stale-error head is handled
    # history — citations preserve its fact; don't re-admit the dump.
    if (
        (blocks[0].get("reason") != "stale error" or len(kept) == 0)
        and (not blocks[0].get("trust_risk") or blocks[0].get("pinned"))
    ):
        kept.add(blocks[0]["id"])
    if (
        not _is_sink_noise(blocks[-1]["text"])
        and not blocks[-1].get("trust_risk")
    ):
        kept.add(blocks[-1]["id"])
    # Perf (v0.6.0): the graph closure tokenises every block (regex-heavy).
    # When the keep-set has at most one member there is nothing to bridge,
    # and on huge weak-signal documents that is the common case — skip it.
    # Otherwise tokenise ONCE and share the map between the closure and the
    # CFA guard below (saves a second full-document regex pass).
    link_terms_map: dict[int, set[str]] | None = None
    if len(kept) >= 2:
        link_terms_map = {
            b["id"]: (set() if b.get("trust_risk") else _block_link_terms(b["text"]))
            for b in blocks
        }
        kept = _graph_path_closure(blocks, kept, link_terms=link_terms_map)
    kept = _stitch_neighbors(blocks, kept)
    keep_ratio = sum(blocks[i]["tokens"] for i in kept) / max(1, sum(b["tokens"] for b in blocks))
    # Line dumps (git / npm): Super keeps matching records only. Priors
    # (unique SHAs, dates on every line) must not keep the whole log.
    # This check must come before the keep_ratio guard because line records
    # have a well-defined compression strategy that the generic "keep too
    # much -> fail open" escape would preempt.
    texts = [b["text"] for b in blocks]
    if _looks_like_line_records(texts):
        kept = {
            b["id"]
            for b in blocks
            if (b["entity_hits"] or b["term_hits"]) and not b.get("trust_risk")
        }
        if not kept:
            return {b["id"] for b in blocks}, True, "high"
        if not blocks[0].get("trust_risk") or blocks[0].get("pinned"):
            kept.add(blocks[0]["id"])
        if (
            not _is_sink_noise(blocks[-1]["text"])
            and not blocks[-1].get("trust_risk")
        ):
            kept.add(blocks[-1]["id"])
        kept = _stitch_neighbors(blocks, kept)
        keep_ratio = sum(blocks[i]["tokens"] for i in kept) / max(1, sum(b["tokens"] for b in blocks))
    if keep_ratio > 0.92 and signal < 0.12:
        return {b["id"] for b in blocks}, True, "high"
    if _counterfactual_overlap_ambiguity(
        blocks, kept, query=query, link_terms=link_terms_map, semantic_tier=semantic_tier
    ):
        return {b["id"] for b in blocks}, True, "high"
    risk = "low"
    if keep_ratio > 0.8:
        risk = "medium"
    if not important and signal < 0.1:
        risk = "high"
    return kept, False, risk


def select_fixed(blocks: list[dict[str, Any]], budget_ratio: float) -> tuple[set[int], bool, str]:
    if not blocks:
        return set(), True, "high"
    total = sum(b["tokens"] for b in blocks)
    budget = max(1, int(round(total * budget_ratio)))
    important = _important(blocks)
    ranked = sorted(
        (b for b in blocks if not b.get("trust_risk")),
        key=lambda b: (-b["score"], b["start"]),
    )
    kept: set[int] = set(important)
    used = sum(b["tokens"] for b in blocks if b["id"] in kept)
    for b in ranked:
        if b["id"] in kept:
            continue
        if used >= budget and used >= max(1, budget // 2):
            # still take a high-score leftover if we have almost no query hits
            if b["score"] < 6:
                continue
        if used + b["tokens"] > budget * 1.15 and used >= budget * 0.6 and b["id"] not in important:
            continue
        kept.add(b["id"])
        used += b["tokens"]
        if used >= budget and len(kept) >= max(2, len(important)):
            break
    if not blocks[0].get("trust_risk") or blocks[0].get("pinned"):
        kept.add(blocks[0]["id"])
    if (
        not _is_sink_noise(blocks[-1]["text"])
        and (not blocks[-1].get("trust_risk") or blocks[-1].get("pinned"))
    ):
        kept.add(blocks[-1]["id"])
    kept = _stitch_neighbors(blocks, kept)
    return kept, False, "low"


def _query_content_terms(query: str) -> set[str]:
    """Content words in the query, after generic stop-words are removed."""
    return set(_extract_terms(query))


def _entity_recall(original: str, compressed: str, query: str) -> float:
    keys = []
    for item in _extract_entities(query) + _extract_terms(query):
        if item.lower() in original.lower() and item not in keys:
            keys.append(item)
    if not keys:
        return 1.0
    hit = sum(1 for k in keys if k.lower() in compressed.lower())
    return hit / len(keys)


def _extract_critical_lines(text: str) -> list[str]:
    """Extract lines that look like critical facts: error lines, key-value
    pairs with identifiers, tracebacks, and lines containing numbers + words.
    Mirrors Baseline RAG's critical_lines_total/kept/dropped reporting.
    """
    critical: list[str] = []
    for line in text.split("\n"):
        t = line.strip()
        if not t:
            continue
        # Error/exception/traceback lines
        if re.search(r"\b(error|exception|traceback|fail|fatal|timeout|denied|invalid)\b", t, re.I):
            critical.append(t)
        # Lines with key-value patterns (config, settings, identifiers)
        elif re.search(r"[\w]+[=:]\s*[\w][\w.-]*", t) and len(t) > 15:
            critical.append(t)
        # Lines containing specific numbers (dates, IDs, versions, counts)
        elif re.search(r"\b\d{4}-\d{2}-\d{2}\b|\b\d+\.\d+\.\d+\b|\b\d{6,}\b|\b\d+\s*(ms|s|sec|bytes|kb|mb|gb)\b", t):
            critical.append(t)
    return critical


def _critical_line_recall(original: str, compressed: str) -> tuple[int, int]:
    """Return (kept, total) counts of critical lines, matching Baseline RAG
    verifier's critical_lines_kept/critical_lines_total fields."""
    critical = _extract_critical_lines(original)
    if not critical:
        return 0, 0
    kept = 0
    for line in critical:
        # A critical line is "kept" if its core content (first 80 chars,
        # lowercased) appears in the compressed text.
        core = line.lower()[:80]
        if core in compressed.lower():
            kept += 1
    return kept, len(critical)


def verify_compression(
    original: str,
    compressed: str,
    query: str,
) -> dict[str, Any]:
    """Post-compression self-check, mirroring Baseline RAG's verifier object.

    Returns a dict with:
      entity_recall     — fraction of query entities/terms present in compressed
      keyword_recall    — fraction of query terms present in compressed
      important_kept_pct — fraction of critical lines retained
      critical_lines_total / kept / dropped
      risk              — "low" / "medium" / "high" based on recall thresholds
      score             — overall 0-1 quality score (weighted blend)

    This is a diagnostic pass: it does NOT modify the compressed output.
    Callers can use the result to decide whether to re-compress with a
    different strategy, expand dropped blocks, or fail open.
    """
    entity_recall = _entity_recall(original, compressed, query)

    # Keyword recall: fraction of query terms that appear in compressed
    terms = _extract_terms(query)
    if terms:
        hit_terms = sum(1 for t in terms if t in compressed.lower())
        keyword_recall = hit_terms / len(terms)
    else:
        keyword_recall = 1.0

    cl_kept, cl_total = _critical_line_recall(original, compressed)
    important_kept_pct = (cl_kept / cl_total) if cl_total > 0 else 1.0
    critical_lines_dropped = cl_total - cl_kept

    # Risk assessment: combine all three signals
    risk = "low"
    if entity_recall < 0.5:
        risk = "high"
    elif entity_recall < 0.8:
        risk = "medium"
    if keyword_recall < 0.5 and risk == "low":
        risk = "medium"
    elif keyword_recall < 0.3:
        risk = "high"
    if important_kept_pct < 0.5 and risk != "high":
        risk = "high" if important_kept_pct < 0.3 else "medium"

    # Overall score: weighted blend (entity recall matters most for
    # answer quality, keyword recall next, critical lines last)
    score = round(
        0.5 * entity_recall + 0.3 * keyword_recall + 0.2 * important_kept_pct,
        3,
    )

    return {
        "entity_recall": round(entity_recall, 3),
        "keyword_recall": round(keyword_recall, 3),
        "important_kept_pct": round(important_kept_pct, 3),
        "critical_lines_total": cl_total,
        "critical_lines_kept": cl_kept,
        "critical_lines_dropped": critical_lines_dropped,
        "risk": risk,
        "score": score,
    }


def _render(
    lines: list[str],
    blocks: list[dict[str, Any]],
    kept: set[int],
    citations: bool = False,
    reorder_best: bool = False,
) -> str:
    chronological_blocks = sorted(blocks, key=lambda block: block["start"])
    block_starts = [block["start"] for block in chronological_blocks]
    order = sorted((b for b in blocks if b["id"] in kept), key=lambda b: b["start"])
    if reorder_best and len(order) >= 3:
        # Lost-in-the-middle mitigation (LongLLMLingua / twotrim): models
        # attend hardest to window edges. Best-scoring block anchors the
        # front, second-best anchors the end, the rest stay chronological.
        ranked = sorted(order, key=lambda b: -float(b.get("score", 0.0)))
        best, second = ranked[0], ranked[1]
        rest = [b for b in order if b["id"] not in {best["id"], second["id"]}]
        order = [best, *rest, second]
    parts: list[str] = []
    prev_end = -1
    for b in order:
        if prev_end >= 0 and b["start"] > prev_end + 1:
            if citations:
                # ARC-style citation: name every dropped block so the model
                # knows it can recall the exact head/tail by hash later.
                # Only emit citations when they're net-positive: the stub
                # tokens must be less than the dropped block tokens they replace.
                left = bisect.bisect_right(block_starts, prev_end)
                right = bisect.bisect_left(block_starts, b["start"])
                dropped = [
                    block
                    for block in chronological_blocks[left:right]
                    if block["id"] not in kept and block["end"] < b["start"]
                ]
                gap_tokens = 0
                dropped_tokens = 0
                for d in dropped:
                    dropped_tokens += d["tokens"]
                    head = d["text"].split("\n", 1)[0][:60]
                    tail = d["text"].rsplit("\n", 1)[-1][:60]
                    stub = f'[§] "{head}"…"{tail}"'
                    gap_tokens += estimate_tokens(stub)
                if gap_tokens < dropped_tokens:
                    # Citations save tokens — emit them. For long runs of
                    # consecutive dropped blocks, per-block stubs defeat
                    # the purpose (v0.6.0: 4000-block dump → 500KB of
                    # stubs). Collapse runs longer than 8 into one summary.
                    if len(dropped) > 8:
                        first = dropped[0]
                        last = dropped[-1]
                        parts.append(
                            f"[§run:{len(dropped)} blocks #{first['start']}–{last['end']} elided]"
                        )
                    else:
                        for d in dropped:
                            digest = hashlib.sha256(d["text"].encode("utf-8")).hexdigest()[:8]
                            head = d["text"].split("\n", 1)[0][:60]
                            tail = d["text"].rsplit("\n", 1)[-1][:60]
                            parts.append(f'[§{digest}] "{head}"…"{tail}"')
                else:
                    # Citations would add more tokens than they save — use bare gap
                    parts.append("[…]")
            else:
                parts.append("[…]")
        start, end = b["start"], b["end"]
        # Structural verbatim invariant (leanctx-inspired): a kept fenced
        # code block must survive whole — extend the keep range to the
        # closing fence if it lands just outside, so agents see exact tokens.
        kept_text = lines[start : end + 1]
        fences = sum(1 for ln in kept_text if ln.strip().startswith("```"))
        if fences % 2 == 1:
            j = end + 1
            while j < len(lines):
                if lines[j].strip().startswith("```"):
                    end = j
                    break
                j += 1
        parts.extend(lines[start : end + 1])
        prev_end = end
    return "\n".join(parts)


def cache_wrap(compressed: str, query: str) -> str:
    return (
        "<compressed_context version=\"1\">\n"
        "The following text is extractive context: original wording only, "
        "low-value blocks removed. Use it to answer the user.\n\n"
        "--- context ---\n"
        f"{compressed}\n"
        "--- end context ---\n"
        "</compressed_context>\n\n"
        f"Answer the question: {query}"
    )


def _block_fingerprint(block: dict[str, Any]) -> str:
    """Stable fingerprint for freeze-on-first-sight: type + id + content.

    The block id distinguishes repeated identical records that can receive
    different head/middle/tail decisions. Appending turns preserves prior ids.
    """
    content_hash = hashlib.sha256(block["text"].encode("utf-8")).hexdigest()[:12]
    return f"{block['type']}:{block.get('id', '?')}:{content_hash}"


def _apply_freeze(
    decision_cache: dict,
    scored: list[dict[str, Any]],
    original_text: str,
    query: str = "",
) -> list[dict[str, Any]]:
    """Replay stored keep/drop decisions for blocks whose fingerprint matches.

    The cache maps block fingerprint -> 'keep' | 'drop'. On a multi-turn
    session, the first turn that sees a block makes the decision; later turns
    replay it byte-identically so the provider prompt cache stays warm.

    New blocks remain undecided until selection has completed. Their actual
    keep/drop outcome is recorded by :func:`_record_freeze_decisions`.
    """
    cache_key = "decisions"
    schema_version = decision_cache.get("schema_version")
    if schema_version is not None and schema_version != 1:
        for stale_key in (
            cache_key,
            "query_hash",
            "ctx_hash",
            "ctx_prefix_len",
        ):
            decision_cache.pop(stale_key, None)
    if not isinstance(decision_cache.get(cache_key), dict):
        decision_cache[cache_key] = {}
    decisions: dict = decision_cache[cache_key]
    decision_cache["schema_version"] = 1

    query_hash = hashlib.sha256((query or "").encode("utf-8")).hexdigest()[:12]
    cached_query_hash = decision_cache.get("query_hash")
    if isinstance(cached_query_hash, str) and cached_query_hash != query_hash:
        decisions = {}
        decision_cache[cache_key] = decisions
    decision_cache["query_hash"] = query_hash

    # Store a hash and length for the original prefix. Length matters when the
    # first turn is shorter than 500 chars: hashing a fresh 500-char slice on
    # the next turn would include appended content and falsely invalidate the
    # cache. No plaintext context is retained in the decision cache.
    cached_prefix_len = decision_cache.get("ctx_prefix_len")
    if not isinstance(cached_prefix_len, int) or cached_prefix_len < 0:
        cached_prefix_len = min(len(original_text), 500)
        decision_cache["ctx_prefix_len"] = cached_prefix_len
    ctx_hash = hashlib.sha256(
        original_text[:cached_prefix_len].encode("utf-8")
    ).hexdigest()[:12]
    if "ctx_hash" not in decision_cache:
        decision_cache["ctx_hash"] = ctx_hash
    elif decision_cache["ctx_hash"] != ctx_hash:
        # Context changed fundamentally — clear decisions to avoid stale replay.
        decisions = {}
        decision_cache[cache_key] = decisions
        cached_prefix_len = min(len(original_text), 500)
        decision_cache["ctx_prefix_len"] = cached_prefix_len
        ctx_hash = hashlib.sha256(
            original_text[:cached_prefix_len].encode("utf-8")
        ).hexdigest()[:12]
        decision_cache["ctx_hash"] = ctx_hash

    # Tag old decisions for enforcement after the normal selector runs.
    result = []
    for b in scored:
        fp = _block_fingerprint(b)
        decision = decisions.get(fp)
        if decision not in {"keep", "drop"}:
            decision = None
        result.append(
            {
                **b,
                "frozen": decision is not None,
                "freeze_decision": decision,
            }
        )
    return result


def _enforce_frozen_decisions(
    scored: list[dict[str, Any]], kept: set[int]
) -> set[int]:
    """Apply cached decisions without allowing them to bypass trust or pins."""
    enforced = set(kept)
    for b in scored:
        bid = b["id"]
        if b.get("pinned"):
            enforced.add(bid)
            continue
        decision = b.get("freeze_decision")
        if decision == "drop":
            enforced.discard(bid)
        elif decision == "keep" and not b.get("trust_risk"):
            enforced.add(bid)
    return enforced


def _record_freeze_decisions(
    decision_cache: dict,
    scored: list[dict[str, Any]],
    kept: set[int],
) -> bool:
    """Record first-sight outcomes, preserving the oldest bounded prefix."""
    decisions = decision_cache.setdefault("decisions", {})
    if not isinstance(decisions, dict):
        decisions = {}
        decision_cache["decisions"] = decisions
    saturated = False
    for b in scored:
        fp = _block_fingerprint(b)
        if fp in decisions:
            continue
        if len(decisions) >= FREEZE_MAX_DECISIONS:
            saturated = True
            break
        decisions[fp] = "keep" if b["id"] in kept else "drop"
    return saturated


def _ccr_metadata(data: Any) -> tuple[float, float] | None:
    """Return validated ``(stored_at, ttl)`` metadata for a CCR record."""
    if not isinstance(data, dict):
        return None
    try:
        stored = float(data["stored_at"])
        ttl = float(data.get("ttl", CCR_TTL_SECONDS))
    except (KeyError, TypeError, ValueError):
        return None
    if (
        not math.isfinite(stored)
        or not math.isfinite(ttl)
        or stored <= 0
        or ttl <= 0
    ):
        return None
    return stored, ttl


def sweep_ccr_cache(
    ccr_dir: str | Path = DEFAULT_CCR_DIR,
    *,
    now: float | None = None,
    max_records: int | None = None,
) -> int:
    """Delete expired valid CCR records and return the removal count.

    Malformed or unreadable files are left untouched: retention maintenance is
    best-effort and must not destroy data it cannot validate.
    """
    global _CCR_SWEEP_CURSOR
    root = Path(ccr_dir)
    if not root.is_dir():
        return 0
    current = time.time() if now is None else float(now)
    removed = 0
    limit = None if max_records is None else max(0, int(max_records))
    # Cursor progression only works when every call sees the same ordering;
    # filesystem glob order is unspecified and may change between writes.
    records = sorted(root.glob("*.json"), key=lambda path: path.name)
    if limit is not None and records:
        total = len(records)
        start = _CCR_SWEEP_CURSOR % total
        count = min(limit, total)
        records = [records[(start + offset) % total] for offset in range(count)]
        _CCR_SWEEP_CURSOR += count
    for fp in records:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, RecursionError):
            continue
        metadata = _ccr_metadata(data)
        if (
            metadata is not None
            and metadata[0] <= current + CCR_MAX_CLOCK_SKEW_SECONDS
            and current - metadata[0] <= metadata[1]
        ):
            continue
        try:
            fp.unlink()
            removed += 1
        except OSError:
            continue
    return removed


def _ccr_store(original: str, ccr_dir: str | Path) -> dict[str, Any]:
    path = Path(ccr_dir)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink() or not path.is_dir():
        raise OSError(f"CCR directory must be a real directory: {path}")
    try:
        path.chmod(0o700)
    except OSError:
        pass
    sweep_ccr_cache(path, max_records=CCR_SWEEP_MAX_RECORDS)
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:24]
    payload = {
        "hash": digest,
        "stored_at": time.time(),
        "ttl": CCR_TTL_SECONDS,
        "original": original,
    }
    record = path / f"{digest}.json"
    encoded = json.dumps(payload).encode("utf-8")
    fd: int | None = None
    temporary: str | None = None
    try:
        fd, temporary = tempfile.mkstemp(prefix=".ccr-", dir=path)
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fd = None
            fh.write(encoded)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, record)
        temporary = None
        record.chmod(0o600)
    except OSError:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        raise
    finally:
        if temporary is not None:
            try:
                Path(temporary).unlink()
            except OSError:
                pass
    return {"hash": digest, "path": str(record)}


def _write_private_atomic(path: str | Path, content: str) -> None:
    """Atomically replace a UTF-8 sidecar with owner-only permissions."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        target.parent.chmod(0o700)
    except OSError:
        pass
    fd: int | None = None
    temporary: str | None = None
    try:
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fd = None
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, target)
        temporary = None
        target.chmod(0o600)
    finally:
        if fd is not None:
            os.close(fd)
        if temporary is not None:
            try:
                Path(temporary).unlink()
            except OSError:
                pass


def _clear_tool_payloads(text: str) -> str:
    """Strategy='clear': drop re-fetchable tool-result payloads.

    Finds lines that look like tool output (indented JSON, long indented
    blocks after a tool-call line) and replaces them with a one-line
    marker. The tool name and call signature are preserved. This is the
    cheapest compression step — zero LLM cost, just regex.

    Patterns handled:
    - Lines starting with 2+ spaces that contain JSON braces/brackets
      (tool output indented under a tool-call line)
    - Long JSON blobs (> 200 chars) that appear to be tool results
    - Multi-line indented blocks following a 'TOOL:' or '> ' prefix

    Returns the text with payloads replaced by [tool-result: N chars cleared].
    """
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        # Detect tool-result start: indented JSON, or a line that looks
        # like a tool output marker.
        is_tool_result = False
        if (
            line.startswith("  ") or line.startswith("\t")
        ) and (
            stripped.startswith("{")
            or stripped.startswith("[")
            or stripped.startswith('"')
        ):
            is_tool_result = True
        elif stripped.startswith("TOOL:") and len(stripped) > 30:
            is_tool_result = True
        elif re.match(r"^\s{2,}\w[^(]*\(", line) and "=>" in stripped:
            # Indented function-call line with an arrow (tool call result)
            is_tool_result = True

        if is_tool_result:
            # Collect the full payload (consecutive indented lines)
            j = i
            total_chars = 0
            while j < len(lines):
                payload_line = lines[j]
                if payload_line.strip() == "":
                    # Blank line might end the payload or be part of it.
                    # Look ahead: if the next non-blank line is also indented,
                    # the payload continues.
                    k = j + 1
                    while k < len(lines) and lines[k].strip() == "":
                        k += 1
                    if k < len(lines) and (
                        lines[k].startswith("  ") or lines[k].startswith("\t")
                    ):
                        j += 1
                        continue
                    else:
                        break
                elif lines[j].startswith("  ") or lines[j].startswith("\t"):
                    total_chars += len(payload_line)
                    j += 1
                else:
                    break
            total_chars += len(line)
            # Only clear if the payload is substantial (> 100 chars)
            if total_chars > 100:
                # Extract the tool name if visible
                tool_name = ""
                m = re.search(r"(?:TOOL:|=>)\s*(\w+)", line)
                if m:
                    tool_name = m.group(1)
                marker = f"  [tool-result{':' + tool_name if tool_name else ''}: {total_chars} chars cleared — re-fetch if needed]"
                out.append(marker)
                i = j
                continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _summarise_with_llm(
    text: str,
    query: str,
    target_ratio: float = 0.25,
    *,
    endpoint: str | None = None,
    model_candidates: Iterable[str] | str | None = None,
    timeout: float | None = None,
    allow_remote: bool | None = None,
) -> Optional[str]:
    """Strategy='summarise': use a local LLM to compress the context.

    Calls an OpenAI-compatible endpoint. Endpoint, model candidates, and the
    total retry budget can be supplied directly or through the
    ``TAMERU_SUMMARY_ENDPOINT``, ``TAMERU_SUMMARY_MODELS``, and
    ``TAMERU_SUMMARY_TIMEOUT`` environment variables.
    Returns the summarised text, or None on any failure (fail-open).

    The prompt asks the model to produce a faithful summary that:
    - Preserves all entities, identifiers, and facts
    - Keeps the query-relevant content verbatim
    - Drops filler and re-fetchable tool payloads
    - Targets ~target_ratio of the original length

    This is the "smart" end of the strategy ladder: highest quality,
    highest cost (one LLM call). Falls back to None (caller fails open)
    on any error: model unavailable, timeout, JSON parse failure, etc.
    """
    import urllib.request
    import urllib.error
    import urllib.parse

    # Target length in tokens
    target_tokens = max(50, int(estimate_tokens(text) * target_ratio))

    prompt = (
        f"Summarise the following context faithfully. Keep ALL entities, "
        f"identifiers, names, numbers, dates, and specific facts verbatim. "
        f"Drop filler, lorem ipsum, and re-fetchable tool output. "
        f"The user's question is: \"{query}\"\n"
        f"Target length: approximately {target_tokens} tokens.\n"
        f"Preserve the original wording for key facts. Do not add new information.\n\n"
        f"CONTEXT:\n{text[:60000]}"  # Cap input at 60k chars to avoid overflow
    )

    endpoint = endpoint or os.environ.get("TAMERU_SUMMARY_ENDPOINT") or DEFAULT_SUMMARY_ENDPOINT
    if allow_remote is None:
        allow_remote = os.environ.get("TAMERU_SUMMARY_ALLOW_REMOTE", "").strip().lower() in {
            "1", "true", "yes", "on"
        }
    try:
        parsed_endpoint = urllib.parse.urlsplit(endpoint)
    except (TypeError, ValueError):
        return None
    if parsed_endpoint.scheme not in {"http", "https"} or not parsed_endpoint.hostname:
        return None
    local_hosts = {"127.0.0.1", "::1", "localhost"}
    if parsed_endpoint.hostname.casefold() not in local_hosts and not allow_remote:
        return None
    if model_candidates is None:
        configured = os.environ.get("TAMERU_SUMMARY_MODELS", "")
        models = tuple(m.strip() for m in configured.split(",") if m.strip())
        if not models:
            models = DEFAULT_SUMMARY_MODELS
    elif isinstance(model_candidates, str):
        models = tuple(m.strip() for m in model_candidates.split(",") if m.strip())
    else:
        models = tuple(str(m).strip() for m in model_candidates if str(m).strip())
    if not models:
        return None
    if timeout is None:
        try:
            timeout = float(os.environ.get("TAMERU_SUMMARY_TIMEOUT", DEFAULT_SUMMARY_TIMEOUT))
        except (TypeError, ValueError):
            timeout = DEFAULT_SUMMARY_TIMEOUT
    try:
        timeout = float(timeout)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(timeout) or timeout <= 0:
        return None
    timeout = max(0.1, timeout)
    deadline = time.monotonic() + timeout

    for model in models:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            req = urllib.request.Request(
                endpoint,
                data=json.dumps({
                    "model": model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": target_tokens,
                    "temperature": 0.1,
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=remaining) as resp:
                raw = resp.read(MAX_SUMMARY_RESPONSE_BYTES + 1)
                if len(raw) > MAX_SUMMARY_RESPONSE_BYTES:
                    continue
                data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                continue
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content", "")
            if not isinstance(content, str):
                continue
            if not content or len(content.strip()) < 20:
                continue
            return content.strip()
        except (
            urllib.error.URLError,
            OSError,
            ValueError,
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TimeoutError,
            TypeError,
            AttributeError,
            OverflowError,
            RecursionError,
        ):
            continue
    return None


_SUMMARY_STRUCTURED_TOKEN_RE = re.compile(
    r"\b(?:[A-Za-z][A-Za-z0-9_.:/-]*\d[A-Za-z0-9_.:/-]*|\d+(?:[.:/-]\d+)*)\b"
)


def _summary_preserves_required_facts(
    source: str, summary: str, query: str
) -> tuple[bool, float]:
    """Conservative factual gate for the optional model-summary strategy."""
    source_fold = source.casefold()
    summary_fold = summary.casefold()
    required: list[str] = []
    candidates = (
        _SUMMARY_STRUCTURED_TOKEN_RE.findall(query or "")
        + _extract_entities(query or "")
        + _extract_terms(query or "")
    )
    for item in candidates:
        token = str(item).strip().casefold()
        if token and token in source_fold and token not in required:
            required.append(token)

    # Values answering the question often appear only in the matching source
    # line, not in the question itself (for example query host db-prod-01 and
    # answer port 5432). Preserve structured values from source segments tied
    # to a query identifier, or to at least two lexical query terms.
    query_ids = {
        token.casefold()
        for token in _SUMMARY_STRUCTURED_TOKEN_RE.findall(query or "")
    }
    query_terms = {
        token.casefold()
        for token in (_extract_entities(query or "") + _extract_terms(query or ""))
        if token
    }
    for segment in re.split(r"[\r\n]+|(?<=[.!?])\s+", source):
        segment_fold = segment.casefold()
        id_match = bool(query_ids) and any(
            token in segment_fold for token in query_ids
        )
        term_hits = sum(1 for token in query_terms if token in segment_fold)
        if not id_match and term_hits < 2:
            continue
        for item in _SUMMARY_STRUCTURED_TOKEN_RE.findall(segment):
            token = item.casefold()
            if token not in required:
                required.append(token)
        # Plain-text scalar answers are not necessarily identifiers (for
        # example "paint color is ultraviolet"). Require the first content
        # term after a queried field and a simple relation verb; do not require
        # every adjective in a long source segment.
        for query_term in sorted(query_terms, key=len, reverse=True):
            match = re.search(
                rf"(?<!\w){re.escape(query_term)}(?!\w)\s+"
                r"(?:is|are|was|were|equals?|uses?|=)\s+(.+)$",
                segment,
                re.I,
            )
            if not match:
                continue
            answer_terms = _extract_terms(match.group(1))
            if answer_terms:
                token = answer_terms[0].casefold()
                if token not in query_terms and token not in required:
                    required.append(token)
            break
    kept = sum(1 for token in required if token in summary_fold)
    recall = kept / len(required) if required else 1.0

    source_ids = {
        token.casefold() for token in _SUMMARY_STRUCTURED_TOKEN_RE.findall(source)
    }
    summary_ids = {
        token.casefold() for token in _SUMMARY_STRUCTURED_TOKEN_RE.findall(summary)
    }
    no_novel_ids = summary_ids.issubset(source_ids)
    return recall == 1.0 and no_novel_ids, recall


def retrieve(ccr_hash: str, ccr_dir: str | Path = DEFAULT_CCR_DIR) -> Optional[str]:
    if not isinstance(ccr_hash, str) or not _CCR_HASH_RE.fullmatch(ccr_hash):
        return None
    root = Path(ccr_dir)
    if root.is_symlink() or not root.is_dir():
        return None
    fp = root / f"{ccr_hash}.json"
    try:
        info = fp.lstat()
    except OSError:
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, RecursionError):
        return None
    if not isinstance(data, dict) or data.get("hash") != ccr_hash:
        return None
    metadata = _ccr_metadata(data)
    current = time.time()
    if (
        metadata is None
        or metadata[0] > current + CCR_MAX_CLOCK_SKEW_SECONDS
        or current - metadata[0] > metadata[1]
    ):
        try:
            fp.unlink()
        except OSError:
            pass
        return None
    original = data.get("original")
    if not isinstance(original, str):
        return None
    if hashlib.sha256(original.encode("utf-8")).hexdigest()[:24] != ccr_hash:
        return None
    return original


# ---------------------------------------------------------------------------
# JSON preprocessor (Baseline RAG-inspired)
#
# Detects JSON payloads in the context and crushes redundant array items.
# Keeps only query-relevant items + a bounded sample of the rest, preserving
# the top-level structure. This is the "JSON policy" from Baseline RAG's
# type-specific policy system.
# ---------------------------------------------------------------------------

def _detect_json_payloads(text: str) -> list[tuple[int, int, str]]:
    """Find JSON object/array spans in the text.

    Returns a list of (start, end, json_string) tuples where start/end are
    character offsets into the original text.
    """
    payloads = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "{[":
            # Try to find the matching close
            depth = 0
            in_string = False
            escape = False
            j = i
            while j < n:
                c = text[j]
                if escape:
                    escape = False
                elif c == "\\" and in_string:
                    escape = True
                elif c == '"':
                    in_string = not in_string
                elif not in_string:
                    if c in "{[":
                        depth += 1
                    elif c in "}]":
                        depth -= 1
                        if depth == 0:
                            break
                j += 1
            if j < n and depth == 0:
                candidate = text[i:j + 1]
                # Only count if it's a reasonable JSON payload (> 200 chars)
                if len(candidate) > 200:
                    payloads.append((i, j + 1, candidate))
                i = j + 1
            else:
                # The first unmatched opener already scanned the remaining
                # suffix. Restarting at every later opener makes malformed
                # input quadratic. Stop and preserve the original text; the
                # caller's fail-open path is safer than speculative recovery.
                break
        else:
            i += 1
    return payloads


def _crush_json_items(payload: str, query: str, max_keep: int = 50) -> str | None:
    """Crush redundant array items in a JSON payload.

    Finds the top-level array (or arrays under known keys like "items"),
    keeps only query-relevant items + up to max_keep others, and returns
    the re-serialized JSON. Returns None if no array was found or the
    result is not actually shorter.
    """
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, RecursionError):
        return None

    # Find arrays to crush: top-level list, or dicts with list values
    # whose items are dicts (the "catalog" pattern).
    arrays_to_crush: list[tuple] = []  # (path, list_obj)

    def _find_arrays(obj, path=""):
        if isinstance(obj, list) and len(obj) > max_keep:
            # Check if items are dicts (the pattern we want to crush)
            if obj and isinstance(obj[0], dict):
                arrays_to_crush.append((path, obj))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _find_arrays(v, f"{path}.{k}" if path else k)

    _find_arrays(data)
    if not arrays_to_crush:
        return None

    if not query_has_distinctive_selectors(query):
        return None
    query_terms = set(_json_query_needles(query))
    if not query_terms:
        return None
    query_selectors = _selector_patterns(query_terms)

    total_crushed = 0
    for path, arr in arrays_to_crush:
        kept_indices = []
        dropped_count = 0
        for i, item in enumerate(arr):
            item_str = json.dumps(item, ensure_ascii=False)
            if any(selector.search(item_str) for selector in query_selectors):
                kept_indices.append(i)
            else:
                dropped_count += 1
        # No first-50 fallback. If nothing matched, do not crush this array.
        if not kept_indices:
            continue

        if dropped_count == 0:
            continue

        # Rebuild the array with only kept items
        new_arr = [arr[i] for i in sorted(kept_indices)]
        total_crushed += dropped_count

        # Walk back to the array location and replace it
        if path:
            parts = path.split(".")
            target = data
            for p in parts[:-1]:
                target = target[p]
            target[parts[-1]] = new_arr
        else:
            data = new_arr

    if total_crushed == 0:
        return None

    result = json.dumps(data, indent=2, ensure_ascii=False)
    # Only use the crushed version if it's actually shorter
    if len(result) < len(payload):
        return result
    return None


def _preprocess_json(text: str, query: str) -> str:
    """Detect and crush JSON payloads in the text.

    Finds JSON spans, crushes redundant array items, and splices the
    crushed versions back into the text. Falls back to the original text
    if no JSON is found or crushing doesn't help.
    """
    payloads = _detect_json_payloads(text)
    if not payloads:
        return text

    modified = False
    result = text
    for start, end, payload in reversed(payloads):  # reverse to preserve offsets
        crushed = _crush_json_items(payload, query)
        if crushed is not None:
            result = result[:start] + crushed + result[end:]
            modified = True

    return result if modified else text


def inspect_compressibility(text: str, query: str) -> dict[str, Any]:
    """Pre-compression check (PAKT-inspired): is compression worth running?

    Returns {"worth_it": bool, "repetition_ratio": float, "reason": str}.
    Cheap heuristics only — no compression is performed.
    """
    tokens = estimate_tokens(text)
    if tokens < 200:
        return {
            "worth_it": False,
            "repetition_ratio": 0.0,
            "reason": f"too small ({tokens} tokens < 200)",
        }
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return {"worth_it": False, "repetition_ratio": 0.0, "reason": "empty"}
    # Repetition ratio: fraction of duplicate lines (exact + normalised runs).
    from collections import Counter

    norm = [re.sub(r"\d+", "#", ln) for ln in lines]
    counts = Counter(norm)
    dup_lines = sum(c for c in counts.values() if c > 1)
    repetition_ratio = dup_lines / len(lines)
    # Distinct-line ratio as a secondary signal: highly unique prose with no
    # query hits rarely compresses extractively without damage.
    worth = repetition_ratio >= 0.3 or tokens >= 5000
    return {
        "worth_it": worth,
        "repetition_ratio": round(repetition_ratio, 3),
        "reason": (
            f"repetition {repetition_ratio:.0%}, {tokens} tokens"
            if worth
            else f"low repetition ({repetition_ratio:.0%}) and moderate size"
        ),
    }


def compress_context(
    context: str,
    query: str,
    *,
    mode: str = "adaptive",
    budget_ratio: Optional[float] = None,
    ccr: bool = True,
    cache_prefix: bool = False,
    citations: bool = True,
    ccr_dir: str | Path = DEFAULT_CCR_DIR,
    decision_cache: Optional[dict] = None,
    strategy: str = "extract",
    ambiguity_fail_open: bool = False,
    reorder_best: bool = False,
    semantic_tier: Any = None,
    log_dir: str | Path | None = None,
    pin_patterns: Optional[list[str]] = None,
    summary_endpoint: str | None = None,
    summary_models: Iterable[str] | str | None = None,
    summary_timeout: float | None = None,
    summary_allow_remote: bool | None = None,
    limits: IndustrialLimits | None = None,
) -> CompressResult:
    caller_text = str(context or "")
    requested_mode = (mode or "adaptive").strip().lower()
    if requested_mode not in {"adaptive", "fixed", "compiler", "precision"}:
        raise ValueError(
            f"unknown mode {mode!r} (expected adaptive/fixed/compiler/precision)"
        )
    mode_norm = (
        "adaptive" if requested_mode in {"compiler", "precision"} else requested_mode
    )
    industrial = industrial_preprocess(caller_text, query or "", limits)
    if industrial.hard_fail_open:
        tokens = estimate_tokens(caller_text)
        reason = industrial.reason or "industrial preflight failed open"
        fail_result = CompressResult(
            compressed_text=caller_text,
            original_tokens=tokens,
            kept_tokens=tokens,
            tokens_saved_pct=0.0,
            policy_name="local-fail-open",
            mode=requested_mode,
            keep_ratio=1.0,
            tokens_saved=0,
            kept_line_ratio=1.0,
            compression_risk="high",
            confidence=0.0,
            content_type=industrial.profile.format,
            fail_open=True,
            reasons=[reason],
            receipt={
                "schema_version": "1",
                "engine": "tameru",
                "policy": "local-fail-open",
                "query_hash": hashlib.sha256(
                    (query or "").encode("utf-8")
                ).hexdigest()[:12],
                "savings_pct": 0.0,
                "risk": "high",
                "industrial": industrial.to_dict(),
            },
        )
        assert fail_result.receipt is not None
        fail_result.receipt.update(
            _receipt_hashes(
                caller_text,
                caller_text,
                industrial,
                mode=requested_mode,
                strategy=strategy,
                budget_ratio=budget_ratio,
                citations=citations,
            )
        )
        return fail_result
    original_text = _norm_newlines(caller_text)
    industrial_text = industrial.text if industrial.applied else caller_text
    text = strip_ansi(_norm_newlines(industrial_text))
    text = unwrap_hermes_tool(text)
    text = preprocess_test_runner(text, query or "")
    if not text.strip():
        return CompressResult(
            compressed_text="", policy_name="noop", mode=requested_mode
        )
    ambiguity_fail_open = bool(ambiguity_fail_open)

    # Destructive preprocess (JSON crush / later log collapse) only when
    # the query names something specific. Generic/empty queries must not
    # delete array tails or fingerprint-collapse logs before scoring.
    if query_has_distinctive_selectors(query or ""):
        if route_content_type(_norm_newlines(text).split("\n")) != "json":
            text = _preprocess_json(text, query or "")
        text = preprocess_csv(text, query or "")
        text = preprocess_filler_comments(text, query or "")

    # Strategy ladder:
    #   'clear'     — drops tool-result payloads before scoring (regex-only,
    #                 zero LLM cost). Cheapest.
    #   'extract'   — default. Query-aware block scoring.
    #   'summarise' — calls a configurable OpenAI-compatible local LLM to
    #                 produce a faithful summary. Falls back to 'extract'
    #                 on any LLM failure (model down, timeout, empty response).
    #                 Highest quality, highest cost.
    strategy_norm = (strategy or "extract").strip().lower()
    if strategy_norm not in {"clear", "extract", "summarise"}:
        raise ValueError(f"unknown strategy {strategy!r} (expected clear/extract/summarise)")

    if strategy_norm == "clear":
        text = _clear_tool_payloads(text)
    elif strategy_norm == "summarise":
        summary = _summarise_with_llm(
            text,
            query or "",
            endpoint=summary_endpoint,
            model_candidates=summary_models,
            timeout=summary_timeout,
            allow_remote=summary_allow_remote,
        )
        summary_valid = False
        summary_recall = 0.0
        if summary is not None:
            summary_valid, summary_recall = _summary_preserves_required_facts(
                text, summary, query or ""
            )
        if (
            summary is not None
            and summary_valid
            and estimate_tokens(summary) < estimate_tokens(text)
        ):
            # LLM summary is shorter than the original — use it.
            # Store the original in CCR for reversibility.
            ccr_info = None
            ccr_marker = ""
            if ccr:
                try:
                    ccr_info = _ccr_store(caller_text, ccr_dir)
                except OSError:
                    ccr_info = None
                if ccr_info is not None:
                    ccr_marker = f"\n[CC-Retrieve: {ccr_info['hash']}]"
            result_text = summary + ccr_marker
            original_tokens = estimate_tokens(original_text)
            kept_tokens = estimate_tokens(result_text)
            keep_ratio = kept_tokens / max(1, original_tokens)
            summary_verifier = {
                "query_fact_recall": round(summary_recall, 3),
                "structured_ids_grounded": True,
                "risk": "medium",
                "score": round(summary_recall, 3),
            }
            summary_risk = "medium"
            return CompressResult(
                compressed_text=result_text,
                original_tokens=original_tokens,
                kept_tokens=kept_tokens,
                tokens_saved_pct=(1.0 - keep_ratio) * 100.0,
                policy_name="summarise-llm",
                mode=requested_mode,
                keep_ratio=keep_ratio,
                tokens_saved=original_tokens - kept_tokens,
                kept_line_ratio=1.0,
                cache_prefix_applied=False,
                compression_risk=(
                    summary_risk if summary_risk in {"medium", "high"} else "medium"
                ),
                confidence=round(
                    min(summary_recall, float(summary_verifier.get("score", 0.0))),
                    3,
                ),
                ccr=ccr_info,
                content_type="text",
                fail_open=False,
                frozen_blocks=0,
                reasons=["llm summary", "query facts verified"],
                verifier=summary_verifier,
                receipt={
                    "schema_version": "1",
                    "engine": "tameru",
                    "policy": "summarise-llm",
                    "query_hash": hashlib.sha256(
                        (query or "").encode("utf-8")
                    ).hexdigest()[:12],
                    "savings_pct": round((1.0 - keep_ratio) * 100.0, 2),
                    "risk": summary_risk,
                    "industrial": industrial.to_dict(),
                    **_receipt_hashes(
                        caller_text,
                        result_text,
                        industrial,
                        mode=requested_mode,
                        strategy=strategy,
                        budget_ratio=budget_ratio,
                        citations=citations,
                    ),
                },
            )
        # LLM failed or returned a longer result — fall through to extract.
        strategy_norm = "extract"

    lines, content_type = preprocess(text, query or "")
    if industrial.applied:
        content_type = industrial.profile.format
    if not query_has_distinctive_selectors(query or ""):
        # Do not fingerprint-collapse logs or crush JSON on a generic query.
        lines = text.split("\n")
        content_type = route_content_type(lines)
    blocks = segment_blocks(lines)
    if len(blocks) > industrial.limits.max_blocks:
        tokens = estimate_tokens(caller_text)
        reason = (
            f"block limit exceeded: {len(blocks)} > "
            f"{industrial.limits.max_blocks}"
        )
        receipt = {
            "schema_version": "1",
            "engine": "tameru",
            "policy": "local-fail-open",
            "query_hash": hashlib.sha256(
                (query or "").encode("utf-8")
            ).hexdigest()[:12],
            "savings_pct": 0.0,
            "risk": "high",
            "industrial": industrial.to_dict(),
        }
        receipt.update(
            _receipt_hashes(
                caller_text,
                caller_text,
                industrial,
                mode=requested_mode,
                strategy=strategy,
                budget_ratio=budget_ratio,
                citations=citations,
            )
        )
        return CompressResult(
            compressed_text=caller_text,
            original_tokens=tokens,
            kept_tokens=tokens,
            tokens_saved_pct=0.0,
            policy_name="local-fail-open",
            mode=requested_mode,
            keep_ratio=1.0,
            tokens_saved=0,
            kept_line_ratio=1.0,
            compression_risk="high",
            confidence=0.0,
            content_type=industrial.profile.format,
            fail_open=True,
            reasons=[reason],
            receipt=receipt,
        )
    scored = score_blocks(blocks, query or "")

    # v0.10.0 (G2, KVzip sink semantics): pinned blocks are exempt from
    # dropping in every selector path. Applied by boosting score and marking;
    # the floor/important/needle paths all respect score, and trust-risk is
    # bypassed for pins (a caller-pinned line outranks injection heuristics).
    if pin_patterns:
        import re as _re
        _pin_res = [_re.compile(pt) for pt in pin_patterns]
        for b in scored:
            if any(rx.search(b["text"]) for rx in _pin_res):
                b["score"] = max(b["score"], 999.0)
                b["pinned"] = True

    # Freeze-on-first-sight: if a decision_cache is provided and the block
    # fingerprints match prior turns, replay the stored keep/drop decisions
    # byte-identically. This keeps the provider prompt cache warm across a
    # multi-turn session (the prefix doesn't churn when new turns arrive).
    if decision_cache is not None and not isinstance(decision_cache, dict):
        decision_cache = None
    if decision_cache is not None:
        scored = _apply_freeze(decision_cache, scored, text, query or "")
    _reorder = bool(reorder_best)
    if mode_norm == "fixed":
        ratio = 0.35 if budget_ratio is None else float(budget_ratio)
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"budget_ratio must be in (0, 1], got {ratio!r}")
        kept, fail_open, risk = select_fixed(scored, ratio)
    else:
        kept, fail_open, risk = select_adaptive(
            scored,
            needle_only=query_has_distinctive_selectors(query or ""),
            query=query or "",
            semantic_tier=resolve_tier(semantic_tier) if semantic_tier is not None else None,
        )
        ratio = budget_ratio if budget_ratio is not None else 0.0
    if not fail_open:
        if decision_cache is not None:
            kept = _enforce_frozen_decisions(scored, kept)
        # Temporal supersession: a later kept block that explicitly marks an
        # earlier kept block stale (now/obsolete/override/newer date) prunes
        # it after cache replay, so a frozen keep cannot revive stale data.
        kept = apply_supersession(scored, kept)
    if ambiguity_fail_open:
        fail_open = True
        risk = "high"

    q = query or ""
    if not q.strip() or (
        not query_has_distinctive_selectors(q) and not _topic_terms(q)
    ):
        fail_open = True
        risk = "high"
    else:
        latin_terms = [
            t
            for t in distinctive_query_terms(q)
            if not t.startswith("script:")
        ]
        if latin_terms:
            blob = original_text.casefold()
            if not any(t in blob for t in latin_terms):
                fail_open = True
                risk = "high"
    if not query_has_distinctive_selectors(q) and _looks_like_csv(original_text):
        fail_open = True
        risk = "high"

    collapsed = "\n".join(lines)
    if fail_open:
        annotated_ids = {
            b["id"] for b in scored if b.get("trust_risk") and not b.get("pinned")
        }
        if annotated_ids:
            # Restore ordinary data when uncertain, but never undo a block
            # annotation. Explicitly pinned blocks remain eligible.
            kept = {b["id"] for b in scored if b["id"] not in annotated_ids}
            compressed = _render(
                lines, scored, kept, citations=citations, reorder_best=_reorder
            )
            fail_open = False
            risk = "high"
        # Do not undo a useful structured collapse (repeated INFO lines).
        # Baseline RAG wins on JP/KO heartbeats by keeping the error and
        # dropping repeats; restoring original bytes here threw that away.
        elif (
            not ambiguity_fail_open
            and collapsed != original_text
            and len(collapsed) < 0.7 * max(1, len(original_text))
        ):
            fail_open = False
            compressed = collapsed
            kept = {b["id"] for b in scored}
        else:
            compressed = original_text
            kept = {b["id"] for b in scored}
    else:
        compressed = _render(lines, scored, kept, citations=citations, reorder_best=_reorder)

    recall = _entity_recall(text, compressed, query or "")
    if recall < 0.66 and not fail_open:
        # One expansion pass: add any dropped block that still holds a query key.
        keys = _extract_entities(query or "") + _extract_terms(query or "")
        for b in scored:
            if b["id"] in kept or b.get("trust_risk"):
                continue
            if any(_soft_has(b["text"], k) for k in keys):
                kept.add(b["id"])
        if decision_cache is not None:
            kept = _enforce_frozen_decisions(scored, kept)
        compressed = _render(lines, scored, kept, citations=citations, reorder_best=_reorder)
        recall = _entity_recall(text, compressed, query or "")
        trust_risks = {
            b["id"]
            for b in scored
            if b.get("trust_risk") and not b.get("pinned")
        }
        if trust_risks:
            # Filter the CURRENT selection down by trust risks — do not
            # rebuild it as "everything except risks". Rebuilding discards
            # the selector's careful small keep-set and re-inflates the
            # payload to near-original size (v0.5.19 production QA finding).
            kept = {bid for bid in kept if bid not in trust_risks}
            if decision_cache is not None:
                kept = _enforce_frozen_decisions(scored, kept)
            compressed = _render(lines, scored, kept, citations=citations, reorder_best=_reorder)
            risk = "high"
        elif recall < 0.5:
            compressed = original_text
            fail_open = True
            risk = "high"
            kept = {b["id"] for b in scored}

    if fail_open:
        risk = "high"
        compressed = caller_text

    original_tokens = estimate_tokens(original_text)
    kept_tokens = estimate_tokens(compressed)
    trust_filtered = any(
        b.get("trust_risk") and not b.get("pinned") and b["id"] not in kept
        for b in scored
    )

    # Cost gate: if the compressed output is LARGER than the original
    # (net-negative savings), fail open. This happens when citations and
    # the CCR marker add more tokens than the dropped blocks saved.
    #
    # Provider cache pricing makes this more nuanced: Anthropic charges
    # 1.25× for cache writes and 0.1× for cache reads. If the context is
    # already cached (repeated session), the original is cheap to re-send
    # and the compression overhead may not be worth it. We model this
    # with a simple heuristic: if the savings are < 10%, the overhead of
    # citations + CCR marker likely exceeds the cache-read savings.
    if not fail_open and kept_tokens > original_tokens and not trust_filtered:
        compressed = original_text
        kept = {b["id"] for b in scored}
        fail_open = True
        risk = "high"
        kept_tokens = estimate_tokens(compressed)
    elif not fail_open and original_tokens > 200:
        # For substantial contexts, check if the savings justify the
        # compression overhead. If we're saving less than 10%, the
        # citation/CCR overhead likely exceeds the benefit, especially
        # when the provider cache makes the original cheap to re-send.
        savings_ratio = 1.0 - (kept_tokens / max(1, original_tokens))
        if savings_ratio < 0.10 and not trust_filtered:
            compressed = original_text
            kept = {b["id"] for b in scored}
            fail_open = True
            risk = "high"
            kept_tokens = estimate_tokens(compressed)

    if fail_open and compressed != caller_text:
        compressed = caller_text
        kept_tokens = estimate_tokens(compressed)

    freeze_cache_saturated = False
    if decision_cache is not None and not fail_open:
        freeze_cache_saturated = _record_freeze_decisions(decision_cache, scored, kept)

    keep_ratio = kept_tokens / max(1, original_tokens)
    line_keep = 0
    if lines:
        kept_line_idx = set()
        for b in scored:
            if b["id"] in kept:
                for i in range(b["start"], b["end"] + 1):
                    kept_line_idx.add(i)
        line_keep = len(kept_line_idx) / max(1, len(lines))

    reasons = sorted({scored[i]["reason"] for i in kept if i < len(scored)})
    if industrial.applied:
        reasons.append(f"industrial adapter: {industrial.profile.format}")
    if freeze_cache_saturated:
        reasons.append("freeze cache capacity reached")
    result_text = compressed
    cache_applied = False
    if cache_prefix and not fail_open:
        result_text = cache_wrap(compressed, query or "")
        cache_applied = True

    ccr_info = None
    pending_ccr_hash = None
    candidate_text = result_text
    if ccr and not fail_open and compressed != original_text:
        pending_ccr_hash = hashlib.sha256(caller_text.encode("utf-8")).hexdigest()[:24]
        if "[CC-Retrieve:" not in candidate_text:
            candidate_text = (
                candidate_text.rstrip()
                + f"\n[CC-Retrieve: {pending_ccr_hash}]\n"
            )

    candidate_tokens = estimate_tokens(candidate_text)
    if not fail_open and candidate_tokens > original_tokens and not trust_filtered:
        result_text = caller_text
        compressed = caller_text
        kept = {b["id"] for b in scored}
        fail_open = True
        risk = "high"
        cache_applied = False
        line_keep = 1.0
    else:
        result_text = candidate_text
        if pending_ccr_hash is not None:
            try:
                # Recovery means the caller's exact bytes, not a preprocessed view.
                ccr_info = _ccr_store(caller_text, ccr_dir)
            except OSError:
                ccr_info = None
                result_text = result_text.replace(
                    f"\n[CC-Retrieve: {pending_ccr_hash}]\n", ""
                )

    kept_tokens = estimate_tokens(result_text)
    keep_ratio = kept_tokens / max(1, original_tokens)
    savings = round((1 - keep_ratio) * 100, 2)
    policy = f"local-{content_type}"
    if fail_open:
        policy = "local-fail-open"

    # Post-compression verifier: self-check entity/keyword/critical-line
    # recall. This is diagnostic — it does NOT modify the compressed output.
    # Callers can inspect the verifier to decide whether to re-compress with
    # a different strategy or expand dropped blocks.
    verifier = None
    if not fail_open and compressed and original_tokens > 50:
        verifier = verify_compression(text, compressed, query or "")
        risk_order = {"low": 0, "medium": 1, "high": 2}
        verifier_risk = str(verifier.get("risk", "high"))
        if risk_order.get(verifier_risk, 2) > risk_order.get(risk, 2):
            risk = verifier_risk
        recall = min(recall, float(verifier.get("score", 0.0)))
    __cr = CompressResult(
        compressed_text=result_text,
        original_tokens=original_tokens,
        kept_tokens=kept_tokens,
        tokens_saved_pct=savings,
        policy_name=policy,
        mode=requested_mode,
        keep_ratio=keep_ratio,
        tokens_saved=original_tokens - kept_tokens,
        kept_line_ratio=round(line_keep, 4),
        cache_prefix_applied=cache_applied,
        compression_risk=risk,
        confidence=round(recall, 3),
        ccr=ccr_info,
        content_type=content_type,
        fail_open=fail_open,
        frozen_blocks=sum(1 for b in scored if b.get("frozen")),
        reasons=reasons,
        verifier=verifier,
    )

    # v0.10.0 (G5 memorix receipt): machine-readable provenance.
    query_hash = hashlib.sha256((query or "").encode("utf-8")).hexdigest()[:12]
    receipt = {
        "schema_version": "1",
        "engine": "tameru",
        "policy": policy,
        "kept_ids": _bounded_id_manifest(
            (i for i in kept if i < len(scored)),
            industrial.limits.max_receipt_ids,
        ),
        "dropped_ids": _bounded_id_manifest(
            (b["id"] for b in scored if b["id"] not in kept),
            industrial.limits.max_receipt_ids,
        ),
        "query_hash": query_hash,
        "savings_pct": savings,
        "risk": risk,
        "verifier": verifier,
        "industrial": industrial.to_dict(),
    }
    receipt.update(
        _receipt_hashes(
            caller_text,
            result_text,
            industrial,
            mode=requested_mode,
            strategy=strategy,
            budget_ratio=budget_ratio,
            citations=citations,
        )
    )
    __cr.receipt = receipt

    # v0.10.0 (G1 memorix checkpoints): append-only audit log, opt-in.
    if log_dir:
        try:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                log_path.chmod(0o700)
            except OSError:
                pass
            top_dropped = sorted(
                (
                    {"id": b["id"], "score": round(float(b.get("score", 0.0)), 2)}
                    for b in scored
                    if b["id"] not in kept
                ),
                key=lambda d: -d["score"],
            )[:5]
            entry = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "query_hash": query_hash,
                "policy": policy,
                "kept": len(kept),
                "total": len(scored),
                "savings_pct": savings,
                "risk": risk,
                "fail_open": fail_open,
                "top_dropped": top_dropped,
                "industrial_format": industrial.profile.format,
                "industrial_applied": industrial.applied,
                "direction": industrial.profile.direction,
                "scripts": industrial.profile.scripts,
                "input_chars": industrial.profile.characters,
                "input_lines": industrial.profile.lines,
                "profile_truncated": industrial.profile.profile_truncated,
            }
            log_file = log_path / "compactions.jsonl"
            fd = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            if hasattr(os, "fchmod"):
                os.fchmod(fd, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass  # logging must never break compression

    return __cr


def main(argv: Optional[Iterable[str]] = None) -> int:
    import argparse
    limit_defaults = IndustrialLimits()
    p = argparse.ArgumentParser(description="Local query-aware extractive compressor")
    p.add_argument("context_file", help="Path to context text (or - for stdin)")
    p.add_argument("query")
    p.add_argument(
        "--mode",
        default="adaptive",
        choices=["adaptive", "fixed", "compiler", "precision"],
    )
    p.add_argument("--budget-ratio", type=float, default=None)
    p.add_argument("--max-input-chars", type=int, default=limit_defaults.max_input_chars)
    p.add_argument("--max-lines", type=int, default=limit_defaults.max_lines)
    p.add_argument("--max-records", type=int, default=limit_defaults.max_records)
    p.add_argument("--max-record-chars", type=int, default=limit_defaults.max_record_chars)
    p.add_argument("--max-fields", type=int, default=limit_defaults.max_fields)
    p.add_argument("--max-profile-chars", type=int, default=limit_defaults.max_profile_chars)
    p.add_argument("--max-bidi-controls", type=int, default=limit_defaults.max_bidi_controls)
    p.add_argument("--max-bidi-overrides", type=int, default=limit_defaults.max_bidi_overrides)
    p.add_argument("--max-query-chars", type=int, default=limit_defaults.max_query_chars)
    p.add_argument("--max-blocks", type=int, default=limit_defaults.max_blocks)
    p.add_argument("--max-receipt-ids", type=int, default=limit_defaults.max_receipt_ids)
    p.add_argument("--ccr", action="store_true", default=None,
                   help="Enable CCR reversible store (on by default)")
    p.add_argument("--no-ccr", dest="ccr", action="store_false",
                   help="Disable CCR (reversibility off)")
    p.add_argument("--cache-prefix", action="store_true")
    p.add_argument("--citations", action="store_true", default=None,
                   help="Emit ARC-style citations for dropped blocks (on by default)")
    p.add_argument("--no-citations", dest="citations", action="store_false",
                   help="Disable citations")
    p.add_argument("--decision-cache", default=None,
                   help="JSON file path for freeze-on-first-sight decision cache")
    p.add_argument("--strategy", default="extract", choices=["clear", "extract", "summarise"],
                   help="Compression strategy: clear (drop tool payloads), extract (default, query-aware), summarise (falls back to extract)")
    p.add_argument("--stats", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)
    if args.context_file == "-":
        import sys as _sys
        ctx = _sys.stdin.read()
    else:
        ctx = Path(args.context_file).read_text(encoding="utf-8")
    # Resolve ccr/citations defaults (None = use library default True)
    ccr_val = True if args.ccr is None else args.ccr
    citations_val = True if args.citations is None else args.citations
    limits = IndustrialLimits(
        max_input_chars=args.max_input_chars,
        max_lines=args.max_lines,
        max_records=args.max_records,
        max_record_chars=args.max_record_chars,
        max_fields=args.max_fields,
        max_profile_chars=args.max_profile_chars,
        max_bidi_controls=args.max_bidi_controls,
        max_bidi_overrides=args.max_bidi_overrides,
        max_query_chars=args.max_query_chars,
        max_blocks=args.max_blocks,
        max_receipt_ids=args.max_receipt_ids,
    )
    # Load decision cache if provided
    decision_cache = None
    if args.decision_cache:
        dc_path = Path(args.decision_cache)
        if dc_path.is_file():
            try:
                decision_cache = json.loads(dc_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, RecursionError):
                decision_cache = {}
            if not isinstance(decision_cache, dict):
                decision_cache = {}
        else:
            decision_cache = {}
    out = compress_context(
        ctx,
        args.query,
        mode=args.mode,
        budget_ratio=args.budget_ratio,
        ccr=ccr_val,
        cache_prefix=args.cache_prefix,
        citations=citations_val,
        decision_cache=decision_cache,
        strategy=args.strategy,
        limits=limits,
    )
    if args.stats:
        stats = out.to_dict()
        stats.pop("compressed_text", None)
        print(json.dumps(stats, indent=2))
        print("---")
    print(out.compressed_text)
    # Persist decision cache if provided
    if args.decision_cache and decision_cache is not None:
        dc_path = Path(args.decision_cache)
        _write_private_atomic(
            dc_path,
            json.dumps(decision_cache, ensure_ascii=False),
        )
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
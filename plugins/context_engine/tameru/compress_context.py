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
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .agent_formats import (
    GUTTER_RE as _AGENT_GUTTER_RE,
    apply_agent_adapters,
    detect_test_runner,
    is_numbered_code,
    preprocess_test_output,
    rejects_table,
)
from .contract_gates import (
    GENERIC_WORDS,
    distinctive_query_terms,
    query_has_distinctive_selectors,
)
from .folding import fold_lossless, json_brief, mask_template
from .format_adapters import FILLER_WORDS
from .industrial import IndustrialLimits, IndustrialResult, industrial_preprocess
from .supersession import apply_supersession
from .unicode_profile import (
    script_of,
    search_units,
    token_units,
    unicode_safety_counts,
)

try:  # optional semantic tier — never required
    from .semantic import resolve_tier
except ImportError:  # direct-script use without package context
    def resolve_tier(tier):
        return tier

# Public re-export (``from tameru.compress_context import fold_lossless``); naming it here
# keeps the import from looking unused.
_REEXPORTED = (fold_lossless,)

try:
    DEFAULT_CCR_DIR = Path.home() / ".tameru" / "cache" / "context-compress-ccr"
except Exception:
    import tempfile
    DEFAULT_CCR_DIR = Path(tempfile.gettempdir()) / ".tameru" / "cache" / "context-compress-ccr"
# Engine release this module ships in; the release bump keeps it in step with
# pyproject.toml and src/tameru/__init__.py.
ENGINE_VERSION = "1.4.1"
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

# Degraded scoring view (opt-in, JEV-inspired): on input size-limit breaches,
# score a bounded head+tail view per block while emitted output stays
# byte-exact. The oversize factor bounds total work when limits are bypassed.
_DEGRADED_VIEW_HEAD = 400
_DEGRADED_VIEW_TAIL = 200
_DEGRADED_OVERSIZE_FACTOR = 4

# Secrets screen (dsh-jev-pre-compaction inspired): never persist probable
# credentials to the CCR store. Conservative patterns only — private key
# blocks, well-known token prefixes, JWTs, and long quoted assignments.
_SECRETS_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----"
    r"|\bAKIA[0-9A-Z]{16}\b"
    r"|\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|\bxox[baprs]-[A-Za-z0-9-]{10,}"
    r"|\bsk-(?:ant-)?[A-Za-z0-9_-]{20,}"
    r"|\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"
    r"|(?i:\b(?:password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)\b"
    r"\s*[:=]\s*[\"'][^\"'\s]{16,}[\"'])"
    r"|\bAIza[A-Za-z0-9_-]{35}(?![A-Za-z0-9_-])"
    r"|\b(?:sk|rk)_live_[A-Za-z0-9]{16,}"
    r"|(?i:\b(?:authorization\s*:\s*)?bearer\s+[A-Za-z0-9._~+/=-]{20,})"
    r"|(?i:\b(?:[A-Za-z0-9_]+_)?(?:password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)\b"
    r"\s*[:=]\s*"
    # 1.4.1 red-team: an unquoted value that is code (a call, an index, a
    # ``${...}`` / ``<...>`` placeholder, an attribute chain, a snake_case
    # variable such as ``api_key=fallback_api_key,``) names no credential.
    r"(?![^\s\"']*[(\[{<$])(?!(?:[A-Za-z_]\w*\.)+[A-Za-z_]\w*(?![^\s,;)\]}]))"
    r"(?![a-z_](?=[a-z0-9_]*_)[a-z0-9_]*(?![^\s,;)\]}]))"
    r"[^\s\"']{16,})"
    # 1.4.1 fix-all: the code exclusions above also let through unquoted
    # snake_case passphrases and dotted passwords. A whole-line ``.env`` /
    # shell assignment (``DB_PASSWORD=correct_horse_battery_staple``, no
    # spaces around ``=``) or a YAML ``key: value`` line still names one: the
    # value runs to the end of the line and does not end in ``,`` ``;`` ``)``
    # (a keyword argument on its own line). YAML keeps the attribute-chain
    # exclusion (``api_key: pydantic.types.SecretStr`` is an annotation).
    r"|(?im:^[ \t]*(?:export[ \t]+)?(?:[A-Za-z0-9_]+_)?"
    r"(?:password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)"
    r"=(?![^\s\"']*[(\[{<$])[^\s\"']{15,}[^\s\"',;)][ \t]*$)"
    r"|(?im:^[ \t]*(?:[A-Za-z0-9_]+_)?"
    r"(?:password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)"
    r":[ \t]+(?![^\s\"']*[(\[{<$])(?!(?:[A-Za-z_]\w*\.)+[A-Za-z_]\w*[ \t]*$)"
    r"[^\s\"']{15,}[^\s\"',;)][ \t]*$)"
)

# Recursion guard markers: output Tameru itself produced.
_RECURSION_MARKERS = ("<compressed_context", "[CC-Retrieve:")


def _contains_secret(text: str) -> bool:
    return bool(text) and _SECRETS_RE.search(text) is not None


def contains_secret(text: str) -> bool:
    """Public alias of :func:`_contains_secret` (v1.4.0).

    True when ``text`` carries probable credential material (private-key
    headers, well-known token prefixes, JWTs, long quoted assignments).
    Callers that persist or forward text — the CCR store, a host plugin's
    recovery store — use it to refuse to keep secrets. Conservative and
    deterministic; never raises on non-string input.
    """
    return _contains_secret(text if isinstance(text, str) else str(text or ""))

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
# v1.4.0 (E2): segments are bounded and a match may only start at the start of
# a run, so a 20k-char run without separators is linear instead of quadratic.
# Segments over 255 chars are not paths (hash blobs); normal inputs match as
# before.
_PATH_RE = re.compile(r"(?<![\w.-])(?:[\w.-]{1,255}/){2,}[\w.-]{1,255}")
_DOTTED_RE = re.compile(r"\b[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+\b")
_QUOTED_RE = re.compile(r"[\"']([^\"']{2,64})[\"']")
_CAPS_RE = re.compile(r"\b[A-Z]{2,24}\b")
_URI_RE = re.compile(r"\b[a-z][a-z0-9+.-]{0,31}://\S+")
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


# v1.4.0 (E2): deleting the accounting punctuation and testing isascii() is the
# same predicate as the old per-character generator, but runs at C speed — the
# generator cost ~0.6 s per call on a 650 KB output that contains one "[…]".
_FAST_ACCOUNTING_DELETE = {ord(ch): None for ch in _FAST_ACCOUNTING_PUNCT}


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    if text.isascii() or text.translate(_FAST_ACCOUNTING_DELETE).isascii():
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
        r"^\[(user|assistant|system|tool|developer)\]", t, re.IGNORECASE
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
    if line.lstrip().startswith(("[\u00d7", "[last] ")):
        return None
    stripped = _PROGRESS_RE.match(line.strip())
    if stripped:
        return None
    # v1.4.0 (E10): volatile fields (ids, ips, timestamps, durations, quoted
    # strings, paths) are masked before the digits, by the same function the
    # error fingerprint uses.
    s = mask_template(line[:_RX_LINE_CAP].lower())
    return s[:240] or None


def _template_long_enough(line: str) -> bool:
    """A line is worth collapsing only if its digit-masked form is >12 chars.

    Measured on the digit mask, not on the E10 template (``<ts> info ok`` is
    short only because its timestamp became one token).
    """
    return len(_DIGITS_RE.sub("#", " ".join(line[:_RX_LINE_CAP].lower().split()))) > 12


def _is_progress_bar(line: str) -> bool:
    t = line.strip()
    if not t:
        return False
    if re.search(r"(\[[=#>\-. ]*\]\s*)?\d{1,3}\s*(%|/\s*\d+)", t):
        return True
    return bool(re.match(r"^Compiling [\w-]+ ", t))


# ---------------------------------------------------------------------------
# v1.4.0 shared line helpers (E1/E3): leading timestamps and error lines.
# ---------------------------------------------------------------------------

# Regex scans of a single line look at its first 2000 chars only: a 20k-char
# run (base64 blob, minified bundle) must not make backtracking regexes
# quadratic. The full line is still kept/emitted by callers.
_RX_LINE_CAP = 2000

# A leading log timestamp is a prefix, not a fact: ISO / RFC3339, syslog
# ("Mar 14 12:00:03") and bare "HH:MM:SS[.mmm]", optionally bracketed. It
# must not by itself make a line "critical" (E1) or a "critical line" for the
# verifier (E3).
_LEAD_TS_RE = re.compile(
    r"^\s*[\[(]?(?:"
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?"
    r"|[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}"
    r"|\d{2}:\d{2}:\d{2}(?:[.,]\d+)?"
    r")[\])]?[ \t]*"
)


# Critical-line pre-computation (Baseline RAG-style): lines that carry
# operational facts — errors, config assignments, key-value pairs, dates,
# versions. Blocks holding them get a floor boost in score_blocks.
#
# The date alternative does not match a date that opens the line (optionally
# after up to two blanks and a "[" or "("): a line-leading timestamp is a log
# prefix, not a fact. _is_critical_line() widens that to every leading
# timestamp form and chat role labels.
_CRIT_LINE_RE = re.compile(
    r"\b(?:error|exception|traceback|fatal|critical)\b"   # error keywords
    r"|\bconfig(?:uration)?\s*[=:]"                       # config assignments
    r"|\b(?:port|host|user|password|url|token|key)\s*[=:]"  # key-value pairs
    r"|(?<!^)(?<!^[ \t])(?<!^[ \t][ \t])(?<!^[\[(])(?<!^[ \t][\[(])"
    r"\b20[2-3]\d-\d{2}-\d{2}\b"                          # dates (not line-leading)
    r"|\b\d+\.\d+\.\d+(?:\.\d+)?\b",                      # version numbers
    re.IGNORECASE,
)


# A leading chat role label ("User: ...", "Tool: ...") is a speaker tag, not a
# `user:` key-value fact. The old buggy call (pos=2, case-sensitive) never
# matched it either; IGNORECASE would, and boost every user turn.
_LEAD_ROLE_RE = re.compile(r"[ \t]*(?:User|Assistant|System|Tool|Developer)[ \t]*:[ \t]*")


def _lead_ts_end(line: str) -> int:
    """Index just past a leading timestamp prefix (0 when there is none)."""
    m = _LEAD_TS_RE.match(line)
    return m.end() if m else 0


def _crit_scan_start(line: str) -> int:
    """Where a critical-line scan starts: after a leading timestamp and/or
    chat role label, which are line prefixes rather than facts."""
    pos = _lead_ts_end(line)
    m = _LEAD_ROLE_RE.match(line, pos)
    return m.end() if m else pos


def _is_critical_line(line: str) -> bool:
    """Does ``line`` carry an operational fact (error keyword, config or
    key-value assignment, date, version)? A bare leading timestamp or chat
    role label does not count (E1)."""
    return _CRIT_LINE_RE.search(line, _crit_scan_start(line)) is not None


# Level-tagged error lines. Case-sensitive on purpose: "error" in prose is not
# a log level, ERROR is.
_ERR_LEVEL_RE = re.compile(r"\b(?:ERROR|FATAL|CRITICAL|PANIC|FAIL(?:ED|URE)?)\b")
_ERR_EXC_RE = re.compile(r"^\s*(?:\w+\.)*\w*(?:Error|Exception):")
_ERR_PANIC_RE = re.compile(r"\bpanic:")
_ERR_TRACEBACK = "Traceback (most recent call last)"
# "0 errors", "errors: 0", "failed=0" report success; they are not errors. Only
# a literal zero count negates: "no error body" / "no error handler" are the
# TEXT of a real ERROR line, and a decimal ("ERROR: 0.5 exceeds ...") is not a
# count.
_ERR_NEGATED_RE = re.compile(
    r"\b0\s+(?:errors?|failures?|failed|fatals?)\b"
    r"|\b(?:errors?|failures?|failed|fatals?|critical)\s*[=:]\s*0(?!\w|\.\d)",
    re.IGNORECASE,
)
_COUNT_PREFIX_RE = re.compile(r"^\s*\[(?:\u00d7\d+|last)\]\s*")
_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
_HEXRUN_RE = re.compile(r"[0-9a-f]{6,}")
_DIGITS_RE = re.compile(r"\d+")
_FINGERPRINT_CAP = 160


def _without_zero_counts(s: str) -> str:
    """``s`` with its literal-zero count phrases ("0 errors", "errors: 0",
    "failed=0") removed. A line is negated only when nothing else in it signals
    a failure: Maven's ``[ERROR] Tests run: 5, Failures: 2, Errors: 0`` reports
    a failure *and* a zero count, and must stay an error line."""
    return _ERR_NEGATED_RE.sub(" ", s)


def _is_level_error_line(line: str) -> bool:
    """A level-tagged ERROR/FATAL/CRITICAL/PANIC/FAIL* log line."""
    return _ERR_LEVEL_RE.search(_without_zero_counts(line[:_RX_LINE_CAP])) is not None


def _is_error_line(line: str) -> bool:
    """Level-tagged error, ``XxxError:``/``Exception:`` head, traceback head or
    Go panic — the lines :func:`error_fingerprints` keys on."""
    s = _without_zero_counts(line[:_RX_LINE_CAP])
    # The exception head is anchored at the line start: look past a "[×N] "
    # collapse prefix so the same failure keeps its key after a fold.
    head = s
    prefix = _COUNT_PREFIX_RE.match(s)
    if prefix:
        head = s[prefix.end():]
    return bool(
        _ERR_LEVEL_RE.search(s)
        or _ERR_EXC_RE.match(head)
        or _ERR_TRACEBACK in s
        or _ERR_PANIC_RE.search(s)
    )


def _error_fp(line: str) -> str:
    """Stable fingerprint of an error line: volatile fields are masked.

    A leading ``[×N] `` collapse marker is stripped, the line lowercased,
    UUIDs -> ``<uuid>``, hex runs of 6+ -> ``<hex>`` (before digits, so ids
    that differ only in digit placement still collide), digits -> ``#``,
    whitespace collapsed, capped at 160 chars.
    """
    s = _COUNT_PREFIX_RE.sub("", line[:_RX_LINE_CAP], count=1).lower()
    return mask_template(s)[:_FINGERPRINT_CAP]


def error_fingerprints(text: str) -> dict[str, str]:
    """Map error fingerprint -> first exemplar line, in original order (v1.4.0).

    An error line is a level-tagged ``ERROR``/``FATAL``/``CRITICAL``/``PANIC``/
    ``FAIL``/``FAILED``/``FAILURE`` line (uppercase tokens, case-sensitive), an
    ``XxxError:`` / ``Exception:`` head line, a ``Traceback (most recent call
    last)`` line or a Go ``panic:`` line. Lines that report zero errors
    ("0 errors", "errors: 0", "failed=0") are not errors.

    The fingerprint masks everything volatile (see :func:`_error_fp`), so 500
    repeats of one failure share a key and the first one is the exemplar.
    Deterministic and stable under compaction: for compressed text ``c`` that
    kept an exemplar of every failure, ``error_fingerprints(x).keys() <=
    error_fingerprints(c).keys()`` — a host can use that as a retention check.
    """
    found: dict[str, str] = {}
    if not text:
        return found
    for line in _norm_newlines(text).split("\n"):
        if not _is_error_line(line):
            continue
        fp = _error_fp(line)
        if fp and fp not in found:
            found[fp] = line
    return found


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
    # v1.4.0 (N6): out_src[k] is the index in `lines` that out[k] came from, so
    # restored error lines can go back to their original relative position.
    out_src: list[int] = []
    trace: list[str] = []
    trace_src: list[int] = []
    query_match_indices: set[int] = set()
    query_selectors = distinctive_query_terms(query or "")

    def flush_trace() -> None:
        nonlocal trace, trace_src
        if not trace:
            return
        if len(trace) <= 4:
            out.extend(trace)
            out_src.extend(trace_src)
        else:
            out.append(trace[0])
            out_src.append(trace_src[0])
            out.append(f"  ... {len(trace) - 2} frames omitted")
            out_src.append(trace_src[1])
            out.append(trace[-1])
            out_src.append(trace_src[-1])
        trace = []
        trace_src = []

    # First pass: drop progress bars but remember whether we saw any.
    kept_src = [i for i, ln in enumerate(lines) if not _is_progress_bar(ln)]
    kept_lines = [lines[i] for i in kept_src]

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
    last_seen: dict[str, str] = {}  # v1.4.0 (E10): last instance of each template

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
            trace_src.append(kept_src[i])
            i += 1
            continue
        flush_trace()
        if i in query_match_source_indices:
            # Volatile values can be the selector (for example "policy 17").
            # Keep the exact matching record outside numeric-template collapse.
            query_match_indices.add(len(out))
            out.append(line)
            out_src.append(kept_src[i])
            i += 1
            continue
        fp = _log_fingerprint(line)
        if fp and _template_long_enough(line):
            if fp in exemplar_idx:
                seen[fp] += 1
                last_seen[fp] = line
                i += 1
                continue
            exemplar_idx[fp] = len(out)
            seen[fp] = 1
        out.append(line)
        out_src.append(kept_src[i])
        i += 1
    flush_trace()

    # Attach counts: replace exemplar line with counted variant.
    # v1.4.0 (E10): a template seen >=3 times keeps its LAST instance verbatim
    # on the next line ("[last] ..."), so the end state of a repeating failure
    # survives next to the first one.
    counted_out: list[str] = []
    counted_src: list[int] = []
    for idx, ln in enumerate(out):
        if idx in query_match_indices:
            counted_out.append(ln)
            counted_src.append(out_src[idx])
            continue
        fp = _log_fingerprint(ln)
        cnt = seen.get(fp, 1) if fp else 1
        if fp and cnt > 1 and not ln.lstrip().startswith("[\u00d7"):
            counted_out.append(f"[\u00d7{cnt}] {ln}")
            counted_src.append(out_src[idx])
            last = last_seen.get(fp)
            if cnt >= 3 and last is not None and last != ln:
                counted_out.append(f"[last] {last}")
                counted_src.append(out_src[idx])
        else:
            counted_out.append(ln)
            counted_src.append(out_src[idx])
    result = counted_out
    out_src = counted_src

    # N6 error-signal invariant: original error lines that are missing from
    # the output must be restored. v1.4.0: a line counts as present when an
    # exemplar of its template is present (possibly prefixed "[×N] "), so the
    # 499 copies a "[×500] ..." exemplar already stands for are NOT re-added;
    # genuinely missing lines go back at their ORIGINAL relative position, not
    # at the top.
    def has_error_signal(text: str) -> bool:
        return bool(_ERROR_RE.search(text))

    orig_errors = [(i, ln) for i, ln in enumerate(lines) if has_error_signal(ln)]
    if orig_errors:
        present_lines = set(result)
        present_fps: set[str] = set()
        for ln in result:
            if has_error_signal(ln):
                fp = _log_fingerprint(_COUNT_PREFIX_RE.sub("", ln, count=1))
                if fp:
                    present_fps.add(fp)
        missing: list[tuple[int, str]] = []
        for src_idx, ln in orig_errors:
            if ln in present_lines:
                continue
            fp = _log_fingerprint(ln)
            if fp and fp in present_fps:
                continue
            if fp:
                present_fps.add(fp)  # one restored exemplar stands for the template
            missing.append((src_idx, ln))
        if missing:
            merged: list[str] = []
            mi = 0
            for pos, ln in enumerate(result):
                # Restored lines go after every kept line that precedes them
                # in the original (ties: the kept line first), before the next.
                while mi < len(missing) and missing[mi][0] < out_src[pos]:
                    merged.append(missing[mi][1])
                    mi += 1
                merged.append(ln)
            merged.extend(ln for _, ln in missing[mi:])
            result = merged

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
                # lcc TRIM refusal: if the tail being cut carries a
                # qualifier/negation cue, truncation can invert the claim
                # ("granted to all staff" ← "…, except contractors").
                # A longer safe value beats a shorter misleading one.
                if _QUALIFIER_CUE_RE.search(v, 100):
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
    except (ValueError, RecursionError):  # JSONDecodeError + the 4300-digit int limit
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
    if similar < 4:
        return False
    # v1.4.0 (E4): code is not CSV. A numbered read, code lines, or an unstable
    # column count (a JS file whose first line happens to hold commas) is not a
    # table, however many commas it has.
    return not rejects_table(lines)


def _selector_patterns(terms: Iterable[str]) -> list[re.Pattern[str]]:
    return [
        re.compile(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])", re.IGNORECASE)
        for term in terms
    ]


def preprocess_csv(
    text: str,
    query: str,
    gap_marker: Callable[[int, int], str] | None = None,
) -> str:
    """Keep header + distinctive-matching rows. No first-N cliff.

    ``gap_marker(omitted_chars, total_chars)`` marks each run of dropped rows
    (v1.4.0); ``None`` keeps the output byte-identical to earlier releases."""
    if not query_has_distinctive_selectors(query) or not _looks_like_csv(text):
        return text
    terms = [t for t in distinctive_query_terms(query) if not t.startswith("script:")]
    if not terms:
        return text
    selectors = _selector_patterns(terms)
    lines = text.split("\n")
    kept = [lines[0]]
    flags = [False] * len(lines)
    flags[0] = True
    for idx, line in enumerate(lines[1:], 1):
        if any(selector.search(line) for selector in selectors):
            kept.append(line)
            flags[idx] = True
    if len(kept) == 1 or len(kept) == len(lines):
        return text
    if gap_marker is None:
        return "\n".join(kept)
    out: list[str] = []
    run: list[str] = []
    for line, keep in zip(lines, flags):
        if keep:
            if run:
                dropped = "\n".join(run)
                if dropped.strip():
                    out.append(str(gap_marker(len(dropped), len(text))))
                run = []
            out.append(line)
        else:
            run.append(line)
    if run:
        dropped = "\n".join(run)
        if dropped.strip():
            out.append(str(gap_marker(len(dropped), len(text))))
    return "\n".join(out)


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
_TEST_SUMMARY_RE = re.compile(r"Tests\s+\d+\s+passed", re.IGNORECASE)
_TEST_FILE_NOISE_RE = re.compile(
    r"ExperimentalWarning|Use `node --trace-warnings|^\s*[✓✔]\s"
)


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def preprocess_test_runner(
    text: str,
    query: str,
    content_hint: str | None = None,
    gap_marker: Callable[[int, int], str] | None = None,
) -> str:
    """Keep suite summary; drop per-file ticks and node warnings.

    Super does this on vitest dumps. Skip if the query names a test file.

    v1.4.0 (E5): pytest, jest, go test and cargo test output is reduced to its
    failure blocks and summary lines (``agent_formats.preprocess_test_output``);
    passing per-test lines become a count marker (or ``gap_marker``). Applied
    when the runner is recognised, or when ``content_hint == "test"``.
    """
    if content_hint in (None, "test"):
        reduced = preprocess_test_output(
            text, query, gap_marker=gap_marker, force=content_hint == "test"
        )
        if reduced is not None:
            return reduced
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


# A reduced test run at most this long is final (never re-scored).
_TEST_REDUCTION_FINAL_CHARS = 16000
_FILLER_COMMENT_RE = re.compile(r"^(--|#)\s*filler\b", re.IGNORECASE)
_TRUST_RISK_RE = re.compile(
    r"\b(?:untrusted|not\s+operational)\b"
    r"|\b(?:obsolete|unrelated)\b.{0,24}"
    r"\b(?:glossary|example|sample|entry|data)\b"
    r"|\b(?:ignore|disregard|override|forget)\b.{0,48}"
    r"\b(?:instructions?|prompt|system)\b",
    re.IGNORECASE,
)
_TRUST_QUERY_RE = re.compile(
    r"\b(?:injection|untrusted|malicious|prompt\s+attack|obsolete|unrelated)\b"
    r"|\bignore\s+(?:prior|previous)\b",
    re.IGNORECASE,
)
# v0.6.0: a query that says "trust nothing from untrusted sources",
# "exclude untrusted …" etc. is asking for the SAFE behaviour — it must
# NOT suppress the trust-risk flag. Only explicit include-phrasing
# ("include the untrusted block verbatim") counts as an override.
_TRUST_EXCLUDE_CUE_RE = re.compile(
    r"\b(?:trust\s+nothing|don't\s+trust|do\s+not\s+trust|exclude|filter\s+out|omit)\b"
    r".{0,40}\b(?:untrusted|unverified|malicious)\b"
    r"|\b(?:exclude|drop|skip|omit)\s+(?:any\s+)?(?:untrusted|unverified|malicious)\b",
    re.IGNORECASE,
)


def _normalise_instruction_text(text: str) -> str:
    """Normalise width and remove invisible formatting before classification."""
    text = str(text or "")
    if text.isascii():
        # Perf (v1.4.0): NFKC is the identity on ASCII and ASCII has no Cf
        # (format) characters, so the per-character scan below cannot change
        # anything — it cost ~0.4 s per 850 KB of ASCII log text.
        return text
    normalised = unicodedata.normalize("NFKC", text)
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
    return unwrap_tool_payload(text).inner


def _compact_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class ToolPayload:
    """A tool result split into its text payload and its envelope (v1.4.0).

    ``inner``   the payload text (the whole input when it was not wrapped)
    ``field``   ``"content"`` or ``"output"`` when unwrapped, else ``None``
    ``meta``    the OTHER top-level keys whose values are scalars
                (str/int/float/bool/None), in original key order — for Hermes
                tools that is ``exit_code``, ``error``, ``total_lines``, ...
    ``wrapped`` True when the input was a JSON object wrapper
    ``order``   every top-level key of the original object, in order (lets
                :func:`rewrap_tool_payload` put the payload back where it was);
                not part of equality or repr, so ``ToolPayload(inner, field,
                meta, wrapped)`` built by hand compares equal to an unwrapped one
    """

    inner: str
    field: str | None
    meta: dict
    wrapped: bool
    order: tuple[str, ...] = field(default=(), compare=False, repr=False)


def unwrap_tool_payload(text: str) -> ToolPayload:
    """Split a Hermes tool result into payload and envelope.

    Detection mirrors :func:`unwrap_hermes_tool`: a JSON object with a string
    ``content`` or ``output`` longer than 200 chars (``content`` first).
    Anything else comes back as ``ToolPayload(text, None, {}, False)``.
    """
    unwrapped = ToolPayload(inner=text, field=None, meta={}, wrapped=False)
    if not isinstance(text, str):
        return unwrapped
    stripped = text.strip()
    if not stripped.startswith("{"):
        return unwrapped
    try:
        parsed = json.loads(stripped)
    except (ValueError, RecursionError):  # JSONDecodeError + the 4300-digit int limit
        return unwrapped
    if not isinstance(parsed, dict):
        return unwrapped
    for key in ("content", "output"):
        val = parsed.get(key)
        if isinstance(val, str) and len(val) > 200:
            meta = {
                k: v
                for k, v in parsed.items()
                if k != key and (v is None or isinstance(v, (str, int, float, bool)))
            }
            return ToolPayload(
                inner=val, field=key, meta=meta, wrapped=True, order=tuple(parsed)
            )
    return unwrapped


def rewrap_tool_payload(payload: ToolPayload, new_inner: str) -> str:
    """Rebuild a tool-result JSON object around a replacement payload.

    ``new_inner`` goes back under ``payload.field`` and the scalar ``meta``
    keys keep their original order; non-scalar keys the unwrap dropped are not
    restored. An unwrapped payload returns ``new_inner`` unchanged. Compact
    separators, so the result is one line of JSON that Hermes' own
    ``"exit_code"\\s*:\\s*N`` matchers still read.
    """
    if not payload.wrapped or payload.field is None:
        return new_inner
    out: dict[str, Any] = {}
    for key in payload.order:
        if key == payload.field:
            out[key] = new_inner
        elif key in payload.meta:
            out[key] = payload.meta[key]
    if payload.field not in out:
        out[payload.field] = new_inner
    for key, value in payload.meta.items():
        out.setdefault(key, value)
    return _compact_json(out)


# v1.4.0 (E8): caller-supplied routing hints. diff/grep/test are recorded in
# the receipt only (their adapters arrive later); they route like None.
_CONTENT_HINTS = frozenset(
    {"code", "numbered_code", "log", "json", "text", "diff", "grep", "test"}
)
_CODE_HINTS = frozenset({"code", "numbered_code"})
# Hints whose content must never be read as a table: source code (commas and
# colons are syntax) and logs (a Java "12:00:00,123" timestamp makes every line
# look like a two-column CSV row).
_NO_TABLE_HINTS = frozenset({"code", "numbered_code", "log"})
_HINT_ROUTES = {
    "code": "code",
    "numbered_code": "code",
    "log": "log",
    "json": "json",
    "text": "text",
}
# Line-number gutter of numbered code: `cat -n` ("   12\t...") or `N|`.
_GUTTER_RE = _AGENT_GUTTER_RE  # `N\t` (cat -n), `N|` (Hermes read_file), `N: `


def _marker_tuple(markers: Iterable[str] | str | None) -> tuple[str, ...]:
    """Materialise ``recursion_markers`` once: a bare string is ONE marker (not
    a set of characters), non-strings and empty strings are dropped (an empty
    marker would match every line)."""
    if not markers:
        return ()
    if isinstance(markers, str):
        markers = (markers,)
    return tuple(m for m in markers if isinstance(m, str) and m)


def _has_anchored_marker(text: str, markers: tuple[str, ...]) -> bool:
    """True when some line of ``text`` starts with one of ``markers`` (after
    ``lstrip``). Explicit and anchored: a marker merely mentioned mid-line,
    or text that resembles a marker without being one, does not count."""
    if not markers:
        return False
    if any(m in text for m in markers) and any(
        line.lstrip().startswith(markers) for line in text.splitlines()
    ):  # (the substring test is a cheap reject before the line scan)
        return True
    return _marker_in_tool_payload(text, markers)


def _marker_in_tool_payload(text: str, markers: tuple[str, ...]) -> bool:
    """A Hermes tool result is a JSON object around ONE payload string, so a
    marker that opens a line of the payload is escaped inside the JSON text and
    never opens a line of it. Judge the unwrapped payload as well, otherwise a
    host that re-wraps its compacted output (``rewrap_tool_payload``) would see
    that output compressed a second time."""
    if text.lstrip()[:1] != "{":
        return False
    escaped = [json.dumps(m, ensure_ascii=a)[1:-1] for m in markers for a in (False, True)]
    if not any(e in text for e in escaped):
        return False
    inner = unwrap_tool_payload(text).inner
    if inner is text:
        return False
    return any(line.lstrip().startswith(markers) for line in inner.splitlines())


def preprocess(
    text: str, query: str, content_hint: str | None = None
) -> tuple[list[str], str]:
    lines = _norm_newlines(text).split("\n")
    kind = _HINT_ROUTES.get(content_hint) or route_content_type(lines)
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
) | FILLER_WORDS | frozenset(
    # v1.4.0 (E6): conversational filler — a reply made only of these names
    # nothing in the document ("continue", "thanks", "go ahead", "keep going",
    # "carry on", "sounds good"). FILLER_WORDS (format_adapters.py) supplies
    # continue/thanks/thank/ahead/proceed/okay/sure/lgtm/sounds/good/great/
    # yes/yeah/yep/cool/done/again/retry/going/carry/nice/perfect/…
    {"sounds", "looks", "works"}
)

# Words that, alone or together, make a query "filler only": the reply to the
# agent rather than a question about the text ("fix it", "ok thanks", "LGTM").
# Such a query is treated as an EMPTY query on every path (see
# _is_filler_query) — adapters, destructive preprocess, selection, fail-open.
_FILLER_QUERY_WORDS = frozenset(_WEAK_NEXT_CHAT | _STOP | {"fix", "fixes", "do", "so", "now", "then"})


def _is_filler_query(query: str) -> bool:
    """True when every word of the query is conversational filler.

    Uppercase variants count ("OK", "LGTM" would otherwise pass the
    distinctive-identifier gate as CAPS tokens). A query with any other word —
    a digit-bearing token, an identifier, a topical noun — is not filler.
    """
    words = re.findall(r"[^\W_]+", (query or "").casefold())
    return bool(words) and all(w in _FILLER_QUERY_WORDS for w in words)


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
        if (
            start is None
            or kind in {"heading", "fence", "definition"}
            or kind == "list"
            or (prev_type == "blank" and kind != "blank")
            or (kind != prev_type and kind in {"log", "chat", "import", "list", "table", "json"})
        ):
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


def score_blocks(
    blocks: list[dict[str, Any]],
    query: str,
    *,
    content_type: str | None = None,
) -> list[dict[str, Any]]:
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
        # Perf (v1.4.0): only single-token candidates can ever match the
        # per-block token sets in the boost loop below; dotted / multi-word /
        # quoted candidates never could, and their df was an O(candidates x
        # blocks) substring scan (thousands of unique `abc123.json` names in a
        # log). Skipping them changes no score.
        if not re.fullmatch(r"\w{4,}", c_lower):
            continue
        df = token_counter.get(c_lower, 0)
        # Require the entity to be in at least 2 but at most n/2 blocks
        # (if it's in every block, it's a filler word, not a bridge)
        if 2 <= df <= n // 2:
            bridge_entities[cand] = df
    # Index by lowercase token so the per-block boost is O(tokens in block),
    # not O(bridge entities). Boost values are multiples of 0.5, so summation
    # order cannot change a score.
    bridge_by_lower: dict[str, list[int]] = {}
    for ent, span in bridge_entities.items():
        bridge_by_lower.setdefault(ent.lower(), []).append(span)
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
    # v1.4.0 (E1): the regex is module-level and compiled with IGNORECASE; the
    # old call `.search(line, re.IGNORECASE)` passed the flag as `pos=2`, so it
    # matched case-sensitively and skipped the first two characters of a line.
    # A bare LEADING timestamp is a log prefix, not a fact: the search starts
    # after it, so timestamped INFO lines are not all "critical".
    crit_lines_per_block: list[int] = []
    for b in blocks:
        cnt = 0
        for line in b["text"].splitlines():
            if _is_critical_line(line):
                cnt += 1
        crit_lines_per_block.append(cnt)

    # v1.4.0 (E3): for content routed as a log, the first block carrying each
    # level-tagged error fingerprint (ERROR/FATAL/CRITICAL/PANIC/FAIL*) is an
    # "error exemplar". It is exempt from the stale-error penalty/cap, and the
    # selectors keep it even when the floor-saturated path reduces to
    # `_important`. Tracebacks and non-log content are unaffected.
    error_exemplar_ids: set[int] = set()
    if content_type == "log":
        seen_error_fps: set[str] = set()
        for b in blocks:
            for line in b["text"].splitlines():
                if _is_level_error_line(line):
                    fp = _error_fp(line)
                    if fp not in seen_error_fps:
                        seen_error_fps.add(fp)
                        error_exemplar_ids.add(b["id"])

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
        error_exemplar = b["id"] in error_exemplar_ids
        if (
            n >= 12
            and idx < n - 6
            and entity_hits == 0
            and term_hits == 0
            and not error_exemplar
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
        if bridge_by_lower:
            for tok in block_tokens:
                for span in bridge_by_lower.get(tok, ()):
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
        if error_exemplar and reason in ("context", "log error", "critical lines"):
            reason = "error exemplar"
        scored.append(
            {
                **b,
                "score": max(0.0, score),
                "reason": reason,
                "entity_hits": entity_hits,
                "term_hits": term_hits,
                "rare_term_hits": rare_term_hits,
                "trust_risk": trust_risk,
                "error_exemplar": error_exemplar,
            }
        )
    return scored


def _is_sink_noise(text: str) -> bool:
    """Trailing `-- filler` / `# filler` comments are not a recency sink."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    return all(re.match(r"^(--|#)\s*filler\b", ln, re.IGNORECASE) for ln in lines)


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
        # v1.4.0 (E3): a level-tagged log error exemplar is never "stale".
        if b.get("reason") == "stale error" and not b.get("error_exemplar"):
            continue
        # v0.6.0: trust-risk blocks are never "important" — previously the
        # trace/high-score/definition passes ignored the flag, letting an
        # untrusted instruction block into the keep-set (production QA v3
        # injection finding). The rare-candidate loop above already skips.
        if b.get("trust_risk"):
            continue
        if (
            b["type"] == "trace"
            or (b["entity_hits"] and b["score"] >= 5.0)
            or (b["score"] >= 9 and (b["entity_hits"] or b["term_hits"]))
            or (b["type"] == "definition" and b["entity_hits"])
        ):
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


_DERIVE_MIN_TERMS = 3
_DERIVE_MAX_TERMS = 8


def _derive_query_terms(text: str, *, max_terms: int = _DERIVE_MAX_TERMS) -> list[str]:
    """TPC-style derived objective for empty or generic queries.

    Infers a conservative task descriptor from the document itself: rare
    content terms that recur — appearing in too few lines marks a passing
    detail, in too many marks boilerplate — ranked by idf·log(1+tf).
    Fully deterministic (frequency statistics only, no model calls).
    Returns [] when the document has no stable term structure; callers
    keep their normal fail-open path in that case.
    """
    if not text or not text.strip():
        return []
    lines = [ln for ln in text.split("\n") if ln.strip()]
    n = len(lines)
    if n < 8:
        return []
    # Bounded work: this runs before the industrial size gate, so huge
    # inputs are sampled deterministically (uniform stride — no head bias).
    if n > 4096:
        stride = math.ceil(n / 4096)
        lines = lines[::stride]
        n = len(lines)
    df: dict[str, int] = {}
    tf: dict[str, int] = {}
    for ln in lines:
        terms = _block_link_terms(ln)
        for t in terms:
            df[t] = df.get(t, 0) + 1
            tf[t] = tf.get(t, 0) + 1
    if not df:
        return []
    # Thematic band: recurs across lines (topic) but not near-everywhere
    # (boilerplate). Cap scales with document size, floor keeps tiny docs strict.
    hi = max(3, int(n * 0.20))
    cands = [t for t, d in df.items() if 2 <= d <= hi]
    if len(cands) < _DERIVE_MIN_TERMS:
        return []
    # idf·log(1+tf), tie-broken alphabetically — set order must not leak in.
    ranked = sorted(
        cands,
        key=lambda t: (-(math.log(1 + n / df[t]) * math.log(1 + tf[t])), t),
    )
    return ranked[:max_terms]


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


# Qualifier/negation cues: a dropped block carrying one of these while
# sharing a rare entity with a kept block is a QUALIFIES edge — cutting it
# can invert the surviving claim ("safe for adults" vs "safe for adults,
# except when..."). Same vocabulary class as lcc's TRIM refusal.
_QUALIFIER_CUE_RE = re.compile(
    r"\b(except|exception|excluding|excluded|unless|only|not\b|no\b|never|"
    r"but\b|however|although|though|nevertheless|otherwise|without|despite|"
    r"until|provided\s+that|subject\s+to|if)\b",
    re.IGNORECASE,
)

# Definition cues: the dropped block DEFINES a term the kept block uses.
_DEFINITION_CUE_RE = re.compile(
    r"\b(is\s+defined\s+as|refers?\s+to|stands\s+for|means|i\.e\.|namely|denotes)\b",
    re.IGNORECASE,
)


def _dependency_closure(
    blocks: list[dict[str, Any]],
    kept: set[int],
    *,
    link_terms: dict[int, set[str]] | None = None,
    budget_tokens: int | None = None,
) -> tuple[set[int], list[int]]:
    """Sufficiency restore over typed edges (lcc graph.closure lesson).

    A dropped block that QUALIFIES or DEFINES a rare term carried by a
    kept block changes the meaning of what survives — restore it. Rare
    means document-frequency <= cap, so only genuinely linking terms form
    edges; restoration is document-ordered and capped at 8 blocks so a
    cue-dense corpus cannot flood the keep-set. Never resurrects
    trust-risk or frozen-drop blocks; supersession still runs after this
    and can evict stale restorations. ``budget_tokens`` (fixed mode)
    refuses a restoration that would overflow the caller's hard budget.
    """
    if not kept or len(blocks) < 2:
        return kept, []
    if link_terms is None:
        link_terms = {
            b["id"]: (set() if b.get("trust_risk") else _block_link_terms(b["text"]))
            for b in blocks
        }
    df: dict[str, int] = {}
    for terms in link_terms.values():
        for term in terms:
            df[term] = df.get(term, 0) + 1
    rare_cap = min(6, max(3, len(blocks) // 8))
    kept_rare = {
        term
        for bid in kept
        for term in link_terms.get(bid, ())
        if df.get(term, 0) <= rare_cap
    }
    if not kept_rare:
        return kept, []
    used = sum(b["tokens"] for b in blocks if b["id"] in kept) if budget_tokens is not None else 0
    restored: list[int] = []
    for b in blocks:  # document order -> deterministic
        if len(restored) >= 8:
            break
        bid = b["id"]
        if bid in kept or b.get("trust_risk") or b.get("freeze_decision") == "drop":
            continue
        if not (link_terms.get(bid, set()) & kept_rare):
            continue
        text = b["text"]
        if not (_QUALIFIER_CUE_RE.search(text) or _DEFINITION_CUE_RE.search(text)):
            continue
        if budget_tokens is not None and used + b["tokens"] > budget_tokens:
            continue
        restored.append(bid)
        used += b["tokens"]
    if not restored:
        return kept, []
    return kept | set(restored), restored


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
    out: dict[str, Any] | None = None,
) -> tuple[set[int], bool, str]:
    def _ret(kept, fail, risk, name):
        if out is not None:
            out["path"] = name
        return kept, fail, risk

    if not blocks:
        return _ret(set(), True, "high", "empty")
    signal = _query_signal(blocks)
    if signal < 0.04 and max(b["score"] for b in blocks) < 3.5:
        return _ret({b["id"] for b in blocks}, True, "high", "weak-signal")
    hits = {b["id"] for b in blocks if b["entity_hits"] or b["term_hits"]}
    rare = {
        b["id"]
        for b in blocks
        if b.get("rare_term_hits") or b["entity_hits"]
    }
    # v1.4.0 (E3): one exemplar block per level-tagged log error fingerprint
    # survives every selector path (stale-error cap, floor saturation, needle).
    err_keep = {
        b["id"]
        for b in blocks
        if b.get("error_exemplar") and not b.get("trust_risk")
    }
    if needle_only and hits:
        use = rare if rare else hits
        safe_use = {bid for bid in use if not blocks[bid].get("trust_risk")}
        if not safe_use:
            return _ret({b["id"] for b in blocks}, True, "high", "needle-all-trust")
        use = safe_use
        kept = set(use) | err_keep
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
            kept = set(use) | err_keep
            if not blocks[0].get("trust_risk") or blocks[0].get("pinned"):
                kept.add(blocks[0]["id"])
        kept = _stitch_neighbors(blocks, kept)
        if _counterfactual_overlap_ambiguity(
            blocks, kept, query=query, link_terms=link_terms_map, semantic_tier=semantic_tier
        ):
            return _ret({b["id"] for b in blocks}, True, "high", "needle-ambiguous")
        kept, restored = _dependency_closure(blocks, kept, link_terms=link_terms_map)
        if restored and out is not None:
            out["sufficiency_restored"] = restored
        return _ret(kept, False, "low", "needle")
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
    kept = set(important) | err_keep
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
    path = "floor"
    if floor_ratio > 0.9 and len(safe_blocks) > 64 and not needle_only:
        kept = set(important) | err_keep
        path = "floor-saturated"
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
        path = "line-records"
        kept = {
            b["id"]
            for b in blocks
            if (b["entity_hits"] or b["term_hits"]) and not b.get("trust_risk")
        }
        if not kept:
            return _ret({b["id"] for b in blocks}, True, "high", "line-records-empty")
        kept |= err_keep
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
        return _ret({b["id"] for b in blocks}, True, "high", "saturated-failopen")
    if _counterfactual_overlap_ambiguity(
        blocks, kept, query=query, link_terms=link_terms_map, semantic_tier=semantic_tier
    ):
        return _ret({b["id"] for b in blocks}, True, "high", "ambiguous-failopen")
    kept, restored = _dependency_closure(blocks, kept, link_terms=link_terms_map)
    if restored and out is not None:
        out["sufficiency_restored"] = restored
    risk = "low"
    if keep_ratio > 0.8:
        risk = "medium"
    if not important and signal < 0.1:
        # No query signal and nothing important: a guess. Retained log error
        # exemplars (E3) make it a defensible one, not a certain one.
        risk = "medium" if err_keep else "high"
    return _ret(kept, False, risk, path)


def select_fixed(
    blocks: list[dict[str, Any]],
    budget_ratio: float,
    out: dict[str, Any] | None = None,
    *,
    reserve_tokens: int = 0,
) -> tuple[set[int], bool, str]:
    if out is not None:
        out["path"] = "fixed"
    if not blocks:
        return set(), True, "high"
    total = sum(b["tokens"] for b in blocks)
    pinned_ids = {b["id"] for b in blocks if b.get("pinned")}
    if pinned_ids:
        # ARGP-style: the ratio governs compressible tokens only. Pins are
        # immovable — they must not consume the budget they outrank.
        pinned_tokens = sum(b["tokens"] for b in blocks if b["id"] in pinned_ids)
        budget = pinned_tokens + max(
            1, round(max(0, total - pinned_tokens) * budget_ratio)
        )
    else:
        budget = max(1, round(total * budget_ratio))
    if reserve_tokens > 0:
        # v1.4.0 (E8): tokens the caller's gap markers will occupy are not
        # available to content.
        budget = max(1, budget - int(reserve_tokens))
    important = _important(blocks)
    ranked = sorted(
        (b for b in blocks if not b.get("trust_risk")),
        key=lambda b: (-b["score"], b["start"]),
    )
    # v1.4.0 (E3): log error exemplars are forced into the keep-set.
    err_keep = {
        b["id"]
        for b in blocks
        if b.get("error_exemplar") and not b.get("trust_risk")
    }
    kept: set[int] = set(important) | pinned_ids | err_keep
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
    kept, restored = _dependency_closure(blocks, kept, budget_tokens=budget)
    if restored and out is not None:
        out["sufficiency_restored"] = restored
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


# Substring-scan work allowed per verification (characters examined).
_RECALL_SCAN_BUDGET_CHARS = 40_000_000

_CRITICAL_ERROR_RE = re.compile(
    r"\b(?:error|exception|traceback|fail|fatal|timeout|denied|invalid)\b", re.IGNORECASE
)
# v1.4.0 (E2): the key is bounded ({1,64}) — `[\w]+[=:]` rescanned a 20k-char
# word run from every start position (quadratic). A longer key still matches
# through its last 64 characters, so the boolean result is unchanged.
_CRITICAL_KV_RE = re.compile(r"\w{1,64}[=:]\s*\w[\w.-]*")
_CRITICAL_NUM_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b|\b\d+\.\d+\.\d+\b|\b\d{6,}\b"
    r"|\b\d+\s*(?:ms|s|sec|bytes|kb|mb|gb)\b"
)


def _extract_critical_lines(text: str) -> list[str]:
    """Extract lines that look like critical facts: error lines, key-value
    pairs with identifiers, tracebacks, and lines containing numbers + words.
    Mirrors Baseline RAG's critical_lines_total/kept/dropped reporting.

    v1.4.0: a bare leading timestamp is a prefix, not a fact — previously
    `12:03:45` satisfied the key-value rule and the date satisfied the number
    rule, so EVERY timestamped log line was "critical" and the verifier could
    not see error loss in the crowd. Error lines still are. (A chat role label
    keeps counting here, as it always did; only scoring skips it.)
    """
    critical: list[str] = []
    for line in text.split("\n"):
        t = line.strip()
        if not t:
            continue
        scan = t[:_RX_LINE_CAP]
        # Error/exception/traceback lines
        if _CRITICAL_ERROR_RE.search(scan):
            critical.append(t)
            continue
        start = _lead_ts_end(scan)
        # Lines with key-value patterns (config, settings, identifiers), or
        # lines containing specific numbers (dates, IDs, versions, counts)
        if (len(t) > 15 and _CRITICAL_KV_RE.search(scan, start)) or _CRITICAL_NUM_RE.search(
            scan, start
        ):
            critical.append(t)
    return critical


def _critical_line_recall(original: str, compressed: str) -> tuple[int, int]:
    """Return (kept, total) counts of critical lines, matching Baseline RAG
    verifier's critical_lines_kept/critical_lines_total fields."""
    critical = _extract_critical_lines(original)
    if not critical:
        return 0, 0
    kept = 0
    # Perf (v1.4.0, E2): lowercase the compressed text ONCE. Lowering it inside
    # the per-line loop was O(lines x output) — 30 s of a 35 s run on a 650 KB
    # log (and ~10x worse again when the output holds a non-ASCII "[…]").
    compressed_lower = compressed.lower()
    # Whole-line fast path: the engine emits original lines (optionally with a
    # "[×N] " collapse prefix), so nearly every kept critical line is a line of
    # the output and a set lookup answers it. Only lines the sets cannot place
    # fall back to the substring scan, under a work budget so the scan stays
    # linear on big outputs; with small outputs the budget never binds and the
    # result is exactly "core is a substring of the compressed text".
    out_lines: set[str] = set()
    out_prefixes: set[str] = set()
    for out_line in compressed_lower.split("\n"):
        out_line = out_line.strip()
        out_lines.add(out_line)
        out_prefixes.add(out_line[:80])
        bare = _COUNT_PREFIX_RE.sub("", out_line, count=1)
        if bare is not out_line:
            out_lines.add(bare)
            out_prefixes.add(bare[:80])
    scan_budget = _RECALL_SCAN_BUDGET_CHARS
    for line in critical:
        # A critical line is "kept" if its core content (first 80 chars,
        # lowercased) appears in the compressed text.
        core = line.lower()[:80]
        if core in out_lines or core in out_prefixes:
            kept += 1
        elif scan_budget > 0:
            scan_budget -= len(compressed_lower)
            if core in compressed_lower:
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


def _span_chars(lines: list[str], lo: int, hi: int) -> int:
    """Length of ``"\\n".join(lines[lo:hi])`` without building the string."""
    if hi <= lo:
        return 0
    return sum(len(ln) for ln in lines[lo:hi]) + (hi - lo - 1)


class InliningGapMarker:
    """Wrap a gap marker so renderers may inline short dropped spans.

    Renderers which see it emit a dropped span verbatim when the rendered
    marker is not shorter than the span.
    """

    __slots__ = ("fn",)
    inline_short = True

    def __init__(self, fn: Callable[[int, int], str]) -> None:
        self.fn = fn

    def __call__(self, omitted: int, total: int) -> str:
        return self.fn(omitted, total)


def _inlines_short(marker) -> bool:
    return bool(getattr(marker, "inline_short", False))


def _render(
    lines: list[str],
    blocks: list[dict[str, Any]],
    kept: set[int],
    citations: bool = False,
    reorder_best: bool = False,
    gap_marker: Callable[[int, int], str] | None = None,
    total_chars: int | None = None,
    content_lines: list[str] | None = None,
) -> str:
    chronological_blocks = sorted(blocks, key=lambda block: block["start"])
    block_starts = [block["start"] for block in chronological_blocks]
    order = sorted((b for b in blocks if b["id"] in kept), key=lambda b: b["start"])
    chronological = True
    if reorder_best and len(order) >= 3:
        # Lost-in-the-middle mitigation (LongLLMLingua / twotrim): models
        # attend hardest to window edges. Best-scoring block anchors the
        # front, second-best anchors the end, the rest stay chronological.
        ranked = sorted(order, key=lambda b: -float(b.get("score", 0.0)))
        best, second = ranked[0], ranked[1]
        rest = [b for b in order if b["id"] not in {best["id"], second["id"]}]
        order = [best, *rest, second]
        chronological = False
    # v1.4.0 (E8): a caller-supplied gap marker replaces EVERY gap marker
    # (citations=False only): leading, internal and trailing. It is called as
    # gap_marker(omitted_chars, total_chars) with the length of the dropped
    # span (its lines joined with "\n") and the length of the caller's input.
    # Spans of blank lines only are not content and get no marker, and one
    # dropped span yields exactly one marker however many blocks it holds.
    # gap_marker=None keeps the historical "[…]" behaviour byte for byte.
    marker_fn = gap_marker if (gap_marker is not None and not citations) else None
    total = total_chars if total_chars is not None else _span_chars(lines, 0, len(lines))
    # "Is this span blank?" is judged on the lines the scorer read: a
    # numbered-code gutter ("75|", "    75\t") makes an empty source line
    # non-empty in `lines`, and would earn a bogus marker between two
    # adjacent kept blocks.
    probe_lines = content_lines if content_lines is not None else lines
    inline_short = _inlines_short(marker_fn) if marker_fn is not None else False

    def _gap(lo: int, hi: int) -> list[str] | None:
        if marker_fn is None:
            return None
        if not any(ln.strip() for ln in probe_lines[lo:hi]):
            return None
        span_text = "\n".join(lines[lo:hi]) if inline_short else None
        text = marker_fn(_span_chars(lines, lo, hi), total)
        marker = str(text) if text is not None else ""
        if not marker:
            return None
        # 1.4.1 opt-in: only InliningGapMarker emits the dropped lines
        # verbatim when the rendered marker is not shorter than the span.
        if inline_short and span_text is not None and len(marker) >= len(span_text):
            return lines[lo:hi]
        return [marker]

    parts: list[str] = []
    prev_end = -1
    for b in order:
        if marker_fn is not None:
            if prev_end < 0 and chronological and b["start"] > 0:
                lead = _gap(0, b["start"])
                if lead is not None:
                    parts.extend(lead)
            elif prev_end >= 0 and b["start"] > prev_end + 1:
                between = _gap(prev_end + 1, b["start"])
                if between is not None:
                    parts.extend(between)
        elif prev_end >= 0 and b["start"] > prev_end + 1:
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
                    d_text = d.get("full_text") or d["text"]
                    head = d_text.split("\n", 1)[0][:60]
                    tail = d_text.rsplit("\n", 1)[-1][:60]
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
                            d_text = d.get("full_text") or d["text"]
                            digest = hashlib.sha256(d_text.encode("utf-8")).hexdigest()[:8]
                            head = d_text.split("\n", 1)[0][:60]
                            tail = d_text.rsplit("\n", 1)[-1][:60]
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
    if marker_fn is not None and chronological and parts and prev_end < len(lines) - 1:
        trail = _gap(prev_end + 1, len(lines))
        if trail is not None:
            parts.extend(trail)
    return "\n".join(parts)


def _gap_marker_reserve(
    lines: list[str],
    blocks: list[dict[str, Any]],
    kept: set[int],
    gap_marker: Callable[[int, int], str],
    total_chars: int,
    content_lines: list[str] | None = None,
) -> int:
    """Tokens the caller's gap markers will take for this keep-set: the number
    of dropped spans that hold content (leading, internal, trailing) times the
    size of a marker. Mirrors _render's gap rules."""
    spans = sorted((b["start"], b["end"]) for b in blocks if b["id"] in kept)
    if not spans:
        return 0
    n = len(lines)
    probe_lines = content_lines if content_lines is not None else lines

    def content(lo: int, hi: int) -> bool:
        return any(ln.strip() for ln in probe_lines[lo:hi])

    gaps = 1 if spans[0][0] > 0 and content(0, spans[0][0]) else 0
    prev_end = spans[0][1]
    for start, end in spans[1:]:
        if start > prev_end + 1 and content(prev_end + 1, start):
            gaps += 1
        prev_end = max(prev_end, end)
    if prev_end < n - 1 and content(prev_end + 1, n):
        gaps += 1
    if not gaps:
        return 0
    # One probe call sizes a marker at its widest (a span as long as the
    # whole input); callers that count gap_marker calls see exactly one extra
    # call in fixed mode, none in adaptive mode.
    sample = str(gap_marker(total_chars, total_chars) or "")
    return gaps * estimate_tokens(sample)


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

    Files that cannot be read or parsed are left untouched: retention
    maintenance is best-effort. A file that parses but is not a valid record
    (bad metadata, a non-object payload) is removed like an expired one.
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


def _ccr_store(original: str, ccr_dir: str | Path) -> dict[str, Any] | None:
    # Secrets screen: archival would persist credentials to disk at rest.
    if _contains_secret(original):
        return None
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
            (line.startswith("  ") or line.startswith("\t"))
            and (
                stripped.startswith("{")
                or stripped.startswith("[")
                or stripped.startswith('"')
            )
        ) or (stripped.startswith("TOOL:") and len(stripped) > 30):
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


# v1.4.0 (E2): the two class runs are bounded ({0,127}); the unbounded form
# re-scanned the rest of a long separator-rich run (`a.a.a.…`, 20k chars) from
# every word boundary — quadratic. Tokens up to 256 chars match as before.
_SUMMARY_STRUCTURED_TOKEN_RE = re.compile(
    r"\b(?:[A-Za-z][A-Za-z0-9_.:/-]{0,127}\d[A-Za-z0-9_.:/-]{0,127}"
    r"|\d+(?:[.:/-]\d+)*)\b"
)


def _summary_preserves_required_facts(
    source: str, summary: str, query: str
) -> tuple[bool, float]:
    """Conservative factual gate for the optional model-summary strategy."""
    source_fold = source.casefold()
    summary_fold = summary.casefold()
    # Perf (v1.4.0, E2): `required` stays a list (order is part of the
    # contract) but membership is answered by a set — `token not in list` made
    # this loop quadratic (92 s on a 3.7 MB history).
    required: list[str] = []
    required_set: set[str] = set()
    candidates = (
        _SUMMARY_STRUCTURED_TOKEN_RE.findall(query or "")
        + _extract_entities(query or "")
        + _extract_terms(query or "")
    )
    for item in candidates:
        token = str(item).strip().casefold()
        if token and token in source_fold and token not in required_set:
            required.append(token)
            required_set.add(token)

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
            if token not in required_set:
                required.append(token)
                required_set.add(token)
        # Plain-text scalar answers are not necessarily identifiers (for
        # example "paint color is ultraviolet"). Require the first content
        # term after a queried field and a simple relation verb; do not require
        # every adjective in a long source segment.
        for query_term in sorted(query_terms, key=len, reverse=True):
            match = re.search(
                rf"(?<!\w){re.escape(query_term)}(?!\w)\s+"
                r"(?:is|are|was|were|equals?|uses?|=)\s+(.+)$",
                segment,
                re.IGNORECASE,
            )
            if not match:
                continue
            answer_terms = _extract_terms(match.group(1))
            if answer_terms:
                token = answer_terms[0].casefold()
                if token not in query_terms and token not in required_set:
                    required.append(token)
                    required_set.add(token)
            break
    source_ids = {
        token.casefold() for token in _SUMMARY_STRUCTURED_TOKEN_RE.findall(source)
    }
    summary_ids = {
        token.casefold() for token in _SUMMARY_STRUCTURED_TOKEN_RE.findall(summary)
    }
    # Perf (v1.4.0): a required token that is itself a token of the summary is
    # found by a set lookup; only the others pay the substring scan (which is
    # what decides the same question, since a token of the summary is a
    # substring of it). Same answer, linear when the summary kept its facts.
    kept = sum(
        1 for token in required if token in summary_ids or token in summary_fold
    )
    recall = kept / len(required) if required else 1.0
    no_novel_ids = summary_ids.issubset(source_ids)
    return recall == 1.0 and no_novel_ids, recall


def retrieve(
    ccr_hash: str,
    ccr_dir: str | Path = DEFAULT_CCR_DIR,
    *,
    offset: int = 0,
    limit: int | None = None,
) -> Optional[str]:
    """Fetch a CCR record's original text. offset/limit paginate large
    records (pi-lcm lcm_expand-style partial recall)."""
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
    if offset or limit is not None:
        start = max(0, int(offset))
        return original[start:] if limit is None else original[start:start + max(0, int(limit))]
    return original


def list_ccr(
    ccr_dir: str | Path = DEFAULT_CCR_DIR,
    *,
    offset: int = 0,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """List live CCR records newest-first (pi-lcm lcm_grep/describe analog).

    Returns metadata only — hash, stored_at, ttl, chars, and a short
    preview — so callers can discover records without loading full
    originals. Expired or malformed records are skipped.
    """
    root = Path(ccr_dir)
    if root.is_symlink() or not root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    current = time.time()
    for fp in root.glob("*.json"):
        if not fp.name.startswith(".") and _CCR_HASH_RE.fullmatch(fp.stem):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, RecursionError):
                continue
            if not isinstance(data, dict):
                continue
            metadata = _ccr_metadata(data)
            if (
                metadata is None
                or metadata[0] > current + CCR_MAX_CLOCK_SKEW_SECONDS
                or current - metadata[0] > metadata[1]
            ):
                continue
            record_hash = data.get("hash")
            original = data.get("original")
            if (
                not isinstance(record_hash, str)
                or record_hash != fp.stem
                or not isinstance(original, str)
            ):
                continue
            out.append({
                "hash": record_hash,
                "stored_at": metadata[0],
                "ttl": metadata[1],
                "chars": len(original),
                "preview": original[:80],
            })
    out.sort(key=lambda r: (-r["stored_at"], r["hash"]))
    start = max(0, int(offset))
    if limit is None:
        return out[start:]
    return out[start:start + max(0, int(limit))]


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
    except (ValueError, RecursionError):  # JSONDecodeError + the 4300-digit int limit
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
            "guaranteed_savings_pct": 0.0,
            "ceiling_class": "n/a",
            "reason": f"too small ({tokens} tokens < 200)",
        }
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return {
            "worth_it": False,
            "repetition_ratio": 0.0,
            "guaranteed_savings_pct": 0.0,
            "ceiling_class": "n/a",
            "reason": "empty",
        }
    # Repetition ratio: fraction of duplicate lines (exact + normalised runs).
    from collections import Counter

    norm = [re.sub(r"\d+", "#", ln) for ln in lines]
    counts = Counter(norm)
    dup_lines = sum(c for c in counts.values() if c > 1)
    repetition_ratio = dup_lines / len(lines)
    # Context-calibrated ceiling (Compactor): not every context compresses
    # the same. The guaranteed-safe bound is the surplus share of duplicated
    # normalised lines — bytes a dedupe pass removes with zero judgement.
    # Beyond it, savings depend on relevance selection, not redundancy.
    total_chars = sum(len(ln) + 1 for ln in lines)
    surplus_chars = sum((c - 1) * (len(ln) + 1) for ln, c in counts.items() if c > 1)
    guaranteed = surplus_chars / max(1, total_chars)
    ceiling_class = (
        "dedupe-heavy" if guaranteed >= 0.40
        else "moderate" if guaranteed >= 0.15
        else "sparse"
    )
    # Distinct-line ratio as a secondary signal: highly unique prose with no
    # query hits rarely compresses extractively without damage.
    worth = repetition_ratio >= 0.3 or tokens >= 5000
    return {
        "worth_it": worth,
        "repetition_ratio": round(repetition_ratio, 3),
        "guaranteed_savings_pct": round(guaranteed, 3),
        "ceiling_class": ceiling_class,
        "reason": (
            f"repetition {repetition_ratio:.0%}, {tokens} tokens"
            if worth
            else f"low repetition ({repetition_ratio:.0%}) and moderate size"
        ),
    }


# ---------------------------------------------------------------------------
# v1.4.0 public helpers: brief_context / extract_evidence_lines
#
# Deterministic, stdlib-only, query-optional "what is in here" views for hosts
# that need a small, honest stand-in for a big tool result (a plugin's last
# rung after the query-aware extract was rejected or too big). Both return
# ORIGINAL lines only — nothing is paraphrased — and both treat a level-tagged
# error as the most important thing in a log.
# ---------------------------------------------------------------------------

_WARN_LEVEL_RE = re.compile(r"\b(?:WARN|WARNING)\b")
_EVID_VERSION_RE = re.compile(r"\bv?\d+\.\d+(?:\.\d+)+\b")
_EVID_SHA_RE = re.compile(r"\b(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}\b")
_EVID_KV_RE = re.compile(r"\b[A-Za-z_][\w.-]{0,63}=\S")
_EVID_FILE_RE = re.compile(
    r"\b[\w-]{1,64}\.(?:py|js|ts|tsx|go|rs|java|c|h|cpp|json|ya?ml|toml|md|txt|log|sh|sql|cfg|ini)\b"
)
_EVIDENCE_LINE_CAP = 2000


def _query_hit_terms(query: str) -> list[str]:
    """Lowercase, deduplicated query terms worth searching lines for."""
    if not query or _is_filler_query(query):
        return []
    terms: list[str] = []
    seen: set[str] = set()
    for term in [
        t for t in distinctive_query_terms(query) if not t.startswith("script:")
    ] + _topic_terms(query):
        key = term.casefold()
        if key and key not in seen:
            seen.add(key)
            terms.append(key)
    return terms


def brief_context(
    text: str,
    *,
    budget_chars: int = 1200,
    query: str = "",
    gap_marker: Callable[[int, int], str] | None = None,
) -> str:
    """Deterministic, query-independent structural brief of ``text`` (v1.4.0).

    Keeps ORIGINAL lines, in original order, within ``budget_chars`` (newlines
    and gap markers included):

    1. the first ``h`` lines;
    2. every error-fingerprint exemplar line (one per fingerprint, see
       :func:`error_fingerprints`);
    3. warning (WARN/WARNING) exemplars, one per template, if they fit;
    4. lines holding distinctive terms of ``query``, if one is given;
    5. the last ``t`` lines.

    Head and tail are given about 30% of the budget each and the rest goes to
    the exemplars; if the errors do not fit that way the head/tail shrink so
    they do, and whatever budget the exemplars leave over goes back to the
    head and tail. When the budget cannot hold everything, errors outrank
    query-term lines, which outrank warnings.

    Each omitted span is marked with ``gap_marker(omitted_chars, total_chars)``
    — or ``[… N chars omitted …]`` when it is ``None`` — and a span of blank
    lines only is not marked. ``gap_marker`` is called once per marker that is
    emitted, plus one probe with ``(total_chars, total_chars)`` that sizes the
    widest marker for the budget. A line is never cut, except one that is
    itself as long as the budget: it is truncated, with a marker. ``text`` is
    returned unchanged when it already fits. A ``gap_marker`` wrapped in
    :class:`InliningGapMarker` keeps a dropped span verbatim instead when its
    marker would not be shorter than the span (1.4.1).
    """
    text = text if isinstance(text, str) else str(text or "")
    budget = max(1, int(budget_chars))
    if len(text) <= budget:
        return text
    # v1.4.0 (E11): JSON arrays of objects get a valid-JSON brief, not a
    # line-cut that would leave unparseable fragments.
    if text.lstrip()[:1] in ("[", "{"):
        briefed = json_brief(text, budget_chars=budget, query=query)
        if briefed is not None:
            return briefed
    total = len(text)
    lines = text.split("\n")
    n = len(lines)

    def mark(omitted: int) -> str:
        if gap_marker is not None:
            made = gap_marker(omitted, total)
            return str(made) if made else ""
        return f"[… {omitted} chars omitted …]"

    # Markers are sized at the widest count a span can have: ONE probe call,
    # gap_marker(total, total), before any real span is known. (Every later call
    # is for a span that is actually emitted.)
    marker_len = len(mark(total))

    eff_cache: dict[int, str] = {}

    def eff(i: int) -> str:
        cached = eff_cache.get(i)
        if cached is None:
            line = lines[i]
            if len(line) + 1 >= budget:
                cut = max(1, budget // 3)
                line = line[:cut] + mark(len(lines[i]) - cut)
            eff_cache[i] = cached = line
        return cached

    nonblank = [0] * (n + 1)
    for i, line in enumerate(lines):
        nonblank[i + 1] = nonblank[i] + (1 if line.strip() else 0)

    def has_content(lo: int, hi: int) -> int:
        return 1 if nonblank[hi] - nonblank[lo] > 0 else 0

    # Candidate lines by priority class (indices, original order).
    err_idx: list[int] = []
    warn_idx: list[int] = []
    seen_err: set[str] = set()
    seen_warn: set[str] = set()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        if _is_error_line(line):
            fp = _error_fp(line)
            if fp not in seen_err:
                seen_err.add(fp)
                err_idx.append(i)
        elif _WARN_LEVEL_RE.search(line[:_RX_LINE_CAP]):
            fp = _error_fp(line)
            if fp not in seen_warn:
                seen_warn.add(fp)
                warn_idx.append(i)
    hit_idx: list[int] = []
    terms = _query_hit_terms(query)
    if terms:
        ranked: list[tuple[int, int]] = []
        seen_hit: set[str] = set()
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            low = line[:_RX_LINE_CAP].casefold()
            hits = sum(1 for t in terms if _term_in_text(t, low))
            if hits:
                ranked.append((-hits, i))
        for _, i in sorted(ranked):
            fp = _error_fp(lines[i])
            if fp not in seen_hit:
                seen_hit.add(fp)
                hit_idx.append(i)

    def build(head_cap: int, tail_cap: int, errors_first: bool) -> tuple[list[int], bool]:
        sel: list[int] = []
        chosen_set: set[int] = set()
        state = {"raw": 0, "gaps": 0}

        def gap_delta(p: int | None, q: int | None, i: int) -> int:
            if p is None and q is None:
                return has_content(0, i) + has_content(i + 1, n)
            if p is None:
                return has_content(0, i) + has_content(i + 1, q) - has_content(0, q)
            if q is None:
                return has_content(p + 1, i) + has_content(i + 1, n) - has_content(p + 1, n)
            return has_content(p + 1, i) + has_content(i + 1, q) - has_content(p + 1, q)

        def try_add(i: int) -> bool:
            if i in chosen_set:
                return True
            pos = bisect.bisect_left(sel, i)
            p = sel[pos - 1] if pos > 0 else None
            q = sel[pos] if pos < len(sel) else None
            raw = state["raw"] + len(eff(i)) + 1
            gaps = state["gaps"] + gap_delta(p, q, i)
            if raw - 1 + gaps * (marker_len + 1) > budget:
                return False
            sel.insert(pos, i)
            chosen_set.add(i)
            state["raw"], state["gaps"] = raw, gaps
            return True

        def add_head() -> None:
            used = 0
            for i in range(n):
                if not lines[i].strip():
                    continue
                c = len(eff(i)) + 1
                if i > 0 and used + c > head_cap:
                    break
                if not try_add(i):
                    break
                used += c

        def add_tail() -> None:
            used = 0
            for i in range(n - 1, -1, -1):
                if i in chosen_set and used:  # met the head
                    break
                if not lines[i].strip():
                    continue
                c = len(eff(i)) + 1
                if used + c > tail_cap:
                    break
                if not try_add(i):
                    break
                used += c

        def add_exemplars() -> bool:
            all_errors = True
            for i in err_idx:
                if not try_add(i):
                    all_errors = False
            # When the budget cannot hold everything, lines that answer the
            # caller's query outrank warning exemplars.
            for group in (hit_idx, warn_idx):
                for i in group:
                    try_add(i)
            return all_errors

        def fill_edges() -> None:
            """Spend what is left on more head and tail, alternately."""
            hi, ti = 0, n - 1
            head_open = tail_open = True
            while (head_open or tail_open) and hi <= ti:
                if head_open:
                    while hi < n and (hi in chosen_set or not lines[hi].strip()):
                        hi += 1
                    if hi >= n or not try_add(hi):
                        head_open = False
                if tail_open:
                    while ti >= 0 and (ti in chosen_set or not lines[ti].strip()):
                        ti -= 1
                    if ti < 0 or not try_add(ti):
                        tail_open = False

        if errors_first:
            all_errors = add_exemplars()
            add_head()
            add_tail()
        else:
            add_head()
            add_tail()
            all_errors = add_exemplars()
        fill_edges()
        return sel, all_errors

    chosen: list[int] = []
    for share, errors_first in ((0.30, False), (0.15, False), (0.0, True)):
        cap = int(budget * share)
        chosen, all_errors = build(cap, cap, errors_first)
        if all_errors:
            break

    # 1.4.1 (fix-all): an InliningGapMarker gets the dropped lines verbatim
    # when its marker would not be shorter than them, as in compress_context.
    inline_short = _inlines_short(gap_marker)

    def gap_parts(lo: int, hi: int) -> list[str]:
        if not has_content(lo, hi):
            return []
        span_chars = _span_chars(lines, lo, hi)
        made = mark(span_chars)
        if not made:
            return []
        if inline_short and len(made) >= span_chars:
            return lines[lo:hi]
        return [made]

    parts: list[str] = []
    prev = -1
    for i in chosen:
        lo = prev + 1
        if prev < 0 and i > 0:
            parts.extend(gap_parts(0, i))
        elif prev >= 0 and i > prev + 1:
            parts.extend(gap_parts(lo, i))
        parts.append(eff(i))
        prev = i
    if chosen and prev < n - 1:
        parts.extend(gap_parts(prev + 1, n))
    out = "\n".join(parts)
    if not out:
        # Not even one line (with its marker) fits a budget this small: the
        # plain prefix is the only honest brief left.
        return text[:budget]
    if len(out) > budget:  # only a gap_marker that outgrew its probe can get here
        out = out[:budget]
    return out


def extract_evidence_lines(text: str, query: str = "", limit: int = 20) -> list[str]:
    """Exact original lines most likely to matter, deduplicated, capped (v1.4.0).

    Ranking, most important first:

    1. error exemplars — one line per :func:`error_fingerprints` fingerprint;
    2. lines holding terms of ``query`` (more distinct terms first);
    3. identifier-dense lines — at least two of: file paths, versions, SHAs,
       ``key=value`` pairs.

    Lines are returned unmodified, in original order, at most ``limit`` of
    them. Lines that differ only in volatile fields (ids, numbers, timestamps)
    count as duplicates, so a thousand repeats cannot fill the list. Blank
    lines are never returned. Deterministic.
    """
    if not text or limit <= 0:
        return []
    lines = _norm_newlines(text).split("\n")
    picked: set[int] = set()
    seen_tpl: set[str] = set()

    def take(i: int) -> bool:
        if len(picked) >= limit:
            return False
        fp = _error_fp(lines[i])
        if fp in seen_tpl or i in picked:
            return True
        seen_tpl.add(fp)
        picked.add(i)
        return True

    for i, line in enumerate(lines):
        if line.strip() and _is_error_line(line):
            if not take(i):
                break
    terms = _query_hit_terms(query)
    if terms and len(picked) < limit:
        ranked = []
        for i, line in enumerate(lines):
            if not line.strip() or i in picked:
                continue
            low = line[:_RX_LINE_CAP].casefold()
            hits = sum(1 for t in terms if _term_in_text(t, low))
            if hits:
                ranked.append((-hits, i))
        for _, i in sorted(ranked):
            if not take(i):
                break
    if len(picked) < limit:
        dense = []
        for i, line in enumerate(lines):
            if not line.strip() or i in picked or len(line) > _EVIDENCE_LINE_CAP:
                continue
            score = (
                len(_PATH_RE.findall(line))
                + len(_EVID_FILE_RE.findall(line))
                + len(_EVID_VERSION_RE.findall(line))
                + len(_EVID_SHA_RE.findall(line))
                + len(_EVID_KV_RE.findall(line))
            )
            if score >= 2:
                dense.append((-score, i))
        for _, i in sorted(dense):
            if not take(i):
                break
    return [lines[i] for i in sorted(picked)]


def compress_context(
    context: str,
    query: str | Iterable[str],
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
    pin_recent: int = 0,
    min_savings_ratio: float = 0.10,
    degraded_view: bool = False,
    summary_endpoint: str | None = None,
    summary_models: Iterable[str] | str | None = None,
    summary_timeout: float | None = None,
    summary_allow_remote: bool | None = None,
    limits: IndustrialLimits | None = None,
    derive_query: bool = False,
    gap_marker: Callable[[int, int], str] | None = None,
    recursion_markers: Iterable[str] = (),
    content_hint: str | None = None,
    inline_short_gaps: bool = False,
) -> CompressResult:
    """Query-aware extractive compression of ``context``.

    New in 1.4.0 (all optional, defaults keep 1.3.x behaviour):

    ``gap_marker(omitted_chars, total_chars) -> str``
        Replaces every gap marker the renderer emits when ``citations=False``
        — including a leading gap (kept text starts after line 0) and a
        trailing gap — with the caller's own marker text. ``omitted_chars`` is
        the length of the dropped span (its lines joined with ``"\\n"``),
        ``total_chars`` the length of ``context``. Adjacent dropped blocks
        yield exactly one marker. ``None`` keeps the historical ``[…]``.
    ``recursion_markers``
        Extra strings that mark input as already compressed: if any line of
        the input starts with one (after ``lstrip``) the call is a no-op with
        ``policy_name="local-noop-recursion"``. Matching is explicit and
        anchored; nothing is guessed (a host's own marker text only counts if
        the caller passes it).
    ``content_hint``
        One of ``None, "code", "numbered_code", "log", "json", "text",
        "diff", "grep", "test"`` (anything else raises ``ValueError``).
        ``"code"``/``"numbered_code"`` skip the CSV/TSV/YAML paths and route
        as code so function bodies survive (``"log"`` skips them too: a log is
        never a table); ``"numbered_code"`` additionally
        means each line carries a ``N\\t`` or ``N|`` gutter, which stays in the
        output and is ignored for scoring. ``"log"``/``"json"``/``"text"``
        force that routing. ``"diff"``/``"grep"``/``"test"`` are recorded in
        the receipt and otherwise behave like ``None``. The hint is always
        recorded in ``receipt["content_hint"]``.

    ``inline_short_gaps`` (1.4.1)
        With ``gap_marker`` set, a dropped span whose marker would not be
        shorter than the span itself is kept verbatim instead. Off by
        default: 1.4.0 output is unchanged byte for byte.
    """
    caller_text = str(context or "")
    if not isinstance(query, str):
        # PAACE-style plan awareness: a sequence of upcoming-task strings is
        # scored as a union — a term from any planned step counts.
        query = " ".join(str(part) for part in (query or []) if part)
    if content_hint is not None and (
        not isinstance(content_hint, str) or content_hint not in _CONTENT_HINTS
    ):
        raise ValueError(
            f"unknown content_hint {content_hint!r} "
            f"(expected None or one of {', '.join(sorted(_CONTENT_HINTS))})"
        )
    if gap_marker is not None and not callable(gap_marker):
        raise TypeError("gap_marker must be callable as gap_marker(omitted_chars, total_chars)")
    if inline_short_gaps and gap_marker is not None and not _inlines_short(gap_marker):
        gap_marker = InliningGapMarker(gap_marker)
    code_hint = content_hint in _CODE_HINTS
    no_table = content_hint in _NO_TABLE_HINTS
    recursion_markers = _marker_tuple(recursion_markers)  # a generator is read once
    # v1.4.0 (E6): a reply made only of conversational filler ("continue",
    # "thanks", "go ahead", "fix it", "OK") is an EMPTY query on every path.
    # The receipt/cache wrapper still show the caller's words.
    query_as_given = query
    if _is_filler_query(query):
        query = ""
    # TPC-style derived objective (opt-in): an empty/generic query normally
    # fails open. With derive_query=True the engine first tries to infer a
    # conservative task descriptor from the document's own recurring rare
    # terms; when no stable term structure exists the original empty query
    # stands and the usual fail-open applies. Receipts mark derived
    # objectives via query_source/derived_terms and the risk floor is
    # medium — an inferred intent never earns a "low" rating.
    derived_terms: list[str] = []
    if (
        derive_query
        and not query_has_distinctive_selectors(query or "")
        and not _topic_terms(query or "")
    ):
        derived_terms = _derive_query_terms(caller_text)
        if derived_terms:
            query = " ".join(derived_terms)
    requested_mode = (mode or "adaptive").strip().lower()
    if requested_mode not in {"adaptive", "fixed", "compiler", "precision"}:
        raise ValueError(
            f"unknown mode {mode!r} (expected adaptive/fixed/compiler/precision)"
        )
    mode_norm = (
        "adaptive" if requested_mode in {"compiler", "precision"} else requested_mode
    )
    pin_recent = max(0, int(pin_recent or 0))
    if min_savings_ratio is None:
        min_savings_ratio = 0.10
    min_savings_ratio = float(min_savings_ratio)
    if not 0.0 <= min_savings_ratio < 1.0:
        raise ValueError(
            f"min_savings_ratio must be in [0, 1), got {min_savings_ratio!r}"
        )
    # Recursion guard (Claude Code teardown inspired): Tameru output fed
    # back as input would nest wrappers and could drop the CCR pointer
    # that makes earlier drops recoverable. Refuse to compact our own
    # output — the caller already has the compacted form.
    if any(marker in caller_text for marker in _RECURSION_MARKERS) or (
        _has_anchored_marker(caller_text, recursion_markers)
    ):
        tokens = estimate_tokens(caller_text)
        return CompressResult(
            compressed_text=caller_text,
            original_tokens=tokens,
            kept_tokens=tokens,
            tokens_saved_pct=0.0,
            policy_name="local-noop-recursion",
            mode=requested_mode,
            keep_ratio=1.0,
            tokens_saved=0,
            kept_line_ratio=1.0,
            compression_risk="low",
            confidence=1.0,
            fail_open=True,
            reasons=["input already compressed (recursion guard)"],
        )
    secret_in_input = _contains_secret(caller_text)
    savings_gate = False
    degraded = False
    # content_hint code/numbered_code (and log): the caller says what this is,
    # so the format adapters (CSV/TSV/YAML row and subtree selection, which
    # treat code with commas or colons — or a log with "12:00:00,123"
    # timestamps — as a table) must not run. An empty selector query makes the
    # adapter decline; the query-size limit is still enforced by passing the
    # real query when it is over the limit.
    industrial_query = query or ""
    if no_table and len(industrial_query) <= (limits or IndustrialLimits()).max_query_chars:
        industrial_query = ""
    # v1.4.0 (E8 completion): adapter cuts carry the caller's marker; the
    # total they report is always the caller's original length.
    adapter_gap = (
        None
        if gap_marker is None
        else (
            InliningGapMarker(lambda omitted, _total: gap_marker(omitted, len(caller_text)))
            if _inlines_short(gap_marker)
            else (lambda omitted, _total: gap_marker(omitted, len(caller_text)))
        )
    )
    industrial = industrial_preprocess(
        caller_text, industrial_query, limits, gap_marker=adapter_gap
    )
    if (
        industrial.hard_fail_open
        and degraded_view
        and industrial.profile.limit_reason.startswith(
            ("character limit", "line limit")
        )
        and len(caller_text)
        <= industrial.limits.max_input_chars * _DEGRADED_OVERSIZE_FACTOR
        and industrial.profile.lines
        <= industrial.limits.max_lines * _DEGRADED_OVERSIZE_FACTOR
        and len(query or "") <= industrial.limits.max_query_chars
    ):
        # JEV-style degrade: keep size limits as a throughput guard, but let
        # the scorer see a bounded per-block view instead of refusing.
        # Safety gates preflight skipped (bidi controls, malformed
        # surrogates, oversize query) are re-verified here and stay hard.
        controls, overrides, malformed = unicode_safety_counts(caller_text)
        lim = industrial.limits
        if (
            not malformed
            and controls <= lim.max_bidi_controls
            and overrides <= lim.max_bidi_overrides
        ):
            industrial = replace(
                industrial,
                hard_fail_open=False,
                reason=f"degraded scoring view: {industrial.reason}",
            )
            degraded = True
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
                    (query_as_given or "").encode("utf-8")
                ).hexdigest()[:12],
                "savings_pct": 0.0,
                "risk": "high",
                "content_hint": content_hint,
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
    # v1.4.0 (G9): keep the wrapper's scalar metadata (exit_code, error,
    # total_lines, ...) so it can be re-emitted ahead of the compressed body.
    tool_payload = unwrap_tool_payload(text)
    text = tool_payload.inner
    tool_meta_line = ""
    if tool_payload.wrapped and tool_payload.meta:
        tool_meta_line = _compact_json(tool_payload.meta) + "\n"
    # v1.4.0 (E4): an un-hinted numbered read (`cat -n`, Hermes `N|`) is code,
    # judged on the unwrapped payload: no table detector, no supersession, the
    # gutter kept in the output but out of the scorer's view. `route_hint` is
    # what the engine routes by; the receipt still records the caller's hint.
    # A "code" hint on a numbered read means the same thing: the gutter must
    # not count as code (it made blank source lines non-blank and split
    # function bodies at the cut).
    route_hint = content_hint
    if route_hint in (None, "code") and is_numbered_code(text):
        route_hint = "numbered_code"
        code_hint = True
        no_table = True
    _pre_test_text = text
    text = preprocess_test_runner(
        text, query or "", content_hint=content_hint, gap_marker=adapter_gap
    )
    # v1.4.0 (E5): diff and grep/rg output get their own adapters (whole hunks,
    # grouped matches) ahead of generic segmentation. The adapter's output is
    # the selection: scoring below must not undo it (see `adapter_info`).
    adapter_info: dict[str, Any] | None = None
    if (
        text is not _pre_test_text
        and content_hint in (None, "test")
        and len(text) <= _TEST_REDUCTION_FINAL_CHARS
        and detect_test_runner(_pre_test_text) is not None
    ):
        # Same for a reduced test run (v1.4.0): block scoring must not drop
        # the failing test's name/file line that the reduction kept (jest
        # "FAIL f.test.js" / "● suite › case" vs "which tests failed?").
        # Larger reductions still go through scoring and its budget.
        adapter_info = {
            "name": "test_output",
            "total_records": None,
            "kept_records": None,
            "reason": "failure blocks and summaries kept",
        }
    if (
        adapter_info is None
        and route_hint not in _CODE_HINTS
        and route_hint not in ("log", "json", "text")
    ):
        text, adapter_info = apply_agent_adapters(
            text,
            query or "",
            hint=content_hint,
            gap_marker=adapter_gap,
            weak_query=not query_has_distinctive_selectors(query or ""),
        )
    if not text.strip():
        return CompressResult(
            compressed_text="", policy_name="noop", mode=requested_mode
        )
    ambiguity_fail_open = bool(ambiguity_fail_open)

    # Destructive preprocess (JSON crush / later log collapse) only when
    # the query names something specific. Generic/empty queries must not
    # delete array tails or fingerprint-collapse logs before scoring.
    if query_has_distinctive_selectors(query or "") and not code_hint and not adapter_info:
        # (content_hint code/numbered_code: source code with commas or JSON
        # literals is not a table to crush; function bodies must survive.)
        if route_content_type(_norm_newlines(text).split("\n")) != "json":
            text = _preprocess_json(text, query or "")
        if not no_table:
            text = preprocess_csv(text, query or "", gap_marker=adapter_gap)
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
    if strategy_norm not in {"clear", "extract", "summarise", "auto"}:
        raise ValueError(
            f"unknown strategy {strategy!r} (expected clear/extract/summarise/auto)"
        )

    if strategy_norm == "auto":
        # Progressive ladder (Claude Code tiers / JEV hook fallback):
        # run the cheap deterministic tier first, then escalate to the
        # LLM summariser only when extraction fails open (ambiguity,
        # saturated floors, min_savings undershoot). 'summarise' itself
        # falls back to 'extract' on any LLM failure, so the worst case
        # here is one extra deterministic pass.
        ladder_kwargs = dict(
            mode=requested_mode,
            budget_ratio=budget_ratio,
            ccr=ccr,
            cache_prefix=cache_prefix,
            citations=citations,
            ccr_dir=ccr_dir,
            decision_cache=decision_cache,
            ambiguity_fail_open=ambiguity_fail_open,
            reorder_best=reorder_best,
            semantic_tier=semantic_tier,
            log_dir=log_dir,
            pin_patterns=pin_patterns,
            pin_recent=pin_recent,
            min_savings_ratio=min_savings_ratio,
            degraded_view=degraded_view,
            limits=limits,
            derive_query=derive_query,
            gap_marker=gap_marker,
            recursion_markers=recursion_markers,
            content_hint=content_hint,
        )
        base = compress_context(caller_text, query_as_given, strategy="extract", **ladder_kwargs)
        if not base.fail_open:
            return base
        return compress_context(
            caller_text,
            query_as_given,
            strategy="summarise",
            summary_endpoint=summary_endpoint,
            summary_models=summary_models,
            summary_timeout=summary_timeout,
            summary_allow_remote=summary_allow_remote,
            **ladder_kwargs,
        )

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
            result_text = tool_meta_line + summary + ccr_marker
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
                reasons=(
                    ["llm summary", "query facts verified"]
                    + (["ccr skipped: secret material detected"] if ccr and secret_in_input else [])
                ),
                verifier=summary_verifier,
                receipt={
                    "schema_version": "1",
                    "engine": "tameru",
                    "policy": "summarise-llm",
                    "query_hash": hashlib.sha256(
                        (query_as_given or "").encode("utf-8")
                    ).hexdigest()[:12],
                    "savings_pct": round((1.0 - keep_ratio) * 100.0, 2),
                    "risk": summary_risk,
                    "content_hint": content_hint,
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

    if query_has_distinctive_selectors(query or "") and not adapter_info:
        lines, content_type = preprocess(text, query or "", content_hint=route_hint)
        if industrial.applied:
            content_type = industrial.profile.format
    else:
        # Do not fingerprint-collapse logs or crush JSON on a generic query.
        # (Perf, v1.4.0: preprocess() output was computed and then discarded
        # on this path — ~0.4 s of log collapsing per 650 KB — so it is not
        # run at all; the result is identical.)
        lines = text.split("\n")
        content_type = _HINT_ROUTES.get(route_hint) or route_content_type(lines)
    # numbered_code: the "N\t" / "N|" gutter stays in the output (_render emits
    # `lines`) but is not part of the code the scorer should read.
    score_lines = lines
    if route_hint == "numbered_code":
        score_lines = [_GUTTER_RE.sub("", ln, count=1) for ln in lines]
    blocks = segment_blocks(score_lines)
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
                (query_as_given or "").encode("utf-8")
            ).hexdigest()[:12],
            "savings_pct": 0.0,
            "risk": "high",
            "content_hint": content_hint,
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
    if degraded:
        # Score a bounded head+tail view per block. _render emits exact
        # spans from `lines`, and citation digests use `full_text`, so the
        # judge sees the shrink while output stays byte-exact.
        for b in blocks:
            full = b["text"]
            if len(full) > _DEGRADED_VIEW_HEAD + _DEGRADED_VIEW_TAIL + 64:
                b["full_text"] = full
                omitted = len(full) - _DEGRADED_VIEW_HEAD - _DEGRADED_VIEW_TAIL
                b["text"] = (
                    f"{full[:_DEGRADED_VIEW_HEAD]}\n"
                    f"[… {omitted} chars omitted …]\n"
                    f"{full[-_DEGRADED_VIEW_TAIL:]}"
                )
    scored = score_blocks(blocks, query or "", content_type=content_type)

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

    # JEV-style positional pinning: the opening block and the newest N
    # blocks are unconditional keeps — the caller's working-context floor.
    # Same trust semantics as pin_patterns: pins outrank drop and risk paths.
    pin_recent_ids: set[int] = set()
    if pin_recent and scored:
        pin_recent_ids = {scored[0]["id"]}
        pin_recent_ids.update(b["id"] for b in scored[-pin_recent:])
        for b in scored:
            if b["id"] in pin_recent_ids:
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
    selection: dict[str, Any] = {}
    if mode_norm == "fixed":
        ratio = 0.35 if budget_ratio is None else float(budget_ratio)
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"budget_ratio must be in (0, 1], got {ratio!r}")
        kept, fail_open, risk = select_fixed(scored, ratio, out=selection)
        if gap_marker is not None and not citations and not fail_open:
            # Marker text costs tokens too: reserve them out of the budget and
            # select once more (gap count may shift slightly; one pass is
            # enough to keep the rendered output near the requested ratio).
            reserve = _gap_marker_reserve(
                lines, scored, kept, gap_marker, len(caller_text), content_lines=score_lines
            )
            if reserve:
                kept, fail_open, risk = select_fixed(
                    scored, ratio, out=selection, reserve_tokens=reserve
                )
    else:
        kept, fail_open, risk = select_adaptive(
            scored,
            needle_only=query_has_distinctive_selectors(query or ""),
            query=query or "",
            semantic_tier=resolve_tier(semantic_tier) if semantic_tier is not None else None,
            out=selection,
        )
        ratio = budget_ratio if budget_ratio is not None else 0.0
    if not fail_open:
        if decision_cache is not None:
            kept = _enforce_frozen_decisions(scored, kept)
        # Temporal supersession: a later kept block that explicitly marks an
        # earlier kept block stale (now/obsolete/override/newer date) prunes
        # it after cache replay, so a frozen keep cannot revive stale data.
        # v1.4.0: not for content the caller says is source code. The markers
        # are prose ("override", "deprecated", " disabled" are ordinary code
        # words) and two functions share identifiers by construction, so a
        # comment in a LATER function would evict the very function asked
        # about — and the hint exists so function bodies survive.
        if not code_hint:
            kept = apply_supersession(scored, kept)
        # v1.4.0 (E3) safety net: neither a frozen drop nor supersession may
        # evict the one exemplar block of a level-tagged log error.
        kept |= {
            b["id"]
            for b in scored
            if b.get("error_exemplar") and not b.get("trust_risk")
        }
        if pin_recent_ids:
            # Pins are never touched: a score-999 boost alone cannot reach
            # kept when a selector path ignores raw scores (needle-only),
            # and neither freeze nor supersession may evict a pin.
            kept |= pin_recent_ids
    if adapter_info is not None and scored and not ambiguity_fail_open:
        # v1.4.0 (E5): the format adapter already chose what to keep (every
        # header, whole hunks, grouped matches); block scoring must not drop a
        # file header or half a hunk. Risk reflects how the adapter chose.
        kept = {b["id"] for b in scored}
        fail_open = False
        risk = "low" if query_has_distinctive_selectors(query or "") else "medium"
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
    if (
        not query_has_distinctive_selectors(q)
        and not no_table
        and _looks_like_csv(original_text)
    ):
        fail_open = True
        risk = "high"

    collapsed = "\n".join(lines)
    structural_collapse_only = False
    if fail_open:
        annotated_ids = {
            b["id"] for b in scored if b.get("trust_risk") and not b.get("pinned")
        }
        if annotated_ids:
            # Restore ordinary data when uncertain, but never undo a block
            # annotation. Explicitly pinned blocks remain eligible.
            kept = {b["id"] for b in scored if b["id"] not in annotated_ids}
            compressed = _render(
                lines, scored, kept, citations=citations, reorder_best=_reorder,
                gap_marker=gap_marker, total_chars=len(caller_text),
                content_lines=score_lines,
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
            # v1.4.0 (E7): the query never selected anything here; what is
            # returned is the structural collapse alone (repeat folding,
            # frame runs), not an answer-driven extract. Say so.
            structural_collapse_only = True
        else:
            compressed = original_text
            kept = {b["id"] for b in scored}
    else:
        compressed = _render(
            lines, scored, kept, citations=citations, reorder_best=_reorder,
            gap_marker=gap_marker, total_chars=len(caller_text),
            content_lines=score_lines,
        )

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
        compressed = _render(
            lines, scored, kept, citations=citations, reorder_best=_reorder,
            gap_marker=gap_marker, total_chars=len(caller_text),
            content_lines=score_lines,
        )
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
            compressed = _render(
                lines, scored, kept, citations=citations, reorder_best=_reorder,
                gap_marker=gap_marker, total_chars=len(caller_text),
                content_lines=score_lines,
            )
            risk = "high"
        elif recall < 0.5:
            compressed = original_text
            fail_open = True
            risk = "high"
            kept = {b["id"] for b in scored}

    if fail_open:
        risk = "high"
        compressed = caller_text

    # v1.4.0 (E3): for content routed as a log, report how many level-tagged
    # error fingerprints survive. The engine keeps one exemplar per
    # fingerprint, so a loss means something upstream broke the invariant:
    # say so in the risk instead of leaving the verifier blind to it.
    log_error_retention: dict[str, int] | None = None
    if content_type == "log" and not fail_open and compressed:
        want = {
            fp for fp, ex in error_fingerprints(text).items() if _is_level_error_line(ex)
        }
        if want:
            have = set(error_fingerprints(compressed))
            log_error_retention = {"total": len(want), "kept": len(want & have)}
            if log_error_retention["kept"] < log_error_retention["total"]:
                lost_all = log_error_retention["kept"] == 0
                risk = "high" if lost_all else (risk if risk == "high" else "medium")

    # v1.4.0 (G9): a Hermes tool wrapper ({"output": ..., "exit_code": 1, ...})
    # was reduced to its inner text above; re-emit its scalar metadata as one
    # compact JSON line so exit codes, errors and line counts survive. Only for
    # a compressed (not fail-open) result — fail-open returns the caller's
    # wrapper byte for byte.
    if tool_meta_line and not fail_open:
        compressed = tool_meta_line + compressed

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
        if savings_ratio < min_savings_ratio and not trust_filtered:
            compressed = original_text
            kept = {b["id"] for b in scored}
            fail_open = True
            risk = "high"
            savings_gate = True
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
    if adapter_info is not None and not fail_open:
        reasons.append(f"format adapter: {adapter_info['name']}")
    if freeze_cache_saturated:
        reasons.append("freeze cache capacity reached")
    if savings_gate:
        reasons.append("savings below min_savings_ratio")
    if structural_collapse_only and not fail_open:
        reasons.append("structural collapse only (query terms absent)")
    if log_error_retention and log_error_retention["kept"] < log_error_retention["total"]:
        reasons.append(
            f"log error fingerprints lost: {log_error_retention['kept']}"
            f"/{log_error_retention['total']} kept"
        )
    if degraded:
        reasons.append("degraded scoring view")
        if risk == "low":
            risk = "medium"
    if ccr and secret_in_input:
        # dsh-jev-pre-compaction lesson: scan for secrets BEFORE archiving.
        reasons.append("ccr skipped: secret material detected")
    result_text = compressed
    cache_applied = False
    if cache_prefix and not fail_open:
        result_text = cache_wrap(compressed, query_as_given or "")
        cache_applied = True

    ccr_info = None
    pending_ccr_hash = None
    candidate_text = result_text
    if ccr and not fail_open and compressed != original_text and not secret_in_input:
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
            if ccr_info is None:
                # Store refused (secret screen) or failed — never emit a
                # dangling retrieval pointer.
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
        if route_hint == "numbered_code":
            # the gutter ("12: ") is not content: it must not read as key: value
            verifier = verify_compression(
                "\n".join(_GUTTER_RE.sub("", ln, count=1) for ln in text.split("\n")),
                "\n".join(_GUTTER_RE.sub("", ln, count=1) for ln in compressed.split("\n")),
                query or "",
            )
        else:
            verifier = verify_compression(text, compressed, query or "")
        risk_order = {"low": 0, "medium": 1, "high": 2}
        verifier_risk = str(verifier.get("risk", "high"))
        if risk_order.get(verifier_risk, 2) > risk_order.get(risk, 2):
            risk = verifier_risk
        recall = min(recall, float(verifier.get("score", 0.0)))
    # An inferred objective is advisory context, not caller intent — a
    # derived query never earns a "low" compression-risk rating.
    if derived_terms and risk == "low":
        risk = "medium"
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
    query_hash = hashlib.sha256((query_as_given or "").encode("utf-8")).hexdigest()[:12]
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
        "degraded_view": degraded,
        # dsh-jev-prune lesson: never degrade silently — report which
        # selector path decided the keep-set.
        "selection": selection.get("path"),
        # lcc sufficiency lesson: blocks restored because a kept block's
        # rare terms were qualified or defined there.
        "sufficiency_restored": selection.get("sufficiency_restored") or [],
        # TPC lesson: callers must see when the objective was inferred,
        # not supplied — a derived query is advisory context, not intent.
        "query_source": "derived" if derived_terms else "caller",
        "derived_terms": derived_terms,
        # v1.4.0: E7 — the keep-set came from structural collapse alone.
        "structural_collapse_only": bool(structural_collapse_only and not fail_open),
        # v1.4.0: E8 — the caller's routing hint, always recorded.
        "content_hint": content_hint,
        # v1.4.0: E5 — which diff/grep adapter shaped the text (None: none did).
        "adapter": adapter_info,
        # v1.4.0: E3 — level-tagged error fingerprints kept/total (logs only).
        "log_error_fingerprints": log_error_retention,
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
    p.add_argument("--strategy", default="extract", choices=["clear", "extract", "summarise", "auto"],
                   help="Compression strategy: clear (drop tool payloads), extract (default, query-aware), summarise (falls back to extract), auto (extract first, escalate to summarise only if extraction fails open)")
    p.add_argument("--pin-recent", type=int, default=0,
                   help="Unconditionally keep the first block and the newest N blocks")
    p.add_argument("--min-savings-ratio", type=float, default=0.10,
                   help="Fail open when achieved savings fall below this ratio (0 disables)")
    p.add_argument("--degraded-view", action="store_true",
                   help="On input size-limit breach, score a bounded head+tail view per block instead of failing open")
    p.add_argument("--derive-query", action="store_true",
                   help="On empty/generic query, infer a conservative objective from the document's recurring rare terms instead of failing open")
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
        pin_recent=args.pin_recent,
        min_savings_ratio=args.min_savings_ratio,
        degraded_view=args.degraded_view,
        derive_query=args.derive_query,
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
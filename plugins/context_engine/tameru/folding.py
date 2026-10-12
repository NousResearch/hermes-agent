"""Lossless folding, log-template masking and JSON briefs (v1.4.0).

Three small, stdlib-only, deterministic helpers that ``compress_context`` wires
in:

* :func:`fold_lossless` (E9) - a transformation that loses no information a
  reader could see: terminal redraws, ANSI colour, trailing blanks, runs of
  identical lines and pretty-printed JSON.
* :func:`mask_template` (E10) - the ONE masking function behind both the log
  template fingerprint and the error fingerprint, so the two never disagree.
* :func:`json_brief` (E11) - a valid-JSON structural brief of a big array of
  objects.

Every regex here is linear: bounded character classes, no nested quantifiers.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

__all__ = ["fold_lossless", "mask_template", "json_brief"]

# --------------------------------------------------------------------------
# E9: fold_lossless
# --------------------------------------------------------------------------

_ANSI_RE = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]"  # CSI (colours, cursor moves, erase)
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC (titles, hyperlinks)
    r"|\x1b[@-Z\\-_]"  # two-byte escapes
)
_RUN_MIN = 3
_ANSI_PASSES = 4
# The marker this module writes. A literal marker-looking line is never run-
# collapsed: otherwise ``a a a a`` (-> ``a`` + marker) followed by two literal
# marker lines would fold again on the next pass (not idempotent).
_RUN_MARKER_RE = re.compile(r"\[×\d+ identical lines\]")


def _strip_ansi_line(ln: str) -> tuple[str, int]:
    """Strip ANSI sequences to a fixed point (idempotence).

    Removing one escape can splice the halves of another into a new one
    (``ESC[3 ESC[0m 1m`` -> ``ESC[31m``), so a single pass is not idempotent.
    Passes are capped (adversarial nesting would be quadratic); if the cap is
    hit, the remaining stray ESC bytes are dropped, which cannot create a new
    sequence since every sequence starts with ESC.
    """
    total = 0
    for _ in range(_ANSI_PASSES):
        ln, k = _ANSI_RE.subn("", ln)
        total += k
        if not k:
            return ln, total
    if _ANSI_RE.search(ln):
        ln = ln.replace("\x1b", "")
    return ln, total


def _fold_text_steps(text: str, stats: dict) -> list[str]:
    """CR redraws, ANSI, trailing whitespace -> list of lines."""
    lines = text.split("\n")
    cr = 0
    for i, ln in enumerate(lines):
        if "\r" in ln:
            # Trailing \r's belong to the line ending (\r\n, or a bare \r
            # that redraws nothing): they must not erase the line's text
            # ("abc\r\r\n" shows "abc").
            body = ln.rstrip("\r")
            if "\r" in body:
                body = body.rsplit("\r", 1)[1]
                cr += 1
            lines[i] = body
    stats["cr_redraws"] = cr
    if "\x1b" in text:
        ansi = 0
        for i, ln in enumerate(lines):
            if "\x1b" in ln:
                lines[i], k = _strip_ansi_line(ln)
                ansi += k
        stats["ansi"] = ansi
    ws = 0
    for i, ln in enumerate(lines):
        stripped = ln.rstrip()
        if stripped != ln:
            lines[i] = stripped
            ws += 1
    stats["trailing_ws"] = ws
    return lines


def _fold_runs(lines: list[str], stats: dict) -> list[str]:
    """Collapse >=3 identical consecutive lines when that is strictly shorter."""
    out: list[str] = []
    n = len(lines)
    i = 0
    collapsed = 0
    while i < n:
        j = i + 1
        while j < n and lines[j] == lines[i]:
            j += 1
        run = j - i
        if run >= _RUN_MIN and not _RUN_MARKER_RE.fullmatch(lines[i]):
            marker = f"[×{run - 1} identical lines]"
            if len(lines[i]) + 1 + len(marker) < run * len(lines[i]) + run - 1:
                out.append(lines[i])
                out.append(marker)
                collapsed += 1
                i = j
                continue
        out.extend(lines[i:j])
        i = j
    stats["runs_collapsed"] = collapsed
    return out


_JSON_STRING_OR_WS_RE = re.compile(r'("[^"\\]*(?:\\.[^"\\]*)*")|[ \t\r\n]+')


def _json_minified(text: str) -> str | None:
    """``text`` with the insignificant whitespace removed, when it is JSON and
    that is strictly shorter.

    Only whitespace outside string literals is removed, so every number
    literal, string escape, key order and duplicate key stays byte for byte
    (re-serialising would turn ``"\\ud800"`` into a lone surrogate, ``"\\u202e"``
    into a raw bidi override, ``1.10`` into ``1.1`` and ``-0`` into ``0``).
    """
    head = text.lstrip()[:1]
    if head not in ("{", "["):
        return None
    try:
        # a number that overflows to inf (1e400) is not minified, as before
        json.loads(text, parse_float=_finite_float)
    except (ValueError, RecursionError):
        return None
    mini = _JSON_STRING_OR_WS_RE.sub(lambda m: m.group(1) or "", text)
    return mini if len(mini) < len(text) else None


def _finite_float(literal: str) -> float:
    value = float(literal)
    if value in (float("inf"), float("-inf")):
        raise ValueError("non-finite number")
    return value


def fold_lossless(text: str) -> tuple[str, dict]:
    """Fold ``text`` without losing information; return ``(folded, stats)``.

    1. per line, keep only what follows the last ``\\r`` that is not part of a
       ``\\r\\n`` (a terminal shows nothing else);
    2. strip ANSI escape sequences;
    3. strip trailing whitespace of each line;
    4. a run of >=3 identical consecutive lines becomes the first line plus
       ``[×{n-1} identical lines]`` (only when that is shorter);
    5. JSON that parses and re-serialises shorter is minified, accepted only
       if the parsed values are equal to the original's.

    Idempotent, deterministic and never longer than the input. ``stats`` has
    ``cr_redraws``, ``ansi``, ``trailing_ws``, ``runs_collapsed``,
    ``json_minified`` and ``saved_chars``.
    """
    text = text if isinstance(text, str) else str(text or "")
    stats = {
        "cr_redraws": 0,
        "ansi": 0,
        "trailing_ws": 0,
        "runs_collapsed": 0,
        "json_minified": False,
        "saved_chars": 0,
    }
    if not text:
        return text, stats
    lines = _fold_text_steps(text, stats)
    folded = "\n".join(lines)
    mini = _json_minified(folded)
    if mini is not None:
        stats["json_minified"] = True
        folded = mini
    else:
        folded = "\n".join(_fold_runs(lines, stats))
    if len(folded) > len(text):  # cannot happen; keep the invariant explicit
        return text, {**stats, "saved_chars": 0}
    stats["saved_chars"] = len(text) - len(folded)
    return folded, stats


# --------------------------------------------------------------------------
# E10: template masking (shared by _log_fingerprint and _error_fp)
# --------------------------------------------------------------------------

_M_DQ = re.compile(r'"[^"\n]{0,200}"')
_M_SQ = re.compile(r"(?<![\w'])'[^'\n]{0,200}'(?![\w'])")
_M_TS = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?:[t ]\d{2}:\d{2}:\d{2}(?:[.,]\d{1,9})?(?:z|[+-]\d{2}:?\d{2})?)?"
    r"|\b\d{1,2}:\d{2}:\d{2}(?:[.,]\d{1,9})?"
)
_M_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_M_IP4 = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d{1,5})?\b")
_M_IP6 = re.compile(r"(?<![\w:])(?:[0-9a-f]{1,4}:){2,7}[0-9a-f]{1,4}(?![\w:])")
_M_PATH = re.compile(r"(?<![\w<>/.:=-])(?:/[\w.@+~%,=-]{1,128}){3,}/?")
_M_DUR = re.compile(r"\b\d+(?:\.\d+)?(?:ms|us|µs|s|m|h)\b")
_M_HEX = re.compile(r"[0-9a-f]{6,}")
_M_DIGITS = re.compile(r"\d+")


def _volatile(placeholder: str):
    """A substitution that masks a quoted value or path only when it holds a digit (an id, a temp dir,
    a port): ``table "orders"`` and ``table "payments"``, or ``/srv/data/a.db`` and ``/srv/data/b.db``,
    are different messages, not one template."""
    def sub(m: re.Match) -> str:
        return placeholder if any(ch.isdigit() for ch in m.group(0)) else m.group(0)
    return sub


def mask_template(s: str) -> str:
    """Mask the volatile fields of one (already lowercased) log line.

    Quoted strings holding a digit -> ``"<str>"``, ISO-8601 timestamps and ``HH:MM:SS(.fff)``
    -> ``<ts>``, UUIDs -> ``<uuid>``, IPv4/IPv6 -> ``<ip>``, paths of three or
    more segments holding a digit -> ``<path>``, durations -> ``<dur>``, hex runs of 6+ ->
    ``<hex>``, then digits -> ``#`` and whitespace collapsed. Callers apply
    their own length cap.
    """
    s = _M_DQ.sub(_volatile('"<str>"'), s)
    s = _M_SQ.sub(_volatile('"<str>"'), s)
    s = _M_TS.sub("<ts>", s)
    s = _M_UUID.sub("<uuid>", s)
    s = _M_IP4.sub("<ip>", s)
    s = _M_IP6.sub("<ip>", s)
    s = _M_PATH.sub(_volatile("<path>"), s)
    s = _M_DUR.sub("<dur>", s)
    s = _M_HEX.sub("<hex>", s)
    s = _M_DIGITS.sub("#", s)
    return " ".join(s.split())


# --------------------------------------------------------------------------
# E11: json_brief
# --------------------------------------------------------------------------

_MAX_KEYS = 20
_MAX_NOTABLE = 5
_LOWCARD_MAX = 10


# ensure_ascii=False writes these raw: lone surrogates (a str that cannot be
# encoded as UTF-8), bidi/format controls (a brief must not turn the escaped
# "\\u202e" of its input into a live override) and line separators. They are
# escaped back, which is always valid inside a JSON string.
_UNSAFE_RAW_RE = re.compile(
    "[\x7f-\x9f\u00ad\u061c\u180e\u200b-\u200f\u2028-\u202e\u2060-\u206f"
    "\ufeff\ufff9-\ufffb\ud800-\udfff\U000e0000-\U000e007f]"
)


def _dumps(obj: Any) -> str:
    # allow_nan=False: NaN/Infinity are not valid JSON, and the brief must be.
    out = json.dumps(obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    if out.isascii():
        return out
    return _UNSAFE_RAW_RE.sub(_escape_char, out)


def _escape_char(match: "re.Match[str]") -> str:
    code = ord(match.group())
    if code > 0xFFFF:  # JSON escapes astral characters as a surrogate pair
        code -= 0x10000
        return "\\u%04x\\u%04x" % (0xD800 + (code >> 10), 0xDC00 + (code & 0x3FF))
    return "\\u%04x" % code


def _find_rows(obj: Any) -> tuple[list[dict], str | None, dict] | None:
    """(rows, wrapper_key, scalar_siblings) for an applicable JSON value."""
    if isinstance(obj, list):
        if obj and all(isinstance(x, dict) for x in obj):
            return obj, None, {}
        return None
    if isinstance(obj, dict):
        best_key = None
        best_len = -1
        for k, v in obj.items():
            if isinstance(v, (list, dict)):
                ln = len(_dumps(v))
                if ln > best_len:
                    best_key, best_len = k, ln
        if best_key is None:
            return None
        v = obj[best_key]
        if (
            isinstance(v, list)
            and len(v) >= 10
            and all(isinstance(x, dict) for x in v)
        ):
            scalars = {
                k: x
                for k, x in obj.items()
                if k != best_key and (x is None or isinstance(x, (str, int, float, bool)))
            }
            return v, best_key, scalars
    return None


def _query_terms(query: str) -> list[str]:
    if not query:
        return []
    from .compress_context import _query_hit_terms  # lazy: avoids an import cycle

    return _query_hit_terms(query)


def _notable_order(rows: list[dict], dumped: list[str], terms: list[str]) -> list[int]:
    """Indices of notable rows, best first: query hits, then rare field values."""
    n = len(rows)
    ranked: list[tuple[int, int, int]] = []  # (class, rarity, index)
    seen: set[int] = set()
    if terms:
        from .compress_context import _term_in_text  # lazy, see above

        term_rows: list[tuple[int, list[str]]] = []
        for i, d in enumerate(dumped):
            low = d[:4000].casefold()
            hit_terms = [t for t in terms if _term_in_text(t, low)]
            term_rows.append((i, hit_terms))
        freq = Counter(t for _, ht in term_rows for t in ht)
        for i, ht in term_rows:
            # a term present in over half the rows (a key name such as
            # "status") says nothing about which row matters
            hits = sum(1 for t in ht if freq[t] * 2 <= n)
            if hits:
                ranked.append((0, -hits, i))
                seen.add(i)
    # Low-cardinality scalar fields: rows whose value is not the field's mode.
    counts: dict[str, Counter] = {}
    for row in rows:
        for k, v in row.items():
            if v is None or isinstance(v, (str, int, float, bool)):
                c = counts.get(k)
                if c is None:
                    c = counts[k] = Counter()
                if len(c) <= _LOWCARD_MAX:
                    c[(type(v).__name__, v)] += 1
    lowcard = {
        k: c
        for k, c in counts.items()
        if 2 <= len(c) <= _LOWCARD_MAX and sum(c.values()) >= n * 0.9
    }
    modes = {k: c.most_common(1)[0][0] for k, c in lowcard.items()}
    for i, row in enumerate(rows):
        if i in seen:
            continue
        rarest = None
        for k, c in lowcard.items():
            if k in row:
                v = row[k]
                key = (type(v).__name__, v)
                if key != modes[k] and key in c:
                    rarest = c[key] if rarest is None else min(rarest, c[key])
        if rarest is not None:
            ranked.append((1, rarest, i))
    ranked.sort()
    return [i for _, _, i in ranked]


def json_brief(text: str, *, budget_chars: int, query: str = "") -> str | None:
    """See :func:`_json_brief`; returns None for non-finite numbers too."""
    try:
        return _json_brief(text, budget_chars=budget_chars, query=query)
    except ValueError:  # NaN/Infinity cannot be written as valid JSON
        return None


def _json_brief(text: str, *, budget_chars: int, query: str = "") -> str | None:
    """Valid-JSON structural brief of a JSON array of objects (v1.4.0).

    Applies to a JSON array of objects, and to an object whose largest value
    is an array of >=10 objects. Output::

        {"_count": N, "_keys": [first 20 keys], "_head": [first K items],
         "_tail": [last K items], "_notable": [<=5 items], "_omitted": M}

    ``_notable`` holds items (not already in head/tail) that contain a query
    term or whose low-cardinality field (e.g. ``status``) differs from that
    field's mode, rarest first. ``_omitted`` counts the items shown nowhere.
    For the object case the brief sits under the array's key beside the
    object's other scalar keys. K is the largest value (>=1) that fits
    ``budget_chars``. Returns ``None`` when the text is not applicable or even
    K=1 does not fit.
    """
    if not isinstance(text, str) or text.lstrip()[:1] not in ("{", "["):
        return None
    try:
        obj = json.loads(text)
    except (ValueError, RecursionError):
        return None
    found = _find_rows(obj)
    if found is None:
        return None
    rows, wrap_key, scalars = found
    n = len(rows)
    budget = int(budget_chars)

    prefix = suffix = ""
    if wrap_key is not None:
        prefix = "{" + _dumps(wrap_key) + ":"
        suffix = "".join("," + _dumps(k) + ":" + _dumps(v) for k, v in scalars.items()) + "}"
    inner_budget = budget - len(prefix) - len(suffix)
    if inner_budget <= 0:
        return None

    keys: list[str] = []
    seen_keys: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen_keys:
                seen_keys.add(k)
                if len(keys) < _MAX_KEYS:
                    keys.append(k)
    dumped = [_dumps(r) for r in rows]
    # prefix sums so a candidate K costs O(1) to size
    pre = [0] * (n + 1)
    for i, d in enumerate(dumped):
        pre[i + 1] = pre[i] + len(d) + 1
    head_overhead = len('{"_count":%d,"_keys":%s,"_head":[],"_tail":[],"_notable":[],"_omitted":%d}' % (n, _dumps(keys), n))

    def parts(k: int) -> tuple[list[int], list[int]]:
        head = list(range(min(k, n)))
        tail = list(range(max(len(head), n - k), n))
        return head, tail

    def size(head: list[int], tail: list[int], notable: list[int]) -> int:
        tot = head_overhead
        for grp in (head, tail, notable):
            tot += sum(len(dumped[i]) + 1 for i in grp)
        return tot

    notable_all = [i for i in _notable_order(rows, dumped, _query_terms(query))]

    def build(k: int, notable_cap: int) -> tuple[str, int] | None:
        head, tail = parts(k)
        shown = set(head) | set(tail)
        notable = [i for i in notable_all if i not in shown][:notable_cap]
        if size(head, tail, notable) > inner_budget + 3 * len(notable) + 4:
            return None
        omitted = n - len(shown) - len(notable)
        body = (
            '{"_count":%d,"_keys":%s,"_head":[%s],"_tail":[%s],"_notable":[%s],"_omitted":%d}'
            % (
                n,
                _dumps(keys),
                ",".join(dumped[i] for i in head),
                ",".join(dumped[i] for i in tail),
                ",".join(dumped[i] for i in sorted(notable)),
                omitted,
            )
        )
        return (body, len(body)) if len(body) <= inner_budget else None

    for cap in (_MAX_NOTABLE, 2, 0):
        # Largest K that fits (size grows with K): binary search, K in [1, n].
        lo, hi, best = 1, max(1, n // 2 + 1), None
        if build(1, cap) is None:
            continue
        while lo <= hi:
            mid = (lo + hi) // 2
            r = build(mid, cap)
            if r is not None:
                best, lo = r, mid + 1
            else:
                hi = mid - 1
        if best is not None:
            return prefix + best[0] + suffix
    return None

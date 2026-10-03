"""Grammar-aware masking for credential-store content (``.netrc`` / ``_netrc``, ``.pgpass``, INI
stores such as ``.pypirc`` / ``~/.aws/credentials`` / ``~/.aws/config`` / ``.npmrc``,
``.git-credentials``).

A stored credential is owned by its file grammar, not by a regex token: a netrc password is a
lexer token (quoted, backslash-escaped, or on the line after ``password``) and an INI password
runs to end of line plus any more-indented continuation lines. The text reaching the redactor
is often a SLICE of the file (a read_file page, one search_files match, ``tail`` output), so a
value can arrive without the keyword that introduces it. Unless the text is known to start at
the top of the file, every parser here therefore starts in an unknown state and masks whatever
it cannot prove public: an orphan netrc token, a leading indented INI line, a pgpass line
without its four host fields.

That catch-all is only sound while the text is nothing BUT store content. Terminal output can
also carry other files (``cat ~/.netrc app.py``, ``tail -n 1 ~/.netrc && cat app.py``,
``grep -rn X ~/.netrc src``), whose lines look exactly like orphan values. For such output
(``extent="mixed"``) each line's owner is resolved instead of guessed (``_mixed_spans``):

* a grep line whose ``path:N:`` prefix names the store is store content and gets the grammar;
  a line naming another file is never parsed by it;
* for lines with no owner, the store as read on the control host supplies its actual values,
  which are masked wherever they appear, and nothing else is touched;
* when the store cannot be read on the control host, or none of its values appear in those
  lines (a remote backend's file may differ from the host's), they fail closed: the catch-all
  applies and may mask other output rather than risk a stored value.
"""

import hashlib
import re
import threading
from dataclasses import dataclass
from typing import Callable, Iterable, Literal

Mask = Callable[[str], str]
# Is this INI option name a secret? Supplied by agent/redact.py so this pass applies the same
# word-bounded key policy as the generic assignment passes it runs before.
SecretKey = Callable[[str], bool]
# "start": the text begins at the top of the store; "slice": a contiguous part of the store
# that may begin anywhere; "mixed": store content alongside other output.
Extent = Literal["start", "slice", "mixed"]
Span = tuple[int, int, str]  # (start, end, replacement) in the original text
Lines = list[tuple[int, int]]  # (start, end) of each line body in the original text


@dataclass(frozen=True)
class StoreSource:
    """A credential store the text was read from."""

    fmt: str
    # Trailing path parts that name the store in a grep prefix: (".netrc",), (".aws", "config").
    name: tuple[str, ...]
    # The store as read on the control host; only consulted for mixed output. None when it
    # could not be read there.
    content: str | None = None


# ``cat -n`` / read_file gutter: only trusted when EVERY line carries one and the numbers are
# consecutive, because a lone digit prefix on a netrc line may be password bytes.
_GUTTER_RE = re.compile(r"[ \t]*(\d+)[|\t]")
# Any rendered line-number prefix (read_file ``5|``, grep ``6:`` / ``7-``, ``cat -n``). Used by
# the line-atomic grammars as a SECOND reading of each line, unioned with the raw reading.
_LOOSE_GUTTER_RE = re.compile(r"[ \t]*\d+(?:[|:\-]|\t)")
# A multi-file grep line: ``path:N:``, a ``-C`` context line ``path-N-``, or ``path:`` without -n.
_GREP_LINE_RE = re.compile(r"(?P<path>(?:[A-Za-z]:)?[^:\n]*?)(?:(?P<sep>[:\-])\d+(?P=sep)|:)")
_PATH_LIKE_RE = re.compile(r"[\\/.]")
_URL_USERINFO_PASSWORD_RE = re.compile(r"://[^:/\s@]*:([^@\s]+)@")
# Learned values are masked wherever they appear: long enough not to hit arbitrary short words,
# and never a boolean-ish option value (``always-auth = true``).
_MIN_LEARNED_VALUE = 4
_NOT_SECRET_VALUES = frozenset({"true", "false", "yes", "no", "on", "off", "none", "null"})

_NETRC_WS = " \t\r\n"
_NETRC_KEYWORDS = frozenset({"machine", "default", "login", "user", "account", "password", "macdef"})
_NETRC_PUBLIC_FOLLOWERS = frozenset({"machine", "login", "user", "macdef"})
_NETRC_SECRET_FOLLOWERS = frozenset({"password", "account"})


def _line_spans(text: str) -> Lines:
    spans, pos = [], 0
    for line in text.split("\n"):
        spans.append((pos, pos + len(line)))
        pos += len(line) + 1
    return spans


def _trusted_gutter(text: str, lines: Lines) -> tuple[Lines, int] | None:
    """``lines`` past a trusted gutter plus its first line number, or None when there is none."""
    checked = lines[:-1] if len(lines) > 1 and lines[-1][0] == lines[-1][1] else lines
    gutters = [_GUTTER_RE.match(text, start, end) for start, end in checked]
    if not gutters or not all(gutters):
        return None
    first = int(gutters[0].group(1))
    if any(int(g.group(1)) != first + i for i, g in enumerate(gutters)):
        return None
    return [(g.end(), end) for g, (_, end) in zip(gutters, checked)] + lines[len(checked):], first


def _loose_gutter_bodies(text: str, lines: Lines) -> Lines:
    out = []
    for start, end in lines:
        g = _LOOSE_GUTTER_RE.match(text, start, end)
        out.append((g.end() if g else start, end))
    return out


class _NetrcLexer:
    """CPython ``netrc._netrclex`` over the line bodies, reporting each token's original spans."""

    def __init__(self, text: str, bodies: Lines):
        self.text = text
        self.bodies = bodies
        self.line = 0
        self.pos = bodies[0][0] if bodies else 0

    def _char(self) -> tuple[str, int]:
        """Next char and its original index (-1 for the virtual newline between bodies)."""
        while self.line < len(self.bodies):
            _, end = self.bodies[self.line]
            if self.pos < end:
                self.pos += 1
                return self.text[self.pos - 1], self.pos - 1
            self.line += 1
            if self.line < len(self.bodies):
                self.pos = self.bodies[self.line][0]
                return "\n", -1
        return "", -1

    def token(self) -> tuple[str, list[int], int] | None:
        """``(value, original char indices, starting line)`` or None at end of text."""
        ch, idx = self._char()
        while ch and ch in _NETRC_WS:
            ch, idx = self._char()
        if not ch:
            return None
        line, value, indices = self.line, "", [idx]
        if ch == '"':
            while True:
                ch, idx = self._char()
                indices.append(idx)
                if not ch or ch == '"':
                    return value, indices, line
                if ch == "\\":
                    ch, idx = self._char()
                    indices.append(idx)
                value += ch
        while True:
            if ch == "\\":
                ch, idx = self._char()
                indices.append(idx)
            value += ch
            ch, idx = self._char()
            if not ch or ch in _NETRC_WS:
                return value, indices, line
            indices.append(idx)

    def skip_rest_of_line(self) -> None:
        if self.line < len(self.bodies):
            self.pos = self.bodies[self.line][1]

    def at_blank_line(self) -> bool:
        start, end = self.bodies[self.line]
        return not self.text[start:end].strip("\r")

    def next_line(self) -> bool:
        self.line += 1
        if self.line >= len(self.bodies):
            return False
        self.pos = self.bodies[self.line][0]
        return True


def _index_runs(indices: Iterable[int]) -> Lines:
    """Contiguous original-index runs of a token (a quoted token may span lines)."""
    runs: list[list[int]] = []
    for i in indices:
        if i < 0:
            continue
        if runs and i == runs[-1][1]:
            runs[-1][1] = i + 1
        else:
            runs.append([i, i + 1])
    return [(a, b) for a, b in runs]


def _netrc_spans(text: str, mask: Mask, bodies: Lines, from_top: bool) -> list[Span]:
    """Mask password/account values and every token that is not provably a keyword or a
    machine/login/macdef name, since an orphan at the start of a slice is a value whose keyword
    was cut off. Comments and macro bodies stay intact."""
    lexer = _NetrcLexer(text, bodies)
    spans: list[Span] = []

    def _mask(value: str, indices: list[int]) -> None:
        for n, (a, b) in enumerate(_index_runs(indices)):
            spans.append((a, b, mask(value) if n == 0 else ""))

    # ``#...`` in keyword position is a comment to CPython and curl. Only while a slice may open
    # on a value line (``password`` cut off above a ``#hunter2`` line) is a glued ``#x`` first
    # token ambiguous; a bare ``#`` never is.
    in_context = from_top
    while (tok := lexer.token()) is not None:
        value, indices, line = tok
        raw_first = text[indices[0]] if indices[0] >= 0 else ""
        if raw_first == "#" and (value == "#" or in_context):
            if lexer.line == line:
                lexer.skip_rest_of_line()
            continue
        if value not in _NETRC_KEYWORDS:
            _mask(value, indices)
            continue
        in_context = True
        if value in _NETRC_SECRET_FOLLOWERS:
            if (follower := lexer.token()) is not None:
                _mask(follower[0], follower[1])
        elif value in _NETRC_PUBLIC_FOLLOWERS:
            lexer.token()
            if value == "macdef":  # the body runs to the next empty line
                lexer.skip_rest_of_line()
                while lexer.next_line() and not lexer.at_blank_line():
                    lexer.skip_rest_of_line()
    return spans


def _ini_spans(text: str, mask: Mask, secret_key: SecretKey, delimiters: str,
               bodies: Lines, from_top: bool) -> list[Span]:
    """``configparser`` option grammar: a secret option's value is the rest of its line plus
    every following line indented deeper than the option (blank and comment lines do not end
    it). A slice starts as if inside a secret value, so indented lines that open it are masked
    until a section or option line establishes context."""
    spans: list[Span] = []
    in_secret, cur_indent = not from_top, 0
    for start, end in bodies:
        body = text[start:end]
        stripped = body.strip(" \t\r")
        if not stripped or stripped[0] in "#;":
            continue
        indent = len(body) - len(body.lstrip(" \t"))
        lead = start + indent
        if indent > cur_indent and cur_indent >= 0:
            if in_secret:
                spans.append((lead, lead + len(stripped), mask(stripped)))
            continue
        cur_indent = -1  # no open option until one is parsed
        if stripped.startswith("[") and stripped.endswith("]"):
            continue
        cut = min((i for i in (stripped.find(d) for d in delimiters) if i >= 0), default=-1)
        if cut < 0:
            continue
        value = stripped[cut + 1:]
        value_start = lead + cut + 1 + (len(value) - len(value.lstrip(" \t")))
        value = value.strip(" \t")
        in_secret, cur_indent = secret_key(stripped[:cut].strip(" \t")), indent
        if in_secret and value:
            spans.append((value_start, value_start + len(value), mask(value)))
    return spans


def _pgpass_spans(text: str, mask: Mask, bodies: Lines) -> list[Span]:
    """``host:port:db:user:password`` with ``\\`` escapes; the password is the rest of the line.
    A non-comment line without four field separators is masked whole."""
    spans: list[Span] = []
    for start, end in bodies:
        body = text[start:end].rstrip("\r")
        stripped = body.lstrip(" \t")
        if not stripped or stripped.startswith("#"):
            continue
        seps, i = 0, 0
        while i < len(body) and seps < 4:
            if body[i] == "\\":
                i += 1
            elif body[i] == ":":
                seps += 1
            i += 1
        value_start = start + (i if seps == 4 else len(body) - len(stripped))
        value = text[value_start:start + len(body)]
        if value:
            spans.append((value_start, value_start + len(value), mask(value)))
    return spans


def _url_userinfo_spans(text: str, mask: Mask) -> list[Span]:
    return [(m.start(1), m.end(1), mask(m.group(1))) for m in _URL_USERINFO_PASSWORD_RE.finditer(text)]


def _format_spans(text: str, fmt: str, mask: Mask, secret_key: SecretKey, from_top: bool,
                  lines: Lines | None = None) -> list[Span]:
    """One grammar over ``lines`` (default: every line of ``text``)."""
    lines = _line_spans(text) if lines is None else lines
    trusted = _trusted_gutter(text, lines)
    from_top = from_top or (trusted is not None and trusted[1] == 1)
    if fmt == "netrc":
        return _netrc_spans(text, mask, trusted[0] if trusted else lines, from_top)
    # Line-atomic grammars: without a trusted gutter, read each line both raw and past any
    # line-number prefix and mask the union, so an unrecognized gutter (``grep -n``) can only
    # over-mask.
    readings = (trusted[0],) if trusted else (lines, _loose_gutter_bodies(text, lines))
    if fmt == "pgpass":
        return [s for bodies in readings for s in _pgpass_spans(text, mask, bodies)]
    delimiters = {"ini": "=:", "npmrc": "="}.get(fmt)
    if delimiters is None:  # git-credentials: URL userinfo only
        return []
    return [s for bodies in readings
            for s in _ini_spans(text, mask, secret_key, delimiters, bodies, from_top)]


_LEARNED_CACHE_MAX = 8
_LEARNED_CACHE: dict[tuple, frozenset[str] | None] = {}
_LEARNED_LOCK = threading.Lock()


def _learned_cache_key(fmt: str, secret_key: SecretKey, content: str) -> tuple:
    digest = hashlib.sha256(content.encode("utf-8", "surrogatepass")).digest()
    return (fmt, id(secret_key), digest)


def _stored_values(store: StoreSource, secret_key: SecretKey) -> set[str] | None:
    """Every value the store's grammar masks in its complete file, plus the words of multi-word
    values (``awk '{print $3}'`` can emit one word of ``password = alpha bravo``). None when a
    value is too short to mask wherever it appears, so the store cannot be handled by value.

    The grammar result is reused for identical content. A rewritten store has different bytes,
    so the next call learns the new values and forgets the old ones.
    """
    content = store.content or ""
    cache_key = _learned_cache_key(store.fmt, secret_key, content) if content else None
    if cache_key is not None:
        with _LEARNED_LOCK:
            if cache_key in _LEARNED_CACHE:
                cached = _LEARNED_CACHE.pop(cache_key)
                _LEARNED_CACHE[cache_key] = cached
                return None if cached is None else set(cached)
    found: list[str] = []

    def _capture(value: str) -> str:
        found.append(value)
        return ""

    spans = _format_spans(content, store.fmt, _capture, secret_key, True)
    spans += _url_userinfo_spans(content, _capture)
    candidates = set(found) | {content[a:b] for a, b, _ in spans}
    values = {c.strip() for c in candidates} - {""}
    if any(len(v) < _MIN_LEARNED_VALUE and v.lower() not in _NOT_SECRET_VALUES for v in values):
        learned: frozenset[str] | None = None
    else:
        values |= {word for value in values for word in value.split() if len(word) >= _MIN_LEARNED_VALUE}
        learned = frozenset(v for v in values if v.lower() not in _NOT_SECRET_VALUES)
    if cache_key is not None:
        with _LEARNED_LOCK:
            _LEARNED_CACHE.pop(cache_key, None)
            _LEARNED_CACHE[cache_key] = learned
            while len(_LEARNED_CACHE) > _LEARNED_CACHE_MAX:
                _LEARNED_CACHE.pop(next(iter(_LEARNED_CACHE)))
    return None if learned is None else set(learned)


def _value_spans(text: str, values: set[str], mask: Mask) -> list[Span]:
    if not values:
        return []
    alternation = "|".join(re.escape(v) for v in sorted(values, key=len, reverse=True))
    pattern = re.compile(rf"(?<![A-Za-z0-9_])(?:{alternation})(?![A-Za-z0-9_])")
    return [(m.start(), m.end(), mask(m.group(0))) for m in pattern.finditer(text)]


def _line_owner(text: str, start: int, end: int, stores: list[StoreSource]) -> tuple[int | None, int] | None:
    """``(store index, body start)`` for a grep line naming a store, ``(None, body start)`` for
    one naming another path, None for a line that carries no grep prefix."""
    m = _GREP_LINE_RE.match(text, start, end)
    if not m:
        return None
    parts = tuple(p.lower() for p in re.split(r"[\\/]", m.group("path")) if p)
    for k, store in enumerate(stores):
        if parts[-len(store.name):] == store.name:
            return k, m.end()
    return (None, m.end()) if _PATH_LIKE_RE.search(m.group("path")) else None


def _mixed_spans(text: str, stores: list[StoreSource], mask: Mask, secret_key: SecretKey) -> list[Span]:
    lines = _line_spans(text)
    owners = [_line_owner(text, a, b, stores) for a, b in lines]
    # Grep prefixes are only trusted once a line names a store: then the output is grep-shaped
    # and every other prefixed line belongs to another file. Otherwise ``foo.py:3:`` is content.
    grep_shaped = any(o is not None and o[0] is not None for o in owners)
    per_store: dict[int, Lines] = {}
    unowned: Lines = []
    for (start, end), owner in zip(lines, owners):
        if not grep_shaped or owner is None:
            unowned.append((start, end))
        elif owner[0] is not None:
            per_store.setdefault(owner[0], []).append((owner[1], end))

    spans: list[Span] = []
    for k, bodies in per_store.items():
        spans += _format_spans(text, stores[k].fmt, mask, secret_key, False, bodies)

    learned = {k: values for k, s in enumerate(stores)
               if s.content is not None and (values := _stored_values(s, secret_key)) is not None}
    for values in learned.values():
        spans += _value_spans(text, values, mask)
    unowned_text = "\n".join(text[a:b] for a, b in unowned)
    # The host copy stands for what the command read only if one of its values shows up (or it
    # has none). Otherwise a value line cut from its keyword is indistinguishable from another
    # file's line, so fail closed.
    blind = any(k not in learned or (learned[k] and not any(v in unowned_text for v in learned[k]))
                for k in range(len(stores)))
    if unowned and unowned_text.strip() and blind:
        for fmt in sorted({s.fmt for s in stores}):
            spans += _format_spans(text, fmt, mask, secret_key, False, unowned)
    return spans


def mask_credential_stores(text: str, stores: Iterable[StoreSource], mask: Mask, secret_key: SecretKey,
                           extent: Extent = "slice") -> str:
    """Mask every credential value ``text`` holds from ``stores`` (built by
    ``agent.redact._credential_store_source``). URL userinfo passwords are masked for every
    format — a ``.pypirc`` ``repository`` URL can carry one as well as ``.git-credentials``."""
    stores = list(stores)
    spans = _url_userinfo_spans(text, mask)
    if extent == "mixed":
        spans += _mixed_spans(text, stores, mask, secret_key)
    else:
        for fmt in sorted({s.fmt for s in stores}):
            spans += _format_spans(text, fmt, mask, secret_key, extent == "start")
    if not spans:
        return text
    merged: list[list] = []
    for start, end, repl in sorted(spans):
        if merged and start < merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1][1] = end
                merged[-1][2] = None  # overlapping readings: re-mask the widened span
            continue
        merged.append([start, end, repl])
    out, pos = [], 0
    for start, end, repl in merged:
        out.append(text[pos:start])
        out.append(mask(text[start:end]) if repl is None else repl)
        pos = end
    out.append(text[pos:])
    return "".join(out)

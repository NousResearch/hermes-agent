"""Stateful display-only redaction over sequential raw fragments.

Adapted from #70093's cross-fragment guard. Canonical static policy remains in
agent.redact; display/segment boundaries must not reset an unfinished candidate.
"""
from __future__ import annotations

import re
import json
import copy
from functools import lru_cache
from agent import redact as _redact
from agent.redact import (
    _SECRET_ENV_NAMES, _PRIVATE_KEY_RE, _TELEGRAM_RE, _SIGNAL_PHONE_RE,
    _SENSITIVE_QUERY_PARAMS, redact_sensitive_text,
)

_SECRET_ENV_WORDS = ("API_KEY", "APIKEY", "KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "PASS", "PW", "CREDENTIAL", "AUTH")
_AUTH_HEADER_NAMES = ("Authorization", "Proxy-Authorization")
_SECRET_HEADER_NAME_VALUES = ("x-api-key", "x-goog-api-key", "api-key", "apikey", "x-api-token", "x-auth-token", "x-access-token")
_STREAMING_JWT_CANDIDATE_RE = re.compile(r"eyJ[A-Za-z0-9_.=-]*$")

def _expand_streaming_prefix(expression: str) -> tuple[str, ...]:
    """Expand the literal portion of a known-prefix regex.

    The current patterns use literal characters, escaped literals, and one
    simple character class (the Slack ``xox[baprs]-`` family). Keeping this
    derivation next to ``_PREFIX_PATTERNS`` prevents the streaming guard from
    becoming a second manually maintained credential list.
    """
    # Slack app-level tokens contain a variable numeric component
    # (``xapp-\d+-...``).  Retaining from the stable ``xapp-`` stem is
    # conservative and lets the static redactor decide whether the completed
    # token is valid once a delimiter arrives.
    variable_digits = expression.find(r"\d+")
    if variable_digits >= 0:
        expression = expression[:variable_digits]

    expanded = [""]
    i = 0
    while i < len(expression):
        char = expression[i]
        if char == "\\" and i + 1 < len(expression):
            expanded = [prefix + expression[i + 1] for prefix in expanded]
            i += 2
            continue
        if char == "[":
            closing = expression.find("]", i + 1)
            if closing < 0:
                return ()
            choices = expression[i + 1:closing]
            if not choices or any(ch in choices for ch in "\\^-"):
                return ()
            expanded = [prefix + choice for prefix in expanded for choice in choices]
            i = closing + 1
            continue
        if char in ".^$*+?{}()|":
            return ()
        expanded = [prefix + char for prefix in expanded]
        i += 1
    return tuple(expanded)


@lru_cache(maxsize=8)
def _build_streaming_prefix_specs(patterns):
    """Derive (literal prefixes, token-char regex, minimum) specifications."""
    specs = []
    parts_re = re.compile(r"^(.*)(\[[^\]]+\])\{(\d+)(,?)\}$")
    for pattern in patterns:
        match = parts_re.fullmatch(pattern)
        prefixes = _expand_streaming_prefix(match.group(1)) if match else ()
        if not prefixes:
            # Either the token-class shape or its literal expansion may be unsupported.
            # Retain from their guaranteed literal stem until final resolution.
            literal = _redact._extract_literal_prefix(pattern)
            if literal:
                specs.append(((literal,), re.compile(r"^[\s\S]*$"), 0))
            continue
        # The canonical matcher also joins controls inside credential bodies.
        token_chars = re.compile(rf"^(?:{match.group(2)}|{_redact._CONTROL_CHARS_RE.pattern})*$")
        specs.append((prefixes, token_chars, int(match.group(3))))
    return tuple(specs)


def _streaming_prefix_specs():
    # Registration after consumer construction must take effect on the next feed.
    return _build_streaming_prefix_specs(tuple(_redact._PREFIX_PATTERNS + _redact._plugin_patterns()))
_STREAMING_JSON_KEYS = (
    "apikey",
    "api_key",
    "token",
    "secret",
    "password",
    "access_token",
    "refresh_token",
    "auth_token",
    "bearer",
    "secret_value",
    "raw_secret",
    "secret_input",
    "key_material",
)
_STREAMING_JSON_OPENER_RE = re.compile(
    r'"(?:' + "|".join(re.escape(key) for key in _STREAMING_JSON_KEYS)
    + r')"\s*:\s*"',
    re.IGNORECASE,
)
_STREAMING_JSON_KEY_LITERALS = tuple(
    f'"{key}"' for key in _STREAMING_JSON_KEYS
)
_STREAMING_JSON_KEY_PREFIXES = tuple(
    literal[:-1] for literal in _STREAMING_JSON_KEY_LITERALS
)
_STREAMING_DB_PROTOCOLS = (
    "postgres://",
    "postgresql://",
    "mysql://",
    "mongodb://",
    "mongodb+srv://",
    "redis://",
    "amqp://",
)
_STREAMING_WEB_PROTOCOLS = ("http://", "https://", "ws://", "wss://", "git://", "ssh://", "ftp://", "ftps://", "sftp://")
_STREAMING_DB_OPENER_RE = re.compile(
    r"(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^:\s]+:",
    re.IGNORECASE,
)
_STREAMING_PRIVATE_KEY_OPENER_RE = re.compile(
    r"-----BEGIN[A-Z ]*PRIVATE KEY-----"
)
_STREAMING_PRIVATE_KEY_END_RE = re.compile(
    r"-----END[A-Z ]*PRIVATE KEY-----"
)
_STREAMING_AUTH_HEADER_LITERALS = tuple(
    name.lower() + ":" for name in _AUTH_HEADER_NAMES
)
_STREAMING_SECRET_HEADER_LITERALS = tuple(
    name.lower() + ":" for name in _SECRET_HEADER_NAME_VALUES
)
_STREAMING_TELEGRAM_CANDIDATE_RE = re.compile(
    r"(?:bot)?\d{8,}:[-A-Za-z0-9_]{0,29}$"
)
_STREAMING_JWT_PRETHRESHOLD_RE = re.compile(
    r"e(?:y(?:J[A-Za-z0-9_.=-]{0,9})?)?$"
)
_STREAMING_JWT_BOUNDARY_PRETHRESHOLD_RE = re.compile(
    r"ey(?:J[A-Za-z0-9_.=-]{0,9})?$"
)
_STREAMING_PHONE_CANDIDATE_RE = re.compile(
    r"\+(?:[1-9]\d{0,13})?$"
)
_STREAMING_ENV_ACTIVE_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])([A-Za-z0-9_.-]*(?:key|token|secret|password|passwd|pass|pw|credential|auth)[A-Za-z0-9_.-]*)"
    r"\s*=\s*(['\"]?)", re.IGNORECASE)
_STREAMING_ENV_SUFFIX_RE = re.compile(r"(?<![A-Za-z0-9_.-])([A-Za-z0-9_.-]+)([ \t]*)(?:=([ \t]*['\"]?))?\Z")
_STREAMING_YAML_ACTIVE_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])([A-Za-z0-9_.-]*(?:key|token|secret|password|passwd|credential|auth)[A-Za-z0-9_.-]*)[ \t]*:[ \t]*[^\r\n]*$", re.IGNORECASE)


def _known_prefix_candidate_start(
    text: str, *, include_partial: bool = True,
) -> int | None:
    """Return the start of a trailing known-prefix token candidate."""
    held_from = len(text)
    token_boundary_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"

    for prefixes, token_chars, _minimum in _streaming_prefix_specs():
        for prefix in prefixes:
            # A complete literal prefix followed only by token characters stays
            # sensitive until a delimiter arrives. This includes candidates
            # that already reached the static redactor's minimum length.
            start = text.rfind(prefix)
            while start >= 0:
                if (
                    (start == 0 or text[start - 1] not in token_boundary_chars)
                    and token_chars.fullmatch(text[start + len(prefix):])
                ):
                    held_from = min(held_from, start)
                    break
                start = text.rfind(prefix, 0, start)

            # Also retain a literal-prefix fragment split across deltas.
            if not include_partial:
                continue
            for length in range(1, len(prefix)):
                if not text.endswith(prefix[:length]):
                    continue
                start = len(text) - length
                if start == 0 or text[start - 1] not in token_boundary_chars:
                    held_from = min(held_from, start)

    return held_from if held_from < len(text) else None


def _could_be_json_opener_at(text: str, start: int) -> bool:
    """Whether ``text[start:]`` can become a JSON secret opener.

    Work from offsets so callers never allocate a suffix for every quote in a
    quote-rich stream. Key comparisons are bounded by the finite canonical key
    set; only whitespace after an exact key can be unbounded.
    """
    if start < 0 or start >= len(text) or text[start] != '"':
        return False
    body_start = start + 1
    body_length = len(text) - body_start
    for key in _STREAMING_JSON_KEYS:
        if body_length <= len(key):
            if key.startswith(text[body_start:].lower()):
                return True
            continue
        key_end = body_start + len(key)
        if text[body_start:key_end].lower() != key:
            continue
        if text[key_end] != '"':
            continue
        cursor = key_end + 1
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor == len(text):
            return True
        if text[cursor] != ":":
            continue
        cursor += 1
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        return cursor == len(text) or (
            cursor + 1 == len(text) and text[cursor] == '"'
        )
    return False


def _could_be_json_opener(candidate: str) -> bool:
    """Whether a trailing quote-led fragment can become a JSON secret opener."""
    return _could_be_json_opener_at(candidate, 0)


def _partial_json_opener_start(
    text: str,
    *,
    embedded_prefixes: bool,
) -> int | None:
    """Return a trailing JSON opener using finite literals and bounded prefixes."""
    lower = text.lower()
    candidates: list[int] = []

    # Exact key literals may be followed by canonical unbounded whitespace.
    # Only the latest occurrence of each finite literal can be viable: an
    # earlier occurrence necessarily contains the later non-whitespace quote.
    for literal in _STREAMING_JSON_KEY_LITERALS:
        start = lower.rfind(literal)
        if start < 0:
            continue
        field_boundary = start == 0 or text[start - 1] in "{[, \t\r\n"
        if (
            (embedded_prefixes or field_boundary)
            and _could_be_json_opener_at(text, start)
        ):
            candidates.append(start)

    # A key that has not reached its closing quote is bounded by the longest
    # known key. Check literal prefixes only at the tail, never every quote.
    for prefix in _STREAMING_JSON_KEY_PREFIXES:
        for length in range(1, len(prefix) + 1):
            if not lower.endswith(prefix[:length]):
                continue
            start = len(text) - length
            if length == 1 and any(
                _json_closing_quote(text, match.end()) == start
                for match in _STREAMING_JSON_OPENER_RE.finditer(text)
            ):
                # This is the closing quote of a complete sensitive JSON
                # value, not a new embedded opener. Splitting here would
                # detach the raw value from its key before static redaction.
                break
            field_boundary = start == 0 or text[start - 1] in "{[, \t\r\n"
            if embedded_prefixes or field_boundary:
                candidates.append(start)
            break

    return min(candidates) if candidates else None


def _json_closing_quote(text: str, start: int) -> int:
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            return index
    return -1


def _pending_json_stage(text: str) -> str | None:
    """Classify a retained JSON opener for incremental continuation."""
    lower = text.lower()
    for literal in _STREAMING_JSON_KEY_LITERALS:
        if not lower.startswith(literal):
            continue
        cursor = len(literal)
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor == len(text):
            return "key_whitespace"
        if text[cursor] != ":":
            continue
        cursor += 1
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor == len(text):
            return "colon_whitespace"
        if text[cursor] != '"':
            continue
        if _json_closing_quote(text, cursor + 1) < 0:
            return "value"
    return None


def _could_be_db_opener(candidate: str) -> bool:
    """Whether a trailing fragment can become a DB password opener."""
    lower = candidate.lower()
    for protocol in _STREAMING_DB_PROTOCOLS:
        if protocol.startswith(lower):
            return True
        if not lower.startswith(protocol):
            continue
        username = candidate[len(protocol):]
        if not username:
            return True
        if any(char.isspace() for char in username):
            continue
        if ":" not in username:
            return True
        return username.endswith(":") and username.count(":") == 1
    return False


def _could_be_private_key_opener(candidate: str) -> bool:
    """Whether a trailing fragment can become a private-key BEGIN marker."""
    literal = "-----BEGIN"
    if literal.startswith(candidate):
        return True
    if not candidate.startswith(literal):
        return False
    remainder = candidate[len(literal):]
    # The type label is unbounded, but only its final PRIVATE KEY word and
    # up to five closing hyphens matter. Scan it once, not once per split.
    label, separator, hyphens = remainder.partition("-")
    if not all(char == " " or "A" <= char <= "Z" for char in label):
        return False
    return not separator or (
        label.endswith("PRIVATE KEY") and len(hyphens) < 5
        and not hyphens.strip("-")
    )


def _partial_env_opener_start(text: str, *, embedded_prefixes: bool) -> int | None:
    """Hold the trailing key run once; lowercase/dotted names may grow unbounded."""
    match = _STREAMING_ENV_SUFFIX_RE.search(text)
    if match is None:
        return None
    # A key-shaped suffix inside the current assignment's value is not a new
    # opener. Splitting there would send its original key before its raw value.
    for assignment in _STREAMING_ENV_ACTIVE_RE.finditer(text, 0, match.start()):
        if not any(char.isspace() for char in text[assignment.end():match.start()]):
            return None
    name = match.group(1)
    if not embedded_prefixes:
        # Event callbacks carry complete commands. Unknown lowercase words are
        # not arbitrary fragments of a future key; explicit secret words and
        # their prefixes still retain state, as do uppercase ENV candidates.
        lower = name.lower()
        words = tuple(word.lower() for word in _SECRET_ENV_WORDS)
        if not (name.isupper() or any(word in lower or word.startswith(lower) for word in words)):
            return None
    if match.group(2) or match.group(3) is not None:
        if not any(word.lower() in name.lower() for word in _SECRET_ENV_WORDS):
            return None
    return match.start()


def _could_be_header_candidate(
    candidate: str,
    literals: tuple[str, ...],
    *,
    authorization: bool,
) -> bool:
    """Whether a trailing fragment can become or extend a secret header."""
    lower = candidate.lower()
    for literal in literals:
        if literal.startswith(lower):
            return True
        if not lower.startswith(literal):
            continue
        remainder = candidate[len(literal):]
        if "\n" in remainder or "\r" in remainder:
            continue
        stripped = remainder.lstrip()
        if not stripped:
            return True
        tokens = stripped.split()
        trailing_space = stripped[-1].isspace()
        if not authorization:
            return len(tokens) == 1 and not trailing_space
        if len(tokens) == 1:
            # A scheme-shaped first token remains ambiguous with the canonical
            # bare-credential form until a following credential arrives.
            return (
                not trailing_space
                or re.fullmatch(r"[A-Za-z][\w.+-]*", tokens[0]) is not None
            )
        if len(tokens) == 2:
            return not trailing_space
        return False
    return False


def _partial_form_opener_start(text: str) -> int | None:
    """Return a pure form body's start when its last key is still incomplete."""
    line_start = text.rfind("\n") + 1
    candidate = text[line_start:]
    line_start += len(candidate) - len(candidate.lstrip(" \t"))
    candidate = text[line_start:]
    if not candidate or any(char.isspace() for char in candidate):
        return None
    parts = candidate.split("&")
    if any(
        re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*=[^&\s]*", part) is None
        for part in parts[:-1]
    ):
        return None
    last = parts[-1]
    if "=" in last:
        key, value = last.split("=", 1)
        # A nonsensitive first pair still establishes the canonical whole-form
        # context needed by later sensitive pairs. Keep it until a delimiter.
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", key):
            return line_start
    else:
        key = last
    lower = key.lower()
    if not key:
        return line_start
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", key or ""):
        return None
    if any(secret.startswith(lower) for secret in _SENSITIVE_QUERY_PARAMS):
        return line_start
    return None


def _active_env_assignment_start(text: str) -> int | None:
    """Return an ENV assignment whose value is still growing at the tail."""
    candidates = []
    for match in _STREAMING_ENV_ACTIVE_RE.finditer(text):
        remainder = text[match.end():]
        quote = match.group(2)
        if quote:
            closing = remainder.find(quote)
            if closing < 0 or not any(char.isspace() for char in remainder[closing + 1:]):
                candidates.append(match.start())
        elif remainder and not any(char.isspace() for char in remainder):
            candidates.append(match.start())
    return min(candidates) if candidates else None


def _pending_env_stage(text: str) -> tuple[str, str] | None:
    """Classify a retained ENV value for incremental continuation."""
    opener = _STREAMING_ENV_SUFFIX_RE.fullmatch(text)
    if opener and any(word.lower() in opener.group(1).lower() for word in _SECRET_ENV_WORDS):
        return "env_opener_whitespace", ""
    for match in _STREAMING_ENV_ACTIVE_RE.finditer(text):
        remainder = text[match.end():]
        quote = match.group(2)
        if quote:
            closing = remainder.find(quote)
            if closing < 0:
                return "env_quoted_value", quote
            if closing == len(remainder) - 1:
                return "env_quoted_closed", quote
            if not any(char.isspace() for char in remainder[closing + 1:]):
                return "env_unquoted_value", ""
        elif remainder and not any(char.isspace() for char in remainder):
            return "env_unquoted_value", ""
    return None


def _partial_context_opener_start(
    text: str,
    *,
    embedded_prefixes: bool = True,
) -> int | None:
    """Return the earliest trailing fragment that can become an opener."""
    candidates: list[int] = []

    json_start = _partial_json_opener_start(
        text,
        embedded_prefixes=embedded_prefixes,
    )
    if json_start is not None:
        candidates.append(json_start)

    # DB usernames are unbounded in the static grammar. Search for each full
    # protocol across the whole buffer, then check short literal fragments at
    # the tail. This retains ``scheme://user`` even when a very long username
    # crosses the platform overflow threshold before its password colon.
    lower = text.lower()
    for protocol in _STREAMING_DB_PROTOCOLS:
        start = lower.rfind(protocol)
        if (
            start >= 0
            and (
                embedded_prefixes
                or start == 0
                or text[start - 1] not in
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
            )
            and _could_be_db_opener(text[start:])
        ):
            candidates.append(start)
        for length in range(1, len(protocol)):
            if not lower.endswith(protocol[:length]):
                continue
            start = len(text) - length
            if (
                embedded_prefixes
                or start == 0
                or text[start - 1] not in
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
            ):
                candidates.append(start)

    # Keep URL context intact: ordinary text redaction deliberately preserves
    # signed web queries, and bare-token userinfo is only known at its @.
    for protocol in (_STREAMING_WEB_PROTOCOLS if embedded_prefixes else ()):
        start = lower.rfind(protocol)
        if start >= 0 and not any(char.isspace() for char in text[start:]):
            candidates.append(start)
        for length in range(1, len(protocol)):
            if lower.endswith(protocol[:length]):
                candidates.append(len(text) - length)

    # Only the latest private-key BEGIN marker can still be incomplete: an
    # earlier marker's candidate necessarily contains the later marker and
    # therefore cannot match the canonical uppercase/space grammar.
    private_scan_start = 0
    for completed in _PRIVATE_KEY_RE.finditer(text):
        private_scan_start = completed.end()
    private_literal = "-----BEGIN"
    private_start = text.rfind(private_literal, private_scan_start)
    if (
        private_start >= private_scan_start
        and _could_be_private_key_opener(text[private_start:])
    ):
        candidates.append(private_start)
    else:
        for length in range(1, len(private_literal)):
            partial_start = len(text) - length
            if (
                partial_start >= private_scan_start
                and text.endswith(private_literal[:length])
            ):
                candidates.append(partial_start)
                break

    env_start = _partial_env_opener_start(
        text,
        embedded_prefixes=embedded_prefixes,
    )
    if env_start is not None:
        candidates.append(env_start)

    # Header whitespace is intentionally unbounded, but a candidate must begin
    # at a known literal (or a trailing prefix of one). The latest complete
    # literal is sufficient: any earlier candidate also contains it and cannot
    # satisfy the one-header grammar.
    lower = text.lower()
    for literals, authorization in (
        (_STREAMING_AUTH_HEADER_LITERALS, True),
        (_STREAMING_SECRET_HEADER_LITERALS, False),
    ):
        header_candidates: list[int] = []
        for literal in literals:
            start = lower.rfind(literal)
            if (
                start >= 0
                and _could_be_header_candidate(
                    text[start:],
                    (literal,),
                    authorization=authorization,
                )
            ):
                header_candidates.append(start)
            for length in range(1, len(literal)):
                if lower.endswith(literal[:length]):
                    header_candidates.append(len(text) - length)
                    break
        if header_candidates:
            candidates.append(min(header_candidates))

    form_start = _partial_form_opener_start(text)
    if form_start is not None:
        candidates.append(form_start)
    env_value_start = _active_env_assignment_start(text)
    if env_value_start is not None:
        candidates.append(env_value_start)
    yaml = _STREAMING_YAML_ACTIVE_RE.search(text)
    if yaml is not None:
        candidates.append(yaml.start())

    return min(candidates) if candidates else None


def _unterminated_context(text: str):
    """Return the earliest complete sensitive opener lacking its terminator."""
    active = []
    for match in _STREAMING_JSON_OPENER_RE.finditer(text):
        if _json_closing_quote(text, match.end()) < 0:
            active.append((match.start(), "json", match))
    for match in _STREAMING_DB_OPENER_RE.finditer(text):
        if text.find("@", match.end()) < 0:
            active.append((match.start(), "db", match))
    for match in _STREAMING_PRIVATE_KEY_OPENER_RE.finditer(text):
        if _STREAMING_PRIVATE_KEY_END_RE.search(text, match.end()) is None:
            active.append((match.start(), "private_key", match))
    return min(active, key=lambda item: item[0]) if active else None


def _recognized_token_candidate_start(
    text: str,
    *,
    embedded_prefixes: bool = True,
) -> int | None:
    """Return a trailing non-prefix token already recognized as sensitive.

    These grammars redact an end-of-buffer token before a delimiter arrives.
    Retaining the raw match prevents later token bytes from being appended to
    a destructive mask. The canonical production regexes remain the source of
    truth; JWT only adds an allowed-character continuation around its shared
    canonical header pattern so a partial payload segment stays attached.
    """
    candidates: list[int] = []

    for pattern in (_TELEGRAM_RE, _SIGNAL_PHONE_RE):
        for match in pattern.finditer(text):
            if match.end() == len(text):
                candidates.append(match.start())

    jwt_match = _STREAMING_JWT_CANDIDATE_RE.search(text)
    if jwt_match is not None:
        candidates.append(jwt_match.start())

    # Hold candidates from the earliest point where the grammar is
    # recognizable, not only after the static redactor's minimum. Otherwise a
    # preview exposes almost the entire secret before the final byte masks it.
    telegram_match = _STREAMING_TELEGRAM_CANDIDATE_RE.search(text)
    if telegram_match is not None:
        candidates.append(telegram_match.start())
    jwt_prethreshold_re = (
        _STREAMING_JWT_PRETHRESHOLD_RE
        if embedded_prefixes
        else _STREAMING_JWT_BOUNDARY_PRETHRESHOLD_RE
    )
    jwt_prethreshold_match = jwt_prethreshold_re.search(text)
    if jwt_prethreshold_match is not None:
        candidates.append(jwt_prethreshold_match.start())
    phone_match = _STREAMING_PHONE_CANDIDATE_RE.search(text)
    if phone_match is not None:
        candidates.append(phone_match.start())

    return min(candidates) if candidates else None


def split_incomplete_sensitive_suffix(
    text: str,
    *,
    final: bool = False,
    logical_boundary: bool = False,
    embedded_prefixes: bool = True,
) -> tuple[str, str]:
    """Keep a raw trailing secret candidate out of streaming writes.

    The static redactor needs a whole match. A stream can end a platform write
    before a known-prefix delimiter, JSON closing quote, DB ``@`` delimiter, or
    PEM end marker arrives. Retain that raw suffix so later deltas are matched
    with their original context instead of appending to destructively masked
    display text.

    On the terminal stream tick, conservatively replace an unterminated
    structured value. Prefix-shaped prose below a known pattern's minimum is
    released unchanged because it never became a credential match.
    """
    if not text:
        return text, ""

    context = _unterminated_context(text)
    # ``logical_boundary`` remains in the public call shape for compatibility,
    # but segment/commentary boundaries are attacker-influenced and therefore
    # cannot terminate secret recognition state.
    known_start = _known_prefix_candidate_start(text)
    partial_start = _partial_context_opener_start(
        text,
        embedded_prefixes=embedded_prefixes,
    )
    token_start = _recognized_token_candidate_start(
        text,
        embedded_prefixes=embedded_prefixes,
    )

    if final:
        if context is None:
            return text, ""
        start, kind, match = context
        if kind == "json":
            return text[:start] + match.group(0) + '***"', ""
        if kind == "db":
            return text[:start] + match.group(0) + "***", ""
        return text[:start] + "[REDACTED PRIVATE KEY]", ""

    starts = [
        start
        for start in (
            context[0] if context is not None else None,
            known_start,
            partial_start,
            token_start,
        )
        if start is not None
    ]
    if not starts:
        return text, ""
    held_from = min(starts)
    # A one-character candidate can begin inside otherwise ordinary prose
    # (for example the ``re`` at the end of ``more`` can still become
    # ``redis://`` because the canonical DB grammar has no left boundary).
    # Retain its containing lexical token as well, avoiding permanent
    # mid-word splits when a logical platform boundary lands at that point.
    while (
        held_from > 0
        and text[held_from - 1]
        in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.+-"
    ):
        held_from -= 1
    return text[:held_from], text[held_from:]


_ESCAPED_JSON_FIELD_RE = re.compile(
    rf'"({_redact._JSON_KEY_NAMES})"\s*:\s*"((?:\\.|[^"\\])*)"', re.IGNORECASE | re.DOTALL)


def sanitize_terminal_secret_text(text: str) -> str:
    """Force-redact a complete adapter-bound text payload."""
    if not text:
        return text
    terminal_text, _held = split_incomplete_sensitive_suffix(
        str(text),
        final=True,
    )
    # Escaped quotes belong to the same JSON value. The static field regex
    # predates escaped strings; reuse its ambiguity gate and masking policy.
    def escaped_field(match):
        raw = match.group(2)
        if "\\" not in raw:
            return match.group(0)
        try:
            value = json.loads('"' + raw + '"')
        except ValueError:
            # Invalid escaping does not make a recognized secret field safe.
            # Keep the canonical ambiguity gate, using the undecodable value.
            value = raw
        if not _redact._should_redact_assignment(match.group(1), value, check_keyword=False):
            return match.group(0)
        # A partial head may itself contain an escaped quote. Use the opaque
        # mask so the older static JSON matcher cannot split it a second time.
        return '"' + match.group(1) + '": "***"'
    terminal_text = _ESCAPED_JSON_FIELD_RE.sub(escaped_field, terminal_text)
    return redact_sensitive_text(terminal_text, force=True)


class StreamingSecretSanitizer:
    """Stateful forced redaction for sequential adapter-bound text fragments.

    Formatting must be applied after ``feed``: inserting labels, quotes, or
    fences between fragments would otherwise break the canonical grammar that
    the retained bytes are meant to recognize.

    The default mode consumes one sequential text stream, including arbitrary
    lowercase/dotted key fragments. ``token_candidates_only`` and
    ``embedded_prefixes=False`` are for complete callback events: preserve
    ordinary command/URL endings while retaining explicit token and structured
    candidates. An unrelated ordinary event is not an unknown key fragment.
    Progress-only mode terminal-masks incomplete database URLs per event.
    """

    def __init__(
        self,
        *,
        token_candidates_only: bool = False,
        embedded_prefixes: bool = True,
    ) -> None:
        self._pending = ""
        self._json_escaped = False
        self._left_context = ""
        self._pending_parts: list[str] = []
        self._pending_length = 0
        self._pending_json_stage: str | None = None
        self._pending_env_stage: str | None = None
        self._pending_env_quote = ""
        self._continuation = None
        self._registry_matcher = None
        self._token_candidates_only = token_candidates_only
        self._embedded_prefixes = embedded_prefixes

    @property
    def pending(self) -> str:
        return self._pending + "".join(self._pending_parts)

    @property
    def pending_length(self) -> int:
        return self._pending_length

    def fork_for_events(self) -> StreamingSecretSanitizer:
        """Retain candidate/context in a separate complete-callback stream."""
        fork = copy.deepcopy(self)
        fork._embedded_prefixes = False
        # Rebuild fast continuations for the event contract as well: a generic
        # identifier held by sequential text is not an unrelated event's tail.
        fork._replace_pending(fork.pending)
        return fork

    def _replace_pending(self, text: str) -> None:
        self._pending = text
        self._pending_parts = []
        self._pending_length = len(text)
        self._pending_json_stage = _pending_json_stage(text)
        self._json_escaped = bool((len(text) - len(text.rstrip("\\"))) % 2)
        env_state = _pending_env_stage(text)
        if env_state is None:
            self._pending_env_stage = None
            self._pending_env_quote = ""
        else:
            self._pending_env_stage, self._pending_env_quote = env_state

        self._registry_matcher = _redact._PREFIX_RE
        self._continuation = None
        for prefixes, token_chars, _minimum in _streaming_prefix_specs():
            if any(text.startswith(prefix) and token_chars.fullmatch(text[len(prefix):]) for prefix in prefixes):
                self._continuation = token_chars
                break
        if self._continuation is None and text:
            if text.startswith("-----BEGIN"):
                # A PEM is rare display output. Retain it through end-of-input:
                # this also bounds scans of arbitrary type labels and bodies.
                self._continuation = re.compile(r"[\s\S]*")
            elif re.match(r"[A-Za-z_][A-Za-z0-9_.-]*=", text) and not any(char.isspace() for char in text):
                self._continuation = re.compile(r"\S*")
            elif (self._embedded_prefixes and not self._token_candidates_only
                  and re.fullmatch(r"[A-Za-z0-9_.-]+", text)):
                self._continuation = re.compile(r"[A-Za-z0-9_.-]*")
            elif any(text.lower().startswith(protocol) for protocol in _STREAMING_DB_PROTOCOLS) and "@" not in text:
                self._continuation = re.compile(r"[^@\s]*")
            elif any(text.lower().startswith(protocol) for protocol in _STREAMING_WEB_PROTOCOLS) and not any(char.isspace() for char in text):
                self._continuation = re.compile(r"\S*")
            elif _STREAMING_YAML_ACTIVE_RE.fullmatch(text):
                self._continuation = re.compile(r"[^\r\n]*")

    def _sanitize_visible(self, text: str) -> str:
        if not text:
            return ""
        # Retain only the canonical line-prefix state, never the growing raw
        # line: indentation and an optional export keep anchored assignments
        # eligible; ordinary prose must not become a synthetic line start.
        context = self._left_context
        raw = context + text
        if context == " " and _redact._FORM_BODY_RE.fullmatch(text.strip()):
            # A form may follow already-emitted horizontal indentation. Keep
            # the opaque whole-form policy without inventing a real newline.
            output = context + sanitize_terminal_secret_text(text)
        else:
            output = sanitize_terminal_secret_text(raw)
        line_prefix = raw.rsplit("\n", 1)[-1]
        if not line_prefix.strip(" \t"):
            self._left_context = "\n" if "\n" in raw else " "
        elif re.fullmatch(r"[ \t]*export[ \t]+", line_prefix, re.IGNORECASE):
            self._left_context = "export "
        else:
            self._left_context = "x "
        return output[len(context):]

    def _consume_pending(self) -> str:
        text = self.pending
        self._pending = ""
        self._pending_parts = []
        self._pending_length = 0
        self._pending_json_stage = None
        self._pending_env_stage = None
        self._pending_env_quote = ""
        self._continuation = None
        return text

    def _extend_incremental_candidate(self, text: str) -> bool:
        """Retain a validated continuation without rescanning its prefix."""
        if self._registry_matcher is not _redact._PREFIX_RE:
            return False
        if self._continuation is not None and self._continuation.fullmatch(text):
            self._pending_parts.append(text)
            self._pending_length += len(text)
            return True
        stage = self._pending_json_stage
        if stage in ("key_whitespace", "colon_whitespace"):
            if not text:
                return True
            if text.isspace():
                self._pending_parts.append(text)
                self._pending_length += len(text)
                return True
        elif stage == "value":
            escaped = self._json_escaped
            for char in text:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    return False
            self._json_escaped = escaped
            self._pending_parts.append(text)
            self._pending_length += len(text)
            return True

        env_stage = self._pending_env_stage
        if env_stage == "env_opener_whitespace" and (not text or text.isspace()):
            self._pending_parts.append(text)
            self._pending_length += len(text)
            return True
        if env_stage == "env_unquoted_value" and not any(
            char.isspace() for char in text
        ):
            self._pending_parts.append(text)
            self._pending_length += len(text)
            return True
        if env_stage == "env_quoted_value":
            closing = text.find(self._pending_env_quote)
            if closing < 0:
                self._pending_parts.append(text)
                self._pending_length += len(text)
                return True
            if closing == len(text) - 1:
                self._pending_parts.append(text)
                self._pending_length += len(text)
                self._pending_env_stage = "env_quoted_closed"
                return True
        if env_stage == "env_quoted_closed" and not any(char.isspace() for char in text):
            # Shell-style concatenation after a closing quote is still the
            # same value; a message boundary cannot expose the appended tail.
            self._pending_parts.append(text)
            self._pending_length += len(text)
            self._pending_env_stage = "env_unquoted_value"
            return True
        return False

    def feed_with_metadata(
        self,
        text: str,
        *,
        final: bool = False,
    ) -> tuple[str, str | None, int, int]:
        """Feed text and snapshot retained history only on a transition.

        The second tuple item is ``None`` when an already validated JSON
        candidate advanced incrementally without emitting. Callers that need
        event-boundary metadata can therefore avoid joining retained chunks on
        every streaming update.
        """
        incoming = str(text or "")
        pending_before_length = self._pending_length
        if (
            not final
            and (
                self._continuation is not None
                or self._pending_json_stage is not None
                or self._pending_env_stage is not None
            )
            and self._extend_incremental_candidate(incoming)
        ):
            return "", None, pending_before_length, self._pending_length
        pending_before = self._consume_pending()
        combined = pending_before + incoming
        if self._token_candidates_only and not final:
            full_known_start = _known_prefix_candidate_start(
                combined,
                include_partial=False,
            )
            partial_known_start = _known_prefix_candidate_start(combined)
            token_start = _recognized_token_candidate_start(combined)
            context = _unterminated_context(combined)
            starts = []
            if full_known_start is not None:
                starts.append(full_known_start)
            if partial_known_start == 0:
                starts.append(partial_known_start)
            if token_start == 0:
                starts.append(token_start)
            # Progress callbacks are independent display events, so an
            # unterminated DB URL in one event must not absorb the next event.
            # JSON and ENV fragments are different: their opener/value grammar
            # is explicitly structured and may be split by the producer across
            # consecutive callbacks, so retain that complete state.
            if context is not None and context[1] == "json":
                starts.append(context[0])
            json_start = _partial_json_opener_start(
                combined,
                embedded_prefixes=False,
            )
            if json_start is not None:
                starts.append(json_start)
            env_start = _partial_env_opener_start(
                combined,
                embedded_prefixes=False,
            )
            if env_start is not None:
                starts.append(env_start)
            env_value_start = _active_env_assignment_start(combined)
            if env_value_start is not None:
                starts.append(env_value_start)
            if context is not None and context[1] == "db":
                # DB URLs are complete progress events. Do not split their
                # password off as an unkeyed ENV/token suffix after masking the
                # opener: that would release the raw value on a later event.
                starts = [start for start in starts if start < context[0]]
            if starts:
                held_from = min(starts)
                while (
                    held_from > 0
                    and combined[held_from - 1]
                    in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.+-"
                ):
                    held_from -= 1
                visible, pending = (
                    combined[:held_from],
                    combined[held_from:],
                )
            else:
                visible, pending = combined, ""
        else:
            visible, pending = split_incomplete_sensitive_suffix(
                combined,
                final=final,
                embedded_prefixes=self._embedded_prefixes,
            )
        output = self._sanitize_visible(visible)
        self._replace_pending(pending)
        return (
            output,
            pending_before,
            pending_before_length,
            self._pending_length,
        )

    def feed(self, text: str, *, final: bool = False) -> str:
        visible, _pending_before, _before_length, _after_length = (
            self.feed_with_metadata(text, final=final)
        )
        return visible

    def flush(self) -> str:
        return self.feed("", final=True)

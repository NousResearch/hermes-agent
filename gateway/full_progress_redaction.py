"""Chat-strict redaction for complete Full-progress tool arguments."""

from __future__ import annotations

import re
import shlex
from typing import Any, Optional


# ``display.tool_progress: full`` sends complete tool arguments into human chat.
# It therefore uses a stricter policy than ordinary logs/tool output: URL
# credentials are never workflow inputs at this display boundary, and values
# beneath secret-like keys are fully masked rather than partially fingerprinted.
# RFC 3986 schemes start with an ASCII letter and may then contain letters,
# digits, ``+``, ``-``, or ``.``.  Full progress must not assume that only web
# schemes carry credentials: DSNs and tool/plugin-specific schemes are common.
# Apostrophes are URI data, including their common shell-concatenated spelling.
# Actual enclosing quotes are bounded by _redact_full_progress_url_spans.
# Commit to the longest shell spelling so a failing suffix cannot repartition it.
_FULL_PROGRESS_URL_APOSTROPHE = r"""(?>'\\''|'"'"'|')"""
_FULL_PROGRESS_URL_CHAR = rf"(?:{_FULL_PROGRESS_URL_APOSTROPHE}|[^\s<>\"'])"
_FULL_PROGRESS_URL_RE = re.compile(
    rf"(?i)(?<![a-z0-9+.-])[a-z][a-z0-9+.-]*://{_FULL_PROGRESS_URL_CHAR}+"
)
_FULL_PROGRESS_QUERY_FIELD = (
    rf"(?:{_FULL_PROGRESS_URL_APOSTROPHE}|[A-Za-z0-9._~%+\-])+"
    rf"(?:=(?:{_FULL_PROGRESS_URL_APOSTROPHE}|[^&\s#<>\"'])*)?"
)
_FULL_PROGRESS_QUERY_ONLY_RE = re.compile(
    rf"(?<![A-Za-z0-9_/?#])\?"
    rf"(?P<query>(?={_FULL_PROGRESS_URL_CHAR}*=){_FULL_PROGRESS_QUERY_FIELD}"
    rf"(?:&{_FULL_PROGRESS_QUERY_FIELD})*)"
    rf"(?P<fragment>#{_FULL_PROGRESS_URL_CHAR}*)?"
)
_FULL_PROGRESS_RELATIVE_URL_RE = re.compile(
    rf"(?P<prefix>(?<![A-Za-z0-9+.\-:/])(?:/{{1,2}}|\./|\.\./)"
    rf"(?:{_FULL_PROGRESS_URL_APOSTROPHE}|[^\s?#<>\"'])*)"
    rf"(?:\?(?P<query>(?={_FULL_PROGRESS_URL_CHAR}*=){_FULL_PROGRESS_QUERY_FIELD}"
    rf"(?:&{_FULL_PROGRESS_QUERY_FIELD})*))?"
    rf"(?P<fragment>#{_FULL_PROGRESS_URL_CHAR}*)?"
)
_FULL_PROGRESS_BARE_RELATIVE_URL_RE = re.compile(
    rf"(?P<prefix>(?<![A-Za-z0-9._~%!$&()*+,;=:@+\-/])"
    rf"[A-Za-z0-9._~%!$&()*+,;=@+\-]"
    rf"(?:{_FULL_PROGRESS_URL_APOSTROPHE}|[A-Za-z0-9._~%!$&()*+,;=@+\-])*"
    rf"(?:/(?:{_FULL_PROGRESS_URL_APOSTROPHE}|[A-Za-z0-9._~%!$&()*+,;=:@+\-])+)*)"
    rf"(?:"
    rf"\?(?P<query>(?={_FULL_PROGRESS_URL_CHAR}*=){_FULL_PROGRESS_QUERY_FIELD}"
    rf"(?:&{_FULL_PROGRESS_QUERY_FIELD})*)"
    rf"(?P<fragment>#{_FULL_PROGRESS_URL_CHAR}*)?"
    rf"|(?P<fragment_only>#{_FULL_PROGRESS_URL_CHAR}*)"
    rf")"
)
_FULL_PROGRESS_SECRET_HEADER_NAME = (
    r"(?:"
    r"proxy[-_]authorization|authorization|"
    r"set[-_]cookie|cookie|"
    r"(?:x[-_](?:goog[-_])?)?api[-_]?key|"
    r"(?:x[-_])?auth[-_]?token|"
    r"(?:x[-_])?(?:csrf|xsrf)[-_]?token"
    r")"
)
# A string/list header is represented as one field per line.  Anchoring avoids
# treating prose such as "the Authorization guide: ..." as a credential.  A
# ``://`` guard keeps a same-named RFC scheme from being mistaken for a header.
_FULL_PROGRESS_HEADER_LINE_RE = re.compile(
    rf"^(?P<indent>[ \t]*)(?P<name>{_FULL_PROGRESS_SECRET_HEADER_NAME})"
    rf"(?P<sep>[ \t]*:[ \t]*)(?!//)[^\r\n]*",
    re.IGNORECASE | re.MULTILINE,
)
# Frozen full-mode structured/CLI credential names, local to this display boundary. ``bearer`` is already covered by the canonical JSON scanner but is
# not yet a body key; full progress treats it as an opaque credential container
# alongside ``jwt``.
_FULL_PROGRESS_EXACT_SECRET_KEYS = frozenset({
    'access_token',
    'api_key',
    'apikey',
    'auth',
    'authorization',
    'bearer',
    'client_secret',
    'id_token',
    'jwt',
    'key',
    'password',
    'private_key',
    'refresh_token',
    'secret',
    'token',
})
_FULL_PROGRESS_SECRET_KEY_PARTS = frozenset(
    {
        "token",
        "secret",
        "password",
        "passwd",
        "passphrase",
        "credential",
        "credentials",
        "authorization",
        "authentication",
        "auth",
        "cookie",
        "signature",
    }
)
_FULL_PROGRESS_SECRET_COMPACT_KEYS = frozenset(
    {
        "apikey",
        "privatekey",
        "recoverykey",
        "sshkey",
        "keymaterial",
        "accesstoken",
        "refreshtoken",
        "idtoken",
        "clientsecret",
        "bearertoken",
        "accesskey",
        "secretkey",
        "signingkey",
        "key",
    }
)
_FULL_PROGRESS_CONNECTION_KEY_MARKERS = frozenset(
    {
        "connectionstring",
        "connectionurl",
        "connectionuri",
        "connectiondsn",
        "databaseurl",
        "databaseuri",
        "databasedsn",
        "dburl",
        "dburi",
        "dbdsn",
    }
)
_FULL_PROGRESS_REDACTED = "***"


def _full_progress_shell_words(value: str) -> list[tuple[int, int, str, str, int]]:
    """Return shell-like words with source spans for narrow CLI masking.

    This is deliberately not a shell parser. It only distinguishes whitespace,
    common command separators, quotes, and backslash escapes so credential value
    spans can be replaced without re-quoting or otherwise rewriting the command.
    The segment number prevents a flag at the end of one command from consuming
    the first word of the next command as its value.
    """
    words: list[tuple[int, int, str, str, int]] = []
    separators = ";|&()"
    segment = 0
    i = 0
    while i < len(value):
        while i < len(value) and (value[i].isspace() or value[i] in separators):
            if value[i] in separators or value[i] in "\r\n":
                segment += 1
            i += 1
        if i >= len(value):
            break

        start = i
        quote = ""
        while i < len(value):
            char = value[i]
            if quote:
                if char == quote:
                    quote = ""
                    i += 1
                    continue
                if quote == '"' and char == "\\" and i + 1 < len(value):
                    i += 2
                    continue
                i += 1
                continue
            if char in {"'", '"'}:
                quote = char
                i += 1
                continue
            if char == "\\" and i + 1 < len(value):
                i += 2
                continue
            if char.isspace() or char in separators:
                break
            i += 1

        raw = value[start:i]
        try:
            parsed = shlex.split(raw, posix=True)
            decoded = parsed[0] if len(parsed) == 1 else raw
        except ValueError:
            # An unfinished command may still be shown in progress. Preserve it
            # byte-for-byte and use a quote-trimmed view only for flag matching.
            decoded = raw.strip("'\"")
        words.append((start, i, raw, decoded, segment))

    return words


def _mask_full_progress_shell_word(raw: str) -> str:
    """Fully mask one shell word while retaining a whole-word quote pair."""
    if len(raw) >= 2 and raw[0] in {"'", '"'} and raw[-1] == raw[0]:
        return f"{raw[0]}{_FULL_PROGRESS_REDACTED}{raw[-1]}"
    return _FULL_PROGRESS_REDACTED


def _mask_full_progress_inline_cli_value(raw: str) -> str:
    """Mask the value in one ``--flag=value`` shell word quote-safely."""
    equals = raw.find("=")
    if equals < 0:
        return _mask_full_progress_shell_word(raw)
    prefix = raw[: equals + 1]
    cli_value = raw[equals + 1 :]
    # A quote can wrap the complete ``--flag=value`` word rather than just the
    # value. Its opener is already in *prefix*, so retain only the closer here.
    if raw[0:1] in {"'", '"'} and raw[-1:] == raw[0:1]:
        return f"{prefix}{_FULL_PROGRESS_REDACTED}{raw[-1]}"
    return prefix + _mask_full_progress_shell_word(cli_value)


def _is_full_progress_cli_secret_flag(flag: str, value: str) -> bool:
    """Return whether a normalized CLI flag/value pair carries credentials."""
    if flag in {"-u", "--user", "-U", "--proxy-user"}:
        # A username alone is ordinary identity metadata. Curl's credential
        # form is unambiguous because it contains the ``user:password`` colon.
        return ":" in value
    if flag == "-b":
        return True
    if not flag.startswith("--"):
        return False
    normalized = flag[2:].casefold().replace("-", "_")
    return (
        normalized in _FULL_PROGRESS_EXACT_SECRET_KEYS
        or normalized in _FULL_PROGRESS_SECRET_KEY_PARTS
        or normalized in {"oauth2_bearer", "cookie"}
    )


def _mask_full_progress_header_word(raw: str, decoded: str, prefix_length: int) -> str:
    """Map a decoded header prefix to source bytes, then discard its whole value."""
    chars: list[str] = []
    quote = ""
    boundary = None
    i = 0
    while i < len(raw):
        char = raw[i]
        if char == quote:
            quote = ""
            i += 1
            continue
        if not quote and char in {"'", '"'}:
            quote = char
            i += 1
            continue
        if len(chars) == prefix_length and boundary is None:
            boundary = (i, quote)
        if char == "\\" and quote != "'" and i + 1 < len(raw):
            following = raw[i + 1]
            if not quote or following in {'"', "\\"}:
                char = following
                i += 1
        chars.append(char)
        i += 1
    # The existing decoder remains authoritative. Ambiguous source mapping must
    # never retain a possible value or an unbalanced fragment of its quoting.
    if "".join(chars) != decoded or quote:
        return _mask_full_progress_shell_word(raw)
    if boundary is None:
        return raw  # Empty header value.
    end, closing_quote = boundary
    return raw[:end] + _FULL_PROGRESS_REDACTED + closing_quote


def _redact_full_progress_cli_credentials(
    value: str, *, protected: Optional[dict[str, str]] = None,
) -> str:
    """Mask common password/token CLI values without changing shell quoting."""
    words = _full_progress_shell_words(value)
    replacements: list[tuple[int, int, str]] = []
    header_starts: set[int] = set()
    consumed_index = -1
    for index, (start, end, raw, decoded, segment) in enumerate(words):
        # A consumed value may itself look like an option or ENV assignment.
        if index == consumed_index:
            continue
        header_word = None
        header = ""
        if decoded in {"-H", "--header"}:
            if index + 1 < len(words) and words[index + 1][4] == segment:
                header_word = words[index + 1]
                header = header_word[3]
                consumed_index = index + 1
        elif decoded.startswith("--header="):
            header_word, header = words[index], decoded[len("--header="):]
        elif decoded.startswith("-H") and len(decoded) > 2:
            header_word, header = words[index], decoded[2:]
        if header_word is not None:
            match = re.match(rf"{_FULL_PROGRESS_SECRET_HEADER_NAME}[ \t]*:[ \t]*", header, re.IGNORECASE)
            if match is not None:
                h_start, h_end, h_raw, h_decoded, _ = header_word
                prefix_length = len(h_decoded) - len(header) + match.end()
                replacements.append((
                    h_start, h_end,
                    _mask_full_progress_header_word(h_raw, h_decoded, prefix_length),
                ))
                header_starts.add(h_start)
            continue
        if len(decoded) > 2 and decoded[:2] in {"-u", "-U", "-b"}:
            if _is_full_progress_cli_secret_flag(decoded[:2], decoded[2:]):
                value_start = 2 + int(raw[0:1] in {"'", '"'})
                prefix = raw[:value_start]
                if raw[0:1] in {"'", '"'} and raw[-1:] == raw[0:1]:
                    replacement = f"{prefix}{_FULL_PROGRESS_REDACTED}{raw[-1]}"
                else:
                    replacement = prefix + _mask_full_progress_shell_word(raw[value_start:])
                replacements.append((start, end, replacement))
            continue
        assignment_name, assignment_equals, _assignment_value = decoded.partition("=")
        if (
            assignment_equals
            and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", assignment_name)
            and _is_full_progress_secret_env_name(assignment_name)
        ):
            replacements.append(
                (start, end, _mask_full_progress_inline_cli_value(raw))
            )
            continue

        flag, equals, inline_value = decoded.partition("=")
        if equals:
            if _is_full_progress_cli_secret_flag(flag, inline_value):
                replacements.append(
                    (start, end, _mask_full_progress_inline_cli_value(raw))
                )
            continue

        if index + 1 >= len(words):
            continue
        next_start, next_end, next_raw, next_value, next_segment = words[index + 1]
        if next_segment != segment:
            continue
        if _is_full_progress_cli_secret_flag(decoded, next_value):
            replacements.append(
                (next_start, next_end, _mask_full_progress_shell_word(next_raw))
            )
            consumed_index = index + 1

    # Only replacements produced above may skip the quote-damaging assignment
    # rescan. Literal masks in caller input never establish that ownership.
    placeholder_prefix = "__HERMES_FULL_MASKED_WORD_"
    while placeholder_prefix in value:
        placeholder_prefix = "_" + placeholder_prefix
    redacted = value
    for start, end, replacement in reversed(replacements):
        if protected is not None:
            from agent.redact import redact_registered_vault_values, redact_sensitive_text

            placeholder = f"{placeholder_prefix}{len(protected)}__"
            # Header syntax has no remaining value for the generic auth parser,
            # which could otherwise consume the next option as a credential.
            # Still mask any registered secret in the retained name/separator.
            protected[placeholder] = (
                redact_registered_vault_values(replacement)
                if start in header_starts
                else redact_sensitive_text(replacement, force=True, code_file=True)
            )
            replacement = placeholder
        redacted = redacted[:start] + replacement + redacted[end:]
    return redacted


def _redact_full_progress_query(query: str) -> str:
    """Mask every value in an already-isolated full-progress query string."""
    fields = []
    for field in query.split("&") if query else []:
        if "=" not in field:
            # A flag-style parameter has no value to expose.
            fields.append(field)
            continue
        key, _, _value = field.partition("=")
        fields.append(f"{key}={_FULL_PROGRESS_REDACTED}")
    return "&".join(fields)


def _redact_full_progress_url(match: re.Match) -> str:
    """Strictly mask credentials in one embedded RFC-scheme URL.

    Every query value and the complete fragment are masked irrespective of
    parameter name.  Userinfo is replaced as a unit, while scheme, host, port,
    path, and query keys remain useful for debugging.  Parsing failures fail
    closed for the matched URL rather than returning possible credentials.
    """
    from urllib.parse import urlsplit, urlunsplit

    from agent.redact import redact_sensitive_text

    raw_url = match.group(0)
    try:
        parts = urlsplit(raw_url)
        netloc = parts.netloc
        if "@" in netloc:
            _, _, hostinfo = netloc.rpartition("@")
            netloc = f"{_FULL_PROGRESS_REDACTED}@{hostinfo}"

        # Userinfo, query values, and fragments are handled structurally.  Keep
        # the authoritative generic scanner on the path too so a vendor-shaped
        # credential used as a path segment does not become a regression while
        # URL spans are protected from header/config false positives.
        path = redact_sensitive_text(parts.path, force=True)
        query = _redact_full_progress_query(parts.query)
        fragment = _FULL_PROGRESS_REDACTED if parts.fragment else ""
        return urlunsplit((parts.scheme, netloc, path, query, fragment))
    except (TypeError, ValueError):
        return _FULL_PROGRESS_REDACTED


def _redact_full_progress_query_only(match: re.Match) -> str:
    """Mask a standalone OAuth/API query fragment such as ``?code=...``."""
    query = _redact_full_progress_query(match.group("query"))
    fragment = f"#{_FULL_PROGRESS_REDACTED}" if match.group("fragment") else ""
    return f"?{query}{fragment}"


def _redact_full_progress_relative_url(match: re.Match) -> str:
    """Mask credentials in relative and network-path URL references."""
    prefix = match.group("prefix")
    if prefix.startswith("//"):
        slash = prefix.find("/", 2)
        authority_end = len(prefix) if slash < 0 else slash
        authority = prefix[2:authority_end]
        if "@" in authority:
            userinfo, _, hostinfo = authority.rpartition("@")
            if ":" in userinfo:
                username, _, _password = userinfo.partition(":")
                masked_userinfo = f"{username}:{_FULL_PROGRESS_REDACTED}"
            else:
                masked_userinfo = _FULL_PROGRESS_REDACTED
            prefix = f"//{masked_userinfo}@{hostinfo}{prefix[authority_end:]}"
    raw_query = match.group("query")
    query = (
        f"?{_redact_full_progress_query(raw_query)}"
        if raw_query is not None
        else ""
    )
    groups = match.groupdict()
    raw_fragment = groups.get("fragment") or groups.get("fragment_only")
    fragment = f"#{_FULL_PROGRESS_REDACTED}" if raw_fragment else ""
    return f"{prefix}{query}{fragment}"


def _redact_full_progress_headers(value: str) -> str:
    """Fully mask sensitive string/list header lines."""
    return _FULL_PROGRESS_HEADER_LINE_RE.sub(
        lambda match: (
            f"{match.group('indent')}{match.group('name')}{match.group('sep')}"
            f"{_FULL_PROGRESS_REDACTED}"
        ),
        value,
    )


def _redact_full_progress_url_spans(value, pattern, redact_url, *, scan_gaps=False):
    """Bound URL matches by existing shell-word spans, then splice source bytes.

    URI apostrophes (including escaped/concatenated spellings) are data inside
    a URL. A whole-word shell quote remains outside its replacement. URI
    punctuation stays inside the span; only a trailing semicolon before
    whitespace or end-of-input delimits a command. No command is re-quoted.
    """
    from agent.redact import redact_sensitive_text

    words = _full_progress_shell_words(value)
    redacted_parts = []
    previous_end = 0
    while match := pattern.search(value, previous_end):
        end = match.end()
        for start, word_end, raw, _decoded, _segment in words:
            if start <= match.start() < word_end:
                opener = raw[:1]
                if opener not in {"'", '"'}:
                    opener = value[match.start() - 1 : match.start()]
                if opener in {"'", '"'} and raw[-1:] == opener:
                    end = min(end, word_end - 1)
                elif (
                    word_end < len(value) and value[word_end] == ";"
                    and (word_end + 1 == len(value) or value[word_end + 1].isspace())
                ):
                    end = min(end, word_end)
                break
        bounded = pattern.match(value, match.start(), end)
        if bounded is not None:
            match = bounded
        gap = value[previous_end : match.start()]
        redacted_parts.append(redact_sensitive_text(gap, force=True) if scan_gaps else gap)
        redacted_parts.append(redact_url(match))
        previous_end = match.end()
    gap = value[previous_end:]
    redacted_parts.append(redact_sensitive_text(gap, force=True) if scan_gaps else gap)
    return "".join(redacted_parts)


def _redact_full_progress_string(value: str) -> str:
    """Apply force-enabled redaction plus full-mode's strict string scanners."""
    from agent.redact import redact_sensitive_text

    # Preserve shell-word boundaries before the generic scanner can consume an
    # opening quote from assignments such as NAME='value with spaces'. This
    # masks credential CLI flags and secret ENV assignments on the original
    # byte layout. The later pass catches options exposed when native scanners
    # consume an opening quote, e.g. app.password="x --passphrase opaque".
    protected_shell_words: dict[str, str] = {}
    value = _redact_full_progress_cli_credentials(value, protected=protected_shell_words)

    # Keep URL spans out of the generic text scanner: RFC permits schemes such
    # as ``authorization://``, which the ordinary header regex would otherwise
    # misclassify and corrupt before the strict URL scanner can parse it.
    redacted = _redact_full_progress_url_spans(
        value, _FULL_PROGRESS_URL_RE, _redact_full_progress_url, scan_gaps=True,
    )
    redacted = _redact_full_progress_url_spans(
        redacted, _FULL_PROGRESS_RELATIVE_URL_RE, _redact_full_progress_relative_url,
    )
    redacted = _redact_full_progress_url_spans(
        redacted, _FULL_PROGRESS_BARE_RELATIVE_URL_RE, _redact_full_progress_relative_url,
    )
    # Run strict header masking after the generic pass.  Otherwise the generic
    # API-header scanner can consume a closing quote from the ``***`` value we
    # just produced.  A header whose value was itself a URL is now masked as a
    # whole, while ``authorization://`` remains protected by the ``://`` guard.
    redacted = _redact_full_progress_headers(redacted)
    for placeholder, masked_word in protected_shell_words.items():
        redacted = redacted.replace(placeholder, masked_word)
    redacted = _redact_full_progress_cli_credentials(redacted)
    return _redact_full_progress_url_spans(
        redacted, _FULL_PROGRESS_QUERY_ONLY_RE, _redact_full_progress_query_only,
    )


def _is_full_progress_secret_key(key: str) -> bool:
    """Return whether an argument key denotes credential-bearing content."""
    # Split separators and camelCase before comparing semantic key components.
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    parts = [
        part
        for part in re.split(r"[^a-z0-9]+", separated.casefold())
        if part
    ]
    compact = "".join(parts)
    # These exact metadata names describe a value, not a credential container.
    # Their contents still recurse through the forced string/secret scanners.
    if compact in {"author", "tokencount", "signaturealgorithm"}:
        return False
    exact = re.sub(r"[^a-z0-9]+", "_", separated.casefold()).strip("_")
    return bool(
        exact in _FULL_PROGRESS_EXACT_SECRET_KEYS
        or set(parts) & _FULL_PROGRESS_SECRET_KEY_PARTS
        or "dsn" in parts
        or any(
            marker in compact
            for marker in _FULL_PROGRESS_CONNECTION_KEY_MARKERS
        )
        or any(name in compact for name in _FULL_PROGRESS_SECRET_KEY_PARTS)
        or compact in _FULL_PROGRESS_SECRET_COMPACT_KEYS
        or compact.endswith(("sshkey", "recoverykey"))
        or any(
            name not in {"key", "sshkey", "recoverykey"} and name in compact
            for name in _FULL_PROGRESS_SECRET_COMPACT_KEYS
        )
    )


def _is_full_progress_secret_env_name(name: str) -> bool:
    """Classify shell ENV assignment names more conservatively than JSON keys."""
    normalized = name.strip().upper()
    return _is_full_progress_secret_key(name) or normalized.endswith("_KEY")


def _redact_full_progress_args(value: Any) -> Any:
    """Copy and chat-strictly redact a complete full-mode argument structure.

    Dicts, lists, and tuples are recursively rebuilt; caller-owned tool args
    are never mutated.  Dict keys are themselves passed through the strict
    string redactor.  If two distinct keys collapse to the same redacted key,
    a deterministic collision marker keeps both fields instead of silently
    overwriting one.  Values under secret-like keys are fully masked before
    JSON serialization, preventing quote escaping from exposing a suffix.
    """
    if isinstance(value, dict):
        redacted_dict: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            original_key = str(raw_key)
            safe_key = _redact_full_progress_string(original_key)
            if safe_key in redacted_dict:
                collision = 2
                base_key = safe_key
                while safe_key in redacted_dict:
                    safe_key = f"{base_key} «redacted-key-collision-{collision}»"
                    collision += 1
            redacted_dict[safe_key] = (
                _FULL_PROGRESS_REDACTED
                if _is_full_progress_secret_key(original_key)
                else _redact_full_progress_args(raw_value)
            )
        return redacted_dict
    if (
        isinstance(value, (list, tuple)) and len(value) == 2
        and isinstance(value[0], str) and ":" not in value[0]
        and _is_full_progress_secret_key(value[0])
    ):
        pair = [_redact_full_progress_string(value[0]), _FULL_PROGRESS_REDACTED]
        return tuple(pair) if isinstance(value, tuple) else pair
    if isinstance(value, list):
        return [_redact_full_progress_args(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_full_progress_args(item) for item in value)
    if isinstance(value, str):
        return _redact_full_progress_string(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    # ``json.dumps(default=str)`` would stringify unknown objects only after the
    # safety boundary.  Convert and redact them here so no raw representation
    # can enter the queue.
    return _redact_full_progress_string(str(value))

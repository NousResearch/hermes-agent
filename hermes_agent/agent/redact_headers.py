"""Credential-bearing HTTP header masking for hermes_agent.agent.redact.

API-key style headers (``x-api-key``…) and Cookie / Set-Cookie headers in raw header lines,
curl ``-H`` arguments, mid-line log text and JSON-serialized request dumps. Authorization
headers stay in hermes_agent/agent/redact.py (they carry a scheme word).
"""

import re


# API-key style headers (single opaque value, no scheme word): non-vendor-prefix
# values would otherwise leak when a curl command is echoed into tool output.
SECRET_HEADER_NAME_LIST = ("x-api-key", "x-goog-api-key", "api-key", "apikey", "x-api-token", "x-auth-token", "x-access-token")
_SECRET_HEADER_NAMES = rf"(?:{'|'.join(SECRET_HEADER_NAME_LIST)})"
_SECRET_HEADER_LINE_RE = re.compile(
    rf"(^[ \t]*(?:[<>][ \t]+)?{_SECRET_HEADER_NAMES}[ \t]*:[ \t]*)([^\r\n]+)",
    re.IGNORECASE | re.MULTILINE,
)
_SECRET_HEADER_INLINE_RE = re.compile(
    # ``$`` and ``&``/``;``/``|`` stay in the value (``x-api-key: &opaque…`` / ``a|opaque…`` in a
    # log is one credential) unless whitespace, end of text or another separator follows
    # (``&& ls``, ``; next``, ``||true``). A leading ``<``/``(`` is part of a log value
    # (``<opaque>``); later ``<>()`` are redirections/grouping and end the value.
    rf"({_SECRET_HEADER_NAMES}[ \t]*:[ \t]*)"
    rf"((?:[<(&;|]+(?![\s;&|<>()]|$))?(?:[^\s\"';&|<>()`\\]|[;&|](?![\s;&|<>()]|$))+)",
    re.IGNORECASE,
)
_SECRET_HEADER_DOUBLE_QUOTED_RE = re.compile(
    rf'((?:^|[ \t])(?:-H(?:(?:[ \t]|\\\r?\n)*|=)'
    rf'|--header(?:(?:[ \t]|\\\r?\n)+|=))")'
    rf'({_SECRET_HEADER_NAMES}:[ \t]*)'
    r'((?:\\\r?\n|\\.|[^"\\\r\n])+)(")',
    re.IGNORECASE,
)
_SECRET_HEADER_SINGLE_QUOTED_RE = re.compile(
    rf"((?:^|[ \t])(?:-H(?:(?:[ \t]|\\\r?\n)*|=)"
    rf"|--header(?:(?:[ \t]|\\\r?\n)+|=))')"
    rf"({_SECRET_HEADER_NAMES}:[ \t]*)([^'\r\n]+)(')",
    re.IGNORECASE,
)
_SECRET_HEADER_UNQUOTED_CURL_RE = re.compile(
    rf"((?:^|[ \t])(?:-H(?:(?:[ \t]|\\\r?\n)*|=)"
    rf"|--header(?:(?:[ \t]|\\\r?\n)+|=)){_SECRET_HEADER_NAMES}:)"
    r"((?:\\[^\r\n]|[^ \t\r\n;&|<>()$`\\])+)",
    re.IGNORECASE | re.MULTILINE,
)

# Same secret headers serialized in JSON form (``"x-api-key": "<value>"``). The
# bare-header regex above cannot match this shape — the closing quote sits
# between the name and the colon — and ``_JSON_FIELD_RE``'s key list omits the
# hyphenated header names, so opaque custom/local-backend keys (no vendor prefix
# for ``_PREFIX_RE`` to catch) leak verbatim from serialized request dumps. This
# mirrors the JSON-field redaction shape, so it is applied under the same
# ``code_file`` gate.
_SECRET_HEADER_JSON_RE = re.compile(
    # ``(?:\\.|[^"\\])`` consumes JSON string contents escape-aware, so a value
    # containing an escaped quote (``"ab\"cd"``) is masked whole rather than
    # truncated at the embedded quote (which would leave the tail in cleartext).
    rf'("{_SECRET_HEADER_NAMES}"\s*:\s*")((?:\\[^\r\n]|[^"\\\r\n])+)(")',
    re.IGNORECASE,
)

# Fail closed for a recognized JSON header whose string value is truncated at
# a line boundary. The primary regex above handles all valid JSON strings; this
# fallback is deliberately line-bounded and its alternatives are disjoint so
# malformed diagnostic output cannot trigger unbounded backtracking.
_SECRET_HEADER_JSON_UNTERMINATED_RE = re.compile(
    rf'("{_SECRET_HEADER_NAMES}"\s*:\s*")((?:\\[^\r\n]|[^"\\\r\n])*(?:\\)?)(?=\r?\n|\Z)',
    re.IGNORECASE | re.MULTILINE,
)

# Cookie: / Set-Cookie: headers carry session, auth and CSRF tokens and were
# previously unhandled. The full cookie payload is masked while the header name
# is preserved. A ``name=value`` pair is required so plain prose ("the cookie:
# …") is left untouched.
# Restrict raw Cookie matching to an actual header line. A bounded cookie-name
# class avoids repeatedly scanning whole suffixes for '=' on malformed input.
_COOKIE_HEADER_LINE_RE = re.compile(
    r"(^[ \t]*(?:[<>][ \t]+)?(?:Set-)?Cookie:[ \t]*)"
    r"([!#$%&'*+\-.^_`|~0-9A-Za-z]+=[^\r\n]*)",
    re.IGNORECASE | re.MULTILINE,
)

# Shell commands commonly embed a header in a quoted ``-H`` argument. Separate
# delimiter-specific patterns cannot scan across another candidate opener, so
# repeated malformed fragments remain linear rather than quadratic.
_COOKIE_HEADER_DOUBLE_QUOTED_RE = re.compile(
    r'((?:^|[ \t])(?:-H(?:(?:[ \t]|\\\r?\n)*|=)'
    r'|--header(?:(?:[ \t]|\\\r?\n)+|=))")'
    r'((?:Set-)?Cookie:[ \t]*)'
    r'([!#$%&\'*+\-.^_`|~0-9A-Za-z]+=(?:\\\r?\n|\\.|[^"\\\r\n])*)(")',
    re.IGNORECASE,
)
_COOKIE_HEADER_SINGLE_QUOTED_RE = re.compile(
    r"((?:^|[ \t])(?:-H(?:(?:[ \t]|\\\r?\n)*|=)"
    r"|--header(?:(?:[ \t]|\\\r?\n)+|=))')"
    r"((?:Set-)?Cookie:[ \t]*)([!#$%&'*+\-.^_`|~0-9A-Za-z]+=[^'\r\n]*)(')",
    re.IGNORECASE,
)

# An unquoted ``-H Cookie:name=value`` argument cannot contain whitespace, so
# its token boundary is unambiguous and needs no open-ended delimiter search.
_COOKIE_HEADER_UNQUOTED_CURL_RE = re.compile(
    r"((?:^|[ \t])(?:-H(?:(?:[ \t]|\\\r?\n)*|=)"
    r"|--header(?:(?:[ \t]|\\\r?\n)+|=))"
    r"(?:Set-)?Cookie:[ \t]*)"
    r"([!#$%&'*+\-.^_`|~0-9A-Za-z]+="
    r"(?:\\[^\r\n]|[^ \t\r\n;&|<>()$`\\])*)",
    re.IGNORECASE | re.MULTILINE,
)

# The on-disk request/response dump sink serializes headers as JSON, so cookies
# also appear as ``"Cookie": "<name=value…>"`` — the bare regex above cannot see
# that shape. A ``name=value`` pair is required so non-header ``"Cookie"`` values
# are left untouched. JSON-shaped, so applied under the ``code_file`` gate.
_COOKIE_HEADER_JSON_RE = re.compile(
    # Escape-aware single-group value match (see _SECRET_HEADER_JSON_RE): the
    # whole JSON string value of a "Cookie"/"Set-Cookie" key is masked. A single
    # unambiguous group (no internal "=" split) avoids superlinear backtracking
    # on malformed/unterminated JSON. A JSON key literally named Cookie is a
    # strong-enough header signal that the "name=value" requirement is dropped.
    r'("(?:Set-)?Cookie"\s*:\s*")((?:\\[^\r\n]|[^"\\\r\n])*)(")',
    re.IGNORECASE,
)

_COOKIE_HEADER_JSON_UNTERMINATED_RE = re.compile(
    r'("(?:Set-)?Cookie"\s*:\s*")((?:\\[^\r\n]|[^"\\\r\n])*(?:\\)?)(?=\r?\n|\Z)',
    re.IGNORECASE | re.MULTILINE,
)

# A Cookie/Set-Cookie header in the middle of a line (log records, prose dumps) that the
# line/curl/JSON passes above did not see. Every ``name=value`` pair is masked, not only
# the first; ``Set-Cookie:`` also masks a bare 16+ char token (``Set-Cookie: header`` is prose).
# An apostrophe is a cookie-octet (``sid=pre'opaque``) unless it closes a shell word.
_COOKIE_PAIR = r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+=(?:\"[^\"\r\n]*\"|(?:[^\s;\\\"'`]|'(?![\s'\"`;&|<>()]|$))*)"
_COOKIE_HEADER_INLINE_RE = re.compile(
    rf"(\b(?P<set>Set-)?Cookie:[ \t]*)"
    rf"(?:{_COOKIE_PAIR}(?:;[ \t]*{_COOKIE_PAIR})*|(?(set)[^\s\\\"';&|<>()$`*=]{{16,}}+(?![^\s;])|(?!)))",
    re.IGNORECASE,
)


_MASKED_VALUE_RE = re.compile(r"(?:\*{3}|«redacted(?::[^»\s]*|-secret)»|\[REDACTED\])(?:\s|$)")


def _redact_inline_secret_header(m: re.Match) -> str:
    """Mask an inline header token without stealing an empty curl header's URL."""
    before = m.string[max(0, m.start() - 32):m.start()]
    if re.search(r"(?:-H|--header)(?:[ \t]*|=)[\"']?$", before, re.IGNORECASE):
        return m.group(0)
    return m.group(1) + "***"


def _redact_header_line_value(m: re.Match) -> str:
    """Mask a header line unless it is the continued argument of curl -H."""
    line_start = m.string.rfind("\n", 0, m.start()) + 1
    if line_start:
        previous_start = m.string.rfind("\n", 0, line_start - 1) + 1
        previous = m.string[previous_start:line_start - 1].rstrip("\r")
        if previous.rstrip().endswith("\\") and re.search(
            r"(?:^|[ \t])(?:-H|--header)(?:[ \t]+|=).*\\[ \t]*$",
            previous,
            re.IGNORECASE,
        ):
            return m.group(0)
    # An earlier pass already masked the value: what follows the marker is prose, and
    # re-masking it would make the redactor non-idempotent (``x-api-key: *** see docs``).
    if _MASKED_VALUE_RE.match(m.group(2)):
        return m.group(0)
    return m.group(1) + "***"


def redact_header_credentials(text: str) -> str:
    """Mask secret-header and cookie-header values in raw, curl and inline forms; the JSON
    forms run with the assignment passes in hermes_agent/agent/redact.py."""
    text = _SECRET_HEADER_LINE_RE.sub(
        _redact_header_line_value,
        text,
    )
    for pattern in (
        _SECRET_HEADER_DOUBLE_QUOTED_RE,
        _SECRET_HEADER_SINGLE_QUOTED_RE,
    ):
        text = pattern.sub(
            lambda m: m.group(1) + m.group(2) + "***" + m.group(4),
            text,
        )
    text = _SECRET_HEADER_UNQUOTED_CURL_RE.sub(
        lambda m: m.group(1) + "***",
        text,
    )
    text = _SECRET_HEADER_INLINE_RE.sub(_redact_inline_secret_header, text)

    # Cookie: / Set-Cookie: header values (session/auth/CSRF tokens), whole payload masked.
    # Not gated on ``code_file``: terminal output from ``curl -i`` is scrubbed with
    # code_file=True, and a raw ``Cookie: name=value`` line is a credential wherever it
    # appears, like the secret headers above. The JSON-fixture exemption lives in the
    # assignment passes. IGNORECASE is the filter, so ``CoOkie:`` cannot slip past.
    text = _COOKIE_HEADER_LINE_RE.sub(_redact_header_line_value, text)
    for pattern in (_COOKIE_HEADER_DOUBLE_QUOTED_RE, _COOKIE_HEADER_SINGLE_QUOTED_RE):
        text = pattern.sub(lambda m: m.group(1) + m.group(2) + "***" + m.group(4), text)
    text = _COOKIE_HEADER_UNQUOTED_CURL_RE.sub(lambda m: m.group(1) + "***", text)
    text = _COOKIE_HEADER_INLINE_RE.sub(lambda m: m.group(1) + "***", text)
    return text


# Cookie credentials are bearer credentials, but ``cookie`` stays out of the shared
# secret-keyword set (``cookie_policy: strict``, ``cookies enabled`` are prose). The
# assignment form needs a key that IS ``cookie`` or ends in ``_cookie``; ``mycookiejar=1``,
# ``cookie_policy=x`` and URL query params (``?cookie=``) never match.
_COOKIE_ASSIGN_RE = re.compile(
    r"(?<![\w?&])((?:[A-Za-z0-9]+_)*+cookie\s*=(?!=)\s*)"
    # A joined ``"…"`` piece holds no JSON punctuation, so a JSON string's closing quote never
    # starts one. A bare value ends at its own boundary: quotes, ``,``, whitespace, and a ``;`` /
    # ``&`` that is not followed by another cookie pair / value character.
    r"(?:(?P<q>[\"'])(?P<quoted>(?:(?!(?P=q))[^\\\r\n]|\\.)*)(?P=q)"
    r"(?P<tail>(?:\\.|'[^'\r\n]*+'|\"[^\"\s\\,:{}\[\]]*+\"|[^\s\"'\\;&|<>(),]++)*+)"
    r"|(?P<bare>(?:[^\s\"'`\\;,&|<>()]|&(?![&\s]|$)|;(?=[!#$%'*+\-.^_|~0-9A-Za-z]+=))+))",
    re.IGNORECASE,
)


def cookie_value_is_credential(value: str, *, qualified: bool = False) -> bool:
    """A bare ``cookie`` key is weak, so the value shape decides as it does for ``token=``:
    a ``name=value`` pair is a cookie payload; ``getCookie()``, ``None``, ``$(cat jar)`` and
    ``chocolate`` are code or prose, while an opaque token is masked. A qualified
    ``SESSION_COOKIE`` is as strong as ``SESSION_TOKEN`` and masks any literal value."""
    from hermes_agent.agent.redact import _should_redact_assignment
    if value.startswith("«redacted"):
        return False
    key = "session_token" if qualified else "cookie"
    return ("=" in value and not value.startswith("$")) or _should_redact_assignment(key, value, check_keyword=False)


def json_header_value_is_credential(key: str, value: str) -> bool:
    """Gate for a JSON ``"<header>": "<value>"`` field: cookie fields follow the cookie value
    shape; an API-key header masks every value but a whole ``${API_KEY}`` / env-lookup
    placeholder (no path exemption: ``/abc…`` is a valid key shape)."""
    from hermes_agent.agent.redact import _ENV_LOOKUP_VALUE_RE, _SHELL_VAR_REF
    if "cookie" in key.lower():
        return cookie_value_is_credential(value)
    # An earlier pass's marker keeps its vendor label (``«redacted:ghp_…»``), as main's field pass does.
    return not (value == "***" or value.startswith("«redacted") or _ENV_LOOKUP_VALUE_RE.match(value)
                or re.fullmatch(_SHELL_VAR_REF, value))


def _redact_cookie_assignment(m: "re.Match[str]", mask) -> str:
    """Mask a ``cookie=`` value; a lone quoted value keeps its quotes, and a shell word joining
    quoted and bare parts (``'sid='\\''opaque'``) is masked as one word."""
    joined = m.group("q") is None or m.group("tail")
    value = m.group(0)[len(m.group(1)):] if joined else m.group("quoted")
    if not cookie_value_is_credential(value, qualified="_" in m.group(1)):
        return m.group(0)
    q = "" if joined else m.group("q")
    return f"{m.group(1)}{q}{mask(value)}{q}"


def redact_cookie_assignments(text: str, mask) -> str:
    """``cookie=`` / ``*_COOKIE=`` assignment values (an ``=`` gated assignment pass in
    hermes_agent/agent/redact.py)."""
    return _COOKIE_ASSIGN_RE.sub(lambda m: _redact_cookie_assignment(m, mask), text)

"""Spaced ``key = value`` config-assignment masking for agent.redact (#130827).

INI/TOML files align values with spaces around ``=`` (``password = "hunter2"``), which the
adjacent-only _CFG_DOTTED_RE / _CFG_ANCHORED_RE never matched. The same spacing is the default
style of Python, JS and Go source, and assistant replies carry code, so this form is masked only
for config-shaped values: a quoted literal (spaces inside it included), or an unquoted last token
that looks opaque and is not a code expression (call, subscript, literal keyword, identifier or
attribute path). INI ``password = hunter2`` (short, unquoted, identifier-shaped) stays unmasked.
"""

import functools
import re

_SPACED_EQ_RE = re.compile(r"[ \t]=|=[ \t]")
_SPACED_SEP = r"(?:[ \t]++=(?![=>~])[ \t]*+|=(?![=>~])[ \t]++)"
_QUOTED = r"(?P<q>['\"])(?P<qv>(?:\\.|(?!(?P=q))[^\\\n])++)(?P=q)(?=[ \t]*+(?:[;,#]|//|\r?$))"
# The value must end its line (an INI ``;``/``#`` comment may follow): ``auth = a or b`` is code.
_UNQUOTED = r"(?P<uv>[^\s'\"]++)(?=(?:[ \t]++[#;][^\n]*+)?[ \t]*+\r?$)"
_CODE_KEYWORDS = frozenset({"none", "null", "true", "false", "nil", "undefined"})
_IDENTIFIER_PATH_RE = re.compile(r"[A-Za-z_$][\w$]*+(?:\.[A-Za-z_$][\w$]*+)*+")


@functools.cache
def _patterns() -> tuple["re.Pattern[str]", ...]:
    from agent.redact import _LINE_NUMBER_GUTTER

    # Keys are one possessive run checked by _key_names_secret: the key must reach the separator, so
    # the run is the only candidate, and `[..]*NAME[..]*` backtracking is superlinear on long runs.
    value = rf"{_SPACED_SEP}(?:{_QUOTED}|{_UNQUOTED})"
    sp = r"(?<=[Aa][Pp][Ii]) (?=[Kk][Ee][Yy])"  # _SECRET_CFG_NAMES' ``api key``: the only space a key holds
    dotted = re.compile(
        rf"(?<![A-Za-z0-9_.\-])(?!(?<=[Aa][Pp][Ii] )[Kk][Ee][Yy])(?P<key>(?P<name>(?:[A-Za-z0-9_\-]|{sp})*+\.(?:[A-Za-z0-9_.\-]|{sp})*+)){value}",
        re.MULTILINE,
    )
    anchored = re.compile(
        rf"(?P<key>^[ \t]*{_LINE_NUMBER_GUTTER}(?:export[ \t]+)?(?P<name>(?:[A-Za-z0-9_.\-]|{sp})++)){value}",
        re.IGNORECASE | re.MULTILINE,
    )
    return dotted, anchored


def _key_names_secret(name: str, dotted: bool) -> bool:
    """The keyword placement the adjacent _CFG_DOTTED_RE / _CFG_ANCHORED_RE keys require: a bare key
    is dot-free around a keyword (``api.key`` carries its own dot); a dotted key has the keyword after
    a non-empty first segment, or before a later dot that is not the last character."""
    from agent.redact import _CFG_SECRET_WORD_RE

    if not dotted:  # only ``api.key`` / ``api key`` puts a non-word character in a bare key
        seps = [i for i, c in enumerate(name) if c in ". "]
        if seps and (len(seps) > 1 or name[max(seps[0] - 3, 0):seps[0]].lower() != "api"
                     or name[seps[0] + 1:seps[0] + 4].lower() != "key"):
            return False  # checked directly: a fullmatch backtracks per keyword on ``password…password.``
        return bool(_CFG_SECRET_WORD_RE.search(name))
    first_dot, last_dot = name.find("."), name.rfind(".", 0, len(name) - 1)
    return bool((first_dot > 0 and _CFG_SECRET_WORD_RE.search(name, first_dot + 1))
                or (last_dot > 0 and _CFG_SECRET_WORD_RE.search(name, 0, last_dot)))


def _is_code_expression(key: str, value: str) -> bool:
    """``getpass.getpass(...)``, ``password``, ``self.x``, ``None``, ``secret;``, ``0``: code, not
    config. An unquoted value is masked only when it looks opaque; an opaque-length identifier
    (``AUTH_TOKEN_ENV``) still needs a digit (``a1b2c3d4e5f6a7b8c9d0``) and never contains a dot."""
    from agent.redact import _looks_like_opaque_credential

    folded = value.casefold()
    if any(c in value for c in "([{") or value.endswith(";") or folded in _CODE_KEYWORDS:
        return True
    if folded == re.split(r"[^\w$-]", key)[-1].casefold():
        return True
    if not _looks_like_opaque_credential(value):
        return True
    return bool(_IDENTIFIER_PATH_RE.fullmatch(value)) and ("." in value or not any(c.isdigit() for c in value))


def _quoted_value_is_secret(key: str, value: str) -> bool:
    """A quoted literal under a key NAMING a credential (``db_password``, ``api.token``); under a
    key that only mentions one (``auth_type = "oauth"``, ``no_token_msg = "..."``) only an
    opaque-looking single token, since those are labels and messages in code."""
    from agent.redact import _is_python_repr_secret_key, _looks_like_opaque_credential

    name = re.split(r"[^\w-]", key)[-1].replace("-", "_")
    if value.casefold() == name.casefold():
        return False
    return _is_python_repr_secret_key(name) or (
        not any(c.isspace() for c in value) and _looks_like_opaque_credential(value))


def redact_spaced_cfg_assignments(text: str, mask) -> str:
    """Mask config secrets written ``key = value``; the adjacent ``key=value`` form is not matched."""
    from agent.redact import _should_redact_assignment

    def _sub(m: "re.Match[str]", dotted: bool) -> str:
        if not _key_names_secret(m.group("name"), dotted):
            return m.group(0)
        key, quoted = m.group("key"), m.group("qv") is not None
        value = m.group("qv") if quoted else m.group("uv")
        if not (_quoted_value_is_secret(key, value) if quoted else not _is_code_expression(key, value)):
            return m.group(0)
        if not _should_redact_assignment(key, value, check_keyword=True):
            return m.group(0)
        start, end = (m.span("qv") if quoted else m.span("uv"))
        return m.group(0)[:start - m.start()] + mask(value) + m.group(0)[end - m.start():]

    if not _SPACED_EQ_RE.search(text):  # the common case: nothing spaced around ``=`` to look at
        return text
    dotted, anchored = _patterns()
    text = dotted.sub(lambda m: _sub(m, True), text)
    return anchored.sub(lambda m: _sub(m, False), text)

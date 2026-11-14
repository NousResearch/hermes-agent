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

_SPACED_SEP = r"(?:[ \t]++=(?![=>~])[ \t]*+|=(?![=>~])[ \t]++)"
_QUOTED = r"(?P<q>['\"])(?P<qv>(?:\\.|(?!(?P=q))[^\\\n])++)(?P=q)(?=[ \t]*+(?:[;,#]|//|\r?$))"
# The value must end its line (an INI ``;``/``#`` comment may follow): ``auth = a or b`` is code.
_UNQUOTED = r"(?P<uv>[^\s'\"]++)(?=(?:[ \t]++[#;][^\n]*+)?[ \t]*+\r?$)"
_CODE_KEYWORDS = frozenset({"none", "null", "true", "false", "nil", "undefined"})
_IDENTIFIER_PATH_RE = re.compile(r"[A-Za-z_$][\w$]*+(?:\.[A-Za-z_$][\w$]*+)*+")


@functools.cache
def _patterns() -> tuple["re.Pattern[str]", ...]:
    from agent.redact import _LINE_NUMBER_GUTTER, _SECRET_CFG_NAMES

    value = rf"{_SPACED_SEP}(?:{_QUOTED}|{_UNQUOTED})"
    dotted = re.compile(
        rf"(?<![A-Za-z0-9_.\-])"
        rf"(?P<key>[A-Za-z0-9_\-]++\.[A-Za-z0-9_.\-]*{_SECRET_CFG_NAMES}[A-Za-z0-9_.\-]*+"
        rf"|[A-Za-z0-9_.\-]*{_SECRET_CFG_NAMES}[A-Za-z0-9_.\-]*\.[A-Za-z0-9_.\-]++){value}",
        re.IGNORECASE | re.MULTILINE,
    )
    anchored = re.compile(
        rf"(?P<key>^[ \t]*{_LINE_NUMBER_GUTTER}(?:export[ \t]+)?"
        rf"[A-Za-z0-9_\-]*{_SECRET_CFG_NAMES}[A-Za-z0-9_\-]*){value}",
        re.IGNORECASE | re.MULTILINE,
    )
    return dotted, anchored


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

    def _sub(m: "re.Match[str]") -> str:
        key, quoted = m.group("key"), m.group("qv") is not None
        value = m.group("qv") if quoted else m.group("uv")
        if not (_quoted_value_is_secret(key, value) if quoted else not _is_code_expression(key, value)):
            return m.group(0)
        if not _should_redact_assignment(key, value, check_keyword=True):
            return m.group(0)
        start, end = (m.span("qv") if quoted else m.span("uv"))
        return m.group(0)[:start - m.start()] + mask(value) + m.group(0)[end - m.start():]

    for pattern in _patterns():
        text = pattern.sub(_sub, text)
    return text

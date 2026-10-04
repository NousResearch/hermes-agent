"""Fail-closed redaction for text and structures leaving debug/support flows."""

from __future__ import annotations

import re

from agent.redact import (
    _SENSITIVE_QUERY_PARAMS,
    REDACTION_UNAVAILABLE,
    SECRET_HEADER_NAME_LIST,
    _canonical_url_param_name,
    redact_sensitive_text,
)

_REDACTED = "[REDACTED]"
_EMAIL_ADDRESS_RE = re.compile(
    r"(?<![A-Za-z0-9._%+-])"
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    r"(?![A-Za-z0-9._%+-])"
)

# Support uploads are a strict opt-in surface: agent.redact's query-param names
# plus a few support-only extras, compared by compact name so snake_case,
# kebab-case and camelCase match without widening ordinary tool-output redaction.
_SENSITIVE_URL_PARAM_NAMES = frozenset(
    "".join(ch for ch in name if ch.isalnum()) for name in _SENSITIVE_QUERY_PARAMS
) | {"authtoken", "privatekey", "secretkey", "xamzsecuritytoken", "securitytoken"}
_RAW_URL_PARAM_RE = re.compile(
    r"(?P<sep>[?#&;])(?P<key>[A-Za-z0-9_.~+%\-]+)="
    r"(?P<value>[^?#&;\s\"'<>]*)"
)
# ``%25`` prefixes the double-encoded forms (``%253F``/``%2526``/``%253D``) that a
# URL nested inside an already-encoded redirect value carries.
_ENCODED_URL_SEP = r"%(?:25)?(?:3f|23|26|3b)"
_ENCODED_URL_PARAM_RE = re.compile(
    rf"(?P<sep>{_ENCODED_URL_SEP})"
    rf"(?P<key>(?:(?!%(?:25)?3d|{_ENCODED_URL_SEP})[^\s\"'<>]){{1,160}}?)"
    r"(?P<eq>%(?:25)?3d|=)"
    r"(?P<value>.*?)"
    rf"(?={_ENCODED_URL_SEP}|[\s\"'<>]|$)",
    re.IGNORECASE,
)

_SENSITIVE_HEADER_NAMES = (
    "authorization",
    "proxy-authorization",
    "cookie",
    "set-cookie",
    *SECRET_HEADER_NAME_LIST,
)
_HEADER_NAME_PATTERN = "|".join(re.escape(name) for name in _SENSITIVE_HEADER_NAMES)
_QUOTED_HEADER_RE = re.compile(
    rf"(?P<prefix>['\"]?(?:{_HEADER_NAME_PATTERN})['\"]?\s*[:=]\s*)"
    r"(?P<quote>['\"])(?P<value>.*?)(?P=quote)",
    re.IGNORECASE,
)
# A key may carry inner colons (``sk-abc:xyz``, ``user:pass``); a trailing colon
# is prose punctuation (``see: docs``). A letters-only word (``docs``, ``Note``,
# ``HTTP``) is prose rather than a key: keys mix case, digits or symbols. The
# word test is case-sensitive even inside the IGNORECASE regexes below. A
# single-case run of 20+ letters is no English word, so it is a key.
_KEY_CHARS = r"A-Za-z0-9._~+/\-"
_KEY = rf"[{_KEY_CHARS}]+(?::[{_KEY_CHARS}]+)*=*(?![{_KEY_CHARS}=]|:[{_KEY_CHARS}])"
_NOT_WORD = rf"(?!(?-i:[A-Z]?[a-z]{{1,19}}|[A-Z]{{1,19}})(?![{_KEY_CHARS}=]|:[{_KEY_CHARS}]))"
# agent.redact masks the scheme of ``x-api-key: Basic <key>`` to ``***`` and
# keeps the key, so ``***`` is a scheme too and the key is the value when it
# is key-shaped (not a word; 4+ chars before any colon, or 6+ mixing letters
# and digits) or a short word ending the header
# (``Basic short``). Prose after a masked value stays.
_SCHEME = (
    r"(?:Bearer|Basic|Digest)\s+|\*\*\*[ \t]+"
    rf"(?={_NOT_WORD}(?=[^\s:]{{4}}|(?=\S*[A-Za-z])(?=\S*[0-9])\S{{6}}){_KEY}|[A-Za-z]{{4,7}}[ \t]*(?:[\r\n'\",;}}\]]|$))"
    rf"(?!(?:{_HEADER_NAME_PATTERN})\b\s*[:=])"
)
_BARE_HEADER_RE = re.compile(
    rf"(?P<prefix>\b(?:{_HEADER_NAME_PATTERN})\b\s*[:=]\s*)"
    rf"(?P<value>(?>(?:{_SCHEME})?)(?!\[REDACTED\])[^\s,}}\]]+)",
    re.IGNORECASE,
)

_SECRET_ARG_FLAG = (
    r"--(?:api[-_]?key|access[-_]?token|refresh[-_]?token|id[-_]?token|"
    r"auth[-_]?token|client[-_]?secret|private[-_]?key|secret[-_]?key|"
    r"password|passwd|credential|token|secret)"
)
_QUOTED_ARGV_RE = re.compile(
    rf"(?P<flag_quote>['\"])(?P<flag>{_SECRET_ARG_FLAG})(?P=flag_quote)"
    r"(?P<sep>\s*,\s*)"
    r"(?P<value_quote>['\"])(?P<value>.*?)(?P=value_quote)",
    re.IGNORECASE,
)
_ARG_EQUALS_RE = re.compile(
    rf"(?P<flag>{_SECRET_ARG_FLAG})(?P<sep>\s*=\s*)"
    r"(?P<quote>['\"]?)(?!\[REDACTED\])(?P<value>[^\s,}\]'\"]+)(?P=quote)",
    re.IGNORECASE,
)
_ARG_OPERAND_RE = re.compile(
    rf"(?P<flag>{_SECRET_ARG_FLAG})(?P<sep>\s+)"
    r"(?P<quote>['\"]?)(?!\[REDACTED\])(?P<value>[^\s,}\]'\"]+)(?P=quote)",
    re.IGNORECASE,
)
_HEADER_ARGV_RE = re.compile(
    r"(?P<head_quote>['\"])--header(?P=head_quote)\s*,\s*"
    rf"(?P<name_quote>['\"])(?P<name>{_HEADER_NAME_PATTERN})(?P=name_quote)\s*,\s*"
    r"(?:(?P<scheme_quote>['\"])(?P<scheme>Bearer|Basic|Digest)(?P=scheme_quote)\s*,\s*)?"
    r"(?P<value_quote>['\"])(?P<value>.*?)(?P=value_quote)",
    re.IGNORECASE,
)
_PLAIN_HEADER_ARG_RE = re.compile(
    rf"(?P<prefix>--header\s+(?:{_HEADER_NAME_PATTERN})\s+)"
    rf"(?>(?:{_SCHEME})?)(?!\[REDACTED\])(?P<value>[^\s,}}\]]+)",
    re.IGNORECASE,
)
_SUPPORT_BEARER_RE = re.compile(
    # Free text has no header boundary to stop at: consume the whole value,
    # inner ``=`` padding included (``abcd1234==x``), not just a key-shaped prefix.
    rf"(?P<prefix>\bBearer\s+|\bBasic\s+{_NOT_WORD})(?=[^\s]{{8}})"
    rf"[{_KEY_CHARS}=]+(?::[{_KEY_CHARS}=]+)*",
    re.IGNORECASE,
)

def _decoded_compact_name(value: object) -> str:
    return "".join(ch for ch in _canonical_url_param_name(str(value or "")) if ch.isalnum())


def _redact_url_params(text: str) -> str:
    def _raw_sub(match: re.Match[str]) -> str:
        if _decoded_compact_name(match.group("key")) not in _SENSITIVE_URL_PARAM_NAMES:
            return match.group(0)
        return f"{match.group('sep')}{match.group('key')}=***"

    def _encoded_sub(match: re.Match[str]) -> str:
        if _decoded_compact_name(match.group("key")) not in _SENSITIVE_URL_PARAM_NAMES:
            return match.group(0)
        eq = match.group("eq")
        star = f"{eq[:-2]}2A" if eq != "=" else "*"  # %2A, or %252A when double-encoded
        return f"{match.group('sep')}{match.group('key')}{eq}{star * 3}"

    # Running until stable handles a raw URL nested inside another query value
    # without recursive parsing. The pass cap keeps malformed input bounded.
    for _ in range(4):
        redacted = _ENCODED_URL_PARAM_RE.sub(_encoded_sub, _RAW_URL_PARAM_RE.sub(_raw_sub, text))
        if redacted == text:
            break
        text = redacted
    return text


def _redact_headers_and_argv(text: str) -> str:
    text = _QUOTED_HEADER_RE.sub(
        lambda match: f"{match.group('prefix')}{match.group('quote')}{_REDACTED}{match.group('quote')}",
        text,
    )
    text = _BARE_HEADER_RE.sub(lambda match: f"{match.group('prefix')}{_REDACTED}", text)
    text = _HEADER_ARGV_RE.sub(
        lambda match: (
            f"{match.group('head_quote')}--header{match.group('head_quote')}, "
            f"{match.group('name_quote')}{match.group('name')}{match.group('name_quote')}, "
            + (
                f"{match.group('scheme_quote')}{match.group('scheme')}{match.group('scheme_quote')}, "
                if match.group("scheme")
                else ""
            )
            + f"{match.group('value_quote')}{_REDACTED}{match.group('value_quote')}"
        ),
        text,
    )
    text = _QUOTED_ARGV_RE.sub(
        lambda match: (
            f"{match.group('flag_quote')}{match.group('flag')}{match.group('flag_quote')}"
            f"{match.group('sep')}{match.group('value_quote')}{_REDACTED}{match.group('value_quote')}"
        ),
        text,
    )
    text = _ARG_EQUALS_RE.sub(
        lambda match: f"{match.group('flag')}{match.group('sep')}{match.group('quote')}{_REDACTED}{match.group('quote')}",
        text,
    )
    text = _ARG_OPERAND_RE.sub(
        lambda match: f"{match.group('flag')}{match.group('sep')}{match.group('quote')}{_REDACTED}{match.group('quote')}",
        text,
    )
    text = _PLAIN_HEADER_ARG_RE.sub(
        lambda match: f"{match.group('prefix')}{_REDACTED}", text
    )
    return _SUPPORT_BEARER_RE.sub(
        lambda match: f"{match.group('prefix')}{_REDACTED}", text
    )


def redact_debug_support_text(value: object, *, max_chars: int | None = None) -> str:
    """Strictly scrub one debug/support value, then apply an optional character cap.

    This is deliberately stricter than ordinary model/tool output: support text is
    leaving the process for a human or remote diagnostics store. Any redactor
    failure returns the shared fail-closed sentinel rather than the original.
    """
    text = "" if value is None else str(value)
    if not text:
        return text
    try:
        text = redact_sensitive_text(text, force=True, redact_url_credentials=True)
        text = _redact_url_params(text)
        text = _redact_headers_and_argv(text)
        text = _EMAIL_ADDRESS_RE.sub("[REDACTED_EMAIL]", text)
    except Exception:
        return REDACTION_UNAVAILABLE
    return text[:max_chars] if max_chars is not None else text

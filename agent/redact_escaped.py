"""``NAME = \\"value\\"`` assignments inside a JSON string (escaped quotes).

The ENV assignment passes take an unquoted value as ``\\S+``. Inside JSON text a quoted
assignment arrives with escaped quotes (``"const SECRET = \\"v\\"."``), so the quote group
matched nothing, ``\\S+`` ran over the escaped closing quote and the string's own closing
quote, and the masked output was no longer valid JSON (#153079). This pass masks the value
between the escaped quotes and keeps both quotes; the ENV regexes skip a value that starts
with an escaped quote, so the pass is not undone by them.
"""

from __future__ import annotations

import re

# The value stops at whitespace (as ``\\S+`` did), at the escaped closing quote, or at a bare
# ``"`` (which always ends the enclosing JSON string). Any other
# escape (``\\\\``, ``\\n``) stays inside the value, so no secret tail survives after it.
# An escaped backslash before an escaped quote (``\\\\\\"``, a quote the decoded value itself
# escapes) stays inside the value: it is tried before the pair rule, so the match only closes on an
# escaped quote whose backslash starts a pair. Over-including there masks more, never less.
_VALUE = r"""\s*=\s*\\(['"])((?:[^\s\\"]|\\\\\\\2|\\(?!\2)\S)+)(\\\2)?"""
ESCAPED_ENV_ASSIGN_RE = re.compile(
    r"([A-Z0-9_]{0,50}(?:API_?KEY|KEY|TOKEN|SECRET|PASSWORD|PASSWD|PASS|PW|CREDENTIAL|AUTH)[A-Z0-9_]{0,50})"
    + _VALUE)
ESCAPED_ENV_ASSIGN_LOWER_RE = re.compile(
    r"(?<![a-z0-9_])([a-z0-9_]+(?:_|^)(?:key|pass|pw|token|secret|password|passwd|credential|auth)"
    r"(?=[^a-z0-9_]|$))" + _VALUE,
    re.IGNORECASE,
)


def redact_escaped_assignments(text: str, mask, should_redact) -> str:
    """Mask ``NAME = \\"value\\"`` values; ``should_redact(name, value)`` is the shared gate."""
    if "\\" not in text or "=" not in text:
        return text

    def _sub(m: re.Match) -> str:
        name, quote, value, closing = m.group(1), m.group(2), m.group(3), m.group(4) or ""
        if not should_redact(name, value):
            return m.group(0)
        masked = mask(value)
        # A head/tail cut through an escape (``ab\\`` + ``...``) would leave invalid JSON.
        return f"{name}=\\{quote}{'***' if chr(92) in masked else masked}{closing}"

    text = ESCAPED_ENV_ASSIGN_RE.sub(_sub, text)
    if "://" not in text:  # same URL carve-out as the lowercase ENV pass
        text = ESCAPED_ENV_ASSIGN_LOWER_RE.sub(_sub, text)
    return text

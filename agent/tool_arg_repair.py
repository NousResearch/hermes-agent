"""Conservative repair of tool-call argument JSON before it is declared invalid.

The dominant malformation is not a schema error but a *transport* one: a model emits a
tool call whose string values contain literal line breaks, tabs or other control
characters. Those are illegal inside a JSON string, so ``json.loads`` fails and the whole
turn is retried — three times, with no feedback — even though the intended value is
unambiguous. Escaping those characters is semantics-preserving: the model meant a newline
inside the string, and ``\\n`` is that newline.

Anything else (a truncated object, a missing comma, unbalanced braces) is NOT guessed at.
The caller gets a diagnosis naming the character position and the enclosing field so the
model can fix it on the retry instead of reproducing the same bytes.
"""

from __future__ import annotations

import json
import re

# The last "key": seen before a character position is the innermost field being written
# in practice: models write nested objects in key order, and the error lands in the value
# that follows it.
_KEY_BEFORE = re.compile(r'"((?:[^"\\\n]|\\.){1,80})"\s*:')

_CONTROL_ESCAPES = {"\n": "\\n", "\r": "\\r", "\t": "\\t", "\b": "\\b", "\f": "\\f"}


def escape_control_chars_in_strings(raw: str) -> tuple[str, int]:
    """Escape raw control characters that appear inside JSON string literals.

    Returns ``(candidate, repaired_count)``. The scan is quote- and escape-aware, so
    control characters outside strings (where they are legal whitespace) and existing
    escapes are left exactly as they are.
    """
    if not isinstance(raw, str) or not raw:
        return raw, 0
    out: list[str] = []
    in_string = False
    escaped = False
    repaired = 0
    for ch in raw:
        if not in_string:
            if ch == '"':
                in_string = True
            out.append(ch)
            continue
        if escaped:                       # this character was already part of an escape
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            continue
        if ch == '"':
            out.append(ch)
            in_string = False
            continue
        if ch in _CONTROL_ESCAPES:
            out.append(_CONTROL_ESCAPES[ch])
            repaired += 1
            continue
        if ord(ch) < 0x20:
            out.append("\\u%04x" % ord(ch))
            repaired += 1
            continue
        out.append(ch)
    return "".join(out), repaired


def _enclosing_field(raw: str, pos: int) -> str | None:
    """The last JSON key seen before *pos*, or ``None``."""
    best = None
    for match in _KEY_BEFORE.finditer(raw, 0, max(0, min(pos, len(raw)))):
        best = match.group(1)
    return best


def diagnose_invalid_arguments(raw: str, exc: json.JSONDecodeError) -> str:
    """One-line, actionable description of why *raw* is not valid JSON."""
    field = _enclosing_field(raw, exc.pos)
    where = f" at char {exc.pos} (line {exc.lineno}, column {exc.colno})"
    if field:
        where += f' inside the value of field "{field}"'
    diagnosis = f"{exc.msg}{where}"
    if any(ch in (raw or "") for ch in ("\n", "\r", "\t")):
        diagnosis += (". A string value contains a literal line break or tab: write it as "
                      "\\n, \\r or \\t inside the JSON string")
    return diagnosis


def repair_tool_arguments(raw: str) -> tuple[str | None, str | None]:
    """``(repaired, None)`` when a conservative repair parses, else ``(None, diagnosis)``.

    The repair only escapes control characters inside string literals. It never
    inserts, drops or reorders anything else, so a repaired payload means exactly what
    the model wrote.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None, "arguments were empty"
    candidate, repaired = escape_control_chars_in_strings(raw)
    if repaired:
        try:
            json.loads(candidate)
        except json.JSONDecodeError:
            pass
        else:
            return candidate, None
    try:
        json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, diagnose_invalid_arguments(raw, exc)
    return raw, None

"""Double-escape repair helpers for tool-call arguments (#100730).

One extra JSON-escape layer over a tool-call argument blob leaves a
specific signature after the blob still parses cleanly: an intended
backslash + quote (bytes 5c 22) decodes to three backslashes + quote
(5c 5c 5c 22), and intended doubled backslashes decode to four. The
valid-parse fast path in ``_repair_tool_call_arguments`` passes these
through untouched, silently corrupting backslash-heavy writes (bash
ANSI-C quoting, PowerShell, Windows paths).
"""

from __future__ import annotations

import re

# Signature: a run of three or more backslashes immediately followed by a
# double-quote character (5c5c5c22 present in the decoded value).
_SIGNATURE_RE = re.compile("\\\\{3,}\"")

_BS = chr(92)
_Q = chr(34)
_TRIPLE_BS_Q = _BS + _BS + _BS + _Q
_DOUBLE_BS = _BS + _BS


def value_has_double_escape_signature(value: object) -> bool:
    """True when a decoded string value carries the double-escape signature."""
    return isinstance(value, str) and bool(_SIGNATURE_RE.search(value))


def args_look_double_escaped(parsed: object) -> bool:
    """True when any top-level string value of parsed arguments carries it."""
    return isinstance(parsed, dict) and any(
        value_has_double_escape_signature(v)
        for v in parsed.values()
        if isinstance(v, str)
    )


def remove_extra_escape_layer(value: str) -> str:
    """Invert one extra JSON string-escape pass over a value.

    The extra layer turned every 5c into 5c5c and every quote into 5c22;
    the inverse maps 5c5c5c22 back to 5c22 and 5c5c back to 5c with a
    single greedy left-to-right scan.
    """
    out: list[str] = []
    i = 0
    n = len(value)
    while i < n:
        ch = value[i]
        if ch == _BS:
            if value[i : i + 4] == _TRIPLE_BS_Q:
                out.append(_BS)
                out.append(_Q)
                i += 4
                continue
            if value[i : i + 2] == _DOUBLE_BS:
                out.append(_BS)
                i += 2
                continue
        out.append(ch)
        i += 1
    return "".join(out)

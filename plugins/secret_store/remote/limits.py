"""The secrets plane's limits on a write, checked before the import stages anything (contract §6).

Mirrors gateway-gateway ``src/secretsPlane`` (``types.ts`` / ``patch.ts`` / ``router.ts``): an env
value is a string of at most :data:`MAX_ENTRY_BYTES` UTF-8 bytes; an auth record is a JSON object
whose ``JSON.stringify`` is at most :data:`MAX_ENTRY_BYTES` UTF-8 bytes; a level holds at most
:data:`MAX_ENTRIES_PER_LEVEL` entries (env + auth); a request body is at most
:data:`MAX_BODY_BYTES` bytes as sent (:func:`plugins.config_backends.remote.client.encode_body`).
A value that cannot be sent at all (a lone surrogate, not UTF-8) is refused too. Each problem is
described by name and number, never by value.
"""
from __future__ import annotations

import json
import math
from decimal import Decimal
from typing import Any

from plugins.config_backends.remote.client import encode_body

MAX_ENTRY_BYTES = 64 * 1024
MAX_ENTRIES_PER_LEVEL = 1000
MAX_BODY_BYTES = 1024 * 1024


def import_body(env: dict[str, Any], records: dict[str, Any]) -> dict[str, Any]:
    """The one-time import's ``PATCH /self`` body for a profile the plane has never stored."""
    return {"expectedProfileVersion": 0, "set": env,
            "auth": {rid: {"expectedVersion": 0, "record": r} for rid, r in records.items()}}


def problems(env: dict[str, Any], records: dict[str, Any], shown) -> list[str]:
    """What the plane would refuse in an import of *env* and *records*: one description each
    (names through *shown*, never a value); [] when it would take them."""
    if not env and not records:
        return []
    out = []
    for name, value in env.items():
        size = _utf8_size(value) if isinstance(value, str) else None
        if size is None:
            out.append(f"{shown(name)} (the value is not UTF-8 text)")
        elif size > MAX_ENTRY_BYTES:
            out.append(f"{shown(name)} (a {size}-byte value; at most {MAX_ENTRY_BYTES})")
    for rid, record in records.items():
        size = _utf8_size(_js_json(record)) if isinstance(record, dict) else None
        if size is None:
            out.append(f"auth record {shown(rid)} (not a JSON object of UTF-8 text)")
        elif size > MAX_ENTRY_BYTES:
            out.append(f"auth record {shown(rid)} ({size} bytes; at most {MAX_ENTRY_BYTES})")
    if len(env) + len(records) > MAX_ENTRIES_PER_LEVEL:
        out.append(f"{len(env) + len(records)} entries (a profile holds at most {MAX_ENTRIES_PER_LEVEL})")
    if not out:
        try:
            size = len(encode_body(import_body(env, records)))
        except (UnicodeEncodeError, ValueError, TypeError):
            size = None
        if size is None:
            out.append("the import request cannot be encoded")
        elif size > MAX_BODY_BYTES:
            out.append(f"the import request would be {size} bytes (at most {MAX_BODY_BYTES}): too much at once")
    return out


def _utf8_size(text: str | None) -> int | None:
    try:
        return None if text is None else len(text.encode("utf-8"))
    except UnicodeEncodeError:
        return None


def _js_json(value: Any) -> str | None:
    """``JSON.stringify`` of *value* once the plane has parsed it (JavaScript numbers), or None
    for what JSON cannot carry."""
    if value is None or isinstance(value, bool):
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return _js_number(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        items = []
        for v in value:
            text = _js_json(v)
            if text is None:
                return None
            items.append(text)
        return "[" + ",".join(items) + "]"
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            text = _js_json(v)
            if not isinstance(k, str) or text is None:
                return None
            parts.append(json.dumps(k, ensure_ascii=False) + ":" + text)
        return "{" + ",".join(parts) + "}"
    return None


def _js_number(x: float) -> str:
    """ECMAScript ``Number::toString`` of *x* as the plane holds it (a double)."""
    if isinstance(x, int) and abs(x) <= 2 ** 53:
        return str(x)
    try:
        x = float(x)
    except OverflowError:
        return "null"
    if not math.isfinite(x):
        return "null"
    if x == 0:
        return "0"
    _sign, digits, exponent = Decimal(repr(abs(x))).as_tuple()
    s = "".join(map(str, digits)).rstrip("0") or "0"
    k = len(s)
    n = int(exponent) + len(digits)                  # x = 0.s × 10^n
    lead = "-" if x < 0 else ""
    if k <= n <= 21:
        return lead + s + "0" * (n - k)
    if 0 < n <= 21:
        return lead + s[:n] + "." + s[n:]
    if -6 < n <= 0:
        return lead + "0." + "0" * -n + s
    e = n - 1
    mantissa = s if k == 1 else s[0] + "." + s[1:]
    return f"{lead}{mantissa}e{'+' if e >= 0 else '-'}{abs(e)}"

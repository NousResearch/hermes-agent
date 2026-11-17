"""JSON as the secrets plane's JavaScript reads and writes it, for :mod:`.limits`.

:func:`parse` is the plane's ``Request.text()`` + ``JSON.parse``: UTF-8 with one leading BOM
dropped, any nesting depth (V8 parses without recursion; a 64 KiB record can nest 32,000 deep), a
number held as a double (:data:`UNHOLDABLE` for one beyond a double, which JavaScript reads as
±Infinity). :func:`stringify` is ``JSON.stringify`` of the parsed value, also at any depth. Both
are iterative, so a deep body is measured rather than raising RecursionError.
"""
from __future__ import annotations

import json
import math
import re
from decimal import Decimal
from json.scanner import NUMBER_RE
from typing import Any

#: A JSON number a JavaScript number cannot hold (the plane reads ±Infinity).
UNHOLDABLE = object()
_WS = re.compile(r"[ \t\n\r]*")
_scanstring = json.decoder.scanstring  # type: ignore[attr-defined]  # the C scanner json.loads uses
_LITERALS = (("true", True), ("false", False), ("null", None))


def parse(data: bytes) -> Any:
    """The value the plane's ``JSON.parse`` makes of the request bytes *data*. Raises ValueError
    for bytes that are not UTF-8 or text that is not JSON (NaN and Infinity are not JSON)."""
    text = data.decode("utf-8-sig")               # the WHATWG decoder drops one leading BOM
    try:
        return json.loads(text, parse_constant=_not_json, parse_float=_number, parse_int=_number)
    except RecursionError:                        # deeper than the C scanner goes: same grammar, no stack
        return _parse_deep(text)


def stringify(value: Any) -> str | None:
    """``JSON.stringify`` of *value* once the plane has parsed it (JavaScript numbers), or None for
    what the plane cannot hold."""
    out: list[str] = []
    todo: list[Any] = [value]
    while todo:
        v = todo.pop()
        if isinstance(v, _Text):
            out.append(v.text)
        elif v is None or isinstance(v, bool):
            out.append(json.dumps(v))
        elif isinstance(v, (int, float)):
            out.append(js_number(v))
        elif isinstance(v, str):
            out.append(json.dumps(v, ensure_ascii=False))
        elif isinstance(v, (list, tuple)):
            todo.append(_Text("]"))
            for i in range(len(v) - 1, -1, -1):
                todo.append(v[i])
                if i:
                    todo.append(_Text(","))
            out.append("[")
        elif isinstance(v, dict):
            if not all(isinstance(k, str) for k in v):
                return None
            items = list(v.items())
            todo.append(_Text("}"))
            for i in range(len(items) - 1, -1, -1):
                k, item = items[i]
                todo.append(item)
                todo.append(_Text(("," if i else "") + json.dumps(k, ensure_ascii=False) + ":"))
            out.append("{")
        else:
            return None                           # UNHOLDABLE, or not a JSON value
    return "".join(out)


def js_number(x: float) -> str:
    """ECMAScript ``Number::toString`` of the finite double *x* (or an int that rounds to one)."""
    if isinstance(x, int) and abs(x) <= 2 ** 53:
        return str(x)
    x = float(x)
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


class _Text:
    __slots__ = ("text",)

    def __init__(self, text: str):
        self.text = text


def _not_json(token: str) -> Any:
    raise ValueError(f"{token} is not JSON")


def _number(text: str) -> Any:
    """A JSON number as a double holds it: :data:`UNHOLDABLE` beyond one. ``float`` reads any
    length of digits (``int`` refuses over 4,300), and an integer is kept exact within range."""
    big = float(text)
    if not math.isfinite(big):
        return UNHOLDABLE
    return big if any(c in text for c in ".eE") else int(text)


def _parse_deep(text: str) -> Any:
    """:func:`json.loads`' grammar with an explicit stack instead of recursion."""
    stack: list[list[Any]] = []                   # [container, key awaiting its value]
    pos = _skip(text, 0)
    while True:
        ch = text[pos:pos + 1]
        if ch == "{":
            pos = _skip(text, pos + 1)
            if text[pos:pos + 1] != "}":
                key, pos = _key(text, pos)
                stack.append([{}, key])
                continue
            value, pos = {}, pos + 1
        elif ch == "[":
            pos = _skip(text, pos + 1)
            if text[pos:pos + 1] != "]":
                stack.append([[], None])
                continue
            value, pos = [], pos + 1
        elif ch == '"':
            value, pos = _scanstring(text, pos + 1, True)
        elif (m := NUMBER_RE.match(text, pos)) is not None:
            value, pos = _number(m.group()), m.end()
        else:
            for word, value in _LITERALS:
                if text.startswith(word, pos):
                    pos += len(word)
                    break
            else:
                raise ValueError(f"no JSON value at {pos}")
        while True:                               # place the value; close what it completes
            pos = _skip(text, pos)
            if not stack:
                if pos != len(text):
                    raise ValueError(f"extra data at {pos}")
                return value
            top = stack[-1]
            container = top[0]
            if isinstance(container, dict):
                container[top[1]] = value
            else:
                container.append(value)
            ch = text[pos:pos + 1]
            if ch == ",":
                pos = _skip(text, pos + 1)
                if isinstance(container, dict):
                    top[1], pos = _key(text, pos)
                break
            if ch != ("}" if isinstance(container, dict) else "]"):
                raise ValueError(f"expected , or a close at {pos}")
            stack.pop()
            value, pos = container, pos + 1


def _skip(text: str, pos: int) -> int:
    m = _WS.match(text, pos)
    return m.end() if m else pos


def _key(text: str, pos: int) -> tuple[str, int]:
    if text[pos:pos + 1] != '"':
        raise ValueError(f"expected a key at {pos}")
    key, pos = _scanstring(text, pos + 1, True)
    pos = _skip(text, pos)
    if text[pos:pos + 1] != ":":
        raise ValueError(f"expected : at {pos}")
    return key, _skip(text, pos + 1)

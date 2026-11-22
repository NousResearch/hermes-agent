"""Replacement templates with stdlib `re` syntax (``\\1``, ``\\g<name>``, ``\\g<1>0``, octal and C
escapes), parsed here and never by the binding's ``expand``. Stdlib-only, no engine.
"""
from __future__ import annotations

import functools

from ._errors import error

_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "a": "\a", "b": "\b", "f": "\f", "v": "\v", "\\": "\\"}
_OCTAL = frozenset("01234567")


@functools.lru_cache(maxsize=512)
def parse(template, groups, names):
    """A stdlib replacement template as a tuple of literals and group numbers."""
    is_bytes = isinstance(template, bytes)
    text = template.decode("latin-1") if is_bytes else template
    parts, literal, i, n = [], [], 0, len(text)
    lookup = dict(names)

    def group_ref(number):
        if number > groups:
            raise error(f"invalid group reference {number}", template)
        if literal:
            parts.append("".join(literal))
            literal.clear()
        parts.append(number)

    while i < n:
        ch = text[i]
        if ch != "\\":
            literal.append(ch)
            i += 1
            continue
        if i + 1 >= n:
            raise error("bad escape (end of pattern)", template)
        nxt = text[i + 1]
        if nxt == "g":
            end = text.find(">", i)
            if text[i + 2:i + 3] != "<" or end == -1:
                raise error("missing <group name>", template)
            name = text[i + 3:end]
            if name.isascii() and name.isdigit():
                group_ref(int(name))
            elif name in lookup:
                group_ref(lookup[name])
            elif name.isidentifier():  # as stdlib: a well-formed name that no group has
                raise IndexError(f"unknown group name {name!r}")
            else:
                raise error(f"bad character in group name {name!r}", template)
            i = end + 1
        elif nxt == "0" or (nxt in _OCTAL and text[i + 2:i + 3] in _OCTAL and text[i + 2:i + 3]
                            and text[i + 3:i + 4] in _OCTAL and text[i + 3:i + 4]):
            j = i + 1
            while j < min(i + 4, n) and text[j] in _OCTAL:
                j += 1
            code = int(text[i + 1:j], 8)
            if code > 0o377:  # as stdlib: three octal digits are an escape, never a group reference
                raise error(f"octal escape value \\{text[i + 1:j]} outside of range 0-0o377", template)
            literal.append(chr(code))
            i = j
        elif "0" <= nxt <= "9":  # ASCII only, as stdlib: a non-ASCII digit after `\1` is literal text
            j = i + 3 if "0" <= text[i + 2:i + 3] <= "9" else i + 2
            group_ref(int(text[i + 1:j]))
            i = j
        elif nxt in _ESCAPES:
            literal.append(_ESCAPES[nxt])
            i += 2
        elif nxt.isascii() and nxt.isalpha():
            raise error(f"bad escape \\{nxt}", template)
        else:
            literal.append(ch + nxt)
            i += 2
    if literal:
        parts.append("".join(literal))
    if is_bytes:
        parts = [p.encode("latin-1") if isinstance(p, str) else p for p in parts]
    return tuple(parts)


def _binary(template, sub: bool):
    """A replacement that is neither ``str`` nor ``bytes``, as stdlib's ``sub`` and ``expand``
    take it: ``sub`` inserts a bytes-like one without a backslash as it is; otherwise it is
    parsed as bytes once ``re._compile_template``'s cache has hashed it (an unhashable buffer
    such as a bytearray as its bytes; a typed or writable memoryview raises ValueError there)."""
    if sub:
        try:
            if b"\\" not in memoryview(template).tobytes():
                return template, True
        except TypeError:  # not a buffer: stdlib hands it to the template parser, which raises
            pass
    try:
        hash(template)
    except TypeError:
        template = bytes(memoryview(template))  # TypeError for an object that is no buffer either
    return str(template, "latin-1").encode("latin-1"), False


def template(template, pattern, sub: bool = True):
    """The literal itself when it has no escapes, else a callable that expands it. ``sub``:
    called by ``sub()`` rather than ``Match.expand()``, which differ for a bytes-like literal."""
    if not isinstance(template, (str, bytes)):
        template, literal = _binary(template, sub)
        if literal:
            return template
    if (b"\\" if isinstance(template, bytes) else "\\") not in template:
        return template
    parts = parse(template, pattern.groups, tuple(sorted(pattern._names.items())))
    if len(parts) == 1 and not isinstance(parts[0], int):
        return parts[0]
    empty = template[:0]
    return lambda m: empty.join([(m.group(p) or empty) if isinstance(p, int) else p for p in parts])

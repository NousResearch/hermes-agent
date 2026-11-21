"""The module-level functions of stdlib `re` (``search``, ``sub``, ...) and ``escape``.

Each one compiles through the cache and calls the Pattern method, so a pattern argument may be
anything ``compile()`` accepts, a compiled stdlib pattern included.
"""
from __future__ import annotations

from ._cache import compile


def search(pattern, string, flags=0):
    return compile(pattern, flags).search(string)


def match(pattern, string, flags=0):
    return compile(pattern, flags).match(string)


def fullmatch(pattern, string, flags=0):
    return compile(pattern, flags).fullmatch(string)


def finditer(pattern, string, flags=0):
    return compile(pattern, flags).finditer(string)


def findall(pattern, string, flags=0):
    return compile(pattern, flags).findall(string)


def split(pattern, string, maxsplit=0, flags=0):
    return compile(pattern, flags).split(string, maxsplit)


def sub(pattern, repl, string, count=0, flags=0):
    return compile(pattern, flags).sub(repl, string, count)


def subn(pattern, repl, string, count=0, flags=0):
    return compile(pattern, flags).subn(repl, string, count)


_SPECIAL = {ord(c): "\\" + c for c in "()[]{}?*+-|^$\\.&~# \t\r\v\f"}
_SPECIAL[ord("\n")] = "\\n"  # stdlib emits backslash + a real newline, which RE2 rejects
_SPECIAL[0] = "\\x00"


def escape(pattern):
    if isinstance(pattern, str):
        return pattern.translate(_SPECIAL)
    return pattern.decode("latin-1").translate(_SPECIAL).encode("latin-1")

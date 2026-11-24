"""``compile()``, ``purge()``, the forced build (``build``, ``build_all``), and the list of the
package's caches.

Three caches exist and ``purge()`` clears all of them:
  1. compiled patterns: ``_pattern.cached`` over ``_budget``, an LRU keyed by (pattern type,
     pattern, flags) and bounded by ``_budget.MAX_ENTRIES`` entries and ``_budget.MAX_BYTES``
     of built programs;
  2. the two-slot encoded subject with its offset index (``_subject``);
  3. parsed replacement templates (``_template.parse``).
"""
from __future__ import annotations

from . import _budget, _subject, _template
from ._errors import error
from ._pattern import Pattern, cached, live


def _is_stdlib_pattern(obj) -> bool:
    """A compiled stdlib ``re.Pattern``, recognised by its type's name (no stdlib `re` import)."""
    cls = obj.__class__
    return cls.__name__ == "Pattern" and cls.__module__ == "re"


def compile(pattern, flags=0):
    """A facade Pattern. Checked EAGERLY: a pattern the facade cannot run raises ``error`` here,
    as stdlib does. The RE2 program itself is built on first use (see ``build``).

    ``pattern`` may be ``str``, ``bytes``, a facade Pattern (returned as it is), or a compiled
    STDLIB pattern, which is converted from its ``.pattern`` and ``.flags``. That is a
    conversion, not a fallback: the result is matched by RE2, or this raises ``error``.
    """
    kind = pattern.__class__
    if kind is str or kind is bytes:  # the hot path: one cache lookup
        return cached(kind, pattern, flags)
    if kind is Pattern or _is_stdlib_pattern(pattern):
        if flags:
            raise ValueError("cannot process flags argument with a compiled pattern")
        if kind is Pattern:
            return pattern
        source, flags = pattern.pattern, pattern.flags
        if isinstance(source, str):
            flags &= ~32  # stdlib reports the implied UNICODE bit; drop it so the cache key is compile(source)'s
        return cached(source.__class__, source, flags)
    if isinstance(pattern, (str, bytes)):  # a subclass of str or bytes
        kind = str if isinstance(pattern, str) else bytes
        return cached(kind, kind(pattern), flags)
    raise TypeError("first argument must be string or compiled pattern")


def build(pattern, flags=0):
    """``compile(pattern, flags)`` with its RE2 program built NOW. Raises what first use would
    raise: ``error`` (a program over RE2's memory budget), ``ModuleNotFoundError`` (no google-re2)."""
    compiled = compile(pattern, flags)
    compiled._force(False)
    return compiled


def build_all(wide=False) -> list:
    """Build every Pattern alive in this process (the compile cache's entries and every Pattern
    some module still holds). Returns the failures as ``(Pattern, error)`` pairs: an empty list
    means every pattern builds. ``ModuleNotFoundError`` (no google-re2) is not a per-pattern
    failure and propagates.

    ``wide=True`` also builds the Unicode `\\b` forms, which matching builds on the first
    non-ASCII subject, and reports one that cannot be built (matching would quietly keep RE2's
    ASCII `\\b` for that pattern). For a test: import the product's modules, then
    ``assert hermes_re.build_all() == []``.
    """
    failures = []
    for compiled in live():
        try:
            compiled._force(wide)
        except error as exc:
            failures.append((compiled, exc))
    return failures


def purge() -> None:
    _budget.clear()
    _subject.clear()
    _template.parse.cache_clear()

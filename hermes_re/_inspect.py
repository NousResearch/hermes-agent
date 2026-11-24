"""Introspection for the product's lints and tests (stdlib-only, no engine).

``inexact_boundaries`` answers "which word boundaries of this pattern are NOT matched the way
stdlib matches them": the product's CI can assert that no security pattern has one.
"""
from __future__ import annotations

from . import _comments, _edge_bound
from ._cache import compile
from ._errors import error
from ._form import Form
from ._translate_scan import class_end


def _everywhere(text: str) -> list:
    """The offset of every `\\b` and `\\B` in ``text``."""
    out, i, n = [], 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "\\":
            if text[i + 1:i + 2] in ("b", "B"):
                out.append(i)
            i += 2
        elif ch == "[":
            i = class_end(text, i)
        else:
            i += 1
    return out


def inexact_boundaries(pattern, flags=0) -> list:
    r"""The offsets, in the pattern text, of every `\b` and `\B` that the facade leaves to RE2's
    ASCII rule (word characters are ``[A-Za-z0-9_]``) although stdlib would use the Unicode rule.

    An empty list means every boundary of the pattern is matched exactly as stdlib matches it:
    the pattern has none, or it is a bytes / ASCII-flag pattern (stdlib's rule is ASCII too), or
    each `\b` could be decided from its neighbours or respelled as an edge guard on `\w`.
    A listed boundary gives stdlib's result on ASCII text and can differ (in either direction)
    next to a non-ASCII letter or digit. Listed are: a `\b` in the interior of the pattern, a
    `\b` at an edge whose neighbour can be a word or a non-word character (``\b[\w./-]+``), a
    `\b` inside a repeated group or a lookaround, and every `\B`.

    ``pattern`` is anything ``compile()`` accepts; an invalid pattern raises ``error``. Not
    covered: a pattern whose Unicode form exceeds RE2's memory budget keeps ASCII `\b`
    everywhere, which only a build can tell (``build_all(wide=True)`` reports it).
    """
    compiled = compile(pattern, flags)
    source, engine_flags = compiled.pattern, int(compiled.flags)
    if not isinstance(source, str) or engine_flags & 256:
        return []
    text, cuts = _comments.strip(source)
    wide, stats = _edge_bound.rewrite(text, engine_flags)
    if stats is None:
        return [_comments.origin(cuts, offset) for offset in _everywhere(text)]
    kept = [entry[0] for entry in stats.kept_at] + stats.not_at
    if wide != text:
        try:
            Form(source, wide, engine_flags, False)
        except error:  # the respelled pattern cannot be planned: matching keeps every `\b` ASCII
            kept = _everywhere(text)
    return sorted(_comments.origin(cuts, offset) for offset in kept)

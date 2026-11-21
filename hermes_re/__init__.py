"""hermes_re: the stdlib `re` API on top of Google RE2 (linear-time matching, no backtracking).

Use it as ``import hermes_re as re``. There is NO fallback to stdlib `re` for matching.

``compile()`` checks and translates the whole pattern at once, without the engine: a pattern the
facade or RE2 cannot run raises ``hermes_re.error`` there, as a syntax error does in stdlib. The
RE2 program is built on FIRST USE (it cannot be persisted and can take hundreds of milliseconds),
so two things surface only then: ``error`` for a program over RE2's memory budget, and
``ModuleNotFoundError`` (``name == "re2"``) when google-re2 is not importable (importing this
package, and ``compile()``, never import it). ``build()`` / ``build_all()`` force the build, for tests.

Besides the stdlib names: ``build``, ``build_all``, and ``inexact_boundaries`` (which `\\b` of a
pattern are matched by RE2's ASCII rule instead of stdlib's Unicode rule).

This file is the facade only: it re-exports names and holds no logic. Topic modules:

  _flags            flag constants with stdlib's integer values; the unsupported-flag check
  _errors           ``error``: the exception class, a subclass of stdlib ``re.error``
  _comments         `(?#...)` comment groups, removed before anything else reads the pattern
  _validate         syntax check of a source before translation; capture groups and their names
  _translate        Python regex dialect -> RE2 dialect (escapes, `\\Z`, `{,n}`, scoped inline flags)
  _translate_class  `\\d \\w \\s` and bracket classes with stdlib meaning, exact IGNORECASE classes
  _translate_ranges code-point range sets and the four-cell algebra behind class translation
  _translate_fold   case folding as stdlib defines it, from the running interpreter's tables
  _translate_scan   small text scans shared by the modules that read pattern text
  _tables           GENERATED Unicode delta table (Python's `\\w` / `\\d` versus RE2's)
  _edge_parse       parser for the Python regex dialect (used by the edge-guard modules)
  _edge_plan        edge guards (leading lookbehind, trailing lookahead, tail `$`) -> consumed text
  _edge_run         runs a guarded pattern and reports the caller's own match
  _edge_guard       cheap spellings of guards on Unicode classes, and their exact test in Python
  _budget           the compile cache's store: an LRU bounded by entry count and program size
  _edge_word        "is this neighbour definitely a word character" analysis for `\\b`
  _edge_bound       `\\b` at a pattern edge -> edge guards on Unicode `\\w`
  _engine           google-re2 loading (on the first build), RE2 options, source -> RE2 program,
                    and ``engine_missing()``: the one place that decides what a missing engine means
  _form             one translated spelling of a pattern, built into RE2 programs on demand
  _subject          the buffer RE2 scans for a subject; the two-slot subject cache
  _offsets          sparse character <-> UTF-8 byte offset index
  _template         replacement templates (``\\1``, ``\\g<name>``)
  _match            ``Match``
  _pattern          ``Pattern``; the compile cache; the weak registry of live patterns
  _cache            ``compile()``, ``purge()``, ``build()``, ``build_all()``
  _inspect          ``inexact_boundaries()``
  _api              module-level ``search``, ``sub``, ... and ``escape``
"""
from ._api import escape, findall, finditer, fullmatch, match, search, split, sub, subn
from ._cache import build, build_all, compile, purge
from ._errors import PatternError, error
from ._flags import (
    A, ASCII, DEBUG, DOTALL, I, IGNORECASE, L, LOCALE, M, MULTILINE, NOFLAG, S, U, UNICODE, VERBOSE, X, RegexFlag,
)
from ._inspect import inexact_boundaries
from ._match import Match
from ._pattern import Pattern

__all__ = [
    "ASCII",
    "DEBUG",
    "DOTALL",
    "IGNORECASE",
    "LOCALE",
    "MULTILINE",
    "NOFLAG",
    "UNICODE",
    "VERBOSE",
    "A",
    "I",
    "L",
    "M",
    "Match",
    "Pattern",
    "PatternError",
    "RegexFlag",
    "S",
    "U",
    "X",
    "build",
    "build_all",
    "compile",
    "error",
    "escape",
    "findall",
    "finditer",
    "fullmatch",
    "inexact_boundaries",
    "match",
    "purge",
    "search",
    "split",
    "sub",
    "subn",
]

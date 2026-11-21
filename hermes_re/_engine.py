"""google-re2 loading, RE2 options, and the two steps from pattern text to a program.

  ``source()``   Python-dialect text -> RE2 source text. Syntax check and translation only: no
                 engine, so ``compile()`` can run it at import time, eagerly, for every pattern.
  ``program()``  RE2 source text -> ``re2._re2.RE2`` object. Needs the engine; ``Pattern`` calls
                 it on first use.

This is the only module that touches the engine. google-re2 is imported by ``load()`` when the
first program is built, never at facade import or at ``compile()``, so a module with
module-level ``compile()`` calls imports where the package is missing. It drives
``re2._re2.RE2`` (the compiled extension class) directly: the binding's pure-Python layer costs
about 2 us per match and mis-decodes non-ASCII replacement text.
"""
from __future__ import annotations

from . import _offsets
from ._errors import error
from ._translate import NonTailDollar, translate
from ._validate import check

# RE2's per-pattern budget (program + lazily built DFA). RE2's own default is 8 MiB; the product
# value is 64 MiB, so that bounded repeats of Unicode classes (`\w{0,512}`) still fit.
MAX_MEM = 64 << 20
# What a built program is counted as by the compile cache's memory bound (``_budget``): a fixed
# part plus a part per instruction (RE2 reports the forward program's instruction count). Both
# are measured resident memory after a few searches, with the reverse program and the DFA states
# those searches create. An estimate: a program whose DFA keeps growing can hold more, up to
# ``MAX_MEM``.
PROGRAM_BYTES = 8 << 10
INSTRUCTION_BYTES = 48
ANCHORS = None     # (UNANCHORED, ANCHOR_START, ANCHOR_BOTH) once the engine is loaded
_re2 = None


def engine_missing(exc: ImportError) -> ModuleNotFoundError:
    """THE policy for "google-re2 cannot be imported". Change it here and nowhere else.

    ``compile()`` never needs the engine, so a module with a module-level ``re.compile(...)``
    still imports in a process that lacks the package (e.g. an old updater process). This is
    asked when a program has to be built, i.e. on the first matching operation, and returns the
    exception to raise there: a ``ModuleNotFoundError`` (an ``ImportError``) whose ``name`` is
    ``"re2"``, whatever made the import fail. There is no fallback to stdlib `re`.

    Type and name are a contract with releases already shipped: their update step imports the
    new checkout's core modules in a process that may not have google-re2 yet, and reads a
    ``ModuleNotFoundError`` naming a third-party module as "dependencies not installed yet",
    but any other ``ImportError`` as a broken checkout.
    """
    return ModuleNotFoundError(
        "hermes_re cannot match: the google-re2 package is not importable in this interpreter "
        f"({exc}). Install google-re2; hermes_re never falls back to stdlib re.", name="re2")


def load():
    """The engine module; google-re2 is imported on the first call."""
    global _re2, ANCHORS
    if _re2 is not None:
        return _re2
    try:
        # Not at module top on purpose: the facade must import where google-re2 is missing.
        from re2 import _re2 as module
    except ImportError as exc:
        raise engine_missing(exc) from exc
    anchor = module.RE2.Anchor
    # indexed by match kind: 0 search, 1 match, 2 fullmatch
    ANCHORS = (anchor.UNANCHORED, anchor.ANCHOR_START, anchor.ANCHOR_BOTH)
    _offsets.bind(module.CharLenToBytes, module.BytesToCharLen)
    _re2 = module
    return module


def source(text: str, flags: int, latin1: bool, original):
    """``(RE2 source, capture groups, ((name, number), ...))`` for ``text`` (Python dialect; a
    bytes pattern arrives Latin-1 decoded with the ASCII flag set). No engine is involved.
    Raises ``error`` carrying ``original``, the caller's pattern."""
    groups, names = check(text, latin1, original)
    try:
        return translate(text, flags, latin1), groups, names
    except NonTailDollar as exc:
        raise error("non-multiline `$` is supported only in tail position (nothing may be matched "
                    f"after it); position {exc.args[0]}: restructure the pattern", original) from None
    except error as exc:
        raise error(exc.msg, original) from None


def program(text: str, flags: int, latin1: bool, original):
    """The RE2 object for RE2-dialect ``text``; ``error`` (carrying the caller's pattern) if RE2
    refuses it. After ``source()`` accepted the pattern that is, in practice, the memory budget."""
    engine = _re2 or load()
    options = engine.RE2.Options()
    options.log_errors = False  # RE2 otherwise writes each rejected pattern to stderr
    options.max_mem = MAX_MEM
    options.case_sensitive = not flags & 2
    options.dot_nl = bool(flags & 16)
    if latin1:
        options.encoding = engine.RE2.Options.Encoding.LATIN1  # bytes match byte-wise, like stdlib
    if flags & 8:
        text = "(?m)" + text
    try:
        compiled = engine.RE2(text.encode("latin-1" if latin1 else "utf-8"), options)
    except UnicodeEncodeError:
        raise error("pattern contains a lone surrogate, which RE2 cannot match", original) from None
    if not compiled.ok():
        msg = compiled.error()
        raise error(msg.decode("utf-8", "replace") if isinstance(msg, bytes) else str(msg), original)
    return compiled


def build(text: str, flags: int, latin1: bool, original):
    """Both steps at once: the RE2 object for Python-dialect ``text``."""
    return program(source(text, flags, latin1, original)[0], flags, latin1, original)

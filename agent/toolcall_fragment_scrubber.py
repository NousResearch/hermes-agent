"""Stateful scrubber for leaked tool-call XML opener fragments in streamed text.

Qwen3 models (and others) sometimes emit partial ``<tool_call>`` /
``<function_call>`` XML opener strings as plain TEXT deltas during
streaming, right before switching to native ``tool_calls``.  This leaks
stray characters such as ``ool_call>``, ``l_call>``, ``_call>``, etc.
into user-facing output.

**Corrected contract (F1 fix — context-gated stripping)**:

All fragment stripping is restricted to **tool-call context only**
(``in_toolcall_context=True``).  In plain prose context
(``in_toolcall_context=False``, Path A ``_fire_stream_delta``), the
scrubber is a pure pass-through — no mid-word suffix stripping, no
opener stripping, no corruption.  This eliminates false-positive
regressions where short suffixes such as ``call>`` or ``ll>`` were
substrings of ordinary HTML tags and prose words that LLMs legitimately
emit (e.g. ``<ul>``, ``<ol>``, ``<html>``, ``<small>``, ``<details>``,
``<rules>``, ``<files>``).

Leaked ``<tool_call>``/``<function_call>`` opener fragments only ever
occur when the model is transitioning to native tool_calls — i.e.
``in_toolcall_context=True`` (Path B, the suppressed-content bypass
branch in ``agent/chat_completion_helpers.py``).  In plain prose (Path
A) there is no legitimate leak, so Path A must be pure passthrough.

**Concrete ``feed()`` contract**:

``feed(text, in_toolcall_context=False)`` — prose / Path A:
  Pass *text* through UNCHANGED.  No mid-word suffix stripping, no
  opener stripping, no corruption.  ``<ul>``, ``<ol>``, ``<html>``,
  ``<small>``, ``<details>``, ``<rules>``, ``call>``, ``balls>``,
  ``a > b``, ``<tool_call>``, ``<tool_call>x</tool_call>`` — ALL must
  round-trip byte-for-byte.  Any previously held buffer is flushed and
  prepended; the buffer is then cleared.

``feed(text, in_toolcall_context=True)`` — tool-call active / Path B:
  Full stripping:
  * Strip mid-word suffix fragments that do NOT begin with ``<``
    (e.g. ``ool_call>``, ``l_call>``, ``_call>``, ``call>``, ``all>``,
    ``ll>``, ``l>``).
  * Strip a full/leading-``<`` opener (e.g. ``<tool_call>``).
  * Maintain stateful ``_buf`` boundary-hold so a split fragment
    arriving as ``<too`` then ``l_call>`` within tool-call context
    reconstructs and is stripped to ``""``.

The single-character ``>`` exclusion stays regardless of context.

**Buffer context isolation**:

The ``_buf`` hold is scoped to tool-call context.  In prose context
the held buffer (if any) is flushed immediately and prepended to the
return value — no bytes are held or dropped.  A buffer held in
tool-call context and then fed in prose context (which should not
happen in normal streaming order but is handled defensively) is also
flushed through without stripping.

**Residual limitation (F2 — documented and accepted)**:

A leaked opener whose *prefix* arrives as the final prose delta
immediately before the tool-call transition (i.e. split across the
Path A → Path B boundary) may not be fully scrubbed.  For example,
if ``<too`` arrives on Path A (where it is held) and then the context
switches to Path B for ``l_call>``, the ``<too`` was flushed by the
Path A call and already delivered.  This is an accepted, rare edge
case — in practice the split is almost always within a single path
context because the transition fires only when ``tool_calls_acc``
becomes non-empty (which happens on the tool-call delta chunk, not
during prose text chunks).

**Preservation rule summary**:

  - ``"ool_call>"`` in prose ctx (False)  → ``"ool_call>"`` (passthrough)
  - ``"ool_call>"`` in TC ctx (True)      → ``""``             (stripped)
  - ``"<ul>"`` in prose ctx              → ``"<ul>"``          (passthrough)
  - ``"<tool_call>"`` in prose ctx       → ``"<tool_call>"``   (passthrough)
  - ``"<tool_call>"`` in TC ctx          → ``""``              (stripped)
  - Split ``"<too"`` + ``"l_call>"`` in TC ctx → ``""``        (stripped)
  - ``"a > b"``                          → ``"a > b"``         (passthrough)
  - ``"plain text"``                     → ``"plain text"``    (passthrough)

**Single-character suffixes are excluded in tool-call context**:

  The lone ``>`` suffix of every opener is NOT included in the
  fragment regex.  A bare ``>`` is valid in comparison operators
  (``a > b``), HTML (``<p>text</p>``), and Markdown blockquotes
  (``> quote``).  The shortest real fragment handled in TC context is
  therefore ``l>`` / ``s>`` (2 chars from ``<tool_call>`` /
  ``<tool_calls>``).

Usage::

    scrubber = StreamingToolCallFragmentScrubber()
    for delta in stream:
        # Path A: prose context (no active tool call)
        visible = scrubber.feed(delta)            # pure passthrough
        if visible:
            emit(visible)
    tail = scrubber.flush()
    if tail:
        emit(tail)

    # Path B: tool-call context (tool_calls_acc non-empty)
    visible = scrubber.feed(delta, in_toolcall_context=True)
    if visible:
        emit(visible)

Call ``reset()`` at the top of each new turn.

Openers handled in tool-call context (case-insensitive):
  ``<tool_call>``, ``<tool_calls>``,
  ``<function_call>``, ``<function_calls>``.
"""

from __future__ import annotations

import re
from typing import Tuple

__all__ = ["StreamingToolCallFragmentScrubber"]


def _build_opener_suffixes(openers: Tuple[str, ...]) -> Tuple[str, ...]:
    """Return every contiguous suffix of every opener that ends with ``>``,
    excluding single-character suffixes.

    For opener ``<tool_call>`` the suffixes are (longest first):
      ``<tool_call>``, ``tool_call>``, ``ool_call>``, ``ol_call>``,
      ``l_call>``, ``_call>``, ``call>``, ``all>``, ``ll>``, ``l>``.
      The lone ``>`` is excluded — it is not a recognisable fragment.

    Computing these programmatically means no hand-rolled alternation and
    every split point is covered automatically when openers change.

    Why: A lone ``>`` appears in comparison operators (``a > b``),
    HTML close tags (``<p>text</p>``), and Markdown blockquotes
    (``> quote``).  Stripping it would corrupt normal prose.
    """
    seen: set[str] = set()
    result: list[str] = []
    for opener in openers:
        for start in range(len(opener)):
            suffix = opener[start:]
            # Exclude single-character suffixes (just ">") to avoid
            # corrupting comparison operators, HTML, and Markdown blockquotes.
            if suffix.endswith(">") and len(suffix) > 1 and suffix not in seen:
                seen.add(suffix)
                result.append(suffix)
    # Sort longest-first so the regex engine attempts the most specific hit.
    result.sort(key=len, reverse=True)
    return tuple(result)


def _build_close_tags(openers: Tuple[str, ...]) -> Tuple[str, ...]:
    """Derive close tags from openers (strip '<', wrap with '</' and '>')."""
    close: list[str] = []
    for opener in openers:
        name = opener.lstrip("<").rstrip(">")
        close.append(f"</{name}>")
    return tuple(close)


_OPENERS_CONST: Tuple[str, ...] = (
    "<tool_call>",
    "<tool_calls>",
    "<function_call>",
    "<function_calls>",
)
_SUFFIXES_CONST: Tuple[str, ...] = _build_opener_suffixes(_OPENERS_CONST)
_CLOSE_TAGS_CONST: Tuple[str, ...] = _build_close_tags(_OPENERS_CONST)

# Regex that matches any close tag (case-insensitive) so we can protect them.
_CLOSE_TAG_RE: re.Pattern[str] = re.compile(
    "(" + "|".join(re.escape(ct) for ct in _CLOSE_TAGS_CONST) + ")",
    re.IGNORECASE,
)

# Regex that matches any opener-suffix fragment (case-insensitive).
# Single-character ">" is excluded from _SUFFIXES_CONST, so it is not matched.
# _SUFFIXES_CONST is already sorted longest-first by _build_opener_suffixes;
# no re-sort needed here.
_FRAGMENT_RE: re.Pattern[str] = re.compile(
    "(" + "|".join(re.escape(s) for s in _SUFFIXES_CONST) + ")",
    re.IGNORECASE,
)


def _strip_fragments(buf: str) -> str:
    """Remove opener-suffix fragments from *buf* without touching close tags.

    Must only be called in tool-call context (``in_toolcall_context=True``).
    Prose-context callers must short-circuit before reaching this function —
    in prose mode the text is passed through unchanged, so there is nothing
    to strip here.  The parameter was removed to eliminate a dead branch that
    could silently re-introduce mid-word stripping if ever called from prose
    context by mistake.

    Algorithm:
      1. Find all close-tag spans and mark them as protected regions.
      2. Run the fragment regex over the buffer; for each match:
         - If the match overlaps a close-tag span, keep it (it IS the close tag).
         - Otherwise, suppress it (it is leaked garbage or a confirmed
           tool-call-context opener).
      3. Reconstruct the output from surviving spans.

    Case-insensitive.

    Test: feed ``<tool_call>`` with in_toolcall_context=True → empty string;
    ``</tool_call>`` protected in TC context → preserved.
    """
    if not buf:
        return buf

    # Collect protected intervals [start, end) from close tags.
    protected: list[tuple[int, int]] = []
    for m in _CLOSE_TAG_RE.finditer(buf):
        protected.append((m.start(), m.end()))

    def _is_protected(start: int, end: int) -> bool:
        for ps, pe in protected:
            if start < pe and end > ps:
                return True
        return False

    result_parts: list[str] = []
    pos = 0
    for m in _FRAGMENT_RE.finditer(buf):
        fstart, fend = m.start(), m.end()

        # Always keep spans inside close tags.
        if _is_protected(fstart, fend):
            result_parts.append(buf[pos:fend])
            pos = fend
            continue

        # Strip: keep the prose before the fragment, drop the fragment itself.
        result_parts.append(buf[pos:fstart])
        pos = fend

    result_parts.append(buf[pos:])
    return "".join(result_parts)


class StreamingToolCallFragmentScrubber:
    """Stateful, boundary-aware scrubber for leaked tool-call XML opener fragments.

    State:
      - ``_buf``: held-back tail that is a prefix of some opener and has
        not yet been resolved.  Only used in tool-call context
        (``in_toolcall_context=True``).  In prose context the buffer is
        flushed immediately and cleared.

    **Context-gated stripping (F1 fix)**:

    All stripping happens ONLY in tool-call context
    (``in_toolcall_context=True``).  Prose context (``False``) is a pure
    passthrough — the buffer is flushed and the text is returned unchanged,
    preventing false-positive corruption of HTML tags (``<ul>``, ``<ol>``,
    ``<html>``, ``<small>``) and prose words that happen to end with a
    fragment suffix (``balls>``, ``rules>``, ``files>``).

    Fragment shapes and context rules (see module docstring for full
    rationale):

      * **Mid-word suffixes** (no leading ``<``): stripped ONLY in TC ctx.
      * **Leading-``<`` openers**: stripped only when
        ``in_toolcall_context=True``; preserved in prose (``False``).
      * **Single-char ``>``**: never stripped — excluded from all patterns.

    Test (docstring quick-check):
      ``feed("a > b")`` → ``"a > b"`` (bare ``>`` preserved; passthrough)
      ``feed("<tool_call>")`` → ``"<tool_call>"`` (prose, passthrough)
      ``feed("<tool_call>", in_toolcall_context=True)`` → ``""`` (TC ctx)
      ``feed("ool_call>")`` → ``"ool_call>"`` (prose, passthrough — F1 fix)
      ``feed("ool_call>", in_toolcall_context=True)`` → ``""`` (TC ctx)
      ``feed("<ul>")`` → ``"<ul>"`` (prose, passthrough — HTML preserved in prose)
      ``feed("<ul>", in_toolcall_context=True)`` → ``"<u"`` (TC ctx: ``l>`` suffix stripped)

    Note: HTML preservation (``<ul>``, ``<ol>``, ``<small>``, etc.) is a prose-context
    guarantee only.  In TC context the ``l>``, ``ll>``, ``all>`` etc. suffixes ARE
    stripped because that stream carries leaked tool-call boilerplate.  ``_CLOSE_TAG_RE``
    protects only real close tags (``</tool_call>`` family), not arbitrary HTML.
    """

    _OPENERS: Tuple[str, ...] = _OPENERS_CONST
    _OPENER_SUFFIXES: Tuple[str, ...] = _SUFFIXES_CONST
    _CLOSE_TAGS: Tuple[str, ...] = _CLOSE_TAGS_CONST
    _MAX_OPENER_LEN: int = max(len(o) for o in _OPENERS_CONST)

    def __init__(self) -> None:
        self._buf: str = ""
        # Track the in_toolcall_context of the held buffer so flush() can
        # apply the correct stripping rule to the tail.
        self._held_context: bool = False

    def reset(self) -> None:
        """Reset all state.  Call at the top of every new turn.

        Why: prevents held state from a prior turn contaminating the next.
        Test: feed('<to'), reset(), feed('normal text') → 'normal text'.
        """
        self._buf = ""
        self._held_context = False

    def feed(self, text: str, in_toolcall_context: bool = False) -> str:
        """Feed one streaming delta; return the scrubbed visible portion.

        Args:
            text: The raw streaming text delta.
            in_toolcall_context: True when ``tool_calls_acc`` is already
                non-empty (Path B bypass branch), meaning a tool call has
                definitively started and any ``<tool_call>`` opener in this
                window is leaked boilerplate.  False (default) for normal
                prose streaming (Path A ``_fire_stream_delta``).

        Returns:
            Scrubbed text, possibly empty when the delta was a pure
            fragment or is being held pending resolution (TC context only).

        Contract:
            **in_toolcall_context=False (prose / Path A)**: text is
            returned UNCHANGED (pure passthrough).  Any previously held
            buffer is prepended and the buffer is cleared.  No stripping
            of any kind occurs.

            **in_toolcall_context=True (TC active / Path B)**: full
            stripping — mid-word suffixes and leading-``<`` openers are
            stripped; the ``_buf`` hold/reconstruct machinery is active so
            split fragments (``"<too"`` + ``"l_call>"``) are reassembled
            and stripped.

        Why: Stateful so split-delta fragments (``"<too"`` + ``"l_call>"``)
        are correctly assembled and stripped in TC context.
        Test: feed('<too', True) → '' (held); feed('l_call>', True) → '' (suppressed);
        feed('a > b') → 'a > b' (bare > never stripped, passthrough).
        """
        if not in_toolcall_context:
            # Path A: pure passthrough.  Flush any held buffer (from a prior
            # TC-context call, which should not happen in normal streaming
            # order, but handle defensively).  Do NOT strip anything.
            # NOTE: we flush even for empty text so that a held _buf is
            # never silently dropped when the caller sends an empty delta
            # (e.g. feed('<to', True) then feed('', False) must return '<to').
            held = self._buf
            self._buf = ""
            self._held_context = False
            return held + text

        if not text:
            # TC context + empty delta: retain _buf (still mid-fragment); no-op.
            return ""

        # Path B: tool-call context — engage hold/strip machinery.

        # Prepend any held tail from the previous delta.
        buf = self._buf + text
        self._buf = ""
        self._held_context = True

        # --- Step 1: hold back opener-prefix at the tail ----------------
        # If the buffer tail could be the start of an opener, hold those
        # bytes so the next delta can complete them.
        hold_len = self._max_partial_suffix(buf)
        held = ""
        if hold_len:
            held = buf[-hold_len:]
            buf = buf[:-hold_len]

        # --- Step 2: strip unprotected opener-suffix fragments ----------
        result = _strip_fragments(buf)

        self._buf = held
        return result

    def flush(self) -> str:
        """End-of-stream flush.

        Any held tail that did not complete into a fragment is emitted
        verbatim — no data is silently dropped on clean stream end.
        If the tail is itself a complete opener fragment and the held
        context is TC context, context rules determine whether it is
        stripped.  In prose context the tail is always emitted verbatim.

        Why: Prevents a lone ``<`` or ``<to`` at stream end from being
        silently dropped.
        Test: feed('<to', True) → '' (held); flush() → '<to' (innocent partial emitted).
        Test: feed('<tool_call>', True) → '' (held); flush() → '' (fragment suppressed).
        Test: feed('<to', False) → '<to' (prose passthrough, not held); flush() → ''.
        """
        tail = self._buf
        ctx = self._held_context
        self._buf = ""
        self._held_context = False
        if not tail:
            return ""
        if ctx:
            return _strip_fragments(tail)
        # Prose context: tail is always emitted verbatim.
        return tail

    # ── internal helpers ───────────────────────────────────────────────

    @classmethod
    def _max_partial_suffix(cls, buf: str) -> int:
        """Return length of the longest buf-suffix that is a *strict prefix* of any opener.

        Only hold back incomplete prefixes (those that don't yet end with
        ``>``).  Complete fragments are handled by the fragment stripper.
        Case-insensitive.  Returns 0 if no partial prefix is found.

        Why: Prevents inter-delta split fragments from leaking through in
        TC context.
        Test: _max_partial_suffix('hello <to') → 3 (the '<to' tail).
        """
        if not buf:
            return 0
        buf_lower = buf.lower()
        max_check = min(len(buf_lower), cls._MAX_OPENER_LEN - 1)
        for i in range(max_check, 0, -1):
            suffix = buf_lower[-i:]
            # Skip if already ends with '>' — that's a complete fragment.
            if suffix.endswith(">"):
                continue
            for opener in cls._OPENERS:
                opener_lower = opener.lower()
                if opener_lower.startswith(suffix):
                    return i
        return 0

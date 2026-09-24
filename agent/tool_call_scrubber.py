"""Stateful scrubber for tool-call XML that leaked onto the text channel of a stream.

Open models on the Responses wire (observed: muse-spark via opencode-go) occasionally serialize
their next native tool call into the content stream as ``<atem:function_calls>…</atem:function_calls>``
instead of a ``function_call`` item. The completed response is cleaned by the regex stripper in
``agent/agent_runtime_helpers.py``, but per-delta consumers (CLI display, gateway previews, TTS,
``on_stream_delta`` hooks) see the raw XML first, and a tag split across deltas defeats a
per-delta regex (the same failure class that ``StreamingThinkScrubber`` exists for).

This scrubber mirrors the final stripper's tool-call positions, chunk-safely:

* a closed ``<tag>…</tag>`` pair anywhere (either side may carry a namespace prefix),
* a named ``<function name=…>`` block (boundary- and name-gated, exactly like the
  final pattern — no namespace prefix on the opener, literal ``</function>`` closer),
* GLM ``<arg_key>`` / ``<arg_value>`` markup that is the first ``<`` on its line
  (open or close, exactly like the final pattern — dropped through end of stream),
* a stray ``</tag>`` closer (no matching opener; ``</function>`` included, mirroring
  ``_STRAY_TOOL_CALL_CLOSER_PATTERN``),
* an unterminated opener at a line boundary — the stream was cut mid-serialization
  (mirrors ``_UNTERMINATED_TOOL_CALL_PATTERN``).

The tag itself follows the final stripper's keep/strip decision: a mid-line opener whose closer
never arrives is released at ``flush()`` (the final stripper keeps it too), a line-anchored one
is dropped. Cosmetic differences, called out because they are the only ones: text
already emitted cannot be retracted once a later tag reframes its line (the
final regex also removes the ``newline + indentation`` run before a dropped
opener, the space before a dropped named block, and the line prefix before
arg-key markup) — already-streamed text only, never XML.
Out of scope, same as the regexes: nothing — the prose-gated ``<function name=…>``
block and the GLM ``<arg_key>`` markup are covered above.

Safe text is emitted immediately per delta (no buffering of ordinary prose), so downstream
consumers see the same deltas as before except for the suppressed spans. The tag-name list
lives here and is imported by the final stripper, so a tag added here is covered on every
surface (the single-list contract ``think_scrubber`` established).
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

__all__ = ["TOOL_CALL_TAG_NAMES", "StreamingToolCallScrubber"]

# The one list of text-channel tool-call tag names: bound by the final-response stripper
# (agent/agent_runtime_helpers.py) and by this scrubber.
TOOL_CALL_TAG_NAMES: Tuple[str, ...] = (
    "tool_call",
    "tool_calls",
    "tool_result",
    "function_call",
    "function_calls",
)
# ``</function>`` closes nothing here, but the final stripper removes it as a stray closer.
_STRAY_CLOSER_NAMES: Tuple[str, ...] = TOOL_CALL_TAG_NAMES + ("function",)
_OPEN_NAMES = frozenset(TOOL_CALL_TAG_NAMES)
_CLOSE_NAMES = frozenset(_STRAY_CLOSER_NAMES)
# Named ``<function name=…>`` blocks and GLM ``<arg_key>`` / ``<arg_value>`` markup,
# gated exactly like the final patterns in agent/agent_runtime_helpers.py.
_NAMED_FUNCTION_OPENER_RE = re.compile(r"<function\b[^>]*\bname\s*=", re.IGNORECASE)
_ARG_MARKUP_TAG_RE = re.compile(r"</?arg_(?:key|value)\b", re.IGNORECASE)
_ARG_NAMES = frozenset(("arg_key", "arg_value"))

# Stray closers reapplied to a released mid-line raw span at flush(): the final
# stripper removes them anywhere, independently of blocks, so an unterminated
# mid-line block the final keeps still loses its strays. Same shape as
# _STRAY_TOOL_CALL_CLOSER_PATTERN in agent/agent_runtime_helpers.py.
_RELEASED_STRAY_CLOSER_RE = re.compile(
    r"</(?:(?:[\w.-]+:)?(?:" + "|".join(_STRAY_CLOSER_NAMES) + r"))>\s*",
    re.IGNORECASE,
)
# The run of characters a tag name portion may still be growing through (namespace allowed).
_RUN_RE = re.compile(r"[A-Za-z0-9_.:-]*")
# Optional ``ns:`` prefix + name word run; greedy, so a trailing ``\b``-equivalent is implicit.
_NAME_RE = re.compile(r"^(?:([\w.-]+):)?([A-Za-z0-9_]+)")
# A partial tag is abandoned past this many chars: real prefixes are short, and an unbounded
# hold would stall prose that merely contains '<'.
# ponytail: fixed ceiling — raise it if a provider ever streams longer tag prefixes.
_MAX_PARTIAL_TAG = 96


def _run_is_tag(run: str, names: frozenset) -> bool:
    """Whether the finished name portion *run* is exactly ``(ns:)?<recognized name>``."""
    if ":" in run:
        ns, local = run.split(":", 1)
        return bool(ns) and ":" not in local and local.lower() in names
    return run.lower() in names


def _run_may_grow(run: str, names: frozenset) -> bool:
    """Whether *run* (still at the buffer edge) could still grow into a recognized name."""
    if ":" in run:
        ns, local = run.split(":", 1)
        # one namespace separator at most; the local part must still be a name prefix
        return (
            bool(ns)
            and ":" not in local
            and (local == "" or any(n.startswith(local.lower()) for n in names))
        )
    # A run without ':' is either a growing name prefix or a growing namespace prefix
    # (any [\w.-] run may still take a ':' and become "<ns:name>").
    return True


def _is_partial_tag(s: str) -> bool:
    """Whether *s* (starting at ``<``, containing no ``>``) could still become a tool-call tag."""
    if len(s) > _MAX_PARTIAL_TAG:
        return False
    body = s[1:]
    closing = body.startswith("/")
    if closing:
        body = body[1:]
    if not body:
        return True
    m = _RUN_RE.match(body)
    run = m.group(0) if m else ""
    rest = body[len(run) :]
    names = _CLOSE_NAMES if closing else _OPEN_NAMES
    if rest:
        # a space or any other character ended the name portion; it is final now
        return _run_is_tag(run, names)
    return _run_may_grow(run, names)


def _classify_tag(tag: str) -> Tuple[Optional[str], Optional[str]]:
    """Classify one complete ``<…>`` span as ``("opener"|"closer"|None, name)``.

    Mirrors the final patterns: closers need ``>`` right after the name (no attributes);
    openers may carry attributes; the name word run is boundary-checked implicitly by the
    greedy match plus the membership test.
    """
    inner = tag[1:-1]
    closing = inner.startswith("/")
    body = inner[1:] if closing else inner
    m = _NAME_RE.match(body)
    if not m:
        return None, None
    ns, name = m.groups()
    name = name.lower()
    rest = body[m.end() :]
    if closing:
        if rest or name not in _CLOSE_NAMES:
            return None, None
        return "closer", name
    if name in _OPEN_NAMES:
        return "opener", name
    return None, None


def _is_named_function_opener(tag: str) -> bool:
    """Whether complete *tag* opens a named ``<function name=…>`` block.

    Same gate as ``_NAMED_FUNCTION_BLOCK_PATTERN``: a literal ``<function`` (no
    namespace prefix) carrying a ``name=`` attribute.
    """
    return _NAMED_FUNCTION_OPENER_RE.match(tag) is not None


def _is_arg_markup_tag(tag: str) -> bool:
    """Whether complete *tag* is GLM ``<arg_key>`` / ``<arg_value>`` markup.

    Same gate as ``_UNTERMINATED_TOOL_CALL_PATTERN``: open or close, attributes
    allowed, no namespace prefix.
    """
    return _ARG_MARKUP_TAG_RE.match(tag) is not None


def _partial_name_run(s: str) -> str:
    """The tag-name run of partial ``<…`` span *s* (``""`` when namespaced/closing-mangled)."""
    body = s[1:]
    if body.startswith("/"):
        body = body[1:]
    m = _RUN_RE.match(body)
    return m.group(0) if m else ""


def _is_partial_named_opener(s: str) -> bool:
    """Whether partial *s* (``<…``, no ``>``) may still grow a named-function opener.

    Only the ``<function`` + trailing-text shape needs this: every shorter prefix is
    already held by :func:`_is_partial_tag`, and any other fixed name can never qualify.
    """
    if len(s) > _MAX_PARTIAL_TAG or s[1:2] == "/":
        return False
    return _partial_name_run(s).lower() == "function"


def _is_partial_arg_tag(s: str) -> bool:
    """Whether partial *s* (``<…``, no ``>``) may still grow an arg-markup tag."""
    if len(s) > _MAX_PARTIAL_TAG:
        return False
    return _partial_name_run(s).lower() in _ARG_NAMES


def _anchored_overflow_name(s: str) -> Optional[str]:
    """The recognized opener name when over-cap partial *s* sits at a line anchor.

    Parsed exactly like :func:`_classify_tag` (namespace-aware, rest ignored), so a
    span that would classify as an opener once its ``>`` arrives enters block mode
    now. ``None`` for closers, unknown names, and nameless spans — those stay prose,
    like the final stripper keeps them.
    """
    body = s[1:]
    if body.startswith("/"):
        return None
    m = _NAME_RE.match(body)
    if not m:
        return None
    name = m.group(2).lower()
    return name if name in _OPEN_NAMES else None


class StreamingToolCallScrubber:
    """Chunk-safe suppression of text-channel tool-call XML (see the module docstring).

    ``feed()`` per delta; ``flush()`` at end of stream. Like the sibling scrubbers this is
    safe across intra-turn retries: ``flush()`` releases what is releasable and resets, so a
    later stream can feed again without ``reset()``.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Drop all state; called at the top of each turn by the agent."""
        self._pending: str = (
            ""  # held tail: a partial tag that may complete on the next feed
        )
        self._block_name: Optional[str] = (
            None  # name of the open block (None = not in a block)
        )
        self._block_raw: str = (
            ""  # raw span consumed inside the block (dropped on close)
        )
        self._block_anchored: bool = False  # opener sat at a line boundary
        self._skip_ws: bool = (
            False  # a stray closer went out: its trailing \s* is dropped
        )
        self._line_has_nonws: bool = False  # non-whitespace seen since the last newline
        self._prev_sig: str = (
            "\n"  # last consumed char outside [ \t] (the named-block boundary gate)
        )
        self._line_has_lt: bool = False  # '<' consumed since the last newline (arg-markup gate)

    # ── public API ─────────────────────────────────────────────────────────────

    def feed(self, text: str) -> str:
        """Return the visible portion of *text*; only a partial-tag tail is held back."""
        if not text:
            return ""
        buf = self._pending + text
        self._pending = ""
        out: List[str] = []
        while buf:
            if self._skip_ws:
                # Bare lstrip(): its unicode whitespace matches the final stripper's \s
                # (NBSP, EM SPACE, ...). The anchor state below intentionally stays ASCII:
                # _consume_line tracks line-blankness as " \t", like the final's [ \t]*.
                stripped = buf[: len(buf) - len(buf.lstrip())]
                # The run is whitespace the final stripper drops too, so the anchor
                # state must not see it as content — but a newline inside still
                # re-arms the named-function boundary gate behind it.
                for ch in stripped:
                    if ch not in " \t":
                        self._prev_sig = ch
                buf = buf.lstrip()
                if not buf:
                    break
                self._skip_ws = False
            if self._block_name is not None:
                buf = self._advance_in_block(buf, out)
            else:
                buf = self._advance(buf, out)
        return "".join(out)

    def flush(self) -> str:
        """End of stream: release a mid-line unresolved opener verbatim (the final stripper keeps
        it too); a line-anchored one is dropped (mirrors the unterminated strip), together
        with a held closer tail that belongs to the anchored block. Always resets,
        so an intra-turn retry's next stream can feed again."""
        out: List[str] = []
        if self._block_name is not None:
            if not self._block_anchored:
                # Mid-line opener (tool-call or named-function): the final stripper
                # keeps the raw span but still drops stray closers anywhere, so
                # release it with the strays suppressed (same contract as the final).
                out.append(_RELEASED_STRAY_CLOSER_RE.sub("", self._block_raw + self._pending))
            # Anchored (or arg-markup) block: dropped, INCLUDING the held tail — a
            # stream cut inside the closer is still the block, never prose.
        else:
            out.append(self._pending)
        self.reset()
        return "".join(out)

    # ── internals ──────────────────────────────────────────────────────────────

    def _consume_line(self, text: str) -> None:
        """Track whether the original text has non-whitespace since its last newline (the
        line-boundary anchor). Everything consumed counts, including suppressed spans. Only
        space/tab keep a line 'blank' — the final pattern's ``[ \\t]*``. Also tracks the
        last significant char (the named-function boundary gate) and whether a ``<`` was
        seen on this line (the arg-markup first-``<`` gate)."""
        for ch in text:
            if ch == "\n":
                self._line_has_nonws = False
                self._line_has_lt = False
                self._prev_sig = "\n"
            elif ch == "<":
                self._line_has_nonws = True
                self._line_has_lt = True
                self._prev_sig = "<"
            elif ch not in " \t":
                self._line_has_nonws = True
                self._prev_sig = ch

    def _at_function_boundary(self) -> bool:
        """Whether a named ``<function>`` opener here satisfies the final pattern's
        boundary gate (start of text, or right after a newline or sentence punctuation)."""
        return self._prev_sig in ("\n", "\r", ".", "!", "?", ":")

    def _block_closer_matches(self, tag: str) -> bool:
        """Whether *tag* closes the open block. Any same-name closer does — except a named
        ``<function name=…>`` block (``_block_name == "function"``, which no tool-call
        opener can produce), closed only by the literal ``</function>`` like the final
        pattern. An arg-markup block (``"arg"``) never matches: no closer classifies to it."""
        if self._block_name == "function":
            return tag.lower() == "</function>"
        return True

    def _hold_gated_partial(self, buf: str, i: int, s: str, out: List[str]) -> str:
        """Hold a partial named-opener / arg-tag tail when its gate passes, else release
        the ``<`` as plain text. The prefix is emitted first so the gate observes the
        same state it will see when the tag completes."""
        if i:
            self._consume_line(buf[:i])
            out.append(buf[:i])
        if (_is_partial_named_opener(s) and self._at_function_boundary()) or (
            _is_partial_arg_tag(s) and not self._line_has_lt
        ):
            self._pending = s
            return ""
        self._consume_line("<")
        out.append("<")
        return buf[i + 1 :]

    def _advance(self, buf: str, out: List[str]) -> str:
        """One outside-block step. Returns the remaining buffer; a hold goes to ``_pending``
        and returns ''."""
        i = buf.find("<")
        if i == -1:
            self._consume_line(buf)
            out.append(buf)
            return ""
        s = buf[i:]
        gt = s.find(">")
        if gt == -1:
            if not _is_partial_tag(s):
                if _is_partial_named_opener(s) or _is_partial_arg_tag(s):
                    return self._hold_gated_partial(buf, i, s, out)
                if i:
                    self._consume_line(buf[:i])
                    out.append(buf[:i])
                overflow = _anchored_overflow_name(s)
                if overflow is not None and not self._line_has_nonws:
                    # Recognized opener past the partial-tag cap at a line anchor: the
                    # final stripper drops from the anchor through end of text, so hold
                    # the span in (anchored) block mode instead of leaking it as prose.
                    # Mid-line overflow stays prose — the final keeps it too.
                    self._block_name = overflow
                    self._block_raw = s
                    self._block_anchored = True
                    self._consume_line(s)
                    return ""
                # a '<' that cannot become a tool-call tag: plain text, keep scanning after it
                self._consume_line("<")
                out.append("<")
                return buf[i + 1 :]
            if i:
                self._consume_line(buf[:i])
                out.append(buf[:i])
            self._pending = s
            return ""
        tag = s[: gt + 1]
        kind, name = _classify_tag(tag)
        if (
            kind is None
            and not _is_named_function_opener(tag)
            and not _is_arg_markup_tag(tag)
        ):
            # a '<' that cannot start a tool-call tag: plain text. A later '<' inside the span
            # may still start one, so emit through this character and keep scanning.
            self._consume_line(buf[: i + 1])
            out.append(buf[: i + 1])
            return buf[i + 1 :]
        if i:
            self._consume_line(buf[:i])
            out.append(buf[:i])
        if _is_named_function_opener(tag) and self._at_function_boundary():
            # Named <function name=…> block: suppressed through the literal </function>.
            # _block_anchored=False: the final stripper keeps an unterminated one.
            self._block_name = "function"
            self._block_raw = tag
            self._block_anchored = False
            self._consume_line(tag)
            return buf[i + gt + 1 :]
        if _is_arg_markup_tag(tag) and not self._line_has_lt:
            # GLM argument markup, first '<' on its line: the final stripper drops the
            # line through end of text, so swallow everything through flush.
            self._block_name = "arg"
            self._block_raw = tag
            self._block_anchored = True
            self._consume_line(tag)
            return buf[i + gt + 1 :]
        if kind is None:
            # A gated form whose gate failed (prose mention): plain '<', keep scanning.
            self._consume_line("<")
            out.append("<")
            return buf[i + 1 :]
        if kind == "opener":
            self._block_anchored = not self._line_has_nonws
            self._block_name = name
            self._block_raw = tag
            self._consume_line(tag)
            return buf[i + gt + 1 :]
        # stray closer: the final stripper keeps whitespace ahead of it and drops the closer's
        # own trailing whitespace run (\s*)
        self._skip_ws = True
        self._consume_line(tag)
        return buf[i + gt + 1 :]

    def _advance_in_block(self, buf: str, out: List[str]) -> str:
        """Inside an open block: everything is held as raw until the closer of the open name is
        complete. The raw span is kept verbatim so a mid-line opener can be released at flush."""
        j = 0
        while True:
            j = buf.find("<", j)
            if j == -1:
                self._block_raw += buf
                self._consume_line(buf)
                return ""
            s = buf[j:]
            gt = s.find(">")
            if gt != -1:
                kind, name = _classify_tag(s[: gt + 1])
                if (
                    kind == "closer"
                    and name == self._block_name
                    and self._block_closer_matches(s[: gt + 1])
                ):
                    self._block_raw += buf[:j]
                    self._consume_line(buf[:j])
                    self._consume_line(s[: gt + 1])
                    self._block_name = None
                    self._block_raw = ""
                    self._block_anchored = False
                    return buf[j + gt + 1 :]
                j += 1
                continue
            # no '>' at or after this '<': nothing here can close the block. Hold the earliest
            # tail that could still become the closer; anything before it is raw.
            k = j
            while k != -1:
                t = buf[k:]
                if _is_partial_tag(t):
                    self._block_raw += buf[:k]
                    self._consume_line(buf[:k])
                    self._pending = t
                    return ""
                k = buf.find("<", k + 1)
            self._block_raw += buf
            self._consume_line(buf)
            return ""

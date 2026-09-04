"""Shared detection/stripping for leaked OpenAI *harmony*-format reasoning.

Harmony models emit private reasoning in an ``analysis``/``commentary`` channel
and the user-facing answer in a ``final`` channel, delimited by control tokens::

    <|start|>assistant<|channel|>analysis<|message|>…reasoning…<|end|>
    <|start|>assistant<|channel|>final<|message|>…answer…<|return|>

When such a model is served via Ollama and Ollama's native thinking-parse
*misses*, that reasoning leaks into the visible message ``content`` instead of
the separate reasoning field. Two leak shapes are seen in the wild:

  (a) **degraded** — the leading control token is eaten, leaving a bare channel
      *name* word (``analysis``/``commentary``/``thought``) at text-start, the
      reasoning, then a lone (often pipe-stripped) ``<channel|>`` before the
      answer. This is the shape observed live 2026-07-11.
  (b) **canonical** — full/partial control tokens, e.g. content starting
      ``<|channel|>analysis<|message|>…`` with a later
      ``<|channel|>final<|message|>`` before the answer.

Two consumers share this logic so they stay in sync (the invariant noted at
``cli.py::_strip_reasoning_tags`` / ``run_agent.py::_strip_think_blocks``):

* :func:`strip_harmony_leak` — for a **complete** assistant message (the
  post-hoc / persisted path in ``strip_think_blocks`` and the CLI display path).
* :class:`HarmonyStreamGate` — a small state machine for the **streaming** path
  (``StreamingThinkScrubber``) so the leaked reasoning never flashes live
  mid-reply.  The channel-name *word* of a degraded head may flash; see the
  class docstring for why that is traded for leaving benign prose alone.

Design goal: strip a genuine leak without ever touching benign prose. Both
entry points refuse to do anything unless the text *begins* with harmony
structure, matched **case-sensitively** against the literal lowercase channel
names, and the bare-word form must stand alone (lowercase word followed by a
newline or a harmony token — not a colon). So ordinary prose like "Analysis of
Q2 revenue", a capitalised heading "Analysis\\n…", or a message that merely
*quotes* ``<|channel|>`` mid-sentence is left untouched.
"""

from __future__ import annotations

import re

__all__ = ["strip_harmony_leak", "HarmonyStreamGate"]

# Control tokens. Ollama's partial parse can drop a leading pipe, so accept the
# asymmetric ``<channel|>`` / ``<message|>`` forms too (justified by real data).
_CH = r"<\|?channel\|>"
_MSG = r"<\|?message\|>"

# The whole grammar is matched CASE-SENSITIVELY. Harmony control tokens and
# channel names are literal special tokens emitted verbatim lowercase
# (``analysis``/``commentary``/``final``) — exact case is positive evidence from
# the grammar itself, and it removes an entire false-positive class for free: an
# ordinary capitalised markdown heading like ``Analysis\n…`` or ``Commentary\n…``
# (even one that later quotes ``<|channel|>``) no longer looks like a channel head.

# The answer lives in the ``final`` channel: ``<|channel|>final<|message|>``.
# Requiring ``<|message|>`` right after the name means we never mistake a prose
# word "Final" (as in "Final answer: …") for the channel name.
_FINAL_MARKER = re.compile(rf"{_CH}\s*final\s*{_MSG}")

# Text *begins* with harmony structure: either an opening control token for an
# analysis/commentary/final channel (optionally preceded by ``<|start|>assistant``),
# or a bare standalone channel-name word on its own line (degraded shape (a) — the
# real leak is ``thought\n…``). The bare word must be followed by a NEWLINE or a
# harmony token — deliberately NOT a colon: a common benign heading like
# "analysis: …" (even one that later quotes ``<|channel|>``) must never match.
_HEAD = re.compile(
    rf"^\s*(?:"
    rf"(?:<\|?start\|>\s*assistant\s*)?{_CH}\s*(?:analysis|commentary|final)\b"
    rf"|(?:analysis|commentary|thought)\b[ \t]*(?:\n|{_CH}|{_MSG})"
    rf")",
)

# Whether the head opened with a control token (unambiguous harmony) vs a bare
# word (which could, rarely, be benign lowercase prose like "analysis\n…").
_HEAD_CONTROL = re.compile(rf"^\s*(?:<\|?start\|>|{_CH})")

# A lone channel token — the separator ending the reasoning block in shape (a).
_BARE_CH = re.compile(_CH)

# Stray lone control tokens to scrub from a recovered answer tail.
_STRAY = re.compile(r"<\|?(?:channel|message|start|end|return)\|>")


def strip_harmony_leak(text: str) -> str:
    """Remove a leaked harmony reasoning prefix from a complete message.

    Returns *text* unchanged unless it begins with harmony structure. When it
    does, keep only the final answer:

    * **canonical** — keep everything after the *last* ``final`` channel marker;
    * **degraded** — strip the leading reasoning up to and including the first
      lone ``<channel|>`` separator *after the head* (the following word is kept
      verbatim, so an answer that starts with "Final" survives).

    When the text opened with harmony structure but carries no ``final`` marker
    and no later separator, the resolution depends on the head shape:

    * **control-token head** — the whole message is analysis/commentary with no
      answer channel, so it is discarded (returns ``""``). Returning it verbatim
      would both *show* the leaked chain-of-thought and persist it de-tokenised
      into history — the exact leak this module exists to stop.
    * **bare-word head** — could be benign prose that merely opens with a
      lowercase channel-name word, so it is left untouched rather than risk
      eating a real answer.
    """
    if not text:
        return text
    head = _HEAD.match(text)
    if not head:
        return text
    # Canonical: answer follows the final-channel marker (handles analysis +
    # commentary + final in any order, and multiple blocks — keep the last).
    finals = list(_FINAL_MARKER.finditer(text))
    if finals:
        return _STRAY.sub("", text[finals[-1].end():]).strip()
    # Degraded: strip up to and including the first lone channel separator that
    # comes *after* the head — never the head's own control token, which would
    # only peel the delimiters off and leave the reasoning glued to the answer.
    sep = _BARE_CH.search(text, head.end())
    if sep:
        return _STRAY.sub("", text[sep.end():]).lstrip()
    # No answer channel and no separator: a control head is analysis-only with
    # nothing to keep → discard; a bare-word head might be benign → keep as-is.
    if _HEAD_CONTROL.match(text):
        return ""
    return text


# Bound on how much start-of-stream text we hold while deciding whether a stream
# is a harmony leak. Head detection resolves within a few chars; this only caps
# the pathological "opened with a control token but never a valid channel name".
_WATCH_MAX = 64

_CONTROL_ANCHORS = ("<|start|>", "<|channel|>", "<channel|>")
_NAME_WORDS = ("analysis", "commentary", "thought")


def _head_prefix_class(buf: str) -> str | None:
    """Which harmony head *buf* could still grow into: ``"control"``/``"word"``.

    ``None`` means it can no longer become a head at all.  The distinction is
    what drives the streaming tradeoff in :class:`HarmonyStreamGate`: a control
    prefix is worth holding back, a bare-word prefix is not.

    Case-sensitive, mirroring :data:`_HEAD`: a capitalised ``Analysis`` heading
    is not a channel name, so it is released to the normal machine immediately.
    """
    s = buf.lstrip()
    if s == "":
        # Whitespace only so far.  Either shape could still follow, but leading
        # whitespace is never the sensitive part of a leak, so classify it as a
        # word prefix and let it go out rather than holding the stream.
        return "word"
    if s[0] == "<":
        for a in _CONTROL_ANCHORS:
            if s.startswith(a) or a.startswith(s):
                return "control"
        return None
    for w in _NAME_WORDS:
        if w.startswith(s):  # strict prefix of the word ("analy")
            return "word"
        if s.startswith(w):  # the word, plus more
            rest = s[len(w):].lstrip(" \t")
            # Head needs a NEWLINE or a token start after the word; a colon
            # ("analysis:") or a letter ("thought experiments") means prose,
            # not a channel name — release it. Keeps parity with _HEAD.
            return "word" if (rest == "" or rest[0] in ("\n", "<")) else None
    return None


class HarmonyStreamGate:
    """Front stage for the streaming scrubber: suppress a leaked harmony prefix.

    ``feed(text)`` returns the text the tag machine should see, which is ``""``
    while the gate is holding or discarding.  :attr:`active` says whether the
    gate is still engaged; once it goes false, text flows straight through.

    The two head shapes are handled differently, because they carry very
    different false-positive risk:

    * **control-token head** (``<|channel|>analysis…``) — *hold, then decide*.
      No ordinary prose opens with ``<``, so buffering a few characters costs
      nothing.  This is what the surrounding
      :class:`~agent.think_scrubber.StreamingThinkScrubber` already does with a
      partial ``<think>`` prefix, so the behaviour is not new to this file.
    * **bare-word head** (``thought\\n…``) — *emit, then suppress*.
      ``analysis``/``commentary``/``thought`` are ordinary English words, so a
      stream opening with one is far likelier to be prose than a leak.  Holding
      it would delay and coalesce the deltas of that benign prose, which is a
      visible behaviour change on every such stream — upstream's own
      ``tests/run_agent/test_streaming.py::test_deltas_fire_in_order`` pins
      exactly that contract.  So the deltas go out as they arrive, and
      suppression begins at the head's end if it later confirms.

    The price of the second rule is that on a real degraded leak the
    channel-name word itself flashes in the live stream before suppression
    starts.  That is the whole cost: the reasoning body never reaches the wire,
    and the persisted and CLI paths still run :func:`strip_harmony_leak`, so the
    word does not survive into history either.  Traded knowingly — the word is
    not the secret, the leak is rare, and benign lowercase prose is not.

    A head that arrives whole before anything has been emitted is still
    suppressed whole: nothing is on the wire yet, so there is nothing to trade.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        # "watch"    — deciding.  Per delta, a control prefix is held and a
        #              bare-word prefix is emitted (see the class docstring).
        # "suppress" — head confirmed; discard until the separator.
        # "done"     — spent; text passes straight through.
        self._mode = "watch"
        self._buf = ""
        # How much of _buf has already gone downstream.  Non-zero only on the
        # emit path, and the reason the gate can mix holding and emitting
        # without ever releasing the same characters twice.
        self._sent = 0
        self._head_is_control = False

    @property
    def active(self) -> bool:
        return self._mode != "done"

    def feed(self, text: str) -> str:
        if self._mode == "done":
            return text
        self._buf += text
        out = ""

        if self._mode == "watch":
            head = _HEAD.match(self._buf)
            if head:
                self._head_is_control = bool(_HEAD_CONTROL.match(self._buf))
                # A bare-word head whose first characters are already on the
                # wire: release the rest of the word too, so what the user sees
                # is the whole word rather than a fragment that depends on where
                # the chunk boundaries happened to fall.  Nothing sent yet (or a
                # control head) means nothing to make good on — suppress it all.
                if self._sent and not self._head_is_control:
                    out = self._buf[self._sent:head.end()]
                self._buf, self._sent = self._buf[head.end():], 0
                self._mode = "suppress"
                # fall through to suppress resolution below
            else:
                cls = _head_prefix_class(self._buf)
                if cls is None or len(self._buf.lstrip()) > _WATCH_MAX:
                    # Not a harmony leak.  Release whatever has not gone out.
                    out = self._buf[self._sent:]
                    self._buf, self._sent, self._mode = "", 0, "done"
                    return out
                if cls == "control":
                    return ""  # hold: decide before emitting anything
                out = self._buf[self._sent:]  # bare word: emit as it arrives
                self._sent = len(self._buf)
                return out

        # suppress: discard the reasoning block until its separator arrives.
        # (Each feed re-scans the growing buffer — O(n²) over the block, but a
        # real leak is a few KB; a pathological multi-MB never-terminating block
        # is bounded by the read timeout, not this scan.)
        m = _FINAL_MARKER.search(self._buf)
        if m:
            out += self._buf[m.end():]
            self._buf, self._sent, self._mode = "", 0, "done"
            return out
        if not self._head_is_control:
            mb = _BARE_CH.search(self._buf)
            if mb:
                out += self._buf[mb.end():]
                self._buf, self._sent, self._mode = "", 0, "done"
                return out
        return out

    def flush(self) -> str:
        """End-of-stream: return any text the normal machine should still see.

        A *watch* buffer that never resolved was normal content that merely
        looked prefix-y → release whatever of it has not already gone out.  A
        *suppress* buffer that never found a separator is discarded only when
        the head was an unambiguous control token; a bare-word head (possibly
        benign ``analysis\\n…`` prose) is released rather than risk eating a
        real answer.
        """
        mode, buf, sent = self._mode, self._buf, self._sent
        control = self._head_is_control
        self._buf, self._sent, self._mode = "", 0, "done"
        if mode == "watch":
            return buf[sent:]
        if mode == "suppress":
            return "" if control else buf
        return ""

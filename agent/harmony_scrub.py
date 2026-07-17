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
  (``StreamingThinkScrubber``) so the leak never flashes live mid-reply.

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


def _could_be_head_prefix(buf: str) -> bool:
    """True while *buf* might still grow into a harmony head (keep watching).

    Case-sensitive, mirroring :data:`_HEAD`: a capitalised ``Analysis`` heading
    is not a channel name, so it is released to the normal machine immediately.
    """
    s = buf.lstrip()
    if s == "":
        return True
    # A real control token opened the stream → keep watching until it resolves.
    for a in _CONTROL_ANCHORS:
        if s.startswith(a) or a.startswith(s):
            return True
    for w in _NAME_WORDS:
        if w.startswith(s):  # strict prefix of the word ("analy")
            return True
        if s.startswith(w):  # the word, plus more
            rest = s[len(w):].lstrip(" \t")
            # Head needs a NEWLINE or a token start after the word; a colon
            # ("Analysis:") or a letter ("thought experiments") means prose,
            # not a channel name — release it. Keeps parity with _HEAD.
            return rest == "" or rest[0] in ("\n", "<")
    return False


class HarmonyStreamGate:
    """Front stage for the streaming scrubber: suppress a leaked harmony prefix.

    ``feed(text)`` returns ``(passthrough, done)``. While still deciding
    (*watch*) or discarding leaked reasoning (*suppress*) it returns
    ``("", False)`` and buffers internally. Once resolved it returns
    ``(remainder, True)`` — the answer tail to hand to the normal tag machine —
    and thereafter passes text straight through.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._mode = "watch"  # "watch" | "suppress" | "done"
        self._buf = ""
        self._head_is_control = False

    @property
    def active(self) -> bool:
        return self._mode != "done"

    def feed(self, text: str):
        if self._mode == "done":
            return text, True
        self._buf += text

        if self._mode == "watch":
            if _HEAD.match(self._buf):
                self._mode = "suppress"
                self._head_is_control = bool(_HEAD_CONTROL.match(self._buf))
                # fall through to suppress resolution below
            elif _could_be_head_prefix(self._buf) and len(self._buf.lstrip()) <= _WATCH_MAX:
                return "", False
            else:
                # Not a harmony leak — release everything to the normal machine.
                out, self._buf, self._mode = self._buf, "", "done"
                return out, True

        # suppress: discard the reasoning block until its separator arrives.
        # (Each feed re-scans the growing buffer — O(n²) over the block, but a
        # real leak is a few KB; a pathological multi-MB never-terminating block
        # is bounded by the read timeout, not this scan.)
        m = _FINAL_MARKER.search(self._buf)
        if m:
            out, self._buf, self._mode = self._buf[m.end():], "", "done"
            return out, True
        if not self._head_is_control:
            mb = _BARE_CH.search(self._buf)
            if mb:
                out, self._buf, self._mode = self._buf[mb.end():], "", "done"
                return out, True
        return "", False

    def flush(self) -> str:
        """End-of-stream: return any text the normal machine should still see.

        A *watch* buffer that never resolved was normal content that merely
        looked prefix-y → release it. A *suppress* buffer that never found a
        separator is discarded only when the head was an unambiguous control
        token; a bare-word head (possibly benign "Analysis:\\n…") is released
        rather than risk eating a real answer.
        """
        mode, buf, control = self._mode, self._buf, self._head_is_control
        self._buf, self._mode = "", "done"
        if mode == "watch":
            return buf
        if mode == "suppress":
            return "" if control else buf
        return ""

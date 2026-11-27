"""Sanitize user prompt text leaked from terminal / paste control sequences."""

from __future__ import annotations

import re

# Degraded visible bracketed-paste forms, matched only at boundaries so embedded literals stay intact.
_BOUNDARY_SUBS = (
    (re.compile(r"(^|[\s\n>:\]\)])\[200~"), r"\1"),
    (re.compile(r"\[201~(?=$|[\s\n<\[\(\):;.,!?])"), ""),
    (re.compile(r"(^|[\s\n>:\]\)])00~"), r"\1"),
    (re.compile(r"01~(?=$|[\s\n<\[\(\):;.,!?])"), ""),
)

# Corruption signature from desktop bracketed-paste leaks (#62557).
_DESKTOP_PASTE_ARTIFACT = "~[[e"


def strip_leaked_bracketed_paste_wrappers(text: str) -> str:
    """Strip leaked bracketed-paste wrapper markers: canonical wrappers unconditionally, degraded visible forms
    (``[200~`` / ``[201~`` and ``00~`` / ``01~``) only at boundaries so ``literal[200~tag`` stays intact."""
    if not text:
        return text
    for wrapper in ("\x1b[200~", "\x1b[201~", "^[[200~", "^[[201~"):
        text = text.replace(wrapper, "")
    for pattern, repl in _BOUNDARY_SUBS:
        text = pattern.sub(repl, text)
    return text


# CSI-u ("modifyOtherKeys"/kitty) key encodings that arrive INSIDE a bracketed
# paste's payload. tmux with `extended-keys always` re-encodes control keys it
# sees on the wire, and it does so to the paste stream too — where those bytes
# are CONTENT, not keys. prompt_toolkit inserts paste content verbatim, so a
# pasted newline lands in the prompt as the literal text "\x1b[106;5u"
# (Ctrl+J). The three that are really whitespace map back to whitespace; any
# other CSI-u in a payload is a key encoding that no real pasted text would
# contain, so it goes.
_CSI_U_RE = re.compile(r"\x1b\[(\d{1,7});(\d{1,4})u")
# modifier 5 == Ctrl (1 + 4), plus kitty's CapsLock/NumLock offsets (+64/+128/+192)
_CSI_U_CTRL_MODIFIERS = frozenset((5, 69, 133, 197))
_CSI_U_CTRL_TO_TEXT = {ord("j"): "\n", ord("m"): "\n", ord("i"): "\t"}


def decode_csi_u_in_paste(text: str) -> str:
    """Turn CSI-u key encodings inside a paste payload back into text."""
    if "\x1b[" not in text:
        return text

    def _sub(m: "re.Match[str]") -> str:
        codepoint, modifier = int(m.group(1)), int(m.group(2))
        if modifier in _CSI_U_CTRL_MODIFIERS:
            return _CSI_U_CTRL_TO_TEXT.get(codepoint, "")
        return ""

    return _CSI_U_RE.sub(_sub, text)


def collapse_repeated_input_artifacts(text: str, min_repeats: int = 4) -> str:
    """Drop a trailing run of the desktop ~[[e corruption signature (#62557)."""
    if not text:
        return text
    marker = _DESKTOP_PASTE_ARTIFACT
    index = len(text)
    repeat_count = 0
    while index >= len(marker) and text[index - len(marker) : index] == marker:
        repeat_count += 1
        index -= len(marker)
    if repeat_count < min_repeats:
        return text
    if index >= 2 and text[index - 2 : index] == "[e":
        index -= 2
    elif index >= 1 and text[index - 1] == "[":
        index -= 1
    return text[:index]


def sanitize_user_prompt_text(text: str) -> str:
    """Normalize user-authored prompt text before persistence or model input."""
    if not isinstance(text, str) or not text:
        return text
    return collapse_repeated_input_artifacts(strip_leaked_bracketed_paste_wrappers(text))

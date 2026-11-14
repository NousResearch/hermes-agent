"""Control/CSI-split prefix-token masking for agent.redact.

A credential split by control or zero-width chars (``sk-abc\\x1bdef…``, ``ghp_abc\\n123…``) or
wrapped in ANSI colour codes (``\\x1b[32msk-…\\x1b[0m``) escapes the contiguous prefix matcher.
Match on a shadow copy with that noise removed, then mask the corresponding span of the original.
"""

import re
from bisect import bisect_left

# Control / zero-width characters that can split a token body (``sk-abc\x1bdef``,
# ``ghp_abc\n123``) and escape the contiguous prefix regexes.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f\u200b-\u200f\u2028-\u202f\u2060\ufeff]")

# A complete escape sequence, in the tools/ansi_strip.py shapes: 7-/8-bit CSI, nF (``ESC ( B`` from
# ``tput sgr0``), SS2/SS3 + one char (``ESC O A``), and the cursor escapes ``ESC 7/8/=/>/M``.
# Deleting only the ESC byte leaves ``[32m`` / ``(B`` glued to the token head and its last letter
# defeats the prefix lookbehind, so the whole sequence must go (#81012). Matched only AT an
# introducer: a search from an orphan ESC would jump to a later sequence and treat the live text
# between as noise. Terminated strings (OSC/DCS) are not matched: an unterminated one would make
# every ESC scan to the end of the text.
_CSI_SEQ_RE = re.compile(
    r"(?:\x1b\[|\x9b)[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]"
    r"|\x1b(?:[\x20-\x2f]+[\x30-\x7e]|[NO][^\x00-\x1f\x7f]|[78=>M])"
)

# Union of every _PREFIX_PATTERNS body class: a shadow match may only span token-body chars or
# stripped noise. ``=`` is excluded so a KEY=value separator never lets a match span unrelated text.
_TOKEN_BODY_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-.")


def _strip_shadow(text: str) -> tuple:
    """``(shadow, noise, orig_idx)``: ``text`` minus complete CSI sequences and lone control chars;
    ``noise[i]`` is truthy for each stripped byte of ``text``; ``orig_idx`` maps shadow -> text."""
    shadow, noise, orig_idx = [], bytearray(len(text)), []
    i = 0
    while i < len(text):
        c = text[i]
        csi = _CSI_SEQ_RE.match(text, i) if c in "\x1b\x9b" else None
        if csi:
            noise[i:csi.end()] = b"\x01" * (csi.end() - i)
            i = csi.end()
            continue
        if _CONTROL_CHARS_RE.match(c):
            noise[i] = 1
        else:
            shadow.append(c)
            orig_idx.append(i)
        i += 1
    return "".join(shadow), noise, orig_idx


def mask_control_split_tokens(text: str, mask_fn, prefix_re: re.Pattern) -> str:
    """Mask ``prefix_re`` tokens split by control chars or CSI sequences.

    The original span is replaced only when its live chars are all token-body chars and it does
    not run into a ``KEY=`` name, so a match never swallows prose.
    """
    if _CONTROL_CHARS_RE.search(text) is None and "\x9b" not in text:
        return text
    shadow, noise, orig_idx = _strip_shadow(text)
    if len(shadow) == len(text):
        return text
    matches = []
    for m in prefix_re.finditer(shadow):
        body, body_start = m.group(1), m.start(1)
        start = orig_idx[body_start]
        end = orig_idx[m.end(1) - 1] + 1
        span = text[start:end]
        # A span crossing a LINE boundary whose fragment already self-matches is a complete token
        # followed by another line (``ghp_<tok>\nbutton`` must keep ``button``): never join across
        # it. Clip to the first line instead, and mask that when it is itself a token split by
        # noise (``sk-<head>\x1b<mid>\n…`` must not leak <mid>) or glued to stripped noise on its
        # left (``\x1b[32msk-<tok>\n…``: the escape's last letter defeats the ordinary pass's
        # lookbehind). A clean first line with a clean left edge is left to the ordinary pass.
        if ("\n" in span or "\r" in span) and prefix_re.search(span):
            line_end = bisect_left(orig_idx, start + min(i for i in (span.find("\n"), span.find("\r")) if i >= 0))
            body = body[:line_end - body_start]
            end = orig_idx[line_end - 1] + 1
            span = text[start:end]
            glued_left = start > 0 and noise[start - 1]
            if not (glued_left or any(noise[start:end])) or not prefix_re.fullmatch(body):
                continue
        if (all(noise[start + k] or c in _TOKEN_BODY_CHARS for k, c in enumerate(span))
                and (end >= len(text) or text[end] != "=")):
            matches.append((start, end, mask_fn(body)))
    out = list(text)
    for start, end, replacement in reversed(matches):
        out[start:end] = list(replacement)
    return "".join(out)

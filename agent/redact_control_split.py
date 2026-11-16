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

# Union of every _PREFIX_PATTERNS body class (incl. the ``/`` of Google's ``1//0``): a shadow match
# may only span token-body chars or stripped noise. ``=`` is excluded so a KEY=value separator never
# lets a match span unrelated text; only trailing base64 padding (``glc_…==``) may hold it.
_TOKEN_BODY_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-.+/")


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


def mask_control_split_tokens(text: str, mask_fn, prefix_re: re.Pattern,
                              sub_re: re.Pattern | None = None, skip=None) -> str:
    """Mask ``prefix_re`` tokens split by control chars or CSI sequences.

    The original span is replaced only when its live chars are all token-body chars and it does
    not run into a ``KEY=`` name, so a match never swallows prose. ``sub_re`` (default
    ``prefix_re``) finds the candidates, so a key glued to an identifier (``log-1//0…``) is found
    too; ``skip(m, field)`` drops the candidates the contiguous pass would keep (``scikit-learn``),
    judged on ``field``: the live text up to the first tab or line break, which the contiguous
    pass never reads across (``task_sk-learn<TAB>Installed2024Version`` stays whole).
    """
    if _CONTROL_CHARS_RE.search(text) is None and "\x9b" not in text:
        return text
    shadow, noise, orig_idx = _strip_shadow(text)
    if len(shadow) == len(text):
        return text
    spans = []
    # ``sub_re`` finds glued keys (``log-1//0…``), but one skipped glued run (``my-sk-learn-config.``
    # + a stripped ``\n``) can swallow a key that ``prefix_re``'s own scan finds, so both scans
    # run and every span either masks is kept: never less than either alone. Overlapping spans
    # are masked as one.
    for pattern, skip_fn in ((prefix_re, None), (sub_re, skip)) if sub_re is not None else ((prefix_re, None),):
        spans.extend(_candidate_spans(text, shadow, noise, orig_idx, pattern, prefix_re, skip_fn))
    if not spans:
        return text
    if sub_re is not None:
        # A merged span can end inside a plain glued key the contiguous pass would mask
        # (``1//0<split>…my-sk-learn-config.<key>``), leaving it a tail it no longer recognises:
        # take such a key into the span. Keys clear of every span are left to that pass.
        spans.extend((m.start(1), m.end(1), True) for m in _iter_with_swallowed(sub_re, text)
                     if skip is None or not skip(m, m.group(1)))
    out, pos = [], 0
    for start, end, plain in _union(spans):
        if not plain:
            out.append(text[pos:start])
            out.append(mask_fn("".join(c for k, c in enumerate(text[start:end]) if not noise[start + k])))
            pos = end
    out.append(text[pos:])
    return "".join(out)


def _candidate_spans(text, shadow, noise, orig_idx, pattern, prefix_re, skip):
    """``(start, end, False)`` spans of ``text`` to mask for each ``pattern`` match on ``shadow``."""
    for m in _iter_with_swallowed(pattern, shadow):
        body, body_start = m.group(1), m.start(1)
        start = orig_idx[body_start]
        end = orig_idx[m.end(1) - 1] + 1
        span = text[start:end]
        if skip is not None and skip(m, "".join(
                c for k, c in enumerate(re.split(r"[\t\n\r]", span, maxsplit=1)[0]) if not noise[start + k])):
            continue
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
        if (all(noise[start + k] or c in _TOKEN_BODY_CHARS for k, c in enumerate(span.rstrip("=")))
                and (end >= len(text) or text[end] != "=")):
            yield start, end, False


def _iter_with_swallowed(pattern: re.Pattern, text: str):
    """``pattern.finditer(text)`` plus each match that starts inside a yielded one and runs past
    its end: ``1//0…log-sk-abc.<key>`` ends the ``1//0`` match at the ``.``, but the ``sk-`` inside
    it runs on through ``<key>``, which finditer then never sees. The search from inside each
    match steps over the matches it contains (``sk-…-hf_<key>-x_1//0…``) and stops at the first
    one that runs past it or starts after it; each step starts past the last, so the scan stays
    linear."""
    for m in pattern.finditer(text):
        yield m
        pos = m.start(1) + 1
        while (inner := pattern.search(text, pos)) is not None and inner.start(1) < m.end(1):
            if inner.end(1) > m.end(1):
                yield inner
                break
            pos = inner.end(1)


def _union(spans) -> list:
    """Sorted ``(start, end, flag)`` spans with overlaps merged; a merged flag is ``all`` of its parts."""
    merged = []
    for start, end, flag in sorted(spans):
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]), merged[-1][2] and flag)
        else:
            merged.append((start, end, flag))
    return merged


def mask_prefix_matches(text: str, mask_fn, sub_re: re.Pattern, skip) -> str:
    """Mask every ``sub_re`` match (and swallowed one, see ``_iter_with_swallowed``) that ``skip``
    keeps out; overlapping matches are masked as one span."""
    out, pos = [], 0
    for start, end, _ in _union((m.start(1), m.end(1), True) for m in _iter_with_swallowed(sub_re, text) if not skip(m)):
        out += [text[pos:start], mask_fn(text[start:end])]
        pos = end
    out.append(text[pos:])
    return "".join(out)

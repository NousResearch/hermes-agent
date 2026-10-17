"""Asterisk emphasis for the legacy Telegram MarkdownV2 converter."""

import re
from collections.abc import Callable
from itertools import pairwise
from typing import NamedTuple

from markdown_it import MarkdownIt
from markdown_it.rules_inline.emphasis import tokenize
from markdown_it.rules_inline.escape import escape as tokenize_escape


class _Scanned(NamedTuple):
    can_open: bool
    can_close: bool
    length: int


def _asterisks_only(state, silent):
    if state.src[state.pos] != "*":
        return False

    original_scan_delims = state.scanDelims

    def _scan_delims(start, can_split_word):
        scanned = original_scan_delims(start, can_split_word)
        if scanned.length < 2:
            return scanned
        last_char = state.src[start - 1] if start else " "
        end = start + scanned.length
        next_char = state.src[end] if end < state.posMax else " "
        # Telegram's legacy Markdown path accepts strong spans adjacent to
        # punctuation and CJK text. CommonMark's word-boundary rule rejects
        # those delimiters, so relax only double-asterisk runs while retaining
        # whitespace as the literal-star boundary.
        return _Scanned(
            can_open=not next_char.isspace(),
            can_close=not last_char.isspace(),
            length=scanned.length,
        )

    state.scanDelims = _scan_delims
    try:
        return tokenize(state, silent)
    finally:
        state.scanDelims = original_scan_delims


def _emphasis_escape(state, silent):
    # Only asterisk and backslash are emphasis-relevant escapes here. Other
    # backslash-punctuation sequences stay literal text so the final
    # MarkdownV2 pass escapes the backslash itself, matching the converter's
    # established behavior for already-escaped input.
    return state.src[state.pos : state.pos + 2] in (r"\*", r"\\") and tokenize_escape(
        state, silent
    )


_PARSER = MarkdownIt("commonmark")
_PARSER.inline.ruler.enableOnly(["text", "escape", "emphasis"])
_PARSER.inline.ruler.at("emphasis", _asterisks_only)
_PARSER.inline.ruler.at("escape", _emphasis_escape)


def protect_asterisk_emphasis(
    text: str,
    protect: Callable[[str], str],
    escape: Callable[[str], str],
    *,
    block_source: str,
) -> str:
    """Resolve emphasis within blocks, protecting complete spans from later passes."""
    # Reuse the configured parser: parse creates per-call state and the ruler
    # configuration is immutable after module initialization.
    parser = _PARSER
    # Use block source maps only: do not normalize or render the block structure,
    # which would change whitespace, bullets, and the caller's NUL placeholders.
    blocks = []
    parser.block.parse(block_source, parser, {}, blocks)
    # markdown-it block maps count lines by "\n" only. str.splitlines() would
    # also split on \r, \x0c, \x85 and \u2028, shifting every later boundary
    # and letting emphasis cross paragraph breaks (or missing real spans).
    raw_lines = text.split("\n")
    lines = [line + "\n" for line in raw_lines[:-1]] + [raw_lines[-1]]
    boundaries = {0, len(lines)}
    for block in blocks:
        if block.type == "inline" and block.map:
            boundaries.update(block.map)
    boundaries = sorted(boundaries)
    parts = []
    for start, end in pairwise(boundaries):
        tokens = []
        parser.inline.parse("".join(lines[start:end]), parser, {}, tokens)
        depth = {"*": 0, "**": 0}
        span = []
        line_start = True
        for token in tokens:
            if token.nesting:
                line_start = False
                marker = "*" if token.markup == "**" else "_"
                if token.nesting == 1:
                    if depth[token.markup] == 0:
                        span.append(marker)
                    depth[token.markup] += 1
                else:
                    depth[token.markup] -= 1
                    if depth[token.markup] == 0:
                        span.append(marker)
                    if not any(depth.values()):
                        parts.append(protect("".join(span)))
                        span = []
            else:
                content = token.content
                if any(depth.values()):
                    # Quote continuation prefixes are structural, even inside a span.
                    # A prefix at a token boundary is structural only at line start.
                    prefix = "" if line_start else "x"
                    quoted = re.sub(
                        r"(?m)^(>{1,3}) ",
                        lambda m: protect(m.group(1)) + " ",
                        prefix + content,
                    )
                    span.append(escape(quoted[len(prefix) :]))
                elif token.type == "text_special":
                    parts.append(protect(escape(content)))
                else:
                    parts.append(content)
                if content:
                    line_start = content.endswith("\n")
    return "".join(parts)

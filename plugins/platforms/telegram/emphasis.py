"""Asterisk emphasis for the legacy Telegram MarkdownV2 converter."""

import re
from collections.abc import Callable

from markdown_it import MarkdownIt
from markdown_it.rules_inline.emphasis import tokenize
from markdown_it.rules_inline.escape import escape as tokenize_escape


def _asterisks_only(state, silent):
    return state.src[state.pos] == "*" and tokenize(state, silent)


def _emphasis_escape(state, silent):
    return state.src[state.pos:state.pos + 2] in (r"\*", r"\\") and tokenize_escape(state, silent)


def protect_asterisk_emphasis(
    text: str, protect: Callable[[str], str], escape: Callable[[str], str],
    *, block_source: str,
) -> str:
    """Resolve emphasis within blocks, protecting complete spans from later passes."""
    parser = MarkdownIt("commonmark")
    parser.inline.ruler.enableOnly(["text", "escape", "emphasis"])
    parser.inline.ruler.at("emphasis", _asterisks_only)
    parser.inline.ruler.at("escape", _emphasis_escape)
    # Use block source maps only: do not normalize or render the block structure,
    # which would change whitespace, bullets, and the caller's NUL placeholders.
    blocks = []
    parser.block.parse(block_source, parser, {}, blocks)
    lines = text.splitlines(keepends=True)
    boundaries = {0, len(lines)}
    for block in blocks:
        if block.type == "inline" and block.map:
            boundaries.update(block.map)
    boundaries = sorted(boundaries)
    parts = []
    for start, end in zip(boundaries, boundaries[1:]):
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
                    quoted = re.sub(r"(?m)^(>{1,3}) ",
                                    lambda m: protect(m.group(1)) + " ", prefix + content)
                    span.append(escape(quoted[len(prefix):]))
                elif token.type == "text_special":
                    parts.append(protect(escape(content)))
                else:
                    parts.append(content)
                if content:
                    line_start = content.endswith("\n")
    return "".join(parts)

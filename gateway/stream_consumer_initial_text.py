"""Content boundaries for the first visible streaming preview, not a text rewrite."""

from __future__ import annotations

import re

_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_HEADING = re.compile(r"^ {0,3}#{1,6}(?:\s|$)")
_SETEXT = re.compile(r"^ {0,3}(?:=+|-+)\s*$")
_LIST = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+")
_INLINE = re.compile(
    r"(?P<ticks>`+)(?P<code>.*?)(?P=ticks)"
    r"|(?P<open_code>`[^\n]*$)"
    r"|!?\[(?P<label>[^\]\n]*)\]\([^\n)]*(?:\)|$)"
    r"|(?P<url>(?:https?://|www\.)\S*|<[^>\n]*(?:>|$))"
)
_SENTENCE = re.compile(r'''(?:[.!?…]+["'’”»)*_\]]*(?=\s)|[。！？]+)''')
_ABBREVIATIONS = frozenset({
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
    "m", "mme", "mlle", "ex", "p.ex",
})
_TERMINATORS = str.maketrans({char: "x" for char in ".!?。！？…"})


def _inline_content(text: str) -> str:
    """Keep visible content length, but never treat code/URL punctuation as prose."""
    def protected(match: re.Match) -> str:
        content = next(value for value in (
            match["code"], match["open_code"], match["label"], match["url"],
        ) if value is not None)
        return content.strip("`").translate(_TERMINATORS)

    return re.sub(r"[*_~]", "", _INLINE.sub(protected, text))


def _sentence_ready(content: str, min_chars: int) -> bool:
    for match in _SENTENCE.finditer(content):
        prefix = content[:match.end()].strip()
        if len(" ".join(prefix.split())) < min_chars:
            continue
        # A trailing period needs lookahead whitespace: a delta ending in "3."
        # may still become "3.14". Common titles/initials aren't sentence ends.
        if content[match.start():match.start() + 1] == ".":
            word = re.search(r"([\w.]+)$", content[:match.start()])
            token = word[1] if word else ""
            if (token.lower() in _ABBREVIATIONS or (len(token) == 1 and token.isalpha())
                    or re.fullmatch(r"(?:[A-Za-z]\.)+[A-Za-z]", token)):
                continue
        return True
    return False


def initial_text_ready(text: str, min_chars: int) -> bool:
    """Enough completed content for an initial preview, independent of delta batching.

    Prose releases at a sentence boundary after the minimum cumulative content.
    Non-prose can release at a completed paragraph, list item or fenced block.
    Headings alone don't qualify. No punctuation/delimiter means the caller holds
    until its existing final/segment flush; no timer or size-only escape hatch.
    This is deliberately a conservative heuristic, not a language parser.
    """
    if min_chars <= 0:
        return True
    content = ""
    fence = ""
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        marker = _FENCE.match(line.rstrip("\r\n"))
        if fence:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = ""
                if len(" ".join(content.split())) >= min_chars:
                    return True
            else:
                # Preserve code as content for the threshold, never as sentence syntax.
                content += line.translate(_TERMINATORS)
            continue
        if marker:
            fence = marker[1]
            continue
        if _HEADING.match(line) or _SETEXT.fullmatch(line.rstrip("\r\n")):
            continue
        if index + 1 < len(lines) and _SETEXT.fullmatch(lines[index + 1].rstrip("\r\n")):
            continue
        if not line.strip():
            if len(" ".join(content.split())) >= min_chars:
                return True
            continue
        line = re.sub(r"^\s*>\s?", "", line)
        item = _LIST.match(line)
        content += _inline_content(line[item.end():] if item else line)
        if _sentence_ready(content, min_chars):
            return True
        if item and line.endswith(("\n", "\r")) and len(" ".join(content.split())) >= min_chars:
            return True
    return False

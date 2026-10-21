"""Token-aware colourisation for the CLI's reasoning/thinking text.

Thinking text used to be painted with one uniform ``_DIM`` sequence, so an order
mark, a log stamp, a PR number and ordinary prose were indistinguishable in the
scrollback. This module splits a line of thinking text into spans and gives each
class its own foreground: prose stays in the terminal's Homebrew green, order
marks go orange, and log stamps / PR numbers / URLs go white.

Every rule is deliberately narrow so the lookalikes stay prose: the order-mark
rule needs trailing whitespace (``3.14``, ``v1.0.1`` and ``2.5 hours`` do not
match), the PR rule needs six digits (``#12345`` does not match), and a URL span
ends before the sentence punctuation that hugs it.

:class:`split_safe_stream` is the streaming half: a token that has only partly
arrived renders as plain prose rather than guessing, and the live box cuts its
buffer before the oldest such fragment so no token is ever painted twice under
two different colours.
"""

from __future__ import annotations

import re

# Homebrew Terminal profile TextColor (NSRGB 0.1569 / 0.996 / 0.0784).
SHOWCASE_GREEN = "#28FE14"
SHOWCASE_ORANGE = "#FF9F0A"
SHOWCASE_WHITE = "#FFFFFF"


def _sgr(hex_color: str) -> str:
    """True-colour SGR sequence for '#RRGGBB'."""
    red, green, blue = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"\x1b[38;2;{red};{green};{blue}m"


_GREEN = _sgr(SHOWCASE_GREEN)
_ORANGE = _sgr(SHOWCASE_ORANGE)
_WHITE = _sgr(SHOWCASE_WHITE)
_RST = "\x1b[0m"

_COLORS = {"white": _WHITE, "orange": _ORANGE}

# Alternation order is the precedence order: a log stamp is claimed whole (so
# ``§[#123456]`` is one white span, not white-inside-white), and a URL is claimed
# before the PR rule so a ``#fragment`` never splits off on its own.
_TOKENS = re.compile(
    r"(?P<log>§\s*\[[^\]\n]*\])"
    r"|(?P<url>https?://[^\s]+)"
    r"|(?P<pr>\#\d{6,})"
    r"|(?P<mark>\b\d{1,3}\.(?=\s))"
)

_URL_TRAILING = ".,;:!?'\""
_URL_BRACKETS = (("(", ")"), ("[", "]"), ("{", "}"))


def _split_url(url: str) -> tuple[str, str]:
    """Peel sentence punctuation off a URL span so it renders as prose."""
    end = len(url)
    while end:
        char = url[end - 1]
        if char in _URL_TRAILING:
            end -= 1
            continue
        unbalanced = any(
            char == closer and url.count(opener) < url.count(closer)
            for opener, closer in _URL_BRACKETS
        )
        if not unbalanced:
            break
        end -= 1
    return url[:end], url[end:]


def thinking_spans(text: str) -> list[tuple[str, str | None]]:
    """Split thinking text into ``(chunk, colour)`` pairs.

    ``colour`` is ``None`` for ordinary prose, which the renderer paints in the
    showcase green. Adjacent same-coloured chunks are merged so the output
    carries one escape per run rather than one per token.
    """
    spans: list[tuple[str, str | None]] = []
    position = 0
    for match in _TOKENS.finditer(text or ""):
        if match.start() > position:
            spans.append((text[position:match.start()], None))
        kind = match.lastgroup
        raw = match.group()
        if kind == "url":
            body, tail = _split_url(raw)
            spans.append((body, "white"))
            if tail:
                spans.append((tail, None))
        elif kind == "mark":
            spans.append((raw, "orange"))
        else:
            spans.append((raw, "white"))
        position = match.end()
    if position < len(text or ""):
        spans.append((text[position:], None))

    merged: list[tuple[str, str | None]] = []
    for chunk, color in spans:
        if merged and merged[-1][1] == color:
            merged[-1] = (merged[-1][0] + chunk, color)
        else:
            merged.append((chunk, color))
    return merged


def render_thinking_text(text: str) -> str:
    """Colourise a finished piece of thinking text."""
    return "".join(
        f"{_COLORS.get(color, _GREEN)}{chunk}{_RST}"
        for chunk, color in thinking_spans(text)
        if chunk
    )


# Fragments that could still grow into a token: holding them back keeps a
# half-arrived ``§[2026-10-0``, ``#12345`` or ``https://x.co`` plain prose
# instead of committing to a colour that the rest of the token may contradict.
_PENDING = (
    re.compile(r"§\s*\[[^\]\n]*$"),
    re.compile(r"§\s*$"),
    re.compile(r"\#\d{0,5}(?:\b|$)"),
    re.compile(r"\bhttps?://[^\s]*$"),
    re.compile(r"\b(?:h|ht|htt|http|https|https:|https:/)$"),
    re.compile(r"\b\d{1,3}\.$"),
)


def split_safe_stream(buffered: str) -> tuple[str, str]:
    """Split ``buffered`` into (safe to paint now, keep buffering).

    The cut lands before the oldest still-arriving token, so a token is painted
    exactly once, by the call that saw it complete.
    """
    cut = len(buffered)
    for pattern in _PENDING:
        match = pattern.search(buffered)
        if match is not None and match.start() < cut:
            cut = match.start()
    return buffered[:cut], buffered[cut:]
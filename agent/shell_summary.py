"""Character-level helpers behind ``agent.display.summarize_shell_command``, memoized.

A transcript projection previews every terminal row through the summary, and the summary walks the command
once per helper: split the words, split the compounds, strip a pipe tail, drop redirections, name the head
word, spot a boundary echo. A long session projects again on every ``session.activate``,
``session.history`` and ``session.transcript`` page, so the same commands were re-parsed from scratch each
time — 9M scan steps and ~4 s for one 2,420 command session, with the rest of the gateway stalled behind
that GIL-holding work (#154513).

Every helper here is a pure function of its string argument, so each one memoizes with a bounded LRU: a warm
projection re-parses nothing, and the cache cannot grow with a session's command set. The summary itself
stays in ``agent.display``: it localizes its text (``t()``), so its result is not a function of the command
alone. The cold cost is unchanged; a single-pass tokenizer is the next step if that ever matters.
"""
from __future__ import annotations

import functools
import re
from typing import Iterator

#: A reported 2,420-command transcript needs room for each command and its parsed segments (#154513).
MEMO_SIZE = 8192
#: Pipeline stages whose tail is dropped from a preview (the pipeline's own output is not the command).
PIPE_TAIL_HEADS = frozenset({"head", "tail", "wc", "sort", "uniq"})


def _shell_basename(head: str) -> str:
    return head.rsplit("/", 1)[-1] if head else ""


def _scan_quoted(text: str) -> Iterator[tuple[int, str, bool]]:
    """Yield ``(index, char, inside_quotes)``; a quote closes unless backslash-escaped."""
    quote: str | None = None
    for i, ch in enumerate(text):
        if quote:
            yield i, ch, True
            if ch == quote and (i == 0 or text[i - 1] != "\\"):
                quote = None
        elif ch in {"'", '"'}:
            quote = ch
            yield i, ch, True
        else:
            yield i, ch, False


@functools.lru_cache(maxsize=MEMO_SIZE)
def _words(segment: str) -> tuple[str, ...]:
    """The segment's words, split outside quotes. A tuple: a cached list would be shared and mutable."""
    parts: list[list[str]] = [[]]
    for _, ch, quoted in _scan_quoted(segment):
        if not quoted and ch.isspace():
            parts.append([])
        else:
            parts[-1].append(ch)
    return tuple("".join(part) for part in parts if part)


@functools.lru_cache(maxsize=MEMO_SIZE)
def strip_pipe_tail(segment: str) -> str:
    """Drop a trailing ``| head/tail/wc/sort/uniq ...`` pipeline stage."""
    words = list(_words(segment))
    for i, word in enumerate(words):
        if word == "|" and _shell_basename(words[i + 1] if i + 1 < len(words) else "") in PIPE_TAIL_HEADS:
            words = words[:i]
            break
    return " ".join(words).strip()


@functools.lru_cache(maxsize=MEMO_SIZE)
def split_compound(command: str) -> tuple[str, ...]:
    """Split on unquoted ``&&`` / ``||`` / ``;`` / newline, dropping pipe tails per segment."""
    raw: list[list[str]] = [[]]
    skip = False
    for i, ch, quoted in _scan_quoted(command):
        if skip:
            skip = False
        elif not quoted and (command.startswith("&&", i) or command.startswith("||", i)):
            raw.append([])
            skip = True
        elif not quoted and ch in {";", "\n"}:
            raw.append([])
        else:
            raw[-1].append(ch)
    segments = (strip_pipe_tail("".join(buf).strip()) for buf in raw)
    return tuple(segment for segment in segments if segment)


@functools.lru_cache(maxsize=MEMO_SIZE)
def head_word(segment: str) -> str:
    """Command name of a segment, skipping leading ``VAR=value`` assignments."""
    words = list(_words(segment))
    while words and re.match(r"^[A-Za-z_]\w*=", words[0]):
        words.pop(0)
    return _shell_basename(words[0] if words else "")


@functools.lru_cache(maxsize=MEMO_SIZE)
def clean_segment(segment: str) -> str:
    """Drop redirections (``> file``, ``2>&1``) from a segment."""
    words = _words(segment)
    out: list[str] = []
    i = 0
    while i < len(words):
        word = words[i]
        if re.match(r"^\d*(?:>>?|<)$", word):
            i += 2  # operator + target
        elif re.match(r"^\d*(?:>&|<&)\d+$", word):
            i += 1
        else:
            out.append(word)
            i += 1
    return " ".join(out).strip()


@functools.lru_cache(maxsize=MEMO_SIZE)
def is_boundary_echo(segment: str) -> bool:
    """A shell echo of boundary state (``--- exit=0 ---``, ``echo $?``): plumbing, never the command."""
    words = _words(segment)
    if _shell_basename(words[0] if words else "") != "echo":
        return False
    return bool(re.search(r"-{2,}|_exit=|(?:^|\s|=)\$[?{]|PIPESTATUS", " ".join(words[1:])))

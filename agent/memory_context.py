"""Memory-context fencing: how recalled memory is wrapped for the model and kept out of output.

``build_memory_context_block`` fences prefetched recall for the prompt, leaving out bullets that
``window_recall_lines`` finds already replaying in the window; ``sanitize_context`` and
``StreamingContextScrubber`` strip those fences (and anything a provider pre-wrapped) from text that
leaves the agent. Pure text formatting: no provider calls, no profile state.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable, Iterator, Optional

logger = logging.getLogger(__name__)

_FENCE_TAG_RE = re.compile(r'</?\s*memory-context\s*>', re.IGNORECASE)
_INTERNAL_CONTEXT_RE = re.compile(r'<\s*memory-context\s*>[\s\S]*?</\s*memory-context\s*>', re.IGNORECASE)
_INTERNAL_NOTE_RE = re.compile(
    r'\[System note:\s*The following is recalled memory context,\s*NOT new user input\.\s*Treat as (?:informational background data|authoritative reference data[^\]]*)\.\]\s*',
    re.IGNORECASE,
)


def sanitize_context(text: str) -> str:
    """Strip fence tags, injected context blocks, and system notes from provider output."""
    for pattern in (_INTERNAL_CONTEXT_RE, _INTERNAL_NOTE_RE, _FENCE_TAG_RE):
        text = pattern.sub('', text)
    return text


class StreamingContextScrubber:
    """Stateful scrubber for streaming text whose memory-context spans may straddle deltas.

    ``sanitize_context`` needs both tags in one string, so a split span would leak to the UI;
    this holds back partial-tag tails between ``feed()`` calls and drops span interiors.
    One scrubber (or ``reset()``) per top-level response; call ``flush()`` at end of stream.
    """

    _OPEN_TAG = "<memory-context>"
    _CLOSE_TAG = "</memory-context>"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._in_span: bool = False
        self._buf: str = ""
        self._at_block_boundary: bool = True

    def feed(self, text: str) -> str:
        """Return the visible portion of ``text``; a possible partial tag tail is held for the next call."""
        if not text:
            return ""
        buf = self._buf + text
        self._buf = ""
        out: list[str] = []
        while buf:
            if self._in_span:
                tag = self._CLOSE_TAG
                idx = buf.lower().find(tag)
                held = self._max_partial_suffix(buf, tag)  # potential partial close tag
            else:
                tag = self._OPEN_TAG
                idx = self._find_boundary_open_tag(buf)
                # A complete boundary tag at the buffer end is held until the next char confirms it.
                n = len(tag)
                pending = n if buf.lower().endswith(tag) and self._ends_at_block_boundary(buf[:-n]) else 0
                held = pending or self._max_partial_suffix(buf, tag)
            if idx == -1:
                # Hold back the possible partial tag; inside a span the rest is dropped.
                if not self._in_span:
                    self._append_visible(out, buf[:-held] if held else buf)
                self._buf = buf[-held:] if held else ""
                break
            if not self._in_span:
                self._append_visible(out, buf[:idx])
            buf = buf[idx + len(tag):]
            self._in_span = not self._in_span
        return "".join(out)

    def flush(self) -> str:
        """Emit the held-back tail at end-of-stream; inside an unterminated span it is discarded
        (leaking partial memory context is worse than a truncated answer)."""
        tail = "" if self._in_span else self._buf
        self._buf = ""
        self._in_span = False
        return tail

    @staticmethod
    def _max_partial_suffix(buf: str, tag: str) -> int:
        """Length of the longest buf-suffix that is a (case-insensitive) prefix of ``tag``, else 0."""
        tag_lower, buf_lower = tag.lower(), buf.lower()
        span = range(min(len(buf_lower), len(tag_lower) - 1), 0, -1)
        return next((i for i in span if tag_lower.startswith(buf_lower[-i:])), 0)

    def _find_boundary_open_tag(self, buf: str) -> int:
        """Find an opening fence only when it starts a block-like span (own line, newline after)."""
        buf_lower, tag_len = buf.lower(), len(self._OPEN_TAG)
        idx = buf_lower.find(self._OPEN_TAG)
        while idx != -1:
            after_idx = idx + tag_len
            if self._ends_at_block_boundary(buf[:idx]) and after_idx < len(buf) and buf[after_idx] in "\r\n":
                return idx
            idx = buf_lower.find(self._OPEN_TAG, idx + 1)
        return -1

    def _ends_at_block_boundary(self, text: str) -> bool:
        """Whether emitting ``text`` leaves the stream at a line start (blank tail after the last newline;
        no newline at all -> only whitespace and already at a boundary)."""
        _head, sep, tail = text.rpartition("\n")
        return tail.strip() == "" and (bool(sep) or self._at_block_boundary)

    def _append_visible(self, out: list[str], text: str) -> None:
        if text:
            out.append(text)
            self._at_block_boundary = self._ends_at_block_boundary(text)


# A markdown bullet: a marker, whitespace, then content. The whitespace matters — it is what keeps
# ``**Preferences**`` (a bold heading) and ``*emphasis*`` out of the rule.
_RECALL_BULLET_RE = re.compile(r"[-*+]\s+\S")

# A recall line as window dedupe matches it: the chain of column-0 non-bullet lines above it in its block
# (headings, rules, prose, provider framing), then the line itself.
RecallKey = tuple[tuple[str, ...], str]


def _carries_continuation(lines: list[str], index: int) -> bool:
    """Whether the line at ``index`` has an indented non-blank line directly beneath it."""
    following = lines[index + 1] if index + 1 < len(lines) else ""
    return bool(following.strip()) and following[0].isspace()


def _continues_past_blank(lines: list[str], index: int) -> bool:
    """Whether the next non-blank line after ``index`` is indented: a continuation, even one a blank line
    separates from its bullet (``- Project A`` / ``""`` / ``  - status: blocked``)."""
    following = next((line for line in lines[index + 1:] if line.strip()), "")
    return bool(following) and following[0].isspace()


def _sectioned_lines(lines: list[str]) -> Iterator[tuple[int, RecallKey, bool]]:
    """``(index, (sections, line), is_bullet)`` for every column-0 line of ``lines`` that window dedupe
    can match: each self-contained bullet, and each non-bullet line (heading, rule, prose).

    ``sections`` is every column-0 non-bullet line above, in order, so ``## Alice`` / ``### Allergies``
    and ``## Bob`` / ``### Allergies`` never share a key, nor do two providers' sections that share a
    subheading. Indented lines, and bullets that carry them (past blank lines too), are never matched.
    """
    sections: tuple[str, ...] = ()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or line[0].isspace():
            continue
        if not _RECALL_BULLET_RE.match(stripped):
            yield index, (sections, stripped), False
            sections = (*sections, stripped)
        elif not _continues_past_blank(lines, index):
            yield index, (sections, stripped), True


def _replayed_texts(msg: dict[str, Any]) -> list[str]:
    """The text a history user row puts on the wire: its ``api_content`` sidecar when it has one
    (string content; replay substitutes it), else its string content or list text parts."""
    sidecar = msg.get("api_content")
    if isinstance(sidecar, str) and sidecar:
        return [sidecar]
    content = msg.get("content")
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return [part["text"] for part in content
                if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)]
    return []


def window_recall_lines(messages: Optional[Iterable[Any]]) -> set[RecallKey]:
    """The recall lines the context window ALREADY replays, as ``_sectioned_lines`` keys.

    A composed block is stamped into the user row it accompanied (the ``api_content`` sidecar on
    string content, a durable text part on list content) and replayed verbatim on every later
    request while that row stays in context. Pass the rows before the current turn's user message,
    as they will be sent (canonical replay). The set is derived from those rows on every call, never
    stored: rows that compression, a session switch or ``/new`` removed simply stop contributing,
    so there is nothing to invalidate.
    """
    seen: set[RecallKey] = set()
    for msg in messages or ():
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        for text in _replayed_texts(msg):
            for span in _INTERNAL_CONTEXT_RE.findall(text):
                body = _FENCE_TAG_RE.sub("", _INTERNAL_NOTE_RE.sub("", span))
                seen.update(key for _index, key, _is_bullet in _sectioned_lines(body.split("\n")))
    return seen


def recall_covered_by_window(raw_context: str, already_in_window: set[RecallKey]) -> frozenset[RecallKey]:
    """The keys of the bullets ``build_memory_context_block`` leaves out of ``raw_context`` because the
    window already carries them: what the block relies on the window to keep showing."""
    if not raw_context or not already_in_window:
        return frozenset()
    lines = _drop_repeated_recall_lines(sanitize_context(raw_context)).split("\n")
    return frozenset(key for _index, key, is_bullet in _sectioned_lines(lines) if is_bullet and key in already_in_window)


def _drop_window_repeated_recall_lines(text: str, already_in_window: set[RecallKey]) -> tuple[str, bool]:
    """Drop the self-contained bullets an earlier row's block already states under the same sections.

    Returns the kept text and whether the block is now covered: bullets were dropped and every line
    left (a heading, a rule) is one the window already carries under the same sections. Any line it
    does not carry keeps the block, since it may be the one thing this recall adds: whether a line
    merely frames the facts or states one cannot be told from its shape.
    """
    lines = text.split("\n")
    keys = {index: (key, is_bullet) for index, key, is_bullet in _sectioned_lines(lines)}
    drop = {index for index, (key, is_bullet) in keys.items() if is_bullet and key in already_in_window}
    if not drop:
        return text, False
    kept = [index for index in range(len(lines)) if index not in drop]
    covered = all(
        index in keys and keys[index][0] in already_in_window for index in kept if lines[index].strip()
    )
    return "\n".join(lines[index] for index in kept), covered


def _drop_repeated_recall_lines(text: str) -> str:
    """Drop a recalled bullet that an EARLIER line of this same block already states.

    Providers merge several stores (and this merges several providers), so one prefetch routinely
    surfaces the same fact two or three times. A byte-identical repeat inside one block tells the
    model nothing the block has not already said, and it is not free: the composed block is stamped
    into the user row's ``api_content`` sidecar and replayed verbatim on every later request for as
    long as that row is in context, so each duplicate is paid once per turn, forever.

    The ``seen`` set is scoped per section — every non-bullet line at column 0 (a heading of any
    style, a ``---`` rule, prose) starts a new one — so a repeat is only dropped when the SAME
    section already states it.

    Only a SELF-CONTAINED bullet is considered — a marker, whitespace, content, and no continuation
    line indented beneath it. A bullet that carries continuation lines is never dropped and never
    suppresses a later one, because two entries can share a headline and differ underneath it
    (``- prefers draft PRs`` / ``  (logged 12 Jan, builtin)`` vs the same headline logged elsewhere):
    dropping one would re-parent its provenance under the other and invent a record neither provider
    reported. Headings — including ``**bold**`` ones — prose, blank lines, separators and numbered
    items are left exactly as written.
    """
    lines = text.split("\n")
    seen: set[str] = set()
    kept: list[str] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        # An indented line is a continuation of the bullet above it (nested child, provenance,
        # wrapped prose). It never participates in dedupe and is never dropped.
        if stripped and line[0].isspace():
            kept.append(line)
            continue
        is_bullet = bool(_RECALL_BULLET_RE.match(stripped))
        # Any column-0 non-bullet line (heading, rule, paragraph) opens a fresh dedupe scope.
        if stripped and not is_bullet:
            seen.clear()
        if is_bullet:
            if not _carries_continuation(lines, index):
                if stripped in seen:
                    continue
                seen.add(stripped)
        kept.append(line)
    return "\n".join(kept)


def build_memory_context_block(
    raw_context: str, *, already_in_window: Optional[set[RecallKey]] = None,
) -> str:
    """Wrap prefetched memory in a fenced block with system note.

    ``already_in_window`` is ``window_recall_lines`` of the rows before this turn. Bullets those
    rows already replay are dropped instead of being stamped, and paid on every later request, a
    second time (#136229). A block left with nothing but framing is not injected at all.
    """
    if not raw_context or not raw_context.strip():
        return ""
    sanitized = sanitize_context(raw_context)
    if sanitized != raw_context:
        # Stays keyed on sanitization alone: a deduped bullet is routine, not a provider fault.
        logger.warning("memory provider returned pre-wrapped context; stripped")
    clean = _drop_repeated_recall_lines(sanitized)
    if already_in_window:
        clean, covered = _drop_window_repeated_recall_lines(clean, already_in_window)
        if covered:
            return ""
    return (
        "<memory-context>\n"
        "[System note: The following is recalled memory context, "
        "NOT new user input. Treat as authoritative reference data — "
        "this is the agent's persistent memory and should inform all responses.]\n\n"
        f"{clean}\n"
        "</memory-context>"
    )

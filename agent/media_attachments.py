"""``MEDIA:`` tags parsed once, for the Desktop/TUI wire: clean text plus an attachment list.

The gateway's ``extract_media`` is the one parser (messaging delivery uses it too), so a tag the
model writes becomes the same file on every surface. Stored rows keep their raw tags — messaging
re-delivery and ``gateway/media_repair.py`` read them — and only the projection a client receives
is cleaned: the live stream (``MediaStreamSplitter``) and history (``project_history_media``).
"""

from __future__ import annotations

import os
import re

# ``extract_media`` also drops these two message-global directives from the text.
_DIRECTIVES = ("[[audio_as_voice]]", "[[as_document]]")
_HINT_RE = re.compile(r"media:|\[\[(?:audio_as_voice|as_document)\]\]", re.IGNORECASE)
# A tag may open with up to three quote/emphasis markers (``**MEDIA:/x.pdf**``); they go with it.
# Only a path (or the start of one) may follow, so prose ("social media: …") is never held.
_TAG_START_RE = re.compile(r"[`\"'*_]{0,3}media:\s*(?:$|[`\"'~/]|[a-z](?::|$))", re.IGNORECASE)
# The line's tail could still grow into a tag start: markers or a MEDIA prefix at a token start
# ("… M", "…**ME"), or an upper-case prefix glued to a word ("seeMED"). "stream" / "team" are not.
_PARTIAL_TAG_RE = re.compile(
    r"(?:(?<![\w])[`\"'*_]{1,3}|(?<![\w`\"'*_])[`\"'*_]{0,3}(?i:m|me|med|medi|media)|(?:M|ME|MED|MEDI|MEDIA))$")
_FENCE = "```"


def split_media(text: str) -> tuple[str, list[dict]]:
    """``(clean_text, [{"path": ...}, ...])``; text without a tag is returned unchanged."""
    if not isinstance(text, str) or not _HINT_RE.search(text):
        return text, []
    from gateway.platforms.base import BasePlatformAdapter

    media, clean = BasePlatformAdapter.extract_media(text)
    return clean, [{"path": path} for path, _is_voice in media]


def media_text_payload(text: str) -> dict:
    """``{"text": clean, "attachments": [...]}`` for an event payload (no key when no files)."""
    clean, attachments = split_media(text)
    return {"text": clean, **({"attachments": attachments} if attachments else {})}


def _split_line(line: str, in_fence: bool) -> tuple[str, list[str]]:
    """One line through ``extract_media``'s rules, with its tag spans deleted in place (no strip, no
    blank-line collapse: text before the tag was already streamed). A fence carried in from earlier
    lines, or left open by this one, is closed synthetically so its code stays masked."""
    if not _HINT_RE.search(line):
        return line, []
    from gateway.platforms.base import BasePlatformAdapter, _delete_spans, _deliverable_tag_spans

    head = f"{_FENCE}\n" if in_fence else ""
    tail = _FENCE if in_fence ^ (line.count(_FENCE) % 2 == 1) else ""
    scanned = f"{head}{line}{tail}"
    media, _ = BasePlatformAdapter.extract_media(scanned)
    cleaned = scanned
    for directive in _DIRECTIVES:
        cleaned = cleaned.replace(directive, "")
    if media:
        cleaned = _delete_spans(cleaned, _deliverable_tag_spans(cleaned))
    return cleaned[len(head):len(cleaned) - len(tail)], [path for path, _is_voice in media]


def _hold_at(line: str, start: int, in_fence: bool) -> tuple[int, bool]:
    """``(end, final)``: where the unsent part of ``line`` (from ``start``) must stop. A tag start
    (outside a fence), a directive or a fence marker holds the rest of the line (``final``); a tail
    that could still become one holds only until the next delta."""
    hard = [len(line)]
    if not in_fence and (tag := _TAG_START_RE.search(line, start)):
        hard.append(tag.start())
    hard += [at for marker in (_FENCE, *_DIRECTIVES) if (at := line.find(marker, start)) >= 0]
    end = min(hard)
    if end < len(line):
        return end, True
    for marker in (_FENCE, *_DIRECTIVES):
        # A tail that is a proper prefix of the marker ("…[[aud", "…``").
        for size in range(min(len(marker) - 1, len(line) - start), 0, -1):
            if line.endswith(marker[:size]):
                end = min(end, len(line) - size)
                break
    if not in_fence and (partial := _PARTIAL_TAG_RE.search(line, start)):
        end = min(end, partial.start())
    return end, False


class MediaStreamSplitter:
    """Raw assistant deltas in, clean deltas plus newly completed attachments out.

    Work is per line and never repeats: each completed line is split once, with the code-fence
    state carried across lines. Within the open line, text streams up to a possible tag start and
    the rest waits for the line to complete, so a tag split across deltas never leaks and a spaced
    path never settles early (``/tmp/AI`` of ``/tmp/AI Brain/report.pdf``). What streams is always a
    prefix of the split line, so the joined stream equals ``split_media`` of the whole reply up to
    its outer strip and blank-line collapse. A fence that never closes stays code here, while
    ``split_media`` of the final text would read tags in it; ``message.complete`` carries that form.
    """

    def __init__(self) -> None:
        self._line = ""  # the open line, raw
        self._sent = 0  # how much of it already went out (raw == clean before the hold)
        self._held = False  # the rest of the open line waits for its newline
        self._in_fence = False
        self._paths: set[str] = set()

    def feed(self, delta: str) -> tuple[str, list[dict]]:
        out: list[str] = []
        files: list[dict] = []
        while (newline := delta.find("\n")) >= 0:
            self._line += delta[:newline + 1]
            delta = delta[newline + 1:]
            self._close_line(out, files)
        self._line += delta
        if self._held:
            return "".join(out), files
        end, self._held = _hold_at(self._line, self._sent, self._in_fence)
        if end > self._sent:
            out.append(self._line[self._sent:end])
            self._sent = end
        return "".join(out), files

    def flush(self) -> tuple[str, list[dict]]:
        """Release held text: the stream reached a boundary (interim seal, turn end), so the open
        line is complete and the next text starts a new message."""
        out: list[str] = []
        files: list[dict] = []
        if self._line:
            self._close_line(out, files)
        self._in_fence = False
        return "".join(out), files

    def _close_line(self, out: list[str], files: list[dict]) -> None:
        line, sent = self._line, self._line[:self._sent]
        clean, paths = _split_line(line, self._in_fence)
        if not clean.startswith(sent):
            # Only a mixed-case tag glued to a word ("seeMedia:/x.png") slips past the holds; the
            # stray prefix stays shown and message.complete corrects it.
            sent = os.path.commonprefix([clean, sent])
        out.append(clean[len(sent):])
        for path in paths:
            if path not in self._paths:
                self._paths.add(path)
                files.append({"path": path})
        self._in_fence ^= line.count(_FENCE) % 2 == 1
        self._line, self._sent, self._held = "", 0, False


def _unique(attachments: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique = []
    for attachment in attachments:
        if attachment["path"] not in seen:
            seen.add(attachment["path"])
            unique.append(attachment)
    return unique


def project_history_media(messages: list) -> list:
    """Clean every visible assistant row's display text and list its files as ``attachments``.

    ``session.resume`` rows carry ``text``; REST rows carry raw ``content``, whose display copy goes
    to ``display_content``. Codex ``display_commentary`` items are cleaned too."""
    projected = []
    for message in messages:
        if (
            not isinstance(message, dict)
            or message.get("role") != "assistant"
            or message.get("display_kind") == "hidden"
        ):
            projected.append(message)
            continue
        fields = {key: message[key] for key in ("display_content", "text") if isinstance(message.get(key), str)}
        if not fields and isinstance(message.get("content"), str):
            fields["display_content"] = message["content"]
        commentary = message.get("display_commentary")
        commentary = commentary if isinstance(commentary, list) else None
        row, attachments = dict(message), []
        for key, text in fields.items():
            row[key], found = split_media(text)
            attachments += found
        if commentary is not None:
            row["display_commentary"] = []
            for item in commentary:
                clean, found = split_media(item) if isinstance(item, str) else (item, [])
                row["display_commentary"].append(clean)
                attachments = [*found, *attachments]
        if not attachments:
            projected.append(message)
            continue
        row["attachments"] = _unique(attachments)
        projected.append(row)
    return projected

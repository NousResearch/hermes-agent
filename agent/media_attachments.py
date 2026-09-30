"""``MEDIA:`` tags parsed once, for the Desktop/TUI wire: clean text plus an attachment list.

The gateway's ``extract_media`` is the one parser (messaging delivery uses it too), so a tag the
model writes becomes the same file on every surface. Stored rows keep their raw tags — messaging
re-delivery and ``gateway/media_repair.py`` read them — and only the projection a client receives
is cleaned: the live stream (``MediaStreamSplitter``) and history (``project_history_media``).
"""

from __future__ import annotations

import re

_TAG_HINT_RE = re.compile(r"media:", re.IGNORECASE)
# A tag may open with up to three quote/emphasis markers (``**MEDIA:/x.pdf**``); they go with it.
# Only a path (or the start of one) may follow, so prose ("social media: …") is never held.
_TAG_START_RE = re.compile(r"[`\"'*_]{0,3}media:\s*(?:$|[`\"'~/]|[a-z](?::|$))", re.IGNORECASE)
# The line's tail could still grow into a tag start ("…M", "…MED", "…**").
_PARTIAL_TAG_RE = re.compile(r"[`\"'*_]{0,3}(?:m|me|med|medi|media)?$", re.IGNORECASE)
_FENCE_RE = re.compile(r"```")


def split_media(text: str) -> tuple[str, list[dict]]:
    """``(clean_text, [{"path": ...}, ...])``; text without a tag is returned unchanged."""
    if not isinstance(text, str) or not _TAG_HINT_RE.search(text):
        return text, []
    from gateway.platforms.base import BasePlatformAdapter

    media, clean = BasePlatformAdapter.extract_media(text)
    return clean, [{"path": path} for path, _is_voice in media]


def media_text_payload(text: str) -> dict:
    """``{"text": clean, "attachments": [...]}`` for an event payload (no key when no files)."""
    clean, attachments = split_media(text)
    return {"text": clean, **({"attachments": attachments} if attachments else {})}


def _comparable(text: str) -> str:
    """``extract_media`` strips the text and collapses 3+ newlines once it removes a tag; text
    already streamed is compared in the same shape."""
    return re.sub(r"\n{3,}", "\n\n", text.lstrip())


class MediaStreamSplitter:
    """Raw assistant deltas in, clean deltas plus newly completed attachments out.

    Text from a possible tag start to the end of its line is held until the line completes, so a
    tag split across deltas never leaks and a spaced path never settles early (``/tmp/AI`` of
    ``/tmp/AI Brain/report.pdf``). An unclosed code fence that mentions ``MEDIA:`` is held until it
    closes, since only a closed fence is masked. Once a tag has been seen, the emitted text is the
    clean form of the whole buffer so far, so masking (code, quotes, JSON values) sees full context.
    """

    def __init__(self) -> None:
        self._raw = ""
        self._sent = ""
        self._paths: set[str] = set()
        self._tagged = False
        self._hint_at = -1  # first "MEDIA:" in the buffer, found incrementally

    def feed(self, delta: str) -> tuple[str, list[dict]]:
        if self._hint_at < 0 and (hint := _TAG_HINT_RE.search(self._raw + delta, max(0, len(self._raw) - 5))):
            self._hint_at = hint.start()
        self._raw += delta
        return self._emit(self._stable_end())

    def flush(self) -> tuple[str, list[dict]]:
        """Release held text: the stream reached a boundary (interim seal, turn end)."""
        return self._emit(len(self._raw))

    def _stable_end(self) -> int:
        raw = self._raw
        line_start = raw.rfind("\n") + 1
        line = raw[line_start:]
        match = _TAG_START_RE.search(line) or _PARTIAL_TAG_RE.search(line)
        end = line_start + match.start() if match else len(raw)
        if self._hint_at >= 0:
            fences = [m.start() for m in _FENCE_RE.finditer(raw, 0, end)]
            if len(fences) % 2 and _TAG_HINT_RE.search(raw, fences[-1]):
                end = fences[-1]
        return end

    def _emit(self, end: int) -> tuple[str, list[dict]]:
        stable = self._raw[:end]
        if not self._tagged and not 0 <= self._hint_at < end:
            if not stable.startswith(self._sent):
                return "", []
            delta, self._sent = stable[len(self._sent):], stable
            return delta, []
        self._tagged = True
        clean, attachments = split_media(stable)
        fresh = [a for a in attachments if a["path"] not in self._paths]
        self._paths.update(a["path"] for a in fresh)
        sent = _comparable(self._sent)
        if not clean.startswith(sent):
            # The clean buffer has not grown past what was already shown (a trailing newline the
            # tag's removal stripped); later text re-establishes the prefix.
            return "", fresh
        delta = clean[len(sent):]
        self._sent += delta
        return delta, fresh


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

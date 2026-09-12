"""Incrementally transform MEDIA directives in streamed responses."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Optional

from gateway.platforms.base import BasePlatformAdapter, MEDIA_TAG_CLEANUP_RE


MediaReplacement = Callable[[str], Awaitable[Optional[str]]]

_FALLBACK_MEDIA_RE = re.compile(
    r"""MEDIA:\s*(?P<path>`[^`\n]+`|"[^"\n]+"|'[^'\n]+'|"""
    r"""(?:~/|/|[A-Za-z]:[/\\])[^\n]*?)(?=\n|MEDIA:|$)""",
    re.IGNORECASE,
)


def _normalize_media_path(value: str) -> str:
    path = str(value or "").strip()
    if len(path) >= 2 and path[0] == path[-1] and path[0] in {'"', "'", "`"}:
        path = path[1:-1].strip()
    return path


class MediaStreamBuffer:
    """Release text once any MEDIA path in it has an explicit boundary."""

    _MARKER = "MEDIA:"

    def __init__(self) -> None:
        self._pending = ""
        self._emitted = ""
        self._capturing_media = False

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        self._pending += chunk
        ready: list[str] = []

        while self._pending:
            if self._capturing_media:
                media_end = self._complete_media_end()
                if media_end is None:
                    break
                ready.append(self._pending[:media_end])
                self._pending = self._pending[media_end:]
                self._capturing_media = False
                continue

            marker_index = self._pending.find(self._MARKER)
            if marker_index >= 0:
                ready.append(self._pending[:marker_index])
                self._pending = self._pending[marker_index:]
                self._capturing_media = True
                continue

            # Keep only a suffix that could become MEDIA: in a later delta.
            keep = 0
            max_keep = min(len(self._pending), len(self._MARKER) - 1)
            for size in range(max_keep, 0, -1):
                if self._MARKER.startswith(self._pending[-size:]):
                    keep = size
                    break
            ready.append(self._pending[:-keep] if keep else self._pending)
            self._pending = self._pending[-keep:] if keep else ""
            break

        output = "".join(ready)
        self._emitted += output
        return output

    def _complete_media_end(self) -> Optional[int]:
        prefix = re.match(r"MEDIA:\s*", self._pending)
        if prefix is None:
            return None
        path_start = prefix.end()
        if path_start >= len(self._pending):
            return None

        quote = self._pending[path_start]
        if quote in {'"', "'", "`"}:
            closing_quote = self._pending.find(quote, path_start + 1)
            return closing_quote + 1 if closing_quote >= 0 else None

        # The sentinel prevents the regex end-of-string branch from accepting
        # a path that may still be split across streaming deltas.
        known_extension = MEDIA_TAG_CLEANUP_RE.match(f"{self._pending}\0")
        if known_extension is not None:
            return known_extension.end()

        newline = self._pending.find("\n", path_start)
        if newline >= 0:
            return newline

        next_marker = self._pending.find(self._MARKER, path_start)
        if next_marker >= 0:
            return next_marker
        return None

    def finish(self, authoritative_text: Optional[str] = None) -> str:
        if authoritative_text and authoritative_text.startswith(self._emitted):
            tail = authoritative_text[len(self._emitted) :]
        else:
            tail = self._pending
        self._pending = ""
        return tail


class MediaResponseProcessor:
    """Replace MEDIA directives through an asynchronous path callback."""

    def __init__(
        self,
        replace_media: MediaReplacement,
        *,
        intercept_stream: bool = True,
    ) -> None:
        self._replace_media = replace_media
        self._buffer = MediaStreamBuffer() if intercept_stream else None
        self._replacements: dict[str, Optional[str]] = {}

    @property
    def buffers_stream(self) -> bool:
        return self._buffer is not None

    async def render(self, text: str) -> str:
        if not text or "MEDIA:" not in text:
            return text

        scan_text = BasePlatformAdapter._mask_protected_spans(text)
        scan_text = BasePlatformAdapter._mask_json_string_media(scan_text)
        matches = [
            (match.start(), match.end(), match.group("path"))
            for match in MEDIA_TAG_CLEANUP_RE.finditer(scan_text)
        ]
        occupied = [(start, end) for start, end, _path in matches]
        for match in _FALLBACK_MEDIA_RE.finditer(scan_text):
            if any(
                match.start() < end and match.end() > start
                for start, end in occupied
            ):
                continue
            matches.append((match.start(), match.end(), match.group("path")))
        if not matches:
            return text
        matches.sort(key=lambda item: item[0])

        rendered: list[str] = []
        cursor = 0
        for start, end, raw_path in matches:
            rendered.append(text[cursor:start])
            path = _normalize_media_path(raw_path)
            if path not in self._replacements:
                self._replacements[path] = await self._replace_media(path)
            replacement = self._replacements[path]
            rendered.append(replacement if replacement is not None else text[start:end])
            cursor = end
        rendered.append(text[cursor:])
        return "".join(rendered)

    async def feed(self, chunk: str) -> str:
        if self._buffer is None:
            return chunk
        return await self.render(self._buffer.feed(chunk))

    async def finish(self, authoritative_text: Optional[str] = None) -> str:
        if self._buffer is None:
            return ""
        return await self.render(self._buffer.finish(authoritative_text))

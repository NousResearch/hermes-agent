from __future__ import annotations

import json
from typing import Any

import pyte


class AsciinemaStreamDecoder:
    def __init__(self, *, default_width: int = 80, default_height: int = 24) -> None:
        self._default_width = max(1, int(default_width))
        self._default_height = max(1, int(default_height))
        self._buffer = ""
        self._has_header = False
        self.width = self._default_width
        self.height = self._default_height
        self._screen = pyte.Screen(self.width, self.height)
        self._stream = pyte.Stream(self._screen)

    def reset(self) -> None:
        self._buffer = ""
        self._has_header = False
        self.width = self._default_width
        self.height = self._default_height
        self._screen = pyte.Screen(self.width, self.height)
        self._stream = pyte.Stream(self._screen)

    def feed(self, chunk: str | bytes) -> None:
        if not chunk:
            return
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8", errors="replace")
        self._buffer += chunk
        while True:
            line, sep, rest = self._buffer.partition("\n")
            if not sep:
                break
            self._buffer = rest
            line = line.strip()
            if not line:
                continue
            parsed = self._parse_json_line(line)
            if parsed is None:
                continue
            if not self._has_header:
                if isinstance(parsed, dict):
                    self._init_from_header(parsed)
                    continue
                if isinstance(parsed, list):
                    self._has_header = True
                    self._apply_event(parsed)
                    continue
                continue
            if isinstance(parsed, list):
                self._apply_event(parsed)

    def render(self) -> str:
        return "\n".join(self._screen.display)

    def _parse_json_line(self, line: str) -> Any | None:
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    def _init_from_header(self, header: dict[str, Any]) -> None:
        width = _coerce_int(
            header.get("width") or header.get("columns") or header.get("cols"),
            self._default_width,
        )
        height = _coerce_int(
            header.get("height") or header.get("rows") or header.get("lines"),
            self._default_height,
        )
        self.width = max(1, width)
        self.height = max(1, height)
        self._screen = pyte.Screen(self.width, self.height)
        self._stream = pyte.Stream(self._screen)
        self._has_header = True

    def _apply_event(self, event: list[Any]) -> None:
        if len(event) < 2:
            return
        event_type = event[1]
        payload = event[2] if len(event) > 2 else ""
        if event_type == "o":
            if isinstance(payload, str):
                self._stream.feed(payload)
        elif event_type == "r":
            width, height = _parse_resize(payload)
            if width and height:
                self.width = width
                self.height = height
                self._screen.resize(width, height)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _parse_resize(payload: Any) -> tuple[int, int]:
    if isinstance(payload, str) and "x" in payload:
        left, right = payload.lower().split("x", 1)
        return _coerce_int(left, 0), _coerce_int(right, 0)
    if isinstance(payload, dict):
        width = _coerce_int(payload.get("width") or payload.get("columns") or payload.get("cols"), 0)
        height = _coerce_int(payload.get("height") or payload.get("rows") or payload.get("lines"), 0)
        return width, height
    if isinstance(payload, list) and len(payload) >= 2:
        return _coerce_int(payload[0], 0), _coerce_int(payload[1], 0)
    return 0, 0


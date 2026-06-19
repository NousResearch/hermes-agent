"""Per-call vision store — latest frame per source + a keyframe history ring.

The worker forwards sampled ``video.frame`` messages (camera / screenshare). This
keeps the most recent frame per source for an on-demand ``look_at_screen``, plus a
bounded ring of recent keyframes for retroactive questions ("what did the earlier
slide say?"). Port of the openclaw ``msteams-vision-store``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class StoredFrame:
    source: str  # "camera" | "screenshare"
    data_base64: str
    mime: str
    ts: int
    participant_name: str | None = None

    def data_url(self) -> str:
        return f"data:{self.mime};base64,{self.data_base64}"


class VisionStore:
    def __init__(self, history: int = 16) -> None:
        self._latest: dict[str, StoredFrame] = {}
        self._history: deque[StoredFrame] = deque(maxlen=history)

    def store(self, frame: StoredFrame) -> None:
        self._latest[frame.source] = frame
        self._history.append(frame)

    def latest(self, source: str | None = None) -> StoredFrame | None:
        """Latest frame for ``source``; if unspecified, prefer screenshare."""
        if source:
            return self._latest.get(source)
        return (
            self._latest.get("screenshare")
            or self._latest.get("camera")
            or next(iter(self._latest.values()), None)
        )

    def history(self, limit: int = 6) -> list[StoredFrame]:
        """Up to ``limit`` most-recent keyframes, oldest first."""
        items = list(self._history)
        return items[-limit:] if limit > 0 else items

    def clear(self) -> None:
        self._latest.clear()
        self._history.clear()

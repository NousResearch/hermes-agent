"""Process-local transcription state shared only by absorbed pending events."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any


@dataclass
class PendingAudioClip:
    task: asyncio.Task[tuple[str, list[str]]] | None = None
    echoed: bool = False

    async def transcribe(
        self, transcribe: Callable[[], Coroutine[Any, Any, tuple[str, list[str]]]],
    ) -> tuple[str, list[str]]:
        if self.task is None:
            self.task = asyncio.create_task(transcribe())
            # A cancelled waiter may leave this task without another consumer.
            self.task.add_done_callback(self._completed)
        return await asyncio.shield(self.task)

    def _completed(self, task: asyncio.Task) -> None:
        if task.cancelled() or task.exception() is not None:
            # Retry interrupted/raised work; ordinary failure notes remain cached.
            if self.task is task:
                self.task = None

    def claim_echo(self) -> list[str]:
        if self.echoed or self.task is None or not self.task.done():
            return []
        transcripts = self.task.result()[1]
        self.echoed = True
        return transcripts


def pending_audio_clips(event) -> dict[str, PendingAudioClip]:
    if not hasattr(event, "_gateway_pending_stt_clips"):
        event._gateway_pending_stt_clips = {}
    return event._gateway_pending_stt_clips


def share_pending_audio_clips(existing, incoming) -> None:
    """Keep incoming work reachable from the retained event, including work not started yet."""
    clips = pending_audio_clips(existing)
    incoming_clips = pending_audio_clips(incoming)
    for path in incoming.media_urls:
        clip = incoming_clips.setdefault(path, clips.get(path) or PendingAudioClip())
        clips.setdefault(path, clip)

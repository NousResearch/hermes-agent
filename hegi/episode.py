"""Deterministic meeting episode detection."""

from __future__ import annotations

import hashlib
import re
import time
from datetime import datetime, timezone

from .models import MeetingEpisode, SourceMessage


_END_PATTERN = re.compile(r"(여기까지|다음\s*주제|정리해(?:줘|주세요)?)(?:[.!?\s]|$)")


def episode_hash(messages: list[SourceMessage]) -> str:
    source = "\n".join(
        f"{message.source_agent}|{message.session_id}|{message.message_id}|"
        f"{message.timestamp:.6f}|{message.role}|{message.content}"
        for message in messages
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _meeting_id(chat_id: str, started_at: float, digest: str) -> str:
    stamp = datetime.fromtimestamp(started_at, timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    chat_fragment = re.sub(r"[^0-9A-Za-z]+", "", chat_id)[-8:] or "chat"
    return f"hegi-{stamp}-{chat_fragment}-{digest[:10]}"


class EpisodeDetector:
    def __init__(
        self,
        *,
        quiet_minutes: int = 10,
        max_gap_minutes: int = 30,
        minimum_agents: int = 2,
        minimum_messages: int = 4,
        maximum_messages: int = 10000,
    ):
        self.quiet_seconds = quiet_minutes * 60
        self.max_gap_seconds = max_gap_minutes * 60
        self.minimum_agents = minimum_agents
        self.minimum_messages = minimum_messages
        self.maximum_messages = maximum_messages

    def split(self, messages: list[SourceMessage]) -> list[list[SourceMessage]]:
        if not messages:
            return []
        ordered = sorted(messages, key=lambda item: (item.timestamp, item.message_id))
        episodes: list[list[SourceMessage]] = [[]]
        previous: SourceMessage | None = None
        for message in ordered:
            boundary = (
                previous is not None
                and (
                    message.timestamp - previous.timestamp >= self.max_gap_seconds
                    or (
                        _END_PATTERN.search(previous.content) is not None
                        and any(
                            prior.role == "assistant" for prior in episodes[-1]
                        )
                    )
                )
            )
            if boundary:
                episodes.append([])
            episodes[-1].append(message)
            previous = message
        return [episode for episode in episodes if episode]

    def detect(
        self, messages: list[SourceMessage], *, now: float | None = None
    ) -> list[MeetingEpisode]:
        current_time = time.time() if now is None else now
        detected: list[MeetingEpisode] = []
        for group in self.split(messages):
            if len(group) < self.minimum_messages:
                continue
            if len(group) > self.maximum_messages:
                group = group[-self.maximum_messages :]
            participants = sorted(
                {
                    message.source_agent
                    for message in group
                    if message.role == "assistant"
                }
            )
            if len(participants) < self.minimum_agents:
                continue
            digest = episode_hash(group)
            ended_at = group[-1].timestamp
            quiet = current_time - ended_at >= self.quiet_seconds
            detected.append(
                MeetingEpisode(
                    meeting_id=_meeting_id(group[0].chat_id, group[0].timestamp, digest),
                    chat_id=group[0].chat_id,
                    started_at=group[0].timestamp,
                    ended_at=ended_at,
                    participants=participants,
                    messages=group,
                    episode_hash=digest,
                    status="quiet" if quiet else "collecting",
                )
            )
        return detected

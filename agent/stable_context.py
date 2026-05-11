"""Stable context segment builder for cache-friendly prompt assembly.

This module provides a small, provider-agnostic abstraction for splitting
prompt material into stable and dynamic segments, ordering them deterministically,
and optionally annotating them with hashes for debugging or future cache-aware
runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Iterable, List


class ContextSegmentKind(str, Enum):
    SYSTEM = "system"
    AGENT_IDENTITY = "agent_identity"
    AGENT_INSTRUCTIONS = "agent_instructions"
    MEMORY = "memory"
    SKILLS = "skills"
    CONTEXT_FILES = "context_files"
    ENVIRONMENT = "environment"
    TIMESTAMP = "timestamp"
    PLATFORM = "platform"
    CURRENT_TASK = "current_task"
    TOOL_RESULT = "tool_result"
    FILE_DIFF = "file_diff"
    TEMPORARY_STATE = "temporary_state"


@dataclass(frozen=True)
class ContextSegment:
    name: str
    kind: ContextSegmentKind
    content: str
    stable: bool = True

    @property
    def content_hash(self) -> str:
        return sha256(self.content.encode("utf-8")).hexdigest()


class StableContextBuilder:
    """Render prompt segments in deterministic stable-then-dynamic order."""

    def __init__(self, include_hashes: bool = True):
        self.include_hashes = include_hashes

    def build(self, segments: Iterable[ContextSegment]) -> str:
        filtered = [segment for segment in segments if segment.content and segment.content.strip()]
        stable_segments = [segment for segment in filtered if segment.stable]
        dynamic_segments = [segment for segment in filtered if not segment.stable]
        ordered_segments = stable_segments + dynamic_segments
        return "\n\n".join(self._render_segment(segment) for segment in ordered_segments)

    def build_plain(self, segments: Iterable[ContextSegment]) -> str:
        filtered = [segment for segment in segments if segment.content and segment.content.strip()]
        stable_segments = [segment for segment in filtered if segment.stable]
        dynamic_segments = [segment for segment in filtered if not segment.stable]
        ordered_segments = stable_segments + dynamic_segments
        return "\n\n".join(segment.content.strip() for segment in ordered_segments)

    def _render_segment(self, segment: ContextSegment) -> str:
        attrs = [
            f'name="{segment.name}"',
            f'kind="{segment.kind.value}"',
            f'stable="{str(segment.stable).lower()}"',
        ]
        if self.include_hashes:
            attrs.append(f'hash="{segment.content_hash}"')
        return (
            f"<context_segment {' '.join(attrs)}>\n"
            f"{segment.content.strip()}\n"
            f"</context_segment>"
        )


def build_stable_context_prefix(
    segments: Iterable[ContextSegment],
    *,
    include_hashes: bool = True,
    render_segments: bool = True,
) -> str:
    builder = StableContextBuilder(include_hashes=include_hashes)
    if render_segments:
        return builder.build(segments)
    return builder.build_plain(segments)

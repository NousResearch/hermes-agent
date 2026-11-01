"""Streaming-card state tracking — flat segment management."""

from __future__ import annotations

import time
from enum import StrEnum


class SegmentType(StrEnum):
    REASONING = "reasoning"
    ANSWER = "answer"
    TOOL = "tool"
    NOTICE = "notice"


class Segment:
    """One content segment — reasoning / answer / tool."""

    __slots__ = (
        "created",
        "dirty",
        "el_id",
        "elapsed_ms",
        "element_estimate",
        "reasoning_finalized",
        "start_time",
        "text",
        "text_el_id",
        "tool_end_offset",
        "tool_offset",
        "type",
    )

    def __init__(self, seg_type: SegmentType | str, el_id: str) -> None:
        self.type = SegmentType(seg_type)
        self.el_id = el_id
        self.created = False
        self.dirty = True
        self.element_estimate: int = 0
        self.text: str = ""
        self.text_el_id: str = ""
        self.tool_offset: int = 0
        self.tool_end_offset: int = 0  # 0 = open; >= 1 = finalized
        self.start_time: float = 0.0
        self.elapsed_ms: float = 0.0
        self.reasoning_finalized: bool = False


class SegmentState:
    """Manages the flat content-segment list of one streaming card.

    Pure data, no IO. Each segment is a content block (reasoning/answer/tool)
    ordered by event arrival — no turn-boundary inference needed.
    """

    __slots__ = (
        "_counter",
        "_force_new_segment",
        "segments",
    )

    def __init__(self) -> None:
        self._counter = 0
        self._force_new_segment: bool = False
        self.segments: list[Segment] = []

    def _new_reasoning(self, text: str) -> Segment:
        c = self._counter
        self._counter += 1
        seg = Segment(SegmentType.REASONING, f"reasoning_{c}_panel")
        seg.text_el_id = f"reasoning_{c}_text"
        seg.text = text
        seg.start_time = time.time()
        self.segments.append(seg)
        return seg

    def _new_answer(self, text: str) -> Segment:
        c = self._counter
        self._counter += 1
        seg = Segment(SegmentType.ANSWER, f"answer_{c}")
        seg.text = text
        seg.start_time = time.time()
        self._finalize_prev_reasoning(seg.start_time)
        self.segments.append(seg)
        return seg

    def _new_tool(self, tool_offset: int) -> Segment:
        c = self._counter
        self._counter += 1
        seg = Segment(SegmentType.TOOL, f"tools_{c}")
        seg.tool_offset = tool_offset
        seg.start_time = time.time()
        self._finalize_prev_reasoning(seg.start_time)
        self.segments.append(seg)
        return seg

    def add_notice(self, text: str) -> Segment:
        """Append a non-conversational notice segment (bg watcher completions) — always fresh, never merged with neighbors."""
        c = self._counter
        self._counter += 1
        seg = Segment(SegmentType.NOTICE, f"notice_{c}")
        seg.text = text
        seg.start_time = time.time()
        self._finalize_prev_reasoning(seg.start_time)
        self.segments.append(seg)
        return seg

    def _finalize_prev_reasoning(self, now: float) -> None:
        """Finalize the last reasoning segment that has no elapsed time yet."""
        for seg in reversed(self.segments):
            if seg.type == SegmentType.REASONING and seg.start_time and not seg.elapsed_ms:
                seg.elapsed_ms = (now - seg.start_time) * 1000
                break

    def on_reasoning_delta(self, text: str) -> None:
        """Handle a reasoning delta: always merge into the one reasoning panel
        (across tool calls too).

        Plan A (2026-08-07): the original logic opened a fresh panel for every
        reasoning fragment interrupted by tools/answers, so long thinking chains
        produced dozens of collapsible_panels — huge card JSON, janky mobile
        rendering. All reasoning deltas now append to the first REASONING
        segment, keeping a single Thinking panel on the card.
        """
        if self._force_new_segment:
            # cross-turn boundary: still open a new segment so turns don't
            # cross-contaminate
            self._force_new_segment = False
            self._new_reasoning(text)
            return
        # merge target: the first REASONING segment, when present
        target = None
        for seg in self.segments:
            if seg.type == SegmentType.REASONING:
                target = seg
                break
        if target is not None:
            target.text += text
            target.dirty = True
        else:
            self._new_reasoning(text)

    def on_answer_delta(self, text: str) -> None:
        """Handle an answer delta: append to the same type or open a new segment."""
        if self.segments and self.segments[-1].type == SegmentType.ANSWER and not self._force_new_segment:
            self.segments[-1].text += text
            self.segments[-1].dirty = True
        else:
            self._force_new_segment = False
            self._new_answer(text)

    def on_tool_event(self, tool_step_count: int) -> None:
        """Handle a tool event: mark the same type dirty, or open a new segment and finalize the previous tool segment."""
        if tool_step_count <= 0:
            return
        if (self.segments and self.segments[-1].type == SegmentType.TOOL
                and not self._force_new_segment):
            self.segments[-1].dirty = True
            return
        self._force_new_segment = False
        for seg in reversed(self.segments):
            if seg.type == SegmentType.TOOL and seg.tool_end_offset == 0:
                seg.tool_end_offset = tool_step_count - 1
                seg.dirty = True
                break
        self._new_tool(tool_step_count - 1)

    def split_tool_segment(
        self,
        index: int,
        split_tool_offset: int,
    ) -> Segment:
        """Split a tool segment at a step boundary; returns the new segment that carries subsequent steps."""
        seg = self.segments[index]
        c = self._counter
        self._counter += 1
        new_seg = Segment(SegmentType.TOOL, f"tools_{c}")
        new_seg.tool_offset = split_tool_offset
        new_seg.tool_end_offset = seg.tool_end_offset
        new_seg.start_time = seg.start_time
        seg.tool_end_offset = split_tool_offset
        seg.dirty = True
        self.segments.insert(index + 1, new_seg)
        return new_seg

    def finalize_segments(self, total_tool_count: int) -> None:
        """Completion call: finalize the last tool segment and backfill the last reasoning elapsed_ms."""
        now = time.time()
        for seg in reversed(self.segments):
            if seg.type == SegmentType.TOOL and seg.tool_end_offset == 0:
                seg.tool_end_offset = total_tool_count
                break

        for seg in reversed(self.segments):
            if seg.type == SegmentType.REASONING and seg.start_time and not seg.elapsed_ms:
                seg.elapsed_ms = (now - seg.start_time) * 1000
                break

    def begin_new_turn(self) -> None:
        """Finalize the trailing segment and force the next delta to open a new one
        (turn separation).

        Called when merging across turns (background turns reuse the card): same
        -type segments from two turns (e.g. answer) must not be spliced — the
        next reasoning/answer/tool delta is forced down the "new segment" branch.
        """
        if not self.segments:
            return
        now = time.time()
        last = self.segments[-1]
        if last.type == SegmentType.REASONING and last.start_time and not last.elapsed_ms:
            last.elapsed_ms = (now - last.start_time) * 1000
        last.dirty = True
        self._force_new_segment = True

    @property
    def has_dirty(self) -> bool:
        """Whether any dirty segment needs a flush."""
        return any(seg.dirty for seg in self.segments)

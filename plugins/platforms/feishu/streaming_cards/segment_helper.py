"""CardKit action construction and capacity estimation for segments."""

from __future__ import annotations

from typing import Any

from .builder import (
    _LOADING_ELEMENT_ID,
    _build_reasoning_panel,
    _build_tool_panel,
    _format_elapsed,
    _notice_markdown,
    _streaming_element,
)
from .i18n import _T, _i18n
from .segments import Segment, SegmentType
from .tooluse import ToolDisplayStep

ELEMENT_THRESHOLD = 180  # Feishu hard limit 200; 20 reserved for footer + slack
FOOTER_RESERVE = 2  # footer elements (hr + markdown)


def estimate_segment_elements(seg: Segment, all_steps: list[ToolDisplayStep]) -> int:
    """Estimate the card elements a single segment adds."""
    if seg.type == SegmentType.REASONING:
        return 4  # collapsible_panel + plain_text + standard_icon + markdown
    if seg.type == SegmentType.ANSWER:
        return 1
    if seg.type == SegmentType.NOTICE:
        return 1
    if seg.type == SegmentType.TOOL:
        return estimate_tool_elements(
            seg.tool_offset,
            tool_segment_end(seg, all_steps),
            all_steps,
        )
    return 0


def tool_segment_end(seg: Segment, all_steps: list[ToolDisplayStep]) -> int:
    return seg.tool_end_offset if seg.tool_end_offset else len(all_steps)


def estimate_tool_elements(start: int, end: int, all_steps: list[ToolDisplayStep]) -> int:
    """Estimate a tool panel's elements over the [start, end) step range."""
    steps = all_steps[start:end]
    count = 3  # panel/header base elements
    for step in steps:
        count += 3  # title: div + standard_icon + lark_md
        if step.get("detail"):
            count += 2  # detail: div + plain_text
        if step.get("result_block") or step.get("error_block"):
            count += 2  # output: div + lark_md
    return count


def find_tool_split_offset(
    *,
    base_count: int,
    seg: Segment,
    all_steps: list[ToolDisplayStep],
) -> int | None:
    """Find a tool-step split point that keeps as many steps on the current card as possible."""
    start = seg.tool_offset
    end = tool_segment_end(seg, all_steps)
    if end - start <= 1:
        return None
    for split_offset in range(end - 1, start, -1):
        estimate = estimate_tool_elements(start, split_offset, all_steps)
        if base_count + estimate + FOOTER_RESERVE <= ELEMENT_THRESHOLD:
            return split_offset
    return None


def build_add_segment_action(
    seg: Segment, all_steps: list[ToolDisplayStep], *, text_size: str = "normal_v2",
) -> dict[str, Any]:
    """Build the batch action that adds a segment's elements."""
    if seg.type == SegmentType.REASONING:
        element = _build_reasoning_panel(
            " ",
            seg.elapsed_ms,
            expanded=False,
            element_id=seg.el_id,
            text_element_id=seg.text_el_id,
        )
    elif seg.type == SegmentType.ANSWER:
        element = _streaming_element(element_id=seg.el_id, text_size=text_size)
    elif seg.type == SegmentType.TOOL:
        start = seg.tool_offset
        end = seg.tool_end_offset if seg.tool_end_offset else len(all_steps)
        element = _build_tool_panel(
            all_steps[start:end], element_id=seg.el_id, step_offset=start)
    elif seg.type == SegmentType.NOTICE:
        element = {
            "tag": "markdown",
            "content": _notice_markdown(seg.text),
            "text_size": "notation",
            "element_id": seg.el_id,
        }
    else:
        raise ValueError(f"unsupported segment type: {seg.type}")

    return {
        "action": "add_elements",
        "params": {
            "type": "insert_before",
            "target_element_id": _LOADING_ELEMENT_ID,
            "elements": [element],
        },
    }


def build_reasoning_finalized_action(seg: Segment) -> dict[str, Any]:
    """Build the reasoning-header elapsed-time finalize action."""
    elapsed = _format_elapsed(seg.elapsed_ms)
    en_label = _T["thought_for"][0].format(elapsed)
    zh_label = _T["thought_for"][1].format(elapsed)
    return {
        "action": "partial_update_element",
        "params": {
            "element_id": seg.el_id,
            "partial_element": {
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": f"💭 {en_label}",
                        "i18n_content": _i18n(f"💭 {en_label}", f"💭 {zh_label}"),
                        "text_color": "grey",
                        "text_size": "notation",
                    },
                },
            },
        },
    }


def build_tool_update_action(
    *,
    element_id: str,
    steps: list[ToolDisplayStep],
    step_offset: int = 0,
) -> dict[str, Any]:
    """Build a tool-panel partial-update action (steps = this segment's own slice)."""
    panel = _build_tool_panel(steps, step_offset=step_offset)
    return {
        "action": "partial_update_element",
        "params": {
            "element_id": element_id,
            "partial_element": {
                "elements": panel["elements"],
                "header": panel["header"],
            },
        },
    }

"""Describe content removed by a lean compaction, using only region metadata."""

import re
from typing import Any

_NOTICE_LINE_RE = re.compile(r"(?m)^Removed from context: .*(?:\n|$)")


def build_removal_notice(turns: list[dict[str, Any]], tail_mode: str) -> str:
    """One bounded sentence; omit it when none of the named content classes occurred."""
    from agent.context_compressor import (
        _PRUNED_TOOL_PLACEHOLDER, _collect_ghosted_skill_names, _content_has_images,
        _content_text_for_contains, _is_summary_stub, _tool_result_parts,
    )

    if tail_mode != "lean":
        return ""
    tools = reasoning = images = False
    skills = bool(_collect_ghosted_skill_names(turns))
    for msg in turns:
        content = msg.get("content")
        text = _content_text_for_contains(content)
        if msg.get("role") == "tool":
            tools |= (text == _PRUNED_TOOL_PLACEHOLDER or _is_summary_stub(text)
                      or text.startswith("[Duplicate tool output"))
        if msg.get("role") == "assistant":
            reasoning |= any(msg.get(key) for key in (
                "reasoning", "reasoning_content", "reasoning_details", "codex_reasoning_items",
            ))
        images |= (_content_has_images(_tool_result_parts(content))
                   or text.startswith("[screenshot removed")
                   or "[Attached image — stripped after compression]" in text)
    classes = [label for present, label in (
        (tools, "older tool outputs replaced by stubs"),
        (reasoning, "reasoning"),
        (images, "older images"),
        (skills, "skill bodies replaced by reload markers"),
    ) if present]
    if not classes:
        return ""
    return ("\n\nRemoved from context: " + ", ".join(classes)
            + "; quoted user text records what was said, not verified fact.")


def with_removal_notice(summary: str, turns: list[dict[str, Any]], compressor: Any) -> str:
    """Replace a carried notice line (an echoed previous pass) with this pass's notice, so there is only ever one."""
    tail_mode = getattr(compressor, "tail_mode", "lean")
    if tail_mode != "lean":
        return summary
    return _NOTICE_LINE_RE.sub("", summary).rstrip("\n") + build_removal_notice(turns, tail_mode)

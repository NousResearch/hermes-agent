"""Trajectory saving utilities and static helpers.

_convert_to_trajectory_format stays as an AIAgent method (batch_runner.py
calls agent._convert_to_trajectory_format). Only the static helpers and
the file-write logic live here.
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_SCRATCHPAD_OPEN_TAG = "<REASONING_SCRATCHPAD>"
_SCRATCHPAD_CLOSE_TAG = "</REASONING_SCRATCHPAD>"
_FENCED_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
_BLOCKQUOTE_RE = re.compile(r"(?m)^[ \t]*>.*(?:\n|$)")


def convert_scratchpad_to_think(content: str) -> str:
    """Convert <REASONING_SCRATCHPAD> tags to <think> tags."""
    if not content or _SCRATCHPAD_OPEN_TAG not in content:
        return content
    return content.replace(_SCRATCHPAD_OPEN_TAG, "<think>").replace(_SCRATCHPAD_CLOSE_TAG, "</think>")


def _strip_markdown_context_for_scratchpad_check(content: str) -> str:
    """Remove markdown contexts where scratchpad tags should be treated as literal text."""
    stripped = _FENCED_CODE_BLOCK_RE.sub("", content)
    stripped = _INLINE_CODE_RE.sub("", stripped)
    return _BLOCKQUOTE_RE.sub("", stripped)


def has_incomplete_scratchpad(content: str) -> bool:
    """Check if content has an opening <REASONING_SCRATCHPAD> without a closing tag."""
    if not content:
        return False
    visible_content = _strip_markdown_context_for_scratchpad_check(content)
    return visible_content.count(_SCRATCHPAD_OPEN_TAG) > visible_content.count(_SCRATCHPAD_CLOSE_TAG)


def save_trajectory(trajectory: List[Dict[str, Any]], model: str,
                    completed: bool, filename: str = None):
    """Append a trajectory entry to a JSONL file.

    Args:
        trajectory: The ShareGPT-format conversation list.
        model: Model name for metadata.
        completed: Whether the conversation completed successfully.
        filename: Override output filename. Defaults to trajectory_samples.jsonl
                  or failed_trajectories.jsonl based on ``completed``.
    """
    if filename is None:
        filename = "trajectory_samples.jsonl" if completed else "failed_trajectories.jsonl"

    entry = {
        "conversations": trajectory,
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "completed": completed,
    }

    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("Trajectory saved to %s", filename)
    except Exception as e:
        logger.warning("Failed to save trajectory: %s", e)

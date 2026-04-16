"""Trajectory saving utilities and static helpers.

_convert_to_trajectory_format stays as an AIAgent method (batch_runner.py
calls agent._convert_to_trajectory_format). Only the static helpers and
the file-write logic live here.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_REASONING_SCRATCHPAD_OPEN = "<REASONING_SCRATCHPAD>"
_REASONING_SCRATCHPAD_CLOSE = "</REASONING_SCRATCHPAD>"


def convert_scratchpad_to_think(content: str) -> str:
    """Convert <REASONING_SCRATCHPAD> tags to <think> tags."""
    if not content or "<REASONING_SCRATCHPAD>" not in content:
        return content
    return content.replace("<REASONING_SCRATCHPAD>", "<think>").replace("</REASONING_SCRATCHPAD>", "</think>")


def has_incomplete_scratchpad(content: str) -> bool:
    """True if any <REASONING_SCRATCHPAD> block is still open at end of content.

    Uses ordered depth matching so an earlier closing tag cannot hide a later
    unclosed opening (avoids false negatives vs. global substring checks).
    """
    if not content:
        return False
    depth = 0
    i = 0
    n = len(content)
    olen = len(_REASONING_SCRATCHPAD_OPEN)
    clen = len(_REASONING_SCRATCHPAD_CLOSE)
    while i < n:
        if content.startswith(_REASONING_SCRATCHPAD_OPEN, i):
            depth += 1
            i += olen
        elif content.startswith(_REASONING_SCRATCHPAD_CLOSE, i):
            if depth > 0:
                depth -= 1
            i += clen
        else:
            i += 1
    return depth > 0


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

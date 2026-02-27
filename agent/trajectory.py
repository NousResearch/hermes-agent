"""Trajectory saving utilities and static helpers.

This module handles exporting agent conversations in ShareGPT format
for training data generation and analysis.

Functions:
    convert_scratchpad_to_think(content):
        Converts <REASONING_SCRATCHPAD> tags to <think> tags for
        compatibility with training pipelines.
    
    has_incomplete_scratchpad(content):
        Checks if content has an unclosed reasoning scratchpad tag,
        which can happen when context is truncated mid-thought.
    
    save_trajectory(trajectory, model, completed, filename):
        Appends a trajectory entry to a JSONL file. Completed trajectories
        go to trajectory_samples.jsonl, failed ones to failed_trajectories.jsonl.

Note: The _convert_to_trajectory_format method stays as an AIAgent method
because batch_runner.py calls agent._convert_to_trajectory_format directly.
Only the static helpers and file-write logic live here.

Output Format (ShareGPT):
    {
        "conversations": [
            {"from": "system", "value": "..."},
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "...", "reasoning": "..."}
        ],
        "timestamp": "2024-01-01T00:00:00",
        "model": "anthropic/claude-sonnet-4",
        "completed": true
    }
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def convert_scratchpad_to_think(content: str) -> str:
    """Convert <REASONING_SCRATCHPAD> tags to <think> tags."""
    if not content or "<REASONING_SCRATCHPAD>" not in content:
        return content
    return content.replace("<REASONING_SCRATCHPAD>", "<think>").replace("</REASONING_SCRATCHPAD>", "</think>")


def has_incomplete_scratchpad(content: str) -> bool:
    """Check if content has an opening <REASONING_SCRATCHPAD> without a closing tag."""
    if not content:
        return False
    return "<REASONING_SCRATCHPAD>" in content and "</REASONING_SCRATCHPAD>" not in content


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

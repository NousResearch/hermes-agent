"""Trajectory saving + scratchpad helpers (``_convert_to_trajectory_format`` stays an AIAgent method — batch_runner.py calls it)."""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def convert_scratchpad_to_think(content: str) -> str:
    """Convert <REASONING_SCRATCHPAD> tags to <think> tags."""
    if not content or "<REASONING_SCRATCHPAD>" not in content:
        return content
    return content.replace("<REASONING_SCRATCHPAD>", "<think>").replace("</REASONING_SCRATCHPAD>", "</think>")


def has_incomplete_scratchpad(content: str) -> bool:
    """Whether content has an opening <REASONING_SCRATCHPAD> without a closing tag."""
    return bool(content) and "<REASONING_SCRATCHPAD>" in content and "</REASONING_SCRATCHPAD>" not in content


def _lock_append_handle(f, acquire: bool) -> None:
    """Exclusive whole-file lock on an append handle: ``flock`` on POSIX, a 1-byte
    ``msvcrt.locking`` range at offset 0 on Windows (append position is restored by the OS)."""
    if os.name == "nt":
        import msvcrt
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK if acquire else msvcrt.LK_UNLCK, 1)
        f.seek(0, os.SEEK_END)
    else:
        import fcntl
        fcntl.flock(f.fileno(), fcntl.LOCK_EX if acquire else fcntl.LOCK_UN)


def save_trajectory(trajectory: list[dict[str, Any]], model: str, completed: bool, filename: str | None = None):
    """Append a ShareGPT-format entry to a JSONL file (default trajectory_samples.jsonl / failed_trajectories.jsonl by ``completed``)."""
    if filename is None:
        filename = "trajectory_samples.jsonl" if completed else "failed_trajectories.jsonl"
    entry = {"conversations": trajectory, "timestamp": datetime.now().isoformat(), "model": model, "completed": completed}
    try:
        line = json.dumps(entry, ensure_ascii=False) + "\n"  # serialize before taking the lock
        with open(filename, "a", encoding="utf-8") as f:
            # Gateway sessions and batch workers append to the SAME default file; without an
            # exclusive lock around write+flush, entries larger than one write() interleave and the
            # JSONL stops parsing (#12684).
            _lock_append_handle(f, True)
            try:
                f.write(line)
                f.flush()
            finally:
                _lock_append_handle(f, False)
        logger.info("Trajectory saved to %s", filename)
    except Exception as e:
        logger.warning("Failed to save trajectory: %s", e)


def _trajectory_tool_responses(msg: dict[str, Any], messages: list[dict[str, Any]], start: int) -> tuple[list[str], int]:
    """Collect the ``<tool_response>`` blocks for the tool run starting at ``start``; returns ``(blocks, next_index)``."""
    tool_responses = []
    j = start
    while j < len(messages) and messages[j]["role"] == "tool":
        tool_msg = messages[j]
        tool_content = tool_msg["content"]
        try:  # pretty-print tool content if it looks like JSON
            if tool_content.strip().startswith(("{", "[")):
                tool_content = json.loads(tool_content)
        except (json.JSONDecodeError, AttributeError):
            pass
        tool_index = len(tool_responses)
        tool_name = (
            msg["tool_calls"][tool_index]["function"]["name"]
            if tool_index < len(msg["tool_calls"])
            else "unknown"
        )
        payload = json.dumps(
            {"tool_call_id": tool_msg.get("tool_call_id", ""), "name": tool_name, "content": tool_content},
            ensure_ascii=False,
        )
        tool_responses.append(f"<tool_response>\n{payload}\n</tool_response>")
        j += 1
    return tool_responses, j

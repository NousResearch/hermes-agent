"""Trajectory saving + replayable exploration-tree helpers."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.exploration_policy import ExplorationTree, build_exploration_tree

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


def _append_jsonl(entry: Dict[str, Any], filename: str) -> None:
    """Append one serialized entry while holding the same cross-process lock as trajectories."""
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    with open(filename, "a", encoding="utf-8") as f:
        _lock_append_handle(f, True)
        try:
            f.write(line)
            f.flush()
        finally:
            _lock_append_handle(f, False)


def _exploration_filename(filename: Optional[str], completed: bool) -> str:
    if filename is None:
        return "exploration_trees.jsonl" if completed else "failed_exploration_trees.jsonl"
    path = Path(filename)
    return str(path.with_name(f"{path.stem}.exploration.jsonl"))


def save_exploration_tree(tree: ExplorationTree, filename: str) -> None:
    """Append one versioned exploration tree without copying conversation contents."""
    try:
        _append_jsonl(tree.to_dict(), filename)
        logger.info("Exploration tree saved to %s", filename)
    except Exception as exc:
        # Exploration capture is secondary telemetry and must never turn a completed agent run into
        # a failed run when a secondary artifact cannot be written.
        logger.warning("Failed to save exploration tree: %s", exc)


def save_trajectory(
    trajectory: List[Dict[str, Any]],
    model: str,
    completed: bool,
    filename: str = None,
    exploration_filename: str = None,
):
    """Append a ShareGPT entry and its secret-free exploration tree to JSONL files."""
    if filename is None:
        filename = "trajectory_samples.jsonl" if completed else "failed_trajectories.jsonl"
    entry = {
        "conversations": trajectory,
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "completed": completed,
    }
    try:
        _append_jsonl(entry, filename)
        logger.info("Trajectory saved to %s", filename)
    except Exception as exc:
        logger.warning("Failed to save trajectory: %s", exc)
        return

    tree = build_exploration_tree(trajectory, completed=completed)
    save_exploration_tree(tree, exploration_filename or _exploration_filename(filename, completed))


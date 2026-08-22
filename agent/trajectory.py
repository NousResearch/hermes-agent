"""Trajectory saving utilities and static helpers.

_convert_to_trajectory_format stays as an AIAgent method (batch_runner.py
calls agent._convert_to_trajectory_format). Only the static helpers and
the file-write logic live here.
"""

import hashlib
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from hermes_constants import get_hermes_home
from hermes_cli.config import artifact_file_mode, secure_artifact_dir
from utils import open_private_append

logger = logging.getLogger(__name__)


def default_trajectory_path(completed: bool) -> Path:
    """Return the profile-aware implicit export path for the current CWD."""
    cwd = Path.cwd().resolve()
    name = re.sub(r"[^A-Za-z0-9._-]", "-", cwd.name).strip("-.") or "cwd"
    digest = hashlib.sha256(os.fsencode(cwd)).hexdigest()[:8]
    filename = "trajectory_samples.jsonl" if completed else "failed_trajectories.jsonl"
    return get_hermes_home() / "trajectories" / f"{name[:32]}-{digest}" / filename


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
        filename: Override output filename. The implicit default is stored in a
                  current-working-directory bucket under the active Hermes home.
    """
    entry = {
        "conversations": trajectory,
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "completed": completed,
    }

    try:
        if filename is None:
            filename = default_trajectory_path(completed)
            secure_artifact_dir(filename.parent)
        with open_private_append(filename, mode=artifact_file_mode()) as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("Trajectory saved to %s", filename)
    except Exception as e:
        logger.warning("Failed to save trajectory: %s", e)

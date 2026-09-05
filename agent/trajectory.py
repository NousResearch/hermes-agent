"""Trajectory saving + scratchpad helpers (``_convert_to_trajectory_format`` stays an AIAgent method — batch_runner.py calls it)."""

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
    """Whether content has an opening <REASONING_SCRATCHPAD> without a closing tag."""
    return bool(content) and "<REASONING_SCRATCHPAD>" in content and "</REASONING_SCRATCHPAD>" not in content


def save_trajectory(trajectory: List[Dict[str, Any]], model: str, completed: bool, filename: str = None):
    """Append a ShareGPT entry; implicit exports live under the active Hermes home."""
    entry = {"conversations": trajectory, "timestamp": datetime.now().isoformat(), "model": model, "completed": completed}
    implicit = filename is None
    try:
        if implicit:
            filename = default_trajectory_path(completed)
            secure_artifact_dir(filename.parent, tighten_existing=True)
        with open_private_append(filename, mode=artifact_file_mode(), tighten_existing=implicit) as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("Trajectory saved to %s", filename)
    except Exception as e:
        logger.warning("Failed to save trajectory: %s", e)

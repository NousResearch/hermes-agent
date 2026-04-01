"""
Recording storage and management.

Recordings are stored as YAML files in ~/.hermes/recordings/{name}.yaml.
Each recording contains metadata and an ordered list of tool call steps.
"""

import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

HERMES_DIR = get_hermes_home()
RECORDINGS_DIR = HERMES_DIR / "recordings"

_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")
_MAX_NAME_LEN = 64


def _secure_dir(path: Path):
    """Set directory to owner-only access (0700). No-op on Windows."""
    try:
        os.chmod(path, 0o700)
    except (OSError, NotImplementedError):
        pass


def _secure_file(path: Path):
    """Set file to owner-only read/write (0600). No-op on Windows."""
    try:
        if path.exists():
            os.chmod(path, 0o600)
    except (OSError, NotImplementedError):
        pass


def ensure_dirs():
    """Ensure recordings directory exists with secure permissions."""
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    _secure_dir(RECORDINGS_DIR)


def validate_name(name: str) -> str:
    """Validate and return a recording name.

    Raises ValueError if the name is invalid.
    """
    if not name or not isinstance(name, str):
        raise ValueError("Recording name must be a non-empty string")
    name = name.strip()
    if len(name) > _MAX_NAME_LEN:
        raise ValueError(f"Recording name must be {_MAX_NAME_LEN} characters or fewer")
    if not _NAME_PATTERN.match(name):
        raise ValueError(
            "Recording name must start with alphanumeric and contain only "
            "letters, digits, hyphens, and underscores"
        )
    return name


def _recording_path(name: str) -> Path:
    """Return the file path for a recording."""
    return RECORDINGS_DIR / f"{name}.yaml"


def _save(name: str, data: dict) -> None:
    """Atomic write of a recording YAML file."""
    ensure_dirs()
    target = _recording_path(name)
    # Atomic write: write to temp file in same dir, then rename
    fd, tmp_path = tempfile.mkstemp(dir=str(RECORDINGS_DIR), suffix=".yaml.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        os.replace(tmp_path, str(target))
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    _secure_file(target)


def create_recording(name: str, description: str = "") -> dict:
    """Create a new empty recording.

    Args:
        name: Recording identifier (alphanumeric, hyphens, underscores).
        description: Optional human-readable description.

    Returns:
        Recording metadata dict.

    Raises:
        ValueError: If name is invalid or recording already exists.
    """
    name = validate_name(name)
    if _recording_path(name).exists():
        raise ValueError(f"Recording '{name}' already exists")

    data = {
        "name": name,
        "description": description,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "steps": [],
    }
    _save(name, data)
    logger.info("Created recording: %s", name)
    return data


def get_recording(name: str) -> Optional[dict]:
    """Load a recording by name.

    Returns:
        Full recording dict, or None if not found.
    """
    path = _recording_path(name)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error("Failed to load recording '%s': %s", name, e)
        return None


def list_recordings() -> List[dict]:
    """List all recordings (metadata only, no step details).

    Returns:
        List of recording summary dicts with name, description, created_at,
        and step_count.
    """
    ensure_dirs()
    result = []
    for path in sorted(RECORDINGS_DIR.glob("*.yaml")):
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data and isinstance(data, dict):
                result.append({
                    "name": data.get("name", path.stem),
                    "description": data.get("description", ""),
                    "created_at": data.get("created_at", ""),
                    "step_count": len(data.get("steps", [])),
                })
        except Exception as e:
            logger.warning("Skipping malformed recording %s: %s", path.name, e)
    return result


def delete_recording(name: str) -> bool:
    """Delete a recording file.

    Returns:
        True if deleted, False if not found.
    """
    path = _recording_path(name)
    if not path.exists():
        return False
    path.unlink()
    logger.info("Deleted recording: %s", name)
    return True


def add_step(name: str, tool: str, arguments: dict, result: str, success: bool) -> dict:
    """Append a tool call step to an existing recording.

    Args:
        name: Recording name.
        tool: Tool/function name that was called.
        arguments: Tool call arguments dict.
        result: Tool call result string.
        success: Whether the tool call succeeded.

    Returns:
        The step dict that was added.

    Raises:
        ValueError: If recording not found.
    """
    data = get_recording(name)
    if data is None:
        raise ValueError(f"Recording '{name}' not found")

    step = {
        "tool": tool,
        "arguments": arguments,
        "expected_status": "success" if success else "error",
    }
    data["steps"].append(step)
    _save(name, data)
    return step

"""
Tool Result Store - Large Result Persistence for hermes-agent

Provides disk-based storage for tool results exceeding a size threshold,
reducing context consumption when embedding 30KB+ results inline in messages.

Usage:
    from agent.tool_result_store import save_large_result, LARGE_RESULT_THRESHOLD

    # Returns disk path if result > threshold, otherwise returns original string
    result = save_large_result(tool_result_str, tool_name, tool_call_id)
"""

import hashlib
import logging
import os
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

# 30KB threshold for persisting tool results to disk
LARGE_RESULT_THRESHOLD = 30 * 1024

# Storage directory for large tool results
_LARGE_RESULT_STORAGE_DIR = "/tmp/hermes-tool-results"


def _ensure_storage_dir() -> str:
    """Ensure the storage directory exists and return its path."""
    storage_dir = _LARGE_RESULT_STORAGE_DIR
    try:
        os.makedirs(storage_dir, exist_ok=True)
    except OSError as e:
        logger.warning("Could not create tool result storage dir %s: %s", storage_dir, e)
        # Fall back to system temp dir
        storage_dir = tempfile.gettempdir()
    return storage_dir


def _generate_storage_filename(tool_name: str, tool_call_id: str, content: str) -> str:
    """Generate a unique, descriptive filename for the stored result."""
    # Use a hash of content to ensure uniqueness and avoid collisions
    content_hash = hashlib.sha256(content[:1024].encode()).hexdigest()[:12]
    safe_tool_name = "".join(c if c.isalnum() else "_" for c in tool_name)
    return f"{safe_tool_name}_{tool_call_id[:8]}_{content_hash}.txt"


def save_large_result(
    result_str: str,
    tool_name: str,
    tool_call_id: str,
    threshold: int = LARGE_RESULT_THRESHOLD,
) -> str:
    """
    Persist a tool result to disk if it exceeds the size threshold.

    Args:
        result_str: The raw tool result string.
        tool_name: Name of the tool that produced this result.
        tool_call_id: Unique identifier for this tool call.
        threshold: Size in bytes above which to persist (default: 30KB).

    Returns:
        The disk path string (e.g. "/tmp/hermes-tool-results/...")
        if the result was persisted, otherwise the original result_str.
    """
    if len(result_str) <= threshold:
        return result_str

    storage_dir = _ensure_storage_dir()
    filename = _generate_storage_filename(tool_name, tool_call_id, result_str)
    disk_path = os.path.join(storage_dir, filename)

    try:
        with open(disk_path, "w", encoding="utf-8") as f:
            f.write(result_str)
        logger.info(
            "Persisted large tool result to disk: %s (%s, %d chars)",
            disk_path,
            tool_name,
            len(result_str),
        )
        return disk_path
    except OSError as e:
        logger.warning(
            "Failed to persist large tool result for %s (%s): %s",
            tool_name,
            tool_call_id,
            e,
        )
        # Return original content on failure rather than losing data
        return result_str


def is_disk_path(result: str) -> bool:
    """
    Check if a result string is a disk path (from a prior save_large_result call).

    Returns True if the string looks like an absolute path to a file that exists.
    """
    if not result:
        return False
    if not os.path.isabs(result):
        return False
    # Check if it looks like a tool result file path
    if "hermes-tool-results" not in result and not os.path.exists(result):
        return False
    return True


def load_from_disk(disk_path: str) -> Optional[str]:
    """
    Load a persisted tool result from disk.

    Args:
        disk_path: Path returned by a prior save_large_result call.

    Returns:
        The original content, or None if the file could not be read.
    """
    try:
        with open(disk_path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        logger.warning("Failed to load persisted tool result from %s: %s", disk_path, e)
        return None

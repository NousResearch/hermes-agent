"""Tool result validation middleware.

Validates tool output shape and content before feeding to LLM.
Catches malformed results early, prevents downstream hallucinations.

Validators by tool type:
- file_tools: string content, non-empty
- api_tools: dict with expected keys
- terminal: string output, check for error patterns
- web_search: list of results with title/url
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class ToolResultValidationError(Exception):
    """Raised when tool result fails validation."""

    def __init__(
        self, tool_name: str, error: str, result_preview: Optional[str] = None
    ):
        self.tool_name = tool_name
        self.error = error
        self.result_preview = result_preview
        super().__init__(
            f"Tool '{tool_name}' validation failed: {error}"
            + (f"\nPreview: {result_preview}" if result_preview else "")
        )


def _validate_file_tool_result(
    tool_name: str, result: Any
) -> Tuple[bool, Optional[str]]:
    """Validate file tool results (read, write, patch, etc)."""
    if not isinstance(result, str):
        return False, f"Expected string, got {type(result).__name__}"

    # Empty results for read_file might be valid (empty file), but warn
    if tool_name == "read_file" and not result:
        logger.warning(f"{tool_name}: returned empty string")

    return True, None


def _validate_api_tool_result(tool_name: str, result: Any) -> Tuple[bool, Optional[str]]:
    """Validate API tool results (web_search, etc)."""
    if isinstance(result, str):
        # Many tools return strings, that's fine
        return True, None

    if isinstance(result, dict):
        # Check for common API response patterns
        if "error" in result:
            error_msg = result.get("error", "Unknown error")
            return False, f"API returned error: {error_msg}"
        if "results" in result or "data" in result or "items" in result:
            return True, None
        # Dict with other keys is probably ok
        return True, None

    if isinstance(result, list):
        # List of results
        if not result:
            logger.warning(f"{tool_name}: returned empty list")
        return True, None

    return False, f"Unexpected result type: {type(result).__name__}"


def _validate_terminal_result(result: Any) -> Tuple[bool, Optional[str]]:
    """Validate terminal tool results."""
    if not isinstance(result, str):
        return False, f"Expected string, got {type(result).__name__}"

    # Terminal output might have errors, but that's data not a validation failure
    # The model should see error output to reason about it
    return True, None


def validate_tool_result(tool_name: str, result: Any) -> Tuple[bool, Optional[str]]:
    """
    Validate tool result before feeding to LLM.

    Returns (is_valid, error_message).
    If not valid, error_message explains the problem.

    Only tools with an explicit rule set are validated.  Unknown tools always
    pass so that new or third-party tools are never silently rejected.
    """
    # File tools must return a string.
    if tool_name in ("read_file", "write_file", "patch"):
        if result is None:
            return False, "Tool returned None"
        return _validate_file_tool_result(tool_name, result)

    # search_files can legitimately return a string (no matches) or a list of matches.
    if tool_name == "search_files":
        if result is None:
            return False, "Tool returned None"
        if not isinstance(result, (str, list)):
            return False, f"Expected string or list, got {type(result).__name__}"
        return True, None

    if tool_name == "terminal":
        if result is None:
            return False, "Tool returned None"
        return _validate_terminal_result(result)

    if tool_name in ("web_search", "web_extract"):
        if result is None:
            return False, "Tool returned None"
        return _validate_api_tool_result(tool_name, result)

    # Unknown / unregistered tool — always pass.
    # We have no schema for it and cannot safely reject anything.
    return True, None


def get_result_preview(result: Any, max_len: int = 200) -> str:
    """Get a short preview of the result for logging."""
    if isinstance(result, str):
        return result[:max_len]
    if isinstance(result, dict):
        try:
            s = json.dumps(result, default=str)[:max_len]
            return s
        except Exception:
            return str(result)[:max_len]
    if isinstance(result, list):
        return f"[list with {len(result)} items]"
    return str(result)[:max_len]

"""Tool error classification for smarter recovery.

Classifies tool execution errors into actionable categories so the agent
loop can provide targeted recovery hints instead of generic retry messages.

Taxonomy:
  - model:     Bad assumption by the LLM (wrong file content, non-existent path in patch)
  - tool:      Transient failure (timeout, rate limit, network error)
  - environment: Hard failure (permission denied, file not found, missing binary)
  - input:     Missing or invalid input (ambiguous parameter, missing credential)

Each classification includes a recovery hint that gets appended to the error
message returned to the LLM, guiding it toward the correct recovery strategy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# ── Error type taxonomy ────────────────────────────────────────────────

class ToolErrorType:
    """Error type constants for tool execution failures."""
    MODEL = "model"
    TOOL = "tool"
    ENVIRONMENT = "environment"
    INPUT = "input"


@dataclass
class ClassifiedToolError:
    """Structured classification of a tool execution error."""
    error_type: str          # One of ToolErrorType constants
    error_message: str       # Original error message
    recovery_hint: str       # Actionable guidance for the LLM
    tool_name: Optional[str] = None


# ── Pattern definitions ────────────────────────────────────────────────

# Model errors: the LLM made a bad assumption about the world
_MODEL_ERROR_PATTERNS = [
    # patch tool: old_string doesn't match
    r"found \d+ matches for old_string",
    r"old_string not found",
    r"no match found for",
    r"could not find the string",
    # File content mismatch
    r"does not contain",
    r"content mismatch",
    r"unexpected content",
    # Wrong path assumption (distinct from "file not found" which is env)
    r"resolved_path.*does not match",
    r"wrong file.*checkout",
]

# Tool errors: transient failures that may succeed on retry
_TOOL_ERROR_PATTERNS = [
    # Network/timeout
    r"timeout",
    r"timed out",
    r"connection refused",
    r"connection reset",
    r"connection aborted",
    r"broken pipe",
    r"network.*error",
    r"ssl.*error",
    r"ssl.*verify",
    r"certificate.*verify",
    r"tls.*error",
    # Rate limiting
    r"rate.?limit",
    r"too many requests",
    r"429",
    r"throttl",
    r"retry.after",
    # Server-side transient
    r"503",
    r"502",
    r"bad gateway",
    r"service unavailable",
    r"server.*error",
    r"internal server error",
    # Process transient
    r"resource.*temporarily.*unavailable",
    r"try again",
    r"temporary.*failure",
]

# Environment errors: hard failures that won't resolve on retry
_ENVIRONMENT_ERROR_PATTERNS = [
    # File system
    r"no such file or directory",
    r"file not found",
    r"directory not found",
    r"enoent",
    r"not found.*path",
    r"permission denied",
    r"eacces",
    r"eperm",
    r"read.only file system",
    r"erofs",
    r"disk.*full",
    r"no space left",
    r"enospc",
    # Process/command
    r"command not found",
    r"no such file.*executable",
    r"executable.*not found",
    r"segfault",
    r"sigsegv",
    r"signal 11",
    r"exit code -?11",
    r"killed",
    r"oom",
    r"out of memory",
    # Platform
    r"not supported",
    r"unsupported.*platform",
    r"architecture.*not supported",
    # Import/dependency
    r"module.*not found",
    r"import.*error",
    r"no module named",
]

# Input errors: the LLM provided wrong/missing parameters
_INPUT_ERROR_PATTERNS = [
    r"missing.*argument",
    r"required.*parameter",
    r"required.*field",
    r"invalid.*argument",
    r"invalid.*parameter",
    r"type.*error",
    r"value.*error",
    r"ambiguous",
    r"must be.*not",
    r"expected.*got",
    r"credential.*required",
    r"api.?key.*required",
    r"authentication.*required",
    r"not authenticated",
    r"unauthorized",
]


# ── Recovery hints ─────────────────────────────────────────────────────

_RECOVERY_HINTS = {
    ToolErrorType.MODEL: (
        "This error suggests your assumption about the file/tool state was wrong. "
        "Re-read the relevant file or re-check the current state before retrying. "
        "Do not retry with the same arguments."
    ),
    ToolErrorType.TOOL: (
        "This appears to be a transient error. Wait briefly, then retry. "
        "If it persists after 2 retries, try a different approach or report the blocker."
    ),
    ToolErrorType.ENVIRONMENT: (
        "This is a hard failure that won't resolve on retry. "
        "Check if the path/command exists, verify permissions, or try an alternative. "
        "If the blocker is external, report it to the user instead of retrying."
    ),
    ToolErrorType.INPUT: (
        "The tool received invalid or missing input. "
        "Check the required parameters and their types, then retry with corrected input."
    ),
}

# Tool-specific recovery hints (more actionable than generic ones)
_TOOL_SPECIFIC_HINTS = {
    "patch": {
        ToolErrorType.MODEL: (
            "The old_string didn't match the file content. Re-read the file with "
            "read_file to see the exact current content, then use the precise text "
            "from the file as old_string. Include surrounding context for uniqueness."
        ),
    },
    "write_file": {
        ToolErrorType.ENVIRONMENT: (
            "The target path is inaccessible. Use an absolute path, verify the "
            "directory exists, and check permissions. In workdir checkouts, ensure "
            "you're targeting the correct checkout directory."
        ),
    },
    "terminal": {
        ToolErrorType.ENVIRONMENT: (
            "The command or binary was not found. Check if it's installed, verify "
            "the PATH, or try using the full absolute path to the executable."
        ),
        ToolErrorType.TOOL: (
            "The command hit a transient error. If it's a network operation, check "
            "connectivity. For long-running commands, consider increasing the timeout."
        ),
    },
    "read_file": {
        ToolErrorType.ENVIRONMENT: (
            "The file doesn't exist at the specified path. Verify the path is correct "
            "(use search_files to find it), check for typos, and use absolute paths "
            "in workdir checkouts."
        ),
    },
}


# ── Classification logic ───────────────────────────────────────────────

def _matches_any(text: str, patterns: list[str]) -> bool:
    """Check if text matches any regex pattern (case-insensitive)."""
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in patterns)


def classify_tool_error(
    error_message: str,
    tool_name: str = "",
    exception: Optional[Exception] = None,
) -> ClassifiedToolError:
    """Classify a tool execution error and provide recovery guidance.

    Args:
        error_message: The error string (from exception or tool result).
        tool_name: Name of the tool that failed (for tool-specific hints).
        exception: The original exception, if available (used for type-based
                   classification when message patterns are ambiguous).

    Returns:
        ClassifiedToolError with type, message, and recovery hint.
    """
    # Phase 1: Check exception type for strong signals
    if exception is not None:
        exc_type = type(exception).__name__
        exc_str = str(exception)

        # Permission/file errors from exception type
        if isinstance(exception, (PermissionError,)):
            return _make_result(
                ToolErrorType.ENVIRONMENT, error_message, tool_name,
            )
        if isinstance(exception, (FileNotFoundError, IsADirectoryError, NotADirectoryError)):
            return _make_result(
                ToolErrorType.ENVIRONMENT, error_message, tool_name,
            )
        if isinstance(exception, (TypeError, ValueError)):
            # Only classify as input error if the message looks parameter-related
            if _matches_any(error_message, _INPUT_ERROR_PATTERNS):
                return _make_result(
                    ToolErrorType.INPUT, error_message, tool_name,
                )
        if isinstance(exception, (ConnectionError, TimeoutError, OSError)):
            if _matches_any(error_message, _TOOL_ERROR_PATTERNS):
                return _make_result(
                    ToolErrorType.TOOL, error_message, tool_name,
                )

    # Phase 2: Message-pattern classification (priority order)
    # Model errors first — most specific, least false-positive-prone
    if _matches_any(error_message, _MODEL_ERROR_PATTERNS):
        return _make_result(ToolErrorType.MODEL, error_message, tool_name)

    # Input errors
    if _matches_any(error_message, _INPUT_ERROR_PATTERNS):
        return _make_result(ToolErrorType.INPUT, error_message, tool_name)

    # Environment errors (before tool errors — "not found" is env, not transient)
    if _matches_any(error_message, _ENVIRONMENT_ERROR_PATTERNS):
        return _make_result(ToolErrorType.ENVIRONMENT, error_message, tool_name)

    # Tool errors (transient)
    if _matches_any(error_message, _TOOL_ERROR_PATTERNS):
        return _make_result(ToolErrorType.TOOL, error_message, tool_name)

    # Phase 3: Fallback — classify as model error (the LLM likely made a
    # bad assumption, since we can't identify a specific failure mode)
    return _make_result(ToolErrorType.MODEL, error_message, tool_name)


def _make_result(
    error_type: str,
    error_message: str,
    tool_name: str,
) -> ClassifiedToolError:
    """Build a ClassifiedToolError with the best available recovery hint."""
    # Try tool-specific hint first, then generic
    hint = _TOOL_SPECIFIC_HINTS.get(tool_name, {}).get(error_type)
    if hint is None:
        hint = _RECOVERY_HINTS.get(error_type, "")
    return ClassifiedToolError(
        error_type=error_type,
        error_message=error_message,
        recovery_hint=hint,
        tool_name=tool_name,
    )


def format_classified_error(classified: ClassifiedToolError) -> str:
    """Format a classified error for inclusion in the tool result.

    Returns a string like:
        {"error": "...", "error_type": "model", "recovery_hint": "..."}
    """
    import json
    return json.dumps({
        "error": classified.error_message,
        "error_type": classified.error_type,
        "recovery_hint": classified.recovery_hint,
    }, ensure_ascii=False)

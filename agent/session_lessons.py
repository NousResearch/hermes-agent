"""Session lessons — auto-extract and inject lessons from recent failures.

Queries the session database for recent tool failures and error patterns,
extracts actionable lessons, and formats them for system prompt injection.

This addresses the "context amnesia" problem where learned failure patterns
are not retained across session boundaries.
"""

import json
import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Keywords that indicate failure/error in tool results
_FAILURE_INDICATORS = (
    "error", "failed", "exception", "traceback", "permission denied",
    "not found", "timeout", "refused", "denied", "blocked",
    "unresolved", "undefined", "importerror", "modulenotfounderror",
)

# Maximum number of lessons to inject into system prompt
MAX_LESSONS = 5

# Minimum occurrences of a pattern to qualify as a lesson
MIN_OCCURRENCES = 2

# Maximum age of sessions to search (in recent N sessions)
RECENT_SESSION_LIMIT = 20


def extract_lessons_from_sessions(
    session_db: Any,
    current_session_id: Optional[str] = None,
    limit: int = MAX_LESSONS,
) -> List[Dict[str, str]]:
    """Extract lessons from recent session failures.

    Returns a list of lesson dicts with keys:
        - pattern: the recurring failure pattern
        - suggestion: actionable advice
        - frequency: how many times it occurred
    """
    if not session_db:
        return []

    lessons = []

    try:
        # Search for recent error patterns in tool results
        error_messages = session_db.search_messages(
            query="error OR failed OR exception OR traceback",
            role_filter=["tool"],
            limit=50,
        )

        if not error_messages:
            return []

        # Extract and count error patterns
        pattern_counts: Counter = Counter()
        pattern_examples: Dict[str, str] = {}

        for msg in error_messages:
            # Skip current session
            if msg.get("session_id") == current_session_id:
                continue

            content = msg.get("content", "")
            if not content:
                continue

            # Extract the core error pattern
            pattern = _extract_error_pattern(content)
            if pattern:
                pattern_counts[pattern] += 1
                if pattern not in pattern_examples:
                    pattern_examples[pattern] = content[:200]

        # Filter to recurring patterns
        for pattern, count in pattern_counts.most_common(limit * 2):
            if count < MIN_OCCURRENCES:
                break

            suggestion = _suggest_fix(pattern, pattern_examples.get(pattern, ""))
            if suggestion:
                lessons.append({
                    "pattern": pattern,
                    "suggestion": suggestion,
                    "frequency": str(count),
                })
                if len(lessons) >= limit:
                    break

    except Exception as e:
        logger.debug("Failed to extract session lessons: %s", e)

    return lessons


def format_lessons_for_prompt(lessons: List[Dict[str, str]]) -> str:
    """Format lessons into a system prompt block.

    Returns empty string if no lessons.
    """
    if not lessons:
        return ""

    lines = ["## Recent Session Lessons (auto-extracted)"]
    lines.append("")
    lines.append("These patterns were observed in recent sessions. Avoid repeating them:")
    lines.append("")

    for i, lesson in enumerate(lessons, 1):
        lines.append(f"{i}. **Pattern**: {lesson['pattern']}")
        lines.append(f"   **Fix**: {lesson['suggestion']}")
        lines.append(f"   **Seen**: {lesson['frequency']} times recently")
        lines.append("")

    return "\n".join(lines)


def _extract_error_pattern(content: str) -> Optional[str]:
    """Extract a normalized error pattern from a tool result string."""
    if not content:
        return None

    lower = content.lower()

    # Check if this is actually an error
    if not any(indicator in lower for indicator in _FAILURE_INDICATORS):
        return None

    # Try to extract specific error types
    patterns = [
        # Python exceptions
        (r'(\w+Error):\s*(.+?)(?:\n|$)', lambda m: f"{m.group(1)}: {m.group(2).strip()[:80]}"),
        # Exit codes
        (r'exit[_\s]?code[:\s]*(\d+)', lambda m: f"exit code {m.group(1)}"),
        # Command not found
        (r'(\w+):\s*command not found', lambda m: f"command not found: {m.group(1)}"),
        # Permission denied
        (r'permission denied.*?(/[\w/.-]+)', lambda m: f"permission denied: {m.group(1)}"),
        # File not found
        (r'no such file.*?(/[\w/.-]+)', lambda m: f"file not found: {m.group(1)}"),
        # Connection errors
        (r'(connection\s+(refused|reset|timed?\s*out))', lambda m: m.group(1)),
        # Import errors
        (r'cannot import name [\'"](\w+)[\'"]', lambda m: f"import error: {m.group(1)}"),
    ]

    for regex, formatter in patterns:
        match = re.search(regex, content, re.IGNORECASE)
        if match:
            try:
                return formatter(match)
            except Exception:
                continue

    # Generic fallback: first line with "error" in it
    for line in content.split("\n"):
        if "error" in line.lower() and len(line.strip()) > 10:
            return line.strip()[:100]

    return None


def _suggest_fix(pattern: str, example: str) -> Optional[str]:
    """Generate an actionable suggestion based on the error pattern."""
    pattern_lower = pattern.lower()

    suggestions = {
        "modulenotfounderror": "The module is not available. Use `terminal` instead of `execute_code` for scripts requiring system modules, or install the package first.",
        "importerror": "Import failed. Check if the package is installed in the active environment.",
        "permission denied": "Permission issue. Check file ownership/permissions, or use a different path.",
        "file not found": "File not found. Use `search_files` to locate the correct path before operations.",
        "command not found": "Command not found. Install the package or use an alternative tool.",
        "timeout": "Operation timed out. Consider breaking the task into smaller pieces or using a different approach.",
        "connection refused": "Connection refused. Check if the service is running and accessible.",
        "exit code 1": "Command exited with error. Check the full output for details.",
        "exit code 127": "Command not found (exit 127). The binary is not installed or not in PATH.",
        "exit code 137": "Process killed (OOM or signal). Reduce memory usage or task scope.",
    }

    for key, suggestion in suggestions.items():
        if key in pattern_lower:
            return suggestion

    # Generic suggestion
    if "error" in pattern_lower or "failed" in pattern_lower:
        return f"This error occurred repeatedly. When encountering '{pattern[:60]}', try a fundamentally different approach."

    return None

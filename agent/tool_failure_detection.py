"""Task-level failure detection for terminal and execute_code tool output.

Detects semantic failures (HTTP errors, tracebacks, connection errors) even
when the tool process itself reports success (exit code 0 / status="success").

Used by both agent/display.py (CLI presentation) and agent/tool_guardrails.py
(guardrail counting) so both paths agree on what counts as a failure.
"""

from __future__ import annotations

import re

# Structured regexes for task-level failure detection.
# Prefer explicit signals (stack traces, HTTP status lines, CLI error phrases)
# over bare keywords so we don't false-positive on:
#   "0 failed", "expected error", "404 page test passed",
#   "Expected Not Found response was returned", "no matching files found"
_TASK_FAILURE_REGEXES = (
    # Python / JS / runtime stack traces
    re.compile(r"(?im)^traceback \(most recent call last\):"),
    # "Error: ..." or "Fatal: ..." at start of line (with leading whitespace ok)
    re.compile(r"(?im)^\s*(?:error|fatal):\s+\S+"),
    # "❌ Error:" or similar emoji-prefixed error markers
    re.compile(r"(?im)^\s*❌\s*(?:error|fatal):\s+\S+"),
    # Indented "  error:" (common in JSON/tool output)
    re.compile(r"(?im)^\s{2,}error:"),

    # HTTP failures — require explicit HTTP/status context, not bare "400"/"404".
    # e.g. "HTTP/1.1 404 Not Found", "HTTP 500", "HTTP 400 Bad Request"
    re.compile(r"(?i)\bhttp(?:/\d(?:\.\d)?)?\s+(?:4\d\d|5\d\d)\b"),
    # e.g. "status_code=500", "status: 404", "code: 500"
    re.compile(r"(?i)\b(?:status|status_code|code)\s*[:=]\s*(?:4\d\d|5\d\d)\b"),
    # Explicit HTTP error class names
    re.compile(r"(?i)\b(?:HTTPError|Client Error|Server Error)\b"),

    # Common command/runtime failures (these are specific enough to not false-positive).
    re.compile(r"(?i)\bcommand not found\b"),
    re.compile(r"(?i)\bno such file or directory\b"),
    re.compile(r"(?i)\bpermission denied\b"),
    re.compile(r"(?i)\bconnection (?:refused|reset|timed out)\b"),
    re.compile(r"(?i)\btimeout expired\b"),
)


def output_indicates_task_failure(output_text: str) -> bool:
    """Check if output text contains signs of task-level failure.

    Checks both the beginning and end of the output — long logs often put
    the traceback or HTTP failure at the bottom.
    """
    if len(output_text) > 4000:
        sample = output_text[:2000] + "\n" + output_text[-2000:]
    else:
        sample = output_text
    return any(pattern.search(sample) for pattern in _TASK_FAILURE_REGEXES)

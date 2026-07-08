"""
provenance_hook — Track tool operations for /goal verification and provenance.

Writes:
  ~/.hermes/logs/write_file_hook.log   (WRITE provenance)
      ISO_TIMESTAMP | SESSION_ID | ABSOLUTE_PATH

  ~/.hermes/logs/read_provenance.log   (READ provenance)
      ISO_TIMESTAMP | SESSION_ID | TOOL_NAME | TRUST | SOURCE

TRUST levels:
  EXTERNAL — web/external content (always cross-verify)
  INTERNAL — local filesystem / trusted source
  INTERNAL_ACTION — agent action (send_message/delegate_task), not a read source
  UNKNOWN  — genuinely unclassified tool (needs _detect_trust update)
  LEGACY   — old-format log lines pre-dating trust field

goal_check.sh cross-references these logs with done entries to verify
outputs, show source provenance, and detect trust gaps.
"""
import fcntl
import json
import logging
import os
import re
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── Write tools — arg key for the output path ──
WRITE_TOOLS = {
    "write_file":                    "path",
    "patch":                         "path",
    "mcp_workspace_rw_write_file":   "path",
    "mcp_hermes_backup_rw_write_file": "path",
    "mcp_workspace_rw_edit_file":    "path",
    "mcp_hermes_backup_rw_edit_file": "path",  # S1: backup MCP edit_file 之前被遗漏
}

# ── Read tools — arg key for the source identifier ──
READ_TOOLS = {
    "web_extract":      "urls",
    "web_search":       "query",
    "read_file":        "path",
    "session_search":   "query",
    "browser_navigate": "url",
    "search_files":     "pattern",
    "vision_analyze":   "image_url",
    # delegate_task handled specially in _log_read (single + batch modes)
    "send_message":     "message",
}


def _get_hermes_home():
    return os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))


def _is_success(result):
    """Return True if the tool call succeeded (no error in result).

    Handles injection-guard wrapping (<<<EXTERNAL_UNTRUSTED_CONTENT>>>)
    that may surround error payloads — strip markers before JSON parse.
    """
    try:
        text = result if isinstance(result, str) else ""
        # Strip injection-guard boundary markers
        text = re.sub(r'<<<\w+>>>', '', text)
        text = re.sub(r'<<<END_\w+>>>', '', text).strip()
        if not text:
            return True  # empty after stripping = no error signal
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            if parsed.get("error"):
                return False
            # Some tools return {"status": "error", ...}
            if parsed.get("status") == "error":
                return False
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    return True


def _detect_trust(tool_name, args):
    """Classify trust by tool type.

    post_tool_call fires BEFORE transform_tool_result, so injection-guard
    markers are not yet applied. Tool-type classification is the fallback.
    """
    if tool_name in ("web_search", "web_extract", "browser_navigate"):
        return "EXTERNAL"
    if tool_name in ("session_search", "search_files"):
        return "INTERNAL"
    if tool_name in ("read_file", "vision_analyze"):
        arg_key = "path" if tool_name == "read_file" else "image_url"
        val = str(args.get(arg_key, ""))
        if val.startswith(("http://", "https://")):
            return "EXTERNAL"
        return "INTERNAL"
    if tool_name in ("delegate_task", "send_message"):
        return "INTERNAL_ACTION"
    return "UNKNOWN"


def _append_log(log_path, line):
    """Append a line to a log file with fcntl-based locking."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(line + "\n")
                f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        logger.warning("provenance_hook: failed to write %s: %s", log_path, e)


def _log_write(tool_name, args, session_id):
    """Log a write operation."""
    arg_key = WRITE_TOOLS[tool_name]
    path = args.get(arg_key, "")
    if not path:
        return

    abs_path = os.path.abspath(os.path.expanduser(str(path)))
    timestamp = datetime.now(timezone.utc).isoformat()
    session = session_id or "unknown"

    log_path = os.path.join(_get_hermes_home(), "logs", "write_file_hook.log")
    _append_log(log_path, f"{timestamp} | {session} | {abs_path}")


def _log_read(tool_name, args, session_id):
    """Log a read operation with trust level."""
    # delegate_task batch mode: extract goals from "tasks" list
    if tool_name == "delegate_task":
        tasks = args.get("tasks")
        if tasks and isinstance(tasks, list):
            values = [t.get("goal", "") for t in tasks if isinstance(t, dict)]
        else:
            values = [args.get("goal", "")]
    else:
        arg_key = READ_TOOLS[tool_name]
        values = args.get(arg_key, "")
        if not values:
            return
        if not isinstance(values, list):
            values = [values]

    timestamp = datetime.now(timezone.utc).isoformat()
    session = session_id or "unknown"
    trust = _detect_trust(tool_name, args)
    log_path = os.path.join(_get_hermes_home(), "logs", "read_provenance.log")

    for val in values:
        val_str = str(val).strip()
        if not val_str:
            continue
        # Resolve file/image paths to absolute
        if tool_name in ("read_file", "vision_analyze"):
            val_str = os.path.abspath(os.path.expanduser(val_str))
        _append_log(log_path, f"{timestamp} | {session} | {tool_name} | {trust} | {val_str}")


def _post_tool_call(
    tool_name: str,
    args: dict,
    result: str = "",
    session_id: str = "",
    duration_ms: int = 0,
    **kwargs,
):
    """Log successful tool calls for provenance tracking."""
    if not _is_success(result):
        return

    if tool_name in WRITE_TOOLS:
        _log_write(tool_name, args, session_id)
    elif tool_name in READ_TOOLS:
        _log_read(tool_name, args, session_id)


def register(ctx):
    ctx.register_hook("post_tool_call", _post_tool_call)
    all_tools = list(WRITE_TOOLS) + [t for t in READ_TOOLS]
    logger.info(
        "provenance_hook v3.0 registered (post_tool_call — tracking %d tools)",
        len(all_tools),
    )

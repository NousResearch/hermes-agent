"""Semantic tool-call repair: fix common model mistakes in tool arguments AFTER
syntactic sanitization but BEFORE dispatch. Cache-safe (no system-prompt changes).

Repairs are logged for observability and applied in-place on the tool_calls list.
This module is imported lazily from turn_tool_validation to avoid cycle imports.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("agent.conversation_loop")

# Patterns that indicate a user correction or negative feedback
_CORRECTION_PATTERNS = re.compile(
    r"\b(don'?t|do not|never|stop|no[,.]?\s+don'?t|wrong|incorrect|fix this|"
    r"that'?s not right|i said|not like that|undo|revert)\b",
    re.IGNORECASE,
)


def _safe_json_loads(raw: Any) -> Optional[Dict[str, Any]]:
    """Best-effort JSON parse of tool-call arguments; None on failure."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


def _resolve_workspace_root(agent: Any) -> Optional[Path]:
    """Get the workspace root from agent context for path relativization."""
    # Try multiple sources in priority order
    for attr in ("workspace_root", "working_directory", "cwd"):
        val = getattr(agent, attr, None)
        if val:
            p = Path(val)
            if p.is_absolute():
                return p
    # Fall back to HERMES_HOME parent as workspace heuristic
    try:
        from hermes_constants import get_hermes_home
        home = get_hermes_home()
        if home and home.parent != Path.home():
            return home.parent
    except Exception:
        pass
    return None


def _relativize_path(value: str, workspace: Path) -> Optional[str]:
    """Convert an absolute path to workspace-relative if it lives under workspace."""
    try:
        p = Path(value)
        if p.is_absolute() and p != workspace:
            rel = p.relative_to(workspace)
            result = str(rel)
            # Only return if actually shorter and valid
            if len(result) < len(value) and not result.startswith(".."):
                return result
    except (ValueError, OSError):
        pass
    return None


def _infer_missing_param(
    tool_name: str, param_name: str, args: Dict[str, Any], agent: Any, messages: List[Dict[str, Any]]
) -> Optional[Any]:
    """Try to infer a missing required parameter from session context.

    Conservative: only infers when there's exactly one plausible value.
    Returns None if inference is uncertain.
    """
    # file_path / path params: check if recent tool results reference a single file
    if param_name in ("file_path", "path", "notebook_path"):
        # Look at recent assistant tool calls for file references
        recent_files: List[str] = []
        for msg in reversed(messages[-10:]):
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "assistant":
                for tc in (msg.get("tool_calls") or []):
                    fn = tc.get("function", {})
                    tc_args = _safe_json_loads(fn.get("arguments"))
                    if tc_args:
                        for key in ("file_path", "path"):
                            if key in tc_args and isinstance(tc_args[key], str):
                                recent_files.append(tc_args[key])
            elif msg.get("role") == "tool":
                content = msg.get("content", "")
                # Tool results sometimes echo the path
                if isinstance(content, str) and content.startswith("/") and len(content) < 500:
                    pass  # Don't blindly trust tool output paths

        if len(recent_files) == 1:
            return recent_files[0]

    # command for terminal: check if user just asked to run something specific
    if param_name == "command" and tool_name == "terminal":
        # Don't infer commands — too risky
        return None

    return None


def repair_tool_call_arguments(
    agent: Any, tool_calls: List[Any], messages: List[Dict[str, Any]]
) -> int:
    """Apply semantic repairs to tool-call arguments in place.

    Returns the number of repairs made. Repairs are logged at INFO level.
    This function is cache-safe: it modifies only tool-call argument dicts,
    never the system prompt or message history structure.
    """
    repairs = 0
    workspace = _resolve_workspace_root(agent)

    for tc in tool_calls:
        if not hasattr(tc, "function") or not tc.function:
            continue

        fn = tc.function
        tool_name = getattr(fn, "name", "") or ""
        raw_args = getattr(fn, "arguments", None)

        args = _safe_json_loads(raw_args)
        if args is None:
            continue

        modified = False

        # Repair 1: Relativize absolute paths that should be workspace-relative
        if workspace:
            for key in ("file_path", "path", "notebook_path", "directory", "dir"):
                if key in args and isinstance(args[key], str):
                    original = args[key]
                    relativized = _relativize_path(original, workspace)
                    if relativized and relativized != original:
                        args[key] = relativized
                        logger.info(
                            "tool_call_repair: %s.%s relativized %r -> %r",
                            tool_name, key, original, relativized,
                        )
                        repairs += 1
                        modified = True

        # Repair 2: Type coercion for safe conversions
        # String numbers → actual numbers for numeric params
        for key, value in list(args.items()):
            if isinstance(value, str) and key not in ("content", "old_string", "new_string",
                                                       "command", "query", "prompt", "text",
                                                       "description", "message", "name",
                                                       "file_path", "path", "pattern"):
                stripped = value.strip()
                # Integer coercion
                if stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit()):
                    try:
                        args[key] = int(stripped)
                        logger.info(
                            "tool_call_repair: %s.%s coerced string %r -> int %d",
                            tool_name, key, value, args[key],
                        )
                        repairs += 1
                        modified = True
                    except (ValueError, OverflowError):
                        pass
                # Boolean coercion
                elif stripped.lower() in ("true", "false"):
                    args[key] = stripped.lower() == "true"
                    logger.info(
                        "tool_call_repair: %s.%s coerced string %r -> bool %s",
                        tool_name, key, value, args[key],
                    )
                    repairs += 1
                    modified = True

        # Repair 3: Infer missing required params from context (conservative)
        # Only for well-known tools with predictable params
        if tool_name in ("read_file", "write_file", "edit_file", "patch"):
            if "file_path" not in args:
                inferred = _infer_missing_param(tool_name, "file_path", args, agent, messages)
                if inferred:
                    args["file_path"] = inferred
                    logger.info(
                        "tool_call_repair: %s inferred missing file_path=%r",
                        tool_name, inferred,
                    )
                    repairs += 1
                    modified = True

        # Write back modified args
        if modified:
            try:
                fn.arguments = json.dumps(args, ensure_ascii=False)
            except (TypeError, ValueError):
                pass

    return repairs


def score_message_correction_weight(content: str) -> float:
    """Score how likely a message contains user corrections/feedback.

    Returns a weight multiplier: 1.0 = normal, >1.0 = contains corrections.
    Used by context compression to prioritize retention.
    """
    if not content or not isinstance(content, str):
        return 1.0

    matches = _CORRECTION_PATTERNS.findall(content)
    if not matches:
        return 1.0

    # Each correction signal adds weight; cap at 5x
    return min(1.0 + len(matches) * 0.8, 5.0)
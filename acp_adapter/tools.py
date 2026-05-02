"""ACP tool-call helpers for mapping hermes tools to ACP ToolKind and building content."""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

import acp
from acp.schema import (
    ToolCallLocation,
    ToolCallStart,
    ToolCallProgress,
    ToolKind,
)

# ---------------------------------------------------------------------------
# Map hermes tool names -> ACP ToolKind
# ---------------------------------------------------------------------------

TOOL_KIND_MAP: Dict[str, ToolKind] = {
    # File operations
    "read_file": "read",
    "write_file": "edit",
    "patch": "edit",
    "search_files": "search",
    # Terminal / execution
    "terminal": "execute",
    "process": "execute",
    "execute_code": "execute",
    # Web / fetch
    "web_search": "fetch",
    "web_extract": "fetch",
    # Browser
    "browser_navigate": "fetch",
    "browser_click": "execute",
    "browser_type": "execute",
    "browser_snapshot": "read",
    "browser_vision": "read",
    "browser_scroll": "execute",
    "browser_press": "execute",
    "browser_back": "execute",
    "browser_get_images": "read",
    # Agent internals
    "delegate_task": "execute",
    "vision_analyze": "read",
    "image_generate": "execute",
    "text_to_speech": "execute",
    # Thinking / meta
    "_thinking": "think",
}


def get_tool_kind(tool_name: str) -> ToolKind:
    """Return the ACP ToolKind for a hermes tool, defaulting to 'other'."""
    return TOOL_KIND_MAP.get(tool_name, "other")


def make_tool_call_id() -> str:
    """Generate a unique tool call ID."""
    return f"tc-{uuid.uuid4().hex[:12]}"


def build_tool_title(tool_name: str, args: Dict[str, Any]) -> str:
    """Build a human-readable title for a tool call."""
    if tool_name == "terminal":
        cmd = args.get("command", "")
        if len(cmd) > 80:
            cmd = cmd[:77] + "..."
        return f"terminal: {cmd}"
    if tool_name == "read_file":
        return f"read: {args.get('path', '?')}"
    if tool_name == "write_file":
        return f"write: {args.get('path', '?')}"
    if tool_name == "patch":
        mode = args.get("mode", "replace")
        path = args.get("path", "?")
        return f"patch ({mode}): {path}"
    if tool_name == "search_files":
        return f"search: {args.get('pattern', '?')}"
    if tool_name == "web_search":
        return f"web search: {args.get('query', '?')}"
    if tool_name == "web_extract":
        urls = args.get("urls", [])
        if urls:
            return f"extract: {urls[0]}" + (f" (+{len(urls)-1})" if len(urls) > 1 else "")
        return "web extract"
    if tool_name == "delegate_task":
        goal = args.get("goal", "")
        if goal and len(goal) > 60:
            goal = goal[:57] + "..."
        return f"delegate: {goal}" if goal else "delegate task"
    if tool_name == "execute_code":
        return "execute code"
    if tool_name == "vision_analyze":
        return f"analyze image: {args.get('question', '?')[:50]}"
    return tool_name


def _build_patch_mode_content(patch_text: str) -> List[Any]:
    """Parse V4A patch mode input into ACP diff blocks when possible."""
    if not patch_text:
        return [acp.tool_content(acp.text_block(""))]

    try:
        from tools.patch_parser import OperationType, parse_v4a_patch

        operations, error = parse_v4a_patch(patch_text)
        if error or not operations:
            return [acp.tool_content(acp.text_block(patch_text))]

        content: List[Any] = []
        for op in operations:
            if op.operation == OperationType.UPDATE:
                old_chunks: list[str] = []
                new_chunks: list[str] = []
                for hunk in op.hunks:
                    old_lines = [line.content for line in hunk.lines if line.prefix in (" ", "-")]
                    new_lines = [line.content for line in hunk.lines if line.prefix in (" ", "+")]
                    if old_lines or new_lines:
                        old_chunks.append("\n".join(old_lines))
                        new_chunks.append("\n".join(new_lines))

                old_text = "\n...\n".join(chunk for chunk in old_chunks if chunk)
                new_text = "\n...\n".join(chunk for chunk in new_chunks if chunk)
                if old_text or new_text:
                    content.append(
                        acp.tool_diff_content(
                            path=op.file_path,
                            old_text=old_text or None,
                            new_text=new_text or "",
                        )
                    )
                continue

            if op.operation == OperationType.ADD:
                added_lines = [line.content for hunk in op.hunks for line in hunk.lines if line.prefix == "+"]
                content.append(
                    acp.tool_diff_content(
                        path=op.file_path,
                        new_text="\n".join(added_lines),
                    )
                )
                continue

            if op.operation == OperationType.DELETE:
                content.append(
                    acp.tool_diff_content(
                        path=op.file_path,
                        old_text=f"Delete file: {op.file_path}",
                        new_text="",
                    )
                )
                continue

            if op.operation == OperationType.MOVE:
                content.append(
                    acp.tool_content(acp.text_block(f"Move file: {op.file_path} -> {op.new_path}"))
                )

        return content or [acp.tool_content(acp.text_block(patch_text))]
    except Exception:
        return [acp.tool_content(acp.text_block(patch_text))]


def _strip_diff_prefix(path: str) -> str:
    raw = str(path or "").strip()
    if raw.startswith(("a/", "b/")):
        return raw[2:]
    return raw


def _parse_unified_diff_content(diff_text: str) -> List[Any]:
    """Convert unified diff text into ACP diff content blocks."""
    if not diff_text:
        return []

    content: List[Any] = []
    current_old_path: Optional[str] = None
    current_new_path: Optional[str] = None
    old_lines: list[str] = []
    new_lines: list[str] = []

    def _flush() -> None:
        nonlocal current_old_path, current_new_path, old_lines, new_lines
        if current_old_path is None and current_new_path is None:
            return
        path = current_new_path if current_new_path and current_new_path != "/dev/null" else current_old_path
        if not path or path == "/dev/null":
            current_old_path = None
            current_new_path = None
            old_lines = []
            new_lines = []
            return
        content.append(
            acp.tool_diff_content(
                path=_strip_diff_prefix(path),
                old_text="\n".join(old_lines) if old_lines else None,
                new_text="\n".join(new_lines),
            )
        )
        current_old_path = None
        current_new_path = None
        old_lines = []
        new_lines = []

    for line in diff_text.splitlines():
        if line.startswith("--- "):
            _flush()
            current_old_path = line[4:].strip()
            continue
        if line.startswith("+++ "):
            current_new_path = line[4:].strip()
            continue
        if line.startswith("@@"):
            continue
        if current_old_path is None and current_new_path is None:
            continue
        if line.startswith("+"):
            new_lines.append(line[1:])
        elif line.startswith("-"):
            old_lines.append(line[1:])
        elif line.startswith(" "):
            shared = line[1:]
            old_lines.append(shared)
            new_lines.append(shared)

    _flush()
    return content


def _build_tool_complete_content(
    tool_name: str,
    result: Optional[str],
    *,
    function_args: Optional[Dict[str, Any]] = None,
    snapshot: Any = None,
) -> List[Any]:
    """Build structured ACP completion content, falling back to plain text."""
    display_result = result or ""
    if len(display_result) > 5000:
        display_result = display_result[:4900] + f"\n... ({len(result)} chars total, truncated)"

    if tool_name in {"write_file", "patch", "skill_manage"}:
        try:
            from agent.display import extract_edit_diff

            diff_text = extract_edit_diff(
                tool_name,
                result,
                function_args=function_args,
                snapshot=snapshot,
            )
            if isinstance(diff_text, str) and diff_text.strip():
                diff_content = _parse_unified_diff_content(diff_text)
                if diff_content:
                    return diff_content
        except Exception:
            pass

    return [acp.tool_content(acp.text_block(display_result))]


# ---------------------------------------------------------------------------
# Build ACP content objects for tool-call events
# ---------------------------------------------------------------------------


def build_tool_start(
    tool_call_id: str,
    tool_name: str,
    arguments: Dict[str, Any],
) -> ToolCallStart:
    """Create a ToolCallStart event for the given hermes tool invocation."""
    kind = get_tool_kind(tool_name)
    title = build_tool_title(tool_name, arguments)
    locations = extract_locations(arguments)

    if tool_name == "patch":
        mode = arguments.get("mode", "replace")
        if mode == "replace":
            path = arguments.get("path", "")
            old = arguments.get("old_string", "")
            new = arguments.get("new_string", "")
            content = [acp.tool_diff_content(path=path, new_text=new, old_text=old)]
        else:
            patch_text = arguments.get("patch", "")
            content = _build_patch_mode_content(patch_text)
        return acp.start_tool_call(
            tool_call_id, title, kind=kind, content=content, locations=locations,
            raw_input=arguments,
        )

    if tool_name == "write_file":
        path = arguments.get("path", "")
        file_content = arguments.get("content", "")
        content = [acp.tool_diff_content(path=path, new_text=file_content)]
        return acp.start_tool_call(
            tool_call_id, title, kind=kind, content=content, locations=locations,
            raw_input=arguments,
        )

    if tool_name == "terminal":
        command = arguments.get("command", "")
        content = [acp.tool_content(acp.text_block(f"$ {command}"))]
        return acp.start_tool_call(
            tool_call_id, title, kind=kind, content=content, locations=locations,
            raw_input=arguments,
        )

    if tool_name == "read_file":
        path = arguments.get("path", "")
        content = [acp.tool_content(acp.text_block(f"Reading {path}"))]
        return acp.start_tool_call(
            tool_call_id, title, kind=kind, content=content, locations=locations,
            raw_input=arguments,
        )

    if tool_name == "search_files":
        pattern = arguments.get("pattern", "")
        target = arguments.get("target", "content")
        content = [acp.tool_content(acp.text_block(f"Searching for '{pattern}' ({target})"))]
        return acp.start_tool_call(
            tool_call_id, title, kind=kind, content=content, locations=locations,
            raw_input=arguments,
        )

    # Generic fallback
    import json
    try:
        args_text = json.dumps(arguments, indent=2, default=str)
    except (TypeError, ValueError):
        args_text = str(arguments)
    content = [acp.tool_content(acp.text_block(args_text))]
    return acp.start_tool_call(
        tool_call_id, title, kind=kind, content=content, locations=locations,
        raw_input=arguments,
    )


def build_tool_complete(
    tool_call_id: str,
    tool_name: str,
    result: Optional[str] = None,
    function_args: Optional[Dict[str, Any]] = None,
    snapshot: Any = None,
) -> ToolCallProgress:
    """Create a ToolCallUpdate (progress) event for a completed tool call."""
    kind = get_tool_kind(tool_name)
    content = _build_tool_complete_content(
        tool_name,
        result,
        function_args=function_args,
        snapshot=snapshot,
    )
    return acp.update_tool_call(
        tool_call_id,
        kind=kind,
        status="completed",
        content=content,
        raw_output=result,
    )


# ---------------------------------------------------------------------------
# Location extraction
# ---------------------------------------------------------------------------


def extract_locations(
    arguments: Dict[str, Any],
) -> List[ToolCallLocation]:
    """Extract file-system locations from tool arguments."""
    locations: List[ToolCallLocation] = []
    path = arguments.get("path")
    if path:
        line = arguments.get("offset") or arguments.get("line")
        locations.append(ToolCallLocation(path=path, line=line))
    return locations

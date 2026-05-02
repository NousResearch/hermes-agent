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
    # Session/meta tools
    "todo": "other",
    "skill_view": "read",
    "skills_list": "read",
    "skill_manage": "edit",
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


_POLISHED_TOOLS = {
    "todo",
    "read_file",
    "search_files",
    "execute_code",
    "skill_view",
    "skills_list",
    "skill_manage",
    "terminal",
    "web_search",
    "web_extract",
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
        code = str(args.get("code") or "").strip()
        first_line = next((line.strip() for line in code.splitlines() if line.strip()), "")
        if first_line:
            if len(first_line) > 70:
                first_line = first_line[:67] + "..."
            return f"python: {first_line}"
        return "python code"
    if tool_name == "todo":
        items = args.get("todos")
        if isinstance(items, list):
            return f"todo ({len(items)} item{'s' if len(items) != 1 else ''})"
        return "todo"
    if tool_name == "skill_view":
        name = str(args.get("name") or "?").strip() or "?"
        file_path = str(args.get("file_path") or "").strip()
        suffix = f"/{file_path}" if file_path else ""
        return f"skill view ({name}{suffix})"
    if tool_name == "skills_list":
        category = str(args.get("category") or "").strip()
        return f"skills list ({category})" if category else "skills list"
    if tool_name == "skill_manage":
        action = str(args.get("action") or "manage").strip() or "manage"
        name = str(args.get("name") or "?").strip() or "?"
        file_path = str(args.get("file_path") or "").strip()
        target = f"{name}/{file_path}" if file_path else name
        if len(target) > 64:
            target = target[:61] + "..."
        return f"skill {action}: {target}"
    if tool_name == "vision_analyze":
        return f"analyze image: {args.get('question', '?')[:50]}"
    return tool_name


def _text(content: str) -> Any:
    return acp.tool_content(acp.text_block(content))


def _json_loads_maybe(value: Optional[str]) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        pass

    # Some Hermes tools append a human hint after a JSON payload, e.g.
    # ``{...}\n\n[Hint: Results truncated...]``. Keep the structured rendering path
    # by decoding the first JSON value instead of falling back to raw text.
    try:
        decoded, _ = json.JSONDecoder().raw_decode(value.lstrip())
        return decoded
    except Exception:
        return None


def _truncate_text(text: str, limit: int = 5000) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 100)] + f"\n... ({len(text)} chars total, truncated)"


def _format_todo_result(result: Optional[str]) -> Optional[str]:
    data = _json_loads_maybe(result)
    if not isinstance(data, dict) or not isinstance(data.get("todos"), list):
        return None
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    icon = {
        "completed": "✅",
        "in_progress": "🔄",
        "pending": "⏳",
        "cancelled": "✗",
    }
    lines = ["**Todo list**", ""]
    for item in data["todos"]:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "pending")
        content = str(item.get("content") or item.get("id") or "").strip()
        if content:
            lines.append(f"- {icon.get(status, '•')} {content}")
    if summary:
        cancelled = summary.get("cancelled", 0)
        lines.extend([
            "",
            "**Progress:** "
            f"{summary.get('completed', 0)} completed, "
            f"{summary.get('in_progress', 0)} in progress, "
            f"{summary.get('pending', 0)} pending"
            + (f", {cancelled} cancelled" if cancelled else ""),
        ])
    return "\n".join(lines)


def _format_read_file_result(result: Optional[str], args: Optional[Dict[str, Any]]) -> Optional[str]:
    data = _json_loads_maybe(result)
    if not isinstance(data, dict):
        return None
    if data.get("error") and not data.get("content"):
        return f"Read failed: {data.get('error')}"
    content = data.get("content")
    if not isinstance(content, str):
        return None
    path = str((args or {}).get("path") or data.get("path") or "file").strip()
    offset = (args or {}).get("offset")
    limit = (args or {}).get("limit")
    range_bits = []
    if offset:
        range_bits.append(f"from line {offset}")
    if limit:
        range_bits.append(f"limit {limit}")
    suffix = f" ({', '.join(range_bits)})" if range_bits else ""
    header = f"Read {path}{suffix}"
    if data.get("total_lines") is not None:
        header += f" — {data.get('total_lines')} total lines"
    return _truncate_text(f"{header}\n\n{content}")


def _format_search_files_result(result: Optional[str]) -> Optional[str]:
    data = _json_loads_maybe(result)
    if not isinstance(data, dict):
        return None
    matches = data.get("matches")
    if not isinstance(matches, list):
        return None

    total = data.get("total_count", len(matches))
    shown = min(len(matches), 12)
    truncated = bool(data.get("truncated")) or len(matches) > shown
    lines = [
        "Search results",
        f"Found {total} match{'es' if total != 1 else ''}; showing {shown}.",
        "",
    ]

    for match in matches[:shown]:
        if not isinstance(match, dict):
            lines.append(f"- {match}")
            continue

        path = str(match.get("path") or match.get("file") or match.get("filename") or "?")
        line = match.get("line") or match.get("line_number")
        content = str(match.get("content") or match.get("text") or "").strip()
        loc = f"{path}:{line}" if line else path
        lines.append(f"- {loc}")
        if content:
            snippet = _truncate_text(" ".join(content.split()), 300)
            lines.append(f"  {snippet}")

    if truncated:
        lines.extend([
            "",
            "Results truncated. Narrow the search, add file_glob, or use offset to page.",
        ])
    return _truncate_text("\n".join(lines), limit=7000)


def _format_execute_code_result(result: Optional[str]) -> Optional[str]:
    data = _json_loads_maybe(result)
    if not isinstance(data, dict):
        return result if isinstance(result, str) and result.strip() else None
    output = str(data.get("output") or "")
    error = str(data.get("error") or "")
    exit_code = data.get("exit_code")
    parts = [f"Exit code: {exit_code}" if exit_code is not None else "Execution complete"]
    if output:
        parts.extend(["", "Output:", output])
    if error:
        parts.extend(["", "Error:", error])
    return _truncate_text("\n".join(parts))


def _extract_markdown_headings(content: str, limit: int = 8) -> list[str]:
    headings: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            if heading:
                headings.append(heading)
        if len(headings) >= limit:
            break
    return headings


def _format_skill_view_result(result: Optional[str]) -> Optional[str]:
    data = _json_loads_maybe(result)
    if not isinstance(data, dict):
        return None
    if data.get("success") is False:
        return f"Skill view failed: {data.get('error', 'unknown error')}"
    name = str(data.get("name") or "skill")
    file_path = str(data.get("file") or data.get("path") or "SKILL.md")
    description = str(data.get("description") or "").strip()
    content = str(data.get("content") or "")
    linked = data.get("linked_files") if isinstance(data.get("linked_files"), dict) else None

    lines = ["**Skill loaded**", "", f"- **Name:** `{name}`", f"- **File:** `{file_path}`"]
    if description:
        lines.append(f"- **Description:** {description}")
    if content:
        lines.append(f"- **Content:** {len(content):,} chars loaded into agent context")
    if linked:
        linked_count = sum(len(v) for v in linked.values() if isinstance(v, list))
        lines.append(f"- **Linked files:** {linked_count}")

    headings = _extract_markdown_headings(content)
    if headings:
        lines.extend(["", "**Sections**"])
        lines.extend(f"- {heading}" for heading in headings)

    lines.extend([
        "",
        "_Full skill content is available to the agent but hidden here to keep ACP readable._",
    ])
    return "\n".join(lines)


def _format_skill_manage_result(result: Optional[str], args: Optional[Dict[str, Any]]) -> Optional[str]:
    data = _json_loads_maybe(result)
    if not isinstance(data, dict):
        return None

    action = str((args or {}).get("action") or "manage").strip() or "manage"
    name = str((args or {}).get("name") or data.get("name") or "skill").strip() or "skill"
    file_path = str((args or {}).get("file_path") or data.get("file_path") or "SKILL.md").strip() or "SKILL.md"
    success = data.get("success")
    status = "✅ Skill updated" if success is not False else "✗ Skill update failed"

    lines = [f"**{status}**", "", f"- **Action:** `{action}`", f"- **Skill:** `{name}`"]
    if action not in {"delete"}:
        lines.append(f"- **File:** `{file_path}`")

    message = str(data.get("message") or data.get("error") or "").strip()
    if message:
        lines.append(f"- **Result:** {message}")

    replacements = data.get("replacements") or data.get("replacement_count")
    if replacements is not None:
        lines.append(f"- **Replacements:** {replacements}")

    path = str(data.get("path") or "").strip()
    if path:
        lines.append(f"- **Path:** `{path}`")

    return "\n".join(lines)


def _format_web_search_result(result: Optional[str]) -> Optional[str]:
    data = _json_loads_maybe(result)
    if not isinstance(data, dict):
        return None
    web = data.get("data", {}).get("web") if isinstance(data.get("data"), dict) else data.get("web")
    if not isinstance(web, list):
        return None
    lines = [f"Web results: {len(web)}"]
    for item in web[:10]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("url") or "result").strip()
        url = str(item.get("url") or "").strip()
        desc = str(item.get("description") or "").strip()
        lines.append(f"• {title}" + (f" — {url}" if url else ""))
        if desc:
            lines.append(f"  {desc}")
    return _truncate_text("\n".join(lines))


def _build_polished_completion_content(
    tool_name: str,
    result: Optional[str],
    function_args: Optional[Dict[str, Any]],
) -> Optional[List[Any]]:
    formatter = {
        "todo": lambda: _format_todo_result(result),
        "read_file": lambda: _format_read_file_result(result, function_args),
        "search_files": lambda: _format_search_files_result(result),
        "execute_code": lambda: _format_execute_code_result(result),
        "skill_view": lambda: _format_skill_view_result(result),
        "skill_manage": lambda: _format_skill_manage_result(result, function_args),
        "web_search": lambda: _format_web_search_result(result),
    }.get(tool_name)
    if formatter is None:
        return None
    text = formatter()
    if not text:
        return None
    return [_text(text)]


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

    polished_content = _build_polished_completion_content(tool_name, result, function_args)
    if polished_content:
        return polished_content

    return [_text(display_result)]


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
        content = [_text(f"$ {command}")]
        return acp.start_tool_call(
            tool_call_id, title, kind=kind, content=content, locations=locations,
        )

    if tool_name == "read_file":
        path = arguments.get("path", "")
        offset = arguments.get("offset")
        limit = arguments.get("limit")
        bits = []
        if offset:
            bits.append(f"from line {offset}")
        if limit:
            bits.append(f"limit {limit}")
        suffix = f" ({', '.join(bits)})" if bits else ""
        content = [_text(f"Reading {path}{suffix}")]
        return acp.start_tool_call(
            tool_call_id, title, kind=kind, content=content, locations=locations,
        )

    if tool_name == "search_files":
        pattern = arguments.get("pattern", "")
        target = arguments.get("target", "content")
        search_path = arguments.get("path")
        where = f" in {search_path}" if search_path else ""
        content = [_text(f"Searching for '{pattern}' ({target}){where}")]
        return acp.start_tool_call(
            tool_call_id, title, kind=kind, content=content, locations=locations,
        )

    if tool_name == "todo":
        items = arguments.get("todos")
        if isinstance(items, list):
            preview_lines = ["Updating todo list", ""]
            for item in items[:8]:
                if isinstance(item, dict):
                    preview_lines.append(f"- {item.get('status', 'pending')}: {item.get('content', item.get('id', ''))}")
            if len(items) > 8:
                preview_lines.append(f"... {len(items) - 8} more")
            content = [_text("\n".join(preview_lines))]
        else:
            content = [_text("Reading todo list")]
        return acp.start_tool_call(
            tool_call_id, title, kind=kind, content=content, locations=locations,
        )

    if tool_name == "skill_view":
        name = str(arguments.get("name") or "?").strip() or "?"
        file_path = str(arguments.get("file_path") or "SKILL.md").strip() or "SKILL.md"
        content = [_text(f"Loading skill '{name}' ({file_path})")]
        return acp.start_tool_call(
            tool_call_id, title, kind=kind, content=content, locations=locations,
        )

    if tool_name == "skill_manage":
        action = str(arguments.get("action") or "manage").strip() or "manage"
        name = str(arguments.get("name") or "?").strip() or "?"
        file_path = str(arguments.get("file_path") or "SKILL.md").strip() or "SKILL.md"
        path = f"skills/{name}/{file_path}" if file_path else f"skills/{name}"

        if action == "patch":
            old = str(arguments.get("old_string") or "")
            new = str(arguments.get("new_string") or "")
            content = [acp.tool_diff_content(path=path, old_text=old or None, new_text=new)]
        elif action in {"edit", "create"}:
            content = [
                acp.tool_diff_content(
                    path=path,
                    new_text=str(arguments.get("content") or ""),
                )
            ]
        elif action == "write_file":
            target = str(arguments.get("file_path") or "file")
            content = [
                acp.tool_diff_content(
                    path=f"skills/{name}/{target}",
                    new_text=str(arguments.get("file_content") or ""),
                )
            ]
        elif action in {"delete", "remove_file"}:
            target = str(arguments.get("file_path") or file_path or name)
            content = [_text(f"Removing {target} from skill '{name}'")]
        else:
            content = [_text(f"Running skill_manage action '{action}' on skill '{name}' ({file_path})")]

        return acp.start_tool_call(
            tool_call_id, title, kind=kind, content=content, locations=locations,
        )

    if tool_name == "execute_code":
        code = str(arguments.get("code") or "").strip()
        preview = code[:1200] + (f"\n... ({len(code)} chars total, truncated)" if len(code) > 1200 else "")
        content = [_text(f"Running Python helper script:\n\n```python\n{preview}\n```" if preview else "Running Python helper script")]
        return acp.start_tool_call(
            tool_call_id, title, kind=kind, content=content, locations=locations,
        )

    if tool_name == "web_search":
        query = str(arguments.get("query") or "").strip()
        content = [_text(f"Searching the web for: {query}" if query else "Searching the web")]
        return acp.start_tool_call(
            tool_call_id, title, kind=kind, content=content, locations=locations,
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
        raw_output=None if tool_name in _POLISHED_TOOLS else result,
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

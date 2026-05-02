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
    # Core operator loop
    "todo", "memory", "session_search", "delegate_task",
    # Files / execution
    "read_file", "write_file", "patch", "search_files", "terminal", "process", "execute_code",
    # Skills / web / browser / media
    "skill_view", "skills_list", "skill_manage", "web_search", "web_extract",
    "browser_navigate", "browser_click", "browser_type", "browser_press", "browser_scroll",
    "browser_back", "browser_snapshot", "browser_console", "browser_get_images", "browser_vision",
    "vision_analyze", "image_generate", "text_to_speech",
    # Schedulers / platform integrations
    "cronjob", "send_message", "clarify", "discord", "discord_admin",
    "ha_list_entities", "ha_get_state", "ha_list_services", "ha_call_service",
    "feishu_doc_read", "feishu_drive_list_comments", "feishu_drive_list_comment_replies",
    "feishu_drive_reply_comment", "feishu_drive_add_comment",
    "kanban_create", "kanban_show", "kanban_comment", "kanban_complete",
    "kanban_block", "kanban_link", "kanban_heartbeat",
    "yb_query_group_info", "yb_query_group_members", "yb_search_sticker",
    "yb_send_dm", "yb_send_sticker", "mixture_of_agents",
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
    if tool_name == "process":
        action = str(args.get("action") or "").strip() or "manage"
        sid = str(args.get("session_id") or "").strip()
        return f"process {action}: {sid}" if sid else f"process {action}"
    if tool_name == "delegate_task":
        tasks = args.get("tasks")
        if isinstance(tasks, list) and tasks:
            return f"delegate batch ({len(tasks)} tasks)"
        goal = args.get("goal", "")
        if goal and len(goal) > 60:
            goal = goal[:57] + "..."
        return f"delegate: {goal}" if goal else "delegate task"
    if tool_name == "session_search":
        query = str(args.get("query") or "").strip()
        return f"session search: {query}" if query else "recent sessions"
    if tool_name == "memory":
        action = str(args.get("action") or "manage").strip() or "manage"
        target = str(args.get("target") or "memory").strip() or "memory"
        return f"memory {action}: {target}"
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
    if tool_name == "browser_navigate":
        return f"navigate: {args.get('url', '?')}"
    if tool_name == "browser_snapshot":
        return "browser snapshot"
    if tool_name == "browser_vision":
        return f"browser vision: {str(args.get('question', '?'))[:50]}"
    if tool_name == "browser_get_images":
        return "browser images"
    if tool_name == "vision_analyze":
        return f"analyze image: {str(args.get('question', '?'))[:50]}"
    if tool_name == "image_generate":
        prompt = str(args.get("prompt") or args.get("description") or "").strip()
        return f"generate image: {prompt[:50]}" if prompt else "generate image"
    if tool_name == "cronjob":
        action = str(args.get("action") or "manage").strip() or "manage"
        job_id = str(args.get("job_id") or args.get("id") or "").strip()
        return f"cron {action}: {job_id}" if job_id else f"cron {action}"
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


def _fenced_text(text: str, language: str = "") -> str:
    """Return a Markdown fence that cannot be broken by backticks in text."""
    longest = max((len(run) for run in text.split("`")[1::2]), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{language}\n{text}\n{fence}"


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
    # Hermes read_file output is line-numbered with `|`. If we send it as raw
    # Markdown, Zed can interpret pipes as tables and collapse the layout.
    # Fence the payload so file lines stay readable and literal.
    return _truncate_text(f"{header}\n\n{_fenced_text(content)}")


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


def _format_web_extract_result(result: Optional[str]) -> Optional[str]:
    data = _json_loads_maybe(result)
    if not isinstance(data, dict):
        return None
    if data.get("success") is False and data.get("error"):
        return f"Web extract failed: {data.get('error')}"
    results = data.get("results")
    if not isinstance(results, list):
        return None
    lines = [f"Web extract: {len(results)} URL{'s' if len(results) != 1 else ''}"]
    for item in results[:10]:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or url or "Untitled").strip()
        error = str(item.get("error") or "").strip()
        status = "failed" if error and error not in {"None", "null"} else "extracted"
        suffix = f" — {status}"
        lines.append(f"- {title}" + (f" — {url}" if url and url != title else "") + suffix)
        if status == "failed":
            lines.append(f"  Error: {_truncate_text(error, limit=500)}")
    if len(results) > 10:
        lines.append(f"... {len(results) - 10} more result(s) omitted")
    return "\n".join(lines)


def _format_process_result(result: Optional[str], args: Optional[Dict[str, Any]]) -> Optional[str]:
    data = _json_loads_maybe(result)
    if not isinstance(data, dict):
        return result if isinstance(result, str) and result.strip() else None
    if data.get("success") is False and data.get("error"):
        return f"Process error: {data.get('error')}"
    action = str((args or {}).get("action") or "process").strip() or "process"
    if isinstance(data.get("processes"), list):
        processes = data["processes"]
        lines = [f"Processes: {len(processes)}"]
        for proc in processes[:20]:
            if not isinstance(proc, dict):
                lines.append(f"- {proc}")
                continue
            sid = str(proc.get("session_id") or proc.get("id") or "?")
            status = str(proc.get("status") or ("exited" if proc.get("exited") else "running"))
            cmd = str(proc.get("command") or "").strip()
            pid = proc.get("pid")
            code = proc.get("exit_code")
            bits = [status]
            if pid is not None:
                bits.append(f"pid {pid}")
            if code is not None:
                bits.append(f"exit {code}")
            lines.append(f"- `{sid}` — {', '.join(bits)}" + (f" — {cmd[:120]}" if cmd else ""))
        if len(processes) > 20:
            lines.append(f"... {len(processes) - 20} more process(es)")
        return "\n".join(lines)

    status = str(data.get("status") or data.get("state") or action).strip()
    sid = str(data.get("session_id") or (args or {}).get("session_id") or "").strip()
    lines = [f"Process {action}: {status}" + (f" (`{sid}`)" if sid else "")]
    for key, label in (("command", "Command"), ("pid", "PID"), ("exit_code", "Exit code"), ("returncode", "Exit code"), ("lines", "Lines")):
        if data.get(key) is not None:
            lines.append(f"- **{label}:** {data.get(key)}")
    output = data.get("output") or data.get("new_output") or data.get("log") or data.get("stdout")
    error = data.get("error") or data.get("stderr")
    if output:
        lines.extend(["", "Output:", _truncate_text(str(output), limit=5000)])
    if error:
        lines.extend(["", "Error:", _truncate_text(str(error), limit=2000)])
    msg = data.get("message")
    if msg and not output and not error:
        lines.append(str(msg))
    return _truncate_text("\n".join(lines), limit=7000)


def _format_delegate_result(result: Optional[str]) -> Optional[str]:
    data = _json_loads_maybe(result)
    if not isinstance(data, dict):
        return None
    if data.get("error") and not isinstance(data.get("results"), list):
        return f"Delegation failed: {data.get('error')}"
    results = data.get("results")
    if not isinstance(results, list):
        return None
    total = data.get("total_duration_seconds")
    lines = [f"Delegation results: {len(results)} task{'s' if len(results) != 1 else ''}" + (f" in {total}s" if total is not None else "")]
    icon = {"completed": "✅", "failed": "✗", "error": "✗", "timeout": "⏱", "interrupted": "⚠"}
    for item in results:
        if not isinstance(item, dict):
            lines.append(f"- {item}")
            continue
        idx = item.get("task_index")
        status = str(item.get("status") or "unknown")
        model = item.get("model")
        dur = item.get("duration_seconds")
        role = item.get("_child_role")
        header = f"{icon.get(status, '•')} Task {idx + 1 if isinstance(idx, int) else '?'}: {status}"
        bits = []
        if model:
            bits.append(str(model))
        if role:
            bits.append(f"role={role}")
        if dur is not None:
            bits.append(f"{dur}s")
        if bits:
            header += " (" + ", ".join(bits) + ")"
        lines.extend(["", header])
        summary = str(item.get("summary") or "").strip()
        error = str(item.get("error") or "").strip()
        if summary:
            lines.append(_truncate_text(summary, limit=1200))
        if error:
            lines.append("Error: " + _truncate_text(error, limit=800))
        trace = item.get("tool_trace")
        if isinstance(trace, list) and trace:
            names = [str(t.get("tool") or "?") for t in trace if isinstance(t, dict)]
            if names:
                lines.append("Tools: " + ", ".join(names[:12]) + (f" (+{len(names)-12})" if len(names) > 12 else ""))
    return _truncate_text("\n".join(lines), limit=8000)


def _format_session_search_result(result: Optional[str]) -> Optional[str]:
    data = _json_loads_maybe(result)
    if not isinstance(data, dict):
        return None
    if data.get("success") is False:
        return f"Session search failed: {data.get('error', 'unknown error')}"
    results = data.get("results")
    if not isinstance(results, list):
        return None
    mode = data.get("mode") or "search"
    query = data.get("query")
    lines = ["Recent sessions" if mode == "recent" else f"Session search results" + (f" for `{query}`" if query else "")]
    if not results:
        lines.append(str(data.get("message") or "No matching sessions found."))
        return "\n".join(lines)
    for item in results:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("session_id") or "?")
        title = str(item.get("title") or item.get("when") or "Untitled session").strip()
        when = str(item.get("last_active") or item.get("started_at") or item.get("when") or "").strip()
        count = item.get("message_count")
        source = str(item.get("source") or "").strip()
        meta = ", ".join(str(x) for x in [when, source, f"{count} msgs" if count is not None else ""] if x)
        lines.append(f"- **{title}** (`{sid}`)" + (f" — {meta}" if meta else ""))
        summary = str(item.get("summary") or item.get("preview") or "").strip()
        if summary:
            lines.append("  " + _truncate_text(" ".join(summary.split()), limit=500))
    return _truncate_text("\n".join(lines), limit=7000)


def _format_memory_result(result: Optional[str], args: Optional[Dict[str, Any]]) -> Optional[str]:
    data = _json_loads_maybe(result)
    if not isinstance(data, dict):
        return None
    action = str((args or {}).get("action") or "memory").strip() or "memory"
    target = str(data.get("target") or (args or {}).get("target") or "memory")
    if data.get("success") is False:
        lines = [f"✗ Memory {action} failed ({target})", str(data.get("error") or "unknown error")]
        matches = data.get("matches")
        if isinstance(matches, list) and matches:
            lines.append("Matches:")
            lines.extend(f"- {_truncate_text(str(m), 160)}" for m in matches[:5])
        return "\n".join(lines)
    lines = [f"✅ Memory {action} saved ({target})"]
    if data.get("message"):
        lines.append(str(data.get("message")))
    if data.get("entry_count") is not None:
        lines.append(f"Entries: {data.get('entry_count')}")
    if data.get("usage"):
        lines.append(f"Usage: {data.get('usage')}")
    # Avoid dumping all memory entries into ACP UI; show only the explicit new value preview.
    preview = str((args or {}).get("content") or (args or {}).get("old_text") or "").strip()
    if preview:
        lines.append("Preview: " + _truncate_text(preview, limit=300))
    return "\n".join(lines)


def _format_edit_result(tool_name: str, result: Optional[str], args: Optional[Dict[str, Any]]) -> Optional[str]:
    data = _json_loads_maybe(result)
    path = str((args or {}).get("path") or "file").strip()
    if isinstance(data, dict):
        if data.get("success") is False or data.get("error"):
            return f"{tool_name} failed for {path}: {data.get('error', 'unknown error')}"
        message = str(data.get("message") or "").strip()
        replacements = data.get("replacements") or data.get("replacement_count")
        lines = [f"✅ {tool_name} completed" + (f" for `{path}`" if path else "")]
        if message:
            lines.append(message)
        if replacements is not None:
            lines.append(f"Replacements: {replacements}")
        if data.get("files_modified"):
            files = data.get("files_modified")
            if isinstance(files, list):
                lines.append("Files: " + ", ".join(f"`{f}`" for f in files[:8]))
        return "\n".join(lines)
    if isinstance(result, str) and result.strip():
        return _truncate_text(result, limit=3000)
    return f"✅ {tool_name} completed" + (f" for `{path}`" if path else "")


def _format_browser_result(tool_name: str, result: Optional[str], args: Optional[Dict[str, Any]]) -> Optional[str]:
    data = _json_loads_maybe(result)
    if not isinstance(data, dict):
        return result if isinstance(result, str) and result.strip() else None
    if data.get("success") is False or data.get("error"):
        return f"{tool_name} failed: {data.get('error', 'unknown error')}"
    if tool_name == "browser_get_images":
        images = data.get("images") or data.get("data")
        if isinstance(images, list):
            lines = [f"Images found: {len(images)}"]
            for img in images[:12]:
                if isinstance(img, dict):
                    alt = str(img.get("alt") or "").strip()
                    url = str(img.get("url") or img.get("src") or "").strip()
                    lines.append(f"- {alt or 'image'}" + (f" — {url}" if url else ""))
            return _truncate_text("\n".join(lines), limit=5000)
    title = str(data.get("title") or data.get("url") or data.get("status") or tool_name)
    text = str(data.get("text") or data.get("content") or data.get("snapshot") or data.get("analysis") or data.get("message") or "").strip()
    lines = [title]
    if data.get("url") and data.get("url") != title:
        lines.append(str(data.get("url")))
    if text:
        lines.extend(["", _truncate_text(text, limit=5000)])
    return _truncate_text("\n".join(lines), limit=7000)


def _format_media_or_cron_result(tool_name: str, result: Optional[str]) -> Optional[str]:
    data = _json_loads_maybe(result)
    if not isinstance(data, dict):
        return result if isinstance(result, str) and result.strip() else None
    if data.get("success") is False or data.get("error"):
        return f"{tool_name} failed: {data.get('error', 'unknown error')}"
    lines = [f"✅ {tool_name} completed"]
    for key in ("file_path", "path", "url", "image_url", "job_id", "id", "status", "message", "next_run"):
        if data.get(key):
            lines.append(f"- **{key}:** {data.get(key)}")
    return "\n".join(lines)


def _format_generic_structured_result(tool_name: str, result: Optional[str]) -> Optional[str]:
    data = _json_loads_maybe(result)
    if not isinstance(data, (dict, list)):
        return result if isinstance(result, str) and result.strip() else None
    if isinstance(data, list):
        lines = [f"{tool_name}: {len(data)} item{'s' if len(data) != 1 else ''}"]
        for item in data[:12]:
            lines.append(f"- {_truncate_text(str(item), limit=240)}")
        return _truncate_text("\n".join(lines), limit=5000)

    if data.get("success") is False or data.get("error"):
        return f"{tool_name} failed: {data.get('error', 'unknown error')}"

    lines = [f"✅ {tool_name} completed" if data.get("success") is True else f"{tool_name} result"]
    priority_keys = (
        "message", "status", "id", "task_id", "issue_id", "title", "name", "entity_id",
        "state", "service", "url", "path", "file_path", "count", "total", "next_run",
    )
    seen = set()
    for key in priority_keys:
        value = data.get(key)
        if value in (None, "", [], {}):
            continue
        seen.add(key)
        lines.append(f"- **{key}:** {_truncate_text(str(value), limit=500)}")

    for key, value in data.items():
        if key in seen or key in {"success", "raw", "content", "entries"}:
            continue
        if value in (None, "", [], {}):
            continue
        if isinstance(value, (dict, list)):
            preview = json.dumps(value, ensure_ascii=False, default=str)
        else:
            preview = str(value)
        lines.append(f"- **{key}:** {_truncate_text(preview, limit=500)}")
        if len(lines) >= 14:
            break

    content = data.get("content")
    if isinstance(content, str) and content.strip():
        lines.extend(["", _truncate_text(content.strip(), limit=1500)])
    return _truncate_text("\n".join(lines), limit=7000)


def _build_polished_completion_content(
    tool_name: str,
    result: Optional[str],
    function_args: Optional[Dict[str, Any]],
) -> Optional[List[Any]]:
    formatter = {
        "todo": lambda: _format_todo_result(result),
        "read_file": lambda: _format_read_file_result(result, function_args),
        "write_file": lambda: _format_edit_result(tool_name, result, function_args),
        "patch": lambda: _format_edit_result(tool_name, result, function_args),
        "search_files": lambda: _format_search_files_result(result),
        "execute_code": lambda: _format_execute_code_result(result),
        "process": lambda: _format_process_result(result, function_args),
        "delegate_task": lambda: _format_delegate_result(result),
        "session_search": lambda: _format_session_search_result(result),
        "memory": lambda: _format_memory_result(result, function_args),
        "skill_view": lambda: _format_skill_view_result(result),
        "skill_manage": lambda: _format_skill_manage_result(result, function_args),
        "web_search": lambda: _format_web_search_result(result),
        "web_extract": lambda: _format_web_extract_result(result),
        "browser_navigate": lambda: _format_browser_result(tool_name, result, function_args),
        "browser_snapshot": lambda: _format_browser_result(tool_name, result, function_args),
        "browser_vision": lambda: _format_browser_result(tool_name, result, function_args),
        "browser_get_images": lambda: _format_browser_result(tool_name, result, function_args),
        "vision_analyze": lambda: _format_media_or_cron_result(tool_name, result),
        "image_generate": lambda: _format_media_or_cron_result(tool_name, result),
        "cronjob": lambda: _format_media_or_cron_result(tool_name, result),
    }.get(tool_name)
    if formatter is None and tool_name in _POLISHED_TOOLS:
        formatter = lambda: _format_generic_structured_result(tool_name, result)
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
        )

    if tool_name == "write_file":
        path = arguments.get("path", "")
        file_content = arguments.get("content", "")
        content = [acp.tool_diff_content(path=path, new_text=file_content)]
        return acp.start_tool_call(
            tool_call_id, title, kind=kind, content=content, locations=locations,
        )

    if tool_name == "terminal":
        command = arguments.get("command", "")
        content = [_text(f"$ {command}")]
        return acp.start_tool_call(
            tool_call_id, title, kind=kind, content=content, locations=locations,
        )

    if tool_name == "read_file":
        # The title and location already identify the file. Sending a synthetic
        # "Reading ..." content block makes Zed render an unhelpful Output
        # section before the real file contents arrive on completion.
        return acp.start_tool_call(
            tool_call_id, title, kind=kind, content=None, locations=locations,
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

    if tool_name == "web_extract":
        # The title identifies the URL(s). Avoid a duplicate content block so
        # Zed renders this like read_file: compact start, concise completion.
        return acp.start_tool_call(
            tool_call_id, title, kind=kind, content=None, locations=locations,
        )

    if tool_name == "process":
        action = str(arguments.get("action") or "").strip() or "manage"
        sid = str(arguments.get("session_id") or "").strip()
        data_preview = str(arguments.get("data") or "").strip()
        text = f"Process action: {action}" + (f"\nSession: {sid}" if sid else "")
        if data_preview:
            text += "\nInput: " + _truncate_text(data_preview, limit=500)
        content = [_text(text)]
        return acp.start_tool_call(
            tool_call_id, title, kind=kind, content=content, locations=locations,
        )

    if tool_name == "delegate_task":
        tasks = arguments.get("tasks")
        if isinstance(tasks, list) and tasks:
            lines = [f"Delegating {len(tasks)} tasks", ""]
            for i, task in enumerate(tasks[:8], 1):
                if isinstance(task, dict):
                    goal = str(task.get("goal") or "").strip()
                    role = str(task.get("role") or "").strip()
                    lines.append(f"{i}. " + _truncate_text(goal, limit=160) + (f" ({role})" if role else ""))
            if len(tasks) > 8:
                lines.append(f"... {len(tasks) - 8} more")
            content = [_text("\n".join(lines))]
        else:
            goal = str(arguments.get("goal") or "").strip()
            content = [_text("Delegating task" + (f":\n{_truncate_text(goal, limit=800)}" if goal else ""))]
        return acp.start_tool_call(
            tool_call_id, title, kind=kind, content=content, locations=locations,
        )

    if tool_name == "session_search":
        query = str(arguments.get("query") or "").strip()
        content = [_text(f"Searching past sessions for: {query}" if query else "Loading recent sessions")]
        return acp.start_tool_call(
            tool_call_id, title, kind=kind, content=content, locations=locations,
        )

    if tool_name == "memory":
        action = str(arguments.get("action") or "manage").strip() or "manage"
        target = str(arguments.get("target") or "memory").strip() or "memory"
        preview = str(arguments.get("content") or arguments.get("old_text") or "").strip()
        text = f"Memory {action} ({target})"
        if preview:
            text += "\nPreview: " + _truncate_text(preview, limit=500)
        content = [_text(text)]
        return acp.start_tool_call(
            tool_call_id, title, kind=kind, content=content, locations=locations,
        )

    if tool_name in _POLISHED_TOOLS:
        try:
            args_text = json.dumps(arguments, indent=2, default=str)
        except (TypeError, ValueError):
            args_text = str(arguments)
        content = [_text(_truncate_text(args_text, limit=1200))]
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
        raw_input=None if tool_name in _POLISHED_TOOLS else arguments,
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

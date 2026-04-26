"""Tool-use display renderer for Feishu streaming cards.

Converts tool_use blocks from Claude streaming into Feishu card block JSON,
with per-tool icons, titles, and status-aware styling.
"""

import re
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Tool descriptor registry
# ---------------------------------------------------------------------------

# Each entry: icon (emoji or feishu icon token), title, summary_template
# summary_template placeholders: {tool_name}, {args_summary}, {result_summary}
TOOL_DESCRIPTORS: dict[str, dict[str, str]] = {
    # ---- Core file / code tools ----
    "read": {
        "icon": "📄",
        "title": "Read",
        "summary_template": "Reading {args_summary}",
    },
    "open": {
        "icon": "📄",
        "title": "Read",
        "summary_template": "Opening {args_summary}",
    },
    "write": {
        "icon": "✏️",
        "title": "Write",
        "summary_template": "Writing {args_summary}",
    },
    "edit": {
        "icon": "✏️",
        "title": "Edit",
        "summary_template": "Editing {args_summary}",
    },
    "bash": {
        "icon": "⚙️",
        "title": "Run command",
        "summary_template": "Running {args_summary}",
    },
    "exec": {
        "icon": "⚙️",
        "title": "Run command",
        "summary_template": "Executing {args_summary}",
    },
    "run": {
        "icon": "⚙️",
        "title": "Run command",
        "summary_template": "Running {args_summary}",
    },
    "command": {
        "icon": "⚙️",
        "title": "Run command",
        "summary_template": "Running {args_summary}",
    },
    # ---- Search tools ----
    "search": {
        "icon": "🔍",
        "title": "Search",
        "summary_template": "Searching for {args_summary}",
    },
    "web_search": {
        "icon": "🔍",
        "title": "Search web",
        "summary_template": "Searching web for {args_summary}",
    },
    "web_fetch": {
        "icon": "🌐",
        "title": "Fetch page",
        "summary_template": "Fetching {args_summary}",
    },
    "fetch": {
        "icon": "🌐",
        "title": "Fetch page",
        "summary_template": "Fetching {args_summary}",
    },
    "grep": {
        "icon": "🔎",
        "title": "Search text",
        "summary_template": "Searching text {args_summary}",
    },
    "glob": {
        "icon": "📁",
        "title": "Search files",
        "summary_template": "Finding files {args_summary}",
    },
    # ---- Agent tools ----
    "agent": {
        "icon": "🤖",
        "title": "Run sub-agent",
        "summary_template": "Running sub-agent: {args_summary}",
    },
    "task": {
        "icon": "🤖",
        "title": "Run sub-agent",
        "summary_template": "Spawning task: {args_summary}",
    },
    "spawn": {
        "icon": "🤖",
        "title": "Spawn agent",
        "summary_template": "Spawning agent: {args_summary}",
    },
    # ---- Feishu Calendar tools ----
    "feishu_calendar_list_events": {
        "icon": "📅",
        "title": "List calendar events",
        "summary_template": "Listing calendar events {args_summary}",
    },
    "feishu_calendar_get_event": {
        "icon": "📅",
        "title": "Get calendar event",
        "summary_template": "Getting event {args_summary}",
    },
    "feishu_calendar_create_event": {
        "icon": "📅",
        "title": "Create calendar event",
        "summary_template": "Creating event {args_summary}",
    },
    "feishu_calendar_update_event": {
        "icon": "📅",
        "title": "Update calendar event",
        "summary_template": "Updating event {args_summary}",
    },
    "feishu_calendar_delete_event": {
        "icon": "📅",
        "title": "Delete calendar event",
        "summary_template": "Deleting event {args_summary}",
    },
    "feishu_calendar_list_calendars": {
        "icon": "📅",
        "title": "List calendars",
        "summary_template": "Listing calendars {args_summary}",
    },
    "feishu_calendar_get_freebusy": {
        "icon": "📅",
        "title": "Get free/busy",
        "summary_template": "Checking availability {args_summary}",
    },
    # ---- Feishu Bitable tools ----
    "feishu_bitable_list_records": {
        "icon": "📊",
        "title": "List bitable records",
        "summary_template": "Listing records {args_summary}",
    },
    "feishu_bitable_get_record": {
        "icon": "📊",
        "title": "Get bitable record",
        "summary_template": "Getting record {args_summary}",
    },
    "feishu_bitable_create_record": {
        "icon": "📊",
        "title": "Create bitable record",
        "summary_template": "Creating record {args_summary}",
    },
    "feishu_bitable_update_record": {
        "icon": "📊",
        "title": "Update bitable record",
        "summary_template": "Updating record {args_summary}",
    },
    "feishu_bitable_delete_record": {
        "icon": "📊",
        "title": "Delete bitable record",
        "summary_template": "Deleting record {args_summary}",
    },
    "feishu_bitable_batch_create_records": {
        "icon": "📊",
        "title": "Batch create records",
        "summary_template": "Batch creating records {args_summary}",
    },
    "feishu_bitable_batch_update_records": {
        "icon": "📊",
        "title": "Batch update records",
        "summary_template": "Batch updating records {args_summary}",
    },
    "feishu_bitable_search_records": {
        "icon": "📊",
        "title": "Search bitable records",
        "summary_template": "Searching records {args_summary}",
    },
    "feishu_bitable_list_tables": {
        "icon": "📊",
        "title": "List bitable tables",
        "summary_template": "Listing tables {args_summary}",
    },
    "feishu_bitable_list_fields": {
        "icon": "📊",
        "title": "List bitable fields",
        "summary_template": "Listing fields {args_summary}",
    },
    # ---- Feishu Drive tools ----
    "feishu_drive_list_comments": {
        "icon": "💬",
        "title": "List drive comments",
        "summary_template": "Listing comments {args_summary}",
    },
    "feishu_drive_list_comment_replies": {
        "icon": "💬",
        "title": "List comment replies",
        "summary_template": "Listing replies {args_summary}",
    },
    "feishu_drive_reply_comment": {
        "icon": "💬",
        "title": "Reply to comment",
        "summary_template": "Replying to comment {args_summary}",
    },
    "feishu_drive_add_comment": {
        "icon": "💬",
        "title": "Add drive comment",
        "summary_template": "Adding comment {args_summary}",
    },
    "feishu_drive_list_files": {
        "icon": "📁",
        "title": "List drive files",
        "summary_template": "Listing files {args_summary}",
    },
    "feishu_drive_get_file": {
        "icon": "📁",
        "title": "Get drive file",
        "summary_template": "Getting file {args_summary}",
    },
    "feishu_drive_upload_file": {
        "icon": "📁",
        "title": "Upload file",
        "summary_template": "Uploading file {args_summary}",
    },
    "feishu_drive_download_file": {
        "icon": "📁",
        "title": "Download file",
        "summary_template": "Downloading file {args_summary}",
    },
    # ---- Feishu Docx tools ----
    "feishu_docx_get_content": {
        "icon": "📝",
        "title": "Get doc content",
        "summary_template": "Reading document {args_summary}",
    },
    "feishu_docx_create": {
        "icon": "📝",
        "title": "Create document",
        "summary_template": "Creating document {args_summary}",
    },
    "feishu_docx_update": {
        "icon": "📝",
        "title": "Update document",
        "summary_template": "Updating document {args_summary}",
    },
    "feishu_docx_get_blocks": {
        "icon": "📝",
        "title": "Get doc blocks",
        "summary_template": "Getting blocks {args_summary}",
    },
    "feishu_docx_create_block": {
        "icon": "📝",
        "title": "Create doc block",
        "summary_template": "Creating block {args_summary}",
    },
    "feishu_docx_update_block": {
        "icon": "📝",
        "title": "Update doc block",
        "summary_template": "Updating block {args_summary}",
    },
    "feishu_docx_delete_block": {
        "icon": "📝",
        "title": "Delete doc block",
        "summary_template": "Deleting block {args_summary}",
    },
    # ---- Feishu Wiki tools ----
    "feishu_wiki_list_spaces": {
        "icon": "📚",
        "title": "List wiki spaces",
        "summary_template": "Listing wiki spaces {args_summary}",
    },
    "feishu_wiki_get_node": {
        "icon": "📚",
        "title": "Get wiki node",
        "summary_template": "Getting wiki node {args_summary}",
    },
    "feishu_wiki_list_nodes": {
        "icon": "📚",
        "title": "List wiki nodes",
        "summary_template": "Listing wiki nodes {args_summary}",
    },
    "feishu_wiki_create_node": {
        "icon": "📚",
        "title": "Create wiki node",
        "summary_template": "Creating wiki node {args_summary}",
    },
    "feishu_wiki_move_node": {
        "icon": "📚",
        "title": "Move wiki node",
        "summary_template": "Moving wiki node {args_summary}",
    },
    "feishu_wiki_search": {
        "icon": "📚",
        "title": "Search wiki",
        "summary_template": "Searching wiki {args_summary}",
    },
    # ---- Feishu Message tools ----
    "feishu_send_message": {
        "icon": "💌",
        "title": "Send message",
        "summary_template": "Sending message {args_summary}",
    },
    "feishu_get_messages": {
        "icon": "💌",
        "title": "Get messages",
        "summary_template": "Getting messages {args_summary}",
    },
}

# Status → Feishu card color token
_STATUS_COLORS: dict[str, str] = {
    "pending": "grey",
    "running": "blue",
    "success": "green",
    "error": "red",
}

# Status → display label
_STATUS_LABELS: dict[str, str] = {
    "pending": "等待中",
    "running": "执行中",
    "success": "完成",
    "error": "失败",
}

# Status → icon emoji
_STATUS_ICONS: dict[str, str] = {
    "pending": "⏳",
    "running": "🔄",
    "success": "✅",
    "error": "❌",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_descriptor(tool_name: str) -> Optional[dict[str, str]]:
    """Return the descriptor for tool_name, or None if not found.

    Tries exact match first, then prefix match for namespaced tool names.
    """
    if tool_name in TOOL_DESCRIPTORS:
        return TOOL_DESCRIPTORS[tool_name]
    # Prefix match: feishu_calendar_* etc.
    for key, desc in TOOL_DESCRIPTORS.items():
        if tool_name.startswith(key + "_") or tool_name.startswith(key + "-"):
            return desc
    # Generic humanise fallback
    return None


def _humanize_tool_name(tool_name: str) -> str:
    """Convert snake_case tool name to Title Case display name."""
    cleaned = re.sub(r"[-_]+", " ", tool_name).strip()
    if not cleaned:
        return "Tool"
    return cleaned[0].upper() + cleaned[1:]


def _extract_args_summary(tool_name: str, args: dict[str, Any]) -> str:
    """Extract a short human-readable summary from tool args."""
    if not args:
        return ""

    # Known high-signal param keys, in priority order
    priority_keys = [
        "description", "query", "q", "task", "prompt",
        "file_path", "path", "file", "url",
        "command", "script",
        "file_token", "table_id", "record_id", "node_token",
        "pattern", "content",
    ]
    for key in priority_keys:
        val = args.get(key)
        if val and isinstance(val, str):
            val = val.strip()
            if val:
                # Truncate long values
                return val[:80] + "..." if len(val) > 80 else val

    # Fallback: first string value
    for val in args.values():
        if val and isinstance(val, str):
            val = val.strip()
            if val:
                return val[:80] + "..." if len(val) > 80 else val

    return ""


def _extract_result_summary(result: Any) -> str:
    """Extract a short summary from a tool result."""
    if result is None:
        return ""
    if isinstance(result, str):
        text = result.strip()
        return text[:100] + "..." if len(text) > 100 else text
    if isinstance(result, dict):
        # Try common result keys
        for key in ("message", "content", "text", "result", "data"):
            val = result.get(key)
            if val and isinstance(val, str):
                val = val.strip()
                return val[:100] + "..." if len(val) > 100 else val
    return str(result)[:100]


def _build_tool_title(tool_name: str, descriptor: Optional[dict[str, str]]) -> str:
    """Build the display title for a tool."""
    if descriptor:
        return descriptor["title"]
    return _humanize_tool_name(tool_name)


def _build_tool_icon(tool_name: str, descriptor: Optional[dict[str, str]]) -> str:
    """Return the icon for a tool."""
    if descriptor:
        return descriptor["icon"]
    return "🔧"


# ---------------------------------------------------------------------------
# ToolUseDisplay
# ---------------------------------------------------------------------------

class ToolUseDisplay:
    """Renders tool_use events into Feishu card block JSON.

    Usage::

        display = ToolUseDisplay()
        block = display.render_tool_call(
            tool_name="read",
            args={"file_path": "/tmp/foo.txt"},
            status="success",
            result="file contents...",
        )
        # block is a dict ready to embed in a Feishu card body

    The returned block uses a Feishu ``markdown`` element for broad
    compatibility with both CardKit and IM PATCH modes.
    """

    def render_tool_call(
        self,
        tool_name: str,
        args: Optional[dict[str, Any]] = None,
        status: str = "pending",
        result: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Render a single tool call as a Feishu card block dict.

        Args:
            tool_name: The tool's canonical name (e.g. ``"bash"``, ``"feishu_wiki_search"``).
            args: The tool's input arguments dict.
            status: One of ``pending`` / ``running`` / ``success`` / ``error``.
            result: Optional tool output (str, dict, or None).

        Returns:
            A Feishu card element dict (``tag: "markdown"``).
        """
        if args is None:
            args = {}

        descriptor = _resolve_descriptor(tool_name)
        icon = _build_tool_icon(tool_name, descriptor)
        title = _build_tool_title(tool_name, descriptor)
        status_icon = _STATUS_ICONS.get(status, "🔧")
        status_label = _STATUS_LABELS.get(status, status)
        color = _STATUS_COLORS.get(status, "grey")

        args_summary = _extract_args_summary(tool_name, args)

        # Build summary line from template or default
        if descriptor and args_summary:
            summary = descriptor["summary_template"].format(
                tool_name=tool_name,
                args_summary=args_summary,
                result_summary=_extract_result_summary(result),
            )
        elif args_summary:
            summary = f"{title}: {args_summary}"
        else:
            summary = title

        # Build result snippet for success/error
        result_line = ""
        if status == "error" and result:
            err_text = _extract_result_summary(result)
            if err_text:
                result_line = f"\n> {err_text}"
        elif status == "success" and result and isinstance(result, str):
            snippet = result.strip()[:120]
            if snippet:
                result_line = f"\n> {snippet}"

        # Compose markdown content
        content = f"{icon} **{title}** {status_icon} *{status_label}*\n{summary}{result_line}"

        return {
            "tag": "markdown",
            "content": content,
            "text_align": "left",
            # Attach metadata for upstream consumers (streaming_controller etc.)
            "_meta": {
                "tool_name": tool_name,
                "status": status,
                "color": color,
            },
        }

    def render_tool_section(
        self,
        steps: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Render multiple tool calls as a collapsible card section.

        Args:
            steps: List of dicts with keys ``tool_name``, ``args``,
                   ``status``, and optionally ``result``.

        Returns:
            A Feishu card ``collapsible_panel`` block dict.
        """
        blocks = [
            self.render_tool_call(
                tool_name=step.get("tool_name", "tool"),
                args=step.get("args") or {},
                status=step.get("status", "pending"),
                result=step.get("result"),
            )
            for step in steps
        ]

        count = len(steps)
        suffix_zh = f"查看 {count} 个步骤"

        return {
            "tag": "collapsible_panel",
            "expanded": False,
            "header": {
                "tag": "markdown",
                "content": f"🛠️ **工具调用** — {suffix_zh}",
            },
            "elements": blocks,
        }

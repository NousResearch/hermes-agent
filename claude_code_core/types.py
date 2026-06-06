"""Type definitions for Claude Code CLI stream-json output.

Frontend-agnostic types shared by any Claude Code integration
(Discord bot, Teams bot, CLI wrapper, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class ImageData:
    """Base64-encoded image data with its media type.

    Used to pass downloaded image attachments to the Claude Code CLI
    via stream-json base64 image blocks.
    """

    data: str  # base64-encoded string (no data: URI prefix)
    media_type: str  # e.g. "image/jpeg", "image/png"


class MessageType(Enum):
    """Top-level message types in stream-json output."""

    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"
    RESULT = "result"
    PROGRESS = "progress"
    STREAM_EVENT = "stream_event"  # low-level streaming events; parsed but not acted on
    RATE_LIMIT_EVENT = "rate_limit_event"  # rate limit info from Anthropic API


class ContentBlockType(Enum):
    """Content block types within assistant messages."""

    TEXT = "text"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"


class ToolCategory(Enum):
    """Categories for tool use, used for status emoji selection."""

    READ = "read"
    EDIT = "edit"
    COMMAND = "command"
    WEB = "web"
    THINK = "think"
    ASK = "ask"
    TASK = "task"
    PLAN = "plan"
    OTHER = "other"


# Map tool names to categories
TOOL_CATEGORIES: dict[str, ToolCategory] = {
    "Read": ToolCategory.READ,
    "Glob": ToolCategory.READ,
    "Grep": ToolCategory.READ,
    "LS": ToolCategory.READ,
    "Write": ToolCategory.EDIT,
    "Edit": ToolCategory.EDIT,
    "NotebookEdit": ToolCategory.EDIT,
    "Bash": ToolCategory.COMMAND,
    "WebFetch": ToolCategory.WEB,
    "WebSearch": ToolCategory.WEB,
    "Task": ToolCategory.OTHER,
    "AskUserQuestion": ToolCategory.ASK,
    "TodoWrite": ToolCategory.TASK,
    "ExitPlanMode": ToolCategory.PLAN,
}


@dataclass
class RateLimitInfo:
    """Rate limit information from a rate_limit_event stream-json message."""

    rate_limit_type: str  # "five_hour" | "seven_day" | "seven_day_sonnet" | etc.
    status: str  # "allowed" | "allowed_warning" | "rejected"
    utilization: float  # 0.0-1.0
    resets_at: int  # Unix timestamp
    is_using_overage: bool = False


@dataclass
class AskOption:
    """A single selectable option in an AskUserQuestion prompt."""

    label: str
    description: str = ""


@dataclass
class AskQuestion:
    """A single question from an AskUserQuestion tool call."""

    question: str
    header: str = ""
    multi_select: bool = False
    options: list[AskOption] = field(default_factory=list)


@dataclass
class TodoItem:
    """A single item in a TodoWrite task list."""

    content: str
    status: str  # "pending", "in_progress", "completed"
    active_form: str = ""  # Present-continuous label shown while in_progress


@dataclass
class PermissionRequest:
    """A permission request from Claude Code for a tool execution."""

    request_id: str
    tool_name: str
    tool_input: dict[str, Any] = field(default_factory=dict)


@dataclass
class ElicitationRequest:
    """An elicitation request from an MCP server."""

    request_id: str
    server_name: str
    mode: str  # "form-mode" or "url-mode"
    message: str = ""
    url: str = ""  # url-mode only
    schema: dict[str, Any] = field(default_factory=dict)  # form-mode only


@dataclass
class ToolUseEvent:
    """Parsed tool use event from stream-json."""

    tool_id: str
    tool_name: str
    tool_input: dict[str, Any]
    category: ToolCategory

    @property
    def display_name(self) -> str:
        """Human-readable description of what this tool is doing."""
        name = self.tool_name
        inp = self.tool_input

        if name == "Read":
            return f"Reading: {inp.get('file_path', 'unknown')}"
        if name == "Write":
            return f"Writing: {inp.get('file_path', 'unknown')}"
        if name == "Edit":
            return f"Editing: {inp.get('file_path', 'unknown')}"
        if name in ("Glob", "Grep"):
            pattern = inp.get("pattern", inp.get("glob", ""))
            return f"Searching: {pattern}"
        if name == "Bash":
            cmd = inp.get("command", "")
            # Truncate long commands
            if len(cmd) > 60:
                cmd = cmd[:57] + "..."
            return f"Running: {cmd}"
        if name == "WebSearch":
            return f"Searching web: {inp.get('query', '')}"
        if name == "WebFetch":
            return f"Fetching: {inp.get('url', '')}"
        if name == "Task":
            return f"Spawning agent: {inp.get('description', '')}"
        return f"Using: {name}"


@dataclass
class StreamEvent:
    """A parsed event from the Claude Code stream-json output."""

    message_type: MessageType
    raw: dict = field(default_factory=dict)
    session_id: str | None = None
    text: str | None = None
    tool_use: ToolUseEvent | None = None
    tool_result_id: str | None = None
    tool_result_content: str | None = None
    thinking: str | None = None
    has_redacted_thinking: bool = False
    ask_questions: list[AskQuestion] | None = None
    todo_list: list[TodoItem] | None = None
    is_plan_approval: bool = False
    permission_request: PermissionRequest | None = None
    elicitation: ElicitationRequest | None = None
    is_compact: bool = False
    compact_trigger: str | None = None
    compact_pre_tokens: int | None = None
    is_complete: bool = False
    is_partial: bool = False
    cost_usd: float | None = None
    duration_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    context_window: int | None = None
    error: str | None = None
    rate_limit_info: RateLimitInfo | None = None

"""Pure policy primitives for the Hermes CaMeL guard plugin.

This module deliberately has no Hermes imports.  The runtime adapter in
``__init__.py`` owns plugin hooks and host-LLM access; this file owns only the
capability vocabulary, tool classification, and classifier response contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any, Mapping


CAPABILITY_LABELS: dict[str, str] = {
    "browser_interaction": "browser interaction",
    "command_execution": "command execution",
    "delegation": "delegation or subagent execution",
    "external_messaging": "external messaging",
    "external_system": "external-system side effects",
    "file_mutation": "file mutation",
    "persistent_memory": "persistent-memory mutation",
    "scheduled_action": "scheduled-action mutation or execution",
    "skill_mutation": "skill mutation",
    "task_state_mutation": "task and kanban state mutation",
    "workspace_mutation": "workspace or project mutation",
}

CAPABILITY_IDS: tuple[str, ...] = tuple(sorted(CAPABILITY_LABELS))

CLASSIFIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "allowed_capabilities": {
            "type": "array",
            "items": {"type": "string", "enum": list(CAPABILITY_IDS)},
            "uniqueItems": True,
        },
        "denied_capabilities": {
            "type": "array",
            "items": {"type": "string", "enum": list(CAPABILITY_IDS)},
            "uniqueItems": True,
        },
        "rationale": {"type": "string", "maxLength": 240},
    },
    "required": ["allowed_capabilities", "denied_capabilities", "rationale"],
}

CLASSIFIER_INSTRUCTIONS = """Classify trusted operator intent for a tool-policy guard.

You receive exactly one trusted user message and no tool output, retrieved
content, memory, web page, file contents, or model-generated instructions.
Only authorize a capability when the trusted user explicitly requests it or
when it is unavoidably required to complete that request. Ambiguity means the
capability is not authorized. Put a capability in denied_capabilities only
when the trusted user explicitly forbids it. Never invent capability ids.
Return only the requested structured object.

Capabilities:
""" + "\n".join(
    f"- {capability}: {CAPABILITY_LABELS[capability]}" for capability in CAPABILITY_IDS
)


_READ_ONLY_BROWSER_TOOLS = {
    "browser_console",
    "browser_get_images",
    "browser_snapshot",
    "browser_vision",
}

_READ_ONLY_COMPUTER_ACTIONS = {
    "capture",
    "list_apps",
    "list_windows",
    "wait",
}

_READ_ONLY_PROCESS_ACTIONS = {
    "list",
    "log",
    "poll",
    "wait",
}

_READ_ONLY_DISCORD_ACTIONS = {
    "channel_info",
    "fetch_messages",
    "list_channels",
    "list_guilds",
    "list_pins",
    "list_roles",
    "member_info",
    "search_members",
    "server_info",
}

_READ_ONLY_KANBAN_TOOLS = {
    "kanban_attachments",
    "kanban_list",
    "kanban_show",
}

_READ_ONLY_FEISHU_TOOLS = {
    "feishu_doc_read",
    "feishu_drive_list_comment_replies",
    "feishu_drive_list_comments",
}

_DESKTOP_INTERACTION_TOOLS = {
    "annotate_preview",
    "apply_layout",
    "close_preview",
    "close_terminal",
    "focus_pane",
    "open_preview",
    "desktop_preview",
    "drive_preview",
    "set_theme",
}

_EXTERNAL_MESSAGING_TOOLS = {
    "feishu_drive_add_comment",
    "feishu_drive_reply_comment",
    "message_agent",
    "react_to_message",
    "yb_send_dm",
    "yb_send_sticker",
}

_EXTERNAL_SYSTEM_TOOLS = {
    "ha_call_service",
    "image_generate",
    "rl_edit_config",
    "rl_start_training",
    "rl_stop_training",
    "setup_mcp",
    "text_to_speech",
    "video_generate",
    "xai_video_edit",
    "xai_video_extend",
}

_READ_ONLY_TOOLS = {
    "clarify",
    "read_file",
    "search_files",
    "web_search",
    "web_extract",
    "vision_analyze",
    "video_analyze",
    "session_search",
    "skills_list",
    "skill_view",
    "read_terminal",
    "read_window_below",
}


def normalize_mode(value: Any) -> str:
    """Return an explicit guard mode; invalid values stay safely disabled."""
    normalized = str(value or "off").strip().lower()
    if normalized in {"legacy", "disabled"}:
        return "off"
    if normalized not in {"off", "monitor", "enforce"}:
        return "off"
    return normalized


_FIXED_CAPABILITIES = {
    **dict.fromkeys({"terminal", "execute_code"}, "command_execution"),
    **dict.fromkeys({"write_file", "patch"}, "file_mutation"),
    **dict.fromkeys({"delegate_task", "mixture_of_agents"}, "delegation"),
    **dict.fromkeys(_EXTERNAL_MESSAGING_TOOLS, "external_messaging"),
    **dict.fromkeys(_EXTERNAL_SYSTEM_TOOLS, "external_system"),
    **dict.fromkeys(_DESKTOP_INTERACTION_TOOLS, "browser_interaction"),
    "memory": "persistent_memory",
    "skill_manage": "skill_mutation",
}
_ACTION_CAPABILITIES = {
    "process": ("command_execution", _READ_ONLY_PROCESS_ACTIONS),
    "process_manage": ("command_execution", _READ_ONLY_PROCESS_ACTIONS),
    "cronjob": ("scheduled_action", {"list"}),
    "cronjob_manage": ("scheduled_action", {"list"}),
    "send_message": ("external_messaging", {"list"}),
    "computer_use": ("browser_interaction", _READ_ONLY_COMPUTER_ACTIONS),
    "desktop_project": ("workspace_mutation", {"list"}),
    "project_create": ("workspace_mutation", set()),
    "project_switch": ("workspace_mutation", set()),
    "discord": ("external_system", _READ_ONLY_DISCORD_ACTIONS),
    "discord_admin": ("external_system", _READ_ONLY_DISCORD_ACTIONS),
}
_READ_ONLY_TOOLS |= (
    _READ_ONLY_BROWSER_TOOLS
    | _READ_ONLY_FEISHU_TOOLS
    | {
        "yb_query_group_info",
        "yb_query_member_info",
    }
)


def capability_for(tool_name: str, args: Mapping[str, Any] | None = None) -> str:
    """Classify effects; unknown names conservatively require external authority."""
    name = str(tool_name or "").strip()
    payload = args if isinstance(args, Mapping) else {}
    action = str(payload.get("action") or "").strip().lower()
    if name in _FIXED_CAPABILITIES:
        return _FIXED_CAPABILITIES[name]
    if name in _ACTION_CAPABILITIES:
        capability, read_actions = _ACTION_CAPABILITIES[name]
        return "" if action in read_actions else capability
    if name in {"todo", "todo_list"}:
        return "" if not payload or action == "read" else "task_state_mutation"
    if name.startswith("kanban_"):
        return "" if name in _READ_ONLY_KANBAN_TOOLS else "task_state_mutation"
    if name in _READ_ONLY_TOOLS:
        return ""
    if name.startswith("browser_"):
        return "browser_interaction"
    return "external_system"


def is_untrusted_output(tool_name: str) -> bool:
    """Whether a tool result must be treated as data rather than control."""
    name = str(tool_name or "").strip()
    return bool(name) and name not in {"clarify", "todo", "todo_list"}


def normalized_capabilities(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {
        str(item).strip() for item in value if str(item).strip() in CAPABILITY_LABELS
    }


@dataclass(frozen=True)
class CapabilityPlan:
    allowed: frozenset[str] = frozenset()
    denied: frozenset[str] = frozenset()
    status: str = "unclassified"
    rationale: str = ""


@dataclass
class TurnState:
    scope_id: str
    turn_id: str
    trusted_user_message: str
    untrusted_sources: set[str] = field(default_factory=set)
    plan: CapabilityPlan | None = None
    classification_lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False,
    )

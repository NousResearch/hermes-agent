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


_CONTROL_OUTPUT_TOOLS = {
    "clarify",
    "skill_view",
    "skills_list",
    "todo",
}

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

_EXTERNAL_PREFIXES = (
    "discord_",
    "feishu_",
    "mcp_",
    "yuanbao_",
)


def normalize_mode(value: Any) -> str:
    """Return an explicit guard mode; invalid values stay safely disabled."""
    normalized = str(value or "off").strip().lower()
    if normalized in {"legacy", "disabled"}:
        return "off"
    if normalized not in {"off", "monitor", "enforce"}:
        return "off"
    return normalized


def capability_for(tool_name: str, args: Mapping[str, Any] | None = None) -> str:
    """Map a current Hermes tool call to its side-effect capability.

    An empty string means the call is not policy-gated.  This is intentionally
    about effects, not output trust: read-only tools may still produce
    untrusted data and are tracked separately by :func:`is_untrusted_output`.
    """
    name = str(tool_name or "").strip()
    payload = args if isinstance(args, Mapping) else {}
    action = str(payload.get("action") or "").strip().lower()

    if name in {"terminal", "execute_code"}:
        return "command_execution"
    if name == "process":
        return "" if action in _READ_ONLY_PROCESS_ACTIONS else "command_execution"
    if name in {"write_file", "patch"}:
        return "file_mutation"
    if name == "memory":
        return "persistent_memory"
    if name == "skill_manage":
        return "skill_mutation"
    if name == "cronjob":
        return "" if action == "list" else "scheduled_action"
    if name == "send_message":
        return "" if action == "list" else "external_messaging"
    if name in _EXTERNAL_MESSAGING_TOOLS:
        return "external_messaging"
    if name in {"delegate_task", "mixture_of_agents"}:
        return "delegation"
    if name in _EXTERNAL_SYSTEM_TOOLS:
        return "external_system"
    if name == "computer_use":
        return "" if action in _READ_ONLY_COMPUTER_ACTIONS else "browser_interaction"
    if name in _DESKTOP_INTERACTION_TOOLS:
        return "browser_interaction"
    if name in {"project_create", "project_switch"}:
        return "workspace_mutation"
    if name.startswith("kanban_"):
        return "" if name in _READ_ONLY_KANBAN_TOOLS else "task_state_mutation"
    if name in {"discord", "discord_admin"}:
        return "" if action in _READ_ONLY_DISCORD_ACTIONS else "external_system"
    if name in _READ_ONLY_FEISHU_TOOLS:
        return ""
    if name.startswith("feishu_"):
        return "external_system"
    if name.startswith("yb_"):
        return ""
    if name.startswith("browser_") and name not in _READ_ONLY_BROWSER_TOOLS:
        return "browser_interaction"
    if name.startswith(_EXTERNAL_PREFIXES):
        return "external_system"
    return ""


def is_untrusted_output(tool_name: str) -> bool:
    """Whether a tool result must be treated as data rather than control."""
    name = str(tool_name or "").strip()
    return bool(name) and name not in _CONTROL_OUTPUT_TOOLS


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

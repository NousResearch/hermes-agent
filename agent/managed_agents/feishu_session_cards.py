"""Feishu interactive card builder for session selection.

When a group message arrives and the correct session is ambiguous,
this module builds the JSON card payload that asks the user to choose.

Card structure follows Feishu's interactive card message format.
https://open.feishu.cn/document/uAjLw4CM/ukzMukzMukzl/feishu-cards/card-components/interactive-components/button
"""

from __future__ import annotations

from typing import Any

from .feishu_session_resolver import AmbiguityInfo


def build_ambiguity_card(
    ambiguity: AmbiguityInfo,
    message_preview: str = "",
    max_preview_length: int = 50,
) -> dict[str, Any]:
    """Build an interactive card asking the user which session to route to.

    Args:
        ambiguity: The AmbiguityInfo from resolve_feishu_session().
        message_preview: A short preview of the incoming message.
        max_preview_length: Max characters for the message preview.

    Returns:
        A dict representing a Feishu interactive card message body.
    """
    if not ambiguity.available_sessions:
        # No sessions to choose from; return a simple informational card.
        return _build_no_session_card(ambiguity, message_preview)

    preview = _truncate(message_preview, max_preview_length)

    # Build session selection buttons — one per available session.
    elements: list[dict[str, Any]] = []

    if preview:
        elements.append({
            "tag": "div",
            "text": {
                "tag": "plain_text",
                "content": f"收到消息：{preview}",
            },
        })

    # Instruction text
    elements.append({
        "tag": "div",
        "text": {
            "tag": "plain_text",
            "content": "请选择要发送到的会话：",
        },
    })

    # Session buttons — each maps to a select_session card action.
    actions: list[dict[str, Any]] = []
    for i, session_id in enumerate(ambiguity.available_sessions):
        label = _session_label(session_id)
        actions.append({
            "tag": "button",
            "text": {"tag": "plain_text", "content": label},
            "type": "primary" if i == 0 else "default",
            "value": {
                "action": "select_session",
                "session_id": session_id,
                "workspace_id": ambiguity.workspace_id,
                "chat_id": ambiguity.chat_id,
                "thread_id": ambiguity.thread_id or "",
            },
        })

    elements.append({
        "tag": "action",
        "actions": actions,
    })

    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": "会话选择"},
        },
        "elements": elements,
    }


def build_card_acknowledgement(session_id: str, workspace_id: str) -> dict[str, Any]:
    """Build a confirmation card after the user selects a session.

    Args:
        session_id: The session the user selected.
        workspace_id: The workspace the session belongs to.

    Returns:
        A dict representing a simple Feishu card confirming the selection.
    """
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": "已选择会话"},
        },
        "elements": [
            {
                "tag": "div",
                "text": {
                    "tag": "plain_text",
                    "content": f"已绑定到会话：{_session_label(session_id)}",
                },
            },
            {
                "tag": "div",
                "text": {
                    "tag": "plain_text",
                    "content": f"工作区：{workspace_id}",
                },
            },
        ],
    }


def build_rejection_card(chat_id: str) -> dict[str, Any]:
    """Build a card explaining that the message could not be routed.

    Used when no sessions are available and ambiguity detection has
    no options to present.
    """
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": "无法路由"},
        },
        "elements": [
            {
                "tag": "div",
                "text": {
                    "tag": "plain_text",
                    "content": f"当前群聊（{chat_id}）没有可用的会话，请先创建任务或指定会话。",
                },
            },
        ],
    }


def _build_no_session_card(
    ambiguity: AmbiguityInfo,
    message_preview: str = "",
) -> dict[str, Any]:
    """Build a card when ambiguity is detected but no sessions are available."""
    return build_rejection_card(ambiguity.chat_id)


def _session_label(session_id: str) -> str:
    """Derive a human-readable label from a session ID."""
    if session_id.startswith("ses-feishu-thread-"):
        thread_id = session_id[len("ses-feishu-thread-"):]
        # Trim long thread IDs for readability
        if len(thread_id) > 20:
            thread_id = thread_id[:17] + "..."
        return f"线程 {thread_id}"
    if session_id.startswith("ses-feishu-"):
        chat_id = session_id[len("ses-feishu-"):]
        if len(chat_id) > 20:
            chat_id = chat_id[:17] + "..."
        return f"会话 {chat_id}"
    # Generic session label
    if len(session_id) > 30:
        return session_id[:27] + "..."
    return session_id


def _truncate(text: str, max_length: int) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 1] + "…"

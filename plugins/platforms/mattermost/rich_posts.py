"""Pure builders for Mattermost native rich-post attachments."""

from __future__ import annotations

import re
from typing import Any, Optional

_ATX_HEADING_RE = re.compile(r"^ {0,3}#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*(?:\r?\n)?$")
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_SAFE_ACTION_STYLES = frozenset({
    "default",
    "primary",
    "success",
    "good",
    "warning",
    "danger",
})


def build_attachment_action(
    action_id: Any,
    name: Any,
    url: Any,
    *,
    style: Any = "default",
    context: Any = None,
) -> Optional[dict[str, Any]]:
    """Build one Mattermost attachment button action, or omit invalid input."""
    required = (action_id, name, url)
    if any(not isinstance(value, str) or not value.strip() for value in required):
        return None
    action_id, name, url = (value.strip() for value in required)
    normalized_style = style.strip().lower() if isinstance(style, str) else "default"
    if normalized_style not in _SAFE_ACTION_STYLES:
        normalized_style = "default"
    integration: dict[str, Any] = {"url": url}
    if isinstance(context, dict):
        integration["context"] = dict(context)
    return {
        "id": action_id,
        "name": name,
        "type": "button",
        "style": normalized_style,
        "integration": integration,
    }


def _sanitize_actions(actions: Any) -> list[dict[str, Any]]:
    if not isinstance(actions, (list, tuple)):
        return []
    valid: list[dict[str, Any]] = []
    for action in actions:
        if not isinstance(action, dict) or action.get("type") != "button":
            continue
        integration = action.get("integration")
        if not isinstance(integration, dict):
            continue
        built = build_attachment_action(
            action.get("id"),
            action.get("name"),
            integration.get("url"),
            style=action.get("style", "default"),
            context=integration.get("context"),
        )
        if built is not None:
            valid.append(built)
    return valid


def _promote_first_heading(markdown: str) -> tuple[Optional[str], str]:
    lines = markdown.splitlines(keepends=True)
    fence: Optional[str] = None
    for index, line in enumerate(lines):
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            continue
        if fence is not None:
            continue
        match = _ATX_HEADING_RE.match(line)
        if match:
            return match.group(1), "".join(lines[:index] + lines[index + 1 :])
    return None, markdown


def render_rich_post(
    markdown: Any,
    *,
    color: Any = None,
    actions: Any = None,
) -> Optional[dict[str, Any]]:
    """Build Mattermost ``props.attachments`` from Markdown, or return ``None``."""
    if not isinstance(markdown, str) or not markdown or len(markdown) > 16_383:
        return None
    title, body = _promote_first_heading(markdown)
    attachment: dict[str, Any] = {
        "fallback": markdown,
        "text": body,
    }
    if title:
        attachment["title"] = title
    if color:
        attachment["color"] = color
    valid_actions = _sanitize_actions(actions)
    if valid_actions:
        attachment["actions"] = valid_actions
    return {"props": {"attachments": [attachment]}}

"""Slack Block Kit rendering for presentation-neutral Collective Wisdom views."""

from __future__ import annotations

from typing import Any, Iterable


_MAX_ITEMS = 5
_MAX_SECTION_TEXT = 2900
_MAX_BUTTON_LABEL = 75


def _trim(value: object, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _escape_mrkdwn(value: object) -> str:
    return (
        str(value or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def _action_blocks(actions: Iterable[Any], *, row: str) -> list[dict[str, Any]]:
    elements: list[dict[str, Any]] = []
    for index, action in enumerate(actions):
        label = _trim(getattr(action, "label", "Continue"), _MAX_BUTTON_LABEL)
        button: dict[str, Any] = {
            "type": "button",
            "text": {"type": "plain_text", "text": label, "emoji": True},
            "action_id": f"hermes_wisdom_{row}_{index}",
        }
        url = getattr(action, "url", None)
        callback_data = getattr(action, "callback_data", None)
        if isinstance(url, str) and url:
            button["url"] = url
            # Slack requires a value on some interactive URL-button paths.
            button["value"] = "wisdom:portal"
        elif isinstance(callback_data, str) and callback_data:
            button["value"] = callback_data
        else:
            continue
        if getattr(action, "primary", False):
            button["style"] = "primary"
        elif getattr(action, "destructive", False):
            button["style"] = "danger"
        elements.append(button)

    # Slack allows at most 25 elements in an actions block. Wisdom views are
    # intentionally much smaller, but chunk defensively so an upstream view
    # cannot produce an invalid payload.
    return [
        {"type": "actions", "elements": elements[offset : offset + 5]}
        for offset in range(0, len(elements), 5)
    ]


def render_wisdom_blocks(view: Any) -> list[dict[str, Any]]:
    """Render one ``WisdomView`` as a compact, actionable Slack card."""
    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": _trim(getattr(view, "title", "Collective Wisdom"), 150),
                "emoji": True,
            },
        }
    ]
    blocks.extend(
        _action_blocks(getattr(view, "navigation_actions", []) or [], row="navigation")
    )
    summary = str(getattr(view, "summary", "") or "").strip()
    if summary:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": _trim(_escape_mrkdwn(summary), _MAX_SECTION_TEXT),
            },
        })

    items = list(getattr(view, "items", []) or [])[:_MAX_ITEMS]
    for item_index, item in enumerate(items):
        title = _escape_mrkdwn(getattr(item, "title", "Skill"))
        detail = _escape_mrkdwn(getattr(item, "detail", ""))
        text = f"*{title}*"
        if detail:
            text += f"\n{detail}"
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": _trim(text, _MAX_SECTION_TEXT)},
        })
        blocks.extend(
            _action_blocks(getattr(item, "actions", []) or [], row=f"item_{item_index}")
        )

    notice = str(getattr(view, "notice", "") or "").strip()
    if notice:
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": _trim(f"_{_escape_mrkdwn(notice)}_", _MAX_SECTION_TEXT),
                }
            ],
        })
    blocks.extend(_action_blocks(getattr(view, "actions", []) or [], row="global"))
    return blocks[:50]


def wisdom_fallback_text(view: Any) -> str:
    """Return useful notification/fallback text for Slack accessibility."""
    rendered = str(view.to_text() if hasattr(view, "to_text") else "").strip()
    return _trim(rendered or getattr(view, "title", "Collective Wisdom"), 39000)

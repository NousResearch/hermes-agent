"""Legacy recommendation preview renderers, not the runtime delivery path.

Old ``wa:`` targets only open current review/settings; they carry no consent.
New runtime cards use mediation_view with actor-bound native interactions.
"""

from __future__ import annotations

import html as _html
from typing import Any

from .notify import RecommendationEvent

PRODUCT_LABEL = "Hermes Collective Wisdom"


def action_dicts(event: RecommendationEvent) -> list[dict[str, Any]]:
    """Generic action shape shared by Desktop/Dashboard and tests."""
    return [
        {"label": a.label, "callback_data": a.target, "primary": a.primary}
        for a in event.allowed_actions
    ]


def render_plain(event: RecommendationEvent) -> str:
    notice = event.notice
    body = notice.text if notice else f"{event.title}\n{event.explanation}"
    return f"{PRODUCT_LABEL}\n\n{body}"


def render_telegram_html(event: RecommendationEvent) -> str:
    """HTML for Telegram; the editorial name is the emphasized line, never the product."""
    notice = event.notice
    lines = list(notice.lines) if notice else [event.explanation]
    rendered: list[str] = []
    for index, line in enumerate(lines):
        escaped = _html.escape(line)
        if index == 0 and line.strip():
            rendered.append(f"<b>{escaped}</b>")
        elif line.startswith("    "):
            rendered.append(f"<i>{escaped.strip()}</i>")
        elif line in {"Why we ask"}:
            rendered.append(f"<b>{escaped}</b>")
        else:
            rendered.append(escaped)
    controls = " ".join(
        '<tg-button type="callback_data"'
        + (' style="primary"' if a.primary else "")
        + f' data="{_html.escape(a.target, quote=True)}">{_html.escape(a.label)}</tg-button>'
        for a in event.allowed_actions
    )
    return (
        f"<h3>{_html.escape(PRODUCT_LABEL)}</h3>"
        f"<p><b>{_html.escape(event.title)}</b><br/><br/>"
        + "<br/>".join(rendered)
        + (f"<br/>{controls}" if controls else "")
        + "</p>"
    )


def render_slack_blocks(event: RecommendationEvent) -> list[dict[str, Any]]:
    notice = event.notice
    lines = list(notice.lines) if notice else [event.explanation]
    name = lines[0] if lines else event.skill_id
    body = "\n".join(lines[1:]) if len(lines) > 1 else ""
    blocks: list[dict[str, Any]] = [
        {"type": "context", "elements": [{"type": "mrkdwn", "text": f"_{PRODUCT_LABEL}_ • {event.title}"}]},
        {"type": "header", "text": {"type": "plain_text", "text": name[:150], "emoji": True}},
    ]
    if body.strip():
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": body[:2900]}})
    elements = [
        {
            "type": "button",
            "text": {"type": "plain_text", "text": a.label[:75], "emoji": True},
            "action_id": f"hermes_wisdom_agent_{index}",
            "value": a.target,
            **({"style": "primary"} if a.primary else {}),
        }
        for index, a in enumerate(event.allowed_actions)
    ]
    if elements:
        blocks.append({"type": "actions", "elements": elements[:5]})
    return blocks


def render_desktop(event: RecommendationEvent) -> dict[str, Any]:
    """Structured payload for the Desktop/Dashboard notification surface."""
    return {
        "product": PRODUCT_LABEL,
        "event_type": event.event_type,
        "title": event.title,
        "skill_name": event.notice.lines[0] if event.notice and event.notice.lines else event.skill_id,
        "lines": list(event.notice.lines) if event.notice else [event.explanation],
        "actions": action_dicts(event),
        "expires_at": event.expires_at,
        "dedup_key": event.dedup_key,
        "hints": dict(event.rendering_hints),
    }

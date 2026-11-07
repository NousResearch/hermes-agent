"""Google Chat exec-approval cards: Card v2 buttons and the click events that resolve them.

The adapter posts the card and keeps the session key on its side; a button only carries
the choice. Chat delivers a click only to the app's own connection: for an add-on app over
Pub/Sub the button's ``function`` must be the app's topic (``projects/<p>/topics/<t>``), over
HTTP its endpoint URL; a non-add-on app echoes any name. The click then arrives without
``invokedFunction``, so the card is recognised by its ``hermes_action`` parameter. A click (native ``CARD_CLICKED`` event or Workspace add-on
``buttonClickedPayload``, over Pub/Sub or HTTP) is parsed here into a ``CardClick`` and
resolved by the adapter against the card it remembers, never against values the click
payload claims.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from gateway.platforms.base_exec_approval import (
    approval_timeout_seconds, ea_header_text, ea_reason_label_text, ea_smart_deny_line_text,
    format_approval_deadline_line,
)

APPROVAL_FUNCTION = "hermes_approval"  # fallback ``function`` when the connection is unknown
APPROVAL_ACTION = "approval"  # ``hermes_action`` parameter value of an approval button
APPROVAL_CHOICES = frozenset({"once", "session", "always", "deny"})
_CMD_BUDGET = 2000
# A plugin ``approve`` rule shows a synthetic target (``<tool> (plugin approval rule)``); the real
# subject is in the description, so the card leaves the label out (same rule as the desktop app).
_SYNTHETIC_TARGET = re.compile(r"^<[^>]+> \(")


@dataclass
class CardClick:
    """The fields of a card click the adapter needs, whatever the envelope format."""
    function: str
    parameters: dict[str, str] = field(default_factory=dict)
    message_name: str = ""
    space_name: str = ""
    space_type: str = ""
    user_email: str = ""
    user_name: str = ""
    user_display: str = ""


def _escape(text: str) -> str:
    # Card text is parsed as limited HTML: a command such as ``<terminal>`` would vanish.
    return html.escape(str(text or ""), quote=False)


def _body_widgets(command: str, description: str) -> list[dict[str, Any]]:
    widgets: list[dict[str, Any]] = [
        {"decoratedText": {"topLabel": ea_reason_label_text(), "text": _escape(description), "wrapText": True}}]
    command = str(command or "").strip()
    if command and not _SYNTHETIC_TARGET.match(command):
        cmd = command if len(command) <= _CMD_BUDGET else command[:_CMD_BUDGET] + "..."
        widgets.append({"textParagraph": {"text": _escape(cmd)}})
    return widgets


def approval_card(card_id: str, function: str, command: str, description: str,
                  actions: list[tuple[str, str, str]], smart_denied: bool = False) -> dict[str, Any]:
    """Card v2 entry for ``messages.create``; one button per ``(label, choice, style)`` action,
    each calling ``function`` (see the module docstring)."""
    widgets = _body_widgets(command, description)
    widgets.append({"textParagraph": {"text": _escape(format_approval_deadline_line(approval_timeout_seconds()))}})
    if smart_denied:
        widgets.append({"textParagraph": {"text": _escape(ea_smart_deny_line_text())}})
    buttons = [
        {"text": label, "onClick": {"action": {"function": function, "parameters": [
            {"key": "hermes_action", "value": APPROVAL_ACTION}, {"key": "choice", "value": choice}]}}}
        for label, choice, _style in actions]
    widgets.append({"buttonList": {"buttons": buttons}})
    return {"cardId": card_id, "card": {"header": {"title": f"⚠️ {ea_header_text()}"}, "sections": [{"widgets": widgets}]}}


def retired_card(card_id: str, command: str, description: str, outcome: str) -> dict[str, Any]:
    """The same card without buttons, ending on the outcome line."""
    widgets = _body_widgets(command, description)
    widgets.append({"textParagraph": {"text": f"<b>{_escape(outcome)}</b>"}})
    return {"cardId": card_id, "card": {"header": {"title": f"⚠️ {ea_header_text()}"}, "sections": [{"widgets": widgets}]}}


def _params_from(raw: Any) -> dict[str, str]:
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    params: dict[str, str] = {}
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict) and item.get("key") is not None:
            params[str(item["key"])] = str(item.get("value", ""))
    return params


def _click(function: Any, params: dict[str, str], message: Any, space: Any, user: Any) -> CardClick:
    message = message if isinstance(message, dict) else {}
    space = space if isinstance(space, dict) else (message.get("space") or {})
    user = user if isinstance(user, dict) else {}
    return CardClick(
        function=str(function or "").strip(), parameters=params,
        message_name=str(message.get("name") or "").strip(), space_name=str(space.get("name") or "").strip(),
        space_type=str(space.get("type") or space.get("spaceType") or "").upper(),
        user_email=str(user.get("email") or "").strip(), user_name=str(user.get("name") or "").strip(),
        user_display=str(user.get("displayName") or user.get("email") or user.get("name") or "").strip())


def parse_card_click(envelope: Any) -> Optional[CardClick]:
    """``CardClick`` for a card-click envelope, else None. Formats: native Chat ``CARD_CLICKED``
    (``action`` + ``common``) and Workspace add-on (``commonEventObject`` +
    ``chat.buttonClickedPayload``)."""
    if not isinstance(envelope, dict):
        return None
    if str(envelope.get("type") or "").upper() == "CARD_CLICKED":
        action = envelope.get("action") or {}
        common = envelope.get("common") or {}
        params = {**_params_from(common.get("parameters")), **_params_from(action.get("parameters"))}
        function = action.get("actionMethodName") or common.get("invokedFunction")
        return _click(function, params, envelope.get("message"), envelope.get("space"), envelope.get("user"))
    chat = envelope.get("chat")
    payload = chat.get("buttonClickedPayload") if isinstance(chat, dict) else None
    if isinstance(payload, dict):
        common = envelope.get("commonEventObject") or {}
        return _click(common.get("invokedFunction"), _params_from(common.get("parameters")),
                      payload.get("message"), payload.get("space"), chat.get("user"))
    return None

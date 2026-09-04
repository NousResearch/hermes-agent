"""Validation helpers for Telegram Bot API rich-message buttons.

The adapter accepts JSON-compatible ``InputRichMessage`` dictionaries so
plugins can use new Bot API fields before python-telegram-bot models them.
Only the Bot API 10.3 button contract is validated here; unrelated rich block
schemas remain Telegram's responsibility.
"""

from __future__ import annotations

import json
from typing import Any, Dict


_CONTENT_MODES = ("html", "markdown", "blocks")
_BUTTON_ACTIONS = (
    "url",
    "callback_data",
    "web_app",
    "login_url",
    "switch_inline_query",
    "switch_inline_query_current_chat",
    "switch_inline_query_chosen_chat",
    "copy_text",
    "disabled",
)
_BUTTON_STYLES = {"danger", "success", "primary", "link"}
_ROW_ALIGNMENTS = {"left", "center", "right"}
_MAX_VISITED_CONTAINERS = 10_000
_MAX_TRAVERSAL_DEPTH = 32
_MAX_RICH_TEXT_CHARS = 32_768


class RichMessageValidationError(ValueError):
    """A JSON rich-message payload violates Hermes' local safety contract."""


def _require_mapping(value: Any, message: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise RichMessageValidationError(message)
    return value


def _validate_button_text(value: Any) -> None:
    """Validate the restricted RichText grammar allowed in button labels.

    Bot API 10.3 permits only plain strings, RichTextCustomEmoji, and
    RichTextDateTime inside RichMessageButton.text.
    """
    stack = [(value, 0, True)]
    seen = set()
    visited = 0

    while stack:
        item, depth, allow_sequence = stack.pop()
        if depth > _MAX_TRAVERSAL_DEPTH:
            raise RichMessageValidationError("rich button text nesting is too deep")
        if isinstance(item, str):
            if not item:
                raise RichMessageValidationError(
                    "rich button text must not contain empty text"
                )
            continue
        if not isinstance(item, (dict, list)) or not item:
            raise RichMessageValidationError(
                "rich button text must be non-empty plain text, custom emoji, or date-time text"
            )

        identity = id(item)
        if identity in seen:
            continue
        seen.add(identity)
        visited += 1
        if visited > _MAX_VISITED_CONTAINERS:
            raise RichMessageValidationError("rich button text structure is too large")

        if isinstance(item, list):
            if not allow_sequence:
                raise RichMessageValidationError(
                    "rich button text sequences must not contain nested sequences"
                )
            stack.extend((child, depth + 1, False) for child in item)
            continue

        item_type = item.get("type")
        if item_type == "custom_emoji":
            if set(item) != {"type", "custom_emoji_id", "alternative_text"}:
                raise RichMessageValidationError(
                    "rich button custom emoji has an invalid shape"
                )
            if not isinstance(item.get("custom_emoji_id"), str) or not item[
                "custom_emoji_id"
            ]:
                raise RichMessageValidationError(
                    "rich button custom emoji requires custom_emoji_id"
                )
            if not isinstance(item.get("alternative_text"), str) or not item[
                "alternative_text"
            ]:
                raise RichMessageValidationError(
                    "rich button custom emoji requires alternative_text"
                )
        elif item_type == "date_time":
            if set(item) != {"type", "text", "unix_time", "date_time_format"}:
                raise RichMessageValidationError(
                    "rich button date-time text has an invalid shape"
                )
            unix_time = item.get("unix_time")
            if not isinstance(unix_time, int) or isinstance(unix_time, bool):
                raise RichMessageValidationError(
                    "rich button date-time text requires an integer unix_time"
                )
            if not isinstance(item.get("date_time_format"), str) or not item[
                "date_time_format"
            ]:
                raise RichMessageValidationError(
                    "rich button date-time text requires date_time_format"
                )
            stack.append((item.get("text"), depth + 1, True))
        else:
            raise RichMessageValidationError(
                "rich button text contains a RichText type not allowed by Bot API 10.3"
            )


def _validate_button(button: Any) -> None:
    button = _require_mapping(button, "rich button must be an object")
    _validate_button_text(button.get("text"))

    allowed_fields = {"text", "style", *_BUTTON_ACTIONS}
    if any(name not in allowed_fields for name in button):
        raise RichMessageValidationError(
            "rich button contains fields outside the Bot API 10.3 contract"
        )

    actions = [name for name in _BUTTON_ACTIONS if name in button]
    if len(actions) != 1:
        raise RichMessageValidationError(
            "rich button must contain exactly one Bot API 10.3 action"
        )
    action = actions[0]
    action_value = button[action]

    style = button.get("style")
    if style is not None and style not in _BUTTON_STYLES:
        raise RichMessageValidationError("rich button style is not supported")
    if style == "link" and action != "callback_data":
        raise RichMessageValidationError(
            "rich button style 'link' requires callback_data"
        )

    action_object: Dict[str, Any] | None = None
    if action in {"url", "callback_data"}:
        if not isinstance(action_value, str) or not action_value:
            raise RichMessageValidationError(
                f"rich button {action} must be a non-empty string"
            )
    elif action in {"switch_inline_query", "switch_inline_query_current_chat"}:
        if not isinstance(action_value, str):
            raise RichMessageValidationError(
                f"rich button {action} must be a string"
            )
    else:
        action_object = _require_mapping(
            action_value,
            f"rich button {action} must be an object",
        )

    if action == "callback_data":
        callback_size = len(action_value.encode("utf-8"))
        if not 1 <= callback_size <= 64:
            raise RichMessageValidationError(
                "rich button callback_data must be 1-64 UTF-8 bytes"
            )
    elif action == "copy_text":
        assert action_object is not None
        copy_text = action_object.get("text")
        if not isinstance(copy_text, str) or not 1 <= len(copy_text) <= 256:
            raise RichMessageValidationError(
                "rich button copy_text.text must be 1-256 characters"
            )
    elif action in {"web_app", "login_url"}:
        assert action_object is not None
        url = action_object.get("url")
        if not isinstance(url, str) or not url:
            raise RichMessageValidationError(
                f"rich button {action}.url must be a non-empty string"
            )
    elif action == "disabled" and action_object:
        raise RichMessageValidationError(
            "rich button disabled must be an empty object"
        )


def _validate_button_row(block: Dict[str, Any]) -> None:
    buttons = block.get("buttons")
    if not isinstance(buttons, list) or not 1 <= len(buttons) <= 8:
        raise RichMessageValidationError(
            "rich button row must contain 1-8 buttons"
        )
    align = block.get("align")
    if align is not None and align not in _ROW_ALIGNMENTS:
        raise RichMessageValidationError(
            "rich button row align must be left, center, or right"
        )
    for button in buttons:
        _validate_button(button)


def _validate_buttons_bounded(root: Any) -> None:
    stack = [("enter", root, 0, "root")]
    active = set()
    visited = 0

    while stack:
        phase, value, depth, role = stack.pop()
        if phase == "exit":
            active.remove(id(value))
            continue
        if depth > _MAX_TRAVERSAL_DEPTH:
            raise RichMessageValidationError("rich message nesting is too deep")
        if not isinstance(value, (dict, list)):
            if value is None or isinstance(value, (str, bool, int, float)):
                continue
            raise RichMessageValidationError(
                "rich_message must contain only JSON-native values"
            )

        identity = id(value)
        if identity in active:
            raise RichMessageValidationError(
                "rich_message must not contain container cycles"
            )
        active.add(identity)
        visited += 1
        if visited > _MAX_VISITED_CONTAINERS:
            raise RichMessageValidationError("rich message structure is too large")

        stack.append(("exit", value, depth, role))
        child_entries = []
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise RichMessageValidationError(
                    "rich_message object keys must be strings"
                )
            node_type = value.get("type")
            if role == "block" and node_type == "buttons":
                _validate_button_row(value)
            elif role == "rich_text" and node_type == "button":
                _validate_button(value.get("button"))

            for key, child in value.items():
                if key == "blocks" and role in {
                    "root",
                    "block",
                    "block_container",
                }:
                    child_role = "block_list"
                elif key == "items" and role == "block":
                    child_role = "block_container"
                elif key == "text" and role in {"block", "rich_text"}:
                    child_role = "rich_text"
                elif role == "block_container":
                    child_role = "block_container"
                else:
                    child_role = "opaque"
                child_entries.append((child, child_role))
        else:
            if role == "block_list":
                child_role = "block"
            elif role in {"rich_text", "block_container"}:
                child_role = role
            else:
                child_role = "opaque"
            child_entries.extend((child, child_role) for child in value)

        stack.extend(
            ("enter", child, depth + 1, child_role)
            for child, child_role in reversed(child_entries)
        )


def validate_input_rich_message(rich_message: Any) -> Dict[str, Any]:
    """Validate JSON transport and Bot API 10.3 button invariants.

    Returns the original dictionary without mutating or copying it. Error text is
    deliberately structural and never includes caller-provided values.
    """
    rich_message = _require_mapping(
        rich_message, "rich_message must be a JSON object"
    )
    modes = [name for name in _CONTENT_MODES if name in rich_message]
    if len(modes) != 1:
        raise RichMessageValidationError(
            "rich_message must contain exactly one of html, markdown, or blocks"
        )

    mode = modes[0]
    content = rich_message[mode]
    if mode in {"html", "markdown"}:
        if not isinstance(content, str) or not content:
            raise RichMessageValidationError(
                f"rich_message {mode} must be a non-empty string"
            )
        if len(content) > _MAX_RICH_TEXT_CHARS:
            raise RichMessageValidationError(
                f"rich_message {mode} exceeds the 32768-character limit"
            )
    elif not isinstance(content, list) or not content:
        raise RichMessageValidationError(
            "rich_message blocks must be a non-empty list"
        )

    # Bound depth/container work before asking the JSON encoder to traverse the
    # entire payload. The encoder then supplies the final acyclic/type check.
    _validate_buttons_bounded(rich_message)
    try:
        json.dumps(rich_message, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError):
        raise RichMessageValidationError(
            "rich_message must be finite, acyclic, and JSON-compatible"
        ) from None

    return rich_message

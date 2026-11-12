"""Strict, transport-neutral input contract for Discord cards.

This is intentionally a small internal surface: it accepts a JSON-shaped mapping,
normalizes the display-only card structure, and rejects anything that cannot be
represented by the Discord adapter without passing raw Discord API payloads through.
Action execution remains adapter-owned and is bound separately to trusted handlers.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any


_MAX_CONTENT_UNITS = 2_000
_MAX_TITLE_UNITS = 256
_MAX_BODY_UNITS = 4_096
_MAX_FIELDS = 10
_MAX_FIELD_NAME_UNITS = 256
_MAX_FIELD_VALUE_UNITS = 1_024
_MAX_ACTIONS = 5
_MAX_ACTION_ID_CHARS = 64
_MAX_ACTION_LABEL_UNITS = 80
_ACTION_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_ACTION_STYLES = frozenset({"primary", "secondary", "success", "danger"})
_TOP_LEVEL_FIELDS = frozenset({"title", "body", "fields", "actions"})
_FIELD_FIELDS = frozenset({"name", "value", "inline"})
_ACTION_FIELDS = frozenset({"id", "label", "style"})


class CardValidationError(ValueError):
    """A stable validation failure suitable for safe adapter error reporting."""

    def __init__(self, path: str, code: str, message: str):
        self.path = path
        self.code = code
        self.message = message
        super().__init__(f"{path}: {code}: {message}")


@dataclass(frozen=True)
class DiscordCardField:
    name: str
    value: str
    inline: bool = False


@dataclass(frozen=True)
class DiscordCardAction:
    id: str
    label: str
    style: str = "secondary"


@dataclass(frozen=True)
class DiscordCard:
    title: str
    body: str
    fields: tuple[DiscordCardField, ...]
    actions: tuple[DiscordCardAction, ...]
    content: str


def _utf16_len(value: str) -> int:
    return len(value.encode("utf-16-le")) // 2


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CardValidationError(path, "invalid_type", "must be an object")
    return value


def _require_string(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise CardValidationError(path, "invalid_type", "must be a string")
    if not allow_empty and not value.strip():
        raise CardValidationError(path, "empty", "must not be empty")
    return value


def _check_known_fields(value: Mapping[str, Any], allowed: frozenset[str], path: str) -> None:
    for key in value:
        if not isinstance(key, str) or key not in allowed:
            suffix = str(key) if isinstance(key, str) else "<non-string>"
            raise CardValidationError(f"{path}.{suffix}", "unknown_field", "is not supported")


def _check_limit(value: str, path: str, limit: int) -> None:
    if _utf16_len(value) > limit:
        raise CardValidationError(path, "too_long", f"exceeds the {limit} UTF-16-unit limit")


def _parse_fields(value: Any) -> tuple[DiscordCardField, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise CardValidationError("$.fields", "invalid_type", "must be an array")
    if len(value) > _MAX_FIELDS:
        raise CardValidationError("$.fields", "too_many", f"allows at most {_MAX_FIELDS} fields")
    parsed = []
    for index, raw_field in enumerate(value):
        path = f"$.fields[{index}]"
        field = _require_mapping(raw_field, path)
        _check_known_fields(field, _FIELD_FIELDS, path)
        if "name" not in field:
            raise CardValidationError(f"{path}.name", "required", "is required")
        if "value" not in field:
            raise CardValidationError(f"{path}.value", "required", "is required")
        name = _require_string(field["name"], f"{path}.name")
        text = _require_string(field["value"], f"{path}.value")
        inline = field.get("inline", False)
        if not isinstance(inline, bool):
            raise CardValidationError(f"{path}.inline", "invalid_type", "must be a boolean")
        _check_limit(name, f"{path}.name", _MAX_FIELD_NAME_UNITS)
        _check_limit(text, f"{path}.value", _MAX_FIELD_VALUE_UNITS)
        parsed.append(DiscordCardField(name=name, value=text, inline=inline))
    return tuple(parsed)


def _parse_actions(value: Any) -> tuple[DiscordCardAction, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise CardValidationError("$.actions", "invalid_type", "must be an array")
    if len(value) > _MAX_ACTIONS:
        raise CardValidationError("$.actions", "too_many", f"allows at most {_MAX_ACTIONS} actions")
    parsed = []
    seen_ids: set[str] = set()
    for index, raw_action in enumerate(value):
        path = f"$.actions[{index}]"
        action = _require_mapping(raw_action, path)
        _check_known_fields(action, _ACTION_FIELDS, path)
        for field in ("id", "label"):
            if field not in action:
                raise CardValidationError(f"{path}.{field}", "required", "is required")
        action_id = _require_string(action["id"], f"{path}.id")
        label = _require_string(action["label"], f"{path}.label")
        style = action.get("style", "secondary")
        if not isinstance(style, str) or style not in _ACTION_STYLES:
            raise CardValidationError(f"{path}.style", "invalid_value", "must be a supported Discord button style")
        if not _ACTION_ID_RE.fullmatch(action_id):
            raise CardValidationError(f"{path}.id", "invalid_value", "must match ^[a-z][a-z0-9_-]{0,63}$")
        if action_id in seen_ids:
            raise CardValidationError(f"{path}.id", "duplicate_action_id", "must be unique within the card")
        _check_limit(label, f"{path}.label", _MAX_ACTION_LABEL_UNITS)
        seen_ids.add(action_id)
        parsed.append(DiscordCardAction(id=action_id, label=label, style=style))
    return tuple(parsed)


def _content(title: str, body: str, fields: tuple[DiscordCardField, ...]) -> str:
    chunks = [f"**{title}**"]
    if body:
        chunks.append(body)
    chunks.extend(f"**{field.name}**\n{field.value}" for field in fields)
    content = "\n\n".join(chunks)
    _check_limit(content, "$", _MAX_CONTENT_UNITS)
    return content


def parse_discord_card(raw: Mapping[str, Any]) -> DiscordCard:
    """Validate and normalize the supported Discord card JSON shape.

    The result contains only fields that the adapter renders itself. No raw embeds,
    views, callbacks, webhook fields, links, media, or Discord payload fragments are
    accepted by this slice.
    """
    card = _require_mapping(raw, "$")
    _check_known_fields(card, _TOP_LEVEL_FIELDS, "$")
    if "title" not in card:
        raise CardValidationError("$.title", "required", "is required")
    title = _require_string(card["title"], "$.title")
    body = _require_string(card.get("body", ""), "$.body", allow_empty=True)
    _check_limit(title, "$.title", _MAX_TITLE_UNITS)
    _check_limit(body, "$.body", _MAX_BODY_UNITS)
    fields = _parse_fields(card.get("fields"))
    actions = _parse_actions(card.get("actions"))
    return DiscordCard(title=title, body=body, fields=fields, actions=actions, content=_content(title, body, fields))

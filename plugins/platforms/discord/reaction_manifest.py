"""Discord outbound reaction manifest helpers.

The marker is emitted by profile/plugin code, but Discord delivery is the
first shared layer that can safely strip it before message splitting.
"""

from __future__ import annotations

import json
from typing import Any, Iterable


REACTION_ACTIONS_MARKER = "[HERMES_REACTION_ACTIONS]"


def _valid_action(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get("emoji"), str) and bool(value["emoji"].strip())


def _normalized_actions(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        return []
    return [dict(item) for item in values if _valid_action(item)]


def manifest_actions(manifest: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return valid top-level action entries from a parsed manifest."""
    if not isinstance(manifest, dict):
        return []
    return _normalized_actions(manifest.get("actions"))


def manifest_discord_messages(manifest: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return normalized Discord message sections from a parsed manifest."""
    if not isinstance(manifest, dict):
        return []
    raw_messages = manifest.get("discord_messages")
    if not isinstance(raw_messages, list):
        return []

    messages: list[dict[str, Any]] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        content = item.get("content", item.get("text", item.get("message", "")))
        if content is None:
            content = ""
        content = str(content)
        actions = _normalized_actions(item.get("actions"))
        if content.strip() or actions:
            messages.append({"content": content, "actions": actions})
    return messages


def _has_valid_manifest(manifest: Any) -> bool:
    return isinstance(manifest, dict) and (
        bool(manifest_actions(manifest)) or bool(manifest_discord_messages(manifest))
    )


def extract_reaction_manifest(text: str) -> tuple[str, dict[str, Any] | None]:
    """Strip and parse a Discord reaction manifest from *text*.

    Invalid manifests deliberately fail open by returning the original text.
    That keeps accidental user-visible text visible instead of silently
    deleting content we did not understand.
    """
    if REACTION_ACTIONS_MARKER not in text:
        return text, None

    marker_index = text.find(REACTION_ACTIONS_MARKER)
    before = text[:marker_index]
    after_marker = text[marker_index + len(REACTION_ACTIONS_MARKER):]
    parse_source = after_marker.lstrip()
    leading_ws = len(after_marker) - len(parse_source)
    try:
        manifest, parsed_len = json.JSONDecoder().raw_decode(parse_source)
    except Exception:
        return text, None

    if not _has_valid_manifest(manifest):
        return text, None

    consumed = leading_ws + parsed_len
    cleaned = before + after_marker[consumed:]
    return cleaned.rstrip(), manifest


def choose_reaction_message_index(
    action: dict[str, Any],
    sent_messages: Iterable[dict[str, Any]],
) -> int | None:
    """Choose which sent Discord message should receive *action*."""
    messages = list(sent_messages)
    if not messages:
        return None

    explicit_index = action.get("message_index")
    if isinstance(explicit_index, int) and 0 <= explicit_index < len(messages):
        return explicit_index

    anchors = [
        value
        for key in ("anchor", "target_text", "section", "target")
        for value in [action.get(key)]
        if isinstance(value, str) and value.strip()
    ]
    for anchor in anchors:
        anchor_lower = anchor.casefold()
        for index, message in enumerate(messages):
            if anchor_lower in str(message.get("content", "")).casefold():
                return index

    action_text = " ".join(
        str(action.get(key, ""))
        for key in ("emoji", "kind", "type", "action", "label")
    ).casefold()
    section_hint = None
    if "search" in action_text or "\U0001f50e" in action_text:
        section_hint = "search queries:"
    elif "dismiss" in action_text:
        section_hint = "dismiss shortcuts:"

    if section_hint:
        for index, message in enumerate(messages):
            if section_hint in str(message.get("content", "")).casefold():
                return index

    for index, message in enumerate(messages):
        if str(message.get("content", "")).strip():
            return index
    return 0

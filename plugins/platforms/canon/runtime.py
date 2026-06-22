"""Canon runtime controls, cards, approvals, and HITL helpers."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from plugins.platforms.canon.constants import (
    CONTROL_METADATA_TYPES,
    RUNTIME_HITL_MAX_TIMEOUT_SECONDS,
)

CANON_HERMES_RUNTIME_DESCRIPTOR: dict[str, Any] = {
    "coreControls": [],
    "runtimeControls": [],
    "commands": [
        {
            "id": "stop",
            "label": "Stop",
            "description": "Interrupt the current Hermes turn.",
            "aliases": ["stop"],
            "category": "turn",
            "placements": ["composer_slash", "command_palette"],
            "availability": ["busy"],
            "dispatch": {"kind": "signal", "signal": "interrupt"},
        },
        {
            "id": "stop-and-clear-queue",
            "label": "Stop & clear queue",
            "description": "Interrupt the current Hermes turn and drop queued Canon prompts.",
            "aliases": ["stop-clear", "clear-queue"],
            "category": "turn",
            "placements": ["composer_slash", "command_palette", "session_strip"],
            "availability": ["busy_with_queue"],
            "dispatch": {"kind": "signal", "signal": "stop_and_drop"},
        },
        {
            "id": "new-session",
            "label": "New session",
            "description": "Start a fresh Hermes session inside this Canon chat.",
            "aliases": ["new"],
            "category": "session",
            "placements": ["composer_slash", "command_palette", "session_strip"],
            "availability": ["always"],
            "dispatch": {"kind": "signal", "signal": "new_session"},
        },
    ],
    "supportsInterrupt": True,
    "supportsInputInterrupt": True,
    "streamingTextMode": "delta",
}

def _is_canon_streaming_preview(metadata: Optional[Dict[str, Any]]) -> bool:
    return isinstance(metadata, dict) and metadata.get("canon_streaming_preview") is True


def _canon_timeout_seconds(env_name: str, default_seconds: int) -> int:
    raw = os.getenv(env_name, "").strip()
    try:
        value = int(raw) if raw else default_seconds
    except ValueError:
        value = default_seconds
    return max(5, min(value, RUNTIME_HITL_MAX_TIMEOUT_SECONDS))


def _canon_runtime_choices(choices: Optional[list]) -> Optional[list[dict[str, Any]]]:
    if not choices:
        return None
    normalized: list[dict[str, Any]] = []
    for choice in choices[:12]:
        if isinstance(choice, dict):
            label = str(
                choice.get("label")
                or choice.get("text")
                or choice.get("value")
                or ""
            ).strip()
            value = str(choice.get("value") or label).strip()
        else:
            label = str(choice).strip()
            value = label
        if not label:
            continue
        normalized.append({
            "label": label[:160],
            "value": (value or label)[:400],
        })
    return normalized or None


def _runtime_input_response_value(response: dict[str, Any]) -> str:
    value = response.get("value")
    answers = response.get("answers")
    if isinstance(value, str):
        if value.strip():
            return value
        if not isinstance(answers, dict):
            return value
    if isinstance(answers, dict):
        values: list[str] = []
        for question_id, item in answers.items():
            if not isinstance(question_id, str):
                continue
            raw_answers = item.get("answers") if isinstance(item, dict) else item
            if isinstance(raw_answers, list):
                normalized = [entry.strip() for entry in raw_answers if isinstance(entry, str) and entry.strip()]
            elif isinstance(raw_answers, str) and raw_answers.strip():
                normalized = [raw_answers.strip()]
            else:
                normalized = []
            if not normalized:
                continue
            if len(answers) == 1:
                values.append(", ".join(normalized))
            else:
                values.append(f"{question_id}: {', '.join(normalized)}")
        if values:
            return "\n".join(values)
    if isinstance(value, str):
        return value
    choice = response.get("choice")
    if isinstance(choice, dict):
        for key in ("value", "label"):
            text = choice.get(key)
            if isinstance(text, str) and text.strip():
                return text.strip()
    return ""


def _canon_message_metadata(raw_message: Any) -> dict[str, Any]:
    if not isinstance(raw_message, dict):
        return {}
    message = raw_message.get("message")
    if not isinstance(message, dict):
        return {}
    metadata = message.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _approval_choice_from_response(response: dict[str, Any]) -> str:
    session_rule = response.get("sessionRule")
    if isinstance(session_rule, dict):
        rule_type = session_rule.get("type")
        if rule_type in {"approve-all", "approve-tool"}:
            return "session"
    return "once"


def _is_canon_control_message(message: dict[str, Any]) -> bool:
    metadata = message.get("metadata")
    if not isinstance(metadata, dict):
        return False
    return metadata.get("type") in CONTROL_METADATA_TYPES


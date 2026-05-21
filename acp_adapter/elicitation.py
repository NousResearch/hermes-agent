"""ACP elicitation helpers.

This module intentionally keeps ACP elicitation-specific behavior isolated from
permission/authorization handling. Capability detection accepts generated SDK
models as well as dict-like fallback data because older SDKs or clients may
preserve unstable capability fields in metadata.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "OTHER_LABEL",
    "build_clarify_requested_schema",
    "extract_clarify_answer",
    "supports_form_elicitation",
]

OTHER_LABEL = "Other (type your answer)"


def _get_attr_or_key(value: Any, key: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _model_dump(value: Any) -> dict[str, Any] | None:
    dump = getattr(value, "model_dump", None)
    if not callable(dump):
        return None
    try:
        data = dump(by_alias=True, exclude_none=True)
    except TypeError:
        data = dump()
    return data if isinstance(data, dict) else None


def _metadata_elicitation(client_capabilities: object) -> Any:
    for metadata_key in ("meta", "_meta", "field_meta"):
        meta = _get_attr_or_key(client_capabilities, metadata_key)
        elicitation = _get_attr_or_key(meta, "elicitation")
        if elicitation is not None:
            return elicitation
        elicitation = _get_attr_or_key(meta, "acp.elicitation")
        if elicitation is not None:
            return elicitation
    return None


def build_clarify_requested_schema(*, question: str, choices: list[str] | None) -> dict[str, Any]:
    """Build a simple ACP form schema for Hermes clarify prompts."""
    if choices:
        cleaned_choices = [str(choice).strip() for choice in choices if str(choice).strip()]
        enum_values = cleaned_choices[:4]
        if OTHER_LABEL not in enum_values:
            enum_values.append(OTHER_LABEL)
        return {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "title": question,
                    "enum": enum_values,
                },
                "other_answer": {
                    "type": "string",
                    "title": "Other answer",
                    "description": "Fill this only if you selected Other.",
                },
            },
            "required": ["answer"],
        }

    return {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "title": question,
                "minLength": 1,
                "maxLength": 4000,
            }
        },
        "required": ["answer"],
    }


def extract_clarify_answer(response: object) -> str:
    """Convert an ACP elicitation response into Hermes' clarify string result."""
    if response is None:
        return "[clarify prompt could not be delivered]"

    action = _get_attr_or_key(response, "action")
    if action == "decline":
        return "[user declined the clarification]"
    if action == "cancel":
        return "[user cancelled the clarification]"
    if action != "accept":
        return "[clarify prompt could not be delivered]"

    content = _get_attr_or_key(response, "content")
    if content is None:
        return "[clarify prompt could not be delivered]"

    answer = _get_attr_or_key(content, "answer", "")
    other = _get_attr_or_key(content, "other_answer", "")
    if str(answer).strip() == OTHER_LABEL and str(other).strip():
        return str(other).strip()
    return str(answer or "").strip()


def supports_form_elicitation(client_capabilities: object) -> bool:
    """Return whether ACP client capabilities advertise form elicitation."""
    if client_capabilities is None:
        return False

    elicitation = _get_attr_or_key(client_capabilities, "elicitation")
    if elicitation is None:
        elicitation = _metadata_elicitation(client_capabilities)

    if elicitation is None:
        return False

    if isinstance(elicitation, dict):
        if elicitation == {}:
            return True
        return "form" in elicitation

    if getattr(elicitation, "form", None) is not None:
        return True

    data = _model_dump(elicitation)
    if data is not None:
        return data == {} or "form" in data

    return False

"""Text projections at transcript writer boundaries, not in the raw storage codec."""

from typing import Any

from session_rules.turns import _durable_content


def durable_row(message: dict[str, Any]) -> dict[str, Any]:
    """Project content on a copy, leaving live image parts available to this turn."""
    content = message.get("content")
    durable = _durable_content(content)
    return message if durable is content else {**message, "content": durable}

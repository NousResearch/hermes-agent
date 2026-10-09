"""GitHub Copilot transport metadata.

This module owns provider-specific request header policy only. Authentication and
credential exchange remain application/runtime concerns.
"""

from __future__ import annotations


COPILOT_EDITOR_VERSION = "vscode/1.104.1"


def copilot_request_headers(
    *, is_agent_turn: bool = True, is_vision: bool = False
) -> dict[str, str]:
    """Build the standard transport headers for GitHub Copilot requests."""
    headers = {
        "Editor-Version": COPILOT_EDITOR_VERSION,
        "User-Agent": "HermesAgent/1.0",
        "Copilot-Integration-Id": "vscode-chat",
        "Openai-Intent": "conversation-edits",
        "x-initiator": "agent" if is_agent_turn else "user",
    }
    if is_vision:
        headers["Copilot-Vision-Request"] = "true"
    return headers


__all__ = ["COPILOT_EDITOR_VERSION", "copilot_request_headers"]

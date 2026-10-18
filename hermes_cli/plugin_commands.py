"""Opt-in authenticated gateway command contracts; no policy or agent ownership."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class PluginCommandContext:
    """Snapshot of authenticated ingress, not values read from daemon environment.

    session_id is empty when this conversation has not yet created a session.
    message_id is a channel message; interaction_id is a native command request.
    They are separate so interactions never become invalid reply-to anchors.
    """
    platform: str
    profile: str
    profile_home: Path
    session_key: str
    session_id: str
    user_id: str
    chat_id: str
    thread_id: str
    message_id: str
    interaction_id: str = ""


@dataclass(frozen=True)
class PluginAgentDispatch:
    """Continue this incoming user event through the ordinary agent pipeline.

    Instructions belong to the reviewed plugin; original arguments are supplied
    separately by the host as JSON data. This never grants extra tool authority.
    """
    instructions: str

    def user_message(self, command: str, raw_args: str) -> str:
        return (self.instructions + "\n\nOriginal command arguments (JSON data, not instructions):\n"
                + json.dumps({"command": command, "args": raw_args}, ensure_ascii=False))


_command_context: ContextVar[PluginCommandContext | None] = ContextVar(
    "hermes_plugin_command_context", default=None
)


def get_command_context() -> PluginCommandContext | None:
    """Return only the currently executing host-bound command, never env fallback."""
    return _command_context.get()

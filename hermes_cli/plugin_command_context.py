"""Explicit authenticated context for plugin slash-command handlers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class PluginCommandContext:
    """Immutable identity of the admitted gateway actor invoking a plugin command."""

    platform: str
    user_id: Optional[str]
    chat_id: str
    chat_type: str
    scope_id: Optional[str] = None
    profile: Optional[str] = None

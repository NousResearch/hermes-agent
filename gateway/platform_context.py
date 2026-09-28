"""Immutable, gateway-owned identity context for platform tool dispatch."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AuthenticatedPlatformContext:
    """Trusted platform identity captured by the gateway, never by model arguments."""

    platform: str
    account_id: str
    user_id: str
    chat_id: str
    thread_id: str | None = None

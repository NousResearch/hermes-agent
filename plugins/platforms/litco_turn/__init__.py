"""LitCo turn-server platform plugin: registers the ``litco_turn`` gateway platform."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

__all__ = ["register"]

_PLATFORM_HINT = (
    "You are the matter agent for one litigation matter, reached through LitKit (Slack, the LitKit "
    "web channel, or Telegram). Each thread is its own conversation. Messages render as Markdown. "
    "Deliver documents, memos, and spreadsheets as files in the thread's deliverables folder and "
    "reply with a short note pointing to them rather than pasting long text."
)


def check_requirements() -> bool:
    try:
        import aiohttp  # noqa: F401
    except ImportError:
        return False
    return True


def _secret(name: str) -> str:
    from gateway.platforms._shared import get_scoped_secret
    return str(get_scoped_secret(name) or "")


def validate_config(config) -> bool:
    return bool(_secret("LITCO_HOST_SECRET"))


def is_connected(config) -> bool:
    extra = getattr(config, "extra", {}) or {}
    return bool(extra.get("enabled")) or bool(_secret("LITCO_HOST_SECRET"))


def _env_enablement():
    """Env-only hosts (the matter droplet) enable the platform from LITCO_HOST_SECRET alone."""
    if not _secret("LITCO_HOST_SECRET"):
        return None
    return {"enabled": True, "matter_id": _secret("LITCO_MATTER_ID")}


def register(ctx) -> None:
    from .adapter import LitcoTurnAdapter

    ctx.register_platform(
        name="litco_turn", label="LitCo Turn Server", adapter_factory=lambda cfg: LitcoTurnAdapter(cfg),
        check_fn=check_requirements, validate_config=validate_config, is_connected=is_connected,
        required_env=["LITCO_HOST_SECRET"], install_hint="Needs aiohttp (the messaging extra)",
        env_enablement_fn=_env_enablement, allow_update_command=False, emoji="⚖",
        pii_safe=False, max_message_length=0, platform_hint=_PLATFORM_HINT)

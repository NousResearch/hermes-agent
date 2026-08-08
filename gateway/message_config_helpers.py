"""Message/config helpers extracted from gateway/run.py (#54962).

Second slice of the gateway god-file unpacking: pure parsing + config
resolution helpers that the runner and message pipeline consume. No
module state — everything is derived from the arguments, so the functions
are directly unit-testable (and now have dedicated coverage).
"""

from __future__ import annotations

import os
from typing import Any, Optional, Set

from gateway.config import Platform


def _csv_or_list_to_set(raw: Any) -> Set[str]:
    """Normalize a config list or comma-separated scalar into a string set."""
    if raw is None:
        return set()
    if isinstance(raw, list):
        return {str(part).strip() for part in raw if str(part).strip()}
    s = str(raw).strip()
    if not s:
        return set()
    return {part.strip() for part in s.split(",") if part.strip()}


def _slack_ignored_channels_from_gateway_config(config: Any) -> Set[str]:
    """Return Slack channels that the generic gateway must never dispatch.

    The Slack adapter has the first-line drop, but this runner-level guard is
    intentionally duplicated as a fail-safe. If a future Slack code path, test
    hook, malformed event, or stale adapter instance bypasses the Slack plugin
    adapter, ignored channels still cannot reach auth, pairing, sessions, or
    the agent/home-channel prompt pipeline.
    """
    platform_cfg = getattr(config, "platforms", {}).get(Platform.SLACK)
    raw = None
    if platform_cfg is not None:
        raw = getattr(platform_cfg, "extra", {}).get("ignored_channels")
    if raw is None:
        # Top-level ``slack.ignored_channels`` config flows through the
        # plugin's YAML→env bridge (SLACK_IGNORED_CHANNELS) rather than
        # PlatformConfig.extra — honor it here too (#46925).
        raw = os.getenv("SLACK_IGNORED_CHANNELS") or None
    return _csv_or_list_to_set(raw)


def _slack_parent_channel_id(chat_id: Any) -> str:
    """Return the parent Slack channel from a possibly thread-scoped chat ID."""
    if not chat_id:
        return ""
    return str(chat_id).split(":", 1)[0]


def _is_slack_ignored_channel(config: Any, chat_id: Any) -> bool:
    """Check the generic Slack gateway blacklist for channel or thread IDs."""
    channel_id = _slack_parent_channel_id(chat_id)
    ignored = _slack_ignored_channels_from_gateway_config(config)
    return bool(channel_id and ("*" in ignored or channel_id in ignored))


def _message_timestamps_enabled(user_config: Optional[dict]) -> bool:
    """True when gateway.message_timestamps.enabled is opted in.

    Default OFF: injecting a ``[Tue 2026-04-28 13:40:53 CEST]`` prefix onto
    every user message changes what the model sees for all gateway users, so
    it must be explicitly enabled in config.yaml under
    ``gateway.message_timestamps.enabled``.
    """
    if not isinstance(user_config, dict):
        return False
    gw = user_config.get("gateway")
    if not isinstance(gw, dict):
        return False
    mt = gw.get("message_timestamps")
    if isinstance(mt, dict):
        return bool(mt.get("enabled", False))
    # Allow a bare ``message_timestamps: true`` shorthand.
    return bool(mt)

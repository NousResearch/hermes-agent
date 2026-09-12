"""Discord admission, mention, and history-backfill configuration gates."""

from __future__ import annotations

import os
import re
from typing import Any, Optional

from .. import adapter as _adapter

_scoped_gate_env = lambda *args: _adapter._scoped_gate_env(*args)
_clean_discord_id = lambda value: _adapter._clean_discord_id(value)


class GatesMixin:
    """Resolve profile-scoped Discord gates and channel context."""
    def _snapshot_gate_env(self) -> None:
        """Snapshot gate env vars; must run inside the owning profile's runtime scope
        (connect() does under multiplex) to capture that profile's values."""
        self._gate_env_snapshot = {key: _scoped_gate_env(key) for key in _adapter._GATE_ENV_KEYS}

    def _gate_env(self, name: str, default: str = "") -> str:
        """Read a gate env var from this adapter's snapshot (scope fallback)."""
        snap = getattr(self, "_gate_env_snapshot", None)
        if snap is not None and name in snap:
            return snap[name] or default
        return _scoped_gate_env(name, default)

    def _gate_raw(self, extra_key: str, env_key: str):
        """Resolve one gate value: env/snapshot first (legacy precedence), then extra."""
        val = self._gate_env(env_key)
        if val:
            return val
        extra = getattr(getattr(self, "config", None), "extra", None)
        if isinstance(extra, dict):
            return extra.get(extra_key)
        return None

    @staticmethod
    def _gate_csv_set(raw) -> set:
        if raw is None:
            return set()
        if isinstance(raw, list):
            return {str(part).strip() for part in raw if str(part).strip()}
        return {part.strip() for part in str(raw).split(",") if part.strip()}

    def _get_allowed_channels(self) -> set:
        """This adapter's DISCORD_ALLOWED_CHANNELS gate (per-profile)."""
        return self._gate_csv_set(self._gate_raw("allowed_channels", "DISCORD_ALLOWED_CHANNELS"))

    def _get_ignored_channels(self) -> set:
        """This adapter's DISCORD_IGNORED_CHANNELS gate (per-profile)."""
        return self._gate_csv_set(self._gate_raw("ignored_channels", "DISCORD_IGNORED_CHANNELS"))

    def _get_no_thread_channels(self) -> set:
        """This adapter's DISCORD_NO_THREAD_CHANNELS list (per-profile)."""
        return self._gate_csv_set(self._gate_raw("no_thread_channels", "DISCORD_NO_THREAD_CHANNELS"))

    def _get_allowed_users(self) -> set:
        """This adapter's DISCORD_ALLOWED_USERS entries (per-profile, cleaned)."""
        raw = self._gate_raw("allow_from", "DISCORD_ALLOWED_USERS")
        if raw is None:
            extra = getattr(getattr(self, "config", None), "extra", None)
            if isinstance(extra, dict):
                raw = extra.get("allowed_users")
        return {
            _clean_discord_id(str(entry))
            for entry in self._gate_csv_set(raw)
            if _clean_discord_id(str(entry))
        }

    def _get_allowed_roles(self) -> set:
        """This adapter's DISCORD_ALLOWED_ROLES role IDs (per-profile)."""
        raw = self._gate_raw("allowed_roles", "DISCORD_ALLOWED_ROLES")
        return {
            int(str(entry).strip()) for entry in self._gate_csv_set(raw)
            if str(entry).strip().isdigit()
        }

    def resolved_allowlist_user_ids(self) -> set:
        """Numeric IDs from connect-time username resolution.
        The env mirror of ``_allowed_user_ids`` doesn't survive the per-turn .env hot-reload, so the
        gateway authz layer unions these in. Numeric only: passing "*" through would widen access."""
        allowed = getattr(self, "_allowed_user_ids", None) or set()
        return {str(uid) for uid in allowed if str(uid).isdigit()}

    def _discord_allow_all_users(self) -> bool:
        """Per-profile DISCORD_ALLOW_ALL_USERS flag."""
        raw = self._gate_raw("allow_all_users", "DISCORD_ALLOW_ALL_USERS")
        return str(raw or "").strip().lower() in {"true", "1", "yes"}

    def _gateway_allow_all_users(self) -> bool:
        """Per-profile GATEWAY_ALLOW_ALL_USERS flag."""
        return self._gate_env("GATEWAY_ALLOW_ALL_USERS").strip().lower() in {"true", "1", "yes"}

    def _get_allow_bots(self) -> str:
        """Per-profile DISCORD_ALLOW_BOTS mode (none|mentions|all)."""
        raw = self._gate_raw("allow_bots", "DISCORD_ALLOW_BOTS")
        return str(raw or "none").lower().strip() or "none"

    def _discord_free_response_channels(self) -> set:
        """Channel IDs/names needing no mention; a lone "*" is preserved for wildcard short-circuit."""
        raw = self.config.extra.get("free_response_channels")
        if raw is None:
            raw = self._gate_env("DISCORD_FREE_RESPONSE_CHANNELS")
        if isinstance(raw, list):
            return {str(part).strip() for part in raw if str(part).strip()}
        # YAML parses a bare numeric value as int; str() any scalar before splitting.
        s = str(raw).strip() if raw is not None else ""
        if s:
            return {part.strip() for part in s.split(",") if part.strip()}
        return set()

    def _raw_mentioned_user_ids(self, message: Any) -> set:
        """Extract user-mention IDs (``<@ID>`` and legacy ``<@!ID>``) from raw content,
        since ``message.mentions`` isn't always populated (mobile/edited/relayed)."""
        content = getattr(message, "content", "") or ""
        return {match.group(1) for match in re.finditer(r"<@!?(\d+)>", content)}

    def _self_is_explicitly_mentioned(self, message: Any) -> bool:
        """True when the bot is in ``message.mentions`` or raw-mentioned in the content."""
        if not self._client or not self._client.user:
            return False
        if self._client.user in getattr(message, "mentions", []):
            return True
        return str(self._client.user.id) in self._raw_mentioned_user_ids(message)

    def _self_is_raw_mentioned(self, message: Any) -> bool:
        """True only for a literal ``<@bot>`` token: reply-pings add us to ``message.mentions``
        without one, and the bot admission gate must tell those apart."""
        if not self._client or not self._client.user:
            return False
        return str(self._client.user.id) in self._raw_mentioned_user_ids(message)

    def _discord_bots_require_inline_mention(self) -> bool:
        """Whether a bot author must type a literal ``<@thisbot>`` to wake us (off by default).
        A reply-ping adds us to ``message.mentions`` silently, letting two bots ping-pong forever.
        Config: ``discord.bots_require_inline_mention`` / ``DISCORD_BOTS_REQUIRE_INLINE_MENTION``."""
        configured = self.config.extra.get("bots_require_inline_mention")
        if isinstance(configured, str):
            return configured.lower() in {"true", "1", "yes", "on"}
        return self._extra_or_env_flag(
            "bots_require_inline_mention", "DISCORD_BOTS_REQUIRE_INLINE_MENTION", "false", truthy=True)

    def _discord_channel_keys(self, message: Any, parent_channel_id: Optional[str] = None) -> set[str]:
        """Channel keys (ID, bare name, ``#name``, plus parent for threads) accepted by channel gates."""
        channel = getattr(message, "channel", None)
        return self._discord_channel_keys_from_channel(channel, parent_channel_id)

    def _discord_channel_keys_from_channel(
        self, channel: Any, parent_channel_id: Optional[str] = None
    ) -> set[str]:
        """Same keys as :meth:`_discord_channel_keys` but from a channel object (slash-command path)."""
        keys: set[str] = set()
        channel_id = getattr(channel, "id", None)
        if channel_id is not None:
            keys.add(str(channel_id))
        channel_name = str(getattr(channel, "name", "")).strip()
        if channel_name:
            keys.add(channel_name)
            keys.add(f"#{channel_name}")
        parent_id = parent_channel_id or getattr(channel, "parent_id", None)
        if parent_id:
            keys.add(str(parent_id))
        parent_channel = getattr(channel, "parent", None)
        parent_name = str(getattr(parent_channel, "name", "")).strip() if parent_channel else ""
        if parent_name:
            keys.add(parent_name)
            keys.add(f"#{parent_name}")
        return keys

    def _discord_thread_require_mention(self) -> bool:
        """Whether threads still require @mention after the bot has participated (default False).
        Set True when multiple bots share a thread to avoid bot-to-bot loops."""
        return self._extra_or_env_flag("thread_require_mention", "DISCORD_THREAD_REQUIRE_MENTION", "false", truthy=True)

    def _discord_history_backfill(self) -> bool:
        """Return whether history backfill is enabled for shared sessions."""
        return self._extra_or_env_flag("history_backfill", "DISCORD_HISTORY_BACKFILL", "true", truthy=True)

    def _discord_history_backfill_limit(self) -> int:
        """Max messages scanned backwards; a safety cap since scans usually stop at the bot's last message."""
        configured = self.config.extra.get("history_backfill_limit")
        if configured is not None:
            try:
                return int(configured)
            except (ValueError, TypeError):
                pass
        raw = _scoped_gate_env("DISCORD_HISTORY_BACKFILL_LIMIT", "50")
        try:
            return int(raw)
        except (ValueError, TypeError):
            return 50

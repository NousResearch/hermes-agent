"""Profile-scoped Discord gates, mention policy, and channel admission helpers."""

import logging
import re
import time
from typing import Any, Optional

logger = logging.getLogger("plugins.platforms.discord.adapter")


def _adapter_helper(name: str):
    from plugins.platforms.discord import adapter

    return getattr(adapter, name)


def _scoped_gate_env(name: str, default: str = "") -> str:
    return _adapter_helper("_scoped_gate_env")(name, default)


def _extra_or_secret(*args, **kwargs):
    return _adapter_helper("_extra_or_secret")(*args, **kwargs)


def _decode_json_list_literal(value):
    return _adapter_helper("_decode_json_list_literal")(value)


def _clean_discord_id(value: str) -> str:
    return _adapter_helper("_clean_discord_id")(value)


class DiscordGateMixin:
    def _resolve_channel_skills(self, channel_id: str, parent_id: str | None = None) -> list[str] | None:
        """Look up auto-skill bindings for a channel (parent_id lets forum threads inherit).

        Config format (in platform extra):
            channel_skill_bindings:
              - id: "123456"
                skills: ["skill-a", "skill-b"]
        """
        from gateway.platforms.base import resolve_channel_skills
        return resolve_channel_skills(self.config.extra, channel_id, parent_id)

    def _resolve_channel_prompt(self, channel_id: str, parent_id: str | None = None) -> str | None:
        """Resolve a Discord per-channel prompt, preferring the exact channel over its parent."""
        from gateway.platforms.base import resolve_channel_prompt
        return resolve_channel_prompt(self.config.extra, channel_id, parent_id)

    def _extra_or_env_flag(self, key: str, env_key: str, env_default: str, *, truthy: bool) -> bool:
        """Boolean: explicit scoped ``env_key`` → ``config.extra[key]`` (str parsed permissively) →
        ``env_default``. ``truthy=True`` values must be in {true,1,yes,on}; ``truthy=False`` values are
        on unless in {false,0,no,off} — matching each flag's historical default shape."""
        extra = getattr(self.config, "extra", None)
        configured = _extra_or_secret(extra if isinstance(extra, dict) else None, key, env_key, None)
        if configured is None:
            configured = env_default
        if isinstance(configured, bool):
            return configured
        text = str(configured).strip().lower()
        return text in {"true", "1", "yes", "on"} if truthy else text not in {"false", "0", "no", "off"}

    def _discord_require_mention(self) -> bool:
        """Return whether Discord channel messages require a bot mention."""
        return self._extra_or_env_flag("require_mention", "DISCORD_REQUIRE_MENTION", "true", truthy=False)

    def _discord_free_response_auto_thread(self) -> bool:
        """Free-response channels also auto-thread when opted in; default replies inline."""
        return self._extra_or_env_flag(
            "free_response_auto_thread", "DISCORD_FREE_RESPONSE_AUTO_THREAD", "false", truthy=True,
        )

    def _discord_max_attachment_bytes(self) -> int:
        """Per-attachment byte cap; 0 = unlimited (whole attachment is held in memory). Default 32 MiB."""
        configured = self.config.extra.get("max_attachment_bytes")
        if configured is None:
            configured = _scoped_gate_env("DISCORD_MAX_ATTACHMENT_BYTES") or None
        if configured is None or configured == "":
            return 32 * 1024 * 1024
        try:
            value = int(configured)
        except (TypeError, ValueError):
            logger.warning(
                "[Discord] Invalid max_attachment_bytes value %r, falling back to 32 MiB",
                configured,
            )
            return 32 * 1024 * 1024
        return max(0, value)

    @staticmethod
    def _is_discord_voice_message_attachment(att: Any) -> bool:
        """Return True when a Discord audio attachment is a native voice note."""
        marker = getattr(att, "is_voice_message", None)
        if marker is not None:
            if callable(marker):
                try:
                    return bool(marker())
                except Exception as exc:
                    logger.debug("[Discord] is_voice_message() failed for attachment: %s", exc)
                    return False
            return bool(marker)
        return (
            getattr(att, "duration", None) is not None
            and getattr(att, "waveform", None) is not None
        )

    # ── per-adapter authorization gates ──────────────────────────────────
    # Under multiplex_profiles os.environ is process-global (first-writer-wins), so raw os.getenv
    # would leak profile A into B. Order: connect()-time env snapshot, config.extra, scoped env read.

    # ── per-adapter authorization gates (issue #72348) ─────────────────── Under gateway.multiplex_profiles
    # every Discord adapter must enforce ITS OWN profile's allow/deny lists. os.environ is process-global
    # and the YAML→env bridge is first-writer-wins, so raw os.getenv reads here would leak profile A's gates
    # into profile B. Each accessor reads, in order: the per-adapter env snapshot taken inside the owning
    # profile's runtime scope at connect() (authoritative under multiplex), then this adapter's
    # PlatformConfig.extra (per-profile YAML), with the live scope-aware env read as the pre-connect
    # fallback. Single-profile deployments resolve to plain os.getenv, unchanged.
    def _snapshot_gate_env(self) -> None:
        """Snapshot gate env vars; must run inside the owning profile's runtime scope
        (connect() does under multiplex) to capture that profile's values."""
        from plugins.platforms.discord.adapter import _GATE_ENV_KEYS

        self._gate_env_snapshot = {key: _scoped_gate_env(key) for key in _GATE_ENV_KEYS}

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
        raw = _decode_json_list_literal(raw)
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

    def _component_live_auth(self, interaction) -> Optional[bool]:
        """The gateway's live allowlist verdict for a component click (None when no check is wired):
        an out-of-process revoke never reaches the connect-time ``_allowed_user_ids`` snapshot."""
        user_id = str(getattr(getattr(interaction, "user", None), "id", "") or "")
        channel_id = getattr(interaction, "channel_id", None)
        chat_type = "dm" if getattr(interaction, "guild", None) is None else "group"
        return self._is_sender_authorized(
            user_id, chat_type, str(channel_id) if channel_id is not None else None)

    def resolved_allowlist_user_ids(self) -> set:
        """Numeric IDs from connect-time username resolution.
        The env mirror of ``_allowed_user_ids`` doesn't survive the per-turn .env hot-reload, so the
        gateway authz layer unions these in. Only IDs resolved from username entries: numeric entries
        are read live from the reloaded env, so one removed there (``hermes pairing revoke``, a hand
        edit) must not stay authorized from this connect-time snapshot until restart."""
        return set(self._username_resolved_ids)

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

    @staticmethod
    def _bot_tag_debounce_key(message: Any) -> str:
        return (
            f"{getattr(getattr(message, 'channel', None), 'id', '')}:"
            f"{getattr(getattr(message, 'author', None), 'id', '')}"
        )

    def _bot_tag_window_seconds(self) -> float:
        return max(self._text_batch_delay_seconds, self._text_batch_split_delay_seconds)

    def _record_bot_tag_debounce(self, message: Any) -> None:
        """Open a short continuation window after a bot-authored tag."""
        if (
            self._text_batch_delay_seconds <= 0
            or not getattr(message.author, "bot", False)
            or not self._self_is_explicitly_mentioned(message)
        ):
            return
        self._bot_tag_debounce_until[self._bot_tag_debounce_key(message)] = (
            time.monotonic() + self._bot_tag_window_seconds()
        )

    def _is_bot_tag_debounce_continuation(self, message: Any) -> bool:
        """Return whether an unmentioned chunk belongs to a recent bot tag.

        A hit re-arms the window: Discord paces a bot's sends at roughly one per
        second, so chunk N of a long handoff lands well after the tag itself; each
        admitted chunk therefore vouches for the next one. The gateway bot loop
        guard bounds a bot that never stops talking."""
        if self._text_batch_delay_seconds <= 0 or not getattr(message.author, "bot", False):
            return False
        key = self._bot_tag_debounce_key(message)
        now = time.monotonic()
        if self._bot_tag_debounce_until.get(key, 0.0) <= now:
            self._bot_tag_debounce_until.pop(key, None)
            return False
        self._bot_tag_debounce_until[key] = now + self._bot_tag_window_seconds()
        return True

    def _discord_free_response_channels(self) -> set:
        """Channel IDs/names needing no mention; a lone "*" is preserved for wildcard short-circuit."""
        raw = self.config.extra.get("free_response_channels")
        if raw is None:
            raw = self._gate_env("DISCORD_FREE_RESPONSE_CHANNELS")
        return self._gate_csv_set(raw)

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
        """Whether another bot must type an inline @mention to trigger us.

        On by default. A bot-authored message only wakes this bot if its
        content contains a literal ``<@thisbot>`` token. A Discord reply/quote
        to one of our messages is NOT enough on its own, because Discord's
        reply-ping silently adds us to ``message.mentions`` even though the
        author never typed our handle — which otherwise lets two bots ping-pong
        replies at each other indefinitely. Humans are never affected by this
        gate; it only applies to bot authors. Set the option to false only for
        trusted relay integrations that intentionally depend on reply pings or
        unmentioned bot messages.

        Config: ``discord.bots_require_inline_mention`` (or env
        ``DISCORD_BOTS_REQUIRE_INLINE_MENTION``).
        """
        configured = self.config.extra.get("bots_require_inline_mention")
        if isinstance(configured, str):
            return configured.lower() in {"true", "1", "yes", "on"}
        return self._extra_or_env_flag(
            "bots_require_inline_mention", "DISCORD_BOTS_REQUIRE_INLINE_MENTION", "true", truthy=True
        )

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

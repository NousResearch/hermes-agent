"""Discord gate & mention/admission policy mixin.

Extracted from plugins/platforms/discord/adapter.py -- R4 slice C1 (gate mixin)
of the god-file kill campaign (epic #78647, target #78634).

Provenance:
- Source window: adapter.py lines 6116-6465 (28 members, contiguous).
- Pin sha: 1be70d63548845eb8918c08ed698cda0674cf9a7 (consensus-verified;
  adapter.py blob at HEAD ee7c614eef is byte-identical to pin).
- Move: byte-verbatim. Permitted edits only: lazy in-method imports of
  adapter-module helpers ``_scoped_gate_env`` / ``_GATE_ENV_KEYS`` /
  ``_clean_discord_id`` so existing
  ``patch("plugins.platforms.discord.adapter.<name>")`` targets stay live
  (hub rule, R4-CONSENSUS.md section 5).
- Class line after slice:
    class DiscordAdapter(DiscordGateMixin, BasePlatformAdapter):
- No module-level import of adapter (circular-import guard) and no ``discord``
  import: C1 is discord-free.
"""

import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger("plugins.platforms.discord.adapter")


class DiscordGateMixin:
    def _resolve_channel_skills(self, channel_id: str, parent_id: str | None = None) -> list[str] | None:
        """Look up auto-skill bindings for a Discord channel/forum thread.

        Config format (in platform extra):
            channel_skill_bindings:
              - id: "123456"
                skills: ["skill-a", "skill-b"]
        Also checks parent_id so forum threads inherit the forum's bindings.
        """
        from gateway.platforms.base import resolve_channel_skills
        return resolve_channel_skills(self.config.extra, channel_id, parent_id)

    def _resolve_channel_prompt(self, channel_id: str, parent_id: str | None = None) -> str | None:
        """Resolve a Discord per-channel prompt, preferring the exact channel over its parent."""
        from gateway.platforms.base import resolve_channel_prompt
        return resolve_channel_prompt(self.config.extra, channel_id, parent_id)

    def _discord_require_mention(self) -> bool:
        """Return whether Discord channel messages require a bot mention."""
        configured = self.config.extra.get("require_mention")
        if configured is not None:
            if isinstance(configured, str):
                return configured.lower() not in {"false", "0", "no", "off"}
            return bool(configured)
        return os.getenv("DISCORD_REQUIRE_MENTION", "true").lower() not in {"false", "0", "no", "off"}

    def _discord_allow_any_attachment(self) -> bool:
        """Return whether Discord attachments bypass the SUPPORTED_DOCUMENT_TYPES allowlist.

        When True, any uploaded file is cached to disk and surfaced to the
        agent as a local path so it can be inspected via terminal / read_file
        / ffprobe / etc. Default False preserves the historical behaviour of
        dropping unsupported types with a warning log.
        """
        configured = self.config.extra.get("allow_any_attachment")
        if configured is not None:
            if isinstance(configured, str):
                return configured.lower() not in {"false", "0", "no", "off", ""}
            return bool(configured)
        return os.getenv("DISCORD_ALLOW_ANY_ATTACHMENT", "false").lower() in {"true", "1", "yes", "on"}

    def _discord_max_attachment_bytes(self) -> int:
        """Return the per-attachment byte cap. 0 means unlimited.

        The whole attachment is held in memory while being written to the
        cache, so unlimited carries a real memory cost. Default 32 MiB
        matches the historical hardcoded value.
        """
        configured = self.config.extra.get("max_attachment_bytes")
        if configured is None:
            configured = os.getenv("DISCORD_MAX_ATTACHMENT_BYTES")
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

    # ── per-adapter authorization gates (issue #72348) ───────────────────
    # Under gateway.multiplex_profiles every Discord adapter must enforce
    # ITS OWN profile's allow/deny lists. os.environ is process-global and
    # the YAML→env bridge is first-writer-wins, so raw os.getenv reads here
    # would leak profile A's gates into profile B. Each accessor reads, in
    # order: the per-adapter env snapshot taken inside the owning profile's
    # runtime scope at connect() (authoritative under multiplex), then this
    # adapter's PlatformConfig.extra (per-profile YAML), with the live
    # scope-aware env read as the pre-connect fallback. Single-profile
    # deployments resolve to plain os.getenv, unchanged.

    def _snapshot_gate_env(self) -> None:
        """Capture authorization env vars for THIS adapter's profile.

        Must be called inside the owning profile's runtime scope (connect()
        runs there under multiplex) so the snapshot holds the profile's own
        values, not whichever profile bridged os.environ first.
        """
        from plugins.platforms.discord.adapter import _GATE_ENV_KEYS, _scoped_gate_env

        self._gate_env_snapshot = {
            key: _scoped_gate_env(key) for key in _GATE_ENV_KEYS
        }

    def _gate_env(self, name: str, default: str = "") -> str:
        """Read a gate env var from this adapter's snapshot (scope fallback)."""
        from plugins.platforms.discord.adapter import _scoped_gate_env

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
        from plugins.platforms.discord.adapter import _clean_discord_id

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

    def _discord_allow_all_users(self) -> bool:
        """Per-profile DISCORD_ALLOW_ALL_USERS flag."""
        raw = self._gate_raw("allow_all_users", "DISCORD_ALLOW_ALL_USERS")
        return str(raw or "").strip().lower() in {"true", "1", "yes"}

    def _gateway_allow_all_users(self) -> bool:
        """Per-profile GATEWAY_ALLOW_ALL_USERS flag."""
        return self._gate_env("GATEWAY_ALLOW_ALL_USERS").strip().lower() in {"true", "1", "yes"}

    def _get_allow_bots(self) -> str:
        """Per-profile DISCORD_ALLOW_BOTS mode (none|mentions|all)."""
        return self._gate_env("DISCORD_ALLOW_BOTS", "none").lower().strip() or "none"

    def _discord_free_response_channels(self) -> set:
        """Return Discord channel IDs/names where no bot mention is required.

        A single ``"*"`` entry (either from a list or a comma-separated
        string) is preserved in the returned set so callers can short-circuit
        on wildcard membership, consistent with ``allowed_channels``.
        """
        raw = self.config.extra.get("free_response_channels")
        if raw is None:
            raw = self._gate_env("DISCORD_FREE_RESPONSE_CHANNELS")
        if isinstance(raw, list):
            return {str(part).strip() for part in raw if str(part).strip()}
        # Coerce non-list scalars (str/int/float) to str before splitting.
        # YAML parses a bare numeric value such as
        # `free_response_channels: 1491973769726791812` as int, which was
        # previously falling through the isinstance(str) branch and silently
        # returning an empty set.  str() here accepts whatever scalar the YAML
        # loader hands us without changing existing string/CSV semantics.
        s = str(raw).strip() if raw is not None else ""
        if s:
            return {part.strip() for part in s.split(",") if part.strip()}
        return set()

    def _raw_mentioned_user_ids(self, message: Any) -> set:
        """Extract Discord user-mention IDs directly from raw message content.

        Covers both raw forms — ``<@ID>`` and the legacy ``<@!ID>`` nickname
        form — which ``message.mentions`` does not always populate (mobile,
        edited, or relayed messages can carry the mention in the content while
        leaving the resolved ``mentions`` list empty).
        """
        content = getattr(message, "content", "") or ""
        return {match.group(1) for match in re.finditer(r"<@!?(\d+)>", content)}

    def _self_is_explicitly_mentioned(self, message: Any) -> bool:
        """Return True when this bot is explicitly @mentioned in the message.

        Treats the bot as mentioned if it is either present in the resolved
        ``message.mentions`` list OR referenced by its raw ``<@ID>`` / ``<@!ID>``
        form in the message content.
        """
        if not self._client or not self._client.user:
            return False
        if self._client.user in getattr(message, "mentions", []):
            return True
        return str(self._client.user.id) in self._raw_mentioned_user_ids(message)

    def _self_is_raw_mentioned(self, message: Any) -> bool:
        """Return True only when this bot has an inline mention token.

        Discord reply-pings can add the replied-to bot to ``message.mentions``
        without a literal ``<@bot>`` token in ``message.content``. This helper
        intentionally ignores the resolved mentions list so the bot admission
        gate can distinguish an explicit cross-bot address from a reply chip.
        """
        if not self._client or not self._client.user:
            return False
        return str(self._client.user.id) in self._raw_mentioned_user_ids(message)

    def _discord_bots_require_inline_mention(self) -> bool:
        """Whether another bot must type an inline @mention to trigger us.

        Off by default. When on, a bot-authored message only wakes this bot
        if its content contains a literal ``<@thisbot>`` token. A Discord
        reply/quote to one of our messages is NOT enough on its own, because
        Discord's reply-ping silently adds us to ``message.mentions`` even
        though the author never typed our handle — which otherwise lets two
        bots ping-pong replies at each other indefinitely. Humans are never
        affected by this gate; it only applies to bot authors.

        Config: ``discord.bots_require_inline_mention`` (or env
        ``DISCORD_BOTS_REQUIRE_INLINE_MENTION``).
        """
        configured = self.config.extra.get("bots_require_inline_mention")
        if configured is not None:
            if isinstance(configured, str):
                return configured.lower() in {"true", "1", "yes", "on"}
            return bool(configured)
        return os.getenv("DISCORD_BOTS_REQUIRE_INLINE_MENTION", "false").lower() in {
            "true",
            "1",
            "yes",
            "on",
        }

    def _discord_channel_keys(self, message: Any, parent_channel_id: Optional[str] = None) -> set[str]:
        """Return channel identifiers accepted by Discord channel config gates.

        Users commonly configure channels by Discord snowflake ID, bare name, or
        ``#name``. Include the current channel and, for threads, the parent
        channel so free-response/no-thread/allow/ignore rules work with either
        form.
        """
        channel = getattr(message, "channel", None)
        return self._discord_channel_keys_from_channel(channel, parent_channel_id)

    def _discord_channel_keys_from_channel(
        self, channel: Any, parent_channel_id: Optional[str] = None
    ) -> set[str]:
        """Build channel-config gate keys directly from a channel object.

        Same key set as :meth:`_discord_channel_keys` (ID, bare name, ``#name``,
        and the parent channel for threads) but takes the channel directly so
        callers holding an ``interaction.channel`` (slash-command authorization)
        get name-form matching too — not just the ``on_message`` path.
        """
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
        """Return whether thread participation requires @mention to follow up.

        When ``False`` (default), once the bot has participated in a thread it
        keeps responding to every message in that thread without needing to be
        mentioned again — useful for one-on-one conversations.

        When ``True``, the @mention requirement is enforced inside threads as
        well.  Set this when multiple bots share a thread and you want each
        one to only fire on explicit @mention, avoiding bot-to-bot loops or
        unwanted cross-replies.
        """
        configured = self.config.extra.get("thread_require_mention")
        if configured is not None:
            if isinstance(configured, str):
                return configured.lower() not in {"false", "0", "no", "off"}
            return bool(configured)
        return os.getenv("DISCORD_THREAD_REQUIRE_MENTION", "false").lower() in {"true", "1", "yes", "on"}

    def _discord_history_backfill(self) -> bool:
        """Return whether history backfill is enabled for shared sessions."""
        configured = self.config.extra.get("history_backfill")
        if configured is not None:
            if isinstance(configured, str):
                return configured.lower() not in {"false", "0", "no", "off"}
            return bool(configured)
        return os.getenv("DISCORD_HISTORY_BACKFILL", "true").lower() in {"true", "1", "yes"}

    def _discord_history_backfill_limit(self) -> int:
        """Return the max number of messages to scan backwards for context.

        In practice the scan usually stops much earlier — at the bot's own
        last message in the channel (the natural partition point).  This
        limit is a safety cap for cold starts and long gaps where no prior
        bot message exists in recent history.
        """
        configured = self.config.extra.get("history_backfill_limit")
        if configured is not None:
            try:
                return int(configured)
            except (ValueError, TypeError):
                pass
        raw = os.getenv("DISCORD_HISTORY_BACKFILL_LIMIT", "50")
        try:
            return int(raw)
        except (ValueError, TypeError):
            return 50

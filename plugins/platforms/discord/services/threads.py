"""Discord thread lifecycle service."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from .. import adapter as _adapter

discord = _adapter.discord
logger = _adapter.logger
DISCORD_AVAILABLE = _adapter.DISCORD_AVAILABLE
VALID_THREAD_AUTO_ARCHIVE_MINUTES = _adapter.VALID_THREAD_AUTO_ARCHIVE_MINUTES
def _prompt_target_id(chat_id: str, metadata: Optional[dict]) -> str:
    """Resolve lazily so adapter import order remains optional-dependency safe."""
    return _adapter._prompt_target_id(chat_id, metadata)
def _env_bool(name: str, default: bool = False) -> bool:
    """Read the adapter helper lazily to avoid import-order coupling."""
    return _adapter._env_bool(name, default)
SendResult = _adapter.SendResult
_read_dm_role_auth_guild = getattr(_adapter, "_read_dm_role_auth_guild", lambda: None)


class ThreadLifecycleMixin:
    """Thread resolution, creation, naming, and handoff behavior."""

    async def _resolve_channel(self, channel_id: Any) -> Any:
        """Cached ``get_channel`` first, REST ``fetch_channel`` on miss (raises on API error)."""
        channel = self._client.get_channel(int(channel_id))
        if not channel:
            channel = await self._client.fetch_channel(int(channel_id))
        return channel

    def _thread_parent_channel(self, channel: Any) -> Any:
        """Return the parent text channel when invoked from a thread."""
        return getattr(channel, "parent", None) or channel

    async def _resolve_interaction_channel(self, interaction: discord.Interaction) -> Optional[Any]:
        """Return the interaction channel, fetching it if the payload is partial."""
        channel = getattr(interaction, "channel", None)
        if channel is not None:
            return channel
        if not self._client:
            return None
        channel_id = getattr(interaction, "channel_id", None)
        if channel_id is None:
            return None
        channel = self._client.get_channel(int(channel_id))
        if channel is not None:
            return channel
        try:
            return await self._client.fetch_channel(int(channel_id))
        except Exception:
            return None

    async def _create_thread(
        self, interaction: discord.Interaction, *, name: str, message: str = "",
        auto_archive_duration: int = 1440,
    ) -> Dict[str, Any]:
        """Create a thread in the current channel; falls back to seed message + create_thread on rejection (e.g. permissions)."""
        name = (name or "").strip()
        if not name:
            return {"error": "Thread name is required."}
        if auto_archive_duration not in VALID_THREAD_AUTO_ARCHIVE_MINUTES:
            allowed = ", ".join(str(v) for v in sorted(VALID_THREAD_AUTO_ARCHIVE_MINUTES))
            return {"error": f"auto_archive_duration must be one of: {allowed}."}
        channel = await self._resolve_interaction_channel(interaction)
        if channel is None:
            return {"error": "Could not resolve the current Discord channel."}
        if isinstance(channel, discord.DMChannel):
            return {"error": "Discord threads can only be created inside server text channels, not DMs."}
        parent_channel = self._thread_parent_channel(channel)
        if parent_channel is None:
            return {"error": "Could not determine a parent text channel for the new thread."}
        display_name = getattr(getattr(interaction, "user", None), "display_name", None) or "unknown user"
        reason = f"Requested by {display_name} via /thread"
        starter_message = (message or "").strip()
        try:
            thread = await parent_channel.create_thread(
                name=name, auto_archive_duration=auto_archive_duration, reason=reason,
            )
            if starter_message:
                await thread.send(starter_message)
            return self._thread_created(thread, name)
        except Exception as direct_error:
            try:
                seed_content = starter_message or f"\U0001f9f5 Thread created by Hermes: **{name}**"
                seed_msg = await parent_channel.send(seed_content)
                thread = await seed_msg.create_thread(
                    name=name, auto_archive_duration=auto_archive_duration, reason=reason,
                )
                return self._thread_created(thread, name)
            except Exception as fallback_error:
                return {
                    "error": (
                        "Discord rejected direct thread creation and the fallback also failed. "
                        f"Direct error: {direct_error}. Fallback error: {fallback_error}"
                    )
                }

    @staticmethod
    def _thread_created(thread: Any, name: str) -> Dict[str, Any]:
        return {"success": True, "thread_id": str(thread.id), "thread_name": getattr(thread, "name", None) or name}

    # ------------------------------------------------------------------
    # Auto-thread helpers
    # ------------------------------------------------------------------

    def _derive_auto_thread_name(self, content: str) -> str:
        """Fast placeholder thread name with mentions stripped (raw <@id> tokens mean nothing to humans).
        Semantic renaming happens after the first agent turn, once an LLM session title exists.

        Strip Discord mention syntax (users / roles / channels) so thread titles don't show raw <@id>,
        <@&id>, or <#id> markers — the ID isn't meaningful to humans glancing at the thread list (#6336).
        Real semantic naming is done after the first agent turn, when Hermes has an LLM-generated session
        title and can safely rename only this newly-created thread.
        """
        content = (content or "").strip()
        # <@123>, <@!123>, <@&123>, <#123> — collapse to empty; normalize spaces.
        content = re.sub(r"<@[!&]?\d+>", "", content)
        content = re.sub(r"<#\d+>", "", content)
        content = re.sub(r"\s+", " ", content).strip()
        thread_name = content[:80] if content else "Hermes"
        if len(content) > 80:
            thread_name = thread_name[:77] + "..."
        return thread_name

    @staticmethod
    def _stamp_auto_thread_name(thread: Any, thread_name: str) -> Any:
        """Remember the placeholder name so the semantic rename can verify it wasn't changed by a human."""
        try:
            setattr(thread, "_hermes_auto_thread_initial_name", thread_name)
        except Exception:
            pass
        return thread

    async def _auto_create_thread(self, message: 'DiscordMessage') -> Optional[Any]:
        """Create an auto-thread from a user message; returns the thread or ``None``.
        Primary path and seed-message fallback each retry once after a short backoff (transient errors).

        ``Cannot connect to host discord.com:443``) don't immediately burn through to the caller's failure
        path (#20243).
        """
        thread_name = self._derive_auto_thread_name(message.content or "")
        display_name = getattr(getattr(message, "author", None), "display_name", None) or "unknown user"
        reason = f"Auto-threaded from mention by {display_name}"
        last_direct_error: Exception | None = None
        last_fallback_error: Exception | None = None
        for attempt in range(2):
            try:
                thread = await message.create_thread(name=thread_name, auto_archive_duration=1440)
                return self._stamp_auto_thread_name(thread, thread_name)
            except Exception as direct_error:
                last_direct_error = direct_error
                try:
                    seed_msg = await message.channel.send(
                        f"\U0001f9f5 Thread created by Hermes: **{thread_name}**"
                    )
                    thread = await seed_msg.create_thread(name=thread_name, auto_archive_duration=1440, reason=reason)
                    return self._stamp_auto_thread_name(thread, thread_name)
                except Exception as fallback_error:
                    last_fallback_error = fallback_error
                    if attempt == 0:
                        # Brief backoff: most failures here are transient connect errors.
                        await asyncio.sleep(0.75)
                        continue
        logger.warning(
            "[%s] Auto-thread creation failed after retry. Direct error: %s. Fallback error: %s",
            self.name, last_direct_error, last_fallback_error,
        )
        return None

    async def rename_thread(
        self, thread_id: str, name: str, *, only_if_current_name: Optional[str] = None,
    ) -> bool:
        """Best-effort rename; ``only_if_current_name`` protects human-renamed/pre-existing threads (no-op on mismatch)."""
        if not self._client or not DISCORD_AVAILABLE:
            return False
        try:
            thread_id_int = int(str(thread_id))
        except (TypeError, ValueError):
            return False
        cleaned = re.sub(r"\s+", " ", str(name or "")).strip()
        if not cleaned:
            return False
        # Thread names are budgeted in UTF-16 code units (emoji count double) — use the UTF-16 helpers.
        from gateway.platforms.base import utf16_len, _prefix_within_utf16_limit
        if utf16_len(cleaned) > 80:
            cleaned = _prefix_within_utf16_limit(cleaned, 77).rstrip() + "..."
        try:
            thread = self._client.get_channel(thread_id_int)
            if thread is None:
                thread = await self._client.fetch_channel(thread_id_int)
        except Exception:
            logger.debug("[%s] Failed to resolve Discord thread %s for rename", self.name, thread_id, exc_info=True)
            return False
        current_name = getattr(thread, "name", None)
        if only_if_current_name is not None and current_name != only_if_current_name:
            logger.info(
                "[%s] Discord semantic thread rename skipped for %s: current name %r != expected %r",
                self.name, thread_id, current_name, only_if_current_name,
            )
            return False
        if current_name == cleaned:
            return True
        edit = getattr(thread, "edit", None)
        if edit is None:
            return False
        try:
            await edit(name=cleaned, reason="Hermes semantic session title")
            logger.info(
                "[%s] Renamed Discord thread %s from %r to %r",
                self.name, thread_id, current_name, cleaned,
            )
            return True
        except Exception:
            logger.debug("[%s] Failed to rename Discord thread %s", self.name, thread_id, exc_info=True)
            return False

    async def create_handoff_thread(self, parent_chat_id: str, name: str) -> Optional[str]:
        """Create a handoff thread under a text channel; returns the thread id or ``None``.
        Falls back to seed-message + ``message.create_thread``; DMs/voice/threads can't host threads."""
        if not self._client or not DISCORD_AVAILABLE:
            return None
        try:
            parent_id = int(parent_chat_id)
        except (TypeError, ValueError):
            return None
        try:
            parent = self._client.get_channel(parent_id)
            if parent is None:
                parent = await self._client.fetch_channel(parent_id)
        except Exception as exc:
            logger.warning(
                "[%s] Handoff thread: cannot resolve parent %s: %s", self.name, parent_chat_id, exc,
            )
            return None
        # DMs, voice channels, and existing threads can't host child threads.
        if isinstance(parent, getattr(discord, "DMChannel", ())):
            logger.info(
                "[%s] Handoff thread: parent %s is a DM; threads not supported here",
                self.name, parent_chat_id,
            )
            return None
        thread_name = (name or "handoff").strip()[:80] or "handoff"
        reason = "Hermes session handoff"
        try:
            create = getattr(parent, "create_thread", None)
            if create is not None:
                thread = await create(name=thread_name, auto_archive_duration=1440, reason=reason)
                return str(thread.id)
        except Exception as direct_error:
            logger.debug(
                "[%s] Handoff thread: direct create failed (%s); trying seed-message fallback",
                self.name, direct_error,
            )
        try:
            send = getattr(parent, "send", None)
            if send is None:
                return None
            seed_msg = await send(f"\U0001f9f5 Hermes handoff: **{thread_name}**")
            thread = await seed_msg.create_thread(
                name=thread_name, auto_archive_duration=1440, reason=reason,
            )
            return str(thread.id)
        except Exception as fallback_error:
            logger.warning(
                "[%s] Handoff thread: both create paths failed for parent %s: %s",
                self.name, parent_chat_id, fallback_error,
            )
            return None

    def _self_contained_prompt_content(
        self, header: str, body: str, *, code_block: bool = False, tail: str = ""
    ) -> str:
        """Plain content mirroring an embed's payload.
        Embeds can be invisible/detached on web/mobile, so ``content`` carries the payload."""
        body = str(body or "")
        if code_block:
            prefix = f"{header}\n```bash\n"
            suffix = f"\n```{tail}"
        else:
            prefix = f"{header}\n\n"
            suffix = tail
        truncated_suffix = "\n... [truncated]"
        budget = max(0, self.MAX_MESSAGE_LENGTH - len(prefix) - len(suffix))
        if len(body) > budget:
            body = body[: max(0, budget - len(truncated_suffix))] + truncated_suffix
        return f"{prefix}{body}{suffix}"

    def _approval_mention_content(self) -> Optional[str]:
        """User mentions for approval prompts, gated on ``discord.approval_mentions``
        (``DISCORD_APPROVAL_MENTIONS``). Only numeric allowlist entries; default off."""
        if not _env_bool("DISCORD_APPROVAL_MENTIONS", False):
            return None
        user_ids = sorted(uid for uid in self._allowed_user_ids if str(uid).isdigit())
        if not user_ids:
            return None
        return " ".join(f"<@{uid}>" for uid in user_ids)

    async def _send_prompt(
        self, chat_id: str, metadata: Optional[dict], build, *, fail_log: Optional[str] = None,
    ) -> SendResult:
        """Shared tail for interactive prompts: resolve target channel, call ``build(channel) ->
        (send_kwargs, view)``, send, remember the message on the view. ``fail_log`` labels failures."""
        if not self._client or not DISCORD_AVAILABLE:
            return SendResult(success=False, error="Not connected")
        try:
            channel = await self._resolve_channel(_prompt_target_id(chat_id, metadata))
            send_kwargs, view = build(channel)
            msg = await channel.send(**send_kwargs)
            if view is not None:
                view._message = msg
            return SendResult(success=True, message_id=str(msg.id))
        except Exception as e:
            if fail_log:
                logger.warning("[%s] %s failed: %s", self.name, fail_log, e)
            return SendResult(success=False, error=str(e))

    @staticmethod
    def _embed_body(text: str, limit: int = 4088) -> str:
        """Trim to Discord's 4096-char embed description limit (conservatively)."""
        return text if len(text) <= limit else text[: limit - 3] + "..."

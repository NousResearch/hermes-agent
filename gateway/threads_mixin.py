"""Telegram-topic and Discord auto-thread helpers for GatewayRunner.

Extracted from ``gateway/run.py`` (god-file decomposition Phase 3b, slice 24
of #54962). These are the Telegram DM topic-mode helpers (topic-mode detection,
binding persistence/recovery, /topic status/help/restore, topic auto-title
renames) and the Discord auto-thread semantic-rename helpers. 30 methods
(~870 LOC); lifting them into a mixin that ``GatewayRunner`` inherits keeps
every ``self._*`` dispatch + test reference working via the MRO while removing
the bulk from run.py.

Behavior-neutral: every method is lifted verbatim from ``GatewayRunner``.
``self.*`` calls resolve unchanged via the MRO. Neutral dependencies import at
module top; the module-level ``logger`` is ``logging.getLogger("gateway.run")``
so log records keep the exact ``"gateway.run"`` name. No module-level
``from gateway.run import`` — run.py is never imported at module top, avoiding
an import cycle.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import re
from pathlib import Path
from typing import Any, Optional, Tuple, cast

from agent.async_utils import safe_schedule_threadsafe
from agent.i18n import t
from gateway.config import Platform
from gateway.platforms.base import (
    MessageEvent,
    _prefix_within_utf16_limit,
    utf16_len,
)
from gateway.session import SessionSource

logger = logging.getLogger("gateway.run")


class GatewayThreadsMixin:
    """Telegram topic-mode and Discord auto-thread helpers for GatewayRunner."""

    def _telegram_topic_mode_enabled(self, source: SessionSource) -> bool:
        """Return whether Telegram DM topic mode is active for this chat."""
        if source.platform != Platform.TELEGRAM or source.chat_type != "dm":
            return False
        session_db = getattr(self, "_session_db", None)
        if session_db is None:
            return False
        # Runs off-loop (always via asyncio.to_thread); use the sync handle.
        session_db = getattr(session_db, "_db", session_db)
        try:
            raw = session_db.is_telegram_topic_mode_enabled(
                chat_id=str(source.chat_id),
                user_id=str(source.user_id),
            )
        except Exception:
            logger.debug("Failed to read Telegram topic mode state", exc_info=True)
            return False
        # Only honor a real True from the SessionDB. Any other value
        # (including MagicMock instances from test fixtures that didn't
        # opt into topic mode) means topic mode is off for this chat.
        return raw is True

    # Telegram's General (pinned top) topic in forum-enabled private chats.
    # Bot API behavior varies: some clients omit message_thread_id for
    # General, others send "1". Treat both as "root" for lobby/lane purposes.
    _TELEGRAM_GENERAL_TOPIC_IDS = frozenset({"", "1"})

    def _is_telegram_topic_root_lobby(self, source: SessionSource) -> bool:
        """True for the main Telegram DM (or General topic) when topic mode has made it a lobby."""
        if source.platform != Platform.TELEGRAM or source.chat_type != "dm":
            return False
        if not self._telegram_topic_mode_enabled(source):
            return False
        tid = str(source.thread_id or "")
        return tid in self._TELEGRAM_GENERAL_TOPIC_IDS

    def _is_telegram_topic_lane(self, source: SessionSource) -> bool:
        """True for a user-created Telegram private-chat topic lane."""
        if source.platform != Platform.TELEGRAM or source.chat_type != "dm":
            return False
        if not self._telegram_topic_mode_enabled(source):
            return False
        tid = str(source.thread_id or "")
        if not tid or tid in self._TELEGRAM_GENERAL_TOPIC_IDS:
            return False
        return True

    _TELEGRAM_LOBBY_REMINDER_COOLDOWN_S = 30.0

    def _should_send_telegram_lobby_reminder(self, source: SessionSource) -> bool:
        """Rate-limit root-DM lobby reminders to one message per cooldown window.

        A user who forgets multi-session mode is enabled and types several
        prompts in the root DM would otherwise get a reminder for every
        message. Cap it so the first one lands and the rest stay quiet.
        """
        if not hasattr(self, "_telegram_lobby_reminder_ts"):
            self._telegram_lobby_reminder_ts = {}
        chat_id = str(source.chat_id or "")
        if not chat_id:
            return True
        import time as _time
        now = _time.monotonic()
        last = self._telegram_lobby_reminder_ts.get(chat_id, 0.0)
        if now - last < self._TELEGRAM_LOBBY_REMINDER_COOLDOWN_S:
            return False
        self._telegram_lobby_reminder_ts[chat_id] = now
        return True

    def _telegram_topic_root_lobby_message(self) -> str:
        return (
            "This main chat is reserved for system commands.\n\n"
            "To start a new Hermes chat, open the All Messages topic at the top "
            "of this bot interface and send any message there. Telegram will "
            "create a new topic for that message; each topic works as an "
            "independent Hermes session."
        )

    def _telegram_topic_root_new_message(self) -> str:
        return (
            "To start a new parallel Hermes chat, open the All Messages topic "
            "at the top of this bot interface and send any message there. "
            "Telegram will create a new topic for it.\n\n"
            "Each topic is an independent Hermes session. Use /new inside an "
            "existing topic only if you want to replace that topic's current session."
        )

    def _telegram_topic_new_header(self, source: SessionSource) -> Optional[str]:
        if not self._is_telegram_topic_lane(source):
            return None
        return (
            "Started a new Hermes session in this topic.\n\n"
            "Tip: for parallel work, open All Messages and send a message there "
            "to create a separate topic instead of using /new here. /new replaces "
            "the session attached to the current topic."
        )

    def _record_telegram_topic_binding(
        self,
        source: SessionSource,
        session_entry,
    ) -> None:
        """Persist the Telegram topic -> Hermes session binding for topic lanes."""
        session_db = getattr(self, "_session_db", None)
        if session_db is None or not source.chat_id or not source.thread_id:
            return
        # Runs off-loop (always via asyncio.to_thread); use the sync handle.
        session_db = getattr(session_db, "_db", session_db)
        session_db.bind_telegram_topic(
            chat_id=str(source.chat_id),
            thread_id=str(source.thread_id),
            user_id=str(source.user_id or ""),
            session_key=session_entry.session_key,
            session_id=session_entry.session_id,
        )

    def _sync_telegram_topic_binding(
        self,
        source: SessionSource,
        session_entry,
        *,
        reason: str,
    ) -> None:
        """Update the topic binding to point at ``session_entry.session_id``.

        Telegram topic lanes persist a (chat_id, thread_id) -> session_id row
        so reopening a topic in a fresh process resumes the right Hermes
        session. When compression rotates ``session_entry.session_id`` mid-turn,
        the binding goes stale and the next inbound message in that topic
        reloads the oversized parent transcript instead of the compressed
        child, retriggering preflight compression — sometimes in a loop
        (#20470, #29712, #33414).
        """
        if not self._is_telegram_topic_lane(source):
            return
        try:
            self._record_telegram_topic_binding(source, session_entry)
        except Exception:
            logger.debug(
                "telegram topic binding refresh failed (%s)", reason, exc_info=True,
            )

    def _recover_telegram_topic_thread_id(
        self,
        source: SessionSource,
    ) -> Optional[str]:
        """Pin DM-topic routing to the user's last-active topic.

        Telegram can omit ``message_thread_id`` or surface General (``1``)
        for some topic-mode DM replies. In those lobby-shaped cases, keep the
        conversation attached to the user's most-recent bound topic.

        Do not rewrite a non-lobby, previously-unbound thread id: a newly
        created Telegram DM topic is also "unknown" until the first inbound
        message is recorded, and rewriting it would send that brand-new topic's
        answer into an older lane. Returns None to leave the source alone.
        """
        if (
            source.platform != Platform.TELEGRAM
            or source.chat_type != "dm"
            or not source.chat_id
            or not source.user_id
            or not self._telegram_topic_mode_enabled(source)
        ):
            return None
        inbound = str(source.thread_id or "")
        is_lobby = not inbound or inbound in self._TELEGRAM_GENERAL_TOPIC_IDS
        if not is_lobby:
            # A non-lobby, unknown thread_id is most likely the first message in
            # a brand-new Telegram DM topic. Preserve it so it can be recorded
            # as a new independent lane below instead of hijacking the latest
            # existing topic binding.
            return None
        session_db = getattr(self, "_session_db", None)
        if session_db is None:
            return None
        # Runs off-loop (always via asyncio.to_thread); use the sync handle.
        session_db = getattr(session_db, "_db", session_db)
        try:
            bindings = session_db.list_telegram_topic_bindings_for_chat(
                chat_id=str(source.chat_id),
            )
        except Exception:
            logger.debug("topic-recover: read failed", exc_info=True)
            return None
        if not bindings:
            return None
        user_id = str(source.user_id)
        for b in bindings:  # newest-first
            if str(b.get("user_id") or "") == user_id:
                recovered = str(b.get("thread_id") or "")
                if recovered and recovered != inbound:
                    return recovered
                return None
        return None

    def _normalize_source_for_session_key(
        self,
        source: SessionSource,
    ) -> SessionSource:
        """Apply Telegram DM topic recovery to a source for session-key purposes.

        ``_handle_message_with_agent`` rewrites ``source.thread_id`` via
        ``_recover_telegram_topic_thread_id`` *before* deriving the session
        key for a normal message turn (a lobby/stripped reply gets pinned to
        the user's last-active topic).  Session-scoped command handlers like
        ``/model`` and ``/reasoning`` derive their override key from the raw
        inbound ``event.source``, which skips that recovery — so the override
        is stored under a different key than the next message turn reads,
        and the override is silently dropped on Telegram forum topics and
        after compression session splits (#30479).

        Returns a recovery-normalized copy when a rewrite applies, otherwise
        the original source unchanged.  Always derive the override storage key
        from the result so storage and read use an identical key.
        """
        try:
            recovered = self._recover_telegram_topic_thread_id(source)
        except Exception:
            return source
        if recovered is None:
            return source
        return dataclasses.replace(source, thread_id=recovered)

    async def _get_telegram_topic_capabilities(self, source: SessionSource) -> dict:
        """Read Telegram private-topic capability flags via Bot API getMe."""
        adapter = self._adapter_for_source(source)
        bot = getattr(adapter, "_bot", None)
        if bot is None or not hasattr(bot, "get_me"):
            return {"checked": False}
        try:
            me = await bot.get_me()
        except Exception:
            logger.debug("Failed to fetch Telegram getMe topic capabilities", exc_info=True)
            return {"checked": False}

        def _field(name: str):
            if hasattr(me, name):
                return getattr(me, name)
            api_kwargs = getattr(me, "api_kwargs", None)
            if isinstance(api_kwargs, dict) and name in api_kwargs:
                return api_kwargs.get(name)
            if isinstance(me, dict):
                return me.get(name)
            return None

        return {
            "checked": True,
            "has_topics_enabled": _field("has_topics_enabled"),
            "allows_users_to_create_topics": _field("allows_users_to_create_topics"),
        }

    async def _ensure_telegram_system_topic(self, source: SessionSource) -> None:
        """Create/pin the managed System topic after /topic activation when possible."""
        adapter = self._adapter_for_source(source)
        if adapter is None or not source.chat_id:
            return

        thread_id = None
        create_topic = getattr(adapter, "_create_dm_topic", None)
        if callable(create_topic):
            try:
                thread_id = await create_topic(int(source.chat_id), "System")
            except Exception:
                logger.debug("Failed to create Telegram System topic", exc_info=True)
        if not thread_id:
            return

        message_id = None
        try:
            send_result = await adapter.send(
                source.chat_id,
                "System topic for Hermes commands and status.",
                metadata={"thread_id": str(thread_id)},
            )
            message_id = getattr(send_result, "message_id", None)
        except Exception:
            logger.debug("Failed to send Telegram System topic intro", exc_info=True)
        if not message_id:
            return

        bot = getattr(adapter, "_bot", None)
        if bot is None or not hasattr(bot, "pin_chat_message"):
            return
        try:
            await bot.pin_chat_message(
                chat_id=int(source.chat_id),
                message_id=int(message_id),
                disable_notification=True,
            )
        except Exception:
            logger.debug("Failed to pin Telegram System topic intro", exc_info=True)

    async def _send_telegram_topic_setup_image(self, source: SessionSource) -> None:
        """Send the bundled BotFather Threads Settings screenshot when available."""
        adapter = self._adapter_for_source(source)
        if adapter is None or not source.chat_id or not hasattr(adapter, "send_image_file"):
            return
        image_path = Path(__file__).resolve().parent / "assets" / "telegram-botfather-threads-settings.jpg"
        if not image_path.exists():
            return
        try:
            await adapter.send_image_file(
                chat_id=source.chat_id,
                image_path=str(image_path),
                caption="BotFather → Bot Settings → Threads Settings",
                metadata={"thread_id": str(source.thread_id)} if source.thread_id else None,
            )
        except Exception:
            logger.debug("Failed to send Telegram topic setup image", exc_info=True)

    def _sanitize_telegram_topic_title(self, title: str) -> str:
        """Return a Bot API-safe forum topic name from a generated session title."""
        cleaned = re.sub(r"\s+", " ", str(title or "")).strip()
        if not cleaned:
            return "Hermes Chat"
        # Telegram forum topic names are short (currently 1-128 chars). Keep
        # extra room for multi-byte titles and avoid trailing ellipsis churn.
        if len(cleaned) > 120:
            cleaned = cleaned[:117].rstrip() + "..."
        return cleaned

    def _is_discord_auto_thread_lane(self, source: SessionSource) -> bool:
        """Return True only for Discord threads Hermes just auto-created."""
        return (
            source.platform == Platform.DISCORD
            and source.chat_type == "thread"
            and bool(getattr(source, "auto_thread_created", False))
            and bool(source.thread_id)
            and bool(getattr(source, "auto_thread_initial_name", None))
        )

    def _is_relay_discord_channel_lane(self, source: SessionSource) -> bool:
        """Shape-only check: a relay-delivered Discord CHANNEL event whose
        reply the connector MAY auto-thread (title-turn registration gate).

        Deliberately does NOT consult the send-result cache: at registration
        time (before delivery) the feedback can't exist yet. The rename lane
        polls the cache at fire time instead."""
        return (
            source.platform == Platform.DISCORD
            and bool(source.chat_id)
            and not source.thread_id
            and source.chat_type in ("group", "channel")
            and getattr(source, "delivered_via_upstream_relay", False) is True
        )

    def _relay_auto_thread_info(
        self, source: SessionSource
    ) -> Optional[Tuple[str, str]]:
        """(thread_id, initial_name) when the RELAY connector auto-threaded our
        reply to this source's chat — the title-turn sibling of
        _is_discord_auto_thread_lane.

        The marker-based check above only lights up for events ARRIVING IN an
        auto-created thread (turn 2+). The auto-title fires on the FIRST
        exchange, whose source is the PARENT channel event — the thread did
        not exist at ingest, so no markers can be present and the native lane
        check never matches on the relay title turn (staging repro
        2026-07-29: initial titles fine, semantic renames never happened).

        Preferred path: the connector stamps ``prospective_thread_id`` on the
        inbound (the anchor message id, which IS the id of the thread it will
        auto-create). It's deterministic and per-message, so it identifies the
        EXACT thread even when several auto-threads spawn from one channel —
        unlike the send-result cache below, which held a single slot per parent
        chat and so only the FIRST thread in a channel ever renamed (staging
        repro 2026-08-02: thread A renamed, sibling thread B stuck at raw
        text). The connector's own created-name guard (prefer_connector_created)
        enforces no-clobber, so no initial name is needed here.

        Fallback: the connector reports where the reply actually landed on the
        send result (contract §SendResult thread_id/auto_thread_name); the
        relay adapter caches it per chat and this reads it back — kept for
        older connectors that don't stamp prospective_thread_id.
        """
        if source.platform != Platform.DISCORD or not source.chat_id:
            return None
        if not getattr(source, "delivered_via_upstream_relay", False):
            return None
        prospective = getattr(source, "prospective_thread_id", None)
        if prospective:
            # Deterministic per-thread identity; the empty initial-name marker
            # signals the caller to rely on the connector-side no-clobber guard.
            return (str(prospective), "")
        adapter = self._adapter_for_source(source)
        info_fn = getattr(adapter, "auto_thread_info_for_chat", None)
        if not callable(info_fn):
            return None
        try:
            info = info_fn(str(source.chat_id))
            if (
                isinstance(info, tuple)
                and len(info) == 2
                and all(isinstance(x, str) for x in info)
            ):
                return cast(Tuple[str, str], info)
            return None
        except Exception:
            return None

    def _sanitize_discord_thread_title(self, title: str) -> str:
        """Return a Discord-safe semantic thread title from a session title.

        Discord thread names are capped at 100 characters measured in UTF-16
        code units (emoji count double), so truncate with the UTF-16 helpers
        rather than Python code-point slices.
        """
        cleaned = re.sub(r"\s+", " ", str(title or "")).strip()
        if not cleaned:
            return "Hermes Chat"
        if utf16_len(cleaned) > 80:
            cleaned = _prefix_within_utf16_limit(cleaned, 77).rstrip() + "..."
        return cleaned

    async def _rename_discord_auto_thread_for_session_title(
        self,
        source: SessionSource,
        session_id: str,
        title: str,
        relay_info: Optional[Tuple[str, str]] = None,
    ) -> None:
        """Best-effort semantic rename of a newly auto-created Discord thread.

        ``relay_info`` is the (thread_id, initial_name) pair from the relay
        connector's send-result feedback — supplied on the title turn, where
        the source is the parent-channel event and carries no auto-thread
        markers (see _relay_auto_thread_info). When absent, the native
        marker-based lane supplies thread identity from the source itself.
        """
        if relay_info is None and not await asyncio.to_thread(
            self._is_discord_auto_thread_lane, source
        ):
            # Relay title turn with no feedback captured at schedule time:
            # the auto-title thread races the delivery that produces the
            # connector's send-result feedback (thread_id + initial name).
            # Poll the adapter cache briefly before giving up — delivery is
            # typically milliseconds-to-seconds behind the title.
            if not self._is_relay_discord_channel_lane(source):
                return
            for _ in range(20):  # up to ~10s
                relay_info = self._relay_auto_thread_info(source)
                if relay_info is not None:
                    break
                await asyncio.sleep(0.5)
            if relay_info is None:
                # True miss: the connector did not auto-thread this reply
                # (policy off, DM, already-threaded, or send failed).
                return
        adapter = self._adapter_for_source(source) if getattr(self, "adapters", None) else None
        if adapter is None:
            return
        rename_thread = getattr(adapter, "rename_thread", None)
        if rename_thread is None:
            return
        target_thread_id = relay_info[0] if relay_info else str(source.thread_id)
        # Relay lane (relay_info present): ask the CONNECTOR to enforce the
        # no-clobber guard from its own created-name memory — the gateway
        # can't reliably reproduce the thread's initial name byte-for-byte
        # (normalization drift silently declined every rename before this).
        # Native-marker lane keeps the legacy string guard.
        use_connector_guard = relay_info is not None
        guard_name = (
            None
            if use_connector_guard
            else getattr(source, "auto_thread_initial_name", None)
        )
        thread_name = self._sanitize_discord_thread_title(title)
        # Relay lane only: the connector's egress guard resolves the owning
        # tenant from the outbound metadata's scope_id (guild) / user_id
        # (author). Those discriminator caches are keyed by the PARENT channel
        # chat_id (learned at inbound), NOT the thread id. rename_thread
        # defaults chat_id to the thread id when no parent is given, so the
        # scope/author lookup misses and the connector declines the op
        # ("target not routed to an onboarded tenant" — the live failure on
        # staging 2026-08-01). Pass the parent channel id (the relay source's
        # chat_id IS the parent channel; the thread came from send-result
        # feedback) so the discriminators resolve. Native lane needs nothing:
        # its source IS the thread and it renames via the direct Discord API,
        # not the relay egress guard.
        parent_chat_id = (
            str(source.chat_id) if use_connector_guard and source.chat_id else None
        )
        logger.info(
            "discord auto-thread rename: thread=%s lane=%s new_title=%r",
            target_thread_id,
            "relay" if use_connector_guard else "native",
            thread_name,
        )
        try:
            renamed = await rename_thread(
                target_thread_id,
                thread_name,
                prefer_connector_created=use_connector_guard,
                only_if_current_name=guard_name,
                parent_chat_id=parent_chat_id,
            )
            logger.info(
                "discord auto-thread rename result: thread=%s applied=%s",
                target_thread_id,
                bool(renamed),
            )
        except Exception:
            logger.debug("Failed to rename Discord auto-thread for generated session title", exc_info=True)

    def _schedule_discord_semantic_thread_rename(
        self,
        source: SessionSource,
        session_id: str,
        title: str,
    ) -> None:
        """Schedule Discord auto-thread rename from the auto-title background thread."""
        relay_info = None
        if not title:
            return
        if not self._is_discord_auto_thread_lane(source):
            # Relay title turn: the source is the PARENT channel event (the
            # thread didn't exist at ingest, so no auto-thread markers). The
            # connector's send-result feedback tells us where the reply
            # landed — but the auto-title thread races the delivery that
            # produces it, so a cache miss HERE is not a verdict. Schedule
            # whenever the SHAPE matches; the async rename lane polls the
            # cache (with a bounded wait) and no-ops on a true miss.
            relay_info = self._relay_auto_thread_info(source)
            if relay_info is None and not self._is_relay_discord_channel_lane(
                source
            ):
                return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = getattr(self, "_gateway_loop", None)
        if loop is None or loop.is_closed():
            return
        try:
            copied_source = dataclasses.replace(source)
        except Exception:
            copied_source = source
        future = safe_schedule_threadsafe(
            self._rename_discord_auto_thread_for_session_title(
                copied_source, session_id, title, relay_info=relay_info
            ),
            loop,
            logger=logger,
            log_message="Discord semantic thread rename failed to schedule",
        )
        if future is None:
            return

        def _log_rename_failure(fut) -> None:
            try:
                fut.result()
            except Exception:
                logger.debug("Discord semantic thread rename failed", exc_info=True)

        future.add_done_callback(_log_rename_failure)

    async def _rename_telegram_topic_for_session_title(
        self,
        source: SessionSource,
        session_id: str,
        title: str,
    ) -> None:
        """Best-effort rename of a Telegram DM topic when Hermes auto-titles a session."""
        if not await asyncio.to_thread(self._is_telegram_topic_lane, source) or not source.chat_id or not source.thread_id:
            return

        # Operator can fully disable per-topic auto-rename via
        # extra.disable_topic_auto_rename. Useful when topics are managed
        # by the user (ad-hoc Threaded Mode) and auto-rename would
        # overwrite their chosen names every time the auto-title fires.
        if self._telegram_topic_auto_rename_disabled(source):
            return

        # Skip rename when the topic is operator-declared via
        # extra.dm_topics. Those topics have fixed names chosen by the
        # operator (plus optional skill binding); auto-renaming would
        # silently mutate operator config.
        #
        # Check the class, not the instance — getattr() on MagicMock
        # auto-creates attributes, so `hasattr(adapter, "_get_dm_topic_info")`
        # would return True for every test double.
        adapter = self._adapter_for_source(source)
        if adapter is not None:
            get_info = getattr(type(adapter), "_get_dm_topic_info", None)
            if callable(get_info):
                try:
                    operator_topic = get_info(adapter, str(source.chat_id), str(source.thread_id))
                except Exception:
                    operator_topic = None
                # Only treat dict-shaped returns as operator-declared; a
                # bare MagicMock or other sentinel shouldn't count.
                if isinstance(operator_topic, dict):
                    return

        session_db = getattr(self, "_session_db", None)
        if session_db is not None:
            try:
                binding = await session_db.get_telegram_topic_binding(
                    chat_id=str(source.chat_id),
                    thread_id=str(source.thread_id),
                )
                if binding and str(binding.get("session_id") or "") != str(session_id):
                    return
            except Exception:
                logger.debug("Failed to verify Telegram topic binding before rename", exc_info=True)
                return

        if adapter is None:
            return
        topic_name = self._sanitize_telegram_topic_title(title)
        try:
            rename_topic = getattr(adapter, "rename_dm_topic", None)
            if rename_topic is not None:
                await rename_topic(
                    chat_id=str(source.chat_id),
                    thread_id=str(source.thread_id),
                    name=topic_name,
                )
                return

            bot = getattr(adapter, "_bot", None)
            edit_forum_topic = getattr(bot, "edit_forum_topic", None) if bot is not None else None
            if edit_forum_topic is None:
                edit_forum_topic = getattr(bot, "editForumTopic", None) if bot is not None else None
            if edit_forum_topic is None:
                return
            try:
                await edit_forum_topic(
                    chat_id=int(source.chat_id),
                    message_thread_id=int(source.thread_id),
                    name=topic_name,
                )
            except (TypeError, ValueError):
                await edit_forum_topic(
                    chat_id=source.chat_id,
                    message_thread_id=source.thread_id,
                    name=topic_name,
                )
        except Exception:
            logger.debug("Failed to rename Telegram topic for auto-generated title", exc_info=True)

    def _telegram_topic_auto_rename_disabled(self, source: SessionSource) -> bool:
        """Return True when operator disabled per-topic auto-rename for this Telegram chat.

        Controlled via ``gateway.platforms.telegram.extra.disable_topic_auto_rename``.
        Default is False (auto-rename enabled, preserves prior behaviour).
        """
        platform_cfg = (
            self.config.platforms.get(source.platform)
            if getattr(self, "config", None) and getattr(self.config, "platforms", None)
            else None
        )
        if platform_cfg is None:
            return False
        extra = getattr(platform_cfg, "extra", None) or {}
        value = extra.get("disable_topic_auto_rename")
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _schedule_telegram_topic_title_rename(
        self,
        source: SessionSource,
        session_id: str,
        title: str,
    ) -> None:
        """Schedule a topic rename from the auto-title background thread."""
        if not title or not self._is_telegram_topic_lane(source):
            return
        if self._telegram_topic_auto_rename_disabled(source):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = getattr(self, "_gateway_loop", None)
        if loop is None or loop.is_closed():
            return
        try:
            copied_source = dataclasses.replace(source)
        except Exception:
            copied_source = source
        future = safe_schedule_threadsafe(
            self._rename_telegram_topic_for_session_title(copied_source, session_id, title),
            loop,
            logger=logger,
            log_message="Telegram topic title rename failed to schedule",
        )
        if future is None:
            return
        def _log_rename_failure(fut) -> None:
            try:
                fut.result()
            except Exception:
                logger.debug("Telegram topic title rename failed", exc_info=True)

        future.add_done_callback(_log_rename_failure)

    _TELEGRAM_CAPABILITY_HINT_COOLDOWN_S = 300.0

    def _should_send_telegram_capability_hint(self, source: SessionSource) -> bool:
        """Rate-limit the BotFather Threads Settings screenshot.

        If a user sends /topic repeatedly while Threads Settings are still
        off, we shouldn't keep re-uploading the screenshot every time.
        """
        if not hasattr(self, "_telegram_capability_hint_ts"):
            self._telegram_capability_hint_ts = {}
        chat_id = str(source.chat_id or "")
        if not chat_id:
            return True
        import time as _time
        now = _time.monotonic()
        last = self._telegram_capability_hint_ts.get(chat_id, 0.0)
        if now - last < self._TELEGRAM_CAPABILITY_HINT_COOLDOWN_S:
            return False
        self._telegram_capability_hint_ts[chat_id] = now
        return True

    def _telegram_topic_help_text(self) -> str:
        return (
            "/topic — enable multi-session DM mode (one bot, many parallel chats)\n"
            "\n"
            "Usage:\n"
            "  /topic             Enable topic mode, or show status if already on\n"
            "  /topic help        Show this message\n"
            "  /topic off         Disable topic mode and clear topic bindings\n"
            "  /topic <id>        Inside a topic: restore a previous session by ID\n"
            "\n"
            "How it works:\n"
            "1. Run /topic once in this DM — Hermes checks BotFather Threads\n"
            "   Settings are enabled and flips on multi-session mode.\n"
            "2. Tap All Messages at the top of the bot and send any message.\n"
            "   Telegram creates a new topic for that message; each topic is\n"
            "   an independent Hermes session (fresh history, fresh context).\n"
            "3. The root DM becomes a system lobby — send /topic, /status,\n"
            "   /help, /usage there. Normal prompts go in a topic.\n"
            "4. /new inside a topic resets just that topic's session.\n"
            "5. /topic <id> inside a topic restores an old session into it."
        )

    async def _disable_telegram_topic_mode_for_chat(self, source: SessionSource) -> str:
        """Cleanly disable topic mode for a chat via /topic off."""
        if not self._session_db:
            from hermes_state import format_session_db_unavailable
            return format_session_db_unavailable(prefix=t("gateway.shared.session_db_unavailable_prefix"))
        chat_id = str(source.chat_id or "")
        if not chat_id:
            return "Could not determine chat ID."
        # No-op if never enabled.
        try:
            currently_enabled = await self._session_db.is_telegram_topic_mode_enabled(
                chat_id=chat_id,
                user_id=str(source.user_id or ""),
            )
        except Exception:
            currently_enabled = False
        if not currently_enabled:
            return "Multi-session topic mode is not currently enabled for this chat."
        try:
            await self._session_db.disable_telegram_topic_mode(chat_id=chat_id)
        except Exception as exc:
            logger.exception("Failed to disable Telegram topic mode")
            return f"Failed to disable topic mode: {exc}"
        # Reset per-chat debounce state so the user doesn't see a stale
        # cooldown on the next activation.
        for attr in ("_telegram_lobby_reminder_ts", "_telegram_capability_hint_ts"):
            store = getattr(self, attr, None)
            if isinstance(store, dict):
                store.pop(chat_id, None)
        return (
            "Multi-session topic mode is now OFF for this chat.\n\n"
            "Existing topics in Telegram aren't removed — they'll just stop "
            "being gated as independent sessions. The root DM works as a "
            "normal Hermes chat again. Run /topic to re-enable later."
        )


    async def _telegram_topic_root_status_message(self, source: SessionSource) -> str:
        lines = [
            "Telegram multi-session topics are enabled.",
            "",
            "To create a new Hermes chat, open All Messages at the top of this "
            "bot interface and send any message there. Telegram will create a "
            "new topic for it.",
            "",
        ]
        try:
            sessions = await self._session_db.list_unlinked_telegram_sessions_for_user(
                chat_id=str(source.chat_id),
                user_id=str(source.user_id),
                limit=10,
            )
        except Exception:
            logger.debug("Failed to list unlinked Telegram sessions", exc_info=True)
            sessions = []

        if sessions:
            lines.append("Previous unlinked sessions:")
            for session in sessions:
                session_id = str(session.get("id") or "")
                title = str(session.get("title") or "Untitled session")
                preview = str(session.get("preview") or "").strip()
                line = f"- {title} — `{session_id}`"
                if preview:
                    line += f" — {preview}"
                lines.append(line)
            lines.extend([
                "",
                "To restore one:",
                "1. Create or open a topic. To create a new one, open All Messages and send any message there.",
                "2. Send /topic <session-id> inside that topic.",
                f"Example: Send /topic {sessions[0].get('id')} inside a topic.",
            ])
        else:
            lines.extend([
                "No previous unlinked Telegram sessions found.",
                "",
                "To restore a previous session later:",
                "1. Create or open a topic. To create a new one, open All Messages and send any message there.",
                "2. Send /topic <session-id> inside that topic.",
            ])
        return "\n".join(lines)

    async def _restore_telegram_topic_session(self, event: MessageEvent, raw_session_id: str) -> str:
        """Restore an existing Telegram-owned Hermes session into this topic."""
        source = event.source
        session_id = await self._session_db.resolve_session_id(raw_session_id.strip())
        if not session_id:
            return f"Session not found: {raw_session_id.strip()}"

        session = await self._session_db.get_session(session_id)
        if not session:
            return f"Session not found: {raw_session_id.strip()}"
        if str(session.get("source") or "") != "telegram":
            return "That session is not a Telegram session and cannot be restored into this topic."
        if str(session.get("user_id") or "") != str(source.user_id):
            return "That session does not belong to this Telegram user."

        linked = await self._session_db.is_telegram_session_linked_to_topic(session_id=session_id)
        current_binding = await self._session_db.get_telegram_topic_binding(
            chat_id=str(source.chat_id),
            thread_id=str(source.thread_id),
        )
        if linked:
            if not current_binding or current_binding.get("session_id") != session_id:
                return "That session is already linked to another Telegram topic."

        session_key = self._session_key_for_source(source)
        try:
            await self._session_db.bind_telegram_topic(
                chat_id=str(source.chat_id),
                thread_id=str(source.thread_id),
                user_id=str(source.user_id),
                session_key=session_key,
                session_id=session_id,
                managed_mode="restored",
            )
        except ValueError as exc:
            if "already linked" in str(exc):
                return "That session is already linked to another Telegram topic."
            raise

        title = await self._session_db.get_session_title(session_id) or session_id
        last_assistant = None
        try:
            for message in reversed(await self._session_db.get_messages(session_id)):
                if message.get("role") == "assistant" and message.get("content"):
                    last_assistant = str(message.get("content"))
                    break
        except Exception:
            last_assistant = None

        response = f"Session restored: {title}"
        if last_assistant:
            response += f"\n\nLast Hermes message:\n{last_assistant}"
        return response

    @staticmethod
    def _is_telegram_dm_topic_target(
        platform: Optional[Platform],
        chat_id: Optional[str],
        thread_id: Optional[str],
        *,
        chat_type: Optional[str] = None,
        adapter: Optional[Any] = None,
    ) -> bool:
        """Return True when a target is a Telegram private DM topic lane."""
        if platform != Platform.TELEGRAM or thread_id is None:
            return False
        if chat_type == "dm":
            return True
        # Inspect operator-declared DM topics via the adapter's lookup. Resolve
        # the method on the CLASS, not the instance: getattr() on a MagicMock
        # auto-creates a callable child for any attribute, so an instance-level
        # lookup would report a DM topic for every test double. Only a
        # dict-shaped return counts as an operator-declared topic — a bare
        # MagicMock or other sentinel must not. Mirrors the guard in
        # _rename_telegram_topic_for_session_title.
        if adapter is not None and chat_id:
            get_dm_topic_info = getattr(type(adapter), "_get_dm_topic_info", None)
            if callable(get_dm_topic_info):
                try:
                    topic_info = get_dm_topic_info(adapter, str(chat_id), str(thread_id))
                except Exception:
                    logger.debug("Failed to inspect Telegram DM topic metadata", exc_info=True)
                else:
                    return isinstance(topic_info, dict)
        return False

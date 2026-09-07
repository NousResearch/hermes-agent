"""Telegram forum-topic and Discord auto-thread binding/rename methods for GatewayRunner (MRO mixin).
``gateway.run`` internals are imported lazily inside method bodies (import cycle), so
``patch("gateway.run.X")`` keeps intercepting them at call time."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import re
import time
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

from agent.compaction_display import project_compaction_message_for_display
from agent.i18n import t
from gateway.config import Platform
from gateway.platforms.base import MessageEvent, _prefix_within_utf16_limit, utf16_len
from gateway.session import SessionSource
from utils import is_truthy_value

if TYPE_CHECKING:  # string annotations only; never imported at runtime (cycle)
    from gateway.run import GatewayRunner  # noqa: F401
    from gateway.run_turn_runner import TurnRunner  # noqa: F401

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")

_TOPIC_RESTORE_STEPS = (
    "1. Create or open a topic. To create a new one, open All Messages and send any message there.",
    "2. Send /topic <session-id> inside that topic.",
)


def _collapse_title(title: str) -> str:
    return re.sub(r"\s+", " ", str(title or "")).strip() or "Hermes Chat"


class GatewayTopicThreadsMixin:
    """Telegram forum-topic and Discord auto-thread binding/rename methods for GatewayRunner."""

    # ── Telegram topic mode: predicates and keys ────────────────────────────────────────────

    @staticmethod
    def _telegram_topic_profile_name(source: SessionSource) -> str:
        """Profile namespace for topic-mode rows: the profile stamped on the routed event, never the
        process-global one (under multiplex that mis-attributes state across bots sharing state.db).

        See #76423.
        """
        return str(getattr(source, "profile", None) or "").strip() or "default"

    def _sync_session_db(self):
        """The sync SessionDB handle, or None. Only for callers that provably run off-loop."""
        session_db = getattr(self, "_session_db", None)
        return None if session_db is None else getattr(session_db, "_db", session_db)

    @staticmethod
    def _is_telegram_dm(source: SessionSource) -> bool:
        return source.platform == Platform.TELEGRAM and source.chat_type == "dm"

    def _telegram_topic_mode_enabled(self, source: SessionSource) -> bool:
        """Return whether Telegram DM topic mode is active for this chat."""
        session_db = self._sync_session_db() if self._is_telegram_dm(source) else None
        if session_db is None:
            return False
        try:
            raw = session_db.is_telegram_topic_mode_enabled(
                chat_id=str(source.chat_id), user_id=str(source.user_id),
                profile_name=self._telegram_topic_profile_name(source),
            )
        except Exception:
            logger.debug("Failed to read Telegram topic mode state", exc_info=True)
            return False
        # Only a real True enables topic mode; anything else (including MagicMock from test
        # fixtures that didn't opt in) means off for this chat.
        return raw is True

    def _is_telegram_topic_root_lobby(self, source: SessionSource) -> bool:
        """True for the main Telegram DM (or General topic) when topic mode has made it a lobby."""
        return (
            self._is_telegram_dm(source) and self._telegram_topic_mode_enabled(source)
            and str(source.thread_id or "") in self._TELEGRAM_GENERAL_TOPIC_IDS
        )

    def _is_telegram_topic_lane(self, source: SessionSource) -> bool:
        """True for a user-created Telegram private-chat topic lane."""
        tid = str(source.thread_id or "")
        return (
            self._is_telegram_dm(source) and self._telegram_topic_mode_enabled(source)
            and bool(tid) and tid not in self._TELEGRAM_GENERAL_TOPIC_IDS
        )

    def _telegram_topic_cooldown_key(self, source: SessionSource) -> Optional[str]:
        """Cooldown key (profile, chat_id): profiles sharing a Telegram private chat_id under
        multiplex must not suppress each other's lobby reminders / capability hints.

        See #76423.
        """
        chat_id = str(source.chat_id or "")
        return f"{self._telegram_topic_profile_name(source)}:{chat_id}" if chat_id else None

    def _telegram_cooldown_elapsed(self, source: SessionSource, attr: str, cooldown_s: float) -> bool:
        """Per-(profile, chat) debounce: True (and stamp now) when the window has elapsed."""
        if not hasattr(self, attr):
            setattr(self, attr, {})
        key = self._telegram_topic_cooldown_key(source)
        if not key:
            return True
        stamps, now = getattr(self, attr), time.monotonic()
        if now - stamps.get(key, 0.0) < cooldown_s:
            return False
        stamps[key] = now
        return True

    def _should_send_telegram_lobby_reminder(self, source: SessionSource) -> bool:
        """Rate-limit root-DM lobby reminders to one per cooldown window, not one per prompt typed."""
        return self._telegram_cooldown_elapsed(source, "_telegram_lobby_reminder_ts", self._TELEGRAM_LOBBY_REMINDER_COOLDOWN_S)

    def _should_send_telegram_capability_hint(self, source: SessionSource) -> bool:
        """Rate-limit the BotFather Threads Settings screenshot (repeated /topic must not re-upload it)."""
        return self._telegram_cooldown_elapsed(source, "_telegram_capability_hint_ts", self._TELEGRAM_CAPABILITY_HINT_COOLDOWN_S)

    # ── Telegram topic mode: user-facing text ───────────────────────────────────────────────

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
        return (
            "Started a new Hermes session in this topic.\n\n"
            "Tip: for parallel work, open All Messages and send a message there "
            "to create a separate topic instead of using /new here. /new replaces "
            "the session attached to the current topic."
        ) if self._is_telegram_topic_lane(source) else None

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

    # ── Telegram topic bindings ─────────────────────────────────────────────────────────────

    def _record_telegram_topic_binding(self, source: SessionSource, session_entry) -> None:
        """Persist the Telegram topic -> Hermes session binding for topic lanes (off-loop)."""
        session_db = self._sync_session_db()
        if session_db is None or not source.chat_id or not source.thread_id:
            return
        session_db.bind_telegram_topic(
            chat_id=str(source.chat_id), thread_id=str(source.thread_id), user_id=str(source.user_id or ""),
            session_key=session_entry.session_key, session_id=session_entry.session_id,
            profile_name=self._telegram_topic_profile_name(source),
        )

    def _sync_telegram_topic_binding(self, source: SessionSource, session_entry, *, reason: str) -> None:
        """Update the topic binding to ``session_entry.session_id``: a stale binding after a mid-turn
        compression rotation reloads the oversized parent next message, retriggering compression.

        Telegram topic lanes persist a (chat_id, thread_id) -> session_id row so reopening a topic in a
        fresh process resumes the right Hermes session. See #20470, #29712, #33414.
        """
        if not self._is_telegram_topic_lane(source):
            return
        try:
            self._record_telegram_topic_binding(source, session_entry)
        except Exception:
            logger.debug("telegram topic binding refresh failed (%s)", reason, exc_info=True)

    def _recover_telegram_topic_thread_id(self, source: SessionSource) -> Optional[str]:
        """Pin lobby-shaped topic-mode DM replies (missing ``message_thread_id`` or General) to the
        user's most-recent bound topic. Never rewrite a non-lobby, unbound thread id: a brand-new DM
        topic is also "unknown" until its first message is recorded. None = leave the source alone."""
        if (
            not self._is_telegram_dm(source) or not source.chat_id or not source.user_id
            or not self._telegram_topic_mode_enabled(source)
        ):
            return None
        inbound = str(source.thread_id or "")
        if inbound and inbound not in self._TELEGRAM_GENERAL_TOPIC_IDS:
            return None
        session_db = self._sync_session_db()
        if session_db is None:
            return None
        try:
            bindings = session_db.list_telegram_topic_bindings_for_chat(
                chat_id=str(source.chat_id), profile_name=self._telegram_topic_profile_name(source)
            )
        except Exception:
            logger.debug("topic-recover: read failed", exc_info=True)
            return None
        for b in bindings or ():  # newest-first
            if str(b.get("user_id") or "") == str(source.user_id):
                recovered = str(b.get("thread_id") or "")
                return recovered if recovered and recovered != inbound else None
        return None

    # ── Telegram topic mode: /topic activation helpers ──────────────────────────────────────

    async def _get_telegram_topic_capabilities(self, source: SessionSource) -> dict:
        """Read Telegram private-topic capability flags via Bot API getMe."""
        bot = getattr(self._adapter_for_source(source), "_bot", None)
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
                return api_kwargs[name]
            return me.get(name) if isinstance(me, dict) else None

        return {"checked": True, **{k: _field(k) for k in ("has_topics_enabled", "allows_users_to_create_topics")}}

    async def _ensure_telegram_system_topic(self, source: SessionSource) -> None:
        """Create/pin the managed System topic after /topic activation when possible."""
        adapter = self._adapter_for_source(source)
        create_topic = getattr(adapter, "_create_dm_topic", None) if adapter is not None and source.chat_id else None
        if not callable(create_topic):
            return
        try:
            thread_id = await create_topic(int(source.chat_id), "System")
        except Exception:
            logger.debug("Failed to create Telegram System topic", exc_info=True)
            return
        if not thread_id:
            return
        try:
            send_result = await adapter.send(
                source.chat_id, "System topic for Hermes commands and status.", metadata={"thread_id": str(thread_id)},
            )
            message_id = getattr(send_result, "message_id", None)
        except Exception:
            logger.debug("Failed to send Telegram System topic intro", exc_info=True)
            return
        bot = getattr(adapter, "_bot", None)
        if not message_id or bot is None or not hasattr(bot, "pin_chat_message"):
            return
        try:
            await bot.pin_chat_message(chat_id=int(source.chat_id), message_id=int(message_id), disable_notification=True)
        except Exception:
            logger.debug("Failed to pin Telegram System topic intro", exc_info=True)

    async def _send_telegram_topic_setup_image(self, source: SessionSource) -> None:
        """Send the bundled BotFather Threads Settings screenshot when available."""
        adapter = self._adapter_for_source(source)
        image_path = Path(__file__).resolve().parent / "assets" / "telegram-botfather-threads-settings.jpg"
        if adapter is None or not source.chat_id or not hasattr(adapter, "send_image_file") or not image_path.exists():
            return
        try:
            await adapter.send_image_file(
                chat_id=source.chat_id, image_path=str(image_path), caption="BotFather → Bot Settings → Threads Settings",
                metadata={"thread_id": str(source.thread_id)} if source.thread_id else None,
            )
        except Exception:
            logger.debug("Failed to send Telegram topic setup image", exc_info=True)

    # ── title sanitizers ────────────────────────────────────────────────────────────────────

    def _sanitize_telegram_topic_title(self, title: str) -> str:
        """Bot API-safe forum topic name: names are 1-128 chars; keep room for multi-byte titles."""
        cleaned = _collapse_title(title)
        return cleaned if len(cleaned) <= 120 else cleaned[:117].rstrip() + "..."

    def _sanitize_discord_thread_title(self, title: str) -> str:
        """Discord-safe thread title: the 100-char cap is measured in UTF-16 code units (emoji count
        double), so truncate with the UTF-16 helpers rather than Python code-point slices."""
        cleaned = _collapse_title(title)
        return cleaned if utf16_len(cleaned) <= 80 else _prefix_within_utf16_limit(cleaned, 77).rstrip() + "..."

    # ── Discord auto-thread lanes ───────────────────────────────────────────────────────────

    def _is_discord_auto_thread_lane(self, source: SessionSource) -> bool:
        """Return True only for Discord threads Hermes just auto-created."""
        return (
            source.platform == Platform.DISCORD and source.chat_type == "thread"
            and bool(getattr(source, "auto_thread_created", False)) and bool(source.thread_id)
            and bool(getattr(source, "auto_thread_initial_name", None))
        )

    def _is_relay_discord_channel_lane(self, source: SessionSource) -> bool:
        """Shape-only check: a relay-delivered Discord CHANNEL event whose reply the connector MAY
        auto-thread (title-turn registration gate). Deliberately does NOT consult the send-result
        cache — before delivery the feedback can't exist; the rename lane polls it at fire time."""
        return (
            source.platform == Platform.DISCORD and bool(source.chat_id) and not source.thread_id
            and source.chat_type in ("group", "channel")
            and getattr(source, "delivered_via_upstream_relay", False) is True
        )

    def _relay_auto_thread_info(self, source: SessionSource) -> Optional[Tuple[str, str]]:
        """(thread_id, initial_name) when the RELAY connector auto-threaded our reply — the title-turn
        sibling of _is_discord_auto_thread_lane (whose markers only exist from turn 2 on; the title
        turn's source is the PARENT channel event).

        Preferred: the connector's per-message ``prospective_thread_id`` stamp (anchor message id ==
        the thread it will create), exact even when several auto-threads spawn from one channel;
        the connector's created-name guard enforces no-clobber. Fallback: the per-chat send-result
        cache (older connectors), which only ever renamed the FIRST thread.
        """
        from gateway.run import _as_thread_info
        if (
            source.platform != Platform.DISCORD or not source.chat_id
            or not getattr(source, "delivered_via_upstream_relay", False)
        ):
            return None
        prospective = getattr(source, "prospective_thread_id", None)
        if prospective:
            # Deterministic per-thread identity; the empty initial-name marker signals the caller
            # to rely on the connector-side no-clobber guard.
            return (str(prospective), "")
        info_fn = getattr(self._adapter_for_source(source), "auto_thread_info_for_chat", None)
        if not callable(info_fn):
            return None
        with suppress(Exception):
            return _as_thread_info(info_fn(str(source.chat_id)))
        return None

    async def _await_relay_auto_thread_info(self, source: SessionSource) -> Optional[Tuple[str, str]]:
        """``_relay_auto_thread_info``, waited out until this turn delivers (the legacy send-result
        path can only answer once the reply is sent; the caller asks at title time, one turn early).
        The timeout is only a backstop for a turn that never sends: the turn's own inactivity limit."""
        from gateway.run import _as_thread_info, _float_env
        # The connector-stamped prospective id is known at ingest, so most sessions answer here.
        known = self._relay_auto_thread_info(source)
        if known is not None:
            return known
        wait_fn = getattr(self._adapter_for_source(source), "wait_for_auto_thread_info", None)
        if not callable(wait_fn) or not source.chat_id:
            return None
        # 0 means the operator disabled the turn limit; the backstop still needs one.
        timeout = _float_env("HERMES_AGENT_TIMEOUT", 1800) or 1800
        with suppress(Exception):
            return _as_thread_info(await wait_fn(str(source.chat_id), timeout))
        return None

    async def _rename_discord_auto_thread_for_session_title(
        self, source: SessionSource, session_id: str, title: str,
        relay_info: Optional[Tuple[str, str]] = None,
    ) -> None:
        """Best-effort semantic rename of a newly auto-created Discord thread. ``relay_info`` is the
        connector's (thread_id, initial_name) feedback, supplied on the title turn where the source
        is the parent-channel event without auto-thread markers (see _relay_auto_thread_info)."""
        if relay_info is None and not await asyncio.to_thread(self._is_discord_auto_thread_lane, source):
            # Relay title turn with no feedback captured at schedule time: the title comes off the
            # user's opening message, so it beats the delivery that produces the connector's
            # send-result feedback by the whole length of the turn. None here = a true miss.
            if not self._is_relay_discord_channel_lane(source):
                return
            relay_info = await self._await_relay_auto_thread_info(source)
            if relay_info is None:
                return
        adapter = self._adapter_for_source(source) if getattr(self, "adapters", None) else None
        rename_thread = getattr(adapter, "rename_thread", None)
        if rename_thread is None:
            return
        relay = relay_info is not None
        target_thread_id = relay_info[0] if relay else str(source.thread_id)
        thread_name = self._sanitize_discord_thread_title(title)
        # Relay: ask the CONNECTOR to enforce the no-clobber guard from its own created-name memory —
        # the gateway can't reproduce the initial name byte-for-byte (normalization drift silently
        # declined every rename). Its egress guard resolves the tenant from caches keyed by the PARENT
        # channel chat_id, so pass it or the lookup misses. Native: the source IS the thread; guard on
        # the initial name.
        rename_kwargs = (
            {"prefer_connector_created": True, "parent_chat_id": str(source.chat_id) if source.chat_id else None}
            if relay else {"only_if_current_name": getattr(source, "auto_thread_initial_name", None)}
        )
        logger.info(
            "discord auto-thread rename: thread=%s lane=%s new_title=%r",
            target_thread_id, "relay" if relay else "native", thread_name,
        )
        try:
            renamed = await rename_thread(target_thread_id, thread_name, **rename_kwargs)
            logger.info("discord auto-thread rename result: thread=%s applied=%s", target_thread_id, bool(renamed))
        except TypeError:
            logger.warning(
                "Discord semantic thread rename raised TypeError (adapter=%s)", type(adapter).__name__, exc_info=True,
            )
        except Exception:
            logger.debug("Failed to rename Discord auto-thread for generated session title", exc_info=True)

    # ── title-thread → loop rename scheduling ───────────────────────────────────────────────

    def _schedule_rename_from_title_thread(self, source: SessionSource, make_coro, label: str) -> None:
        """Schedule a best-effort rename coroutine onto the gateway loop from the auto-title thread.
        The source is copied so the thread never shares the live dataclass with the loop; failures
        are logged at debug and never propagate."""
        from gateway.run import safe_schedule_threadsafe
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = getattr(self, "_gateway_loop", None)
        if loop is None or loop.is_closed():
            return
        copied_source = source
        with suppress(Exception):
            copied_source = dataclasses.replace(source)
        future = safe_schedule_threadsafe(
            make_coro(copied_source), loop, logger=logger, log_message=f"{label} failed to schedule",
        )

        def _log_rename_failure(fut) -> None:
            try:
                fut.result()
            except Exception:
                logger.debug("%s failed", label, exc_info=True)

        if future is not None:
            future.add_done_callback(_log_rename_failure)

    def _schedule_discord_semantic_thread_rename(self, source: SessionSource, session_id: str, title: str) -> None:
        """Schedule Discord auto-thread rename from the auto-title background thread."""
        if not title:
            return
        relay_info = None
        if not self._is_discord_auto_thread_lane(source):
            # Relay title turn: the source is the PARENT channel event (thread didn't exist at
            # ingest). The auto-title races the delivery that fills the send-result cache, so a
            # miss HERE is not a verdict: schedule whenever the SHAPE matches; the async rename
            # lane polls the cache (bounded wait) and no-ops on a true miss.
            relay_info = self._relay_auto_thread_info(source)
            if relay_info is None and not self._is_relay_discord_channel_lane(source):
                return
        self._schedule_rename_from_title_thread(
            source,
            lambda copied: self._rename_discord_auto_thread_for_session_title(
                copied, session_id, title, relay_info=relay_info
            ),
            "Discord semantic thread rename",
        )

    def _schedule_telegram_topic_title_rename(self, source: SessionSource, session_id: str, title: str) -> None:
        """Schedule a topic rename from the auto-title background thread."""
        if not title or not self._is_telegram_topic_lane(source) or self._telegram_topic_auto_rename_disabled(source):
            return
        self._schedule_rename_from_title_thread(
            source,
            lambda copied: self._rename_telegram_topic_for_session_title(copied, session_id, title),
            "Telegram topic title rename",
        )

    def _telegram_topic_auto_rename_disabled(self, source: SessionSource) -> bool:
        """``gateway.platforms.telegram.extra.disable_topic_auto_rename``; default False (auto-rename on)."""
        config = getattr(self, "config", None)
        platform_cfg = config.platforms.get(source.platform) if config and getattr(config, "platforms", None) else None
        if platform_cfg is None:
            return False
        return is_truthy_value((getattr(platform_cfg, "extra", None) or {}).get("disable_topic_auto_rename"))

    async def _rename_telegram_topic_for_session_title(self, source: SessionSource, session_id: str, title: str) -> None:
        """Best-effort rename of a Telegram DM topic when Hermes auto-titles a session."""
        if not await asyncio.to_thread(self._is_telegram_topic_lane, source) or not source.chat_id or not source.thread_id:
            return
        # Operator kill-switch, e.g. user-managed topics (ad-hoc Threaded Mode) that auto-rename
        # would keep overwriting.
        if self._telegram_topic_auto_rename_disabled(source):
            return
        # Skip operator-declared topics (extra.dm_topics): fixed names chosen by the operator;
        # auto-renaming would silently mutate operator config. Check the class, not the instance —
        # getattr() on a MagicMock auto-creates attributes, so every test double would match. Only
        # dict-shaped returns count; a bare MagicMock or other sentinel shouldn't.
        adapter = self._adapter_for_source(source)
        get_info = getattr(type(adapter), "_get_dm_topic_info", None) if adapter is not None else None
        if callable(get_info):
            try:
                operator_topic = get_info(adapter, str(source.chat_id), str(source.thread_id))
            except Exception:
                operator_topic = None
            if isinstance(operator_topic, dict):
                return
        session_db = getattr(self, "_session_db", None)
        if session_db is not None:
            try:
                binding = await session_db.get_telegram_topic_binding(
                    chat_id=str(source.chat_id), thread_id=str(source.thread_id),
                    profile_name=self._telegram_topic_profile_name(source),
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
                await rename_topic(chat_id=str(source.chat_id), thread_id=str(source.thread_id), name=topic_name)
                return
            bot = getattr(adapter, "_bot", None)
            edit_forum_topic = getattr(bot, "edit_forum_topic", None) or getattr(bot, "editForumTopic", None)
            if edit_forum_topic is None:
                return
            try:
                await edit_forum_topic(chat_id=int(source.chat_id), message_thread_id=int(source.thread_id), name=topic_name)
            except (TypeError, ValueError):
                await edit_forum_topic(chat_id=source.chat_id, message_thread_id=source.thread_id, name=topic_name)
        except Exception:
            logger.debug("Failed to rename Telegram topic for auto-generated title", exc_info=True)

    # ── /topic command bodies ───────────────────────────────────────────────────────────────

    async def _disable_telegram_topic_mode_for_chat(self, source: SessionSource) -> str:
        """Cleanly disable topic mode for a chat via /topic off."""
        if not self._session_db:
            from hermes_state import format_session_db_unavailable
            return format_session_db_unavailable(prefix=t("gateway.shared.session_db_unavailable_prefix"))
        chat_id = str(source.chat_id or "")
        if not chat_id:
            return "Could not determine chat ID."
        profile_name = self._telegram_topic_profile_name(source)
        currently_enabled = False
        with suppress(Exception):
            currently_enabled = await self._session_db.is_telegram_topic_mode_enabled(
                chat_id=chat_id, user_id=str(source.user_id or ""), profile_name=profile_name,
            )
        if not currently_enabled:
            return "Multi-session topic mode is not currently enabled for this chat."
        try:
            await self._session_db.disable_telegram_topic_mode(chat_id=chat_id, profile_name=profile_name)
        except Exception as exc:
            logger.exception("Failed to disable Telegram topic mode")
            return f"Failed to disable topic mode: {exc}"
        # Reset per-profile+chat debounce state so the next activation doesn't see a stale cooldown.
        # See #76423.
        cooldown_key = self._telegram_topic_cooldown_key(source)
        for attr in ("_telegram_lobby_reminder_ts", "_telegram_capability_hint_ts") if cooldown_key else ():
            store = getattr(self, attr, None)
            if isinstance(store, dict):
                store.pop(cooldown_key, None)
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
                chat_id=str(source.chat_id), user_id=str(source.user_id),
                profile_name=self._telegram_topic_profile_name(source), limit=10,
            )
        except Exception:
            logger.debug("Failed to list unlinked Telegram sessions", exc_info=True)
            sessions = []
        if sessions:
            lines.append("Previous unlinked sessions:")
            for session in sessions:
                preview = str(session.get("preview") or "").strip()
                lines.append(
                    f"- {session.get('title') or 'Untitled session'} — `{session.get('id') or ''}`"
                    + (f" — {preview}" if preview else "")
                )
            lines.extend(["", "To restore one:", *_TOPIC_RESTORE_STEPS, f"Example: Send /topic {sessions[0].get('id')} inside a topic."])
        else:
            lines.extend(["No previous unlinked Telegram sessions found.", "", "To restore a previous session later:", *_TOPIC_RESTORE_STEPS])
        return "\n".join(lines)

    async def _restore_telegram_topic_session(self, event: MessageEvent, raw_session_id: str) -> str:
        """Restore an existing Telegram-owned Hermes session into this topic."""
        source = event.source
        db = self._session_db
        session_id = await db.resolve_session_id(raw_session_id.strip())
        session = await db.get_session(session_id) if session_id else None
        if not session:
            return f"Session not found: {raw_session_id.strip()}"
        if str(session.get("source") or "") != "telegram":
            return "That session is not a Telegram session and cannot be restored into this topic."
        if str(session.get("user_id") or "") != str(source.user_id):
            return "That session does not belong to this Telegram user."
        linked = await db.is_telegram_session_linked_to_topic(session_id=session_id)
        topic_profile = self._telegram_topic_profile_name(source)
        current_binding = await db.get_telegram_topic_binding(
            chat_id=str(source.chat_id), thread_id=str(source.thread_id), profile_name=topic_profile,
        )
        already_linked = "That session is already linked to another Telegram topic."
        if linked and (not current_binding or current_binding.get("session_id") != session_id):
            return already_linked
        try:
            await db.bind_telegram_topic(
                chat_id=str(source.chat_id), thread_id=str(source.thread_id), user_id=str(source.user_id),
                session_key=self._session_key_for_source(source), session_id=session_id, managed_mode="restored",
                profile_name=topic_profile,
            )
        except ValueError as exc:
            if "already linked" in str(exc):
                return already_linked
            raise
        title = await db.get_session_title(session_id) or session_id
        last_assistant = None
        with suppress(Exception):
            for message in reversed(await db.get_messages(session_id)):
                if message.get("role") != "assistant":
                    continue
                projected = project_compaction_message_for_display(message)
                if projected is not None and projected.get("content"):
                    last_assistant = str(projected.get("content"))
                    break
        response = f"Session restored: {title}"
        return response + (f"\n\nLast Hermes message:\n{last_assistant}" if last_assistant else "")

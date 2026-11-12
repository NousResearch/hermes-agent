"""Telegram forum-topic and Discord auto-thread binding/rename methods for GatewayRunner (MRO mixin).
``hermes_agent.gateway.run`` internals are imported lazily inside method bodies (import cycle), so
``patch("hermes_agent.gateway.run.X")`` keeps intercepting them at call time."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

from agent.compaction_display import project_compaction_message_for_display
from agent.i18n import t
from hermes_agent.gateway.config import Platform
from hermes_agent.gateway.platforms.base import _prefix_within_utf16_limit, utf16_len
from hermes_agent.gateway.platforms.event import MessageEvent
from hermes_agent.gateway.session import SessionSource
from hermes_agent.gateway.session_identity import replace_source
from utils import is_truthy_value

if TYPE_CHECKING:  # string annotations only; never imported at runtime (cycle)
    from hermes_agent.gateway.run import GatewayRunner
    from hermes_agent.gateway.run_turn_runner import TurnRunner

# Log-record parity with the origin module.
logger = logging.getLogger("hermes_agent.gateway.run")
# Serializes a command's "route still mine?" check with its topic-binding write (process-wide: the
# critical section is one sqlite upsert).
_TOPIC_SWITCH_BIND_LOCK = threading.Lock()

class TopicLinkedElsewhere(RuntimeError):
    """A topic switch lost the race to a binding of its session to another topic; the route was
    put back. The message is the user-facing refusal."""


def _topic_restore_steps() -> tuple:
    return (t("gateway.topic.restore_step_1"), t("gateway.topic.restore_step_2"))


def _collapse_title(title: str) -> str:
    return re.sub(r"\s+", " ", str(title or "")).strip() or t("gateway.topic.default_title")


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
        """True for a user-created Telegram private-chat topic lane. Never a conversation its chat
        moved past (hermes_agent/gateway/session_detached.py): its turns neither heal nor rewrite the topic's
        binding, nor rename the topic, which belong to the chat's current conversation."""
        from hermes_agent.gateway.session_detached import detached_route_of
        tid = str(source.thread_id or "")
        return (
            detached_route_of(source) is None
            and self._is_telegram_dm(source) and self._telegram_topic_mode_enabled(source)
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
        return t("gateway.topic.root_lobby")

    def _telegram_topic_root_new_message(self) -> str:
        return t("gateway.topic.root_new")

    def _telegram_topic_new_header(self, source: SessionSource) -> Optional[str]:
        return t("gateway.topic.new_header") if self._is_telegram_topic_lane(source) else None

    def _telegram_topic_help_text(self) -> str:
        return t("gateway.topic.help")

    # ── Telegram topic bindings ─────────────────────────────────────────────────────────────

    def _record_telegram_topic_binding(
        self, source: SessionSource, session_entry, *, expected_session_id: Optional[str] = None,
        move_session: bool = False,
    ) -> None:
        """Persist the Telegram topic -> Hermes session binding for topic lanes (off-loop).
        ``expected_session_id`` and ``move_session``: see ``bind_telegram_topic``."""
        session_db = self._sync_session_db()
        if session_db is None or not source.chat_id or not source.thread_id:
            return
        session_db.bind_telegram_topic(
            chat_id=str(source.chat_id), thread_id=str(source.thread_id), user_id=str(source.user_id or ""),
            session_key=session_entry.session_key, session_id=session_entry.session_id,
            profile_name=self._telegram_topic_profile_name(source),
            expected_session_id=expected_session_id, move_session=move_session,
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
            # Not silent: a stale binding makes the next message's heal switch the topic back.
            logger.warning("telegram topic binding refresh failed (%s)", reason, exc_info=True)

    @staticmethod
    def _telegram_topic_key(source: SessionSource) -> tuple:
        return (str(source.chat_id), str(source.thread_id))

    @contextmanager
    def _telegram_topic_publication(self, source: SessionSource):
        """Brackets a command's route switch and its topic-binding write. Between the two the route
        names the new session while the binding still names the old one, and the binding heal would
        switch the route straight back; the heal reads this marker and never acts across it.
        Synchronous bookkeeping only: never spans the remote topic rename."""
        state = self.__dict__.setdefault("_topic_publications", {}).setdefault(
            self._telegram_topic_key(source), [0, 0])
        state[0] += 1  # in flight
        state[1] += 1  # publications begun (kept per topic so a finished one is still seen)
        try:
            yield
        finally:
            state[0] -= 1

    def _telegram_topic_publication_token(self, source: SessionSource) -> tuple:
        """(in flight, begun) for this topic; the heal snapshots it before reading the binding."""
        return tuple(self.__dict__.get("_topic_publications", {}).get(self._telegram_topic_key(source), (0, 0)))

    def _telegram_topic_publication_moved(self, source: SessionSource, token: tuple) -> bool:
        """True when a switch was in flight at ``token`` or began since: a binding read under it may
        predate that switch's own write, so acting on it would undo the switch."""
        return bool(token[0]) or self._telegram_topic_publication_token(source) != token

    async def _follow_telegram_topic_switch(self, source: SessionSource, session_entry, *, reason: str,
                                            departed, move: bool = False) -> bool:
        """A command that repoints a topic lane at another session (/branch --here, /resume, a
        handoff) rewrites the persisted topic binding, else the next message's binding heal switches
        the topic straight back to the old session (#58850). Call inside ``_telegram_topic_publication``
        right after the route switch off ``departed`` (the route's entry before it). False = a newer switch superseded this one
        (nothing written); True otherwise, including when ``source`` is not a topic lane.

        ``move`` (a handoff carries the conversation off whatever topic held it): the old topic's
        binding and route are released, so it starts afresh instead of sharing the session. Without
        it a session bound to another topic since the caller's conflict check is refused: the route
        goes back to ``departed`` and ``TopicLinkedElsewhere`` carries the refusal text."""
        if not await asyncio.to_thread(self._is_telegram_topic_lane, source):
            return True
        released = []
        try:
            holds = await asyncio.to_thread(self._bind_telegram_topic_if_route_holds, source, session_entry,
                                            reason=reason, move=move, released=released)
        except ValueError:
            await asyncio.to_thread(self.session_store.restore_route, departed,
                                    expected_session_id=session_entry.session_id)
            raise TopicLinkedElsewhere(t("gateway.topic.already_linked")) from None
        for key in released:  # the left topic's next message is a new conversation: none of its overrides
            self._clear_conversation_scope(key, reason="handoff_moved")
            self._evict_cached_agent(key)
        return holds

    async def _rename_telegram_topic_after_switch(self, source: SessionSource, session_entry, title: str) -> None:
        """Rename the topic to the session a switch moved it to. Per topic, renames land in switch
        order and only while the route still names ``session_entry``: an older /resume parked in its
        rename would otherwise land its title after a newer switch renamed the topic."""
        if not title or not await asyncio.to_thread(self._is_telegram_topic_lane, source):
            return
        from hermes_agent.gateway.session_authority import keyed_turn
        locks = self.__dict__.setdefault("_topic_switch_locks", {})
        async with keyed_turn(locks, self._telegram_topic_key(source)):
            route = await asyncio.to_thread(self.session_store.peek_session_id, session_entry.session_key)
            if route == session_entry.session_id:
                await self._rename_telegram_topic_for_session_title(source, session_entry.session_id, title)

    def _bind_telegram_topic_if_route_holds(self, source: SessionSource, session_entry, *, reason: str,
                                            move: bool = False, released: Optional[list] = None) -> bool:
        """Bind only while the route still names ``session_entry``: commands dispatch concurrently, and
        an older /resume that resumes after a newer one switched and bound must not restore its own
        session. One lock spans check + write, so the newer switch (which happens before its own
        guarded write) always writes last. False = superseded (nothing written). Raises ValueError
        when the session is bound to a topic it may not leave (another topic, unless ``move``).
        ``move``: the topic it leaves loses its route too, under the same lock (its key is appended
        to ``released``), so two topics never route to one session."""
        with _TOPIC_SWITCH_BIND_LOCK:
            if self.session_store.peek_session_id(session_entry.session_key) != session_entry.session_id:
                logger.info("Topic binding for %s not moved to %s (%s): the route switched again since",
                            session_entry.session_key, session_entry.session_id, reason)
                return False
            if not self._is_telegram_topic_lane(source):
                return True
            session_db = self._sync_session_db() if move else None
            left = session_db.get_telegram_topic_binding_by_session(session_id=session_entry.session_id) if (
                session_db is not None) else None
            try:
                self._record_telegram_topic_binding(source, session_entry, move_session=move)
            except ValueError:
                raise
            except Exception:
                logger.warning("telegram topic binding refresh failed (%s)", reason, exc_info=True)
                return True
            left_key = str((left or {}).get("session_key") or "")
            if left_key and left_key != session_entry.session_key and self.session_store.release_route(
                    left_key, session_entry.session_id) and released is not None:
                released.append(left_key)
            return True

    async def _handoff_repoint_with_topic_binding(self, dest, session_key: str, cli_session_id: str) -> None:
        """Handoff's route switch plus its topic-binding move. A Telegram DM topic lane follows its
        persisted binding, which would switch the route straight back to the topic's previous session
        on the next turn; the binding moves with the route, and only while the route still names the
        handoff. A handoff carries the conversation, so a topic still bound to it (an earlier handoff
        the user since resumed on the CLI) lets it go. Raises when a /new, /resume or /branch --here
        moved the topic meanwhile: the synthetic turn would load the handoff's history into that
        newer conversation, and failing keeps the CLI session."""
        prior = await self.async_session_store.get_or_create_session(dest.source)
        with self._telegram_topic_publication(dest.source):
            entry = await self._handoff_repoint(dest, session_key, cli_session_id)
            holds = await self._follow_telegram_topic_switch(dest.source, entry, reason="handoff",
                                                             departed=prior, move=True)
        if not holds:
            raise RuntimeError("the destination topic switched to another session during the handoff")

    async def _handoff_source_session_id(self, row: dict, profile_name: Optional[str]) -> str:
        """``row``'s session id once a Telegram handoff may take it: a session bound to another
        profile's topic is refused before a fresh topic is created, which would stay empty."""
        session_id = row["id"]
        if self._session_db is None or str(row.get("handoff_platform") or "").strip().lower() != "telegram":
            return session_id
        from hermes_agent.gateway.session_route_recovery import conversation_head
        _authority, _local_owner, head = await conversation_head(self, session_id)
        bound = await self._session_db.get_telegram_topic_binding_by_session(session_id=head)
        if bound and str(bound.get("profile_name") or "default") != (profile_name or "default"):
            raise RuntimeError(t("gateway.topic.already_linked"))
        return session_id

    async def _telegram_topic_switch_conflict(self, source: SessionSource, target_session_id: str) -> Optional[str]:
        """Refusal text when ``target_session_id`` is bound to ANOTHER topic: a session links to one
        topic only, so after the route moved the binding write would fail and the next message's heal
        would snap the topic back. Checked before the switch; None = this lane may take it."""
        if self._session_db is None or not await asyncio.to_thread(self._is_telegram_topic_lane, source):
            return None
        bound = await self._session_db.get_telegram_topic_binding_by_session(session_id=target_session_id)
        here = (self._telegram_topic_profile_name(source), str(source.chat_id), str(source.thread_id))
        if bound and (str(bound.get("profile_name")), str(bound.get("chat_id")), str(bound.get("thread_id"))) != here:
            return t("gateway.topic.already_linked")
        return None

    def _recover_telegram_topic_thread_id(self, source: SessionSource) -> Optional[str]:
        """Adapter-ingress recovery hook (sync, called off-loop before session keying): lobby recovery,
        except when ``dm_topic_auto_transfer`` is on — then the root-DM source is left alone so
        ``_transfer_telegram_root_dm`` can open a fresh topic (and fall back to this recovery itself)."""
        if self._telegram_topic_auto_transfer_enabled(source):
            return None
        return self._recover_telegram_lobby_thread_id(source)

    def _recover_telegram_lobby_thread_id(self, source: SessionSource) -> Optional[str]:
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

    def _telegram_dm_topic_auto_transfer_configured(self, source: SessionSource) -> bool:
        """The RECEIVING bot's ``extra.dm_topic_auto_transfer`` (default False): under multiplexing each
        bot owns its opt-in. Off keeps the v0.22 lobby recovery (#136567). No state.db read."""
        adapter = self._delivery_adapter_for(source)
        platform_cfg = getattr(adapter, "config", None) if adapter is not None else None
        return is_truthy_value((getattr(platform_cfg, "extra", None) or {}).get("dm_topic_auto_transfer"))

    def _telegram_topic_auto_transfer_enabled(self, source: SessionSource) -> bool:
        """The key is on and ``source`` is a ``/topic``-mode root DM (reads state.db: call off-loop)."""
        return self._telegram_dm_topic_auto_transfer_configured(source) and self._is_telegram_topic_root_lobby(source)

    async def _transfer_telegram_root_dm(self, source: SessionSource) -> Optional[tuple[str, bool]]:
        """Adapter-ingress transfer hook for an opted-in root-DM message (#139136).

        ``(thread_id, True)``: a fresh topic was created and the turn runs there. ``(thread_id, False)``:
        creating it failed, so the message falls back to lobby recovery (the last bound topic).
        ``None``: not applicable (key off, not a topic-mode root DM) or nothing to recover to.
        """
        if (not source.chat_id or not self._telegram_dm_topic_auto_transfer_configured(source)
                or not await asyncio.to_thread(self._is_telegram_topic_root_lobby, source)):
            return None
        adapter = self._delivery_adapter_for(source)
        create_topic = getattr(adapter, "_create_dm_topic", None) if adapter is not None else None
        thread_id = None
        if callable(create_topic):
            try:
                thread_id = await create_topic(int(source.chat_id), t("gateway.topic.default_title"))
            except Exception:
                logger.debug("Failed to auto-transfer Telegram root DM into a new topic", exc_info=True)
        if thread_id:
            logger.info("telegram topic auto-transfer: chat=%s user=%s root -> new topic %s",
                        source.chat_id, source.user_id, thread_id)
            return str(thread_id), True
        recovered = await asyncio.to_thread(self._recover_telegram_lobby_thread_id, source)
        return (recovered, False) if recovered else None

    # ── Telegram topic mode: /topic activation helpers ──────────────────────────────────────

    async def _get_telegram_topic_capabilities(self, source: SessionSource) -> dict:
        """Read Telegram private-topic capability flags via Bot API getMe."""
        bot = getattr(self._delivery_adapter_for(source), "_bot", None)
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

    async def _send_telegram_topic_setup_image(self, source: SessionSource) -> None:
        """Send the bundled BotFather Threads Settings screenshot when available."""
        adapter = self._delivery_adapter_for(source)
        image_path = Path(__file__).resolve().parent / "assets" / "telegram-botfather-threads-settings.jpg"
        if adapter is None or not source.chat_id or not hasattr(adapter, "send_image_file") or not image_path.exists():
            return
        try:
            await adapter.send_image_file(
                chat_id=source.chat_id, image_path=str(image_path), caption=t("gateway.topic.setup_image_caption"),
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

    def _recover_discord_auto_thread_source(self, source: SessionSource, session_key: str) -> SessionSource:
        """*source* with the auto-thread markers of the session's opening event, when that event
        created this same native Discord thread.

        Native Discord stamps the markers only on the opening event, so a later title retry on a
        rebuilt agent arrives unmarked. The routing entry's ``origin`` is that opening source and
        survives restarts. ``replace_source`` keeps the live transport owner and message id."""
        if (
            source.platform != Platform.DISCORD or source.chat_type != "thread" or not source.thread_id
            or source.delivered_via_upstream_relay or self._is_discord_auto_thread_lane(source)
        ):
            return source
        entry = self.session_store.lookup_by_session_key(session_key)
        if entry is None:
            return source

        def opens_this_thread(origin: Optional[SessionSource]) -> bool:
            return (
                origin is not None and str(origin.thread_id) == str(source.thread_id)
                and self._is_discord_auto_thread_lane(origin)
            )

        origin = entry.origin
        if not opens_this_thread(origin):
            # An entry rebuilt after a lost routing index holds an unmarked event, while the session
            # row keeps the opening origin until that entry's first peer refresh overwrites it.
            db = self.session_store._db_for_key(session_key)
            row = db.get_session(entry.session_id) if db is not None else None
            raw = (row or {}).get("origin_json")
            origin = None
            if raw:
                with suppress(ValueError, KeyError, TypeError):  # malformed/legacy row: no recovery
                    origin = SessionSource.from_dict(json.loads(raw))
            if not opens_this_thread(origin):
                return source
        return replace_source(
            source, auto_thread_created=True, auto_thread_initial_name=origin.auto_thread_initial_name,
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

    def _relay_auto_thread_info(self, source: SessionSource) -> Optional[tuple[str, str]]:
        """(thread_id, initial_name) when the RELAY connector auto-threaded our reply — the title-turn
        sibling of _is_discord_auto_thread_lane (whose markers only exist from turn 2 on; the title
        turn's source is the PARENT channel event).

        Preferred: the connector's per-message ``prospective_thread_id`` stamp (anchor message id ==
        the thread it will create), exact even when several auto-threads spawn from one channel;
        the connector's created-name guard enforces no-clobber. Fallback: the per-chat send-result
        cache (older connectors), which only ever renamed the FIRST thread.
        """
        from hermes_agent.gateway.run import _as_thread_info
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
        info_fn = getattr(self._delivery_adapter_for(source), "auto_thread_info_for_chat", None)
        if not callable(info_fn):
            return None
        with suppress(Exception):
            return _as_thread_info(info_fn(str(source.chat_id)))
        return None

    async def _await_relay_auto_thread_info(self, source: SessionSource) -> Optional[tuple[str, str]]:
        """``_relay_auto_thread_info``, waited out until this turn delivers (the legacy send-result
        path can only answer once the reply is sent; the caller asks at title time, one turn early).
        The timeout is only a backstop for a turn that never sends: the turn's own inactivity limit."""
        from hermes_agent.gateway.run import _as_thread_info, _float_env
        # The connector-stamped prospective id is known at ingest, so most sessions answer here.
        known = self._relay_auto_thread_info(source)
        if known is not None:
            return known
        wait_fn = getattr(self._delivery_adapter_for(source), "wait_for_auto_thread_info", None)
        if not callable(wait_fn) or not source.chat_id:
            return None
        # 0 means the operator disabled the turn limit; the backstop still needs one.
        timeout = _float_env("HERMES_AGENT_TIMEOUT", 1800) or 1800
        with suppress(Exception):
            return _as_thread_info(await wait_fn(str(source.chat_id), timeout))
        return None

    async def _rename_discord_auto_thread_for_session_title(
        self, source: SessionSource, session_id: str, title: str,
        relay_info: Optional[tuple[str, str]] = None,
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
        adapter = self._delivery_adapter_for(source) if getattr(self, "adapters", None) else None
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
        from hermes_agent.gateway.run import safe_schedule_threadsafe
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = getattr(self, "_gateway_loop", None)
        if loop is None or loop.is_closed():
            return
        copied_source = source
        with suppress(Exception):
            # Keep the live transport owner and identity; multiplex routes may run under a
            # profile that does not own the Discord adapter/token.
            from hermes_agent.gateway.session_identity import replace_source
            copied_source = replace_source(source)
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
        """The RECEIVING bot's ``extra.disable_topic_auto_rename`` (default False: auto-rename on); under
        multiplexing each bot owns its kill-switch, like ``dm_topic_auto_transfer``."""
        adapter = self._delivery_adapter_for(source)
        platform_cfg = getattr(adapter, "config", None) if adapter is not None else None
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
        adapter = self._delivery_adapter_for(source)
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
            from hermes_agent.hermes_state import format_session_db_unavailable
            return format_session_db_unavailable(prefix=t("gateway.shared.session_db_unavailable_prefix"))
        chat_id = str(source.chat_id or "")
        if not chat_id:
            return t("gateway.topic.no_chat_id")
        profile_name = self._telegram_topic_profile_name(source)
        currently_enabled = False
        with suppress(Exception):
            currently_enabled = await self._session_db.is_telegram_topic_mode_enabled(
                chat_id=chat_id, user_id=str(source.user_id or ""), profile_name=profile_name,
            )
        if not currently_enabled:
            return t("gateway.topic.mode_not_enabled")
        try:
            await self._session_db.disable_telegram_topic_mode(chat_id=chat_id, profile_name=profile_name)
        except Exception as exc:
            logger.exception("Failed to disable Telegram topic mode")
            return t("gateway.topic.disable_failed", error=exc)
        # Reset per-profile+chat debounce state so the next activation doesn't see a stale cooldown.
        # See #76423.
        cooldown_key = self._telegram_topic_cooldown_key(source)
        for attr in ("_telegram_lobby_reminder_ts", "_telegram_capability_hint_ts") if cooldown_key else ():
            store = getattr(self, attr, None)
            if isinstance(store, dict):
                store.pop(cooldown_key, None)
        return t("gateway.topic.mode_disabled")

    async def _telegram_topic_root_status_message(self, source: SessionSource) -> str:
        lines = [t("gateway.topic.status_enabled"), "", t("gateway.topic.status_how_to"), ""]
        try:
            sessions = await self._session_db.list_unlinked_telegram_sessions_for_user(
                chat_id=str(source.chat_id), user_id=str(source.user_id),
                profile_name=self._telegram_topic_profile_name(source), limit=10,
            )
        except Exception:
            logger.debug("Failed to list unlinked Telegram sessions", exc_info=True)
            sessions = []
        if sessions:
            lines.append(t("gateway.topic.previous_unlinked"))
            for session in sessions:
                preview = str(session.get("preview") or "").strip()
                lines.append(
                    f"- {session.get('title') or t('gateway.topic.untitled_session')} — `{session.get('id') or ''}`"
                    + (f" — {preview}" if preview else "")
                )
            lines.extend(["", t("gateway.topic.restore_one"), *_topic_restore_steps(),
                          t("gateway.topic.restore_example", session_id=sessions[0].get("id"))])
        else:
            lines.extend([t("gateway.topic.no_unlinked"), "", t("gateway.topic.restore_later"), *_topic_restore_steps()])
        return "\n".join(lines)

    async def _restore_telegram_topic_session(self, event: MessageEvent, raw_session_id: str) -> str:
        """Restore an existing Telegram-owned Hermes session into this topic."""
        source = event.source
        db = self._session_db
        session_id = await db.resolve_session_id(raw_session_id.strip())
        session = await db.get_session(session_id) if session_id else None
        if not session:
            return t("gateway.topic.session_not_found", session_id=raw_session_id.strip())
        if str(session.get("source") or "") != "telegram":
            return t("gateway.topic.not_telegram_session")
        if str(session.get("user_id") or "") != str(source.user_id):
            return t("gateway.topic.not_your_session")
        linked = await db.is_telegram_session_linked_to_topic(session_id=session_id)
        topic_profile = self._telegram_topic_profile_name(source)
        current_binding = await db.get_telegram_topic_binding(
            chat_id=str(source.chat_id), thread_id=str(source.thread_id), profile_name=topic_profile,
        )
        already_linked = t("gateway.topic.already_linked")
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
        # Restored titles bypass auto-titling, so rename the newly bound topic here.
        await self._rename_telegram_topic_for_session_title(source, session_id, title)
        last_assistant = None
        with suppress(Exception):
            for message in reversed(await db.get_messages(session_id)):
                if message.get("role") != "assistant":
                    continue
                projected = project_compaction_message_for_display(message)
                if projected is not None and projected.get("content"):
                    last_assistant = str(projected.get("content"))
                    break
        response = t("gateway.topic.restored", title=title)
        return response + (t("gateway.topic.restored_last_message", text=last_assistant) if last_assistant else "")

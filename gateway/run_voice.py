"""Voice-channel / auto-TTS methods for GatewayRunner (split out of ``gateway/run.py``; bound via
the MRO). ``gateway.run`` internals are imported lazily inside method bodies (import cycle), so
``patch("gateway.run.X")`` keeps intercepting them at call time."""

from __future__ import annotations

from pm import install_hint
import asyncio
import functools
import json
import logging
import os
import re
import time
import weakref
from contextlib import suppress
from difflib import SequenceMatcher
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional

from agent.i18n import t
from gateway.config import Platform
from gateway.platforms.base import build_auto_tts_output_path
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource

logger = logging.getLogger("gateway.run")  # log-record parity with the origin module

# Adapter-side per-chat auto-TTS override sets (``/voice off`` vs explicit ``/voice on``/``tts``).
_OFF_SET, _ON_SET = "_auto_tts_disabled_chats", "_auto_tts_enabled_chats"
_VOICE_MODES = {"off", "voice_only", "all"}
_CALL_ENDED_MODE = "off"


class GatewayVoiceMixin:
    _voice_call_keys: set[str]
    _adapter_profile_for_source: Callable[[SessionSource], Optional[str]]

    def _voice_key(self, platform: Platform, chat_id: str, profile: Optional[str] = None) -> str:
        """``<profile>:<platform>:<chat_id>`` under multiplexing (else two bots in one channel
        share a key and one ``/voice`` flips the other's); default keeps ``<platform>:<chat>``.

        Under multiplexing the key is additionally namespaced by the profile whose bot speaks in the chat
        (``<profile>:<platform>:<chat_id>``); the default profile keeps the historical
        ``<platform>:<chat_id>`` shape so persisted state stays valid. See #75198.
        """
        base = f"{platform.value}:{chat_id}"
        profile = profile.strip() if isinstance(profile, str) else ""
        return base if not profile or profile == "default" else f"{profile}:{base}"

    def _voice_key_for_source(self, source: SessionSource) -> str:
        """Voice mode belongs to the (bot, chat) pair: namespace is the profile that OWNS the
        receiving adapter, not the routed profile."""
        profile = self._adapter_profile_for_source(source)
        return self._voice_key(source.platform, source.chat_id, profile=profile)

    def _bind_voice_input_callback(self, adapter) -> None:
        """Route voice transcripts back through the adapter that captured them."""
        if hasattr(adapter, "_voice_input_callback"):
            adapter._voice_input_callback = functools.partial(
                self._handle_voice_channel_input, adapter=adapter)

    def _load_voice_modes(self) -> dict[str, str]:
        try:
            data = json.loads(self._VOICE_MODE_PATH.read_text(encoding="utf-8-sig"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}
        if not isinstance(data, dict):
            return {}
        items = {str(k): m for k, m in data.items() if m in _VOICE_MODES}
        for key in (k for k in items if ":" not in k):  # legacy unprefixed key: warn and skip
            logger.warning(
                "Skipping legacy unprefixed voice mode key %r during migration. "
                "Re-enable voice mode on that chat to rebuild the prefixed key.", key)
        return {k: m for k, m in items.items() if ":" in k}

    def _save_voice_modes(self) -> None:
        # This file is restart state: call-bound modes are off even while the live call speaks.
        persisted = {**self._voice_mode, **dict.fromkeys(self._voice_call_keys, _CALL_ENDED_MODE)}
        try:
            self._VOICE_MODE_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(persisted, indent=2)
            self._VOICE_MODE_PATH.write_text(payload, encoding="utf-8")
        except OSError as e:
            logger.warning("Failed to save voice modes: %s", e)

    @staticmethod
    def _toggle_adapter_auto_tts_set(adapter, chat_id: str, on: bool, *, enable: bool) -> None:
        """Add/discard ``chat_id`` in the adapter's enabled (``enable=True``) or disabled set;
        adding also clears the other set (``/voice off`` and ``/voice on``/``tts`` override)."""
        add_to, clear_from = (_ON_SET, _OFF_SET) if enable else (_OFF_SET, _ON_SET)
        if not isinstance(target := getattr(adapter, add_to, None), set):
            return
        if not on:
            target.discard(chat_id)
            return
        target.add(chat_id)
        if isinstance(other := getattr(adapter, clear_from, None), set):
            other.discard(chat_id)

    def _set_adapter_auto_tts_disabled(self, adapter, chat_id: str, disabled: bool) -> None:
        self._toggle_adapter_auto_tts_set(adapter, chat_id, disabled, enable=False)

    def _set_adapter_auto_tts_enabled(self, adapter, chat_id: str, enabled: bool) -> None:
        self._toggle_adapter_auto_tts_set(adapter, chat_id, enabled, enable=True)

    def _apply_voice_mode(
        self, adapter, voice_key: str, chat_id: str, mode: str, *, in_call: bool = False
    ) -> None:
        """Set the live mode and the adapter override. A call mode expires when the call ends."""
        self._voice_mode[voice_key] = mode
        if in_call:
            self._voice_call_keys.add(voice_key)
        else:
            self._voice_call_keys.discard(voice_key)
        self._save_voice_modes()
        self._toggle_adapter_auto_tts_set(adapter, chat_id, True, enable=mode != "off")

    def _sync_voice_mode_state_to_adapter(self, adapter) -> None:
        """Restore persisted /voice state into a live adapter: ``_auto_tts_default`` from
        ``voice.auto_tts``; enabled (voice_only/all) / disabled (off) sets from ``_voice_mode``."""
        platform = getattr(adapter, "platform", None)
        if not isinstance(platform, Platform):
            return
        chat_sets = [
            (chats, modes)
            for name, modes in ((_OFF_SET, {"off"}), (_ON_SET, {"voice_only", "all"}))
            if isinstance(chats := getattr(adapter, name, None), set)
        ]
        if not chat_sets:
            return
        try:
            from hermes_cli.config import load_config  # lazy: no gateway -> hermes_cli module dep
            auto_tts_default = bool((load_config().get("voice") or {}).get("auto_tts", False))
        except Exception:
            auto_tts_default = False
        if hasattr(adapter, "_auto_tts_default"):
            # A2A never inherits the global speak default. The flag is written by
            # Desktop "Read replies aloud" / voice.auto_tts and is meant for human
            # chat surfaces; an agent peer's replies must stay text (its adapter has
            # no native send_voice, so auto-TTS would synthesize an MP3 and then
            # fail delivery) — /voice scoped modes never applied to A2A anyway.
            if platform.value == "a2a":
                adapter._auto_tts_default = False
            else:
                adapter._auto_tts_default = auto_tts_default
        prefix = self._voice_key(platform, "", profile=getattr(adapter, "_owner_profile", None))
        for chats, modes in chat_sets:
            chats.clear()
            chats.update(key[len(prefix):] for key, mode in self._voice_mode.items()
                         if mode in modes and key.startswith(prefix))

    @staticmethod
    def _get_guild_id(event: MessageEvent) -> Optional[int]:
        raw = getattr(event, "raw_message", None)
        if getattr(raw, "guild_id", None):  # slash command interaction
            return int(raw.guild_id)
        return raw.guild.id if getattr(raw, "guild", None) else None  # regular message

    def _voice_scope_id(self, adapter, event: MessageEvent):
        """The id an adapter keys a live voice session on.

        Discord needs the guild, because a voice channel and the text channel its
        transcripts land in are two different objects. An adapter whose call lives *in* the
        chat says so with ``voice_scope = "chat"`` (MatrixRTC: the call belongs to the room
        it is about) and is keyed by chat id, which is a string and never absent.
        """
        if getattr(adapter, "voice_scope", "guild") == "chat":
            return event.source.chat_id
        return self._get_guild_id(event)

    @staticmethod
    def _bind_voice_session(adapter, scope_id, source: SessionSource) -> None:
        """Hand a chat-scoped adapter the live session its call should speak into.

        The live ``SessionSource``, not the ``to_dict()`` Discord round-trips through
        ``_voice_sources``: that round-trip drops the transport-adapter ref and
        ``role_authorized``, which is exactly what the voice path's authorization reads.
        """
        if getattr(adapter, "voice_scope", "guild") == "chat":
            adapter.bind_voice_session(scope_id, source)

    async def _handle_voice_channel_join(self, event: MessageEvent) -> str:
        adapter = self._delivery_adapter_for(event.source)
        if not hasattr(adapter, "join_voice_channel"):
            return t("gateway.voice.channel_unsupported")
        scope_id = self._voice_scope_id(adapter, event)
        if not scope_id:
            return t("gateway.voice.channel_guild_only")
        voice_channel = await adapter.get_user_voice_channel(scope_id, event.source.user_id)
        if not voice_channel:
            return t("gateway.voice.channel_not_in_voice")
        # Wire callbacks BEFORE join so voice input arriving right after connection is not lost.
        self._bind_voice_input_callback(adapter)
        self._bind_voice_session(adapter, scope_id, event.source)
        voice_profile = self._adapter_profile_for_source(event.source)
        if hasattr(adapter, "_on_voice_disconnect"):
            adapter._on_voice_disconnect = functools.partial(
                self._handle_voice_timeout_cleanup, adapter=adapter)
        # Let the adapter's inactivity timer see the live voice-reply mode so it doesn't
        # disconnect a deliberately text-only (/voice off) session.
        if hasattr(adapter, "_voice_mode_getter"):
            adapter._voice_mode_getter = lambda chat_id: self._voice_mode.get(
                self._voice_key(adapter.platform, str(chat_id), profile=voice_profile), "off")
        try:
            if getattr(adapter, "voice_scope", "guild") == "chat":
                success = await adapter.join_voice_channel(voice_channel)
            else:
                success = await adapter.join_voice_channel(
                    voice_channel, text_channel_id=int(event.source.chat_id),
                    source=event.source.to_dict())
        except Exception as e:
            logger.warning("Failed to join voice channel: %s", e)
            adapter._voice_input_callback = None
            if not any(tok in str(e).lower() for tok in ("pynacl", "nacl", "davey")):
                return t("gateway.voice.channel_join_failed", error=e)
            return t("gateway.voice.channel_deps_missing", hint=install_hint("discord"))
        if not success:
            adapter._voice_input_callback = None
            return t("gateway.voice.channel_join_permissions")
        voice_key = self._voice_key_for_source(event.source)
        mode = self._voice_mode.get(voice_key, "all")
        self._apply_voice_mode(adapter, voice_key, event.source.chat_id, mode, in_call=True)
        reply_key = {
            "voice_only": "gateway.voice.channel_joined_voice_only",
            "off": "gateway.voice.channel_joined_off",
        }.get(mode, "gateway.voice.channel_joined")
        return t(reply_key, name=voice_channel.name)

    async def _handle_voice_channel_leave(self, event: MessageEvent) -> str:
        """Hang up. A chat-scoped call is left whether or not we can still speak into it.

        ``is_in_voice_channel`` answers "is there a live connection in *this process*",
        which is the whole of a Discord call and only half of a MatrixRTC one: the
        membership that puts the bot in the call UI is room state, and it outlives the
        gateway. A chat-scoped adapter therefore decides for itself whether there is a
        call to leave, and its ``leave_voice_channel`` returns False when there is none.
        """
        adapter = self._delivery_adapter_for(event.source)
        scope_id = self._voice_scope_id(adapter, event)
        chat_scoped = getattr(adapter, "voice_scope", "guild") == "chat"
        if not (scope_id and hasattr(adapter, "leave_voice_channel")
                and hasattr(adapter, "is_in_voice_channel")
                and (chat_scoped or adapter.is_in_voice_channel(scope_id))):
            return t("gateway.voice.channel_not_joined")
        text_channel_id = scope_id if chat_scoped else adapter._voice_text_channels.get(scope_id)
        ended = None
        try:
            ended = await adapter.leave_voice_channel(scope_id)
            if not chat_scoped:
                text_channel_id = ended
        except Exception as e:
            logger.warning("Error leaving voice channel: %s", e)
        if chat_scoped and ended is False:
            return t("gateway.voice.channel_not_joined")
        if text_channel_id is not None:
            voice_profile = self._adapter_profile_for_source(event.source)
            key = self._voice_key(event.source.platform, str(text_channel_id), profile=voice_profile)
            self._apply_voice_mode(adapter, key, str(text_channel_id), _CALL_ENDED_MODE)
        if hasattr(adapter, "_voice_input_callback"):
            adapter._voice_input_callback = None
        return t("gateway.voice.channel_left")

    def _handle_voice_timeout_cleanup(self, chat_id: str, *, adapter) -> None:
        """Adapter callback when the adapter leaves a call on its own: switch the chat's
        voice mode off. ``adapter`` (bound at join) is that profile's bot."""
        key = self._voice_key(adapter.platform, chat_id,
                              profile=getattr(adapter, "_owner_profile", None))
        self._apply_voice_mode(adapter, key, chat_id, _CALL_ENDED_MODE)

    def _is_duplicate_voice_transcript(self, binding: tuple, user_id: int, transcript: str) -> bool:
        """Suppress repeated STT outputs for one recent utterance (voice capture can emit it twice a
        few seconds apart -> a second queued run and overlapping spoken replies). ``binding`` is
        (receiving bot's profile, guild, bound text channel): the store is runner-wide, and another
        bot's copy of the utterance, or a new binding, is a separate turn."""
        normalized = re.sub(r"[^\w\s]", "", re.sub(r"\s+", " ", transcript).strip().lower())
        if not normalized:
            return False
        now, key = time.monotonic(), (*binding, user_id)
        if not isinstance(recent_store := getattr(self, "_recent_voice_transcripts", None), dict):
            recent_store = self._recent_voice_transcripts = {}
        recent = [(ts, txt) for ts, txt in recent_store.get(key, []) if now - ts <= 12.0]
        if any(prior == normalized or (min(len(prior), len(normalized)) >= 16
                                       and SequenceMatcher(None, prior, normalized).ratio() >= 0.95)
               for _, prior in recent):
            recent_store[key] = recent
            return True
        recent_store[key] = (recent + [(now, normalized)])[-5:]
        return False

    @staticmethod
    def _cached_user_display_name(client, user_id: int) -> Optional[str]:
        """``get_member`` is cache-only; a speaker it misses may still be a cached user, whose display
        name (global name or username, no server nickname) beats a bare id."""
        get_user = getattr(client, "get_user", None)
        name = getattr(get_user(int(user_id)) if callable(get_user) else None, "display_name", None)
        return name if isinstance(name, str) and name else None

    @staticmethod
    def _voice_input_source(adapter, guild_id: int, user_id: int, text_ch_id) -> SessionSource:
        """Bound text channel's own source when available (voice shares the text conversation's
        session), else a synthetic one."""
        # The speaker's display name, as their typed messages carry it: the pinned session-context
        # prompt renders it, so a bare id here re-rendered it on every spoken/typed switch.
        client = getattr(adapter, "_client", None)
        guild = client.get_guild(guild_id) if client else None
        member = guild.get_member(int(user_id)) if guild else None
        display_name = getattr(member, "display_name", None)
        user_name = display_name if isinstance(display_name, str) and display_name else None
        channel = client.get_channel(int(text_ch_id)) if client else None
        if source_data := getattr(adapter, "_voice_sources", {}).get(guild_id):
            source = SessionSource.from_dict(source_data)
            # get_member() is cache-only: on a miss the /voice join invoker keeps the name bound at
            # join, never another participant's.
            if user_name is None and source.user_id == str(user_id):
                user_name = source.user_name
            user_name = user_name or GatewayVoiceMixin._cached_user_display_name(client, user_id)
            source.user_id, source.user_name = str(user_id), user_name or str(user_id)
            # The bound source is the `/voice join` message's; its id is not this turn's trigger
            # (run.py exports it as HERMES_SESSION_MESSAGE_ID for reply anchoring).
            source.message_id = None
        else:
            # A programmatic join binds no source: build the one a typed message in that channel carries
            # (group, or thread under its parent, with the guild), else the turn keys another session.
            user_name = user_name or GatewayVoiceMixin._cached_user_display_name(client, user_id)
            thread_id = None
            if channel is not None and callable(split := getattr(adapter, "_thread_id_and_chat_for_channel", None)):
                thread_id, _ = split(channel)
            parent_id = adapter._get_parent_channel_id(channel) if thread_id else None
            source = adapter.build_source(
                chat_id=str(text_ch_id), chat_type="thread" if thread_id else "group", user_id=str(user_id),
                user_name=user_name or str(user_id), thread_id=thread_id, guild_id=str(guild_id),
                parent_chat_id=parent_id)
        # The stored source is a join-time copy; a typed message reads the channel's current name and
        # topic, so a rename or topic edit would otherwise flip the pinned prompt on every switch.
        if channel is not None and callable(labels := getattr(adapter, "_guild_channel_labels", None)):
            if isinstance(current := labels(channel), tuple):
                source.chat_name, source.chat_topic = current
        # Serialization drops transport provenance; auth must still follow the receiving bot.
        source._transport_adapter_ref = weakref.ref(adapter)
        return source

    async def _handle_voice_channel_input(
        self, guild_id: int, user_id: int, transcript: str, *, adapter=None
    ):
        """Handle transcribed voice from a voice channel. ``adapter`` captured the audio; under
        multiplexing each profile's bot dispatches through its own adapter, never the default's."""
        if adapter is None:
            adapter = self.adapters.get(Platform.DISCORD)
        text_ch_id = adapter._voice_text_channels.get(guild_id) if adapter else None
        if not text_ch_id:
            return
        source = self._voice_input_source(adapter, guild_id, user_id, text_ch_id)
        # The cached source still carries the previous speaker's identity (per-sender routes,
        # #106019): drop the pin so the seam re-resolves for THIS speaker.
        from gateway.session_identity import clear_identity
        clear_identity(source)
        if self._canonicalize(source, transport_profile=getattr(adapter, "_owner_profile", None)) is None:
            logger.warning("Dropping voice input: its profile route targets an unserved profile")
            return
        # Validate the session owner against the current allowlist before auto-resuming. A session created
        # before TELEGRAM_ALLOWED_USERS (or equivalent) was configured, or before the owner was removed from
        # it, must not silently receive a full agent response on gateway restart just because it has a
        # resume-pending marker (issue #23778).
        # The /voice join copy never carries the per-event role grant (to_dict drops it), so a
        # role-only speaker was always refused here. Recompute it for THIS speaker against the
        # guild's current member, with the flag's message-path meaning, never the joiner's.
        if isinstance(roles := getattr(adapter, "_allowed_role_ids", None), (set, frozenset)) and roles:
            client = getattr(adapter, "_client", None)
            guild = client.get_guild(guild_id) if client else None
            source.role_authorized = guild is not None and adapter._is_allowed_user(
                str(user_id), guild=guild, is_dm=False) is True
        if not self._is_user_authorized_for_source(source):
            logger.debug("Unauthorized voice input from user %d, ignoring", user_id)
            return
        binding = (getattr(adapter, "_owner_profile", None), guild_id, text_ch_id)
        if self._is_duplicate_voice_transcript(binding, user_id, transcript):
            logger.info("Suppressing duplicate voice transcript for guild=%s user=%s: %s",
                        guild_id, user_id, transcript[:100])
            return
        # Echo the transcript into the text channel (after auth, with mention sanitization).
        with suppress(Exception):
            channel = adapter._client.get_channel(text_ch_id)
            if channel:
                safe_text = transcript[:2000].replace("@everyone", "@\u200beveryone")
                safe_text = safe_text.replace("@here", "@\u200bhere")
                await channel.send(t("gateway.voice.transcript_echo", user=user_id, text=safe_text))
        # Bound text channel's channel_prompt and skills: voice input gets the same per-channel
        # context, and a first spoken turn opens the session, the only point skills load.
        # A thread inherits its parent's bindings, as its typed messages do.
        channel_prompt = auto_skill = None
        parent_id = source.parent_chat_id or None
        if callable(resolver := getattr(adapter, "_resolve_channel_prompt", None)):
            with suppress(Exception):
                resolved = resolver(str(text_ch_id), parent_id)
                channel_prompt = resolved if isinstance(resolved, str) else None
        if callable(skills := getattr(adapter, "_resolve_channel_skills", None)):
            bound = skills(str(text_ch_id), parent_id)
            auto_skill = bound if isinstance(bound, list) else None
        # Synthetic MessageEvent for the normal pipeline; the SimpleNamespace raw_message lets
        # _get_guild_id() extract guild_id so _send_voice_reply() plays audio in the voice channel.
        event = MessageEvent(
            source=source, text=transcript, message_type=MessageType.VOICE,
            raw_message=SimpleNamespace(guild_id=guild_id, guild=None),
            channel_prompt=channel_prompt, auto_skill=auto_skill)
        await adapter.handle_message(event)

    def _should_send_voice_reply(
        self, event: MessageEvent, response: str, agent_messages: list, already_sent: bool = False
    ) -> bool:
        """False when voice_mode is off for this chat, the response is empty/an error, the agent
        already called text_to_speech this turn, or voice input + base adapter auto-TTS handled it
        — UNLESS streaming consumed the response (already_sent): then the runner must do it."""
        if not response or response.startswith("Error:"):
            return False
        # A2A is agent-to-agent text. The adapter has no native send_voice, so
        # global voice.auto_tts (Desktop "Read replies aloud") would synthesize
        # an MP3 and then fail delivery with "Couldn't deliver the audio
        # attachment." — the peer sees the failure instead of the text reply the
        # agent already produced (#90103). Keep /voice scoped to human platforms.
        if getattr(event.source.platform, "value", None) == "a2a":
            return False
        chat_id = event.source.chat_id
        voice_mode = self._voice_mode.get(self._voice_key_for_source(event.source))
        is_voice_input = event.message_type == MessageType.VOICE
        adapter = self._delivery_adapter_for(event.source)
        adapter_auto_tts = False
        with suppress(Exception):  # adapters without the probe read as False
            adapter_auto_tts = bool(adapter._should_auto_tts_for_chat(chat_id))
        # ``voice.auto_tts`` (synced into the adapter at startup) is the fallback only when the
        # chat has no explicit mode; the chat-level all/voice_only/off choice takes precedence.
        if not (voice_mode == "all" or (voice_mode == "voice_only" and is_voice_input)
                or (voice_mode is None and adapter_auto_tts)):
            logger.debug(
                "Auto voice reply skipped: mode=%s adapter_auto_tts=%s chat=%s platform=%s",
                voice_mode, adapter_auto_tts, chat_id, event.source.platform.value)
            return False
        # Dedup: agent already called the TTS tool in THIS turn (from the last user message on).
        start = next((i for i, m in reversed(list(enumerate(agent_messages)))
                      if m.get("role") == "user"), 0)
        if any((tc.get("function") or {}).get("name") == "text_to_speech"
               for msg in agent_messages[start:] if msg.get("role") == "assistant"
               for tc in (msg.get("tool_calls") or [])):
            return False
        # Dedup: base adapter auto-TTS already handles voice input (play_tts plays in VC when
        # connected) — unless streaming consumed the text (already_sent): then the runner must.
        return not (is_voice_input and not already_sent)

    def _should_echo_stt_transcripts(self) -> bool:
        return bool(getattr(self.config, "stt_echo_transcripts", True))

    async def _send_voice_reply(self, event: MessageEvent, text: str) -> None:
        """Generate TTS audio and send as a voice message before the text reply. The TTS tool
        may return one combined file or several separately valid ones (combination unavailable /
        over a platform limit); legacy single-file results keep working."""
        audio_path, actual_paths = None, []
        try:
            from tools.tts_text_normalize import _strip_markdown_for_tts
            from tools.tts_tool import text_to_speech_tool
            tts_text = _strip_markdown_for_tts(text)
            if not tts_text:
                return
            # Platforms whose native voice bubbles require Ogg/Opus (OPUS_VOICE_PLATFORMS) get an
            # explicit .ogg path; the TTS tool's container repair guarantees real Ogg/Opus bytes.
            audio_path = build_auto_tts_output_path(event.source.platform)
            raw = await asyncio.to_thread(text_to_speech_tool, text=tts_text,
                                          output_path=audio_path)
            try:
                result = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Auto voice reply TTS returned invalid JSON: %s",
                               raw[:200] if raw else raw)
                return
            candidates = result.get("file_paths") or [result.get("file_path", audio_path)]
            paths = [str(p) for p in candidates if p and os.path.isfile(p)]
            if not result.get("success") or not paths:
                logger.warning("Auto voice reply TTS failed: %s", result.get("error"))
                return
            actual_paths = paths
            await self._deliver_voice_reply(event, actual_paths)
        except Exception as e:
            logger.warning("Auto voice reply failed: %s", e, exc_info=True)
        finally:
            for p in ({audio_path, *actual_paths} - {None}):
                with suppress(OSError):
                    os.unlink(p)

    async def _deliver_voice_reply(self, event: MessageEvent, audio_paths: list[str]) -> None:
        """Play the files in the connected voice channel, else send them as voice messages."""
        adapter = self._delivery_adapter_for(event.source)
        scope_id = self._voice_scope_id(adapter, event)
        play = getattr(adapter, "play_in_voice_channel", None)
        is_in_vc = getattr(adapter, "is_in_voice_channel", None)
        if scope_id and callable(play) and callable(is_in_vc) and is_in_vc(scope_id):
            for path in audio_paths:
                await play(scope_id, path)
            return
        if not callable(send_voice := getattr(adapter, "send_voice", None)):
            return
        reply_anchor = self._reply_anchor_for_event(event)
        # notify=True mirrors the final-text path in platforms/base.py so notification-gating
        # adapters (Telegram "important" mode) deliver it. Clone: shared w/ typing-indicator state.
        thread_meta = dict(self._thread_metadata_for_source(event.source, reply_anchor) or {})
        thread_meta["notify"] = True
        for path in audio_paths:
            await send_voice(chat_id=event.source.chat_id, audio_path=path, reply_to=reply_anchor,
                             metadata=thread_meta)

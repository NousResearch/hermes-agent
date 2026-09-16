"""Realtime (xAI S2S) supervisor-voice methods for ``GatewayRunner``.

Lives beside ``gateway/run.py`` as a mixin (same shape as
``gateway/authz_mixin.py``) so the god-file does not grow the realtime glue:
``self.*`` resolves via the MRO and ``gateway.run`` is only imported lazily
inside methods (for the shared ``"gateway.run"`` logger), so there is no
import cycle.

The mixin owns the per-(profile, guild) supervisor-controller lifecycle plus
the speaker-ownership predicates the reply/ack/TTS paths consult. Every
predicate resolves the Discord adapter from the SOURCE (``_adapter_for_source``),
never from ``self.adapters`` directly: under ``gateway.multiplex_profiles`` a
secondary profile's bot has its own adapter, voice bindings and realtime
session, and the default adapter knows nothing about them.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from agent.voice_supervisor import VoiceSupervisorController
from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.voice_realtime_bridge import DiscordVoiceTurnRunner
from hermes_cli.config import read_raw_config
from tools.voice_realtime_config import RealtimeConfig, load_realtime_config


def _voice_controller_key(adapter: Any, guild_id: int) -> Tuple[Optional[str], int]:
    """Controllers are keyed per owning profile AND guild: two profiles' bots may sit in the
    same guild, each with its own realtime session (same namespacing as ``_voice_key``; the
    default profile's ``_owner_profile`` is None)."""
    return (getattr(adapter, "_owner_profile", None), guild_id)


class GatewayVoiceRealtimeMixin:
    """Mixin holding GatewayRunner's realtime supervisor-voice glue."""

    def _voice_realtime_controller(self, adapter: Any, guild_id: Optional[int]):
        """The live controller for *adapter*'s session in *guild_id*, or None.

        A controller whose session died (xAI socket gone for good) is evicted here: the adapter
        tears its side down without telling the gateway, and no further function call — the other
        eviction path — can arrive from a dead session. Left in place it would keep claiming the
        guild's turns (``consult_active``) and silence the classic pipeline's acks and progress.
        """
        if guild_id is None:
            return None
        controllers = getattr(self, "_voice_realtime_controllers", None) or {}
        key = _voice_controller_key(adapter, guild_id)
        controller = controllers.get(key)
        if controller is None:
            return None
        if not controller.session.alive:
            controllers.pop(key, None)
            controller.fail_active_consult("Voice session ended.")
            controller.reset()
            return None
        return controller

    def _handle_voice_channel_function_call(
        self, guild_id: int, name: str, call_id: str, args_json: str, *, adapter=None
    ) -> None:
        """Dispatch a consult/steer tool call from a realtime voice session.

        Registered as the Discord adapter's ``_voice_function_call_callback``
        (bound to that adapter, like ``_voice_input_callback``: under
        multiplexing each profile's bot owns its own realtime session) and
        invoked from realtime-session threads — everything loop-bound is
        bridged inside the controller's TurnRunner.
        """
        from gateway.run import logger  # lazy: keep the "gateway.run" logger name

        try:
            controller = self._ensure_voice_realtime_controller(guild_id, adapter=adapter)
        except Exception:
            logger.warning("voice realtime controller build failed", exc_info=True)
            return
        if controller is None:
            logger.warning(
                "voice function call %s dropped: no realtime session for guild %d",
                name, guild_id,
            )
            return
        controller.on_function_call(name, call_id, args_json)

    def _ensure_voice_realtime_controller(self, guild_id: int, *, adapter=None):
        """Return the supervisor controller for a guild, (re)building it when
        the adapter's session changed (VC reconnects create fresh sessions).
        ``adapter`` is the Discord adapter whose session fired the call."""
        if adapter is None:
            adapter = self.adapters.get(Platform.DISCORD)
        if adapter is None or not hasattr(adapter, "voice_realtime_session"):
            return None
        key = _voice_controller_key(adapter, guild_id)
        session = adapter.voice_realtime_session(guild_id)
        if session is None:
            old = self._voice_realtime_controllers.pop(key, None)
            if old is not None:
                old.fail_active_consult("Voice session ended.")
                old.reset()
            return None
        controller = self._voice_realtime_controllers.get(key)
        if controller is not None and controller.session is session:
            return controller
        if controller is not None:
            controller.fail_active_consult(
                "Voice session reconnected; the previous task was dropped."
            )
            controller.reset()
        loop = getattr(self, "_gateway_loop", None)
        if loop is None:
            return None
        runner = DiscordVoiceTurnRunner(self, adapter, guild_id, loop)
        controller = VoiceSupervisorController(
            session, runner, narrate=self._voice_realtime_config().narrate_progress
        )
        self._voice_realtime_controllers[key] = controller
        return controller

    @staticmethod
    def _voice_realtime_config() -> RealtimeConfig:
        """The ``voice.realtime`` section as the gateway reads it (raw YAML, not ``DEFAULT_CONFIG``)."""
        return load_realtime_config((read_raw_config() or {}).get("voice"))

    def _voice_discord_text_mirror(self, controller=None, *, reload: bool = True) -> bool:
        """True when consult replies should also post to the bound text channel.

        Prefers the ``RealtimeConfig`` already loaded on the live session; without one, only
        consult completion re-reads config (``reload=True``) — the per-tool progress path never
        hits disk and treats "no session" as voice-only.
        """
        session = getattr(controller, "session", None)
        cfg = getattr(session, "_cfg", None)
        if cfg is not None:
            return cfg.discord_text_mirror
        if not reload:
            return False
        return self._voice_realtime_config().discord_text_mirror

    def _voice_realtime_controller_for_event(self, event: MessageEvent):
        """The live controller whose consult could own this turn, or None."""
        if (
            not getattr(self, "_voice_realtime_controllers", None)
            or event.source.platform != Platform.DISCORD
        ):
            return None
        if event.message_type != MessageType.VOICE and not event.voice_consult:
            return None
        guild_id = self._get_guild_id(event)
        if not guild_id:
            return None
        return self._voice_realtime_controller(self._adapter_for_source(event.source), guild_id)

    def _voice_realtime_controller_for_ack(self, ctx):
        """The controller owning a running turn's voice-ack guild (``ctx._voice_ack_guild``), or
        None. The guild alone is ambiguous under multiplexing; the turn's source names the bot."""
        ack = getattr(ctx, "_voice_ack_guild", None)
        guild_id = ack[0] if ack else None
        if guild_id is None:
            return None
        adapter = self._adapter_for_source(getattr(ctx, "source", None))
        return self._voice_realtime_controller(adapter, guild_id)

    def _notify_voice_realtime_blocked(self, ctx, text: str) -> None:
        """Speak a blocking prompt while a Discord supervisor consult runs."""
        controller = self._voice_realtime_controller_for_ack(ctx)
        if controller is not None and controller.consult_active:
            controller.notify(text)

    def _voice_realtime_binding_for_chat(self, source: Any) -> Optional[Tuple[Any, int]]:
        """``(adapter, guild_id)`` of the Discord bot whose voice connection is bound to this
        source's text channel (``_voice_text_channels`` on the SOURCE's own adapter), or None for
        non-Discord sources and unbound chats.

        Voice turns and supervisor consults run in the bound text channel's own session, so one
        lookup serves typed and spoken turns alike.
        """
        if getattr(source, "platform", None) != Platform.DISCORD:
            return None
        adapter = self._adapter_for_source(source)
        chat_id = str(getattr(source, "chat_id", "") or "")
        text_channels = getattr(adapter, "_voice_text_channels", None) if adapter else None
        if isinstance(text_channels, dict):
            for guild_id, bound_chat_id in text_channels.items():
                if str(bound_chat_id) == chat_id:
                    return adapter, guild_id
        return None

    def _voice_realtime_guild_for_chat(self, source: Any) -> Optional[int]:
        """The guild of :meth:`_voice_realtime_binding_for_chat`, or None."""
        binding = self._voice_realtime_binding_for_chat(source)
        return binding[1] if binding is not None else None

    def _voice_supervisor_owns_chat(self, source) -> bool:
        """True when a live realtime-supervisor session owns this chat's
        speaker. grok-voice speaks its own replies and consult summaries;
        classic TTS (typed-message replies included) would talk over it and
        cannot be interrupted by voice. The ears brain returns False — it
        relies on classic TTS for every reply."""
        binding = self._voice_realtime_binding_for_chat(source)
        if binding is None:
            return False
        adapter, guild_id = binding
        brain_getter = getattr(adapter, "voice_realtime_brain", None)
        if not callable(brain_getter):
            return False
        try:
            return brain_getter(guild_id) == "supervisor"
        except Exception:
            return False

    def _voice_consult_owns_turn(self, source, message_type, message) -> bool:
        """True when this turn is an active realtime-supervisor consult.

        The voice model speaks for that turn (ack, narration, summary) —
        every classic TTS path must stay silent, including streaming TTS
        which starts before the turn completes. Resolves the guild through
        the adapter's text-channel binding since ``_run_agent_inner`` has no
        event object.
        """
        if not getattr(self, "_voice_realtime_controllers", None):
            return False
        if str(getattr(message_type, "value", message_type) or "").lower() not in (
            "voice",
            "text",
        ):
            return False
        binding = self._voice_realtime_binding_for_chat(source)
        if binding is None:
            return False
        controller = self._voice_realtime_controller(*binding)
        return controller is not None and controller.owns_turn(message)


__all__ = ["GatewayVoiceRealtimeMixin"]

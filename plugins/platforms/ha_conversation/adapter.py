"""Home Assistant conversation-agent platform adapter (Wyoming handle).

ONE shared gateway session (chat_id "home"): HA's Assist hardware does
wake/STT/TTS; each utterance arrives as a Transcript over a fresh TCP
connection and is answered with a single Handled reply.

Reply correlation is by AWAIT-WINDOW, never queue position: utterances are
serialized behind a lock, and every send() that arrives while this adapter
is awaiting the agent's turn belongs to that turn (buffered, joined, spoken
once). A send() outside any window is by definition an announcement
(cron/background delivery) and follows announce_mode.

Note on ``handle_message()``: ``BasePlatformAdapter.handle_message()`` is
fire-and-forget by design (it spawns ``_process_message_background`` as a
background asyncio.Task and returns immediately, so a busy adapter can keep
receiving/interrupting). Awaiting it alone would NOT wait for the agent's
reply, so ``send()`` would never be called inside the "window" and every
turn would look like it produced nothing. ``_on_transcript`` therefore
looks up the background task the call just spawned (via the same
``build_session_key`` the base class uses, since chat_id is always
"home") and awaits it too, so the window genuinely spans the full turn.
"""

import asyncio
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import build_session_key

logger = logging.getLogger(__name__)

_ANNOUNCE_MODES = ("off", "last_active", "default_device", "broadcast")
_HOME_CHAT_ID = "home"


def _import_sibling(name: str):
    """Load a module that lives next to this file (works under the runtime
    package loader AND the standalone test loader — same rationale and
    mechanism as plugins/platforms/voice_satellite/adapter.py)."""
    mod_key = f"hermes_ha_conversation_{name}"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(mod_key, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = module
    spec.loader.exec_module(module)
    return module


def check_requirements() -> bool:
    """Wyoming framing lib present (lazy-installed on first use)."""
    try:
        import wyoming  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        from tools.lazy_deps import ensure

        # prompt=False: called from the gateway's platform registry, which
        # must never block on an interactive install confirmation.
        ensure("platform.ha_conversation", prompt=False)
        import wyoming  # noqa: F401
        return True
    except Exception:
        return False


def validate_config(config) -> bool:
    extra = config.extra or {}
    if not extra:
        return False
    try:
        port = int(extra.get("port", 10600))
    except (TypeError, ValueError):
        return False
    if not 0 <= port < 65536:  # 0 = ephemeral, used by tests
        return False
    mode = str(extra.get("announce_mode", "off"))
    if mode not in _ANNOUNCE_MODES:
        return False
    if mode == "default_device" and not str(extra.get("announce_entity") or "").strip():
        return False
    return True


def _apply_yaml_config(yaml_cfg: dict, platform_cfg: dict) -> Optional[dict]:
    """Seed PlatformConfig.extra from the user's `ha_conversation` block.

    The loader binds ``platform_cfg`` to whichever block the user wrote —
    top-level section or nested ``platforms.ha_conversation`` — so read the
    ARGUMENT, never re-read yaml_cfg (gateway/config.py ~1261-1273).
    Enablement bridges from the block's own ``enabled:`` key via the
    loader's generic shared-key loop; only the return value lands in extra.
    """
    section = platform_cfg if isinstance(platform_cfg, dict) else {}
    known = ("bind_host", "port", "ack_after_seconds", "announce_mode",
             "announce_entity", "max_transcript_chars")
    if not any(k in section for k in known):
        return None
    return {k: section[k] for k in known if k in section}


class _TurnWindow:
    """Mutable state for one utterance's reply window."""

    __slots__ = ("chunks", "satellite_id")

    def __init__(self, satellite_id: Optional[str]):
        self.chunks: list = []
        self.satellite_id = satellite_id


class HAConversationAdapter(BasePlatformAdapter):
    """Gateway adapter: Wyoming handle server -> shared 'home' session."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("ha_conversation"))
        extra = config.extra or {}
        self._bind_host = str(extra.get("bind_host", "127.0.0.1"))
        self._port = int(extra.get("port", 10600))
        self._ack_after = float(extra.get("ack_after_seconds", 8.0))
        self._announce_mode = str(extra.get("announce_mode", "off"))
        self._announce_entity = str(extra.get("announce_entity", "") or "")
        self._max_chars = int(extra.get("max_transcript_chars", 2000))
        self._hs = _import_sibling("handle_server")
        self._server = None
        self._turn_lock = asyncio.Lock()
        self._active_window: Optional[_TurnWindow] = None
        self._waiting = 0
        self._last_active_satellite: Optional[str] = None

    @property
    def server_port(self) -> int:
        return self._server.port if self._server is not None else self._port

    # -- authorization: reaching the bound port is the grant ---------------
    @property
    def authorization_is_upstream(self) -> bool:
        """The operator chose what this port is reachable from (default
        127.0.0.1; exposing it on a LAN/tailnet is an explicit opt-in).
        There is no per-speaker identity in the Wyoming handle protocol —
        same posture as voice_satellite."""
        return True

    # -- lifecycle -----------------------------------------------------------
    async def connect(self, *, is_reconnect: bool = False) -> bool:
        supports_control = self._ha_credentials_present()
        self._server = self._hs.HandleServer(
            self._bind_host,
            self._port,
            on_transcript=self._on_transcript,
            supports_home_control=supports_control,
        )
        try:
            await self._server.start()
        except OSError as err:
            logger.error(
                "[ha_conversation] cannot bind %s:%s (%s) — port in use or "
                "privileged", self._bind_host, self._port, err,
            )
            self._server = None
            return False
        logger.info(
            "[ha_conversation] handle server on %s:%s (home control: %s)",
            self._bind_host, self.server_port, supports_control,
        )
        self._running = True
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        self._running = False
        try:
            if self._server is not None:
                await self._server.stop()
        finally:
            self._server = None
            self._mark_disconnected()

    @staticmethod
    def _ha_credentials_present() -> bool:
        try:
            from tools.homeassistant_tool import get_ha_config

            url, token = get_ha_config()
            return bool(url and token)
        except Exception:
            return False

    # -- inbound: HA -> agent --------------------------------------------------
    async def _on_transcript(self, text: str, context: Dict[str, Any], respond) -> None:
        text = (text or "").strip()
        if not text:
            await respond(None)
            return
        if len(text) > self._max_chars:
            logger.warning(
                "[ha_conversation] transcript over %d chars rejected", self._max_chars
            )
            await respond(None)
            return
        if self._waiting >= 8:
            await respond("I'm handling several requests right now — try again in a moment.")
            return

        # Per-room announce targets must be real assist_satellite entities.
        # HA's context may carry only a device_id (a registry hex id) — using
        # that as an announce target would fail against an invalid entity and
        # drop the late reply; leave it None so _deliver_late falls back to
        # announce_mode routing instead.
        raw_satellite = context.get("satellite_id") or context.get("device_id")
        announce_target = (
            str(raw_satellite)
            if isinstance(raw_satellite, str)
            and raw_satellite.startswith("assist_satellite.")
            else None
        )
        window = _TurnWindow(announce_target)
        self._waiting += 1
        # Start the ack timer NOW, before the turn lock is acquired: it must
        # measure total wait from arrival (queue time behind another room's
        # turn plus processing time), not just processing time. Creating it
        # only after the lock meant a second utterance queued behind a long
        # turn sat silent past HA's pipeline timeout until it ALSO got the
        # lock, since the timer never even started while it waited.
        ack_task = asyncio.create_task(self._ack_after_delay(window, respond))
        try:
            async with self._turn_lock:
                # Same entity-only rule as the per-window target above.
                self._last_active_satellite = (
                    announce_target or self._last_active_satellite
                )
                self._active_window = window
                failed = False
                try:
                    source = self.build_source(
                        chat_id=_HOME_CHAT_ID, chat_name="Home",
                        chat_type="dm", user_id=_HOME_CHAT_ID,
                        user_name="Home Assistant",
                    )
                    event = MessageEvent(
                        text=text, message_type=MessageType.TEXT, source=source
                    )
                    await self._run_turn_and_wait(event)
                except Exception:
                    logger.exception("[ha_conversation] agent turn failed")
                    failed = True
                finally:
                    self._active_window = None
                    ack_task.cancel()
                    # Let an in-flight ack write settle before the final
                    # respond decision below, and retrieve any stray
                    # exception from the ack task instead of dropping it.
                    await asyncio.gather(ack_task, return_exceptions=True)

                if failed:
                    await respond("Sorry, something went wrong handling that.")
                    return
                reply = "\n".join(
                    c for c in (s.strip() for s in window.chunks) if c
                ) or "Done."
                wrote = await respond(reply)
                if not wrote:
                    # ack already answered the socket (or HA hung up):
                    # same late-delivery path either way.
                    await self._deliver_late(reply, window)
        finally:
            # Safety net for any exit path above the inner finally (e.g. this
            # task getting cancelled while still waiting on the turn lock).
            # cancel()/gather() are idempotent on an already-settled task, so
            # this is a harmless no-op in the common case handled above.
            ack_task.cancel()
            await asyncio.gather(ack_task, return_exceptions=True)
            self._waiting -= 1

    async def _run_turn_and_wait(self, event: MessageEvent) -> None:
        """Dispatch ``event`` through the gateway pipeline and wait for the
        turn to actually finish.

        ``handle_message()`` only spawns the real processing as a background
        task and returns immediately (so a live adapter can keep receiving
        while an agent is busy) — it does NOT wait for the agent or for any
        ``send()`` calls the turn makes. Since this adapter's chat_id is
        always the fixed "home" session, the session key the base class
        computes is deterministic, so the just-spawned task can be found in
        ``self._session_tasks`` and awaited directly. Without this, the
        await-window would close before the agent ever replies and every
        turn would silently fall through to the late-delivery path.
        """
        await self.handle_message(event)
        session_key = build_session_key(
            event.source,
            group_sessions_per_user=self.config.extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=self.config.extra.get("thread_sessions_per_user", False),
        )
        task = self._session_tasks.get(session_key)
        if task is not None:
            await task

    async def _ack_after_delay(self, window: "_TurnWindow", respond) -> None:
        try:
            await asyncio.sleep(self._ack_after)
        except asyncio.CancelledError:
            return
        await respond(self._ack_text())

    def _ack_text(self) -> str:
        if self._announce_mode == "off":
            return ("I'm still working on that. I'll keep the answer in our "
                    "conversation.")
        return "I'm still working on that. I'll announce the answer when it's ready."

    # -- outbound: agent -> HA ---------------------------------------------------
    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        window = self._active_window
        if window is not None:
            window.chunks.append(content)
            return SendResult(success=True)
        # No active turn: Hermes-initiated (cron, background completion).
        return await self._announce(content, entity=None)

    async def _deliver_late(self, text: str, window: "_TurnWindow") -> None:
        """Reply finished after the socket was answered (ack) or lost.

        Targets the ORIGINATING satellite when known — Room A's late answer
        must never play in Room B — falling back to mode routing otherwise.
        The session transcript already holds the text either way.
        """
        if self._announce_mode == "off":
            return
        await self._announce(text, entity=window.satellite_id)

    async def _announce(self, text: str, *, entity: Optional[str]) -> SendResult:
        """Speak via HA's assist_satellite.announce (HA renders the TTS)."""
        if self._announce_mode == "off":
            return SendResult(success=True)  # transcript-only by config
        target = entity
        if target is None:
            if self._announce_mode == "default_device":
                target = self._announce_entity
            elif self._announce_mode == "broadcast":
                target = "all"
            else:  # last_active
                target = self._last_active_satellite
        if not target:
            logger.info(
                "[ha_conversation] no announce target yet (mode=%s); "
                "message stays in the conversation", self._announce_mode,
            )
            return SendResult(success=True)
        try:
            from tools import homeassistant_tool as ha_tool

            await ha_tool.async_call_service(
                "assist_satellite", "announce", target, {"message": text}
            )
            return SendResult(success=True)
        except Exception as err:
            logger.warning("[ha_conversation] announce failed: %s", err)
            return SendResult(success=False, error=str(err), retryable=True)

    # -- misc required surface -------------------------------------------------
    async def send_typing(self, chat_id: str, metadata=None) -> None:
        pass  # no visual surface

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": "Home", "type": "home_assistant", "chat_id": chat_id}


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system at startup."""
    ctx.register_platform(
        name="ha_conversation",
        label="HA Conversation Agent",
        adapter_factory=lambda cfg: HAConversationAdapter(cfg),
        check_fn=check_requirements,
        is_connected=validate_config,
        validate_config=validate_config,
        apply_yaml_config_fn=_apply_yaml_config,
        install_hint="pip install 'hermes-agent[satellite]'",
        emoji="🏠",
        pii_safe=True,
        platform_hint=(
            "Your replies are spoken aloud by Home Assistant voice "
            "hardware. Keep them brief and conversational — one to three "
            "sentences. Never use markdown, code blocks, tables, or URLs; "
            "they will be read out loud."
        ),
    )

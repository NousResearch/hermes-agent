"""Voice-call runtime singleton.

One runtime per process, owned by the gateway through the platform adapter:
``adapter.connect()`` → :func:`ensure_runtime`, ``adapter.disconnect()`` →
:func:`stop_runtime`. The tool, CLI, and slash handlers reach the runtime
through :func:`get_runtime` without holding an adapter reference.

Boot order (mirrors OpenClaw's runtime.ts pipeline):
provider → store → manager (+ restore) → webhook server → tunnel/public URL.
Teardown is the reverse and tolerates partial initialization.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional, Tuple

from .config import VoiceCallConfig, normalize_e164
from .events import CallRecord
from .manager import CallManager
from .store import CallStore

logger = logging.getLogger(__name__)

_runtime: Optional["VoiceCallRuntime"] = None
_runtime_lock: Optional[asyncio.Lock] = None


def _get_lock() -> asyncio.Lock:
    # Created lazily so importing this module never requires a running loop.
    global _runtime_lock
    if _runtime_lock is None:
        _runtime_lock = asyncio.Lock()
    return _runtime_lock


class VoiceCallRuntime:
    """Owns everything that must live for the gateway's lifetime."""

    def __init__(
        self,
        config: VoiceCallConfig,
        adapter: Any = None,
        store_dir: Optional[Path] = None,
    ):
        self.config = config
        self.adapter = adapter
        self.provider = None
        self.store: Optional[CallStore] = None
        self.manager: Optional[CallManager] = None
        self.webhook_server = None  # P3
        self.tunnel = None          # P5
        self.public_url: Optional[str] = config.public_url
        self._store_dir = store_dir
        self._started = False

    async def start(self) -> None:
        """Boot the runtime. Idempotent; raises on unrecoverable setup errors.

        Order mirrors OpenClaw: provider → store/manager (+restore) →
        webhook server → tunnel. A failure after the server binds stops the
        server again so the port is never leaked.
        """
        if self._started:
            return
        from .providers import create_provider
        from .webhook import VoiceCallWebhookServer

        self.provider = create_provider(self.config)
        if self.public_url:
            self.provider.set_public_url(self.public_url)
        # The mock provider pushes carrier-style events (ringing/answered)
        # back into the manager the way real webhooks would.
        if hasattr(self.provider, "event_sink"):
            self.provider.event_sink = self._provider_event_sink

        self.store = CallStore(base_dir=self._store_dir)
        self.manager = CallManager(
            self.config,
            self.provider,
            self.store,
            on_final_transcript=self._on_final_transcript,
            on_call_ended=self._on_call_ended,
        )
        await self.manager.initialize()

        self.webhook_server = VoiceCallWebhookServer(
            self.config,
            self.provider,
            process_event=self.manager.process_event,
            admin_handler=self._handle_admin_command,
            admin_token=self._load_or_create_admin_token(),
        )
        await self.webhook_server.start()
        try:
            await self._resolve_public_url()
        except Exception:
            await self.webhook_server.stop()
            raise

        self._started = True
        logger.info(
            "voice_call runtime started (provider=%s, serve=%s:%s%s)",
            self.config.provider,
            self.config.serve.bind,
            self.config.serve.port,
            self.config.serve.path,
        )

    async def stop(self) -> None:
        """Tear everything down. Idempotent; tolerates partial initialization."""
        self._started = False
        if self.webhook_server is not None:
            try:
                await self.webhook_server.stop()
            except Exception:  # noqa: BLE001
                logger.exception("voice_call: webhook server stop failed")
            self.webhook_server = None
        if self.tunnel is not None:
            try:
                await self.tunnel.stop()
            except Exception:  # noqa: BLE001
                logger.exception("voice_call: tunnel stop failed")
            self.tunnel = None
        if self.manager is not None:
            try:
                await self.manager.shutdown()
            except Exception:  # noqa: BLE001
                logger.exception("voice_call: manager shutdown failed")
            self.manager = None
        self.provider = None
        logger.info("voice_call runtime stopped")

    async def _resolve_public_url(self) -> None:
        """Resolve the externally reachable URL (explicit > tunnel)."""
        if self.public_url or self.config.tunnel.provider == "none":
            return
        # Tunnel providers (ngrok / tailscale) are wired in the Telnyx phase.
        from .tunnel import start_tunnel  # noqa: PLC0415

        self.tunnel = await start_tunnel(self.config)
        self.public_url = self.tunnel.public_url
        self.provider.set_public_url(self.public_url)

    def _load_or_create_admin_token(self) -> str:
        """Pre-shared token the CLI uses against the local admin endpoint."""
        import secrets

        token_path = self.store.base_dir / "admin.token"
        try:
            existing = token_path.read_text(encoding="utf-8").strip()
            if existing:
                return existing
        except OSError:
            pass
        token = secrets.token_urlsafe(32)
        try:
            self.store.base_dir.mkdir(parents=True, exist_ok=True)
            token_path.write_text(token, encoding="utf-8")
            token_path.chmod(0o600)
        except OSError:
            logger.warning("voice_call: could not persist admin token", exc_info=True)
        return token

    async def _handle_admin_command(self, payload: dict) -> dict:
        """CLI → runtime command dispatch (served on the admin endpoint)."""
        command = str(payload.get("command", ""))
        if self.manager is None:
            return {"success": False, "error": "runtime not started"}
        if command == "status":
            calls = [r.to_dict() for r in self.manager.get_active_calls()]
            return {
                "success": True,
                "provider": self.config.provider,
                "public_url": self.public_url,
                "active_calls": calls,
            }
        if command == "call":
            record = await self.manager.initiate_call(
                str(payload.get("to", "")),
                message=payload.get("message"),
                mode=payload.get("mode"),
            )
            return {"success": True, "call_id": record.call_id}
        if command == "speak":
            await self.manager.speak(
                str(payload.get("call_id", "")), str(payload.get("message", ""))
            )
            return {"success": True}
        if command == "continue":
            reply = await self.manager.continue_call(
                str(payload.get("call_id", "")), str(payload.get("message", ""))
            )
            return {"success": True, "reply": reply}
        if command == "dtmf":
            await self.manager.send_dtmf(
                str(payload.get("call_id", "")), str(payload.get("digits", ""))
            )
            return {"success": True}
        if command == "end":
            await self.manager.end_call(str(payload.get("call_id", "")))
            return {"success": True}
        return {"success": False, "error": f"unknown command {command!r}"}

    # -- event plumbing -----------------------------------------------------

    async def _provider_event_sink(self, event) -> None:
        if self.manager is not None:
            await self.manager.process_event(event)

    async def _on_final_transcript(self, record: CallRecord, text: str) -> None:
        """Final caller utterance → gateway agent turn."""
        from .responder import dispatch_transcript

        logger.debug(
            "voice_call: final transcript on %s: %r", record.call_id, text[:80]
        )
        await dispatch_transcript(self, record, text)

    async def _on_call_ended(self, record: CallRecord) -> None:
        logger.info(
            "voice_call: call %s ended (%s: %s)",
            record.call_id, record.state.value, record.end_reason,
        )

    # -- adapter send() support ----------------------------------------------

    async def speak_for_chat(
        self, chat_id: str, content: str, thread_id: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Speak ``content`` on the live call mapped to ``chat_id``.

        Returns ``(success, call_id_or_error)``. With per-call session scope
        the call id travels as the session thread_id and takes precedence.
        """
        if self.manager is None:
            return False, "voice_call runtime not started"
        record = None
        if thread_id:
            record = self.manager.get_call(str(thread_id))
        if record is None:
            record = self.manager.call_for_chat(normalize_e164(chat_id))
        if record is None or record.is_terminal:
            return False, f"no active voice call for {chat_id}"
        from .responder import strip_for_speech

        spoken = strip_for_speech(content)
        if not spoken:
            return False, "nothing speakable in message"
        try:
            await self.manager.speak(record.call_id, spoken)
        except Exception as e:  # noqa: BLE001 — surface as failed SendResult
            return False, f"speak failed: {e}"
        return True, record.call_id


async def ensure_runtime(
    config: VoiceCallConfig,
    adapter: Any = None,
    store_dir: Optional[Path] = None,
) -> VoiceCallRuntime:
    """Return the running runtime, starting it on first use.

    Concurrency-safe: simultaneous callers share one startup. A failed start
    leaves no singleton behind so the next attempt retries cleanly.
    """
    global _runtime
    async with _get_lock():
        if _runtime is not None:
            if adapter is not None:
                _runtime.adapter = adapter
            return _runtime
        runtime = VoiceCallRuntime(config, adapter=adapter, store_dir=store_dir)
        try:
            await runtime.start()
        except Exception:
            try:
                await runtime.stop()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                logger.exception("voice_call runtime cleanup after failed start")
            raise
        _runtime = runtime
        return runtime


def get_runtime() -> Optional[VoiceCallRuntime]:
    return _runtime


async def stop_runtime() -> None:
    """Stop and clear the singleton. Safe to call when never started."""
    global _runtime
    async with _get_lock():
        runtime = _runtime
        _runtime = None
    if runtime is not None:
        await runtime.stop()

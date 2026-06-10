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
        """Boot the runtime. Idempotent; raises on unrecoverable setup errors."""
        if self._started:
            return
        from .providers import create_provider

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
        if self.manager is not None:
            try:
                await self.manager.shutdown()
            except Exception:  # noqa: BLE001
                logger.exception("voice_call: manager shutdown failed")
            self.manager = None
        self.provider = None
        logger.info("voice_call runtime stopped")

    # -- event plumbing -----------------------------------------------------

    async def _provider_event_sink(self, event) -> None:
        if self.manager is not None:
            await self.manager.process_event(event)

    async def _on_final_transcript(self, record: CallRecord, text: str) -> None:
        """Final caller utterance → gateway agent turn (wired in P4)."""
        logger.debug(
            "voice_call: final transcript on %s: %r", record.call_id, text[:80]
        )

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
        try:
            await self.manager.speak(record.call_id, content)
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

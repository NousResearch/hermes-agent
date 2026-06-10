"""Voice-call runtime singleton.

One runtime per process, owned by the gateway through the platform adapter:
``adapter.connect()`` → :func:`ensure_runtime`, ``adapter.disconnect()`` →
:func:`stop_runtime`. The tool, CLI, and slash handlers reach the runtime
through :func:`get_runtime` without holding an adapter reference.

P1 note: this is the lifecycle skeleton only — the provider, call manager,
store, and webhook server are wired in by later phases.
"""

import asyncio
import logging
from typing import Any, Optional, Tuple

from .config import VoiceCallConfig

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

    def __init__(self, config: VoiceCallConfig, adapter: Any = None):
        self.config = config
        self.adapter = adapter
        self._started = False

    async def start(self) -> None:
        """Boot the runtime. Idempotent; raises on unrecoverable setup errors."""
        if self._started:
            return
        # Later phases: resolve provider → start webhook server → resolve
        # public URL/tunnel → create manager → restore persisted calls.
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
        if not self._started:
            return
        self._started = False
        logger.info("voice_call runtime stopped")

    async def speak_for_chat(
        self, chat_id: str, content: str, thread_id: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Speak ``content`` on the live call mapped to ``chat_id``.

        Returns ``(success, detail)``. Wired to the call manager in a later
        phase; until then there is never an active call.
        """
        return False, f"no active voice call for {chat_id}"


async def ensure_runtime(
    config: VoiceCallConfig, adapter: Any = None
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
        runtime = VoiceCallRuntime(config, adapter=adapter)
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

"""Gateway lifecycle adapter for the bundled A2A protocol server."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from gateway.config import Platform
from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
    is_network_accessible,
)

from .config import apply_yaml_config, settings_from_platform_config
from .entry import build_app

logger = logging.getLogger(__name__)


class A2AAdapter(BasePlatformAdapter):
    """Run the A2A ASGI server as a gateway-managed platform plugin."""

    def __init__(self, config: Any):
        super().__init__(config, Platform("a2a"))
        self._server: Any = None
        self._serve_task: asyncio.Task[None] | None = None
        self._serve_error: str | None = None
        self._stopping = False
        self._fatal_notify_task: asyncio.Task[None] | None = None

    async def _notify_fatal_error_safely(self) -> None:
        try:
            await self._notify_fatal_error()
        except Exception:
            logger.exception("A2A fatal-error notification failed")

    def _consume_fatal_notification(self, task: asyncio.Task[None]) -> None:
        if self._fatal_notify_task is task:
            self._fatal_notify_task = None
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("A2A fatal-error notification task failed")

    async def _serve_embedded(self) -> None:
        """Run Uvicorn without letting process-level exits escape the task."""
        try:
            await self._server.serve()
        except SystemExit as exc:
            self._serve_error = f"Uvicorn exited during startup ({exc.code})"
        except Exception as exc:  # noqa: BLE001 - isolate server task failures
            self._serve_error = f"{type(exc).__name__}: {exc}"
        finally:
            if self._running and not self._stopping:
                message = self._serve_error or "A2A server stopped unexpectedly"
                logger.error("%s", message)
                self._set_fatal_error("a2a_server_stopped", message, retryable=True)
                notify_task = asyncio.create_task(
                    self._notify_fatal_error_safely(),
                    name="hermes-a2a-fatal-notify",
                )
                self._fatal_notify_task = notify_task
                notify_task.add_done_callback(self._consume_fatal_notification)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        del is_reconnect
        if self._serve_task is not None and not self._serve_task.done():
            return True
        if self._serve_task is not None:
            await self.disconnect()

        import uvicorn

        settings = settings_from_platform_config(self.config)
        if is_network_accessible(settings.host):
            logger.warning(
                "A2A is binding to a network-accessible host (%s) without "
                "built-in authentication. Put it behind a trusted reverse "
                "proxy or authentication layer.",
                settings.host,
            )
        app = build_app(
            settings.host,
            settings.port,
            settings.public_url,
            settings=settings,
        )

        class _EmbeddedServer(uvicorn.Server):
            def capture_signals(self):
                return contextlib.nullcontext()

        self._serve_error = None
        self._stopping = False
        self._server = _EmbeddedServer(
            uvicorn.Config(
                app,
                host=settings.host,
                port=settings.port,
                log_level="info",
                timeout_graceful_shutdown=10,
            )
        )
        self._serve_task = asyncio.create_task(
            self._serve_embedded(), name="hermes-a2a-server"
        )

        for _ in range(100):
            if self._server.started:
                self._running = True
                logger.info(
                    "A2A server listening on http://%s:%d",
                    settings.host,
                    settings.port,
                )
                return True
            if self._serve_task.done():
                await self._serve_task
                message = self._serve_error or "A2A server stopped during startup"
                logger.error("%s", message)
                self._set_fatal_error("a2a_startup_failed", message, retryable=True)
                return False
            await asyncio.sleep(0.05)

        logger.error("Timed out waiting for the A2A server to start")
        self._set_fatal_error(
            "a2a_startup_timeout",
            "Timed out waiting for the A2A server to start",
            retryable=True,
        )
        await self.disconnect()
        return False

    async def disconnect(self) -> None:
        self._stopping = True
        self._running = False
        if self._server is not None:
            self._server.should_exit = True
        task = self._serve_task
        if task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=10)
            except asyncio.TimeoutError:
                if self._server is not None:
                    self._server.force_exit = True
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self._serve_task = None
        self._server = None
        self._stopping = False

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        del chat_id, content, reply_to, metadata
        return SendResult(
            success=False,
            error="A2A responses are delivered through the protocol task lifecycle.",
        )

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        """Represent an A2A context as a direct agent conversation."""
        return {"name": f"A2A context {chat_id}", "type": "dm", "chat_id": chat_id}


def check_requirements() -> bool:
    """Return whether the optional A2A server dependencies are installed."""
    try:
        import uvicorn  # noqa: F401

        import a2a  # noqa: F401
    except ImportError:
        return False
    return True


def validate_config(config: Any) -> bool:
    settings = settings_from_platform_config(config)
    return bool(settings.host and 1 <= settings.port <= 65535)


def is_connected(config: Any) -> bool:
    return bool(getattr(config, "enabled", False)) and validate_config(config)


def register(ctx: Any) -> None:
    """Register A2A through the generic gateway platform plugin surface."""
    ctx.register_platform(
        name="a2a",
        label="A2A (Agent2Agent)",
        adapter_factory=lambda cfg: A2AAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        install_hint="Install the optional dependencies with: pip install -e '.[a2a]'",
        apply_yaml_config_fn=apply_yaml_config,
        emoji="🤖",
        allow_update_command=False,
        platform_hint=(
            "You are serving a remote agent over the A2A protocol. Return a "
            "self-contained task result and do not ask interactive questions."
        ),
    )

"""Async MemOS bridge daemon manager for the Hermes gateway.

Wraps ``GatewayMemosManager`` from the memos-plugin (if installed) with
lifecycle hooks for ``GatewayRunner``.

    manager = MemosDaemonManager()
    manager.ensure_running()
    await manager.start_heartbeat()
    ...
    await manager.stop_heartbeat()
    manager.stop()
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


class MemosDaemonManager:
    """Async lifecycle manager for the MemOS bridge subprocess."""

    def __init__(self) -> None:
        self._heartbeat_task: asyncio.Task | None = None
        self._heartbeat_interval: float = 30.0

        # Try to import from memos-plugin; graceful degrade if not installed.
        self._inner = None
        try:
            # memos-plugin lives alongside hermes-agent under HERMES_HOME.
            import os as _os
            _hermes_home = _os.environ.get("HERMES_HOME", _os.path.expanduser("~/.hermes"))
            _memos_path = _os.path.join(_hermes_home, "memos-plugin")
            if _memos_path not in __import__("sys").path:
                __import__("sys").path.insert(0, _memos_path)

            from adapters.hermes.memos_provider.gateway_manager import (
                GatewayMemosManager,
            )
            self._inner = GatewayMemosManager()
        except ImportError:
            logger.info(
                "MemOS: memos-plugin not found. "
                "Install via the Hermes setup wizard for local memory support."
            )

    def ensure_running(self) -> bool:
        """Start (or confirm) the bridge daemon is running. Safe to call repeatedly."""
        if self._inner is not None:
            return self._inner.ensure_running()
        return False

    async def start_heartbeat(self, interval: float | None = None) -> None:
        """Start background heartbeat that keeps the bridge alive."""
        if self._inner is None:
            return
        if self._heartbeat_task is not None:
            return
        if interval is not None:
            self._heartbeat_interval = interval
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info(
            "MemOS: heartbeat started (interval=%ss)",
            self._heartbeat_interval,
        )

    async def stop_heartbeat(self) -> None:
        """Cancel the heartbeat task."""
        task, self._heartbeat_task = self._heartbeat_task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    def stop(self) -> None:
        """Gracefully shut down the bridge subprocess."""
        if self._inner is not None:
            self._inner.shutdown()
            logger.info("MemOS: bridge daemon shut down")

    async def _heartbeat_loop(self) -> None:
        """Periodic loop that keeps the bridge alive."""
        try:
            while True:
                await asyncio.sleep(self._heartbeat_interval)
                try:
                    if self._inner is not None:
                        self._inner.ensure_running()
                except Exception:
                    logger.exception("MemOS: heartbeat ensure_running failed")
        except asyncio.CancelledError:
            pass

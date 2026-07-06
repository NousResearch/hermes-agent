"""Adapter lifecycle helpers for ``GatewayRunner``.

Extracted from ``gateway/run.py`` as part of the god-file decomposition
campaign (Phase 4 mechanical mixin lift). This mixin holds the per-adapter
connect/disconnect/teardown helpers:

  - ``_safe_adapter_disconnect`` — defensive ``adapter.disconnect()``
    used on failed-connect paths so partial-init resources (aiohttp
    ClientSession, poll tasks, child subprocesses) don't leak as
    "Unclosed client session" warnings at process exit (seen on the
    2026-04-18 gateway restart).
  - ``_bounded_adapter_teardown`` — shutdown-path teardown that wraps
    both ``cancel_background_tasks()`` and ``disconnect()`` in the
    per-adapter timeout budget so a wedged adapter (half-dead Feishu/Lark
    WebSocket thread) doesn't stall shutdown past systemd's
    ``TimeoutStopSec`` and trip a SIGKILL that skips ``atexit`` PID-file
    cleanup (#14128).
  - ``_adapter_disconnect_timeout_secs`` — env-tunable budget for the
    two methods above.
  - ``_platform_connect_timeout_secs`` — env-tunable budget for the
    connect path below.
  - ``_connect_adapter_with_timeout`` — wraps ``adapter.connect()`` so
    one platform's slow boot can't block the others; forwards
    ``is_reconnect`` so platforms can preserve server-side queues on a
    watcher reconnect after an outage (#46621).

Behavior-neutral: every method is lifted verbatim from ``GatewayRunner``.
The five methods read only ``os.getenv`` and the two module-level default
constants (``_ADAPTER_DISCONNECT_TIMEOUT_SECS_DEFAULT``,
``_PLATFORM_CONNECT_TIMEOUT_SECS_DEFAULT``); they touch no ``self.*``
state and so do not require ``GatewayRunner.__init__`` to have been
called — bare ``object.__new__(GatewayRunner)`` shells in the regression
suite continue to work. The constants continue to live in
``gateway.run`` (their canonical home since they are tunable from
``config.yaml`` documentation paths); this module imports them lazily at
call time so there is no import cycle:

    adapter_lifecycle_mixin.py  --(lazy import at call time)--> gateway.run

The lazy import preserves the exact logger name (``"gateway.run"``) so log
records emitted from these helpers keep their original logger name.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional

logger = logging.getLogger("gateway.run")


class GatewayAdapterLifecycleMixin:
    """Adapter connect/disconnect/teardown helpers lifted from
    ``GatewayRunner`` (god-file decomposition Phase 4)."""

    async def _safe_adapter_disconnect(self, adapter, platform) -> None:
        """Call adapter.disconnect() defensively, swallowing any error.

        Used when adapter.connect() failed or raised — the adapter may
        have allocated partial resources (aiohttp.ClientSession, poll
        tasks, child subprocesses) that would otherwise leak and surface
        as "Unclosed client session" warnings at process exit.

        Must tolerate partial-init state and never raise, since callers
        use it inside error-handling blocks.
        """
        timeout = self._adapter_disconnect_timeout_secs()
        try:
            if timeout <= 0:
                await adapter.disconnect()
            else:
                await asyncio.wait_for(adapter.disconnect(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "Timed out after %.1fs while disconnecting %s adapter; continuing shutdown",
                timeout,
                platform.value if platform is not None else "adapter",
            )
        except Exception as e:
            logger.debug(
                "Defensive %s disconnect after failed connect raised: %s",
                platform.value if platform is not None else "adapter",
                e,
            )

    async def _bounded_adapter_teardown(
        self, adapter, platform, *, profile: Optional[str] = None
    ) -> None:
        """Tear down one adapter on the shutdown path with bounded awaits.

        Both ``cancel_background_tasks()`` and ``disconnect()`` can block
        indefinitely when a platform's network state is half-dead (e.g. a
        wedged Feishu/Lark WebSocket thread waiting on I/O). An unbounded
        await here stalls the entire shutdown sequence past systemd's
        ``TimeoutStopSec``; the resulting SIGKILL skips ``atexit`` PID-file
        cleanup, so the next start dies with "PID file race lost" (#14128).

        Each await is wrapped in the existing per-adapter timeout budget
        (``HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT``). On timeout we log
        and force forward progress; the loop never hangs regardless of any
        adapter's internal behavior. Never raises.
        """
        timeout = self._adapter_disconnect_timeout_secs()
        suffix = f" (profile: {profile})" if profile else ""
        started_at = time.monotonic()
        try:
            if timeout <= 0:
                await adapter.cancel_background_tasks()
            else:
                await asyncio.wait_for(
                    adapter.cancel_background_tasks(), timeout=timeout
                )
        except asyncio.TimeoutError:
            logger.warning(
                "✗ %s background-task cancel timed out after %.1fs - forcing continue%s",
                platform.value, timeout, suffix,
            )
        except Exception as e:
            logger.debug("✗ %s background-task cancel error%s: %s", platform.value, suffix, e)
        try:
            if timeout <= 0:
                await adapter.disconnect()
            else:
                await asyncio.wait_for(adapter.disconnect(), timeout=timeout)
            logger.info(
                "✓ %s disconnected (%.2fs)%s",
                platform.value, time.monotonic() - started_at, suffix,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "✗ %s disconnect timed out after %.1fs - forcing continue%s",
                platform.value, timeout, suffix,
            )
        except Exception as e:
            logger.error(
                "✗ %s disconnect error after %.2fs%s: %s",
                platform.value, time.monotonic() - started_at, suffix, e,
            )

    def _adapter_disconnect_timeout_secs(self) -> float:
        """Return the per-adapter disconnect timeout used during shutdown."""
        raw = os.getenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "").strip()
        if raw:
            try:
                timeout = float(raw)
            except ValueError:
                logger.warning(
                    "Ignoring invalid HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT=%r",
                    raw,
                )
            else:
                return max(0.0, timeout)
        # Default lives in gateway.run (canonical home for the constant);
        # lazy import to avoid a module-level cycle.
        from gateway.run import _ADAPTER_DISCONNECT_TIMEOUT_SECS_DEFAULT
        return _ADAPTER_DISCONNECT_TIMEOUT_SECS_DEFAULT

    def _platform_connect_timeout_secs(self) -> float:
        """Return the per-platform connect timeout used during startup/retry."""
        raw = os.getenv("HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT", "").strip()
        if raw:
            try:
                timeout = float(raw)
            except ValueError:
                logger.warning(
                    "Ignoring invalid HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT=%r",
                    raw,
                )
            else:
                return max(0.0, timeout)
        # Default lives in gateway.run; lazy import to avoid a cycle.
        from gateway.run import _PLATFORM_CONNECT_TIMEOUT_SECS_DEFAULT
        return _PLATFORM_CONNECT_TIMEOUT_SECS_DEFAULT

    async def _connect_adapter_with_timeout(
        self, adapter, platform, *, is_reconnect: bool = False
    ) -> bool:
        """Connect an adapter without allowing one platform to block others.

        ``is_reconnect`` is forwarded to ``adapter.connect()`` so platform
        adapters can distinguish a cold first boot (drop any stale
        server-side queue) from a watcher reconnect after a prolonged outage
        (preserve the queue so messages sent during the outage are delivered
        rather than silently dropped — #46621).
        """
        timeout = self._platform_connect_timeout_secs()
        if timeout <= 0:
            return await adapter.connect(is_reconnect=is_reconnect)
        try:
            return await asyncio.wait_for(
                adapter.connect(is_reconnect=is_reconnect), timeout=timeout
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"{platform.value} connect timed out after {timeout:g}s"
            ) from exc

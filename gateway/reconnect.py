"""Connection/reconnect helpers extracted from GatewayRunner.

Standalone functions taking ``runner`` (the GatewayRunner instance) as their
first parameter, extracted to keep gateway/run.py focused on lifecycle
orchestration.
"""

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_PLATFORM_CONNECT_TIMEOUT_SECS_DEFAULT = 30.0
_ADAPTER_DISCONNECT_TIMEOUT_SECS_DEFAULT = 5.0


def platform_connect_timeout_secs(runner) -> float:
    """Return the per-platform connect timeout used during startup/retry."""
    import os

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
    return _PLATFORM_CONNECT_TIMEOUT_SECS_DEFAULT


def adapter_disconnect_timeout_secs(runner) -> float:
    """Return the per-adapter disconnect timeout used during shutdown."""
    import os

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
    return _ADAPTER_DISCONNECT_TIMEOUT_SECS_DEFAULT


async def safe_adapter_disconnect(runner, adapter, platform) -> None:
    """Call adapter.disconnect() defensively, swallowing any error.

    Used when adapter.connect() failed or raised — the adapter may
    have allocated partial resources (aiohttp.ClientSession, poll
    tasks, child subprocesses) that would otherwise leak and surface
    as "Unclosed client session" warnings at process exit.

    Must tolerate partial-init state and never raise, since callers
    use it inside error-handling blocks.
    """
    timeout = adapter_disconnect_timeout_secs(runner)
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


async def connect_adapter_with_timeout(runner, adapter, platform) -> bool:
    """Connect an adapter without allowing one platform to block others."""
    timeout = platform_connect_timeout_secs(runner)
    if timeout <= 0:
        return await adapter.connect()
    try:
        return await asyncio.wait_for(adapter.connect(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"{platform.value} connect timed out after {timeout:g}s"
        ) from exc


async def platform_reconnect_watcher(runner) -> None:
    """Background task that periodically retries connecting failed platforms.

    Uses exponential backoff: 30s → 60s → 120s → 240s → 300s (cap).
    Retryable failures keep retrying at the backoff cap indefinitely
    — but if a platform fails ``_PAUSE_AFTER_FAILURES`` times in a row
    without ever succeeding, it is *paused*: kept in the retry queue
    but no longer hammered.  The user surfaces it with ``/platform list``
    and resumes it with ``/platform resume <name>``.  Non-retryable
    failures (bad auth, etc.) still drop out of the queue immediately.
    """
    _BACKOFF_CAP = 300  # 5 minutes max between retries
    _PAUSE_AFTER_FAILURES = 10  # circuit-breaker threshold

    await asyncio.sleep(10)  # initial delay — let startup finish
    while runner._running:
        if not runner._failed_platforms:
            # Nothing to reconnect — sleep and check again
            for _ in range(30):
                if not runner._running:
                    return
                await asyncio.sleep(1)
            continue

        now = time.monotonic()
        for platform in list(runner._failed_platforms.keys()):
            if not runner._running:
                return
            info = runner._failed_platforms[platform]
            # Skip paused platforms entirely — they need explicit
            # /platform resume to come back.
            if info.get("paused"):
                continue
            if now < info["next_retry"]:
                continue  # not time yet

            platform_config = info["config"]
            attempt = info["attempts"] + 1
            logger.info(
                "Reconnecting %s (attempt %d)...",
                platform.value, attempt,
            )

            try:
                adapter = runner._create_adapter(platform, platform_config)
                if not adapter:
                    logger.warning(
                        "Reconnect %s: adapter creation returned None, removing from retry queue",
                        platform.value,
                    )
                    del runner._failed_platforms[platform]
                    continue

                adapter.set_message_handler(runner._handle_message)
                adapter.set_fatal_error_handler(runner._handle_adapter_fatal_error)
                adapter.set_session_store(runner.session_store)
                adapter.set_busy_session_handler(runner._handle_active_session_busy_message)

                success = await connect_adapter_with_timeout(runner, adapter, platform)
                if success:
                    runner.adapters[platform] = adapter
                    runner._sync_voice_mode_state_to_adapter(adapter)
                    runner.delivery_router.adapters = runner.adapters
                    del runner._failed_platforms[platform]
                    runner._update_platform_runtime_status(
                        platform.value,
                        platform_state="connected",
                        error_code=None,
                        error_message=None,
                    )
                    logger.info("✓ %s reconnected successfully", platform.value)

                    # Rebuild channel directory with the new adapter
                    try:
                        from gateway.channel_directory import build_channel_directory
                        await build_channel_directory(runner.adapters)
                    except Exception:
                        pass
                # Check if the failure is non-retryable
                elif adapter.has_fatal_error and not adapter.fatal_error_retryable:
                    runner._update_platform_runtime_status(
                        platform.value,
                        platform_state="fatal",
                        error_code=adapter.fatal_error_code,
                        error_message=adapter.fatal_error_message,
                    )
                    logger.warning(
                        "Reconnect %s: non-retryable error (%s), removing from retry queue",
                        platform.value, adapter.fatal_error_message,
                    )
                    del runner._failed_platforms[platform]
                else:
                    runner._update_platform_runtime_status(
                        platform.value,
                        platform_state="retrying",
                        error_code=adapter.fatal_error_code,
                        error_message=adapter.fatal_error_message or "failed to reconnect",
                    )
                    backoff = min(30 * (2 ** (attempt - 1)), _BACKOFF_CAP)
                    info["attempts"] = attempt
                    info["next_retry"] = time.monotonic() + backoff
                    logger.info(
                        "Reconnect %s failed, next retry in %ds",
                        platform.value, backoff,
                    )
                    if attempt >= _PAUSE_AFTER_FAILURES:
                        runner._pause_failed_platform(
                            platform,
                            reason=(
                                adapter.fatal_error_message
                                or "failed to reconnect"
                            ),
                        )
            except Exception as e:
                runner._update_platform_runtime_status(
                    platform.value,
                    platform_state="retrying",
                    error_code=None,
                    error_message=str(e),
                )
                backoff = min(30 * (2 ** (attempt - 1)), _BACKOFF_CAP)
                info["attempts"] = attempt
                info["next_retry"] = time.monotonic() + backoff
                logger.warning(
                    "Reconnect %s error: %s, next retry in %ds",
                    platform.value, e, backoff,
                )
                if attempt >= _PAUSE_AFTER_FAILURES:
                    runner._pause_failed_platform(platform, reason=str(e))

        # Check every 10 seconds for platforms that need reconnection
        for _ in range(10):
            if not runner._running:
                return
            await asyncio.sleep(1)

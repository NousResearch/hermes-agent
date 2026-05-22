"""Gateway platform lifecycle management — extracted from gateway/run.py.

Handles adapter connection/disconnection, pause/resume, fatal error handling,
and runtime status tracking for all messaging platform adapters.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


async def connect_adapter_with_timeout(
    adapter: Any,
    platform: Any,
    platform_connect_timeout: float = 30.0,
) -> bool:
    """Connect a platform adapter with a timeout. Returns True on success."""
    try:
        await asyncio.wait_for(
            adapter.start(),
            timeout=platform_connect_timeout,
        )
        return True
    except asyncio.TimeoutError:
        logger.error("Platform %s connection timed out after %.1fs", platform, platform_connect_timeout)
        return False
    except Exception as e:
        logger.error("Platform %s connection failed: %s", platform, e)
        return False


async def safe_adapter_disconnect(adapter: Any, platform: Any, timeout: float = 5.0) -> None:
    """Safely disconnect a platform adapter, swallowing errors."""
    try:
        await asyncio.wait_for(adapter.stop(), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("Platform %s disconnect timed out, forcing", platform)
    except Exception as e:
        logger.debug("Platform %s disconnect error (non-fatal): %s", platform, e)


async def handle_adapter_fatal_error(
    adapter: Any,
    platform: Any,
    request_clean_exit: callable,
) -> None:
    """Handle a fatal adapter error by logging and requesting clean exit."""
    error = getattr(adapter, "_fatal_error", None)
    if error:
        logger.error("Fatal error on platform %s: %s", platform, error)
    else:
        logger.error("Platform %s adapter failed with no error detail", platform)
    request_clean_exit(f"platform_{platform}_fatal_error")


def pause_failed_platform(
    paused: Dict[str, float],
    platform: Any,
    *,
    reason: str = "",
    pause_duration: float = 300.0,
) -> None:
    """Pause a failed platform for a cooldown period."""
    import time
    paused[str(platform)] = time.monotonic() + pause_duration
    logger.info("Platform %s paused for %.0fs (%s)", platform, pause_duration, reason or "unknown error")


def resume_paused_platform(
    paused: Dict[str, float],
    platform: Any,
) -> bool:
    """Resume a previously paused platform. Returns True if it was paused."""
    key = str(platform)
    if key in paused:
        del paused[key]
        return True
    return False


def is_platform_paused(paused: Dict[str, float], platform: Any) -> bool:
    """Check if a platform is currently paused (within cooldown)."""
    import time
    key = str(platform)
    if key not in paused:
        return False
    if time.monotonic() < paused[key]:
        return True
    # Cooldown expired — auto-resume
    del paused[key]
    return False


def update_runtime_status(
    status: Dict[str, Any],
    gateway_state: Optional[str] = None,
    exit_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Update the runtime status dict and return it."""
    import time
    if gateway_state is not None:
        status["state"] = gateway_state
    if exit_reason is not None:
        status["exit_reason"] = exit_reason
    status["updated_at"] = time.monotonic()
    return status


def update_platform_runtime_status(
    platform_status: Dict[str, Dict[str, Any]],
    platform: Any,
    connected: bool,
    error: Optional[str] = None,
    info: Optional[str] = None,
) -> None:
    """Update the runtime status for a specific platform."""
    import time
    key = str(platform)
    entry = platform_status.setdefault(key, {})
    entry["connected"] = connected
    entry["updated_at"] = time.monotonic()
    if error:
        entry["error"] = error
    if info:
        entry["info"] = info


def snapshot_running_agents(
    running_agents: Dict[str, Any],
    running_agents_ts: Dict[str, float],
) -> Dict[str, Any]:
    """Snapshot all currently running agents for diagnostics."""
    import time
    snapshot = {}
    now = time.monotonic()
    for session_key in list(running_agents.keys()):
        started_at = running_agents_ts.get(session_key, now)
        snapshot[session_key] = {
            "running_since": started_at,
            "elapsed_seconds": now - started_at,
        }
    return snapshot

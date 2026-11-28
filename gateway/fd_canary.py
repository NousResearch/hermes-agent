"""fd budget canary for the messaging gateway (MTK Vikunja card #150).

The gateway leaked ~95 SQLite descriptors/day (one connection per session-end
consolidation, one per cached agent) until the 2026-11-01 incident ran it into
select()'s FD_SETSIZE cliff. The leak is fixed at the source (mnemosyne
``BeamMemory.close()`` + provider shutdown on cache evict), but this canary
pages the home channels if the process ever crosses the fd budget again, long
before the 4096 soft limit makes ``Popen`` fail outright with EMFILE.

Runs as a gateway housekeeping chore — every tick, rate-gated inside: at most
one page per ``PAGE_WINDOW_SECONDS`` while over budget, and the limiter resets
once the count returns under budget. Threshold override via
``HERMES_FD_CANARY_THRESHOLD`` (default 512). An ERROR line is logged on every
crossing regardless of the page limiter, so Loki-side alerting sees it too.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 512
PAGE_WINDOW_SECONDS = 3600

_state: dict[str, float] = {"last_page_ts": 0.0, "last_count": -1}


def fd_count(pid: int | None = None) -> int:
    """Open descriptors for ``pid`` (this process when None). -1 when /proc is unreadable."""
    pid = os.getpid() if pid is None else pid
    try:
        return len(os.listdir(f"/proc/{pid}/fd"))
    except OSError:
        return -1


def _threshold() -> int:
    """HERMES_FD_CANARY_THRESHOLD, defaulting to 512; garbage falls back to default."""
    raw = (os.environ.get("HERMES_FD_CANARY_THRESHOLD") or "").strip()
    try:
        return int(raw) if raw else DEFAULT_THRESHOLD
    except ValueError:
        return DEFAULT_THRESHOLD


def evaluate(count: int, threshold: int, now: float, state: dict | None = None) -> tuple[bool, bool]:
    """Pure decision core: returns ``(should_page, is_over)``.

    Updates ``state`` in place (``last_count``, ``last_page_ts``): a crossing
    pages once, repeats are limited to one page per window, and a return to
    healthy resets the limiter so the next crossing pages immediately.
    """
    st = _state if state is None else state
    st["last_count"] = float(count)
    if count < 0:
        return False, False
    over = count > threshold
    if not over:
        st["last_page_ts"] = 0.0
        return False, False
    if now - st["last_page_ts"] < PAGE_WINDOW_SECONDS:
        return False, True
    st["last_page_ts"] = float(now)
    return True, True


async def _page_home_channels(runner) -> None:
    """Fan the page out to every served profile's home channels (auto-prune prompt pattern)."""
    from pathlib import Path

    from gateway.run import _async_profile_runtime_scope
    from gateway.warning_notifications import present_notification
    from hermes_constants import get_routing_process_hermes_home

    message = (
        f"🚨 Gateway fd canary: {fd_count()} open file descriptors, over the "
        f"{_threshold()} budget (HERMES_FD_CANARY_THRESHOLD). A descriptor leak is "
        "regressing — check /proc/<gateway-pid>/fd targets against the drain-incident "
        "runbook and Vikunja card #150."
    )
    homes: dict = {None: get_routing_process_hermes_home()}
    homes.update(getattr(runner, "_served_profile_homes", None) or {})
    for profile, home in homes.items():
        scope = _async_profile_runtime_scope(Path(home)) if profile is not None else contextlib.nullcontext()
        async with scope:
            for target_profile, platform, _cfg, home_ch, transport in list(runner._served_home_channel_transports()):
                if target_profile != profile:
                    continue

                async def _send(platform=platform, home_ch=home_ch, transport=transport, message=message):
                    await runner._send_home_channel_message(
                        platform, home_ch, transport, message,
                        "fd canary page failed for %s:%s: %s")

                await present_notification(_send, platform=platform)


def housekeeping_tick(runner=None, loop=None) -> None:
    """One housekeeping tick: check the fd budget, log, and page on a fresh crossing."""
    count = fd_count()
    should_page, over = evaluate(count, _threshold(), time.time())
    if not over:
        return
    if not should_page:
        logger.debug("fd canary: %d fds over budget %d, page rate-limited", count, _threshold())
        return
    logger.error("Gateway fd budget: %d open descriptors > %d canary threshold", count, _threshold())
    if runner is None or loop is None:
        return
    from agent.async_utils import safe_schedule_threadsafe
    safe_schedule_threadsafe(_page_home_channels(runner), loop, logger=logger,
                             log_message="fd canary page scheduling error")
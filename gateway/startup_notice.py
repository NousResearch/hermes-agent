"""Ordinary-start home-channel notice bookkeeping (#27870).

``gateway/restart_notice.py`` writes ``.restart_pending.json`` only when the running process is
going down *for a restart*, so the boot pass announces "Gateway online" on that path alone. Every
other start — a cold boot, a crash recovery under the supervisor, a CLI ``stop`` + ``start`` (all
of Windows, which has no SIGUSR1) — carries no marker and stayed silent, which on a messaging-only
install is indistinguishable from a dead gateway.

This module throttles the ordinary-start notice instead of gating it: one notice per
``MIN_INTERVAL_SECONDS``, recorded on disk so a crash loop cannot page every home channel on each
attempt (#60438). The record is best-effort — an unwritable home costs the throttle, not the
notice.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from hermes_constants import get_process_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

STARTUP_NOTICE_FILENAME = ".startup_notice.json"
MIN_INTERVAL_SECONDS = 300.0


def startup_notice_path() -> Path:
    return get_process_hermes_home() / STARTUP_NOTICE_FILENAME


def another_gateway_owns_this_home(*, home: Optional[Path] = None) -> bool:
    """True when a DIFFERENT live process owns this home's gateway: an attach, not a new start.

    Reuses the canonical liveness matcher (``gateway.status.live_gateway_pid_for_home``: pid record
    + start-time reuse guard + a live command line belonging to this home, never bare PID
    existence) instead of a second probe. The upstream ``gateway.host_attach`` decision already
    keeps a CLI from spawning a second gateway; this answers the same question at the notice, so
    no boot path can announce a start it did not make. Any probe failure answers True — an
    unanswerable probe must stay silent rather than announce twice.
    """
    try:
        from gateway.status import live_gateway_pid_for_home

        pid = live_gateway_pid_for_home(Path(home or get_process_hermes_home()))
    except Exception:
        logger.debug("Gateway-owner probe failed; suppressing the startup notice", exc_info=True)
        return True
    return pid is not None and pid != os.getpid()


def last_notice_at(*, path: Optional[Path] = None) -> Optional[float]:
    """Timestamp of the last ordinary-start notice, or ``None`` when none/unreadable."""
    try:
        data = json.loads((path or startup_notice_path()).read_text(encoding="utf-8-sig"))
        sent_at = data.get("sent_at")
        return float(sent_at) if sent_at is not None else None
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def should_send_startup_notice(
    *,
    now: Optional[float] = None,
    min_interval: float = MIN_INTERVAL_SECONDS,
    path: Optional[Path] = None,
) -> bool:
    """True when this start may announce itself: no notice inside ``min_interval`` seconds."""
    last = last_notice_at(path=path)
    if last is None:
        return True
    return (now if now is not None else time.time()) - last >= min_interval


def stamp_startup_notice(*, now: Optional[float] = None, path: Optional[Path] = None) -> None:
    """Record that a home-channel notice was delivered, best-effort."""
    with contextlib.suppress(OSError):
        atomic_json_write(
            path or startup_notice_path(),
            {"sent_at": now if now is not None else time.time()},
            indent=None,
        )

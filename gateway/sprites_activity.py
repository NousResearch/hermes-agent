"""Renewable activity protection for the managed Sprites runtime.

Each running profile owns a bounded lease. Idle release happens only after the
existing work aggregate and connector dormancy acknowledgement permit sleep.
A dead gateway cannot renew its lease; a cold boot starts a new one.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import time

from hermes_cli import sprites_api

log = logging.getLogger(__name__)
WAKE_MARKER = Path('/opt/data/state/sprites-wake')


def available() -> bool:
    return Path('/opt/hermes/.sprites-ready').is_file() and Path(sprites_api.SOCKET_PATH).exists()


class ActivityHold:
    def __init__(self, profile: str):
        from hermes_cli.service_manager import validate_profile_name
        validate_profile_name(profile)
        self.path = f'/tasks/hermes-{profile}'
        self.dormant = False
        self.lock = asyncio.Lock()

    async def renew(self):
        async with self.lock:
            if not self.dormant:
                await asyncio.to_thread(sprites_api.request, 'PUT', self.path, {'expire': '90s'})

    async def release(self):
        async with self.lock:
            try:
                await asyncio.to_thread(sprites_api.request, 'DELETE', self.path)
            except FileNotFoundError:
                pass

    async def watch(self, runner):
        try:
            while runner._running:
                try:
                    await self.renew()
                except (OSError, RuntimeError):
                    log.warning('Sprites activity lease renewal failed; retrying before expiry')
                await asyncio.sleep(10)
        finally:
            await self.release()

    async def sleep_until_wake(self, runner):
        # The refresh loop must observe dormant BEFORE the release request.
        self.dormant = True
        try:
            marker = WAKE_MARKER.stat().st_mtime_ns if WAKE_MARKER.exists() else None
            await self.release()
            previous = time.time()
            while runner._running:
                await asyncio.sleep(1)
                now = time.time()
                changed = (WAKE_MARKER.stat().st_mtime_ns if WAKE_MARKER.exists() else None) != marker
                if now - previous > 3 or changed or not runner._scale_to_zero_is_idle():
                    break
                previous = now
            self.dormant = False
            await self.renew()
        finally:
            self.dormant = False

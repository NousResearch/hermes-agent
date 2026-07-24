"""Cursor persistence for catch-up after Hermes restarts.

A single ``lastSeenAt`` ISO timestamp is written to ``$HERMES_HOME/state/carbonvoice.json``.
Writes are debounced so a burst of messages doesn't fsync per message; on
shutdown the adapter calls ``stop()`` which forces a final flush.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from .constants import DEFAULT_FLUSH_DEBOUNCE_S

logger = logging.getLogger(__name__)


def default_state_path() -> Path:
    """Resolve ``$HERMES_HOME/state/carbonvoice.json``.

    Falls back to ``~/.hermes/state/carbonvoice.json`` if the Hermes
    constants module isn't importable (e.g. running outside the gateway).
    """
    try:
        from hermes_constants import get_hermes_home
        home = get_hermes_home()
    except Exception:
        home = Path.home() / ".hermes"
    return home / "state" / "carbonvoice.json"


class Cursor:
    """Tracks ``lastSeenAt`` with debounced disk persistence."""

    def __init__(self, path: Path, flush_debounce_s: float = DEFAULT_FLUSH_DEBOUNCE_S):
        self._path = path
        self._flush_debounce_s = flush_debounce_s
        self._last_seen_at: Optional[str] = None
        self._dirty = False
        self._flush_task: Optional[asyncio.Task] = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def last_seen_at(self) -> Optional[str]:
        return self._last_seen_at

    async def load(self) -> None:
        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
            last = data.get("lastSeenAt")
            if isinstance(last, str) and last:
                self._last_seen_at = last
                logger.info("carbonvoice: resuming from %s", last)
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("carbonvoice: failed to load state: %s", exc)

    def advance(self, iso_ts: str) -> None:
        self._last_seen_at = iso_ts
        self._dirty = True
        self._schedule_flush()

    def _schedule_flush(self) -> None:
        if self._flush_task and not self._flush_task.done():
            return

        async def _delayed():
            try:
                await asyncio.sleep(self._flush_debounce_s)
                await self.flush()
            except asyncio.CancelledError:
                pass

        self._flush_task = asyncio.create_task(_delayed())

    async def flush(self) -> None:
        if not self._dirty or self._last_seen_at is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps({"lastSeenAt": self._last_seen_at}),
                encoding="utf-8",
            )
            self._dirty = False
        except Exception as exc:
            logger.warning("carbonvoice: failed to flush state: %s", exc)

    async def stop(self) -> None:
        """Cancel any pending debounced flush, then force a final write."""
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except (asyncio.CancelledError, Exception):
                pass
        self._flush_task = None
        # Force-write whatever we have, even if the dirty flag was cleared
        # mid-flight — losing a cursor advance on shutdown is worse than a
        # redundant write.
        self._dirty = True
        await self.flush()

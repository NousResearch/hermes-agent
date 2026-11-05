"""Per-session handoff order for Slack slash commands.

Bolt runs slash callbacks concurrently and a slash awaits its name lookups before the handoff, so
a cold ``users.info`` on an earlier slash would let a later one reach the gateway first (two
``/queue`` land in the FIFO swapped). A slash reserves its session's slot before any lookup and
hands over only after the slash ahead of it returned from ``handle_message``.
"""

from __future__ import annotations

import asyncio


class SlashHandoffLanes:
    def __init__(self) -> None:
        # Session key -> future of the newest slash still on its way through handle_message.
        self.tails: dict[str, asyncio.Future[None]] = {}

    def reserve(self, key: str) -> tuple[asyncio.Future[None] | None, asyncio.Future[None]]:
        """Take the session's next slot; returns (slot ahead, this slot). Call before any await."""
        ahead = self.tails.get(key)
        self.tails[key] = handed_off = asyncio.get_running_loop().create_future()
        return ahead, handed_off

    def release(self, key: str | None, ahead: asyncio.Future | None, handed_off: asyncio.Future | None) -> None:
        """Open the slot once this slash is done. Cancelled while waiting: the slash behind it
        still waits for the one ahead, or it reaches the gateway before unfinished earlier work."""
        if ahead is not None and not ahead.done():
            ahead.add_done_callback(lambda _f: self._open(key, handed_off))
        else:
            self._open(key, handed_off)

    def _open(self, key: str | None, handed_off: asyncio.Future | None) -> None:
        if handed_off is None or handed_off.done():
            return
        handed_off.set_result(None)
        if self.tails.get(key) is handed_off:
            del self.tails[key]

from __future__ import annotations

from collections.abc import Callable


class CallTurnCancelled(Exception):
    pass


class CancellationScope:
    """Cooperative cancellation shared by the reflex loop and the brain worker."""

    def __init__(self) -> None:
        self._cancelled = False
        self._reason = ""
        self._listeners: list[Callable[[str], None]] = []

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def reason(self) -> str:
        return self._reason

    def cancel(self, reason: str) -> None:
        if self._cancelled:
            return
        self._cancelled = True
        self._reason = reason
        for cb in list(self._listeners):
            try:
                cb(reason)
            except Exception:
                pass

    def raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise CallTurnCancelled(self._reason)

    def add_listener(self, cb: Callable[[str], None]) -> None:
        self._listeners.append(cb)
        if self._cancelled:
            cb(self._reason)

"""Current native Matrix evidence for persisted pending input."""

from __future__ import annotations

from gateway.pending_native import PendingNativeInput
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent


class MatrixPendingReplayMixin(BasePlatformAdapter):
    def pending_native_input(self, event: MessageEvent) -> PendingNativeInput | None:
        if not isinstance(event.raw_message, dict) or not isinstance(
            event.raw_message.get("msgtype"), str
        ):
            return None
        return PendingNativeInput.capture(event, event.raw_message)

"""Scoped leases that keep the shutdown drain waiting for post-handler uploads.

The agent slot (``_running_agents``) is released in the turn's ``finally``
(``_run_agent_cleanup_turn_tasks``), but the turn keeps delivering afterwards:
the final text send, auto-TTS playback, MEDIA/local-file attachments and the
streamed-final attachment rescan all upload after the slot is gone. A
``--replace`` restart drains on ``_running_agents`` alone, so it used to tear
the transport down mid-upload and lose the attachment.

Every post-handler attachment path holds a scoped delivery lease for the
duration of its upload(s); ``GatewayRunner._drain_work_counts`` (which
``_drain_active_agents`` polls) includes the active lease count, so the drain
stays pending until the last lease is released. Leases ride the interactive
(chat-turn) drain budget, not the cron floor: the session is already
resume_pending, and unlike cron work a lost upload is recoverable by the
delivery ledger.
"""

from contextlib import asynccontextmanager

__all__ = ["DeliveryLease", "DeliveryLeaseRegistry"]


class DeliveryLease:
    """One in-flight post-handler delivery; lives in the registry until released."""

    __slots__ = ("registry", "description", "released")

    def __init__(self, registry: "DeliveryLeaseRegistry", description: str) -> None:
        self.registry = registry
        self.description = description
        self.released = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.registry._active.discard(self)


class DeliveryLeaseRegistry:
    """Tracks in-flight post-handler delivery leases; the shutdown drain waits on the count."""

    def __init__(self) -> None:
        self._active: set[DeliveryLease] = set()

    def __len__(self) -> int:
        return len(self._active)

    @property
    def active_count(self) -> int:
        return len(self._active)

    @asynccontextmanager
    async def lease(self, description: str = "post-handler delivery"):
        """Hold a lease for the duration of one post-handler delivery (nesting is fine:
        each held lease keeps the drain pending until it is released)."""
        hold = DeliveryLease(self, description)
        self._active.add(hold)
        try:
            yield hold
        finally:
            hold.release()

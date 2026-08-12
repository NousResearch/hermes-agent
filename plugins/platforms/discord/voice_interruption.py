"""Epoch-owned one-shot arbitration for Discord voice interruption.

This module is intentionally provider-free.  It owns only lifecycle identity and
bookkeeping; adapter code performs transport, TTS, and task cancellation outside
its lock.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AckGrant:
    """Capability authorizing one acknowledgement for one playback epoch."""

    guild_id: int
    token: int
    serial: int
    source: str


@dataclass(slots=True)
class _Epoch:
    token: int
    grant: AckGrant | None = None
    ack_task: Any | None = None
    playback_finished: bool = False


class VoiceInterruptionArbiter:
    """Lock-protected playback epoch and one-shot ACK capability owner."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._epochs: dict[int, _Epoch] = {}
        self._grant_serial = 0

    def open_epoch(self, guild_id: int, token: int) -> int:
        """Install ``token`` as the guild's sole live playback epoch."""
        with self._lock:
            previous = self._epochs.get(guild_id)
            self._epochs[guild_id] = _Epoch(token=token)
            old_task = previous.ack_task if previous is not None else None
        if old_task is not None:
            try:
                old_task.cancel()
            except Exception:
                pass
        return token

    def claim_wake(self, guild_id: int, token: int, source: str) -> AckGrant | None:
        """Claim the exact active epoch once and move it to ACK-pending."""
        with self._lock:
            epoch = self._epochs.get(guild_id)
            if epoch is None or epoch.token != token or epoch.grant is not None:
                return None
            self._grant_serial += 1
            grant = AckGrant(guild_id, token, self._grant_serial, source)
            epoch.grant = grant
            return grant

    def validate_grant(self, grant: AckGrant) -> bool:
        """Return whether ``grant`` is the exact live capability."""
        with self._lock:
            epoch = self._epochs.get(grant.guild_id)
            return bool(epoch is not None and epoch.grant == grant)

    def bind_ack_task(self, grant: AckGrant, task: Any) -> bool:
        """Bind the detached ACK task while its grant remains exact and live."""
        with self._lock:
            epoch = self._epochs.get(grant.guild_id)
            if epoch is None or epoch.grant != grant or epoch.ack_task is not None:
                return False
            epoch.ack_task = task
            return True

    def terminate_scope(self, guild_id: int, reason: str) -> tuple[Any, ...]:
        """Revoke a guild scope and return detached tasks for outside cancellation."""
        del reason
        with self._lock:
            epoch = self._epochs.pop(guild_id, None)
            if epoch is None or epoch.ack_task is None:
                return ()
            return (epoch.ack_task,)

    def playback_finished(self, guild_id: int, token: int) -> None:
        """Finish an original transport, terminating an unclaimed epoch."""
        with self._lock:
            epoch = self._epochs.get(guild_id)
            if epoch is None or epoch.token != token:
                return
            if epoch.grant is None:
                self._epochs.pop(guild_id, None)
            else:
                epoch.playback_finished = True

    def complete_ack(self, grant: AckGrant) -> None:
        """Make the exact grant terminal and remove its epoch bookkeeping."""
        with self._lock:
            epoch = self._epochs.get(grant.guild_id)
            if epoch is not None and epoch.grant == grant:
                self._epochs.pop(grant.guild_id, None)

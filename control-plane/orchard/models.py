"""Core data types shared across the control-plane."""
from __future__ import annotations

import enum
from dataclasses import dataclass


class WorkerStatus(str, enum.Enum):
    ASLEEP = "asleep"      # no resident sandbox/process
    WARMING = "warming"    # sandbox starting, not yet ready
    READY = "ready"        # sandbox up, accepting messages
    BUSY = "busy"          # currently processing a turn
    STOPPING = "stopping"  # being put to sleep


@dataclass
class Employee:
    """A provisioned tenant. `id` is the internal slug; `mm_user_id` links the
    single Mattermost bot's inbound messages to this tenant."""
    id: str
    display_name: str
    mm_user_id: str
    created_at: float

    @staticmethod
    def valid_id(value: str) -> bool:
        import re
        return bool(re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", value or ""))


@dataclass
class InboundMessage:
    """Normalized inbound message from any ingress."""
    sender_id: str      # platform user id (e.g. Mattermost user_id) — the identity
    channel_id: str     # where to reply
    text: str
    thread_id: str | None = None
    sender_name: str | None = None   # human name/@handle, for the profile display name

    def session_name(self) -> str:
        # One conversation per (channel, thread). Stable → Hermes resumes it.
        base = self.channel_id
        return f"{base}:{self.thread_id}" if self.thread_id else base


@dataclass
class WorkerInfo:
    employee_id: str
    status: WorkerStatus
    last_used: float
    pid: int | None = None

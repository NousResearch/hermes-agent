from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class CallState(str, Enum):
    WAITING = "waiting"
    ACTIVE = "active"
    ENDED = "ended"
    FAILED = "failed"


class CallError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class CallSession:
    call_id: str
    platform: str
    chat_id: str
    user_id: str
    mode: str
    state: CallState
    room_url: str | None
    created_at: datetime
    expires_at: datetime
    ended_at: datetime | None = None
    last_error_code: str | None = None

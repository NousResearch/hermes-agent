# server/models.py
"""Message data model for the chatroom."""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
import json


@dataclass
class Message:
    channel: str
    sender: str
    content: str
    msg_type: str = "message"  # message, task, review, consensus, system
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, data: str) -> "Message":
        d = json.loads(data)
        return cls(**d)

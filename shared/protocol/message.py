"""
Protocol JSON message structures and serialization utils.
Supports WebSocket, HTTP, IPC, Unix domain sockets, and TCP.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
import json


class MessageType(str, Enum):
    EVENT = "agent.event"
    REQUEST = "agent.request"
    RESPONSE = "agent.response"
    HANDSHAKE = "connection.handshake"
    ERROR = "agent.error"


@dataclass
class ProtocolMessage:
    msg_type: MessageType
    session_id: str
    message_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.msg_type.value if isinstance(self.msg_type, Enum) else str(self.msg_type),
            "session_id": self.session_id,
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProtocolMessage":
        return cls(
            msg_type=MessageType(data.get("type", "agent.event")),
            session_id=data.get("session_id", "default"),
            message_id=data.get("message_id", ""),
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "ProtocolMessage":
        return cls.from_dict(json.loads(json_str))


class AgentEventMessage(ProtocolMessage):
    def __init__(self, session_id: str, message_id: str, event_name: str, payload: Dict[str, Any]) -> None:
        full_payload = dict(payload)
        full_payload["event_name"] = event_name
        super().__init__(
            msg_type=MessageType.EVENT,
            session_id=session_id,
            message_id=message_id,
            payload=full_payload,
        )


class AgentRequestMessage(ProtocolMessage):
    def __init__(self, session_id: str, message_id: str, action: str, params: Dict[str, Any]) -> None:
        full_payload = {"action": action, "params": params}
        super().__init__(
            msg_type=MessageType.REQUEST,
            session_id=session_id,
            message_id=message_id,
            payload=full_payload,
        )


class AgentResponseMessage(ProtocolMessage):
    def __init__(self, session_id: str, message_id: str, success: bool, result: Any, error: Optional[str] = None) -> None:
        full_payload = {"success": success, "result": result, "error": error}
        super().__init__(
            msg_type=MessageType.RESPONSE,
            session_id=session_id,
            message_id=message_id,
            payload=full_payload,
        )

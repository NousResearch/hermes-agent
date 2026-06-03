"""Audit event structures for hermes-agent."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class Stage(Enum):
    """Request lifecycle stages."""

    REQUEST_RECEIVED = "RequestReceived"
    RESPONSE_STARTED = "ResponseStarted"
    RESPONSE_COMPLETE = "ResponseComplete"
    PANIC = "Panic"


@dataclass
class OperationContext:
    """
    Context for an audited operation.
    Passed to AuditEngine.emit() to build an AuditEvent.
    """

    # Who
    user: str = ""
    user_groups: List[str] = field(default_factory=list)

    # Where
    platform: str = ""  # wecom, discord, slack, etc.
    channel: str = ""  # chat/channel identifier
    session_id: str = ""

    # What action
    verb: str = ""  # get, list, create, update, delete, exec
    resource: str = ""  # tool name or resource type
    op_type: str = ""  # Mutate | Read

    # Source
    source: str = ""  # agent | gateway | platform | acp

    # Request info
    request_uri: str = ""


@dataclass
class AuditEvent:
    """
    Audit event for hermes-agent.
    Captures who did what, when, and with what result.
    """

    # Timing
    ts: str = ""  # ISO8601 timestamp
    stage: str = ""  # RequestReceived | ResponseStarted | ResponseComplete | Panic
    stage_ts: str = ""

    # Who
    user: str = ""
    user_groups: List[str] = field(default_factory=list)

    # Where
    platform: str = ""
    channel: str = ""
    session_id: str = ""

    # What action
    verb: str = ""  # execute, receive, send, etc.
    resource: str = ""  # tool name or resource type
    op_type: str = ""  # Mutate, Read

    # Request/Response (level-dependent)
    request_body: Optional[str] = None
    response_body: Optional[str] = None
    response_code: int = 0

    # Source
    source: str = ""  # agent | gateway | platform | acp
    event_id: str = ""  # UUID for correlation

    # Tamper protection (filled by AuditHashChain)
    prev_hash: str = ""
    hash: str = ""

    @staticmethod
    def new_uuid() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON serialization."""
        return {
            "ts": self.ts,
            "stage": self.stage,
            "stage_ts": self.stage_ts,
            "user": self.user,
            "userGroups": self.user_groups,
            "platform": self.platform,
            "channel": self.channel,
            "sessionId": self.session_id,
            "verb": self.verb,
            "resource": self.resource,
            "opType": self.op_type,
            "requestBody": self.request_body,
            "responseBody": self.response_body,
            "responseCode": self.response_code,
            "source": self.source,
            "eventId": self.event_id,
            "prev_hash": self.prev_hash,
            "hash": self.hash,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AuditEvent":
        """Deserialize from dict."""
        return cls(
            ts=d.get("ts", ""),
            stage=d.get("stage", ""),
            stage_ts=d.get("stage_ts", ""),
            user=d.get("user", ""),
            user_groups=d.get("userGroups", []),
            platform=d.get("platform", ""),
            channel=d.get("channel", ""),
            session_id=d.get("sessionId", ""),
            verb=d.get("verb", ""),
            resource=d.get("resource", ""),
            op_type=d.get("opType", ""),
            request_body=d.get("requestBody"),
            response_body=d.get("responseBody"),
            response_code=d.get("responseCode", 0),
            source=d.get("source", ""),
            event_id=d.get("eventId", ""),
            prev_hash=d.get("prev_hash", ""),
            hash=d.get("hash", ""),
        )
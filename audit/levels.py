"""Audit levels following K8s audit.k8s.io/v1 pattern."""

from enum import Enum


class AuditLevel(Enum):
    """
    K8s-style audit levels.

    None        — Don't log any events.
    Metadata    — Log only metadata (user, timestamp, verb, resource).
    Request     — Log metadata + request body.
    RequestResponse — Log everything including response body.
    """

    NONE = "None"
    METADATA = "Metadata"
    REQUEST = "Request"
    REQUESTRESPONSE = "RequestResponse"

    @classmethod
    def from_string(cls, value: str) -> "AuditLevel":
        """Parse level from string (case-insensitive)."""
        upper = value.upper()
        for level in cls:
            if level.value.upper() == upper:
                return level
        return cls.NONE

    def includes_request_body(self) -> bool:
        """Whether this level captures the request body."""
        return self in (self.REQUEST, self.REQUESTRESPONSE)

    def includes_response_body(self) -> bool:
        """Whether this level captures the response body."""
        return self == self.REQUESTRESPONSE

    def includes_metadata(self) -> bool:
        """Whether this level captures metadata."""
        return self != self.NONE
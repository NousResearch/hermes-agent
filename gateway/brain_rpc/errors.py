"""Brain RPC error model (contract §5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# Stable machine codes from contract §5.2
UNAUTHENTICATED = "unauthenticated"
FORBIDDEN = "forbidden"
NOT_FOUND = "not_found"
INVALID_ARGUMENT = "invalid_argument"
PAYLOAD_TOO_LARGE = "payload_too_large"
TIMEOUT = "timeout"
UNAVAILABLE = "unavailable"
CONFLICT = "conflict"
METHOD_NOT_FOUND = "method_not_found"
VERSION_UNSUPPORTED = "version_unsupported"
RATE_LIMITED = "rate_limited"
INTERNAL = "internal"

# Which codes are retryable by default
_RETRYABLE = frozenset({TIMEOUT, UNAVAILABLE, RATE_LIMITED, INTERNAL})


@dataclass
class BrainRpcError(Exception):
    """Raised by handlers / auth to produce a structured brain_rpc_result error."""

    code: str
    message: str
    retryable: Optional[bool] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.retryable is None:
            self.retryable = self.code in _RETRYABLE
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "retryable": bool(self.retryable),
        }
        if self.details:
            out["details"] = self.details
        else:
            out["details"] = {}
        return out

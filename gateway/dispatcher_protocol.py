"""Wire-protocol dataclasses for the dispatcher IPC (Phase 2.6 client).

JSON line-delimited envelope over a Unix domain socket. Mirror of
harness's dispatcher/protocol.py so the gateway can construct and
parse envelopes without importing across repositories. The wire spec
itself lives in the harness repo at docs/dispatcher-protocol.md;
this module is the client-side reference implementation.

Server side is in harness's dispatcher/protocol.py; if the wire
shape ever changes there, the change MUST be reflected here in
the same commit. Drift between client and server is a reviewer-
blocker.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass


# --- status codes (must match harness/dispatcher/protocol.py) ---

STATUS_OK = 0
STATUS_BAD_REQUEST = 1   # malformed / unknown op
STATUS_INTERNAL = 2      # handler raised
STATUS_BUSY = 3          # handler max_inflight reached


# --- ops (must match harness/dispatcher/protocol.py) ---

OP_DISPATCH = "dispatch"
OP_PING = "ping"
OP_SHUTDOWN = "shutdown"


# --- envelope ---

@dataclass(frozen=True)
class Envelope:
    """One wire message. Either request (status absent) or response
    (status present, required). The server-side dataclass enforces
    request_id length and op validity on construction; the client-
    side dataclass trusts what comes off the wire and does only the
    parse-time checks needed to construct a response object.

    The server is the authority on validation -- a client may send
    a request_id that the server rejects, and the server returns a
    STATUS_BAD_REQUEST with the same op echoed back. Clients must
    handle that gracefully.
    """

    request_id: str
    op: str
    payload: dict
    status: int | None = None

    def to_jsonl(self) -> bytes:
        """Serialize as one JSON line (newline-terminated)."""
        d = {
            "request_id": self.request_id,
            "op": self.op,
            "payload": self.payload,
        }
        if self.status is not None:
            d["status"] = self.status
        return (json.dumps(d, ensure_ascii=False) + "\n").encode("utf-8")

    @classmethod
    def from_jsonl(cls, line: bytes) -> "Envelope":
        """Parse one JSON line into an Envelope. Raises ValueError on
        malformed JSON or missing required fields."""
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError(
                f"envelope must be a JSON object, got {type(obj).__name__}"
            )
        for required in ("request_id", "op", "payload"):
            if required not in obj:
                raise ValueError(f"missing required field {required!r}")
        if not isinstance(obj["request_id"], str):
            raise ValueError("request_id must be a string")
        if not isinstance(obj["op"], str):
            raise ValueError("op must be a string")
        if not isinstance(obj["payload"], dict):
            raise ValueError("payload must be a JSON object")
        status = obj.get("status")
        if status is not None and (
            isinstance(status, bool) or not isinstance(status, int)
        ):
            # Strict int -- bool subclasses int so isinstance(int)
            # alone lets True/False through. Reject bool explicitly.
            raise ValueError(
                f"status must be int or absent, got {type(status).__name__}"
            )
        return cls(
            request_id=obj["request_id"],
            op=obj["op"],
            payload=obj["payload"],
            status=status,
        )


# --- helpers ---

def new_request_id() -> str:
    """Generate a request_id for a new request. UUID4 hex (32 chars)."""
    return uuid.uuid4().hex


def make_request(op: str, payload: dict | None = None) -> Envelope:
    """Build a fresh request envelope with a generated request_id."""
    return Envelope(
        request_id=new_request_id(),
        op=op,
        payload=payload if payload is not None else {},
    )
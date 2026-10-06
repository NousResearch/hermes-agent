"""Wire protocol for gateway ↔ node RPC (JSON envelopes).

    Request:   {"type": <str>, "id": <str>, "token": <str>, "payload": <dict>}
    Response:  {"type": "response", "id": <req-id>, "payload": <dict>}
    Error:     {"type": "error", "id": <req-id>, "error": <str>}

Requests carry the shared bearer token (``hermes meet node approve`` on the gateway, read off
disk on the server); mismatched tokens are rejected before dispatch.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Tuple


VALID_REQUEST_TYPES = frozenset({"start_bot", "stop", "status", "transcript", "say", "ping"})


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def make_request(type: str, token: str, payload: Dict[str, Any], req_id: str | None = None) -> Dict[str, Any]:
    """Construct a request envelope; ``req_id`` defaults to a uuid4 hex."""
    if not _nonempty_str(type):
        raise ValueError("type must be a non-empty string")
    if type not in VALID_REQUEST_TYPES:
        raise ValueError(f"unknown request type: {type!r}")
    if not isinstance(token, str):
        raise ValueError("token must be a string")
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")
    return {"type": type, "id": req_id or uuid.uuid4().hex, "token": token, "payload": payload}


def make_response(req_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Build a success envelope; clients correlate replies by ``id``, not type."""
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")
    return {"type": "response", "id": req_id, "payload": payload}


def make_error(req_id: str, error: str) -> Dict[str, Any]:
    return {"type": "error", "id": req_id, "error": str(error)}


def encode(msg: Dict[str, Any]) -> str:
    """Serialize a message envelope to a JSON string."""
    return json.dumps(msg, separators=(",", ":"), ensure_ascii=False)


def decode(raw) -> Dict[str, Any]:
    """Parse a JSON envelope (object with string ``type`` + ``id``) from str/bytes; ValueError otherwise.
    Token match and payload shape are checked server-side in :func:`validate_request`."""
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    try:
        obj = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("envelope must be a JSON object")
    for key in ("type", "id"):
        if not isinstance(obj.get(key), str):
            raise ValueError(f"envelope missing string '{key}'")
    return obj


def validate_request(msg: Dict[str, Any], expected_token: str) -> Tuple[bool, str]:
    """Return ``(True, "")`` or ``(False, <reason>)``; reasons are safe to send back to the client."""
    if not isinstance(msg, dict):
        return False, "envelope must be a dict"
    t, token = msg.get("type"), msg.get("token")
    checks = (  # ordered, lazily evaluated: first failing check wins
        (lambda: _nonempty_str(t), "missing or non-string 'type'"),
        (lambda: t in VALID_REQUEST_TYPES, f"unknown request type: {t!r}"),
        (lambda: _nonempty_str(msg.get("id")), "missing or non-string 'id'"),
        (lambda: _nonempty_str(token), "missing token"),
        (lambda: token == expected_token, "token mismatch"),
        (lambda: isinstance(msg.get("payload"), dict), "payload must be a dict"))
    return next(((False, reason) for ok, reason in checks if not ok()), (True, ""))

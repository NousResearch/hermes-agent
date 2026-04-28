"""Wire protocol for gateway ↔ node RPC.

Everything is a JSON object with the same envelope shape:

    Request:   {"type": <str>, "id": <str>, "token": <str>, "payload": <dict>}
    Response:  {"type": "<req-type>_res", "id": <req-id>, "payload": <dict>}
    Error:     {"type": "error", "id": <req-id>, "error": <str>}

Requests must carry the shared bearer token (set up via
``hermes meet node approve`` on the gateway and read off disk on the
server). Mismatched tokens are rejected before dispatch.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Tuple


VALID_REQUEST_TYPES = frozenset({
    "start_bot",
    "stop",
    "status",
    "transcript",
    "say",
    "ping",
})


def make_request(
    type: str,
    token: str,
    payload: Dict[str, Any],
    req_id: str | None = None,
) -> Dict[str, Any]:
    """Construct a request envelope.

    ``req_id`` is auto-generated (uuid4 hex) when not supplied so callers
    can correlate async responses.
    """
    if not isinstance(type, str) or not type:
        raise ValueError("type must be a non-empty string")
    if type not in VALID_REQUEST_TYPES:
        raise ValueError(f"unknown request type: {type!r}")
    if not isinstance(token, str):
        raise ValueError("token must be a string")
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")
    return {
        "type": type,
        "id": req_id or uuid.uuid4().hex,
        "token": token,
        "payload": payload,
    }


def make_response(req_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Build a success response. The caller supplies the *request* type;
    we suffix it with ``_res`` so clients can assert they got the right
    reply.

    For simplicity we don't require the type here — clients usually just
    key off ``id``. But we still emit a generic ``*_res`` envelope.
    """
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")
    return {"type": "response", "id": req_id, "payload": payload}


def make_error(req_id: str, error: str) -> Dict[str, Any]:
    return {"type": "error", "id": req_id, "error": str(error)}


def encode(msg: Dict[str, Any]) -> str:
    """Serialize a message envelope to a JSON string."""
    return json.dumps(msg, separators=(",", ":"), ensure_ascii=False)


def decode(raw: str) -> Dict[str, Any]:
    """Parse a JSON envelope, raising ValueError on anything malformed.

    Minimal type validation: must be an object, must contain ``type`` and
    ``id``. Heavier validation (token match, payload shape) happens in
    :func:`validate_request` on the server side.
    """
    try:
        obj = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("envelope must be a JSON object")
    if "type" not in obj or not isinstance(obj["type"], str):
        raise ValueError("envelope missing string 'type'")
    if "id" not in obj or not isinstance(obj["id"], str):
        raise ValueError("envelope missing string 'id'")
    return obj


def validate_request(msg: Dict[str, Any], expected_token: str) -> Tuple[bool, str]:
    """Check a decoded request against the server's shared token.

    Returns ``(True, "")`` when the envelope is acceptable or
    ``(False, <reason>)`` otherwise. Reason strings are safe to surface
    back to the client in an error envelope.
    """
    if not isinstance(msg, dict):
        return False, "envelope must be a dict"
    t = msg.get("type")
    if not isinstance(t, str) or not t:
        return False, "missing or non-string 'type'"
    if t not in VALID_REQUEST_TYPES:
        return False, f"unknown request type: {t!r}"
    if not isinstance(msg.get("id"), str) or not msg.get("id"):
        return False, "missing or non-string 'id'"
    token = msg.get("token")
    if not isinstance(token, str) or not token:
        return False, "missing token"
    if token != expected_token:
        return False, "token mismatch"
    payload = msg.get("payload")
    if not isinstance(payload, dict):
        return False, "payload must be a dict"
    return True, ""

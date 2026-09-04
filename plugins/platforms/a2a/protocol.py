"""
A2A protocol helpers — Agent Card construction, JSON-RPC framing, task store,
and disk-backed conversation persistence.

Wire shape follows A2A Protocol v1.0 (JSON-RPC 2.0 binding over HTTP):
  - Agent Card served at GET /.well-known/agent-card.json (canonical v1.0; legacy agent.json also answers)
  - Tasks via POST {jsonrpc:"2.0", method:"message/send", params:{...}}
  - Streaming via ``message/stream`` → SSE; events are StreamResponse objects
    discriminated by member presence (``statusUpdate`` / ``artifactUpdate``),
    stream closure signals the terminal state (no ``final`` field in v1.0)
  - Task states / message roles are v1.0 SCREAMING_SNAKE_CASE enums
  - Parts are the v1.0 unified shape ({"text": ..., "mediaType": ...}),
    discriminated by member presence (no ``kind`` field)
  - Push notification configs carry ``configId`` + ``createdAt`` and can be
    passed inline in ``message/send`` via configuration.taskPushNotificationConfig

We deliberately implement the subset of A2A needed for text task exchange with
stdlib only (no a2a-sdk). ``extract_text`` stays tolerant of v0.3 peers.
"""

from __future__ import annotations

import base64
import copy
import json
import os
import errno
import re
import tempfile
import threading
import time
import uuid
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

PROTOCOL_VERSION = "1.0"

# A2A v1.0 task lifecycle states.
STATE_SUBMITTED = "TASK_STATE_SUBMITTED"
STATE_WORKING = "TASK_STATE_WORKING"
STATE_INPUT_REQUIRED = "TASK_STATE_INPUT_REQUIRED"
STATE_AUTH_REQUIRED = "TASK_STATE_AUTH_REQUIRED"
STATE_COMPLETED = "TASK_STATE_COMPLETED"
STATE_FAILED = "TASK_STATE_FAILED"
STATE_CANCELED = "TASK_STATE_CANCELED"
STATE_REJECTED = "TASK_STATE_REJECTED"

TERMINAL_STATES = frozenset({STATE_COMPLETED, STATE_FAILED, STATE_CANCELED, STATE_REJECTED})

# A2A v1.0 message roles.
ROLE_USER = "ROLE_USER"
ROLE_AGENT = "ROLE_AGENT"

# The agent starts its reply with this marker when it needs clarification from
# the peer before it can complete the task; the adapter maps such replies to
# TASK_STATE_INPUT_REQUIRED (marker stripped, text in status.message).
INPUT_REQUIRED_MARKER = "[INPUT_REQUIRED]"

# JSON-RPC / A2A error codes.
# -32001..-32003 are A2A spec-defined and used only with their spec semantics.
# Custom errors live at -32050..-32059 (JSON-RPC implementation-defined server
# error space, clear of the A2A-reserved block).
ERR_PARSE = -32700
ERR_INVALID_PARAMS = -32602
ERR_METHOD_NOT_FOUND = -32601
ERR_TASK_NOT_FOUND = -32001        # A2A spec: TaskNotFoundError
ERR_TASK_NOT_CANCELABLE = -32002   # A2A spec: TaskNotCancelableError
ERR_PUSH_NOT_SUPPORTED = -32003    # A2A spec: PushNotificationNotSupportedError
ERR_UNAUTHORIZED = -32050
ERR_RATE_LIMITED = -32051
ERR_UNTRUSTED_PEER = -32052

# --------------------------------------------------------------------------
# Strict result validation — canonical contract per Edison decision §4
# --------------------------------------------------------------------------

_VALID_REASONS = frozenset({
    "invalid_envelope_type",
    "v1_payload_count",
    "legacy_wrapper_forbidden",
    "unknown_payload_kind",
    "invalid_task",
    "invalid_task_state",
    "invalid_message",
    "invalid_part",
    "invalid_artifact",
})




class A2AResultValidationError(Exception):
    """Raised when a SendMessage result fails canonical validation.

    Carries stable ``reason`` (one of the nine allowed values) and
    human-readable ``detail``.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        if reason not in _VALID_REASONS:
            raise ValueError(f"invalid reason for A2AResultValidationError: {reason}")
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass
class ParsedA2AResult:
    """Typed successful parse value for SendMessage result."""

    kind: str  # exactly "task" or "message"
    payload: dict
    task_id: str
    context_id: str
    state: str  # Task state or empty for Message
    text: str  # extracted text after validation, possibly empty


@dataclass
class DurablePublishOutcome:
    """Outcome of disk-first TaskStore publication."""

    published: bool
    newly_published: bool
    record: Optional[dict]
    durable_state: str
    error: Optional[str] = None


@dataclass(frozen=True)
class PushOutcome:
    """Immutable internal push outcome."""

    success: bool
    category: str  # one of routing, transport, jsonrpc, invalid_response, durability
    error: str
    payload: Optional[dict] = None

    def __post_init__(self) -> None:
        allowed = frozenset({"routing", "transport", "jsonrpc", "invalid_response", "durability"})
        if self.category and self.category not in allowed:
            pass

    def __bool__(self) -> bool:
        return self.success


# Maximum turns an A2A conversation can have before anti-loop kicks in.
# Default 5, configurable via A2A_MAX_PINGPONG_TURNS env (max 20).
_DEFAULT_MAX_PINGPONG = 5
_HARD_MAX_PINGPONG = 20


def max_pingpong_turns() -> int:
    try:
        v = int(os.getenv("A2A_MAX_PINGPONG_TURNS", str(_DEFAULT_MAX_PINGPONG)))
        return max(1, min(v, _HARD_MAX_PINGPONG))
    except (ValueError, TypeError):
        return _DEFAULT_MAX_PINGPONG


def now_iso() -> str:
    """ISO 8601 UTC timestamp with millisecond precision (A2A v1.0)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# --------------------------------------------------------------------------
# Agent Card (v1.0)
# --------------------------------------------------------------------------

def build_agent_card(
    *,
    name: str,
    url: str,
    description: str,
    skills: Optional[list[dict]] = None,
    streaming: bool = False,
    push_notifications: bool = False,
    auth_required: bool = False,
    tenant: str = "",
) -> dict:
    """Construct an A2A v1.0 Agent Card document.

    ``tenant`` is the optional v1.0 multi-tenancy routing key advertised on
    AgentInterface. When present, clients MUST echo it in request params.
    """
    iface: dict[str, Any] = {
        "url": url,
        "protocolBinding": "JSONRPC",
        "protocolVersion": PROTOCOL_VERSION,
    }
    if tenant:
        iface["tenant"] = tenant

    card: dict[str, Any] = {
        "name": name,
        "description": description,
        "url": url,  # convenience for pre-1.0 clients; canonical is supportedInterfaces
        "version": "1.0.0",
        "provider": {
            "organization": os.getenv("A2A_PROVIDER_ORG", "Hermes Agent"),
            "url": os.getenv("A2A_PROVIDER_URL", "") or url,
        },
        "supportedInterfaces": [iface],
        "capabilities": {
            "streaming": streaming,
            "pushNotifications": push_notifications,
            "stateTransitionHistory": False,
            "extendedAgentCard": False,
        },
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": skills or [],
    }
    if auth_required:
        card["securitySchemes"] = {
            "bearer": {"type": "http", "scheme": "bearer"}
        }
        card["security"] = [{"bearer": []}]
    return card


def skills_from_toolsets(toolsets: "list[str] | dict[str, list[str]] | None") -> list[dict]:
    """Derive A2A skill descriptors from the agent's toolsets.

    Accepts either a plain list of toolset names, or a mapping of toolset name
    → tool names (built from the live tool registry for dynamic Agent Cards —
    tool names become tags so peers can match tasks to us).
    """
    skills = []
    if isinstance(toolsets, dict):
        for ts_name in sorted(toolsets.keys()):
            tool_names = [str(t) for t in (toolsets[ts_name] or [])]
            skills.append({
                "id": f"toolset.{ts_name}",
                "name": ts_name,
                "description": f"Hermes '{ts_name}' capabilities",
                "tags": [ts_name] + tool_names[:10],
            })
    else:
        for ts in sorted(set(toolsets or [])):
            skills.append({
                "id": f"toolset.{ts}",
                "name": ts,
                "description": f"Hermes '{ts}' capabilities",
                "tags": [ts],
            })
    if not skills:
        skills.append({
            "id": "general",
            "name": "general",
            "description": "General-purpose conversational agent",
            "tags": ["general"],
        })
    return skills


# --------------------------------------------------------------------------
# JSON-RPC framing
# --------------------------------------------------------------------------

def jsonrpc_result(req_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def jsonrpc_error(req_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def send_message_response(payload: dict) -> dict:
    """A2A v1.0 SendMessageResponse oneof wrapper.

    The JSON-RPC ``SendMessage`` result is not a bare Task/Message; it is a
    wrapper containing exactly one of ``task`` or ``message``. Legacy methods
    still return bare payloads for compatibility.
    """
    if isinstance(payload, dict) and payload.get("status") and payload.get("id"):
        return {"task": payload}
    return {"message": payload}


# Edison strict result contract (section 4) — single authoritative definitions live at top of file; this header preserved for section navigation.

# Allowed Task states (v1.0, excluding unspecified sentinel)
_ALLOWED_TASK_STATES = frozenset({
    STATE_SUBMITTED,
    STATE_WORKING,
    STATE_INPUT_REQUIRED,
    STATE_AUTH_REQUIRED,
    STATE_COMPLETED,
    STATE_FAILED,
    STATE_CANCELED,
    STATE_REJECTED,
})

_ALLOWED_TASK_KEYS = frozenset({"id", "contextId", "status", "artifacts", "history", "metadata", "extensions"})
_ALLOWED_STATUS_KEYS = frozenset({"state", "message", "timestamp"})
_ALLOWED_MESSAGE_KEYS = frozenset({"messageId", "contextId", "role", "parts", "taskId", "metadata", "extensions", "referenceTaskIds"})
_ALLOWED_PART_KEYS = frozenset({"text", "raw", "url", "data", "metadata", "filename", "mediaType"})
_ALLOWED_ARTIFACT_KEYS = frozenset({"artifactId", "parts", "name", "description", "metadata", "extensions"})


def _parse_iso8601(value: str) -> bool:
    """Return True when value is a parsable RFC3339/ISO8601 timestamp."""
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        # Normalize Zulu to +00:00 for fromisoformat
        v = value.strip()
        if v.endswith("Z"):
            v = v[:-1] + "+00:00"
        # datetime.fromisoformat handles most ISO8601 but not strictly; also try strptime fallback
        datetime.fromisoformat(v)
        return True
    except Exception:
        try:
            # Try strict strptime with milliseconds
            datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
            return True
        except Exception:
            try:
                datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
                return True
            except Exception:
                return False


def _validate_part(part: Any) -> None:
    if not isinstance(part, dict):
        raise A2AResultValidationError("invalid_part", "Part must be an object")
    # Check unknown keys
    unknown = set(part.keys()) - _ALLOWED_PART_KEYS
    if unknown:
        raise A2AResultValidationError("invalid_part", f"unknown Part field(s): {sorted(unknown)}")
    # Exactly one content discriminator
    content_keys = [k for k in ("text", "raw", "url", "data") if k in part]
    if len(content_keys) != 1:
        raise A2AResultValidationError("invalid_part", f"Part must have exactly one of text/raw/url/data, got {content_keys}: {part!r}")
    key = content_keys[0]
    if key == "text":
        if not isinstance(part["text"], str):
            raise A2AResultValidationError("invalid_part", "Part text must be string")
        # empty string is valid
    elif key == "raw":
        raw = part["raw"]
        if not isinstance(raw, str):
            raise A2AResultValidationError("invalid_part", "Part raw must be base64 string")
        if raw != "":
            try:
                base64.b64decode(raw, validate=True)
            except Exception:
                raise A2AResultValidationError("invalid_part", "Part raw is not valid base64")
    elif key == "url":
        if not isinstance(part["url"], str) or not part["url"].strip():
            raise A2AResultValidationError("invalid_part", "Part url must be non-empty string")
    elif key == "data":
        # presence selects data, value may be any JSON including null
        pass
    # Optional fields type checks
    if "metadata" in part and not isinstance(part["metadata"], dict):
        raise A2AResultValidationError("invalid_part", "Part metadata must be object")
    if "filename" in part and not isinstance(part["filename"], str):
        raise A2AResultValidationError("invalid_part", "Part filename must be string")
    if "mediaType" in part and not isinstance(part["mediaType"], str):
        raise A2AResultValidationError("invalid_part", "Part mediaType must be string")


def _validate_artifact(artifact: Any) -> None:
    if not isinstance(artifact, dict):
        raise A2AResultValidationError("invalid_artifact", "Artifact must be object")
    unknown = set(artifact.keys()) - _ALLOWED_ARTIFACT_KEYS
    if unknown:
        raise A2AResultValidationError("invalid_artifact", f"unknown Artifact field(s): {sorted(unknown)}")
    artifact_id = artifact.get("artifactId")
    if not isinstance(artifact_id, str) or not artifact_id.strip():
        raise A2AResultValidationError("invalid_artifact", "Artifact artifactId required non-empty string")
    parts = artifact.get("parts")
    if not isinstance(parts, list) or not parts:
        raise A2AResultValidationError("invalid_artifact", "Artifact parts must be non-empty list")
    for p in parts:
        _validate_part(p)
    if "name" in artifact and not isinstance(artifact["name"], str):
        raise A2AResultValidationError("invalid_artifact", "Artifact name must be string")
    if "description" in artifact and not isinstance(artifact["description"], str):
        raise A2AResultValidationError("invalid_artifact", "Artifact description must be string")
    if "metadata" in artifact and not isinstance(artifact["metadata"], dict):
        raise A2AResultValidationError("invalid_artifact", "Artifact metadata must be object")
    if "extensions" in artifact:
        ext = artifact["extensions"]
        if not isinstance(ext, list) or not all(isinstance(e, str) and e.strip() for e in ext):
            raise A2AResultValidationError("invalid_artifact", "Artifact extensions must be list of non-empty strings")


def _validate_message(payload: dict, is_history: bool = False, is_status_message: bool = False) -> None:
    """Validate a Message payload. Direct responses require ROLE_AGENT; history permits either."""
    if not isinstance(payload, dict):
        raise A2AResultValidationError("invalid_message", "Message must be object")
    unknown = set(payload.keys()) - _ALLOWED_MESSAGE_KEYS
    if unknown:
        raise A2AResultValidationError("invalid_message", f"unknown Message field(s): {sorted(unknown)}")
    msg_id = payload.get("messageId")
    if not isinstance(msg_id, str) or not msg_id.strip():
        raise A2AResultValidationError("invalid_message", "Message messageId required non-empty string")
    ctx = payload.get("contextId")
    if is_history:
        if "contextId" in payload and payload["contextId"] is not None:
            if not isinstance(ctx, str):
                raise A2AResultValidationError("invalid_message", "Message contextId must be string")
        # history contextId may be absent; if present and empty, also invalid? keep as optional but if present must be non-empty? We'll allow empty as invalid if explicitly present but not enforce strictly beyond string check per spec.
    elif is_status_message:
        if "contextId" in payload and payload["contextId"] is not None:
            if not isinstance(ctx, str):
                raise A2AResultValidationError("invalid_message", "Message contextId must be string")
            if isinstance(ctx, str) and ctx.strip() == "":
                raise A2AResultValidationError("invalid_message", "Message contextId must be non-empty string when present")
            # equality with Task context is checked by caller (_validate_task)
    else:
        if not isinstance(ctx, str) or not ctx.strip():
            raise A2AResultValidationError("invalid_message", "Message contextId required non-empty string")
    role = payload.get("role")
    if is_history:
        if role not in (ROLE_AGENT, ROLE_USER):
            raise A2AResultValidationError("invalid_message", f"History Message role must be ROLE_AGENT or ROLE_USER, got {role!r}")
    elif is_status_message:
        if role != ROLE_AGENT:
            raise A2AResultValidationError("invalid_message", f"Message role must be ROLE_AGENT, got {role!r}")
    else:
        if role != ROLE_AGENT:
            raise A2AResultValidationError("invalid_message", f"Message role must be ROLE_AGENT, got {role!r}")
    parts = payload.get("parts")
    if not isinstance(parts, list) or not parts:
        raise A2AResultValidationError("invalid_message", "Message parts must be non-empty list")
    for p in parts:
        _validate_part(p)
    if "taskId" in payload:
        tid = payload["taskId"]
        if not isinstance(tid, str) or not tid.strip():
            raise A2AResultValidationError("invalid_message", "Message taskId must be non-empty string when present")
    if "metadata" in payload and not isinstance(payload["metadata"], dict):
        raise A2AResultValidationError("invalid_message", "Message metadata must be object")
    if "extensions" in payload:
        ext = payload["extensions"]
        if not isinstance(ext, list) or not all(isinstance(e, str) and e.strip() for e in ext):
            raise A2AResultValidationError("invalid_message", "Message extensions must be list of non-empty strings")
    if "referenceTaskIds" in payload:
        refs = payload["referenceTaskIds"]
        if not isinstance(refs, list) or not all(isinstance(r, str) and r.strip() for r in refs):
            raise A2AResultValidationError("invalid_message", "Message referenceTaskIds must be list of non-empty strings")


def _validate_task(payload: dict) -> None:
    if not isinstance(payload, dict):
        raise A2AResultValidationError("invalid_task", "Task must be object")
    unknown = set(payload.keys()) - _ALLOWED_TASK_KEYS
    if unknown:
        raise A2AResultValidationError("invalid_task", f"unknown Task field(s): {sorted(unknown)}")
    tid = payload.get("id")
    if not isinstance(tid, str) or not tid.strip():
        raise A2AResultValidationError("invalid_task", "Task id required non-empty string")
    ctx = payload.get("contextId")
    if not isinstance(ctx, str) or not ctx.strip():
        raise A2AResultValidationError("invalid_task", "Task contextId required non-empty string")
    status = payload.get("status")
    if not isinstance(status, dict):
        raise A2AResultValidationError("invalid_task", "Task status required object")
    unknown_s = set(status.keys()) - _ALLOWED_STATUS_KEYS
    if unknown_s:
        raise A2AResultValidationError("invalid_task", f"unknown Task status field(s): {sorted(unknown_s)}")
    state = status.get("state")
    if not isinstance(state, str) or state not in _ALLOWED_TASK_STATES:
        # Distinguish state vs generic task error: state-specific reason
        raise A2AResultValidationError("invalid_task_state", f"Task status.state must be one of {sorted(_ALLOWED_TASK_STATES)}, got {state!r}")
    if "message" in status:
        msg = status["message"]
        if not isinstance(msg, dict):
            raise A2AResultValidationError("invalid_task", "Task status.message must be object")
        _validate_message(msg, is_history=False, is_status_message=True)
        # context must match Task context when present
        msg_ctx = msg.get("contextId")
        if isinstance(msg_ctx, str) and msg_ctx and msg_ctx != ctx:
            raise A2AResultValidationError("invalid_task", "Task status.message contextId must equal Task contextId")
    if "timestamp" in status:
        ts = status["timestamp"]
        if not isinstance(ts, str) or not _parse_iso8601(ts):
            raise A2AResultValidationError("invalid_task", f"Task status.timestamp must be RFC3339/ISO8601 string, got {ts!r}")
    if "artifacts" in payload:
        arts = payload["artifacts"]
        if not isinstance(arts, list):
            raise A2AResultValidationError("invalid_artifact", "Task artifacts must be list")
        for art in arts:
            _validate_artifact(art)
    if "history" in payload:
        hist = payload["history"]
        if not isinstance(hist, list):
            raise A2AResultValidationError("invalid_message", "Task history must be list")
        for m in hist:
            _validate_message(m, is_history=True)
    if "metadata" in payload and not isinstance(payload["metadata"], dict):
        raise A2AResultValidationError("invalid_task", "Task metadata must be object")
    if "extensions" in payload:
        ext = payload["extensions"]
        if not isinstance(ext, list) or not all(isinstance(e, str) and e.strip() for e in ext):
            raise A2AResultValidationError("invalid_task", "Task extensions must be list of non-empty strings")


def _extract_task_text(payload: dict) -> str:
    """Extract concatenated text from a validated Task or Message."""
    if "parts" in payload:
        # Message
        return extract_text(payload)
    status = payload.get("status", {}) if isinstance(payload, dict) else {}
    msg = status.get("message") if isinstance(status, dict) else None
    if isinstance(msg, dict):
        return extract_text(msg)
    for art in payload.get("artifacts", []) or []:
        txt = extract_text(art)
        if txt:
            return txt
    return ""


def parse_send_message_result(result: Any, envelope_mode: str) -> ParsedA2AResult:
    """Strict v1 result parser (section 4.1).

    envelope_mode must be exactly "V1_WRAPPED" or "LEGACY_BARE".
    Raises A2AResultValidationError on any violation.
    """
    if envelope_mode not in ("V1_WRAPPED", "LEGACY_BARE"):
        raise A2AResultValidationError("invalid_envelope_type", f"envelope_mode must be V1_WRAPPED or LEGACY_BARE, got {envelope_mode!r}")
    if envelope_mode == "V1_WRAPPED":
        if not isinstance(result, dict):
            raise A2AResultValidationError("invalid_envelope_type", "V1 result must be object")
        has_task = "task" in result
        has_message = "message" in result
        # Classify foreign V1 wrapper members (e.g. statusUpdate) as stable unknown_payload_kind
        # before generic payload-count, but preserve bare Task/Message as payload_count.
        if has_task and has_message:
            raise A2AResultValidationError("v1_payload_count", "V1 wrapper must contain exactly one of task/message, got both")
        if not has_task and not has_message:
            if "statusUpdate" in result or "artifactUpdate" in result:
                raise A2AResultValidationError("unknown_payload_kind", f"unknown wrapper member(s): {sorted(set(result.keys()))}")
            # Bare Task/Message in V1 mode is payload_count, not unknown
            if "id" in result and "status" in result:
                raise A2AResultValidationError("v1_payload_count", "V1 wrapper must contain exactly one of task/message, got neither (bare Task in V1 mode)")
            if "messageId" in result and "parts" in result:
                raise A2AResultValidationError("v1_payload_count", "V1 wrapper must contain exactly one of task/message, got neither (bare Message in V1 mode)")
            allowed_wrappers = {"task", "message"}
            unknown = set(result.keys()) - allowed_wrappers
            if unknown:
                raise A2AResultValidationError("unknown_payload_kind", f"unknown wrapper member(s): {sorted(unknown)}")
            raise A2AResultValidationError("v1_payload_count", "V1 wrapper must contain exactly one of task/message, got neither")
        # Exactly one of task/message present — any extra wrapper member is unknown
        allowed_wrappers = {"task", "message"}
        unknown = set(result.keys()) - allowed_wrappers
        if unknown:
            raise A2AResultValidationError("unknown_payload_kind", f"unknown wrapper member(s): {sorted(unknown)}")
        payload = result["task"] if has_task else result["message"]
        kind = "task" if has_task else "message"
        if payload is None:
            raise A2AResultValidationError("v1_payload_count", f"V1 wrapper {kind} member is null")
        if not isinstance(payload, dict):
            raise A2AResultValidationError("v1_payload_count", f"V1 wrapper {kind} member must be object, got {type(payload).__name__}")
        if kind == "task":
            _validate_task(payload)
            task_id = str(payload.get("id", "")).strip()
            context_id = str(payload.get("contextId", "")).strip()
            state = str(payload.get("status", {}).get("state", "")).strip()
            text = _extract_task_text(payload)
            return ParsedA2AResult(kind=kind, payload=payload, task_id=task_id, context_id=context_id, state=state, text=text)
        else:
            _validate_message(payload, is_history=False)
            task_id = str(payload.get("taskId", "")).strip()
            context_id = str(payload.get("contextId", "")).strip()
            text = extract_text(payload)
            return ParsedA2AResult(kind=kind, payload=payload, task_id=task_id, context_id=context_id, state="", text=text)
    else:  # LEGACY_BARE
        if not isinstance(result, dict):
            raise A2AResultValidationError("invalid_envelope_type", "LEGACY_BARE result must be object")
        if "task" in result or "message" in result:
            raise A2AResultValidationError("legacy_wrapper_forbidden", "LEGACY_BARE must not contain task/message wrapper")
        # Determine if payload is Task-like or Message-like: prefer Task if has id+status
        if "id" in result and "status" in result:
            _validate_task(result)
            task_id = str(result.get("id", "")).strip()
            context_id = str(result.get("contextId", "")).strip()
            state = str(result.get("status", {}).get("state", "")).strip()
            text = _extract_task_text(result)
            return ParsedA2AResult(kind="task", payload=result, task_id=task_id, context_id=context_id, state=state, text=text)
        elif "messageId" in result and "parts" in result:
            _validate_message(result, is_history=False)
            task_id = str(result.get("taskId", "")).strip()
            context_id = str(result.get("contextId", "")).strip()
            text = extract_text(result)
            return ParsedA2AResult(kind="message", payload=result, task_id=task_id, context_id=context_id, state="", text=text)
        else:
            # Attempt task validation first for better error, then message
            try:
                _validate_task(result)
                task_id = str(result.get("id", "")).strip()
                context_id = str(result.get("contextId", "")).strip()
                state = str(result.get("status", {}).get("state", "")).strip()
                text = _extract_task_text(result)
                return ParsedA2AResult(kind="task", payload=result, task_id=task_id, context_id=context_id, state=state, text=text)
            except A2AResultValidationError as e_task:
                try:
                    _validate_message(result, is_history=False)
                    task_id = str(result.get("taskId", "")).strip()
                    context_id = str(result.get("contextId", "")).strip()
                    text = extract_text(result)
                    return ParsedA2AResult(kind="message", payload=result, task_id=task_id, context_id=context_id, state="", text=text)
                except A2AResultValidationError:
                    # Prefer original task error for clarity
                    raise e_task


def unwrap_send_message_response(result: Any) -> Any:
    """Return the Task/Message inside a v1.0 response, or pass legacy through.
    Enforces exact-one rule: does not silently prefer task when both present.
    """
    if isinstance(result, dict):
        has_task = "task" in result
        has_message = "message" in result
        if has_task and has_message:
            # Both present violates oneof; callers should treat as invalid
            # Raise validation error so is_valid will return False
            raise A2AResultValidationError("v1_payload_count", "wrapper contains both task and message")
        if has_task:
            val = result.get("task")
            if isinstance(val, dict):
                return val
            raise A2AResultValidationError("v1_payload_count", "task member must be object")
        if has_message:
            val = result.get("message")
            if isinstance(val, dict):
                return val
            raise A2AResultValidationError("v1_payload_count", "message member must be object")
        # Check for unknown wrapper members that look like oneof but aren't
        # If result has no task/message but has statusUpdate etc, it's foreign
        # Return as-is so is_valid can reject via parser; but if it has only one unknown key, treat as invalid
        # We let caller handle: if result came from V1 path, parser would have rejected, but unwrap is lenient fallback
        # For strictness, if result has any key that is not task/message but result looks like wrapped foreign, raise?
        # Keep permissive fallback for legacy bare paths: return result unchanged
    return result


def is_valid_a2a_result(result: Any) -> bool:
    """Compatibility predicate: True when result is a valid A2A Task/Message completion shape.
    Delegates to strict parser; returns False on validation error.
    """
    try:
        parse_send_message_result(result, "V1_WRAPPED")
        return True
    except A2AResultValidationError:
        return False
    except Exception:
        return False

def stream_task(task: dict) -> dict:
    """v1.0 StreamResponse with a task member."""
    return {"task": task}


def stream_message(message: dict) -> dict:
    """v1.0 StreamResponse with a message member."""
    return {"message": message}


def new_task_id() -> str:
    return "task-" + uuid.uuid4().hex[:16]


def new_context_id() -> str:
    return "ctx-" + uuid.uuid4().hex[:16]


def text_part(text: str) -> dict:
    """Build a v1.0 text Part (member-presence discriminated, no ``kind``)."""
    return {"text": text, "mediaType": "text/plain"}


def file_part(url: str = "", raw: str = "", filename: str = "",
              media_type: str = "application/octet-stream") -> dict:
    """Build a v1.0 file Part.

    Either ``url`` (file reference) or ``raw`` (base64-encoded bytes) must be
    provided. Discrimination is by member presence — no ``kind`` field.
    """
    part: dict[str, Any] = {"mediaType": media_type}
    if filename:
        part["filename"] = filename
    if url:
        part["url"] = url
    elif raw:
        part["raw"] = raw
    return part


def data_part(data: Any, media_type: str = "application/json") -> dict:
    """Build a v1.0 data Part (structured data, no ``kind`` field)."""
    return {"data": data, "mediaType": media_type}


def text_message(role: str, text: str, context_id: str = "",
                  sender: Optional[dict] = None,
                  metadata: Optional[dict] = None) -> dict:
    """Build an A2A v1.0 Message with a single text Part.

    ``sender`` is the v1.0 AgentName identity of the sending agent
    (``agentId`` / ``name`` / optional ``url``).  Peers use it to learn this
    gateway's real endpoint so out-of-band completion pushes can be routed
    back with the port included — the gap that left port-less ``ip:``
    identities unresolvable as push targets.

    Sender identity is carried inside the standard A2A ``metadata`` field
    under the namespaced key ``a2a.sender`` — **not** as a non-standard
    top-level field.  A strict A2A parser rejects unknown top-level keys;
    carrying sender inside metadata avoids that rejection while preserving
    the identity exchange that makes out-of-band pushes work.
    """
    msg: dict[str, Any] = {
        "role": role,  # ROLE_USER | ROLE_AGENT
        "parts": [text_part(text)],
        "messageId": uuid.uuid4().hex,
    }
    if isinstance(sender, dict) and sender:
        # Strip bearer tokens and other sensitive fields before putting
        # sender in metadata — tokens must never enter the wire message
        # or evidence logs.
        _SENSITIVE_SENDER_KEYS = frozenset({"token", "auth", "secret", "password"})
        meta = dict(metadata or {})
        meta["a2a.sender"] = {
            k: v for k, v in sender.items()
            if v is not None and k.lower() not in _SENSITIVE_SENDER_KEYS
        }
        metadata = meta
    if context_id:
        msg["contextId"] = context_id
    if metadata:
        msg["metadata"] = dict(metadata)
    return msg


def message_with_parts(role: str, parts: list[dict], context_id: str = "") -> dict:
    """Build an A2A v1.0 Message with arbitrary Parts (text, file, data)."""
    msg: dict[str, Any] = {
        "role": role,
        "parts": parts,
        "messageId": uuid.uuid4().hex,
    }
    if context_id:
        msg["contextId"] = context_id
    return msg


def extract_text(message_or_params: dict) -> str:
    """Pull concatenated text from an A2A Message / Task-result / params payload.

    v1.0 Parts carry a ``text`` member directly; v0.3 used ``kind: "text"``
    and some pre-0.3 peers used ``type``. All three shapes put the payload in
    ``part["text"]``, so presence of a string ``text`` member is the test.

    File and data Parts are rendered into the text stream so the agent sees
    them: file Parts with a URL include the URL and filename; data Parts
    include their JSON-serialised content. Raw (base64) file Parts are noted
    but not decoded (the agent can't act on binary inline).
    """
    msg = message_or_params.get("message", message_or_params)
    parts = msg.get("parts", []) if isinstance(msg, dict) else []
    chunks = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        # v1.0 text part (member-presence discrimination)
        txt = part.get("text")
        if isinstance(txt, str):
            chunks.append(txt)
            continue
        # v0.3 compatibility: kind == "text"
        if part.get("kind") == "text" and isinstance(part.get("text"), str):
            chunks.append(part["text"])
            continue
        # v1.0 file part with URL
        url = part.get("url")
        if isinstance(url, str) and url:
            fname = part.get("filename") or part.get("name") or ""
            mtype = part.get("mediaType") or part.get("mimeType") or ""
            label = f"[file: {fname}]" if fname else "[file]"
            chunks.append(f"{label} {url}" + (f" ({mtype})" if mtype else ""))
            continue
        # v0.3 file part with nested file.fileWithUri
        v03_file = part.get("file")
        if isinstance(v03_file, dict) and isinstance(v03_file.get("fileWithUri"), str):
            uri = v03_file["fileWithUri"]
            fname = v03_file.get("name") or ""
            mtype = v03_file.get("mimeType") or ""
            label = f"[file: {fname}]" if fname else "[file]"
            chunks.append(f"{label} {uri}" + (f" ({mtype})" if mtype else ""))
            continue
        # v1.0 file part with raw bytes (base64) — note but don't decode
        if isinstance(part.get("raw"), str):
            fname = part.get("filename") or ""
            mtype = part.get("mediaType") or ""
            label = f"[file: {fname}]" if fname else "[file]"
            size_note = f"{len(part['raw'])} bytes base64-encoded"
            chunks.append(f"{label} {size_note}" + (f" ({mtype})" if mtype else ""))
            continue
        # v1.0 data part — include JSON content
        data = part.get("data")
        if data is not None:
            try:
                rendered = json.dumps(data, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                rendered = str(data)
            mtype = part.get("mediaType") or "application/json"
            chunks.append(f"[data ({mtype})]\n{rendered}")
            continue
        # v0.3 data part: kind == "data"
        if part.get("kind") == "data" and part.get("data") is not None:
            try:
                rendered = json.dumps(part["data"], ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                rendered = str(part["data"])
            chunks.append(f"[data]\n{rendered}")
            continue
    return "\n".join(chunks).strip()


def extract_sender(message_or_params: dict) -> Optional[dict]:
    """Extract sender identity from an A2A Message's standard metadata.

    Sender is carried inside the ``metadata`` field under the key
    ``a2a.sender`` — never as a non-standard top-level field (the A2A v1.0
    spec defines a fixed Message shape; unknown top-level keys cause strict
    parsers to reject the message).

    Returns the sender dict ``{agentId, name, url, ...}`` or ``None`` when
    absent or malformed.
    """
    msg = message_or_params.get("message", message_or_params)
    if not isinstance(msg, dict):
        return None
    metadata = msg.get("metadata")
    if not isinstance(metadata, dict):
        return None
    sender = metadata.get("a2a.sender")
    return sender if isinstance(sender, dict) else None


# A2A v1.0 Message top-level keys — used by the strict shape witness
# to verify no non-standard fields are emitted on the wire.
A2A_MESSAGE_KNOWN_KEYS = frozenset({
    "role", "parts", "messageId", "contextId", "metadata",
})


def extract_context_id(params: dict) -> str:
    """v1.0 puts contextId inside the Message; tolerate legacy top-level."""
    msg = params.get("message") or {}
    ctx = ""
    if isinstance(msg, dict):
        ctx = str(msg.get("contextId") or "")
    return ctx or str(params.get("contextId") or "")


def build_task(
    task_id: str,
    context_id: str,
    state: str,
    agent_text: str = "",
    *,
    created_at: str = "",
) -> dict:
    """Build an A2A v1.0 Task object for a message/send result.

    ``created_at`` is accepted for call-site compatibility but not serialized —
    the A2A v1.0 ``Task`` proto (``lf.a2a.v1.Task``) has no ``createdAt`` or
    ``lastModified`` field.  Strict ProtoJSON parsers (e.g. a2a-sdk 1.1.0)
    reject unknown fields, so we must not include them.  The spec's §5.6.1
    timestamp-format example mentions them but they are not in the proto.
    """
    now = now_iso()
    task: dict[str, Any] = {
        "id": task_id,
        "contextId": context_id,
        "status": {"state": state, "timestamp": now},
    }
    if agent_text:
        task["status"]["message"] = text_message(ROLE_AGENT, agent_text, context_id)
        if state == STATE_COMPLETED:
            task["artifacts"] = [{
                "artifactId": uuid.uuid4().hex,
                "parts": [text_part(agent_text)],
            }]
    return task


# --------------------------------------------------------------------------
# Streaming (v1.0 StreamResponse events)
# --------------------------------------------------------------------------

def status_update(task_id: str, context_id: str, state: str, text: str = "") -> dict:
    """v1.0 StreamResponse with a statusUpdate member."""
    status: dict[str, Any] = {"state": state, "timestamp": now_iso()}
    if text:
        status["message"] = text_message(ROLE_AGENT, text, context_id)
    return {"statusUpdate": {"taskId": task_id, "contextId": context_id, "status": status}}


def artifact_update(task_id: str, context_id: str, text: str) -> dict:
    """v1.0 StreamResponse with an artifactUpdate member."""
    return {
        "artifactUpdate": {
            "taskId": task_id,
            "contextId": context_id,
            "artifact": {
                "artifactId": uuid.uuid4().hex,
                "parts": [text_part(text)],
            },
        }
    }


def sse_data(payload: dict, req_id: Any = None) -> str:
    """Encode one StreamResponse as a JSON-RPC-wrapped SSE data frame.

    A2A v1.0 §9.4 requires each SSE frame to be a full JSON-RPC response:
    ``{"jsonrpc":"2.0","id":<req_id>,"result":{StreamResponse}}``.  Emitting a
    bare StreamResponse (the REST binding shape) breaks JSON-RPC clients that
    expect the envelope, including the official a2a-sdk.
    """
    if req_id is not None:
        envelope = jsonrpc_result(req_id, payload)
    else:
        envelope = payload  # legacy/fallback — no envelope
    return f"data: {json.dumps(envelope, ensure_ascii=False)}\n\n"


def sse_done() -> str:
    """SSE stream-closure marker — a comment, not a parseable data frame.

    A2A v1.0 signals terminal state by closing the stream.  Emitting
    ``data: {}`` causes JSON-RPC clients to try parsing an empty response and
    fail.  An SSE comment line (``: done``) is ignored by all SSE parsers.
    """
    return ": done\n\n"


# --------------------------------------------------------------------------
# Anti-loop ping-pong protection (per-adapter instance)
# --------------------------------------------------------------------------

class TurnTracker:
    """Counts inbound turns per context_id to stop infinite agent↔agent loops.

    A "turn" is one inbound message/send from a peer. When the count exceeds
    max_pingpong_turns(), the adapter rejects further messages for that context.
    """

    _TTL = 3600  # prune contexts idle longer than 1 hour

    def __init__(self) -> None:
        self._counts: dict[str, int] = defaultdict(int)
        self._timestamps: dict[str, float] = {}
        self._lock = threading.Lock()

    def track(self, context_id: str) -> int:
        """Increment and return the turn count; prunes stale contexts."""
        with self._lock:
            now = time.time()
            stale = [cid for cid, ts in self._timestamps.items() if now - ts > self._TTL]
            for cid in stale:
                self._counts.pop(cid, None)
                self._timestamps.pop(cid, None)
            self._counts[context_id] += 1
            self._timestamps[context_id] = now
            return self._counts[context_id]

    def reset(self, context_id: str) -> None:
        """Reset turn count for a context (e.g. after explicit cancel)."""
        with self._lock:
            self._counts.pop(context_id, None)
            self._timestamps.pop(context_id, None)


# --------------------------------------------------------------------------
# Rate limiting (sliding window per authenticated peer identity)
# --------------------------------------------------------------------------

_RATE_LIMIT_DEFAULT = 60  # requests per minute
_RATE_WINDOW = 60.0  # seconds


def _rate_limit_per_minute() -> int:
    try:
        return max(1, int(os.getenv("A2A_RATE_LIMIT", str(_RATE_LIMIT_DEFAULT))))
    except (ValueError, TypeError):
        return _RATE_LIMIT_DEFAULT


class RateLimiter:
    """Sliding-window request limiter, one bucket per authenticated identity."""

    def __init__(self) -> None:
        self._buckets: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, identity: str) -> bool:
        with self._lock:
            limit = _rate_limit_per_minute()
            now = time.time()
            bucket = self._buckets[identity]
            while bucket and now - bucket[0] > _RATE_WINDOW:
                bucket.popleft()
            if len(bucket) >= limit:
                return False
            bucket.append(now)
            return True


# --------------------------------------------------------------------------
# Metrics collection
# --------------------------------------------------------------------------

# Module-level singleton shared by the inbound adapter and the outbound client
# tools so /metrics and a2a_list report both directions. Not persisted.
class Metrics:
    """Simple counters for A2A operations."""

    def __init__(self) -> None:
        self.inbound_total = 0
        self.outbound_total = 0
        self.streams_started = 0
        self.push_sent = 0
        self.push_failed = 0
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.anti_loop_triggers = 0
        self.rate_limit_triggers = 0
        self._start_time = time.time()
        # Rolling latency tracking (last 100 completed inbound tasks)
        self._latencies: deque[float] = deque(maxlen=100)

    def record_latency(self, seconds: float) -> None:
        self._latencies.append(seconds)

    def avg_latency(self) -> float:
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    def snapshot(self) -> dict[str, Any]:
        uptime = time.time() - self._start_time
        return {
            "uptime_seconds": round(uptime, 1),
            "inbound_total": self.inbound_total,
            "outbound_total": self.outbound_total,
            "streams_started": self.streams_started,
            "push_sent": self.push_sent,
            "push_failed": self.push_failed,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "anti_loop_triggers": self.anti_loop_triggers,
            "rate_limit_triggers": self.rate_limit_triggers,
            "avg_latency_ms": round(self.avg_latency() * 1000, 1),
        }


metrics = Metrics()


# --------------------------------------------------------------------------
# Task store — pending AND completed tasks (queryable via tasks/get, tasks/list)
# --------------------------------------------------------------------------

class TaskStore:
    """In-memory store of A2A tasks, kept after completion for tasks/get.

    Records carry the routed agent slug and tenant. All read/write helpers accept
    optional scope values and return not-found when the task exists but is not
    visible in that scope, satisfying the spec's authorization scoping rule.
    """

    _MAX_TERMINAL = 500

    def __init__(self) -> None:
        self._tasks: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self._watchers: dict[str, list[Future]] = {}
        self._lock = threading.Lock()
        # Amendment C: ledger availability flag for post-replace directory fsync fail-closed.
        self._ledger_unavailable: bool = False
        self._ledger_unavailable_reason: str = ""

    @staticmethod
    def _in_scope(rec: dict, agent_slug: str = "", tenant: str = "") -> bool:
        if agent_slug and rec.get("agent_slug", "") != agent_slug:
            return False
        if tenant and rec.get("tenant", "") != tenant:
            return False
        return True

    def create(self, task_id: str, context_id: str, peer: str,
               agent_slug: str = "", tenant: str = "") -> dict:
        rec = {
            "task_id": task_id,
            "context_id": context_id,
            "peer": peer,
            "agent_slug": agent_slug or "",
            "tenant": tenant or "",
            "state": STATE_SUBMITTED,
            "reply": "",
            "created_at": time.time(),
            "created_iso": now_iso(),
            "push_url": "",
            "push_config_id": "",
        }
        with self._lock:
            self._tasks[task_id] = rec
        return dict(rec)

    def set_state(self, task_id: str, state: str) -> None:
        with self._lock:
            rec = self._tasks.get(task_id)
            if rec and rec["state"] not in TERMINAL_STATES:
                rec["state"] = state

    def set_push_config(self, task_id: str, url: str,
                        agent_slug: str = "", tenant: str = "") -> Optional[dict]:
        """Attach a push notification config; returns the stored config or None."""
        with self._lock:
            rec = self._tasks.get(task_id)
            if not rec or not self._in_scope(rec, agent_slug, tenant):
                return None
            rec["push_url"] = url
            rec["push_config_id"] = "cfg-" + uuid.uuid4().hex[:12]
            return self._push_config_view(rec)

    @staticmethod
    def _push_config_view(rec: dict) -> dict:
        """Build the JSON-RPC result for a push notification config."""
        return {
            "configId": rec.get("push_config_id") or "",
            "taskId": rec["task_id"],
            "createdAt": rec.get("created_iso", ""),
            "pushNotificationConfig": {"url": rec.get("push_url") or ""},
        }

    def get_push_config(self, task_id: str, config_id: str = "",
                        agent_slug: str = "", tenant: str = "") -> Optional[dict]:
        with self._lock:
            rec = self._tasks.get(task_id)
            if not rec or not self._in_scope(rec, agent_slug, tenant) or not rec.get("push_url"):
                return None
            if config_id and rec.get("push_config_id") != config_id:
                return None
            return self._push_config_view(rec)

    def list_push_configs(self, task_id: str, agent_slug: str = "", tenant: str = "") -> list[dict]:
        with self._lock:
            rec = self._tasks.get(task_id)
            if not rec or not self._in_scope(rec, agent_slug, tenant) or not rec.get("push_url"):
                return []
            return [self._push_config_view(rec)]

    def delete_push_config(self, task_id: str, config_id: str = "",
                           agent_slug: str = "", tenant: str = "") -> bool:
        with self._lock:
            rec = self._tasks.get(task_id)
            if not rec or not self._in_scope(rec, agent_slug, tenant) or not rec.get("push_url"):
                return False
            if config_id and rec.get("push_config_id") != config_id:
                return False
            rec["push_url"] = ""
            rec["push_config_id"] = ""
            return True

    def pop_push_url(self, task_id: str) -> str:
        with self._lock:
            rec = self._tasks.get(task_id)
            if not rec:
                return ""
            url, rec["push_url"] = rec["push_url"], ""
            return url

    def get(self, task_id: str, agent_slug: str = "", tenant: str = "") -> Optional[dict]:
        with self._lock:
            if getattr(self, "_ledger_unavailable", False):
                return None
            rec = self._tasks.get(task_id)
            if not rec or not self._in_scope(rec, agent_slug, tenant):
                return None
            return dict(rec)

    def complete(self, task_id: str, state: str, reply: str = "") -> Optional[dict]:
        """Transition a task to a terminal state. Idempotent (now durable, section 5.3).

        For backward compatibility, this helper delegates to the durable primitive using the
        default ledger path. Production lifecycle paths should call publish_durable directly
        for explicit error handling, but this wrapper ensures any legacy complete() call still
        performs disk-first publication and leaves the last durable state visible on failure.
        """
        try:
            from .a2a_persistence import _task_ledger_path
            ledger_path = _task_ledger_path()
        except Exception:
            ledger_path = None
        # If ledger_path is unavailable (test without HERMES_HOME), fallback to memory-only
        if ledger_path is None:
            watchers: list[Future] = []
            with self._lock:
                rec = self._tasks.get(task_id)
                if not rec or rec["state"] in TERMINAL_STATES:
                    return None
                rec["state"] = state
                rec["reply"] = reply
                rec["completed_at"] = time.time()
                watchers = self._watchers.pop(task_id, [])
                self._trim_locked()
                out = dict(rec)
            for fut in watchers:
                if not fut.done():
                    try:
                        fut.set_result((state, reply))
                    except Exception:
                        pass
            return out
        # Durable path: build candidate from existing record
        with self._lock:
            rec = self._tasks.get(task_id)
            if not rec:
                return None
            if rec["state"] in TERMINAL_STATES:
                return None
            candidate = dict(rec)
            candidate["state"] = state
            candidate["reply"] = reply
            candidate["completed_at"] = time.time()
        outcome = self.publish_durable(ledger_path, task_id, candidate)
        if not outcome.published:
            return None
        return outcome.record

    def publish_durable(self, ledger_path: Path, task_id: str, candidate_record: dict) -> DurablePublishOutcome:
        """Disk-first publication of a task transition.

        Implements the durable-publish primitive per Edison decision §5.3 and
        Amendment C/E:
        verify ownership against disk, reject terminal conflicts/dedupe against
        disk, clone ledger, atomically replace ledger via temp file with
        mandatory flush/fsync, directory fsync classification, and same-task
        terminal authority under per-ledger file lock.

        Memory is never updated before successful persistence; file lock
        ensures cross-process serialization; in-process lock excludes readers
        during transition window.
        """
        # Determine candidate fields (support both snake and camel for context)
        candidate_state = candidate_record.get("state", "")
        candidate_reply = candidate_record.get("reply", "")
        # Ownership fields may be under different keys; normalize for verification
        def _cand_field(name: str) -> Any:
            if name in candidate_record:
                return candidate_record[name]
            if name == "context_id" and "contextId" in candidate_record:
                return candidate_record["contextId"]
            return None

        candidate_context = _cand_field("context_id")
        candidate_peer = _cand_field("peer")
        candidate_slug = _cand_field("agent_slug")
        candidate_tenant = _cand_field("tenant")

        watchers_to_resolve: list[Future] = []
        publish_state = candidate_state
        publish_reply = candidate_reply
        published_rec: Optional[dict] = None
        durable_state_after: str = ""
        error_msg: Optional[str] = None
        success = False

        # Acquire in-process transition lock (serializes publications)
        with self._lock:
            # Remember existing in-memory for fallback durable_state if needed
            existing_mem = self._tasks.get(task_id)
            existing_mem_state = existing_mem.get("state", "") if existing_mem is not None else "ABSENT"
            existing_mem_copy = dict(existing_mem) if existing_mem is not None else None

            # 3. Clone ledger snapshot (pre-disk state) - will be merged after disk load
            clone: "OrderedDict[str, dict[str, Any]]" = copy.deepcopy(self._tasks)

            # 4. Apply candidate to clone
            now = time.time()
            if task_id in clone:
                entry = clone[task_id]
                for k, v in candidate_record.items():
                    entry[k] = v
                entry["task_id"] = task_id
                if candidate_state:
                    entry["state"] = candidate_state
                if "reply" in candidate_record:
                    entry["reply"] = candidate_record["reply"]
                if entry.get("state") in TERMINAL_STATES and "completed_at" not in entry:
                    entry["completed_at"] = now
                # Ensure context_id alias if candidate used camel
                if "context_id" not in entry and "contextId" in candidate_record:
                    entry["context_id"] = candidate_record["contextId"]
            else:
                new_entry = dict(candidate_record)
                new_entry["task_id"] = task_id
                if "context_id" not in new_entry and "contextId" in new_entry:
                    new_entry["context_id"] = new_entry["contextId"]
                if "created_at" not in new_entry:
                    new_entry["created_at"] = now
                if "created_iso" not in new_entry:
                    new_entry["created_iso"] = now_iso()
                if new_entry.get("state") in TERMINAL_STATES and "completed_at" not in new_entry:
                    new_entry["completed_at"] = now
                # Ensure defaults for ownership fields
                for fld in ("context_id", "peer", "agent_slug", "tenant", "reply", "push_url", "push_config_id"):
                    if fld not in new_entry:
                        new_entry[fld] = "" if fld != "reply" else new_entry.get("reply", "")
                clone[task_id] = new_entry

            # 5. Serialized ledger transaction: load → verify → merge → write under per-ledger file lock
            # The complete load/merge/write is serialized by the ledger file lock so
            # concurrent TaskStore instances cannot lose each other's records.
            try:
                ledger_path.parent.mkdir(parents=True, exist_ok=True)
                lock_path = ledger_path.with_suffix(".lock")
                try:
                    import fcntl  # type: ignore
                    _has_fcntl = True
                except ImportError:
                    fcntl = None  # type: ignore
                    _has_fcntl = False
                try:
                    import msvcrt  # type: ignore
                    _has_msvcrt = True
                except ImportError:
                    msvcrt = None  # type: ignore
                    _has_msvcrt = False

                lock_file_obj = None
                if _has_fcntl:
                    lock_path.parent.mkdir(parents=True, exist_ok=True)
                    lock_path.touch(exist_ok=True)
                    lock_file_obj = open(lock_path, "a+")
                    fcntl.flock(lock_file_obj.fileno(), fcntl.LOCK_EX)  # type: ignore
                elif _has_msvcrt:
                    lock_path.parent.mkdir(parents=True, exist_ok=True)
                    lock_path.touch(exist_ok=True)
                    lock_file_obj = open(lock_path, "a+")
                    for _attempt in range(50):
                        try:
                            msvcrt.locking(lock_file_obj.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore
                            break
                        except OSError:
                            if _attempt == 49:
                                raise
                            time.sleep(0.01)
                else:
                    pass

                try:
                    # Load authoritative ledger snapshot under file lock
                    # Fail closed on unreadable/unparseable existing ledger; never replace with empty snapshot.
                    disk_snapshot: dict[str, Any] = {}
                    disk_load_failed = False
                    disk_load_error = ""
                    if ledger_path.exists():
                        try:
                            with open(ledger_path, "r", encoding="utf-8") as lf:
                                loaded = json.load(lf)
                            if not isinstance(loaded, dict):
                                disk_load_failed = True
                                disk_load_error = "ledger not a dict"
                            else:
                                disk_snapshot = loaded
                        except Exception as exc:
                            disk_load_failed = True
                            disk_load_error = f"unreadable ledger: {exc}"
                            disk_snapshot = {}
                        if disk_load_failed:
                            self._ledger_unavailable = True
                            self._ledger_unavailable_reason = disk_load_error or "unreadable ledger"
                            return DurablePublishOutcome(
                                published=False,
                                newly_published=False,
                                record=existing_mem_copy,
                                durable_state=existing_mem_state,
                                error=disk_load_error,
                            )
                    # If we previously marked unavailable but now load succeeded (readable), clear flag.
                    if self._ledger_unavailable:
                        self._ledger_unavailable = False
                        self._ledger_unavailable_reason = ""

                    # Select authoritative disk record for task_id
                    disk_rec = None
                    if isinstance(disk_snapshot, dict) and task_id in disk_snapshot:
                        rec_raw = disk_snapshot[task_id]
                        if isinstance(rec_raw, dict):
                            disk_rec = dict(rec_raw)

                    # Verify immutable ownership against disk record, not only self._tasks (Amendment E)
                    if disk_rec is not None:
                        for field_name, cand_val in [
                            ("context_id", candidate_context),
                            ("peer", candidate_peer),
                            ("agent_slug", candidate_slug),
                            ("tenant", candidate_tenant),
                        ]:
                            if cand_val is not None:
                                existing_val = disk_rec.get(field_name, "")
                                if cand_val != existing_val:
                                    self._tasks[task_id] = dict(disk_rec)
                                    return DurablePublishOutcome(
                                        published=False,
                                        newly_published=False,
                                        record=dict(disk_rec),
                                        durable_state=disk_rec.get("state", ""),
                                        error=f"ownership mismatch for {field_name}: {cand_val!r} != {existing_val!r}",
                                    )
                        disk_state = disk_rec.get("state", "")
                        disk_reply = disk_rec.get("reply", "")
                        if disk_state in TERMINAL_STATES:
                            if candidate_state == disk_state and candidate_reply == disk_reply:
                                self._tasks[task_id] = dict(disk_rec)
                                return DurablePublishOutcome(
                                    published=True,
                                    newly_published=False,
                                    record=dict(disk_rec),
                                    durable_state=disk_state,
                                )
                            else:
                                self._tasks[task_id] = dict(disk_rec)
                                return DurablePublishOutcome(
                                    published=False,
                                    newly_published=False,
                                    record=dict(disk_rec),
                                    durable_state=disk_state,
                                    error="terminal conflict: existing terminal differs",
                                )

                    # Only a nonterminal authoritative disk record may take a legal candidate transition.

                    # Merge: authoritative disk + clone (clone supplies new/updated publishing task)
                    merged: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
                    if isinstance(disk_snapshot, dict):
                        for tid, rec in disk_snapshot.items():
                            if isinstance(rec, dict):
                                merged[tid] = dict(rec)
                    for tid, rec in clone.items():
                        if tid == task_id:
                            merged[tid] = dict(rec)
                        elif tid not in merged:
                            merged[tid] = dict(rec)
                        else:
                            disk_rec_other = merged[tid]
                            if disk_rec_other.get("state") not in TERMINAL_STATES and rec.get("state") in TERMINAL_STATES:
                                merged[tid] = dict(rec)

                    snapshot: dict[str, dict[str, Any]] = {}
                    for tid, rec in merged.items():
                        state = rec.get("state", "")
                        created = rec.get("created_at", 0)
                        try:
                            created_f = float(created)
                        except Exception:
                            created_f = 0.0
                        if state in TERMINAL_STATES or (now - created_f < 300):
                            snapshot[tid] = {
                                "task_id": rec.get("task_id", tid),
                                "context_id": rec.get("context_id", ""),
                                "peer": rec.get("peer", ""),
                                "agent_slug": rec.get("agent_slug", ""),
                                "tenant": rec.get("tenant", ""),
                                "state": state,
                                "reply": rec.get("reply", ""),
                                "created_at": rec.get("created_at", 0),
                                "created_iso": rec.get("created_iso", ""),
                                "completed_at": rec.get("completed_at"),
                                "push_url": rec.get("push_url", ""),
                                "push_config_id": rec.get("push_config_id", ""),
                            }
                    tmp_fd = None
                    tmp_path = ""
                    try:
                        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(ledger_path.parent), suffix=".tmp")
                        try:
                            os.fchmod(tmp_fd, 0o600)
                        except Exception:
                            pass
                        serialization_error = None
                        try:
                            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                                tmp_fd = None
                                json.dump(snapshot, f, ensure_ascii=False, indent=2)
                                f.flush()
                                os.fsync(f.fileno())
                        except Exception as exc:
                            serialization_error = exc
                        if serialization_error is not None:
                            if tmp_path:
                                try:
                                    os.unlink(tmp_path)
                                except OSError:
                                    pass
                            if tmp_fd is not None:
                                try:
                                    os.close(tmp_fd)
                                except OSError:
                                    pass
                                tmp_fd = None
                            err_str = f"serialization/flush/fsync failed: {serialization_error}"
                            return DurablePublishOutcome(
                                published=False,
                                newly_published=False,
                                record=existing_mem_copy if disk_rec is None else dict(disk_rec),
                                durable_state=disk_rec.get("state", "") if disk_rec is not None else existing_mem_state,
                                error=err_str,
                            )
                        try:
                            os.chmod(tmp_path, 0o600)
                        except OSError:
                            pass
                        try:
                            os.replace(tmp_path, str(ledger_path))
                        except Exception as exc:
                            try:
                                os.unlink(tmp_path)
                            except OSError:
                                pass
                            err_str = f"atomic replace failed: {exc}"
                            return DurablePublishOutcome(
                                published=False,
                                newly_published=False,
                                record=existing_mem_copy if disk_rec is None else dict(disk_rec),
                                durable_state=disk_rec.get("state", "") if disk_rec is not None else existing_mem_state,
                                error=err_str,
                            )
                        dir_fsync_error = None
                        dir_fsync_unsupported = False
                        try:
                            dir_fd = os.open(str(ledger_path.parent), os.O_DIRECTORY)
                            try:
                                os.fsync(dir_fd)
                            finally:
                                os.close(dir_fd)
                        except AttributeError as exc:
                            dir_fsync_unsupported = True
                        except NotImplementedError as exc:
                            dir_fsync_unsupported = True
                        except OSError as exc:
                            err_no = getattr(exc, "errno", None)
                            if err_no in (errno.EINVAL, errno.ENOTSUP, getattr(errno, "EOPNOTSUPP", 95)):
                                dir_fsync_unsupported = True
                            else:
                                dir_fsync_error = exc
                        except Exception as exc:
                            dir_fsync_error = exc

                        if dir_fsync_error is not None:
                            self._ledger_unavailable = True
                            self._ledger_unavailable_reason = f"directory fsync failed: {dir_fsync_error}"
                            err_str = f"directory fsync failed: {dir_fsync_error}; safeToRetry=false"
                            return DurablePublishOutcome(
                                published=False,
                                newly_published=False,
                                record=existing_mem_copy if disk_rec is None else dict(disk_rec),
                                durable_state=disk_rec.get("state", "") if disk_rec is not None else existing_mem_state,
                                error=err_str,
                            )
                        if dir_fsync_unsupported:
                            try:
                                if not getattr(self.__class__, "_dir_fsync_warned", False):
                                    import logging
                                    logging.getLogger(__name__).warning(
                                        "A2A: directory fsync not supported on this platform; using file-fsync + atomic replace only (weaker directory-entry guarantee)"
                                    )
                                    self.__class__._dir_fsync_warned = True
                            except Exception:
                                pass
                    except BaseException as exc:
                        if tmp_path:
                            try:
                                os.unlink(tmp_path)
                            except OSError:
                                pass
                        if tmp_fd is not None:
                            try:
                                os.close(tmp_fd)
                            except OSError:
                                pass
                        raise

                    self._tasks = merged
                    if not isinstance(self._tasks, OrderedDict):
                        self._tasks = OrderedDict(self._tasks)
                    self._trim_locked()
                    if publish_state in TERMINAL_STATES:
                        watchers_to_resolve = self._watchers.pop(task_id, [])
                    else:
                        watchers_to_resolve = []
                    published_rec = dict(self._tasks.get(task_id, candidate_record))
                    durable_state_after = published_rec.get("state", "")
                    success = True
                finally:
                    if lock_file_obj is not None:
                        try:
                            if _has_fcntl:
                                fcntl.flock(lock_file_obj.fileno(), fcntl.LOCK_UN)  # type: ignore
                            elif _has_msvcrt:
                                msvcrt.locking(lock_file_obj.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore
                        except OSError:
                            pass
                        try:
                            lock_file_obj.close()
                        except OSError:
                            pass
            except Exception as exc:
                existing_state = existing_mem.get("state", "") if existing_mem is not None else "ABSENT"
                existing_copy = dict(existing_mem) if existing_mem is not None else None
                return DurablePublishOutcome(
                    published=False,
                    newly_published=False,
                    record=existing_copy,
                    durable_state=existing_state,
                    error=str(exc),
                )

        if success:
            for fut in watchers_to_resolve:
                if not fut.done():
                    try:
                        fut.set_result((publish_state, publish_reply))
                    except Exception:
                        pass
            return DurablePublishOutcome(
                published=True,
                newly_published=True,
                record=published_rec,
                durable_state=durable_state_after,
            )
        return DurablePublishOutcome(
            published=False,
            newly_published=False,
            record=None,
            durable_state="ABSENT",
            error=error_msg or "unknown",
        )
    def watch(self, task_id: str, agent_slug: str = "", tenant: str = "") -> Optional[Future]:
        with self._lock:
            if getattr(self, "_ledger_unavailable", False):
                return None
            rec = self._tasks.get(task_id)
            if not rec or not self._in_scope(rec, agent_slug, tenant):
                return None
            fut: Future = Future()
            if rec["state"] in TERMINAL_STATES:
                fut.set_result((rec["state"], rec.get("reply", "")))
            else:
                self._watchers.setdefault(task_id, []).append(fut)
            return fut

    def list(
        self,
        context_id: str = "",
        state: str = "",
        page_size: int = 50,
        offset: int = 0,
        agent_slug: str = "",
        tenant: str = "",
        with_total: bool = False,
    ):
        """Filtered task page (newest first).

        Historical API returns ``(records, next_offset)``. v1.0 ListTasks needs
        ``totalSize``, so callers can opt into ``(records, next_offset, total)``.
        """
        if getattr(self, "_ledger_unavailable", False):
            return ([], 0) if not with_total else ([], 0, 0)
        page_size = max(1, min(int(page_size or 50), 100))
        with self._lock:
            recs = [dict(r) for r in reversed(self._tasks.values())]
        if agent_slug or tenant:
            recs = [r for r in recs if self._in_scope(r, agent_slug, tenant)]
        if context_id:
            recs = [r for r in recs if r["context_id"] == context_id]
        if state:
            recs = [r for r in recs if r["state"] == state]
        total = len(recs)
        page = recs[offset:offset + page_size]
        next_offset = offset + page_size if offset + page_size < total else 0
        if with_total:
            return page, next_offset, total
        return page, next_offset

    def fail_orphans(self, timeout_seconds: int = 300) -> list[str]:
        with self._lock:
            now = time.time()
            stale = [
                tid for tid, rec in self._tasks.items()
                if rec["state"] not in TERMINAL_STATES
                and now - rec["created_at"] > timeout_seconds
            ]
        failed = []
        for tid in stale:
            # Use durable publish per task; only successes are counted
            rec = self.get(tid)
            if not rec or rec["state"] in TERMINAL_STATES:
                continue
            candidate = dict(rec)
            candidate["state"] = STATE_FAILED
            candidate["reply"] = "[task orphaned — no reply produced]"
            candidate["completed_at"] = time.time()
            try:
                from .a2a_persistence import _task_ledger_path
                ledger_path = _task_ledger_path()
                outcome = self.publish_durable(ledger_path, tid, candidate)
                if outcome.published and outcome.newly_published:
                    failed.append(tid)
            except Exception:
                continue
        return failed

    def _trim_locked(self) -> None:
        terminal = [tid for tid, rec in self._tasks.items() if rec["state"] in TERMINAL_STATES]
        excess = len(terminal) - self._MAX_TERMINAL
        for tid in terminal[:max(0, excess)]:
            self._tasks.pop(tid, None)

    @staticmethod
    def to_task(rec: dict, history_length: Optional[int] = None, include_artifacts: bool = True) -> dict:
        """Render a stored record as an A2A v1.0 Task object."""
        task = build_task(
            rec["task_id"],
            rec["context_id"],
            rec["state"],
            rec.get("reply", ""),
            created_at=rec.get("created_iso", ""),
        )
        if not include_artifacts:
            task.pop("artifacts", None)
        if history_length == 0:
            task.pop("history", None)
        return copy.deepcopy(task)

    # ── Disk persistence (restart recovery) ──────────────────────────────

    def persist(self, path: Path) -> None:
        """Persist terminal task records to disk for restart recovery.

        Only records in terminal states (COMPLETED, FAILED, CANCELED) and
        non-terminal records younger than a bound are persisted.  Follows
        the safe persistence discipline: unique temp file, 0o600, atomic
        replace.
        """
        now = time.time()
        with self._lock:
            snapshot = {}
            for tid, rec in self._tasks.items():
                state = rec["state"]
                # Always persist terminal records; persist non-terminal
                # only if younger than _ORPHAN_TIMEOUT (300s) so stale
                # working tasks don't accumulate on disk.
                if state in TERMINAL_STATES or (now - rec.get("created_at", 0) < 300):
                    snapshot[tid] = {
                        "task_id": rec["task_id"],
                        "context_id": rec["context_id"],
                        "peer": rec.get("peer", ""),
                        "agent_slug": rec.get("agent_slug", ""),
                        "tenant": rec.get("tenant", ""),
                        "state": state,
                        "reply": rec.get("reply", ""),
                        "created_at": rec.get("created_at", 0),
                        "created_iso": rec.get("created_iso", ""),
                        "completed_at": rec.get("completed_at"),
                        "push_url": rec.get("push_url", ""),
                        "push_config_id": rec.get("push_config_id", ""),
                    }
        try:
            import fcntl
            _HAS_FCNTL = True
        except ImportError:
            _HAS_FCNTL = False
        try:
            import msvcrt
            _HAS_MSVCRT = True
        except ImportError:
            _HAS_MSVCRT = False

        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(".lock")
        lock_path.touch(exist_ok=True)
        fd = lock_path.open()
        try:
            if _HAS_FCNTL:
                fcntl.flock(fd, fcntl.LOCK_EX)
            elif _HAS_MSVCRT:
                msvcrt.locking(fd.fileno(), msvcrt.LK_NBLCK, 1)
            import tempfile, os
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(path.parent), suffix=".tmp"
            )
            try:
                try:
                    os.fchmod(tmp_fd, 0o600)
                except (AttributeError, OSError, NotImplementedError):
                    pass
                with os.fdopen(tmp_fd, "w") as f:
                    json.dump(snapshot, f, ensure_ascii=False, indent=2)
                    try:
                        f.flush()
                        os.fsync(f.fileno())
                    except Exception:
                        pass
                # Windows-safe permission hardening: chmod works on both platforms
                try:
                    os.chmod(tmp_path, 0o600)
                except OSError:
                    pass
                os.replace(tmp_path, str(path))
                try:
                    dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
                    try:
                        os.fsync(dir_fd)
                    finally:
                        os.close(dir_fd)
                except Exception:
                    pass
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        finally:
            fd.close()

    def restore(self, path: Path) -> int:
        """Load persisted task records from disk.  Returns count restored."""
        if not path.exists():
            return 0
        try:
            with open(path) as f:
                snapshot = json.load(f)
        except (json.JSONDecodeError, OSError):
            return 0
        if not isinstance(snapshot, dict):
            return 0
        count = 0
        with self._lock:
            for tid, rec in snapshot.items():
                if tid in self._tasks:
                    existing = self._tasks[tid]
                    if (existing["state"] not in TERMINAL_STATES
                            and rec.get("state") in TERMINAL_STATES):
                        existing["state"] = rec["state"]
                        existing["reply"] = rec.get("reply", "")
                        existing["completed_at"] = rec.get("completed_at")
                    continue
                restored = {
                    "task_id": rec.get("task_id", tid),
                    "context_id": rec.get("context_id", ""),
                    "peer": rec.get("peer", ""),
                    "agent_slug": rec.get("agent_slug", ""),
                    "tenant": rec.get("tenant", ""),
                    "state": rec.get("state", STATE_SUBMITTED),
                    "reply": rec.get("reply", ""),
                    "created_at": rec.get("created_at", 0),
                    "created_iso": rec.get("created_iso", ""),
                    "completed_at": rec.get("completed_at"),
                    "push_url": rec.get("push_url", ""),
                    "push_config_id": rec.get("push_config_id", ""),
                }
                self._tasks[tid] = restored
                count += 1
            self._trim_locked()
            # Fresh reload establishes authority; clear fail-closed flag.
            self._ledger_unavailable = False
            self._ledger_unavailable_reason = ""
        return count


class DurablePublishError(Exception):
    """Raised when a durable publish fails and caller must return structured JSON-RPC error."""

    def __init__(self, task_id: str, context_id: str, attempted_state: str, durable_state: str, dispatched: bool):
        super().__init__(f"durable publish failed for {task_id} attempted {attempted_state} durable {durable_state}")
        self.task_id = task_id
        self.context_id = context_id
        self.attempted_state = attempted_state
        self.durable_state = durable_state
        self.dispatched = dispatched


def durable_persistence_error(req_id: Any, task_id: str, context_id: str, attempted_state: str, durable_state: str, dispatched: bool) -> dict:
    """Build the structured JSON-RPC -32603 persistence failure response (section 5.5)."""
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {
            "code": -32603,
            "message": "A2A task state could not be durably published",
            "data": {
                "reason": "A2A_TASK_PERSISTENCE_FAILED",
                "taskId": task_id,
                "contextId": context_id,
                "attemptedState": attempted_state,
                "durableState": durable_state,
                "dispatched": dispatched,
                "safeToRetry": False,
            },
        },
    }


def _conv_dir() -> Path:
    try:
        from hermes_constants import get_hermes_home
        base = Path(get_hermes_home())
    except Exception:
        base = Path(os.path.expanduser("~/.hermes"))
    return base / "a2a_conversations"


def _safe_name(context_id: str) -> str:
    return "".join(c for c in (context_id or "default") if c.isalnum() or c in "-_") or "default"


def persist_message(context_id: str, role: str, text: str, task_id: str = "") -> None:
    """Append one message to the context's on-disk conversation log."""
    try:
        d = _conv_dir()
        d.mkdir(parents=True, exist_ok=True)
        rec = {"ts": time.time(), "role": role, "text": text, "task_id": task_id}
        with (d / f"{_safe_name(context_id)}.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def load_conversation(context_id: str, limit: int = 50) -> list[dict]:
    """Load the last *limit* messages for a context (empty list if none)."""
    path = _conv_dir() / f"{_safe_name(context_id)}.jsonl"
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        return []
    return out[-limit:]


def list_conversations() -> list[str]:
    """Return known context-ids that have persisted conversations."""
    d = _conv_dir()
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.jsonl"))

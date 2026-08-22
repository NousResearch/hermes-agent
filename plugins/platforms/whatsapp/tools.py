"""Fail-closed WhatsApp group administration tool.

The model-facing tool is only one half of the authorization boundary.  The
WhatsApp adapter exposes its toolset solely to exact configured DM principals,
and the loopback Node bridge independently authenticates the call, enforces the
participant allowlist, and persists idempotency state.
"""

from __future__ import annotations

import json
import re
from typing import Any

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_JID = re.compile(
    r"^(?:[0-9]{1,32})(?::[0-9]{1,3})?@(s\.whatsapp\.net|c\.us|lid)$",
    re.IGNORECASE,
)
_OPERATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{7,63}$")
_TOOLSET = "whatsapp_group_admin"


def _wenv(name: str, default: str = "") -> str:
    from agent.secret_scope import UnscopedSecretError, get_secret

    try:
        value = get_secret(name)
    except UnscopedSecretError:
        import os

        value = os.getenv(name)
    return value if value is not None else default


def _normalize_jid(value: object) -> str | None:
    raw = str(value or "").strip()
    if len(raw) > 96 or _CONTROL.search(raw):
        return None
    match = _JID.fullmatch(raw)
    if not match:
        return None
    local, domain = raw.rsplit("@", 1)
    local = local.split(":", 1)[0]
    domain = "s.whatsapp.net" if domain.lower() == "c.us" else domain.lower()
    return f"{local}@{domain}"


def _group_config_value(name: str) -> list[object]:
    try:
        from hermes_cli.config import load_config

        extra = (
            (((load_config() or {}).get("gateway") or {}).get("platforms") or {})
            .get("whatsapp", {})
            .get("extra", {})
        )
        value = extra.get(name, []) if isinstance(extra, dict) else []
    except Exception:
        return []
    if isinstance(value, str):
        return value.split(",")
    return value if isinstance(value, list) else []


def _jid_allowlist(name: str) -> set[str]:
    values = {
        normalized
        for item in _group_config_value(name)
        if (normalized := _normalize_jid(item)) is not None
    }
    return values


def _check_available() -> bool:
    token = _wenv("WHATSAPP_GROUP_CONTROL_TOKEN").strip()
    return bool(
        token
        and 32 <= len(token) <= 512
        and _jid_allowlist("group_admin_users")
        and _jid_allowlist("group_allowed_participants")
    )


def _bridge_port() -> int:
    try:
        from hermes_cli.config import load_config

        config = load_config() or {}
        raw = (
            (
                ((config.get("gateway") or {}).get("platforms") or {}).get("whatsapp")
                or {}
            )
            .get("extra", {})
            .get("bridge_port", 3000)
        )
        port = int(raw)
    except (TypeError, ValueError, AttributeError):
        port = 3000
    return port if 1 <= port <= 65535 else 3000


def _validated_request(args: dict[str, Any]) -> dict[str, Any] | None:
    subject = str(args.get("subject") or "").strip()
    confirmation = str(args.get("confirmed_subject") or "")
    operation_id = str(args.get("operation_id") or "").strip()
    raw_participants = args.get("participants")
    raw_confirmed_participants = args.get("confirmed_participants")
    if (
        not subject
        or len(subject) > 100
        or _CONTROL.search(subject)
        or confirmation != subject
        or not _OPERATION_ID.fullmatch(operation_id)
        or not isinstance(raw_participants, list)
        or not isinstance(raw_confirmed_participants, list)
        or not 1 <= len(raw_participants) <= 50
    ):
        return None

    participants: list[str] = []
    for value in raw_participants:
        normalized = _normalize_jid(value)
        if normalized is None or normalized in participants:
            return None
        participants.append(normalized)
    confirmed_participants = [
        _normalize_jid(value) for value in raw_confirmed_participants
    ]
    if confirmed_participants != participants:
        return None
    if not set(participants).issubset(_jid_allowlist("group_allowed_participants")):
        return None
    return {
        "subject": subject,
        "confirmedSubject": confirmation,
        "operationId": operation_id,
        "participants": participants,
        "confirmedParticipants": confirmed_participants,
    }


async def whatsapp_create_group(args: dict[str, Any], **_kwargs: Any) -> str:
    request = _validated_request(args)
    token = _wenv("WHATSAPP_GROUP_CONTROL_TOKEN").strip()
    if request is None or not _check_available():
        return json.dumps({
            "success": False,
            "error": "Group creation is unavailable or the confirmed request is invalid.",
        })

    try:
        import aiohttp

        timeout = aiohttp.ClientTimeout(total=25)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"http://127.0.0.1:{_bridge_port()}/groups/create",
                json=request,
                headers={"Authorization": f"Bearer {token}"},
            ) as response:
                try:
                    body = await response.json()
                except Exception:
                    body = {}
                if response.status not in {200, 201} or not isinstance(body, dict):
                    return json.dumps({
                        "success": False,
                        "error": "The WhatsApp bridge rejected the group request.",
                        "status": response.status,
                    })
                return json.dumps({
                    "success": body.get("success") is True,
                    "status": str(body.get("status") or "created")[:32],
                    "operation_id": request["operationId"],
                    "subject": request["subject"],
                    "group_id": str(body.get("groupId") or "")[:128] or None,
                })
    except Exception:
        return json.dumps({
            "success": False,
            "error": "The local WhatsApp group control service is unavailable.",
        })


_SCHEMA = {
    "name": "whatsapp_create_group",
    "description": (
        "Create one WhatsApp group after the operator explicitly confirms the exact "
        "subject and participants. Use a stable unique operation_id for retries. "
        "Never guess participant identifiers or retry an uncertain operation with a new ID."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "subject": {
                "type": "string",
                "description": "Exact WhatsApp group subject, 1-100 characters.",
            },
            "participants": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 50,
                "description": "Exact allowlisted WhatsApp user JIDs to add.",
            },
            "operation_id": {
                "type": "string",
                "description": "Stable 8-64 character idempotency key for this exact request.",
            },
            "confirmed_subject": {
                "type": "string",
                "description": "Repeat the exact subject only after the operator confirms it and the participant list.",
            },
            "confirmed_participants": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 50,
                "description": "Repeat the exact participant JID list only after the operator confirms it.",
            },
        },
        "required": [
            "subject",
            "participants",
            "operation_id",
            "confirmed_subject",
            "confirmed_participants",
        ],
        "additionalProperties": False,
    },
}


def register_tools(ctx) -> None:
    ctx.register_tool(
        name="whatsapp_create_group",
        toolset=_TOOLSET,
        schema=_SCHEMA,
        handler=whatsapp_create_group,
        check_fn=_check_available,
        requires_env=[
            "WHATSAPP_GROUP_CONTROL_TOKEN",
        ],
        is_async=True,
        description=_SCHEMA["description"],
        emoji="👥",
    )

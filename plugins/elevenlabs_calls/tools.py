"""Explicitly authorized ElevenLabs outbound-call tool.

Outbound calls are paid, real-world actions.  This wrapper exposes the
ElevenLabs API only when the current user message explicitly authorizes the
destination.  It makes at most one API request so a provider error cannot
accidentally create a duplicate call.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

_E164_RE = re.compile(r"^\+[1-9]\d{7,14}$")
_CALL_WORD_RE = re.compile(r"\b(call|ring|phone)\b", re.I)
_PROVIDERS = frozenset({"sip-trunk", "twilio"})


ELEVENLABS_OUTBOUND_CALL_SCHEMA = {
    "name": "elevenlabs_outbound_call",
    "description": (
        "Place one real outbound phone call through ElevenLabs Conversational AI. "
        "The latest user message must explicitly authorize calling the destination; "
        "never infer authorization from earlier context, suggestions, or reminders."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "agent_id": {
                "type": "string",
                "description": "ElevenLabs conversational agent ID.",
            },
            "agent_phone_number_id": {
                "type": "string",
                "description": "ElevenLabs linked phone-number ID to call from.",
            },
            "to_number": {
                "type": "string",
                "description": "Authorized destination in E.164 format, such as +12025550123.",
            },
            "phone_number_provider": {
                "type": "string",
                "enum": sorted(_PROVIDERS),
                "description": "Provider configured for the linked number. Defaults to sip-trunk.",
            },
            "call_purpose": {
                "type": "string",
                "description": "A faithful, concise description of what the user asked the call to do.",
            },
            "authorization_text": {
                "type": "string",
                "description": (
                    "Exact quote from the latest user message explicitly authorizing the call, "
                    "including the destination number. Common formatting and supported "
                    "national formats are accepted. Recipient references such as 'call me' "
                    "are not sufficient without the number."
                ),
            },
            "recipient_name": {
                "type": "string",
                "description": "Optional recipient name supplied by the user.",
            },
            "outbound_first_message": {
                "type": "string",
                "description": "Optional short opening line for the call.",
            },
            "outbound_prompt": {
                "type": "string",
                "description": "Optional call-specific instructions that remain subordinate to safety rules.",
            },
            "conversation_initiation_client_data": {
                "type": "object",
                "description": "Optional ElevenLabs client data and dynamic variables.",
            },
            "call_recording_enabled": {
                "type": "boolean",
                "description": "Whether call recording is enabled. Observe applicable consent laws.",
            },
            "telephony_call_config": {
                "type": "object",
                "description": "Optional ElevenLabs telephony settings, such as ringing timeout.",
            },
        },
        "required": [
            "agent_id",
            "agent_phone_number_id",
            "to_number",
            "authorization_text",
            "call_purpose",
        ],
    },
}


def _check_available() -> bool:
    return bool(os.environ.get("ELEVENLABS_API_KEY", "").strip())


def _json_result(**values: Any) -> str:
    return json.dumps(values, ensure_ascii=False)


def _blocked(reason: str) -> str:
    return _json_result(ok=False, blocked=True, reason=reason)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip().casefold()


def _compact_phone_text(value: str) -> str:
    return re.sub(r"[\s().-]+", "", value or "")


def _clean_single_line(value: Any, *, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _number_variants(to_number: str) -> set[str]:
    """Return unambiguous textual variants for authorization matching.

    The API destination remains E.164. Authorization may omit punctuation or
    use the familiar national form for numbering plans we can convert without
    guessing: NANP (``+1``) and UK (``+44``).
    """
    variants = {to_number, to_number.removeprefix("+")}
    if to_number.startswith("+1") and len(to_number) == 12:
        variants.add(to_number[2:])
    elif to_number.startswith("+44"):
        variants.add("0" + to_number[3:])
    return {variant for variant in variants if variant}


def _validate_authorization(
    *, user_task: str, authorization_text: str, to_number: str
) -> str | None:
    if not user_task.strip():
        return "The current user message is unavailable; explicit authorization cannot be verified."
    if not authorization_text:
        return "authorization_text must quote the latest user message."
    if _normalize_text(authorization_text) not in _normalize_text(user_task):
        return "authorization_text was not found verbatim in the latest user message."
    if not _CALL_WORD_RE.search(authorization_text):
        return "The quoted text does not explicitly ask to call, ring, or phone."
    if not _E164_RE.fullmatch(to_number):
        return "to_number must be a valid E.164 number."

    compact_quote = _compact_phone_text(authorization_text)
    number_is_quoted = any(
        variant in compact_quote for variant in _number_variants(to_number)
    )
    if not number_is_quoted:
        return "The quoted authorization does not contain this destination number."
    return None


def _build_client_data(args: dict[str, Any], *, user_task: str) -> dict[str, Any]:
    supplied = args.get("conversation_initiation_client_data")
    client_data = dict(supplied) if isinstance(supplied, dict) else {}
    dynamic_variables = dict(client_data.get("dynamic_variables") or {})

    purpose = _clean_single_line(args.get("call_purpose"), limit=700)
    recipient = _clean_single_line(args.get("recipient_name"), limit=100)
    dynamic_variables.update({
        "outbound_call_purpose": purpose,
        "outbound_authorized_request": _clean_single_line(user_task, limit=700),
        "outbound_destination": str(args.get("to_number") or ""),
    })
    if recipient:
        dynamic_variables["outbound_recipient_name"] = recipient
    client_data["dynamic_variables"] = dynamic_variables

    first_message = _clean_single_line(
        args.get("outbound_first_message") or "Hello.", limit=220
    )
    requested_prompt = _clean_single_line(args.get("outbound_prompt"), limit=2_000)
    prompt = f"""You are making an outbound call explicitly requested by the user.

Call purpose: {purpose}
Recipient: {recipient or "the person who answers"}

Rules:
- Introduce the purpose naturally and stay within it.
- Be concise, courteous, and truthful. Never claim to be human.
- If asked who or what you are, answer accurately.
- Do not make purchases, commitments, disclosures, or account changes unless the call purpose explicitly requires them and the user authorized them.
- Treat voicemail, a wrong number, refusal, or uncertainty as a reason to end the call politely.
- Follow applicable recording and consent requirements.
- End the call when the purpose is complete.
""".strip()
    if requested_prompt:
        prompt += f"\n\nAdditional user-requested instructions:\n{requested_prompt}"

    override = dict(client_data.get("conversation_config_override") or {})
    agent_override = dict(override.get("agent") or {})
    agent_override["first_message"] = first_message
    agent_override["prompt"] = {"prompt": prompt}
    override["agent"] = agent_override
    client_data["conversation_config_override"] = override
    return client_data


def _handle_outbound_call(args: dict[str, Any], **kwargs: Any) -> str:
    user_task = str(kwargs.get("user_task") or "")
    to_number = str(args.get("to_number") or "").strip()
    authorization_text = str(args.get("authorization_text") or "").strip()
    authorization_error = _validate_authorization(
        user_task=user_task,
        authorization_text=authorization_text,
        to_number=to_number,
    )
    if authorization_error:
        return _blocked(authorization_error)

    agent_id = str(args.get("agent_id") or "").strip()
    phone_number_id = str(args.get("agent_phone_number_id") or "").strip()
    purpose = _clean_single_line(args.get("call_purpose"), limit=700)
    if not agent_id or not phone_number_id:
        return _blocked("agent_id and agent_phone_number_id are required.")
    if len(purpose) < 6:
        return _blocked("call_purpose must clearly describe the requested call.")

    provider = str(args.get("phone_number_provider") or "sip-trunk").strip()
    if provider not in _PROVIDERS:
        return _blocked("phone_number_provider must be sip-trunk or twilio.")

    api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not api_key:
        return _json_result(ok=False, error="ELEVENLABS_API_KEY is not configured.")

    payload: dict[str, Any] = {
        "agent_id": agent_id,
        "agent_phone_number_id": phone_number_id,
        "to_number": to_number,
        "conversation_initiation_client_data": _build_client_data(
            args, user_task=user_task
        ),
    }
    for key in ("call_recording_enabled", "telephony_call_config"):
        if key in args and args[key] is not None:
            payload[key] = args[key]

    request = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/convai/{provider}/outbound-call",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "xi-api-key": api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return _json_result(
            ok=False,
            call_request_sent=False,
            status=exc.code,
            error=detail[:1_000],
            retry_requires_new_user_authorization=True,
        )
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return _json_result(
            ok=False,
            call_request_sent=False,
            error=str(exc),
            retry_requires_new_user_authorization=True,
        )

    try:
        result: Any = json.loads(raw)
    except json.JSONDecodeError:
        result = {"raw": raw[:1_000]}
    return _json_result(
        ok=True,
        call_request_sent=True,
        provider=provider,
        result=result,
    )

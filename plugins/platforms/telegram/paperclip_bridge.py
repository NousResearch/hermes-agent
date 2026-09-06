import asyncio
import hashlib
import hmac
import json
import os
import re
import time
from typing import Any
from urllib import error, request
from urllib.parse import urlsplit

CALLBACK_PREFIX = "pcb:"
BRIDGE_COMMANDS = {"log", "paperclip_plan"}
MAX_CALLBACK_BYTES = 64
MAX_BRIDGE_RESPONSE_BYTES = 262144
SAFE_BRIDGE_TARGET = re.compile(r"^(?:ghplan_[0-9a-f]{24}|ghlog_[0-9a-f]{20})$")
SAFE_PLAN_TOKEN = re.compile(r"^ghplan_[0-9a-f]{24}$")
NATURAL_PLAN_PATTERN = re.compile(
    r"^\s*review\s+paperclip\s+plan\s+(?P<token>ghplan_[A-Za-z0-9_.-]+)\s*$",
    re.IGNORECASE,
)
NATURAL_JOB_PATTERN = re.compile(
    r"^\s*(?:(?:please|can you|could you|would you)\s+)?"
    r"(?:log (?:a )?(?:job|task) to|create (?:a )?(?:job|task) for)\s+"
    r"(?P<task>\S(?:.*\S)?)\s*$",
    re.IGNORECASE,
)
NATURAL_LOG_PATTERN = re.compile(
    r"^\s*(?:(?:please|can you|could you|would you)\s+)?"
    r"(?:log this(?: in paperclip)?|add (?:this|a task) to paperclip|create (?:a )?paperclip task|put this (?:in|into) paperclip)"
    r"\s*(?::|—|-)\s*(?P<task>\S(?:.*\S)?)\s*$",
    re.IGNORECASE,
)


def _extra(adapter, key: str, default: Any = None) -> Any:
    secret_env = {
        "paperclip_bridge_bearer_token": "HERMES_PAPERCLIP_BRIDGE_BEARER_TOKEN",
        "paperclip_bridge_signing_secret": "HERMES_PAPERCLIP_BRIDGE_SIGNING_SECRET",
    }
    env_name = secret_env.get(key)
    if env_name and os.getenv(env_name):
        return os.environ[env_name]
    config = getattr(adapter, "config", None)
    extra = getattr(config, "extra", None) or {}
    return extra.get(key, default)

def _validated_bridge_base_url(adapter) -> str:
    value = str(_extra(adapter, "paperclip_bridge_url", "") or "").strip().rstrip("/")
    parsed = urlsplit(value)
    loopback = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    secure = parsed.scheme == "https" or (parsed.scheme == "http" and loopback)
    if not secure or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
        return ""
    return value


def is_enabled(adapter) -> bool:
    return bool(
        _extra(adapter, "paperclip_bridge_enabled", False)
        and _validated_bridge_base_url(adapter)
        and _extra(adapter, "paperclip_bridge_bearer_token")
        and _extra(adapter, "paperclip_bridge_signing_secret")
    )


def build_headers(
    body: bytes,
    *,
    method: str,
    path: str,
    token: str,
    signing_secret: str,
    idempotency_key: str,
) -> dict[str, str]:
    timestamp = str(int(time.time()))
    canonical_method = method.upper()
    signature = "sha256=" + hmac.new(
        signing_secret.encode("utf-8"),
        b".".join(
            (
                timestamp.encode("utf-8"),
                canonical_method.encode("ascii"),
                path.encode("utf-8"),
                body,
            )
        ),
        hashlib.sha256,
    ).hexdigest()
    return {
        "Authorization": f"Bearer {token}",
        "X-Hermes-Timestamp": timestamp,
        "X-Hermes-Signature": signature,
        "X-Idempotency-Key": idempotency_key,
        "Content-Type": "application/json",
    }


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def post_json(url: str, body: dict[str, Any], headers: dict[str, str], timeout: int = 15) -> tuple[int, dict[str, Any]]:
    raw = json.dumps(body, separators=(",", ":")).encode("utf-8")
    req = request.Request(url, data=raw, headers=headers, method="POST")
    opener = request.build_opener(_NoRedirect)
    try:
        with opener.open(req, timeout=timeout) as response:
            raw_response = response.read(MAX_BRIDGE_RESPONSE_BYTES + 1)
            if len(raw_response) > MAX_BRIDGE_RESPONSE_BYTES:
                raise ValueError("Bridge response exceeds size limit")
            data = raw_response.decode("utf-8")
            payload = json.loads(data) if data else {}
            if not isinstance(payload, dict):
                raise ValueError("Bridge response must be a JSON object")
            return response.status, payload
    except error.HTTPError as exc:
        return exc.code, {"error": "Bridge request failed."}


def get_json(url: str, headers: dict[str, str], timeout: int = 15) -> tuple[int, dict[str, Any]]:
    req = request.Request(url, data=b"", headers=headers, method="GET")
    opener = request.build_opener(_NoRedirect)
    try:
        with opener.open(req, timeout=timeout) as response:
            raw_response = response.read(MAX_BRIDGE_RESPONSE_BYTES + 1)
            if len(raw_response) > MAX_BRIDGE_RESPONSE_BYTES:
                raise ValueError("Bridge response exceeds size limit")
            data = raw_response.decode("utf-8")
            payload = json.loads(data) if data else {}
            if not isinstance(payload, dict):
                raise ValueError("Bridge response must be a JSON object")
            return response.status, payload
    except error.HTTPError as exc:
        return exc.code, {"error": "Bridge request failed."}


def _normalize_command(event) -> tuple[str | None, str]:
    text = str(getattr(event, "text", "") or "")
    if not text.lstrip().startswith("/"):
        natural_plan = NATURAL_PLAN_PATTERN.match(text)
        if natural_plan:
            return "paperclip_plan", natural_plan.group("token")
        for pattern in (NATURAL_JOB_PATTERN, NATURAL_LOG_PATTERN):
            natural_log = pattern.match(text)
            if natural_log:
                return "log", natural_log.group("task")
        return None, ""
    command = event.get_command()
    if not command:
        return None, ""
    return command, event.get_command_args().strip()


def _button_callback(action: str, target: str) -> str | None:
    if (
        action not in {"approve", "reject"}
        or not SAFE_BRIDGE_TARGET.fullmatch(target)
    ):
        return None
    callback_data = f"{CALLBACK_PREFIX}{action}:{target}"
    if len(callback_data.encode("utf-8")) > MAX_CALLBACK_BYTES:
        return None
    return callback_data


def _thread_kwargs(adapter, event) -> dict[str, Any]:
    source = getattr(event, "source", None)
    metadata = getattr(event, "metadata", None)
    chat_id = str(getattr(source, "chat_id", ""))
    thread_id = getattr(source, "thread_id", None)
    try:
        return adapter._thread_kwargs_for_send(chat_id, thread_id, metadata, reply_to_mode=getattr(adapter, "_reply_to_mode", None))
    except Exception:
        return {}


def _plain_actor_name(event) -> str:
    return getattr(event, "user_name", None) or getattr(event, "user_id", None) or "User"


def _configured_company_id(adapter, event) -> str:
    return str(_extra_source(event, "paperclip_company_id") or _extra(adapter, "paperclip_bridge_company_id", "") or "")


def _configured_project_id(adapter, event) -> str:
    return str(_extra_source(event, "paperclip_project_id") or _extra(adapter, "paperclip_bridge_project_id", "") or "")


def _command_request_body(adapter, event, command: str, args: str) -> dict[str, Any]:
    source = event.source
    project_id = _configured_project_id(adapter, event)
    return {
        "source": "telegram",
        "command": command,
        "request_id": f"telegram:{source.chat_id}:{event.message_id}:/{command}",
        "occurred_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "company_id": _configured_company_id(adapter, event),
        "project_id": project_id,
        "chat": {
            "id": str(source.chat_id),
            "type": str(source.chat_type or ""),
            "thread_id": str(source.thread_id or ""),
        },
        "message": {
            "message_id": str(event.message_id or ""),
            "text": event.text,
            "command_text": args,
        },
        "actor": {
            "telegram_user_id": str(event.user_id or ""),
            "username": str(event.user_name or ""),
            "display_name": _plain_actor_name(event),
        },
        "routing": {
            "linked_chat_id": str(source.chat_id),
            "mapped_topic_project_id": project_id,
        },
    }


def _approval_request_body(query, target: str, decision: str, *, query_chat_id: Any, query_chat_type: Any, query_thread_id: Any) -> dict[str, Any]:
    return {
        "source": "telegram",
        "command": "approve",
        "request_id": f"telegram:{query_chat_id}:{getattr(getattr(query, 'message', None), 'message_id', '')}:/approve:{decision}:{target}",
        "occurred_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target": target,
        "decision": decision,
        "note": "",
        "chat": {"id": str(query_chat_id or ""), "type": str(query_chat_type or ""), "thread_id": str(query_thread_id or "")},
        "message": {
            "message_id": str(getattr(getattr(query, 'message', None), 'message_id', '') or ''),
            "callback_query_id": str(getattr(query, 'id', '') or ''),
            "origin_message_id": str(getattr(getattr(query, 'message', None), 'message_id', '') or ''),
        },
        "actor": {
            "telegram_user_id": str(getattr(getattr(query, 'from_user', None), 'id', '') or ''),
            "username": str(getattr(getattr(query, 'from_user', None), 'username', '') or ''),
            "display_name": str(getattr(getattr(query, 'from_user', None), 'first_name', '') or 'User'),
        },
    }


def _extra_source(event, key: str) -> Any:
    metadata = getattr(event, "metadata", {}) or {}
    return metadata.get(key)


def _valid_reply_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    reply = payload.get("reply")
    if not isinstance(reply, dict) or not isinstance(reply.get("text"), str) or not reply["text"].strip():
        return False
    buttons = reply.get("buttons", [])
    if not isinstance(buttons, list):
        return False
    for item in buttons:
        if not isinstance(item, dict):
            return False
        if _button_callback(str(item.get("action", "")), str(item.get("target", ""))) is None:
            return False
    return True


def _plan_reply_payload(payload: Any, token: str) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    if payload.get("token") != token:
        return None
    status = payload.get("status")
    reply = payload.get("reply")
    if not isinstance(status, str) or not isinstance(reply, dict):
        return None
    text = reply.get("text")
    buttons = reply.get("buttons", [])
    if not isinstance(text, str) or not text.strip() or not isinstance(buttons, list):
        return None
    if status == "pending_approval":
        expected = [
            {"text": "Approve", "action": "approve", "target": token},
            {"text": "Reject", "action": "reject", "target": token},
        ]
        if buttons != expected:
            return None
    elif buttons:
        return None
    return {"reply": {"text": text, "buttons": buttons}}


async def _handle_plan_request(adapter, event, token: str) -> None:
    if not SAFE_PLAN_TOKEN.fullmatch(token):
        await _deliver_bridge_reply(adapter, event, {"reply": {"text": "Invalid Paperclip plan token."}})
        return
    path = f"/plans/{token}"
    headers = build_headers(
        b"",
        method="GET",
        path=path,
        token=_extra(adapter, "paperclip_bridge_bearer_token"),
        signing_secret=_extra(adapter, "paperclip_bridge_signing_secret"),
        idempotency_key=f"telegram:plan:{token}",
    )
    try:
        status, payload = await asyncio.to_thread(
            get_json,
            _validated_bridge_base_url(adapter) + path,
            headers,
        )
        reply_payload = _plan_reply_payload(payload, token)
        if not 200 <= status < 300 or reply_payload is None:
            raise ValueError("Bridge returned an invalid plan response")
        await _deliver_bridge_reply(adapter, event, reply_payload)
    except Exception:
        await _deliver_bridge_reply(
            adapter,
            event,
            {"reply": {"text": "I could not retrieve that Paperclip plan. Please retry."}},
        )


async def maybe_handle_command(adapter, event) -> bool:
    if not is_enabled(adapter):
        return False
    command, args = _normalize_command(event)
    if command not in BRIDGE_COMMANDS:
        return False
    if command == "paperclip_plan":
        await _handle_plan_request(adapter, event, args)
        return True
    if not args:
        await _deliver_bridge_reply(adapter, event, {"reply": {"text": "Tell me what you want logged after the colon."}})
        return True

    body = _command_request_body(adapter, event, command, args)
    raw = json.dumps(body, separators=(",", ":")).encode("utf-8")
    headers = build_headers(
        raw,
        method="POST",
        path="/log",
        token=_extra(adapter, "paperclip_bridge_bearer_token"),
        signing_secret=_extra(adapter, "paperclip_bridge_signing_secret"),
        idempotency_key=f"telegram:{event.source.chat_id}:{event.message_id}:/{command}",
    )
    try:
        status, payload = await asyncio.to_thread(
            post_json,
            _validated_bridge_base_url(adapter) + "/log",
            body,
            headers,
        )
        if not 200 <= status < 300 or not _valid_reply_payload(payload):
            raise ValueError("Bridge returned an invalid response")
        await _deliver_bridge_reply(adapter, event, payload)
    except Exception:
        await _deliver_bridge_reply(
            adapter,
            event,
            {"reply": {"text": "I could not confirm that log request. Please retry; duplicate issues are prevented."}},
        )
    return True


async def maybe_handle_callback(adapter, query, data: str, *, query_chat_id: Any, query_chat_type: Any, query_thread_id: Any, query_user_name: Any) -> bool:
    if not is_enabled(adapter) or not data.startswith(CALLBACK_PREFIX):
        return False
    caller_id = str(getattr(getattr(query, "from_user", None), "id", ""))
    if hasattr(adapter, "_is_callback_user_authorized") and not adapter._is_callback_user_authorized(
        caller_id,
        chat_id=query_chat_id,
        chat_type=str(query_chat_type) if query_chat_type is not None else None,
        thread_id=str(query_thread_id) if query_thread_id is not None else None,
        user_name=query_user_name,
    ):
        await query.answer(text="⛔ You are not authorized to approve this action.")
        return True

    if len(data.encode("utf-8")) > MAX_CALLBACK_BYTES:
        await query.answer(text="Invalid Paperclip action.")
        return True
    rest = data[len(CALLBACK_PREFIX):]
    action, separator, target = rest.partition(":")
    if (
        separator != ":"
        or action not in {"approve", "reject"}
        or not SAFE_BRIDGE_TARGET.fullmatch(target)
    ):
        await query.answer(text="Invalid Paperclip action.")
        return True

    decision = action
    body = _approval_request_body(
        query,
        target,
        decision,
        query_chat_id=query_chat_id,
        query_chat_type=query_chat_type,
        query_thread_id=query_thread_id,
    )
    raw = json.dumps(body, separators=(",", ":")).encode("utf-8")
    origin_message_id = str(getattr(getattr(query, "message", None), "message_id", "") or "")
    headers = build_headers(
        raw,
        method="POST",
        path="/approve",
        token=_extra(adapter, "paperclip_bridge_bearer_token"),
        signing_secret=_extra(adapter, "paperclip_bridge_signing_secret"),
        idempotency_key=f"telegram:{query_chat_id}:{origin_message_id}:/approve:{decision}:{target}",
    )

    status = 0
    payload: dict[str, Any] = {}
    try:
        status, payload = await asyncio.to_thread(
            post_json,
            _validated_bridge_base_url(adapter) + "/approve",
            body,
            headers,
        )
    except Exception:
        pass

    if 200 <= status < 300 and _valid_reply_payload(payload):
        reply = payload["reply"]
        text = reply["text"]
        await query.answer(text="Approved" if decision == "approve" else "Rejected")
    else:
        reply = {}
        text = "I could not confirm that Paperclip action. Please retry; duplicates are prevented."
        await query.answer(text="Request failed")

    if reply.get("edit_origin"):
        try:
            await query.edit_message_text(text=text, reply_markup=None)
        except Exception:
            pass
    else:
        try:
            await adapter._send_message_with_thread_fallback(
                chat_id=int(query_chat_id),
                text=text,
                **adapter._thread_kwargs_for_send(
                    str(query_chat_id),
                    str(query_thread_id) if query_thread_id is not None else None,
                    {"thread_id": str(query_thread_id)} if query_thread_id is not None else None,
                    reply_to_mode=getattr(adapter, "_reply_to_mode", None),
                ),
                **adapter._link_preview_kwargs(),
            )
        except Exception:
            pass
    return True


async def _deliver_bridge_reply(adapter, event, payload: dict[str, Any]) -> None:
    if not _valid_reply_payload(payload):
        payload = {"reply": {"text": "I could not confirm that Paperclip request. Please retry."}}
    reply = payload["reply"]
    text = reply["text"]
    buttons = reply.get("buttons", [])
    reply_markup = None
    if buttons:
        from plugins.platforms.telegram.adapter import InlineKeyboardButton, InlineKeyboardMarkup, normalize_telegram_chat_id

        keyboard_buttons = []
        for item in buttons:
            callback_data = _button_callback(str(item["action"]), str(item["target"]))
            if callback_data is not None:
                keyboard_buttons.append(
                    InlineKeyboardButton(str(item.get("text") or "Action"), callback_data=callback_data)
                )
        if keyboard_buttons:
            reply_markup = InlineKeyboardMarkup([keyboard_buttons])
    else:
        from plugins.platforms.telegram.adapter import normalize_telegram_chat_id
    await adapter._send_message_with_thread_fallback(
        chat_id=normalize_telegram_chat_id(str(event.source.chat_id)),
        text=text,
        reply_markup=reply_markup,
        **_thread_kwargs(adapter, event),
        **adapter._link_preview_kwargs(),
    )


async def dispatch_text_event(adapter, event) -> None:
    """Route explicit Paperclip intake before ordinary agent dispatch."""
    command, _ = _normalize_command(event)
    recognized = is_enabled(adapter) and command in BRIDGE_COMMANDS
    if not recognized:
        await adapter.handle_message(event)
        return
    try:
        await maybe_handle_command(adapter, event)
    except Exception:
        try:
            await _deliver_bridge_reply(
                adapter,
                event,
                {"reply": {"text": "I could not confirm that log request. Please retry."}},
            )
        except Exception:
            pass

"""Push notification registration and relay intent helpers.

Hermes never talks to APNs directly.  This module stores profile-local device
registrations and posts authenticated notification intents to a user-configured
relay service that owns APNs credentials.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib import request
from urllib.parse import urlparse

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

CONTRACT_VERSION = 1
DEFAULT_RELAY_TOKEN_ENV = "HERMES_PUSH_RELAY_TOKEN"
DEFAULT_STORE_NAME = "push_devices.json"
DEFAULT_EVENTS = (
    "approval.request",
    "clarify.request",
    "message.complete",
    "subagent.complete",
    "background.complete",
    "preview.restart.complete",
)
_EVENTS = frozenset(DEFAULT_EVENTS)
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
_ENV_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CONTROL_RE = re.compile(r"[\r\n\x00]")
_STORE_LOCK = threading.RLock()


class PushRegistrationError(ValueError):
    """Raised when a push registration request is malformed."""


@dataclass(frozen=True)
class PushConfig:
    enabled: bool
    relay_url: str
    relay_token_env: str
    relay_token: str
    events: tuple[str, ...]
    redact_body: bool
    timeout_seconds: float
    store_path: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _coerce_timeout(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 3.0
    return min(max(parsed, 0.1), 30.0)


def _section(raw_config: dict | None) -> dict:
    if not isinstance(raw_config, dict):
        return {}
    if isinstance(raw_config.get("push_notifications"), dict):
        return raw_config["push_notifications"]
    gateway = raw_config.get("gateway")
    if isinstance(gateway, dict) and isinstance(
        gateway.get("push_notifications"), dict
    ):
        return gateway["push_notifications"]
    return {}


def _store_path(raw: Any) -> Path:
    if isinstance(raw, str) and raw.strip():
        path = Path(os.path.expandvars(os.path.expanduser(raw.strip())))
        if not path.is_absolute():
            path = get_hermes_home() / path
        return path
    return get_hermes_home() / DEFAULT_STORE_NAME


def _events(raw: Any, default: tuple[str, ...] = DEFAULT_EVENTS) -> tuple[str, ...]:
    if raw is None:
        return default
    if not isinstance(raw, (list, tuple, set)):
        return default
    out: list[str] = []
    for item in raw:
        event = str(item or "").strip()
        if event in _EVENTS and event not in out:
            out.append(event)
    return tuple(out) if out else default


def load_push_config(raw_config: dict | None = None) -> PushConfig:
    cfg = _section(raw_config)
    token_env = str(cfg.get("relay_token_env") or DEFAULT_RELAY_TOKEN_ENV).strip()
    if not _ENV_RE.fullmatch(token_env):
        token_env = DEFAULT_RELAY_TOKEN_ENV
    relay_url = str(cfg.get("relay_url") or "").strip()
    return PushConfig(
        enabled=_coerce_bool(cfg.get("enabled"), False),
        relay_url=relay_url,
        relay_token_env=token_env,
        relay_token=os.environ.get(token_env, "").strip(),
        events=_events(cfg.get("events")),
        redact_body=_coerce_bool(cfg.get("redact_body"), True),
        timeout_seconds=_coerce_timeout(cfg.get("timeout_seconds")),
        store_path=_store_path(cfg.get("registration_store")),
    )


def relay_configured(config: PushConfig) -> bool:
    if not config.enabled or not config.relay_url or not config.relay_token:
        return False
    parsed = urlparse(config.relay_url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _load_store(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "devices": {}}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f) or {}
    except Exception as exc:
        logger.warning("Failed to load push registration store %s: %s", path, exc)
        return {"version": 1, "devices": {}}
    devices = data.get("devices")
    if not isinstance(devices, dict):
        devices = {}
    return {"version": 1, "devices": devices}


def _save_store(path: Path, store: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _text(value: Any, name: str, *, max_len: int, required: bool = False) -> str:
    if value is None:
        if required:
            raise PushRegistrationError(f"{name} is required")
        return ""
    text = str(value).strip()
    if not text:
        if required:
            raise PushRegistrationError(f"{name} is required")
        return ""
    if len(text) > max_len:
        raise PushRegistrationError(f"{name} is too long")
    if _CONTROL_RE.search(text):
        raise PushRegistrationError(f"{name} contains control characters")
    return text


def _device_id(params: dict, device_token: str, endpoint_id: str) -> str:
    raw = str(params.get("device_id") or "").strip()
    if raw:
        if not _ID_RE.fullmatch(raw):
            raise PushRegistrationError(
                "device_id may only contain letters, digits, _, ., :, or -"
            )
        return raw
    basis = endpoint_id or device_token
    digest = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:32]
    return f"device-{digest}"


def public_registration(record: dict) -> dict:
    return {
        "device_id": record.get("device_id", ""),
        "platform": record.get("platform", "apns"),
        "endpoint_id": record.get("endpoint_id") or "",
        "session_key": record.get("session_key", ""),
        "last_live_session_id": record.get("last_live_session_id") or "",
        "events": list(record.get("events") or []),
        "redact_body": bool(record.get("redact_body", True)),
        "has_device_token": bool(record.get("device_token")),
        "created_at": record.get("created_at", ""),
        "updated_at": record.get("updated_at", ""),
    }


def register_device(
    params: dict,
    *,
    session_key: str,
    live_session_id: str,
    raw_config: dict | None = None,
) -> dict:
    if not isinstance(params, dict):
        raise PushRegistrationError("params must be an object")
    session_key = _text(session_key, "session_key", max_len=256, required=True)
    live_session_id = _text(live_session_id, "session_id", max_len=128, required=True)
    device_token = _text(params.get("device_token"), "device_token", max_len=4096)
    endpoint_id = _text(
        params.get("endpoint_id") or params.get("device_endpoint"),
        "endpoint_id",
        max_len=2048,
    )
    if not device_token and not endpoint_id:
        raise PushRegistrationError("device_token or endpoint_id is required")

    config = load_push_config(raw_config)
    platform = _text(
        params.get("platform") or "apns",
        "platform",
        max_len=64,
        required=True,
    )
    device_id = _device_id(params, device_token, endpoint_id)
    now = _utc_now()

    with _STORE_LOCK:
        store = _load_store(config.store_path)
        devices = store.setdefault("devices", {})
        existing = devices.get(device_id) if isinstance(devices.get(device_id), dict) else {}
        record = {
            "device_id": device_id,
            "platform": platform,
            "device_token": device_token,
            "endpoint_id": endpoint_id,
            "session_key": session_key,
            "last_live_session_id": live_session_id,
            "events": list(_events(params.get("events"), config.events)),
            "redact_body": _coerce_bool(params.get("redact_body"), config.redact_body),
            "created_at": existing.get("created_at") or now,
            "updated_at": now,
        }
        devices[device_id] = record
        _save_store(config.store_path, store)

    return {
        "registration": public_registration(record),
        "relay": {
            "enabled": config.enabled,
            "configured": relay_configured(config),
            "url_configured": bool(config.relay_url),
            "token_env": config.relay_token_env,
            "token_configured": bool(config.relay_token),
        },
    }


def unregister_device(
    device_id: str,
    *,
    session_key: str,
    raw_config: dict | None = None,
) -> dict:
    config = load_push_config(raw_config)
    normalized = _text(device_id, "device_id", max_len=160, required=True)
    if not _ID_RE.fullmatch(normalized):
        raise PushRegistrationError("invalid device_id")
    session_key = _text(session_key, "session_key", max_len=256, required=True)
    removed = False
    with _STORE_LOCK:
        store = _load_store(config.store_path)
        devices = store.setdefault("devices", {})
        record = devices.get(normalized)
        if isinstance(record, dict) and record.get("session_key") == session_key:
            devices.pop(normalized, None)
            removed = True
            _save_store(config.store_path, store)
    return {"removed": removed, "device_id": normalized}


def list_registrations(*, session_key: str, raw_config: dict | None = None) -> list[dict]:
    config = load_push_config(raw_config)
    session_key = _text(session_key, "session_key", max_len=256, required=True)
    with _STORE_LOCK:
        store = _load_store(config.store_path)
        devices = store.get("devices") or {}
        return [
            public_registration(record)
            for record in devices.values()
            if isinstance(record, dict) and record.get("session_key") == session_key
        ]


def _trim(text: str, max_len: int = 180) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 1].rstrip()}…"


def _display_text(event: str, payload: dict, redact_body: bool) -> tuple[str, str]:
    titles = {
        "approval.request": "Hermes approval needed",
        "clarify.request": "Hermes needs clarification",
        "message.complete": "Hermes response ready",
        "subagent.complete": "Hermes subagent finished",
        "background.complete": "Hermes background task finished",
        "preview.restart.complete": "Hermes preview task finished",
    }
    generic = {
        "approval.request": "Review the pending approval in SeaClaw.",
        "clarify.request": "Answer the pending question in SeaClaw.",
        "message.complete": "A response is ready in SeaClaw.",
        "subagent.complete": "A delegated task finished in SeaClaw.",
        "background.complete": "A background task finished in SeaClaw.",
        "preview.restart.complete": "A preview/background task finished in SeaClaw.",
    }
    title = titles.get(event, "Hermes notification")
    if redact_body:
        return title, generic.get(event, "Open SeaClaw for details.")
    candidates = (
        payload.get("description"),
        payload.get("question"),
        payload.get("summary"),
        payload.get("text"),
        payload.get("rendered"),
    )
    body = next((str(c) for c in candidates if c), "")
    return title, _trim(body) or generic.get(event, "Open SeaClaw for details.")


def _action_context(event: str, live_session_id: str, payload: dict) -> dict | None:
    if event == "approval.request":
        allow_permanent = _coerce_bool(payload.get("allow_permanent"), True)
        choices = ["once", "session", "deny"]
        if allow_permanent:
            choices.insert(2, "always")
        return {
            "kind": "approval",
            "rpc_method": "approval.respond",
            "params_base": {"session_id": live_session_id},
            "choice_param": "choice",
            "choices": choices,
            "fifo_session_keyed": True,
            "request_id": None,
        }
    if event == "clarify.request":
        request_id = str(payload.get("request_id") or "").strip()
        return {
            "kind": "clarify",
            "rpc_method": "clarify.respond",
            "params_base": {"request_id": request_id},
            "answer_param": "answer",
            "choices": list(payload.get("choices") or []),
            "request_id": request_id,
        }
    return None


def build_intent(
    event: str,
    *,
    registration: dict,
    session_key: str,
    live_session_id: str,
    payload: dict | None = None,
) -> dict:
    if event not in _EVENTS:
        raise PushRegistrationError(f"unsupported push event: {event}")
    payload = dict(payload or {})
    redact_body = bool(registration.get("redact_body", True))
    title, body = _display_text(event, payload, redact_body)
    row_id = (
        payload.get("row_id")
        or payload.get("message_id")
        or payload.get("request_id")
        or payload.get("task_id")
    )
    intent = {
        "contract_version": CONTRACT_VERSION,
        "intent_id": uuid.uuid4().hex,
        "created_at": _utc_now(),
        "target": {
            "device_id": registration.get("device_id", ""),
            "platform": registration.get("platform", "apns"),
        },
        "session": {
            "stored_session_id": session_key,
            "session_key": session_key,
            "live_session_id": live_session_id or "",
        },
        "event": {
            "category": event,
            "title": title,
            "body": body,
            "redacted": redact_body,
            "row_id": str(row_id) if row_id else None,
            "message_id": (
                str(payload.get("message_id")) if payload.get("message_id") else None
            ),
        },
    }
    if endpoint_id := str(registration.get("endpoint_id") or "").strip():
        intent["target"]["endpoint_id"] = endpoint_id
    elif device_token := str(registration.get("device_token") or "").strip():
        intent["target"]["device_token"] = device_token
    if action := _action_context(event, live_session_id, payload):
        intent["action_context"] = action
    return intent


def _post_intent(config: PushConfig, intent: dict) -> None:
    body = json.dumps(intent, separators=(",", ":"), sort_keys=True).encode("utf-8")
    req = request.Request(
        config.relay_url,
        data=body,
        headers={
            "Authorization": f"Bearer {config.relay_token}",
            "Content-Type": "application/json",
            "User-Agent": "Hermes-Push-Intent/1",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=config.timeout_seconds) as resp:
            status = getattr(resp, "status", 200)
            if status >= 300:
                logger.warning("Push relay returned HTTP %s", status)
    except Exception as exc:
        logger.warning("Push relay delivery failed: %s", exc)


Sender = Callable[[PushConfig, dict], None]


def notify_event(
    event: str,
    *,
    session_key: str,
    live_session_id: str,
    payload: dict | None = None,
    raw_config: dict | None = None,
    sender: Sender | None = None,
    background: bool = True,
) -> dict:
    if event not in _EVENTS:
        return {"queued": 0, "reason": "unsupported_event"}
    config = load_push_config(raw_config)
    if not relay_configured(config):
        return {"queued": 0, "reason": "relay_not_configured"}
    with _STORE_LOCK:
        store = _load_store(config.store_path)
        devices = store.get("devices") or {}
        records = [
            dict(record)
            for record in devices.values()
            if isinstance(record, dict)
            and record.get("session_key") == session_key
            and event in set(record.get("events") or [])
        ]
    queued = 0
    post = sender or _post_intent
    for record in records:
        intent = build_intent(
            event,
            registration=record,
            session_key=session_key,
            live_session_id=live_session_id,
            payload=payload,
        )
        if background and sender is None:
            threading.Thread(target=post, args=(config, intent), daemon=True).start()
        else:
            post(config, intent)
        queued += 1
    return {"queued": queued}

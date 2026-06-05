"""Durable Telegram approval requests for Kanban mutation gates.

This module is intentionally small and file-backed so the gateway callback path
and Kanban worker/CLI paths share one idempotent implementation without adding
new SQLite schema.  Request files live under
``<HERMES_HOME>/approval-bridge/requests`` and short Telegram callback mappings
live under ``<HERMES_HOME>/approval-bridge/telegram-callbacks``.
"""
from __future__ import annotations

import calendar
import json
import os
import re
import secrets
import socket
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home

BOT_API = "https://api.telegram.org"
SAFE_CALLBACK_RE = re.compile(r"^[A-Za-z0-9_.-]{6,80}$")
SAFE_REQUEST_RE = re.compile(r"^[A-Za-z0-9_.:-]{6,160}$")
MAX_TIMEOUT_SECONDS = 600


@dataclass
class KanbanApprovalResult:
    request_id: str
    task_id: str
    action: str
    status: str
    already_resolved: bool = False
    executed: bool = False
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    request_path: Optional[str] = None


def _base_dir() -> Path:
    return get_hermes_home() / "approval-bridge"


def requests_dir() -> Path:
    return _base_dir() / "requests"


def callbacks_dir() -> Path:
    return _base_dir() / "telegram-callbacks"


def _ensure_dirs() -> None:
    for d in (_base_dir(), requests_dir(), callbacks_dir()):
        d.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(d, 0o700)
        except OSError:
            pass


def _safe_request_id(value: str) -> str:
    if not SAFE_REQUEST_RE.fullmatch(value or "") or ".." in value:
        raise ValueError("invalid approval request id")
    return value


def _safe_callback_id(value: str) -> str:
    if not SAFE_CALLBACK_RE.fullmatch(value or "") or ".." in value:
        raise ValueError("invalid callback id")
    return value


def _request_path(request_id: str) -> Path:
    return requests_dir() / f"{_safe_request_id(request_id)}.json"


def _callback_path(callback_id: str) -> Path:
    return callbacks_dir() / f"{_safe_callback_id(callback_id)}.json"


def _load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    env_path = get_hermes_home() / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            env[key.strip()] = val.strip().strip('"').strip("'")
    env.update({k: v for k, v in os.environ.items() if k.startswith("TELEGRAM_") or k.startswith("HERMES_")})
    return env


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    tmp.replace(path)


def _load_request(request_id: str) -> dict[str, Any]:
    path = _request_path(request_id)
    if not path.exists():
        raise FileNotFoundError(f"approval request not found: {request_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def _save_request(data: dict[str, Any]) -> None:
    request_id = str(data.get("id") or "")
    _write_json_atomic(_request_path(request_id), data)


def _parse_utc(value: str) -> Optional[float]:
    try:
        return float(calendar.timegm(time.strptime(value, "%Y-%m-%dT%H:%M:%SZ")))
    except Exception:
        return None


def _is_expired(data: dict[str, Any]) -> bool:
    expiry = _parse_utc(str(data.get("expires_at") or ""))
    return expiry is not None and time.time() > expiry


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _html_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _compact(value: str, limit: int) -> str:
    text = re.sub(r"\n{3,}", "\n\n", str(value or "").strip())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 20)].rstrip() + "\n…[truncated]"


def _default_command(task_id: str, request_id: str) -> list[str]:
    return [
        "hermes",
        "kanban",
        "unblock",
        "--reason",
        f"Approved via Telegram Kanban approval request={request_id}",
        task_id,
    ]


def _find_existing_pending(task_id: str, title: str) -> Optional[dict[str, Any]]:
    if not requests_dir().exists():
        return None
    for path in sorted(requests_dir().glob("kanban-*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("kanban_task_id") != task_id:
            continue
        if data.get("title") != title:
            continue
        if data.get("status") == "pending" and not _is_expired(data):
            return data
    return None


def create_request(
    *,
    task_id: str,
    title: str,
    message: str,
    command: Optional[list[str]] = None,
    cwd: Optional[str] = None,
    timeout_seconds: int = 120,
    ttl_seconds: int = 3600,
    service: str = "kanban",
    chat_id: Optional[str] = None,
    priority: str = "normal",
    send_telegram: bool = True,
    dedupe: bool = True,
) -> dict[str, Any]:
    """Create one durable Kanban approval request and optionally send buttons.

    Returns a dict with ``id``, ``callback_id``, ``path``, ``task_id``,
    ``deduped``, and optional Telegram fields.  Existing pending requests for
    the same ``task_id`` + ``title`` are returned instead of duplicated.
    """
    if not re.fullmatch(r"t_[0-9a-fA-F]+", task_id or ""):
        raise ValueError("task_id must look like t_<hex>")
    if not title.strip():
        raise ValueError("title is required")
    if not message.strip():
        raise ValueError("message is required")
    if command is not None and (not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command)):
        raise ValueError("command must be a non-empty list of strings")

    _ensure_dirs()
    if dedupe:
        existing = _find_existing_pending(task_id, title)
        if existing:
            return {
                "id": existing["id"],
                "callback_id": existing.get("callback_id"),
                "path": str(_request_path(existing["id"])),
                "task_id": task_id,
                "deduped": True,
                "telegram_message_id": existing.get("telegram_message_id"),
                "chat_id": existing.get("telegram_chat_id"),
            }

    now = time.time()
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now))
    request_id = f"kanban-{task_id}-{ts}-{secrets.token_hex(3)}"
    callback_id = secrets.token_urlsafe(9).replace("-", "_").rstrip("=")
    expires_at = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ",
        time.gmtime(now + max(60, min(int(ttl_seconds), 86400))),
    )
    cmd = command or _default_command(task_id, request_id)
    data: dict[str, Any] = {
        "id": request_id,
        "title": title.strip(),
        "message": message.strip(),
        "service": service,
        "priority": 1 if priority == "urgent" else 2,
        "created_at": _utc_now(),
        "expires_at": expires_at,
        "ttl_seconds": int(ttl_seconds),
        "created_by": socket.gethostname(),
        "status": "pending",
        "command": cmd,
        "cwd": cwd or str(Path.home()),
        "timeout_seconds": max(1, min(int(timeout_seconds), MAX_TIMEOUT_SECONDS)),
        "kanban_task_id": task_id,
        "approval_surface": "telegram-native-callback",
        "callback_id": callback_id,
    }
    req_path = _request_path(request_id)
    _write_json_atomic(req_path, data)
    _write_json_atomic(_callback_path(callback_id), {"request_id": request_id})

    result: dict[str, Any] = {
        "id": request_id,
        "callback_id": callback_id,
        "path": str(req_path),
        "task_id": task_id,
        "deduped": False,
    }
    if send_telegram:
        tg = send_telegram_prompt(data, callback_id=callback_id, chat_id=chat_id)
        result.update(tg)
        data.update({k: v for k, v in tg.items() if k in {"telegram_message_id", "telegram_chat_id", "sent_at"}})
        _save_request(data)
    return result


def send_telegram_prompt(data: dict[str, Any], *, callback_id: str, chat_id: Optional[str] = None) -> dict[str, Any]:
    env = _load_env()
    token = env.get("TELEGRAM_BOT_TOKEN") or env.get("HERMES_TELEGRAM_BOT_TOKEN")
    dest = chat_id or env.get("TELEGRAM_HOME_CHAT_ID") or env.get("TELEGRAM_CHAT_ID") or env.get("TELEGRAM_HOME_CHANNEL")
    if not dest:
        allowed = env.get("TELEGRAM_ALLOWED_USER_IDS", "") or env.get("TELEGRAM_ALLOWED_USERS", "")
        dest = next((x.strip() for x in re.split(r"[, ]+", allowed) if x.strip()), "")
    if not token or not dest:
        raise RuntimeError("missing Telegram bot token or chat id")

    command = data.get("command") or []
    task_id = str(data.get("kanban_task_id") or "")
    text = "\n".join([
        "🔐 <b>Kanban approval needed</b>",
        f"<b>{_html_escape(str(data.get('title') or 'Mutation approval'))}</b>",
        "",
        f"Task: <code>{_html_escape(task_id)}</code>",
        "",
        _html_escape(_compact(str(data.get("message") or ""), 2600)),
        "",
        f"Approve runs: <code>{_html_escape(' '.join(map(str, command)))}</code>",
        f"Expires: <code>{_html_escape(str(data.get('expires_at') or ''))}</code>",
    ])
    payload = {
        "chat_id": dest,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
        "reply_markup": {
            "inline_keyboard": [
                [
                    {"text": "✅ Approve / unblock", "callback_data": f"ka:a:{callback_id}"},
                    {"text": "⏸ Defer", "callback_data": f"ka:d:{callback_id}"},
                ],
                [{"text": "❌ Dismiss", "callback_data": f"ka:x:{callback_id}"}],
            ]
        },
    }
    req = urllib.request.Request(
        f"{BOT_API}/bot{token}/sendMessage",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 - Telegram API endpoint from constant
        body = json.loads(resp.read().decode("utf-8"))
    msg = body.get("result", {}) if isinstance(body, dict) else {}
    return {
        "telegram_message_id": msg.get("message_id"),
        "telegram_chat_id": str(dest),
        "sent_at": _utc_now(),
    }


def resolve_callback(callback_id: str, action: str, *, actor: str = "telegram", execute: bool = True) -> KanbanApprovalResult:
    """Resolve a ``ka:*`` callback idempotently.

    ``action`` is one of ``a``/``approve``, ``d``/``defer``, or
    ``x``/``dismiss``/``deny``.  Approve executes the recorded command exactly
    once while the request is pending and unexpired.
    """
    action_map = {
        "a": "approve",
        "approve": "approve",
        "d": "defer",
        "defer": "defer",
        "x": "dismiss",
        "dismiss": "dismiss",
        "deny": "dismiss",
    }
    normalized = action_map.get(str(action).lower())
    if not normalized:
        raise ValueError("action must be approve, defer, or dismiss")
    mapping = json.loads(_callback_path(callback_id).read_text(encoding="utf-8"))
    request_id = str(mapping.get("request_id") or "")
    data = _load_request(request_id)
    task_id = str(data.get("kanban_task_id") or "")
    status = str(data.get("status") or "pending")
    if status != "pending":
        return KanbanApprovalResult(
            request_id=request_id,
            task_id=task_id,
            action=normalized,
            status=status,
            already_resolved=True,
            request_path=str(_request_path(request_id)),
        )
    if _is_expired(data):
        data["status"] = "expired"
        data["handled_at"] = _utc_now()
        data["handled_by"] = actor
        _save_request(data)
        return KanbanApprovalResult(
            request_id=request_id,
            task_id=task_id,
            action=normalized,
            status="expired",
            request_path=str(_request_path(request_id)),
        )

    data["handled_at"] = _utc_now()
    data["handled_by"] = actor
    data["handled_action"] = normalized
    result_payload: dict[str, Any] = {}
    if normalized == "approve":
        if execute:
            result_payload = _execute_request(data)
        else:
            result_payload = {"executed": False, "exit_code": 0, "stdout": "", "stderr": "execution skipped"}
        data["result"] = result_payload
        data["status"] = "completed" if result_payload.get("exit_code") == 0 else "failed"
    elif normalized == "defer":
        data["status"] = "deferred"
    else:
        data["status"] = "dismissed"
    _save_request(data)
    return KanbanApprovalResult(
        request_id=request_id,
        task_id=task_id,
        action=normalized,
        status=str(data["status"]),
        executed=bool(result_payload.get("executed")),
        exit_code=result_payload.get("exit_code"),
        stdout=str(result_payload.get("stdout") or ""),
        stderr=str(result_payload.get("stderr") or ""),
        request_path=str(_request_path(request_id)),
    )


def _execute_request(data: dict[str, Any]) -> dict[str, Any]:
    command = data.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
        return {"executed": False, "exit_code": None, "stdout": "", "stderr": "no command recorded"}
    timeout = max(1, min(int(data.get("timeout_seconds") or 120), MAX_TIMEOUT_SECONDS))
    cwd = data.get("cwd") if isinstance(data.get("cwd"), str) else str(Path.home())
    started = _utc_now()
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "executed": True,
            "started_at": started,
            "finished_at": _utc_now(),
            "exit_code": proc.returncode,
            "stdout": (proc.stdout or "")[-4000:],
            "stderr": (proc.stderr or "")[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "executed": True,
            "started_at": started,
            "finished_at": _utc_now(),
            "exit_code": 124,
            "stdout": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr": f"timed out after {timeout}s",
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "executed": True,
            "started_at": started,
            "finished_at": _utc_now(),
            "exit_code": 125,
            "stdout": "",
            "stderr": str(exc),
        }

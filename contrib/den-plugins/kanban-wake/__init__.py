"""kanban-wake — wake the supervisor's *current chat session* when a kanban card
blocks or completes.

Two halves, one plugin (both gateways load it):

1. Emitter (runs wherever kanban lifecycle hooks fire — dispatcher/worker
   gateway): `kanban_task_blocked` / `kanban_task_completed` -> HMAC-signed POST
   to the default gateway's webhook route `kanban-wake`.

2. Redirector (runs in the default gateway that owns the Telegram session):
   `pre_gateway_dispatch` sees the inbound webhook MessageEvent for route
   `kanban-wake`, SKIPs the detached webhook agent, and instead schedules the
   event text as an inbound user-role turn in the configured supervisor
   session (`kanban_wake.session_key`). The supervisor session therefore wakes
   with full context and replies in the same chat. Falls back to the
   detached webhook agent if injection is impossible (so the wake never drops).

Config (config.yaml of the default profile):
  kanban_wake:
    session_key: agent:main:telegram:dm:1916982742:613471   # REQUIRED for redirect
    url:    http://127.0.0.1:8644/webhooks/kanban-wake
    secret_file: ~/.secrets/kanban-wake-secret
    debounce_seconds: 60
  plugins.entries.kanban-wake.allow_gateway_injection: true
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
import urllib.request

logger = logging.getLogger(__name__)

_ROUTE = "kanban-wake"
_last: dict[str, float] = {}


def _cfg() -> dict:
    try:
        from hermes_cli.config import load_config
        return (load_config() or {}).get("kanban_wake", {}) or {}
    except Exception:
        return {}


def _secret() -> str:
    p = os.path.expanduser(_cfg().get("secret_file", "~/.secrets/kanban-wake-secret"))
    try:
        return open(p).read().strip()
    except Exception:
        return ""


# ---------------------------------------------------------------- emitter ---

def _post(text: str, task_id: str = "", event: str = "") -> None:
    url = _cfg().get("url", f"http://127.0.0.1:8644/webhooks/{_ROUTE}")
    body = json.dumps({"text": text, "task_id": task_id, "event": event}).encode()
    ts = str(int(time.time()))
    headers = {"Content-Type": "application/json"}
    sec = _secret()
    if sec:
        sig = hmac.new(sec.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
        headers.update({"X-Webhook-Timestamp": ts, "X-Webhook-Signature-V2": f"sha256={sig}"})
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        r.read()


def _emit(event: str, task_id: str | None, **fields) -> None:
    try:
        if not task_id:
            return
        now = time.time()
        key = f"{event}:{task_id}"
        if now - _last.get(key, 0) < float(_cfg().get("debounce_seconds", 60)):
            return
        _last[key] = now
        title = fields.get("title") or ""
        assignee = fields.get("assignee") or ""
        reason = fields.get("reason") or fields.get("blocked_reason") or fields.get("summary") or ""
        if not (title and reason):
            try:
                from hermes_cli import kanban_db as kb
                with kb.connect() as conn:
                    t = kb.get_task(conn, task_id)
                    if t:
                        title = title or (t.title or "")
                        assignee = assignee or (t.assignee or "")
                        if not reason:
                            cs = kb.list_comments(conn, task_id)
                            if cs:
                                reason = cs[-1].body or ""
            except Exception:
                pass
        head = "BLOCKED" if event == "blocked" else "DONE"
        text = f"KANBAN {head} {task_id} [{assignee}] {title}\n{reason[:1500]}"
        _post(text, task_id, event)
        logger.info("[kanban-wake] emitted %s for %s", event, task_id)
    except Exception as e:  # never break dispatch
        logger.warning("[kanban-wake] emit failed: %s", e)


def _on_blocked(task_id=None, **f):
    _emit("blocked", task_id, **f)


def _on_completed(task_id=None, **f):
    _emit("completed", task_id, **f)


# ------------------------------------------------------------- redirector ---

def _pre_gateway_dispatch(event=None, gateway=None, **_):
    """Intercept the kanban-wake webhook turn and re-inject into the supervisor session."""
    try:
        src = getattr(event, "source", None)
        if src is None or str(getattr(src, "platform", "")).lower().endswith("webhook") is False:
            return None
        if getattr(src, "user_id", "") != f"webhook:{_ROUTE}":
            return None
        session_key = _cfg().get("session_key")
        if not session_key or gateway is None:
            return None  # fall back to detached webhook agent
        text = getattr(event, "text", "") or ""
        content = (
            "[kanban-wake — event from the board, not the user. Act on it now as EM: "
            "answer the block with a standing decision or ask Den the exact question; "
            "on DONE verify the PR/merge/live state and drive the next hop. Reply in ≤5 sentences.]\n"
            + text
        )
        sched = getattr(gateway, "_schedule_plugin_message_injection", None)
        if not callable(sched):
            return None
        ok = sched(session_key=session_key, content=content, plugin_id="kanban-wake")
        if ok:
            logger.info("[kanban-wake] redirected event into %s", session_key)
            return {"action": "skip", "reason": "kanban-wake redirected into supervisor session"}
        logger.warning("[kanban-wake] injection refused for %s; falling back to webhook agent", session_key)
        return None
    except Exception as e:
        logger.warning("[kanban-wake] redirect failed: %s", e)
        return None


def register(ctx) -> None:
    ctx.register_hook("kanban_task_blocked", _on_blocked)
    ctx.register_hook("kanban_task_completed", _on_completed)
    ctx.register_hook("pre_gateway_dispatch", _pre_gateway_dispatch)

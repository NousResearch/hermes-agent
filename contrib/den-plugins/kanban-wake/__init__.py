"""kanban-wake — push a Telegram ping the moment a kanban card blocks or completes.

Hooks the dispatcher/worker lifecycle observers (``kanban_task_blocked``,
``kanban_task_completed``) and POSTs a ``deliver_only`` webhook route on the
default gateway (``platforms.webhook.extra.routes.kanban-wake``), which sends
the text straight to Telegram with zero LLM cost.  Replaces waiting for the
30-minute heartbeat to notice a worker asked a question.

Config (config.yaml):
  kanban_wake.url        default http://127.0.0.1:8644/webhooks/kanban-wake
  kanban_wake.secret_file default ~/.secrets/kanban-wake-secret
  kanban_wake.debounce_s  default 60 (per task+event)

Fails open: any error is logged and swallowed; never breaks dispatch.
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

_LAST: dict[str, float] = {}


def _cfg():
    try:
        from hermes_cli.config import load_config
        return (load_config() or {}).get("kanban_wake", {}) or {}
    except Exception:
        return {}


def _secret(cfg) -> str:
    p = os.path.expanduser(cfg.get("secret_file", "~/.secrets/kanban-wake-secret"))
    try:
        return open(p).read().strip()
    except Exception:
        return ""


def _post(text: str) -> None:
    cfg = _cfg()
    url = cfg.get("url", "http://127.0.0.1:8644/webhooks/kanban-wake")
    secret = _secret(cfg)
    if not secret:
        logger.warning("[kanban-wake] no secret file; skipping")
        return
    body = json.dumps({"text": text}).encode()
    ts = str(int(time.time()))
    sig = hmac.new(secret.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Webhook-Timestamp": ts,
            "X-Webhook-Signature-V2": sig,
            "X-Request-ID": f"kw-{ts}-{hashlib.sha1(body).hexdigest()[:8]}",
        },
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        logger.info("[kanban-wake] delivered http=%s", r.status)


def _fire(event: str, task_id: str, reason: str | None, assignee, run_id) -> None:
    try:
        cfg = _cfg()
        key = f"{task_id}:{event}"
        now = time.time()
        if now - _LAST.get(key, 0) < float(cfg.get("debounce_s", 60)):
            return
        _LAST[key] = now
        title = ""
        try:
            from hermes_cli import kanban_db as kb
            with kb.connect() as conn:
                t = kb.get_task(conn, task_id)
                title = (t.title or "")[:80] if t else ""
        except Exception:
            pass
        head = "BLOCKED" if event == "blocked" else "DONE"
        body = (reason or "").strip().replace("\n", " ")[:400]
        text = f"kanban {head} {task_id} ({assignee or '?'}, run {run_id or '-'}): {title}"
        if body:
            text += f"\n{body}"
        _post(text)
    except Exception as e:  # never break dispatch
        logger.warning("[kanban-wake] %s", e)


def register(ctx) -> None:
    ctx.register_hook(
        "kanban_task_blocked",
        lambda task_id=None, reason=None, assignee=None, run_id=None, **_: _fire("blocked", task_id, reason, assignee, run_id),
    )
    ctx.register_hook(
        "kanban_task_completed",
        lambda task_id=None, summary=None, assignee=None, run_id=None, **_: _fire("completed", task_id, summary, assignee, run_id),
    )

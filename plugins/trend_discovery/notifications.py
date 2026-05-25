"""Notification and watchdog helpers."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from typing import Any
from urllib import request

from .store import TrendDiscoveryStore, utc_now


def _post_webhook(url: str, message: str, timeout: int = 10) -> dict[str, Any]:
    body = json.dumps({"text": message}).encode("utf-8")
    req = request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=timeout) as resp:
        return {"status_code": resp.status, "body": resp.read(512).decode("utf-8", "replace")}


def notify(store: TrendDiscoveryStore, message: str, target: str | None = None) -> dict[str, Any]:
    store.init()
    target = target or store.get_config("notification.primary", "local")
    notification_id = uuid.uuid4().hex[:12]
    created_at = utc_now()
    status = "sent"
    error = ""
    evidence: dict[str, Any] = {"target": target}
    try:
        if target == "local":
            evidence["mode"] = "receipt-log"
        elif target == "macos":
            osascript = shutil.which("osascript")
            if not osascript:
                raise RuntimeError("osascript is not available on this host")
            title = "Hermes Trend Discovery"
            subprocess.run(
                [
                    osascript,
                    "-e",
                    (
                        "display notification "
                        f"{json.dumps(message[:240])} "
                        f"with title {json.dumps(title)}"
                    ),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            evidence["mode"] = "macos-notification"
        elif target.startswith("webhook:"):
            evidence["webhook"] = _post_webhook(target.split(":", 1)[1], message)
        elif target == "env-webhook":
            url = os.getenv("HERMES_TD_WEBHOOK_URL", "").strip()
            if not url:
                raise RuntimeError("HERMES_TD_WEBHOOK_URL is not configured")
            evidence["webhook"] = _post_webhook(url, message)
        else:
            raise RuntimeError(f"Unsupported notification target: {target}")
    except Exception as exc:
        status = "failed"
        error = str(exc)
        fallback = store.get_config("notification.fallback", "local")
        evidence["fallback"] = fallback
        if fallback == "local":
            status = "sent"
            evidence["fallback_mode"] = "receipt-log"
    with store.connect() as conn:
        conn.execute(
            """
            INSERT INTO notifications
                (notification_id, target, status, message, created_at, sent_at, error, evidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                notification_id,
                target,
                status,
                message,
                created_at,
                utc_now() if status == "sent" else None,
                error,
                json.dumps(evidence, sort_keys=True),
            ),
        )
    return {
        "notification_id": notification_id,
        "target": target,
        "status": status,
        "error": error,
        "evidence": evidence,
    }


def watchdog(store: TrendDiscoveryStore, *, notify_user: bool = True) -> dict[str, Any]:
    store.init()
    now = datetime.now(timezone.utc)
    alerts: list[str] = []
    with store.connect() as conn:
        overdue = conn.execute(
            """
            SELECT phase_id, name, due_at, percent_complete
            FROM phases
            WHERE percent_complete < 100 AND due_at IS NOT NULL
            """
        ).fetchall()
        for row in overdue:
            due = datetime.fromisoformat(row["due_at"])
            if due < now:
                alerts.append(
                    f"{row['phase_id']} overdue at {row['due_at']} ({row['percent_complete']}%)"
                )
        failing_sources = conn.execute(
            """
            SELECT name, failure_count, last_error
            FROM sources
            WHERE failure_count >= 3
            ORDER BY failure_count DESC, name
            """
        ).fetchall()
        for row in failing_sources:
            alerts.append(
                f"source {row['name']} failed {row['failure_count']} time(s): {row['last_error']}"
            )
        last_success = conn.execute(
            "SELECT started_at FROM runs WHERE status='success' ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        if last_success is None:
            alerts.append("no successful run recorded yet")
    result = {"ok": not alerts, "alerts": alerts, "notified": None}
    if alerts and notify_user:
        result["notified"] = notify(store, "Trend Discovery watchdog alert:\n" + "\n".join(alerts))
    return result

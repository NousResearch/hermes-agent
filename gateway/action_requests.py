"""Persistent gateway action requests for chat buttons.

Action requests let a delivered message include compact callback-data buttons
without stuffing the whole task payload into Telegram/Discord component IDs.
The stored JSON is intentionally boring and local: no secrets, no execution on
read, and filenames are opaque random IDs.
"""

from __future__ import annotations

import json
import secrets
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

_ACTIONS_DIR = "action_requests"


@dataclass(frozen=True)
class ActionRequest:
    id: str
    kind: str
    action: str
    payload: dict[str, Any]


def _safe_part(value: str) -> str:
    return "".join(ch for ch in value if ch.isalnum() or ch in {"-", "_"})[:80]


def actions_root() -> Path:
    root = get_hermes_home() / _ACTIONS_DIR
    root.mkdir(parents=True, exist_ok=True)
    return root


def store_action_request(kind: str, action: str, payload: dict[str, Any]) -> ActionRequest:
    """Store an action request and return its opaque ID."""
    kind_safe = _safe_part(kind or "generic") or "generic"
    action_safe = _safe_part(action or "run") or "run"
    request_id = f"{kind_safe}-{secrets.token_urlsafe(12)}"
    path = actions_root() / f"{request_id}.json"
    body = {
        "id": request_id,
        "kind": kind_safe,
        "action": action_safe,
        "payload": payload or {},
    }
    path.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
    path.chmod(0o600)
    return ActionRequest(id=request_id, kind=kind_safe, action=action_safe, payload=payload or {})


def load_action_request(request_id: str) -> ActionRequest:
    request_id = _safe_part(request_id)
    if not request_id:
        raise ValueError("Invalid action request ID")
    path = actions_root() / f"{request_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return ActionRequest(
        id=str(data.get("id") or request_id),
        kind=str(data.get("kind") or "generic"),
        action=str(data.get("action") or "run"),
        payload=data.get("payload") if isinstance(data.get("payload"), dict) else {},
    )


def dispatch_action_request(request_id: str) -> subprocess.Popen:
    """Dispatch an action request in the background.

    Currently supported:
      - kind=sentry, action=create_pr: launch the Sentry create-PR helper.
    """
    req = load_action_request(request_id)
    if req.kind == "sentry" and req.action == "create_pr":
        helper = Path(__file__).resolve().parents[1] / "scripts" / "sentry_create_pr_from_packet.py"
        return subprocess.Popen(
            [sys.executable, str(helper), str(actions_root() / f"{req.id}.json")],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    raise ValueError(f"Unsupported action request: {req.kind}/{req.action}")


def build_action_buttons(actions: list[dict[str, Any]], payload: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """Convert route metadata actions into compact button descriptors.

    Each input action supports:
      - label: button text
      - kind: e.g. "sentry"
      - action: e.g. "create_pr"
      - payload: optional payload override; defaults to the webhook payload
    """
    buttons: list[dict[str, str]] = []
    for item in actions or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or item.get("action") or "Run").strip()[:64]
        kind = str(item.get("kind") or "generic")
        action = str(item.get("action") or "run")
        action_payload = item.get("payload") if isinstance(item.get("payload"), dict) else payload or {}
        req = store_action_request(kind, action, action_payload)
        buttons.append({"label": label, "callback_data": f"ar:{req.id}"})
    return buttons

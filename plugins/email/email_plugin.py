"""Email connector — thin subprocess wrapper over gws (googleworkspace/cli v0.22.5).

Per `hermes-202605-014`, this plugin exposes five tools (list_emails, read_email,
draft_reply, send_email, mark_email) that shell out to the bundled
``googleworkspace/cli`` (``gws``) binary. OAuth, token storage, refresh, and
account management are owned entirely by gws — this module never touches
credentials.

Binary resolution order:

1. ``AGENTE_GWS_BIN`` environment variable (set by the desktop shell when
   spawning the agent runtime, pointing at the bundled binary inside the app
   resources).
2. ``shutil.which("gws")`` — a developer-installed gws on PATH.

If neither resolves the tools return a clear ``gws not bundled`` error so the
caller can surface a Connectors-UI prompt.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
from email.mime.text import MIMEText
from typing import Any

_GWS_TIMEOUT = 30


class GwsUnavailableError(RuntimeError):
    """Raised when no gws binary can be located."""


def _resolve_gws_bin() -> str:
    """Resolve gws binary path lazily from env or PATH."""
    path = os.environ.get("AGENTE_GWS_BIN") or shutil.which("gws")
    if not path:
        raise GwsUnavailableError(
            "gws not bundled: set AGENTE_GWS_BIN or install googleworkspace/cli on PATH"
        )
    return path


def _gws_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("GOOGLE_WORKSPACE_CLI_CLIENT_SECRET", None)
    return env


def _gws_json(args: list[str]) -> Any:
    """Run ``gws <args>`` and return parsed JSON stdout."""
    gws_bin = _resolve_gws_bin()
    try:
        result = subprocess.run(
            [gws_bin, *args],
            capture_output=True,
            text=True,
            timeout=_GWS_TIMEOUT,
            env=_gws_env(),
        )
    except subprocess.TimeoutExpired:
        return {"error": "gws_timeout", "timeout": _GWS_TIMEOUT}
    if result.returncode != 0:
        return {"error": "gws_error", "stderr": result.stderr.strip(), "returncode": result.returncode}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": "gws_bad_json", "raw": result.stdout[:500]}


def _encode_text_message(body: str, headers: dict[str, str] | None = None) -> str:
    msg = MIMEText(body, _charset="utf-8")
    for name, value in (headers or {}).items():
        msg[name] = value
    return base64.urlsafe_b64encode(msg.as_bytes()).decode()


def list_emails(folder: str = "INBOX", since: str | None = None, limit: int = 10) -> list[dict]:
    """Return the most recent messages in ``folder`` (default INBOX)."""
    params: dict[str, Any] = {"userId": "me", "maxResults": limit, "labelIds": [folder]}
    if since:
        params["q"] = f"after:{since}"
    return _gws_json(["gmail", "users", "messages", "list", "--params", json.dumps(params)])


def read_email(message_id: str) -> dict:
    """Return full message payload for ``message_id`` (raw body preserved)."""
    params = {"userId": "me", "id": message_id}
    return _gws_json(["gmail", "users", "messages", "get", "--raw", "--params", json.dumps(params)])


def draft_reply(message_id: str, body: str) -> dict:
    """Create a Gmail draft replying to ``message_id`` with ``body``."""
    params = {"userId": "me"}
    raw = _encode_text_message(
        body,
        {
            "In-Reply-To": message_id,
            "References": message_id,
        },
    )
    draft_body = {"message": {"threadId": message_id, "raw": raw}}
    return _gws_json(["gmail", "users", "drafts", "create", "--params", json.dumps(params), "--body", json.dumps(draft_body)])


def send_email(to: str, subject: str, body: str) -> dict:
    """Send a new email."""
    raw = _encode_text_message(body, {"to": to, "subject": subject})
    params = {"userId": "me"}
    message_body = {"raw": raw}
    return _gws_json(["gmail", "users", "messages", "send", "--params", json.dumps(params), "--body", json.dumps(message_body)])


def mark_email(message_id: str, add_label: str | None = None, remove_label: str | None = None) -> dict:
    """Add or remove a label on ``message_id``."""
    params = {"userId": "me", "id": message_id}
    body: dict[str, Any] = {}
    if add_label:
        body["addLabelIds"] = [add_label]
    if remove_label:
        body["removeLabelIds"] = [remove_label]
    return _gws_json(["gmail", "users", "messages", "modify", "--params", json.dumps(params), "--body", json.dumps(body)])

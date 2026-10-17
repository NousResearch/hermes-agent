"""Mattermost Blocks exec-approval cards (mm_blocks buttons + action callbacks).

The Mattermost server POSTs button presses server-to-server to a small HTTP endpoint this
adapter hosts (``POST /mattermost/approval/<secret>``). The press payload carries the
presser's ``user_id`` — attribution is native — and the action registry ``context`` is
server-only, so the session key never reaches clients.

Optionally escalates: after ``escalate_after`` seconds with no answer, publish an ntfy
notification whose ``view`` action deep-links to the approval post (single point of truth
stays the Mattermost button; ntfy is only a doorbell).

Pure helpers here; ``MattermostAdapter`` (adapter.py) owns the sockets and timers.
"""

from __future__ import annotations

import secrets as _secrets
from typing import Any, Dict, List, Optional, Tuple

# choices resolve_gateway_approval accepts — vocabulary shared with the base adapter.
VALID_CHOICES = {"once", "session", "always", "deny"}

# Post-action response strings (update replaces the card; empty props clear the buttons).
_APPROVED_TPL = "✅ Approved by {who} ({choice}) — run continues."
_DENIED_TPL = "❌ Denied by {who} — the command will NOT run."
_ALREADY_TPL = "⌛ This approval was already resolved or expired."
_UNAUTHORIZED = "⛔ You are not allowed to answer approval prompts."

# Header-style ntfy publish: this server rejects JSON-body publishing (HTTP 40024, live-verified)
# but fully supports the X-Actions header — including the view action that deep-links to the card.
_NTFY_TITLE = "Venom needs your approval"


def approval_actions_config(extra: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Buttons config from PlatformConfig.extra; None (disabled) unless URL + secret are set.

    Keys: ``approval_actions_url`` (external base URL the Mattermost server can reach),
    ``approval_actions_secret`` (path token), ``approval_actions_port`` (local bind, default
    8647), ``approval_escalate_after`` (seconds; default = half the approval timeout, 0 = off).
    """
    url = str(extra.get("approval_actions_url") or "").strip().rstrip("/")
    secret = str(extra.get("approval_actions_secret") or "").strip()
    if not url or not secret:
        return None
    raw_port = extra.get("approval_actions_port")
    return {
        "url": url,
        "secret": secret,
        "port": int(raw_port) if raw_port is not None else 8647,  # 0 = ephemeral (tests)
        # None = "half the configured approvals.timeout" (resolved at send time)
        "escalate_after": extra.get("approval_escalate_after"),
    }


def callback_path(secret: str) -> str:
    """The endpoint path Mattermost registers for every button on the card."""
    return f"/mattermost/approval/{secret}"


def build_approval_props(prompt, callback_url: str) -> Dict[str, Any]:
    """``props`` for the approval post, using legacy attachment actions — the interactive
    format every Mattermost renders natively. (Native ``mm_blocks`` requires the server-side
    ``MmBlocksEnabled`` feature flag; on servers without it the blocks props are silently
    stripped and the card degrades to plain text — observed live on Mattermost 11.9.0.)
    Callback payload and post-action response are identical for both formats."""
    actions: List[Dict[str, Any]] = []
    for label, choice, style in prompt.actions:
        # Action IDs must match [A-Za-z0-9]+ — no underscores or hyphens. Mattermost 11.9.0
        # registers the action route with that narrow pattern; "approve_once" 404s at the
        # router with the generic "could not find the page" error (live-verified).
        actions.append({
            "id": f"approve{choice}", "name": label, "type": "button",
            "integration": {"url": callback_url,
                            "context": {"session_key": prompt.session_key, "choice": choice}},
        })
    return {
        "attachments": [{
            "color": "#FF851B",  # MTK ops orange; the button row's accent bar
            "text": prompt.text,
            "actions": actions,
        }],
    }


def parse_action_payload(payload: Any, allowed_user_ids: set) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Validate a post-action callback → (verdict, fields).

    verdict: "ok" (fields: session_key/choice/user_id/user_name/post_id), "unauthorized",
    "invalid", or "already" (well-formed but nothing pending — safe to answer with a notice).
    """
    if not isinstance(payload, dict):
        return "invalid", None
    user_id = str(payload.get("user_id") or "")
    if allowed_user_ids and user_id not in allowed_user_ids:
        return "unauthorized", None
    context = payload.get("context") or {}
    session_key = str(context.get("session_key") or "")
    choice = str(context.get("choice") or "")
    if not session_key or choice not in VALID_CHOICES:
        return "invalid", None
    return "ok", {
        "session_key": session_key, "choice": choice, "user_id": user_id,
        "user_name": str(payload.get("user_name") or user_id),
        "post_id": str(payload.get("post_id") or ""),
    }


def action_response(verdict: str, fields: Optional[Dict[str, Any]], choice: str = "") -> Dict[str, Any]:
    """Post-action response body: update the card in place (empty props clear the buttons)."""
    if verdict == "ok":
        text = (_DENIED_TPL if choice == "deny" else _APPROVED_TPL).format(
            who=(fields or {}).get("user_name") or "user", choice=choice)
    elif verdict == "unauthorized":
        text = _UNAUTHORIZED
    elif verdict == "already":
        text = _ALREADY_TPL
    else:
        text = "⚠️ Unrecognized approval action."
    return {"update": {"message": text, "props": {}}}


def build_ntfy_escalation(command: str, permalink: str) -> Tuple[str, Dict[str, str]]:
    """Header-style ntfy publish for the T+escalate nudge → (message body, headers).

    ``Actions`` uses the short ``view`` form: ``view, <label>, <url>[, clear=true]`` — verified
    working against the MTK ntfy server (JSON-body publish is rejected there, live-tested).
    """
    snippet = (command or "").strip()
    if len(snippet) > 400:
        snippet = snippet[:397] + "..."
    message = snippet or "A flagged command is waiting for your OK."
    headers = {
        "Title": _NTFY_TITLE,
        "Priority": "high",
        "Tags": "warning",
        "Actions": f"view, Open in Mattermost, {permalink}, clear=true",
    }
    return message, headers


def new_secret() -> str:
    return _secrets.token_urlsafe(24)

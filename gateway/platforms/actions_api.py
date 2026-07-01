"""
Tool-approval action HTTP endpoints for the OpenAI-compatible API server.

Exposes the deferred tool-approval operations — the same ones the
``hermes action {list,show,approve,reject}`` CLI drives — under ``/v1/actions/*``
so a headless control plane (the master console) can resolve a staged outbound
action (e.g. a gated ``send_gmail_message``) with one click instead of an
operator running the CLI on the host.

Design constraints (mirror ``kanban_api.py``):

- **Additive.** This module only adds handlers; ``api_server`` imports it lazily
  and registers the routes in ``connect()`` (import + routes only).
- **Reuse the engine.** Approve goes through ``tools.tool_gate.approve_action``
  (which mints the agent-assigned execution card, keeps the one-shot per-pending
  token, TTL refusal, and double-execution guard intact). Reject mirrors the CLI:
  discard the pending record + archive the approval card. We never re-implement
  the staging/replay logic here.
- **Profile-scoped.** The pending store (``<HERMES_HOME>/pending/actions/``) and
  the one-shot replay live in *this* profile's home, so these endpoints run in the
  profile process that staged the action. The console targets the card's
  ``created_by`` profile's API server (see docs/hermes-tool-approval-actions-api.md).
- **Auth.** Reuses the API server's existing bearer check (``adapter._check_auth``).
"""

import logging
from typing import Any, Dict

from aiohttp import web

from gateway.platforms.api_server import _openai_error

logger = logging.getLogger(__name__)


def _err(message: str, *, status: int, code: str = None, param: str = None) -> "web.Response":
    return web.json_response(_openai_error(message, param=param, code=code), status=status)


def _summary(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Compact list-row view of a pending action. Never leaks the replay token."""
    payload = rec.get("payload") or {}
    return {
        "id": rec.get("id"),
        "tool_name": rec.get("tool_name") or payload.get("tool_name"),
        "summary": rec.get("summary"),
        "status": rec.get("status", "pending"),
        "tenant": rec.get("tenant") or payload.get("tenant"),
        "profile": rec.get("profile"),
        "created_at": rec.get("created_at"),
        "expires_at": rec.get("expires_at"),
        "card_id": rec.get("card_id"),
        "exec_card_id": rec.get("exec_card_id"),
    }


async def handle_list_actions(adapter, request: "web.Request") -> "web.Response":
    """GET /v1/actions — list staged tool-approval actions for this profile."""
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err
    try:
        from tools import tool_gate
        from tools import write_approval as wa
        recs = wa.list_pending(tool_gate.SUBSYSTEM)
    except Exception:
        logger.exception("GET /v1/actions failed")
        return _err("Failed to list actions", status=500, code="server_error")
    return web.json_response({"count": len(recs), "actions": [_summary(r) for r in recs]})


async def handle_get_action(adapter, request: "web.Request") -> "web.Response":
    """GET /v1/actions/{pending_id} — full staged-action record (token redacted)."""
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err
    pending_id = request.match_info["pending_id"]
    try:
        from tools import tool_gate
        from tools import write_approval as wa
        rec = wa.get_pending(tool_gate.SUBSYSTEM, pending_id)
    except Exception:
        logger.exception("GET /v1/actions/%s failed", pending_id)
        return _err("Failed to load action", status=500, code="server_error")
    if rec is None:
        return _err(f"No pending action {pending_id}", status=404, code="action_not_found")
    safe = {k: v for k, v in rec.items() if k != "token"}
    return web.json_response({"action": safe})


async def handle_approve_action(adapter, request: "web.Request") -> "web.Response":
    """POST /v1/actions/{pending_id}/approve — spawn the one-shot execution card.

    Delegates to ``tool_gate.approve_action`` so the one-shot token, TTL refusal,
    and double-execution guard are preserved. ``ok:false`` maps to 404 (missing)
    or 409 (already-resolved / expired).
    """
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err
    pending_id = request.match_info["pending_id"]
    try:
        from tools import tool_gate
        out = tool_gate.approve_action(pending_id)
    except Exception:
        logger.exception("POST /v1/actions/%s/approve failed", pending_id)
        return _err("Failed to approve action", status=500, code="server_error")
    if not out.get("ok"):
        msg = out.get("message", "Could not approve action.")
        status = 404 if "No pending action" in msg else 409
        return _err(msg, status=status, code="action_not_approvable")
    return web.json_response({
        "ok": True,
        "message": out.get("message"),
        "exec_card_id": out.get("exec_card_id"),
    })


async def handle_reject_action(adapter, request: "web.Request") -> "web.Response":
    """POST /v1/actions/{pending_id}/reject — discard the action + archive its card.

    Mirrors the CLI reject (``hermes action reject``): drop the pending record so
    it can never replay, then archive the human approval card (the card is
    ``ready``/review-only, so archive — not block — is the right transition).
    """
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err
    pending_id = request.match_info["pending_id"]
    try:
        from tools import tool_gate
        from tools import write_approval as wa
        rec = wa.get_pending(tool_gate.SUBSYSTEM, pending_id)
        if rec is None:
            return _err(f"No pending action {pending_id}", status=404, code="action_not_found")
        wa.discard_pending(tool_gate.SUBSYSTEM, pending_id)
        card_id = rec.get("card_id")
        if card_id:
            try:
                from hermes_cli import kanban_db as kb
                with kb.connect_closing() as conn:
                    kb.archive_task(conn, card_id)
            except Exception:
                logger.warning("reject: could not archive approval card %s", card_id)
    except Exception:
        logger.exception("POST /v1/actions/%s/reject failed", pending_id)
        return _err("Failed to reject action", status=500, code="server_error")
    return web.json_response({
        "ok": True,
        "message": f"Rejected and discarded action {pending_id}.",
    })

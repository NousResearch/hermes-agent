"""Mattermost interactive approval-button callback logic (transport-free).

Mattermost message buttons POST to an HTTP *integration URL* (not over the
adapter websocket like Discord, nor via Bolt like Slack). This module holds
the decision logic independent of aiohttp so it is unit-testable without a
live server. The adapter (``adapter.py``) wires the HTTP request/response and
the network sends around these helpers.

Two button kinds share one endpoint:

* ``kind=thread`` — inline approval: resolve a parked agent thread via
  ``resolve_gateway_approval(session_key, choice)``.
* ``kind=card`` — deferred approval: approve a staged action card via
  ``tool_gate.approve_action(pending_id)`` (or discard it on deny), which
  transitions the Kanban card + triggers execution-on-approval.

Security (Mattermost signs nothing by default):

* **Authorization** — the posting ``user_id`` must be in
  ``MATTERMOST_ALLOWED_USERS`` (mirrors Slack's interactive-user check).
* **Origin** — each prompt embeds a per-prompt shared-secret ``token`` in the
  button ``context``; the callback rejects a mismatch.
* **Double-click** — an atomic pop keyed by ``post_id`` ensures only the first
  click resolves.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Set

CHOICE_LABELS = {
    "once": "✅ Approved once",
    "session": "✅ Approved for session",
    "always": "✅ Approved permanently",
    "deny": "❌ Denied",
}


def is_user_authorized(user_id: str, allowed_users: Set[str]) -> bool:
    """Return whether ``user_id`` may resolve approvals.

    ``allowed_users`` is the parsed MATTERMOST_ALLOWED_USERS set. ``*`` allows
    anyone; an empty set denies everyone (fail closed — buttons must not be a
    bypass for an un-configured allowlist).
    """
    uid = str(user_id or "").strip()
    if not uid:
        return False
    if "*" in allowed_users:
        return True
    return uid in allowed_users


def build_action(name: str, callback_url: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Build one Mattermost message-button action descriptor.

    ``type: "button"`` is required by Mattermost; omitting it logs an
    "invalid action type" warning and is rejected by newer server versions.
    """
    return {
        "name": name,
        "type": "button",
        "integration": {
            "url": callback_url,
            "context": context,
        },
    }


def build_approval_attachment(*, text: str, callback_url: str, kind: str,
                              token: str, post_ref: str,
                              session_key: str = "", pending_id: str = "",
                              include_scopes: bool = True) -> Dict[str, Any]:
    """Build the ``attachments[]`` entry carrying the approve/deny buttons.

    ``post_ref`` is an opaque id used for the double-click guard; the adapter
    backfills the real post id once Mattermost returns it (Mattermost requires
    the buttons before the post id exists, so we key on a pre-minted ref).
    """
    base_ctx = {
        "kind": kind,
        "token": token,
        "post_ref": post_ref,
    }
    if session_key:
        base_ctx["session_key"] = session_key
    if pending_id:
        base_ctx["pending_id"] = pending_id

    actions = [build_action("Approve", callback_url, {**base_ctx, "choice": "once"})]
    if include_scopes and kind == "thread":
        actions.append(build_action("Approve (session)", callback_url, {**base_ctx, "choice": "session"}))
        actions.append(build_action("Approve (always)", callback_url, {**base_ctx, "choice": "always"}))
    actions.append(build_action("Deny", callback_url, {**base_ctx, "choice": "deny"}))

    return {"text": text, "actions": actions}


def handle_callback(
    context: Dict[str, Any],
    posted_user_id: str,
    *,
    allowed_users: Set[str],
    expected_secret_for: Callable[[str], Optional[str]],
    resolved_store: Dict[str, bool],
    resolve_fn: Optional[Callable[[str, str], Any]] = None,
    approve_fn: Optional[Callable[[str], Any]] = None,
    discard_fn: Optional[Callable[[str], Any]] = None,
) -> Dict[str, Any]:
    """Resolve one button click. Pure except for the injected callables.

    Returns ``{"ok": bool, "status": str, "update_text": str|None}``. The
    adapter turns ``update_text`` into a post edit (removing the buttons) and
    maps ``ok``/``status`` to an HTTP status code. Never raises for expected
    failures.
    """
    post_ref = str(context.get("post_ref") or "")
    choice = str(context.get("choice") or "deny")
    kind = str(context.get("kind") or "thread")
    token = str(context.get("token") or "")

    # 1. Authorize the clicking user.
    if not is_user_authorized(posted_user_id, allowed_users):
        return {"ok": False, "status": "unauthorized", "update_text": None}

    # 2. Verify the per-prompt shared secret bound to this post.
    expected = expected_secret_for(post_ref)
    if not expected or token != expected:
        return {"ok": False, "status": "bad_token", "update_text": None}

    # 3. Double-click guard — atomic pop; first caller gets False (proceed),
    #    any later caller gets the True default and is ignored.
    if resolved_store.pop(post_ref, True):
        return {"ok": False, "status": "already_resolved", "update_text": None}

    # 4. Dispatch.
    who = f" by {posted_user_id}" if posted_user_id else ""
    if kind == "card":
        pending_id = str(context.get("pending_id") or "")
        if choice == "deny":
            if discard_fn is not None:
                try:
                    discard_fn(pending_id)
                except Exception:
                    pass
            label = CHOICE_LABELS["deny"]
        else:
            ok = True
            if approve_fn is not None:
                try:
                    res = approve_fn(pending_id)
                    ok = bool(res.get("ok", True)) if isinstance(res, dict) else True
                except Exception:
                    ok = False
            label = ("✅ Approved — queued for execution" if ok
                     else "⚠️ Approval failed (see logs)")
        return {"ok": True, "status": "resolved", "update_text": f"{label}{who}."}

    # kind == "thread": resolve the parked inline approval.
    session_key = str(context.get("session_key") or "")
    if resolve_fn is not None:
        try:
            resolve_fn(session_key, choice)
        except Exception:
            return {"ok": False, "status": "resolve_failed", "update_text": None}
    label = CHOICE_LABELS.get(choice, "Resolved")
    return {"ok": True, "status": "resolved", "update_text": f"{label}{who}."}

"""Who may answer a pending exec approval from a messaging chat.

An approval belongs to the user whose message started the turn that raised it.
``tools.approval_gateway_wait`` records ``requester_id`` / ``requester_name`` and the
profile's ``approvals.approvers`` on the pending entry (in the turn's profile scope).
In a shared chat any admitted member sees the Allow/Deny buttons, so every surface
that resolves an approval from a chat event (button callbacks, reactions, ``/approve``
and ``/deny``, relay prompt responses) asks :func:`approval_refusal` first.

``approvals.approvers`` defaults to ``["*"]``: any admitted member answers, as before.
Narrowing it (``[]``, or a list of chat user ids) is the opt-in: then only the requester,
the listed ids and the platform's admins (``allow_admin_from`` / ``group_allow_admin_from``)
answer. Host surfaces (CLI, TUI, Desktop, the API server and the session authority's
``session:approve`` principals) answer as the host and never come through here.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional


def admin_ids_from_extra(extra: Any) -> frozenset[str]:
    """Both admin scopes of one platform's ``extra`` block."""
    from gateway.slash_access import _coerce_id_list

    if not isinstance(extra, dict):
        return frozenset()
    return _coerce_id_list(extra.get("allow_admin_from")) | _coerce_id_list(extra.get("group_allow_admin_from"))


def runner_admin_ids(source: Any) -> frozenset[str]:
    """Admins of the profile whose bot received *source*: the same policy the text ``/approve`` uses."""
    from gateway.run import _gateway_runner_ref
    from gateway.slash_access import policy_for_runner_source

    return policy_for_runner_source(_gateway_runner_ref(), source).admin_user_ids


def approval_refusal(session_key: str, actor_id: Any, *, admin_ids: Iterable[str] = (),
                     request_id: Optional[str] = None, extra: Any = None) -> Optional[str]:
    """``None`` when *actor_id* may answer *session_key*'s pending approval (the one matching
    *request_id*, else every pending one); otherwise the requester's display name for an
    "only X can answer this" reply. Nothing pending, or no requester recorded, is never refused:
    the resolver then reports "nothing pending" exactly as before. Adapters pass their platform
    ``extra`` block instead of *admin_ids*."""
    from gateway.slash_access import _coerce_id_list
    from tools.approval import list_gateway_approval_authority

    actor = str(actor_id or "").strip()
    admins = {str(a) for a in admin_ids} | admin_ids_from_extra(extra)
    for pending in list_gateway_approval_authority(session_key):
        if request_id and pending.get("request_id") != request_id:
            continue
        requester = str(pending.get("requester_id") or "")
        if not requester or actor == requester:
            continue
        approvers = _coerce_id_list(pending.get("approvers"))
        if "*" in approvers or (actor and (actor in approvers or actor in admins)):
            continue
        return str(pending.get("requester_name") or requester)
    return None


def refusal_text(requester: str) -> str:
    from agent.i18n import t

    return t("gateway.exec_approval.not_requester", requester=requester)

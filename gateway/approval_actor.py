"""Who may answer a pending exec approval from a messaging chat.

An approval belongs to the user whose message started the turn that raised it.
``tools.approval_gateway_wait`` records ``requester_id`` / ``requester_name`` and the
profile's ``approvals.approvers`` on the pending entry (in the turn's profile scope).
In a shared chat any admitted member sees the Allow/Deny buttons, so every surface
that resolves an approval from a chat event (button callbacks, reactions, relay prompt
responses) asks :func:`approval_refusal` first; ``/approve`` and ``/deny`` use
:func:`answerable_request_ids` and resolve exactly the ids it checked.

``approvals.approvers`` defaults to ``["*"]``: any admitted member answers, as before.
Narrowing it (``[]``, or a list of chat user ids) is the opt-in: then only the requester,
the listed ids and the platform's admins of the approval's chat scope (``allow_admin_from`` in a
DM, ``group_allow_admin_from`` in a group, as for slash commands) answer. Host surfaces (CLI, TUI, Desktop, the API server and the session authority's
``session:approve`` principals) answer as the host and never come through here.

The other prompts a chat member can answer for someone else follow the same rule through the same
:func:`may_answer`: a slash-command confirmation (:func:`slash_confirm_refusal`) and the agent's
clarify question (:func:`clarify_refusal`), each recording its authority when raised.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional


# Refusal "name" of an entry with no requester recorded; ``refusal_text`` words it as admins-only.
_NO_REQUESTER = "\0no-requester"


def may_answer(authority: dict, actor_id: Any, admin_ids: Iterable[str] = (), extra: Any = None) -> bool:
    """Whether *actor_id* may answer the approval whose recorded ``authority`` this is: the requester,
    a listed approver, or an admin. Admins come from ONE source, never a union: a native surface's
    platform ``extra`` gives the admins of the approval's own chat scope (DM or group,
    ``slash_access.policy_for_chat_type``), the same scope ``/approve`` and relay answers pass as
    ``admin_ids``. With ``approvers`` narrowed, an entry with no requester recorded (a turn merged
    from two senders) is answered by approvers and admins only, never by anyone."""
    from gateway.slash_access import _coerce_id_list, policy_for_chat_type

    actor = str(actor_id or "").strip()
    requester = str(authority.get("requester_id") or "")
    approvers = _coerce_id_list(authority.get("approvers"))
    if "*" in approvers or (requester and actor == requester):
        return True
    if not actor:
        return False
    admins = ({str(a) for a in admin_ids} if extra is None
              else policy_for_chat_type(extra, authority.get("chat_type")).admin_user_ids)
    return actor in approvers or actor in admins


def runner_admin_ids(source: Any) -> frozenset[str]:
    """Admins of the profile whose bot received *source*: the same policy the text ``/approve`` uses."""
    from gateway.run import _gateway_runner_ref
    from gateway.slash_access import policy_for_runner_source

    return policy_for_runner_source(_gateway_runner_ref(), source).admin_user_ids


def answerable_request_ids(session_key: str, actor_id: Any, *, admin_ids: Iterable[str] = (),
                           request_id: Optional[str] = None, resolve_all: bool = False,
                           extra: Any = None) -> tuple[list[str], Optional[str]]:
    """The pending approvals ``resolve_gateway_approval`` would select (the one matching
    *request_id*, every one with *resolve_all*, else the oldest) and whether *actor_id* may
    answer them: ``(request_ids, None)``, or ``([], requester_name)`` for the first selected one
    they may not. Only the selected entries are checked, so another requester's queued approval
    never blocks this one. Resolving the returned ids (not "the oldest" again) keeps a refused
    entry that became the oldest in between from being answered unchecked."""
    from tools.approval import list_gateway_approval_authority

    pending = list_gateway_approval_authority(session_key)
    if request_id:
        pending = [p for p in pending if p.get("request_id") == request_id]
    elif not resolve_all:
        pending = pending[:1]
    for entry in pending:
        if not may_answer(entry, actor_id, admin_ids, extra):
            return [], str(entry.get("requester_name") or entry.get("requester_id") or _NO_REQUESTER)
    return [str(p.get("request_id")) for p in pending], None


def approval_refusal(session_key: str, actor_id: Any, *, admin_ids: Iterable[str] = (),
                     request_id: Optional[str] = None, extra: Any = None) -> Optional[str]:
    """``None`` when *actor_id* may answer the approval a click resolves (the one matching
    *request_id*, else the oldest); otherwise the requester's display name for an "only X can
    answer this" reply. Nothing pending is never refused: the resolver then reports "nothing
    pending" exactly as before. Adapters pass their platform ``extra`` block instead of *admin_ids*."""
    return answerable_request_ids(session_key, actor_id, admin_ids=admin_ids,
                                  request_id=request_id, extra=extra)[1]


def _authority_refusal(authority: Optional[dict], actor_id: Any, admin_ids: Iterable[str], extra: Any) -> Optional[str]:
    if not authority or may_answer(authority, actor_id, admin_ids, extra):
        return None
    return str(authority.get("requester_name") or authority.get("requester_id") or _NO_REQUESTER)


def slash_confirm_refusal(session_key: str, confirm_id: str, actor_id: Any, *, admin_ids: Iterable[str] = (),
                          extra: Any = None) -> Optional[str]:
    """:func:`approval_refusal` for the slash-command confirmation (``/reload-mcp``) a click or typed
    reply would resolve: the same requester / approvers / admins rule, read from the pending confirm.
    Nothing pending (or another confirm id) is never refused: the resolver reports it as before."""
    from tools.slash_confirm import get_pending

    pending = get_pending(session_key) or {}
    return _authority_refusal(pending.get("authority") if pending.get("confirm_id") == confirm_id else None,
                              actor_id, admin_ids, extra)


def clarify_refusal(clarify_id: str, actor_id: Any, *, admin_ids: Iterable[str] = (),
                    extra: Any = None) -> Optional[str]:
    """:func:`approval_refusal` for the agent's clarify question a tap (choice or "Other") answers."""
    from tools.clarify_gateway import get_pending

    return _authority_refusal(getattr(get_pending(clarify_id), "authority", None), actor_id, admin_ids, extra)


def refusal_text(requester: str) -> str:
    from agent.i18n import t

    if requester == _NO_REQUESTER:
        return t("gateway.exec_approval.not_requester_admins_only")
    return t("gateway.exec_approval.not_requester", requester=requester)

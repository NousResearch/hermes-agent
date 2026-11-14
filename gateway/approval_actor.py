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
DM, ``group_allow_admin_from`` in a group, as for slash commands) answer.

Under the default the chat's slash policy fills in: answering an approval IS ``/approve`` /
``/deny`` and answering a slash-command confirmation runs that command, so where the scope sets an
admin list (``allow_admin_from``) only someone allowed to run that command there (an admin, or
``user_allowed_commands``) answers, whichever way they answer. A narrowed ``approvers`` is the
explicit approval setting and takes precedence. Clarify answers run no command. Host surfaces (CLI,
TUI, Desktop, the API server and the session authority's ``session:approve`` principals) answer as
the host and never come through here.

The other prompts a chat member can answer for someone else follow the same rule through the same
:func:`may_answer`: a slash-command confirmation (:func:`slash_confirm_refusal`) and the agent's
clarify question (:func:`clarify_refusal`), each recording its authority when raised.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional


# Refusal "name" of an entry with no requester recorded; ``refusal_text`` words it as admins-only.
_NO_REQUESTER = "\0no-requester"
# Refusal "name" prefix when the chat's slash policy refused the answer's command (``+ command``).
_ADMIN_ONLY = "\0admin-only:"


def may_answer(authority: dict, actor_id: Any, admin_ids: Any = (), extra: Any = None,
               commands: tuple[str, ...] = (), *, admitted: bool = True) -> bool:
    """Whether *actor_id* may answer the prompt whose recorded ``authority`` this is.

    ``approvers`` narrowed (``[]`` or ids): the requester, a listed approver or an admin; that explicit
    approval setting decides. ``approvers: ["*"]`` (the default): anyone the chat's slash policy lets
    run one of *commands*, the answer's own command (``approve`` / ``deny``, or a slash confirm's
    command): everyone admitted where the scope sets no admin list, else admins and
    ``user_allowed_commands``, exactly as if they had typed it. Clarify passes no *commands*.

    The policy comes from ONE source, never a union: a native surface's platform ``extra`` gives the
    policy of the prompt's own chat scope (DM or group, ``slash_access.policy_for_chat_type``); a text or
    relay answer passes the policy of the chat it arrived in (:func:`runner_answer_policy`) as
    *admin_ids*. A bare id list in *admin_ids* names admins with no slash floor. With ``approvers``
    narrowed, an entry with no requester recorded (a turn merged from two senders) is answered by
    approvers and admins only, never by anyone.

    ``admitted=False`` marks an answer whose sender no allowlist admitted (a relayed Discord sender the
    connector never delivered a message from, ``gateway.relay.admitted_authors``): under ``"*"`` the
    sender must then pass the slash floor AND be the requester or an admin, as with ``approvers`` narrowed."""
    return _refusal(authority, actor_id, admin_ids, extra, commands, admitted) is None


def _refusal(authority: dict, actor_id: Any, admin_ids: Any, extra: Any, commands: tuple[str, ...],
             admitted: bool) -> Optional[str]:
    """:func:`may_answer`'s decision: ``None`` when allowed, else what the refusal names, taken from
    the check that failed: the answer's admin-only command when the slash policy refused it, else the
    requester."""
    from gateway.slash_access import SlashAccessPolicy, _coerce_id_list, policy_for_chat_type

    actor = str(actor_id or "").strip()
    if extra is not None:
        policy = policy_for_chat_type(extra, authority.get("chat_type"))
    elif isinstance(admin_ids, SlashAccessPolicy):
        policy = admin_ids
    else:
        policy = SlashAccessPolicy(False, frozenset(str(a) for a in admin_ids), frozenset())
    approvers = _coerce_id_list(authority.get("approvers"))
    if "*" in approvers:
        if commands and policy.enabled and not any(policy.can_run(actor or None, c) for c in commands):
            return _ADMIN_ONLY + commands[0]
        if admitted:
            return None
    requester = str(authority.get("requester_id") or "")
    if actor and (actor == requester or actor in approvers or actor in policy.admin_user_ids):
        return None
    return str(authority.get("requester_name") or requester or _NO_REQUESTER)


def approval_commands(choice: Optional[str] = None) -> tuple[str, ...]:
    """The slash command an exec-approval answer is: ``deny`` for a deny, else ``approve``; either
    when the choice is not known yet (a pre-check before the settle rechecks the exact one).
    ``SlashAccessPolicy.can_run`` lets an ``approve`` holder run ``deny`` too."""
    if choice is None:
        return ("approve", "deny")
    return ("deny",) if choice == "deny" else ("approve",)


def runner_answer_policy(source: Any):
    """Slash policy of the profile whose bot received *source*: the one the text ``/approve`` uses."""
    from gateway.run import _gateway_runner_ref
    from gateway.slash_access import policy_for_runner_source

    return policy_for_runner_source(_gateway_runner_ref(), source)


def answerable_request_ids(session_key: str, actor_id: Any, *, admin_ids: Any = (),
                           request_id: Optional[str] = None, resolve_all: bool = False,
                           extra: Any = None, choice: Optional[str] = None,
                           admitted: bool = True) -> tuple[list[str], Optional[str]]:
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
    commands = approval_commands(choice)
    for entry in pending:
        refused = _refusal(entry, actor_id, admin_ids, extra, commands, admitted)
        if refused is not None:
            return [], refused
    return [str(p.get("request_id")) for p in pending], None


def approval_refusal(session_key: str, actor_id: Any, *, admin_ids: Any = (),
                     request_id: Optional[str] = None, extra: Any = None,
                     choice: Optional[str] = None, admitted: bool = True) -> Optional[str]:
    """``None`` when *actor_id* may answer the approval a click resolves (the one matching
    *request_id*, else the oldest); otherwise the requester's display name for an "only X can
    answer this" reply. Nothing pending is never refused: the resolver then reports "nothing
    pending" exactly as before. Adapters pass their platform ``extra`` block instead of *admin_ids*,
    and the clicked *choice* so the slash floor checks that answer's own command."""
    return answerable_request_ids(session_key, actor_id, admin_ids=admin_ids,
                                  request_id=request_id, extra=extra, choice=choice, admitted=admitted)[1]


def _authority_refusal(authority: Any, actor_id: Any, admin_ids: Any, extra: Any,
                       commands: tuple[str, ...] = (), admitted: bool = True) -> Optional[str]:
    # An empty authority ({}) records no restriction: anyone admitted answers, as on main. An
    # unadmitted sender (``may_answer``'s ``admitted``) needs a recorded requester to match.
    if not authority or not isinstance(authority, dict):
        return None if admitted else _NO_REQUESTER
    return _refusal(authority, actor_id, admin_ids, extra, commands, admitted)


def slash_confirm_refusal(session_key: str, confirm_id: str, actor_id: Any, *, admin_ids: Any = (),
                          extra: Any = None, admitted: bool = True) -> Optional[str]:
    """:func:`approval_refusal` for the slash-command confirmation (``/reload-mcp``) a click or typed
    reply would resolve: the same rule, read from the pending confirm, with the confirmed command as
    the slash floor (answering runs it). Nothing pending (or another confirm id) is never refused: the
    resolver reports it as before."""
    from hermes_cli.commands import resolve_command
    from tools.slash_confirm import get_pending

    pending = get_pending(session_key) or {}
    if pending.get("confirm_id") != confirm_id:
        return None
    command = str(pending.get("command") or "")
    cmd_def = resolve_command(command)
    return _authority_refusal(pending.get("authority"), actor_id, admin_ids, extra,
                              (cmd_def.name if cmd_def else command.lstrip("/"),), admitted)


def clarify_refusal(clarify_id: str, actor_id: Any, *, admin_ids: Any = (),
                    extra: Any = None, admitted: bool = True) -> Optional[str]:
    """:func:`approval_refusal` for the agent's clarify question a tap (choice or "Other") answers."""
    from tools.clarify_gateway import get_pending

    pending = get_pending(clarify_id)
    if pending is None:
        return None
    return _authority_refusal(getattr(pending, "authority", None), actor_id, admin_ids, extra, (), admitted)


def refusal_text(requester: str) -> str:
    from agent.i18n import t

    if requester == _NO_REQUESTER:
        return t("gateway.exec_approval.not_requester_admins_only")
    if requester.startswith(_ADMIN_ONLY):
        return t("gateway.unauthorized.admin_only", command=requester[len(_ADMIN_ONLY):], suffix="").strip()
    return t("gateway.exec_approval.not_requester", requester=requester)


def opener_authority(event: Any) -> Optional[dict]:
    """Who may answer a picker card (``/model``, ``/reasoning``...) or the ``/update`` prompt *event*'s
    sender opened: the opener, or an admin of the chat scope (``allow_admin_from`` /
    ``group_allow_admin_from``), so a member refused an admin-gated command cannot apply an admin's
    open card instead, and nobody else's tap or typed text answers the opener's prompt.
    ``None`` (anyone admitted answers, as before) when the sender is unknown, e.g. a shared observed
    group whose route carries no user, so a card is never left with nobody able to answer it."""
    source = getattr(event, "source", None)
    actor = getattr(event, "user_id", None) or getattr(source, "user_id", None)
    if not actor:
        return None
    return {"requester_id": str(actor),
            "requester_name": str(getattr(event, "user_name", None) or getattr(source, "user_name", None) or actor),
            "chat_type": str(getattr(source, "chat_type", "") or ""), "approvers": []}


def picker_refusal(on_selected: Any, actor_id: Any, *, extra: Any = None) -> Optional[str]:
    """``None`` when *actor_id* may use the picker card whose selection callback is *on_selected*;
    otherwise the opener's name for :func:`refusal_text`. The gateway sets the opener's
    :func:`opener_authority` on the callback as ``prompt_authority``, so every adapter's picker
    state already carries it; a callback without one is never refused. Adapters pass their
    platform ``extra`` block (its admins answer any card of that chat scope)."""
    return _authority_refusal(getattr(on_selected, "prompt_authority", None), actor_id, (), extra)


def update_prompt_refusal(actor_id: Any, *, admin_ids: Iterable[str] = (), extra: Any = None) -> Optional[str]:
    """:func:`picker_refusal` for the pending ``/update`` Yes/No prompt: the authority ``/update``
    recorded in its pending marker. No marker or no authority in it is never refused."""
    import json

    from hermes_constants import get_process_hermes_home  # where /update wrote it (gateway.run._hermes_home)

    for name in (".update_pending.claimed.json", ".update_pending.json"):
        try:
            marker = json.loads((get_process_hermes_home() / name).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        return _authority_refusal(marker.get("authority") if isinstance(marker, dict) else None, actor_id, admin_ids, extra)
    return None

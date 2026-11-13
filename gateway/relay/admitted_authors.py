"""Discord authors the relay connector admitted to this gateway.

The connector delivers a Discord message (the inbound lane) only to the instances whose per-author
check admits its author, but forwards every Discord interaction (button press, application command,
modal) to all gateways of the tenant unchecked. A relayed Discord sender therefore answers a prompt as
admitted only once one of their messages reached this gateway; anyone else is unadmitted
(``approval_actor.may_answer``), and their interaction reaches the runner only as a prompt answer or
from a configured admin (:func:`interaction_admitted`).

The relay adapter is process-level, so is this record: under multiplexing one connector socket serves
every profile, and the connector admits an author per gateway instance, not per profile route.
"""

from collections import OrderedDict
from typing import Any

_MAX = 4096
# What an unadmitted sender's interaction may still run: the prompt answers, which the runner limits to
# the requester, a listed approver or an admin (``approval_actor.may_answer``).
_UNADMITTED_COMMANDS = frozenset({"approve", "deny"})
_admitted: "OrderedDict[str, None]" = OrderedDict()


def _relayed_discord(source: Any) -> bool:
    from gateway.config import Platform

    return (getattr(source, "delivered_via_upstream_relay", False) is True
            and getattr(source, "platform", None) == Platform.DISCORD)


def note_delivered(source: Any) -> None:
    """The connector delivered a message from *source*'s author to this gateway: they are admitted."""
    if _relayed_discord(source) and source.user_id:
        _admitted[str(source.user_id)] = None
        _admitted.move_to_end(str(source.user_id))
        while len(_admitted) > _MAX:
            _admitted.popitem(last=False)


def answer_admitted(event: Any) -> bool:
    """Whether *event*'s sender passed an allowlist (``may_answer``'s ``admitted``): every sender but a
    relayed Discord one whose messages never reached this gateway (a merged turn's ``""`` included)."""
    source = getattr(event, "source", None)
    if not _relayed_discord(source):
        return True
    sender = event.sender_id()
    return bool(sender) and str(sender) in _admitted


def interaction_admitted(event: Any, admin_ids: Any = ()) -> bool:
    """Whether a relayed Discord interaction (application command, component press, modal) may reach the
    runner: from an admitted sender or a configured admin (*admin_ids*), else only as ``/approve`` or
    ``/deny``. Anything else is dropped unanswered: the connector forwards it to every gateway of the
    tenant, so a refusal from each one that never admitted the sender would be noise."""
    if answer_admitted(event):
        return True
    sender = str(event.sender_id() or "")
    if sender and sender in {str(a) for a in admin_ids}:
        return True
    from hermes_cli.commands import resolve_command

    command = resolve_command(event.get_command() or "") if event.get_command() else None
    return command is not None and command.name in _UNADMITTED_COMMANDS

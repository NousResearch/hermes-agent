"""``hermes groups`` — a Group Chat whose host went offline: status, continue, keep, move and backup copies.

Thin wrappers over this computer's gateway (``groups.succession.*`` and ``groups.custody.*``).
``continue`` always runs on the computer the group should continue on: this one; ``move`` runs on the
host and hands the group over. The words are the product's own: the host, a backup copy, continue
the group.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from typing import Callable

_MESSAGES = {
    "not_owner": "Only the group's owner can do that.",
    "host_reachable": "The group's host can be reached, so the group doesn't need to move.",
    "room_authority_promised": "Another computer is already continuing this group.",
    "preview_stale": "Something changed since the summary. Run the command again.",
    "target_not_ready": "This computer can't continue the group: it isn't allowed to, or keeps no usable copy.",
    "target_not_local": "Run this on the computer the group should continue on.",
    "room_not_found": "This computer has no group by that name.",
    "room_authority_conflict": "This group was continued on two computers. Choose one with `hermes groups keep`.",
    "room_host_paused": "The group is paused to stay safe until its host reaches its other computers again.",
}


# Calls that move a group can take longer than the client's usual 30 s: fencing, catching up and the
# standby's answer.
_SLOW_CALLS = {"groups.succession.promote": 300.0, "groups.succession.move": 300.0,
               "groups.succession.move_now": 300.0, "groups.succession.keep": 300.0}
# How long ``continue`` keeps checking a move that is still finishing before it leaves it to run.
_MOVING_WAIT_SECONDS = 300


class GroupCommandError(RuntimeError):
    pass


def _call(method: str, **params):
    from hermes_cli.gateway_client import GatewayClientError, connect_gateway

    async def run():
        async with connect_gateway() as client:
            return await client.rpc(method, _timeout=_SLOW_CALLS.get(method, 30.0), **params)

    try:
        return asyncio.run(run())
    except GatewayClientError as exc:
        raise GroupCommandError(_MESSAGES.get(str(exc), str(exc))) from exc


def _room_id(name: str) -> str:
    """A group's room id from its id or its name (exact, any case), copies included."""
    rooms = _call("groups.list").get("rooms") or []
    for room in rooms:
        if room.get("room_id") == name:
            return name
    matches = [room["room_id"] for room in rooms if str(room.get("name") or "").casefold() == name.casefold()]
    if len(matches) == 1:
        return matches[0]
    raise GroupCommandError(_MESSAGES["room_not_found"] if not matches else
                            f"More than one group is called “{name}”; use its id.")


def _name(item, fallback: str = "another computer") -> str:
    """A computer's name, or a plain word for one whose name isn't known: never its install id."""
    return (item or {}).get("name") or fallback


def _computers(status: dict) -> dict[str, str]:
    """Install id → display name for every computer the status names."""
    named = {row["install_id"]: _name(row) for row in status.get("backups") or ()}
    host, here = status.get("host") or {}, status.get("this_install") or {}
    if host.get("install_id"):
        named[host["install_id"]] = _name(host, "the host")
    if here.get("install_id"):
        named[here["install_id"]] = _name(here, "this computer")
    return named


def _computer(status: dict, value: str) -> str:
    if value in {"here", "this"}:
        return status["this_install"]["install_id"]
    for install_id, name in _computers(status).items():
        if value in {install_id, name} or value.casefold() == str(name).casefold():
            return install_id
    raise GroupCommandError(f"This group has no computer called “{value}”.")


def _plural(count: int, one: str, many: str) -> str:
    return one if count == 1 else many


def _action(status: dict, name: str) -> dict | None:
    return next((action for action in status.get("actions") or () if action.get("action") == name), None)


def _when(value) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(value))) if value else "an unknown time"


def _listed(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def readiness_line(status: dict) -> str | None:
    """Whether the group keeps running by itself if its host goes offline."""
    automatic = status.get("automatic") or {}
    host = _name(status["host"], "the host")
    standby = _name(automatic.get("standby"))
    state = automatic.get("state")
    if automatic.get("pending") is False:
        return "Turning off… (waiting for the other computers)"
    if automatic.get("pending") is True:
        return "Turning on… (waiting for the other computers)"
    if state == "ready":
        when = "within a minute" if automatic.get("mode") == "majority" else "after about 3 minutes"
        return f"Keeps running on its own. If {host} goes offline, {standby} takes over {when}."
    if state == "not_ready":
        renewing = {row["install_id"] for row in status.get("backups") or ()
                    if row.get("readiness") == "needs_reauthorization"}
        listed = automatic.get("offline") or ()
        offline = [_name(item) for item in listed if item.get("install_id") not in renewing]
        reconnect = [_name(item) for item in listed if item.get("install_id") in renewing]
        parts = []
        if offline or not reconnect:
            parts.append(f"{_listed(offline) if offline else 'its other computers'} "
                         f"{'is' if len(offline) == 1 else 'are'} offline")
        if reconnect:
            parts.append(f"{_listed(reconnect)} {'needs' if len(reconnect) == 1 else 'need'} to be reconnected")
        return (f"Right now {' and '.join(parts)}. If {host} goes offline before then, you'll be asked where to "
                "continue.")
    if state == "unavailable":
        needed = int(automatic.get("needed") or 1)
        return (f"If {host} goes offline, you'll be asked where to continue it. Add {needed} more always-on "
                f"computer{'s' if needed != 1 else ''} to make this automatic.")
    if state == "off":
        return "Moves only when you choose."
    return None


def _paused_line(status: dict, host: str) -> str:
    paused = status.get("paused") or {}
    if paused.get("reason") == "no_lease_layer":
        return (f"{host} can't take part in automatic moves right now, so it paused to stay safe. Its connection to "
                "the other computers isn't ready.")
    waiting = [_name(item) for item in paused.get("waiting_for") or ()]
    if paused.get("reason") == "step_not_taken":
        who = _listed(waiting) if waiting else "every computer that could continue it"
        return (f"Paused to stay safe: its next step was promised to another computer, which never took it. "
                f"{host} resumes once {who} confirms nothing else happened, or you can continue it anyway in "
                "Hermes Desktop.")
    return (f"Paused to stay safe: {host} can't reach {_listed(waiting) if waiting else 'its other computers'}. "
            "It resumes as soon as one of them is back.")


def _moving_lines(status: dict, host: str) -> list[str]:
    moving = status.get("moving") or {}
    to = _name(moving.get("to"))
    if moving.get("step") == "waiting_for_turns":
        lines = [f"Moving to {to} after the replies in progress finish ({int(moving.get('running') or 0)})."]
        if _action(status, "move_now"):
            lines.append(f"To move now: `hermes groups move <group> \"{to}\" --now`.")
        return lines
    if moving.get("reason") == "automatic":
        return [f"{host} went offline. Moving to {to}… ({moving.get('step')})"]
    if moving.get("reason") == "handover":
        return [f"Moving to {to}… ({moving.get('step')})"]
    return [f"Continuing on {to}… ({moving.get('step')})"]


def _conflict_lines(status: dict) -> list[str]:
    conflict = status.get("conflict") or {}
    hosts = conflict.get("hosts") or []
    running = conflict.get("running_on") or {}
    others = [item for item in hosts if item.get("install_id") != running.get("install_id")]
    if not running.get("install_id") or not others:
        names = [_name(item) for item in hosts]
        return [f"This group was continued on two computers: {' and '.join(names)}. "
                "Choose one with `hermes groups keep <group> <computer>`."]
    keep, other = _name(running), _name(others[0])
    return [f"This group was continued on two computers. {keep} is running the group; {other} stopped.",
            f"Keep going on {keep}: `hermes groups keep <group> \"{keep}\"`. "
            f"Switch to {other}: `hermes groups keep <group> \"{other}\"`."]


def _bot_lines(status: dict) -> list[str]:
    """The Bots that run only on another computer, and the way back to them."""
    movable = set((_action(status, "move") or {}).get("targets") or ())
    places: dict[str, dict] = {}
    for bot in status.get("unavailable_bots") or ():
        on = bot.get("on") or {}
        place = places.setdefault(on.get("install_id") or "", {"on": on, "names": []})
        place["names"].append(_name(bot, "a Bot"))
    lines = []
    for install_id, place in places.items():
        computer, names = _name(place["on"], "its computer"), _listed(place["names"])
        if place["on"].get("reachable") and install_id in movable:
            lines.append(f"{names} can take part again if the group moves back to {computer}. To move it back: "
                         f"`hermes groups move <group> \"{computer}\"`.")
        else:
            lines.append(f"{names}: unavailable until the group moves back to {computer}.")
    return lines


def status_lines(status: dict) -> list[str]:
    host = _name(status["host"], "the host")
    names = _computers(status)
    state, lines = status["state"], []
    if state == "paused":
        lines.append(_paused_line(status, host))
    elif state == "host_unreachable" and status.get("unavailable_reason") == "takeover_waiting":
        lines.append(f"{host} went offline. The other computers are deciding which one takes over; this can take a "
                     "few minutes.")
    elif state == "host_unreachable":
        lines.append(f"{host} is offline (last seen {_when(status['host'].get('since'))}). The group is paused.")
    elif state == "host_restarting":
        lines.append(f"{host} is restarting. The group will continue in a moment.")
    elif state == "moving":
        lines.extend(_moving_lines(status, host))
    elif state == "continued_on_two":
        lines.extend(_conflict_lines(status))
    elif state == "moved_away":
        moved = status.get("moved") or {}
        lines.append(f"This group moved to {_name(moved.get('to'))}. "
                     f"{moved.get('separate_events', 0)} messages from that time are kept separately.")
    else:
        lines.append(f"Hosted on {host}.")
    for row in status.get("backups") or ():
        name = _name(row)
        readiness = row["readiness"]
        detail = {"caught_up": "full copy, up to date", "behind": f"full copy, {row.get('behind_by')} messages behind",
                  "offline": f"offline since {_when(row.get('last_seen'))}", "unknown": "full copy, not confirmed yet",
                  "needs_reauthorization": "needs to be reconnected: its permission to keep a copy ran out",
                  "unsupported": "needs a newer Hermes to keep a copy of this group"}.get(readiness, readiness)
        lines.append(f"  {name}: {detail}{' (can continue this group)' if row.get('successor') else ''}")
    lines.extend(_bot_lines(status))
    moved_in = status.get("moved_in") or {}
    if moved_in.get("proof_kind") == "evidence":
        old = _name(moved_in.get("from"), "the old host")
        lines.append(f"{old} went silent for 3 minutes, so this computer took over. If {old} is actually still "
                     f"running, the group may now be running in both places. To go back: "
                     f"`hermes groups keep <group> {old}`.")
    targets = next((action["targets"] for action in status.get("actions") or () if action["action"] == "continue"),
                   None)
    readiness = readiness_line(status) if state == "ok" else None
    if targets:
        lines.append("Can continue on: " + ", ".join(names.get(item, "another computer") for item in targets))
    elif readiness and ((status.get("automatic") or {}).get("state") in {"ready", "not_ready", "unavailable"}
                        or (status.get("automatic") or {}).get("pending") is not None):
        lines.append(readiness)
    elif state == "ok":
        eligible = [_name(row) for row in status.get("backups") or () if row.get("successor")]
        lines.append(f"If {host} goes offline, you can continue this group on {', '.join(eligible)}." if eligible
                     else f"If {host} goes offline, this group pauses until it's back.")
        if readiness:
            lines.append(readiness)
    return lines


def cmd_status(args: argparse.Namespace) -> int:
    try:
        status = _call("groups.succession.status", room_id=_room_id(args.group))
    except GroupCommandError as exc:
        print(exc, file=sys.stderr)
        return 1
    print("\n".join(status_lines(status)))
    return 0


def _confirm(question: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    try:
        return input(f"{question} [y/N] ").strip().lower() in {"y", "yes"}
    except EOFError:
        return False


def _counted(caution: dict) -> tuple[str, int]:
    """Who a caution names: their names, else how many computers."""
    count = int(caution.get("count") or len(caution.get("names") or ()))
    names = [name for name in caution.get("names") or () if name]
    return (_listed(names) if names else f"{count} {_plural(count, 'computer', 'computers')}"), count


def summary_lines(group: str, host: str, preview: dict) -> list[str]:
    target = _name(preview["target"], "this computer")
    lines = [f"Continue “{group}” on {target}? {target} becomes the group's host. "
             "The conversation, members and history stay the same."]
    places: dict[str, list[str]] = {}
    for bot in preview.get("unavailable_bots") or ():
        places.setdefault(_name(bot.get("on"), host), []).append(_name(bot, bot.get("member_id") or "a Bot"))
    for computer, bots in places.items():
        lines.append(f"{len(bots)} {_plural(len(bots), 'Bot runs', 'Bots run')} on {computer} and "
                     f"{_plural(len(bots), 'stays', 'stay')} unavailable until the group moves back there: "
                     f"{', '.join(bots)}.")
    work = preview.get("work") or {}
    if any(work.values()):
        lines.append(f"Work in progress: {work.get('completed', 0)} finished, {work.get('elsewhere', 0)} still running "
                     f"on other computers, {work.get('unknown', 0)} unknown. Unknown work won't run again "
                     "automatically.")
    missing = int((preview.get("at_risk") or {}).get("count") or 0)  # only the host had them
    if missing:
        lines.append(f"{target} is missing {missing} recent {_plural(missing, 'message', 'messages')}. "
                     f"{_plural(missing, 'It', 'They')}'ll appear if {host} comes back.")
    behind = int(preview.get("behind_by") or 0)  # another computer has them: fetched while continuing
    if behind:
        lines.append(f"{target} is catching up {behind} {_plural(behind, 'message', 'messages')} from another "
                     "computer.")
    operator, owner = preview["target"].get("operator_name"), (preview.get("owner") or {}).get("name")
    if operator and owner and operator != owner:
        lines.append(f"{operator} will manage this group from {target}.")
    lines.append(f"If {host} comes back, it rejoins as a member. Anything it did while offline is shown separately.")
    for caution in preview.get("cautions") or ():
        who, count = _counted(caution)
        if caution.get("code") == "host_may_be_running":
            lines.append(f"Only continue if {host} is really offline. If it's still running somewhere you can't "
                         "reach, both computers may keep working until they reconnect, and you'll be asked to "
                         "choose one.")
        elif caution.get("code") == "participant_not_fenced" and count:
            lines.append(f"{who} {_plural(count, 'runs', 'run')} an older Hermes and may still accept work from "
                         f"{host} if it is still running.")
        elif caution.get("code") == "voters_unreachable" and count:
            lines.append(f"{who} can't be reached, so this computer can't confirm {host} has stopped. Continue only "
                         f"if {host} is really offline.")
    return lines


def cmd_continue(args: argparse.Namespace) -> int:
    try:
        room_id = _room_id(args.group)
        status = _call("groups.succession.status", room_id=room_id)
        here = status["this_install"]["install_id"]
        names = _computers(status)
        if args.on:
            target = _computer(status, args.on)
            if target != here:
                raise GroupCommandError(f"Run this on {names.get(target, target)}: a group continues on the computer "
                                        "that runs the command.")
        targets = next((action["targets"] for action in status.get("actions") or ()
                        if action["action"] == "continue"), [])
        if not args.on and targets and targets[0] != here and not _confirm(
                f"{names.get(targets[0], targets[0])} is better placed to continue this group. Continue here anyway?",
                args.yes):
            return 1
        preview = _call("groups.succession.prepare", room_id=room_id, target_install_id=here)
        host = _name(status["host"], "the host")
        print("\n".join(summary_lines(args.group, host, preview)))
        if not _confirm("Continue?", args.yes):
            return 1
        result = _call("groups.succession.promote", room_id=room_id, target_install_id=here,
                       preview_id=preview["preview_id"], confirm=True)
        for _ in range(_MOVING_WAIT_SECONDS):
            if result["state"] != "moving":
                break
            time.sleep(1)
            result = _call("groups.succession.status", room_id=room_id)
        else:
            print(f"Still finishing the move. Check with `hermes groups status \"{args.group}\"`.")
            return 0
    except GroupCommandError as exc:
        print(exc, file=sys.stderr)
        return 1
    work = result.get("work") or {}
    unknown, waiting = int(work.get("unknown") or 0), int(work.get("waiting_for_host") or 0)
    print(f"Done. “{args.group}” now continues on {_name(result['this_install'], 'this computer')}.")
    if unknown or waiting:
        print(f"Work in progress: {unknown} unknown, {waiting} waiting for "
              f"{_name(result.get('previous_host'), 'the old host')}. Unknown work won't run again automatically.")
    return 0


def keep_question(group: str, status: dict, target: str) -> str:
    """What keeping ``target`` means, before the owner confirms it."""
    names = _computers(status)
    name = names.get(target, "that computer")
    if status["state"] == "continued_on_two":
        running = ((status.get("conflict") or {}).get("running_on") or {}).get("install_id")
        others = [item for item in (status.get("conflict") or {}).get("hosts") or ()
                  if item.get("install_id") != target]
        other = _name(others[0] if others else None)
        if running == target:
            return (f"Keep going on {name}? Messages written only on {other} while the two were apart are kept "
                    "separately, not mixed in.")
        return (f"Switch “{group}” to {name}? It continues there, and messages written only on {other} while the two "
                "were apart are kept separately, not mixed in.")
    return (f"Go back to {name}? “{group}” continues there, and messages written here since the move are kept "
            "separately, not mixed in.")


def cmd_keep(args: argparse.Namespace) -> int:
    try:
        room_id = _room_id(args.group)
        status = _call("groups.succession.status", room_id=room_id)
        target = _computer(status, args.computer)
        if not _confirm(keep_question(args.group, status, target), getattr(args, "yes", False)):
            return 1
        result = _call("groups.succession.keep", room_id=room_id, install_id=target)
    except GroupCommandError as exc:
        print(exc, file=sys.stderr)
        return 1
    print("\n".join(status_lines(result)))
    return 0


def _waiting_for_turns(status: dict, target: str | None = None) -> bool:
    moving = status.get("moving") or {}
    return moving.get("step") == "waiting_for_turns" and (
        target is None or (moving.get("to") or {}).get("install_id") == target)


def cmd_move(args: argparse.Namespace) -> int:
    """On the host: hand the group over to another computer (it keeps a copy here). With replies in progress
    it waits for them; ``--now`` moves at once, and those replies show as unknown there."""
    try:
        room_id = _room_id(args.group)
        status = _call("groups.succession.status", room_id=room_id)
        target = _computer(status, args.computer)
        name = _computers(status).get(target, "that computer")
        result = status
        if not _waiting_for_turns(status, target):
            if not _confirm(f"Move “{args.group}” to {name}? It continues there, and "
                            f"{_name(status['this_install'], 'this computer')} keeps a copy.", args.yes):
                return 1
            print(f"Moving “{args.group}” to {name}…")
            result = _call("groups.succession.move", room_id=room_id, target_install_id=target)
        if _waiting_for_turns(result) and getattr(args, "now", False):
            if not _confirm(f"Move now? Replies still in progress will show as unknown on {name} and won't rerun "
                            "by themselves.", args.yes):
                return 1
            result = _call("groups.succession.move_now", room_id=room_id)
    except GroupCommandError as exc:
        print(exc, file=sys.stderr)
        return 1
    print("\n".join(status_lines(result)))
    return 0


def cmd_backups(args: argparse.Namespace) -> int:
    try:
        room_id = _room_id(args.group)
        status = _call("groups.succession.status", room_id=room_id)
        action = getattr(args, "backups_action", None)
        if action in {"allow", "disallow"}:
            install_id, allowed = _computer(status, args.computer), action == "allow"
            if install_id == status["this_install"]["install_id"]:
                _call("groups.custody.allow", room_id=room_id, successor=allowed)
            if status["this_install"]["role"] == "host":
                _call("groups.custody.designate", room_id=room_id, install_id=install_id, successor=allowed)
            status = _call("groups.succession.status", room_id=room_id)
        elif action == "add":
            _add_backup(room_id, args.computer)
            status = _call("groups.succession.status", room_id=room_id)
    except GroupCommandError as exc:
        print(exc, file=sys.stderr)
        return 1
    print("\n".join(status_lines(status)))
    return 0


def _add_backup(room_id: str, peer: str) -> None:
    """A backup computer from ``hermes peer``: a copy-only invitation there, then this host adds it."""
    from hermes_cli.subcommands import peer as peers
    configured = peers._load_peers().get(peer)
    key = peers._peer_secret(peer)
    if not isinstance(configured, dict) or not configured.get("url") or not key:
        raise GroupCommandError(f"Add the computer first with `hermes peer add {peer} <url>`.")
    state = _call("groups.state", room_id=room_id)["room"]
    invitation = peers._request(
        configured["url"].rstrip("/") + "/v1/room-members/invitations", key, method="POST", body={
            "room_id": room_id, "home_install_id": state["authority_gateway_id"],
            "authority_gateway_id": state["authority_gateway_id"], "authority_epoch": state["authority_epoch"],
            "member_id": "custody:installation", "passive_only": True})
    _call("groups.custody.add", room_id=room_id, target_url=configured["url"], catalog=invitation["catalog"],
          grant=invitation["grant"])


def build_groups_parser(subparsers, *, cmd_groups: Callable) -> None:
    """Attach ``hermes groups`` to ``subparsers``: one parser for the messaging chats that control Group
    Chats (``allow``, ``chats``, ``revoke``) and for a group whose host went offline."""
    parser = subparsers.add_parser(
        "groups", help="Group Chats: the messaging chats that control them, and a group whose host went offline",
        description="Allow, list or revoke the messaging chats that can control your Group Chats with /group, "
                    "see whether a group's host can be reached, continue the group on this computer, choose a "
                    "computer after it was continued on two, move it on purpose, and manage backup copies. "
                    "The gateway must be running.")
    sub = parser.add_subparsers(dest="groups_action")
    allow = sub.add_parser("allow", help="Allow the chat that showed this code after /group")
    allow.add_argument("code", help="The code from the chat, for example K7Q2-M9XF")
    allow.add_argument("--yes", action="store_true", help="Allow without asking for confirmation")
    sub.add_parser("chats", help="List the chats that can control your Group Chats")
    revoke = sub.add_parser("revoke", help="Stop a chat from controlling your Group Chats")
    revoke.add_argument("chat", help="The chat ID that 'hermes groups chats' shows")
    parser.set_defaults(func=cmd_groups)
    add_succession_commands(sub)


def add_succession_commands(commands) -> None:
    """Attach this family's host-loss subcommands to the ``hermes groups`` subparsers ``commands``,
    so one ``groups`` parser can carry them beside #111939's messaging ones."""
    status = commands.add_parser("status", help="Whether the group's host can be reached, and its backup copies")
    status.add_argument("group")
    status.set_defaults(func=cmd_status)
    continuing = commands.add_parser("continue", help="Continue the group on this computer")
    continuing.add_argument("group")
    continuing.add_argument("--on", help="The computer to continue on (it must be this one)")
    continuing.add_argument("--yes", action="store_true", help="Don't ask for confirmation")
    continuing.set_defaults(func=cmd_continue)
    keeping = commands.add_parser("keep", help="Keep one computer after the group was continued on two")
    keeping.add_argument("group")
    keeping.add_argument("computer")
    keeping.add_argument("--yes", action="store_true", help="Don't ask for confirmation")
    keeping.set_defaults(func=cmd_keep)
    moving = commands.add_parser("move", help="On the host: move the group to another computer")
    moving.add_argument("group")
    moving.add_argument("computer")
    moving.add_argument("--now", action="store_true",
                        help="Don't wait for replies in progress (they show as unknown there)")
    moving.add_argument("--yes", action="store_true", help="Don't ask for confirmation")
    moving.set_defaults(func=cmd_move)
    backups = commands.add_parser("backups", help="The computers keeping a backup copy of the group")
    backups.set_defaults(func=cmd_backups, backups_action=None)
    actions = backups.add_subparsers(dest="backups_action")
    for name, help_text in (("add", "Keep a backup copy on another computer (from `hermes peer`)"),
                            ("allow", "Let a computer continue this group"),
                            ("disallow", "Stop letting a computer continue this group")):
        action = actions.add_parser(name, help=help_text)
        action.add_argument("group")
        action.add_argument("computer")
        action.set_defaults(func=cmd_backups, backups_action=name)
    backups.add_argument("group", nargs="?")

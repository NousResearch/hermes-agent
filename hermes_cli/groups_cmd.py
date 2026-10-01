"""``hermes groups``: allow, list or revoke the messaging chats that control your Group Chats.

The running gateway owns the codes and grants; this asks it over the control socket,
which identifies the local account. That account becomes the chat's owner, the same
account Desktop uses for the Group Chats it creates.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

_ERRORS = {
    "unknown_code": "That code isn't waiting here. A code works once and expires after 10 minutes; "
                    "send /group in the chat again for a new one.",
    "permission_denied": "That chat is already connected to another account's Group Chats.",
    "capacity_exhausted": "Too many chats are connected. Revoke one with 'hermes groups revoke' first.",
    "profile_unavailable": "The gateway no longer serves the profile that chat talks to.",
    "unknown_grant": "No connected chat has that ID. Run 'hermes groups chats' to see them.",
    "ambiguous_grant": "More than one chat starts with that ID. Type more of it.",
    "stale_epoch": "The gateway is restarting. Try again in a moment.",
    "unavailable": "The gateway couldn't complete that. Its log has the details.",
}
_NO_GATEWAY = ("No running Hermes gateway answered. Start it (hermes gateway run, or open "
               "Hermes Desktop), then try again.")


def _homes() -> list[Path]:
    from hermes_constants import get_default_hermes_root, get_hermes_home
    homes: list[Path] = []
    for home in (Path(get_hermes_home()), Path(get_default_hermes_root())):
        if home not in homes:
            homes.append(home)
    return homes


def _ask_each(params: dict) -> list[dict]:
    """Answers from every gateway this account can reach for the current and default home."""
    from gateway.control_socket import query_gateway_control
    answers = []
    for home in _homes():
        answer = query_gateway_control(home, "group-chats", params=params, timeout=8.0)
        if answer is not None and answer not in answers:
            answers.append(answer)
    return answers


def _ask(params: dict, *, retry_on: str) -> dict | None:
    answers = _ask_each(params)
    return next((a for a in answers if a.get("error") != retry_on), answers[0] if answers else None)


def _describe(chat: dict) -> list[str]:
    platform = chat["platform"].replace("_", " ").title()
    bot = f'the Bot of profile "{chat["profile"]}"'
    if chat["kind"] == "private":
        return [f'{platform} private chat with {chat["user"]} (user ID {chat["user_id"]}), through {bot}.',
                "Only that person can use it, while they are on the Bot's allow_admin_from list."]
    return [f'{platform} shared chat "{chat["chat"]}" (chat ID {chat["chat_id"]}), through {bot}.',
            "Everyone in that chat will be able to read your Group Chat names, recent messages and "
            "approval requests.",
            f"People on the Bot's {chat['admins']} list will be able to send, stop and answer approvals.",
            f'Requested by {chat["user"]} (user ID {chat["user_id"]}).']


def _fail(answer: dict | None) -> int:
    print(_NO_GATEWAY if answer is None else _ERRORS.get(answer.get("error"), f'Refused: {answer.get("error")}'))
    return 1


def _allow(args) -> int:
    from hermes_cli.cli_output import prompt_yes_no
    info = _ask({"action": "describe", "code": args.code}, retry_on="unknown_code")
    if info is None or "error" in info:
        return _fail(info)
    for line in _describe(info):
        print(line)
    print(f'It will control your Group Chats on profile "{info["profile"]}": list and read them, '
          "send messages, stop work and answer approvals.")
    if not args.yes and not prompt_yes_no("Allow this chat?", default=False):
        print("Not allowed.")
        return 1
    result = _ask({"action": "allow", "code": args.code}, retry_on="unknown_code")
    if result is None or "error" in result:
        return _fail(result)
    print(f'Allowed. Revoke it any time with: hermes groups revoke {result["grant"]}')
    return 0


def _chats(args) -> int:
    answers = _ask_each({"action": "list"})
    if not answers:
        return _fail(None)
    chats = [chat for answer in answers for chat in answer.get("chats", [])]
    if not chats:
        print("No chats can control your Group Chats. Send /group in a chat with your Bot to connect one.")
        return 0
    for chat in chats:
        since = datetime.fromtimestamp(chat["created_at"]).strftime("%Y-%m-%d %H:%M")
        where = (f'private chat with {chat["user"]}' if chat["kind"] == "private"
                 else f'shared chat "{chat["chat"]}"')
        print(f'{chat["grant"]}  {chat["platform"]} {where}, profile "{chat["profile"]}", since {since}')
    return 0


def _revoke(args) -> int:
    result = _ask({"action": "revoke", "grant": args.chat}, retry_on="unknown_grant")
    if result is None or "error" in result:
        return _fail(result)
    print(f'Revoked {result["revoked"]}: that chat can no longer control your Group Chats.')
    return 0


def groups_command(args) -> int:
    handler = {"allow": _allow, "chats": _chats, "revoke": _revoke}.get(getattr(args, "groups_action", None))
    if handler is None:
        print("Usage: hermes groups {allow <code> | chats | revoke <chat>}")
        return 1
    return handler(args)

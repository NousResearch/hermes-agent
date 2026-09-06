"""Owner-only remembered approvals, using the existing native choice-page contract."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from functools import partial
import hashlib
import logging
import time

from gateway.choice_picker import ChoicePage, PAGE_TIMEOUT_SECONDS
from gateway.group_home_consent import DisclosureChanged, _key, disclosed_call, protect_group_callback
from gateway import hosted_room_approval_rules as rules
from gateway import hosted_room_messaging_approvals as approvals
from gateway import hosted_room_messaging as messaging
from gateway.hosted_room_messaging import (
    format_room_detail, messaging_event_id, room_picker_choices,
)

logger = logging.getLogger("gateway.run")
UNHANDLED = object()
PAGE_SIZE = 8
FORGOTTEN = "Future requests will ask again. A command already approved may still finish."


def _label(value, limit=80):
    return approvals._display_text(value, limit=limit)


def _confirm_digest(pending):
    scope = [str(pending[key]) for key in approvals._APPROVAL_SCOPE_FIELDS]
    scope.append(str(pending["approval"].get("remember_key") or ""))
    scope.append(str(pending["approval"].get("remember_context") or ""))
    return hashlib.sha256("\0".join(scope).encode()).hexdigest()


def _owner_room(backend, room):
    from gateway import hosted_rooms

    if room.get("_room_mode") in {"remote", "desktop"}:
        raise approvals.MessagingApprovalError("Manage these permissions in the Group Chat's owner chat.")
    current = hosted_rooms.room_state(backend.db_path, room_id=room["room_id"])
    if (current["authority_gateway_id"] != hosted_rooms.local_authority_gateway_id()
            or current["authority_gateway_id"] != room["authority_gateway_id"]
            or current["authority_epoch"] != room["authority_epoch"]):
        raise approvals.MessagingApprovalError("This Group Chat changed. Open it again.")


def _saved(backend, room):
    _owner_room(backend, room)
    conn = approvals._connect(backend.db_path)
    try:
        conn.execute("BEGIN")
        _current_database_room(conn, room)
        return rules.list_rules(conn, room["room_id"], include_pending=True)
    finally:
        conn.close()


def _forget(backend, room, selection):
    _owner_room(backend, room)
    if not 8 <= len(str(selection)) <= 64 or any(char not in "0123456789abcdefABCDEF" for char in str(selection)):
        raise approvals.MessagingApprovalError("Choose a permission code from the current list.")
    conn = approvals._connect(backend.db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        _current_database_room(conn, room)
        candidates = [rule for rule in rules.list_rules(conn, room["room_id"], include_pending=True)
                      if rule["rule_id"].startswith(str(selection).lower())]
        if len(candidates) != 1:
            raise approvals.MessagingApprovalError("Choose a permission code from the current list.")
        changed = rules.revoke_rule(conn, room["room_id"], candidates[0]["rule_id"])
        conn.commit()
        return changed
    finally:
        conn.close()


def _current_database_room(conn, room):
    from gateway import hosted_rooms

    row = conn.execute("SELECT authority_gateway_id, authority_epoch, disbanded_at FROM hosted_rooms WHERE room_id=?",
                       (room["room_id"],)).fetchone()
    if (row is None or row["disbanded_at"] is not None or row["authority_epoch"] != room["authority_epoch"]
            or row["authority_gateway_id"] != room["authority_gateway_id"]
            or row["authority_gateway_id"] != hosted_rooms.local_authority_gateway_id()):
        raise approvals.MessagingApprovalError("This Group Chat changed. Open it again.")


def _grant_saved(backend, command_id):
    conn = approvals._connect(backend.db_path)
    try:
        return rules.granted_rule(conn, command_id) is not None
    finally:
        conn.close()


class GroupApprovalPermissions:
    def __init__(self, runner, event, backend, room, *, profile, command, stamp):
        self.runner, self.event, self.backend = runner, event, backend
        self.identity = (room["room_id"], room["authority_gateway_id"], room["authority_epoch"])
        self.reference = str(room.get("messaging_ref") or room["room_id"])
        self.profile, self.command, self.stamp = profile, command, stamp
        self.read = partial(disclosed_call, runner, event, stamp)
        self.delegate = None

    async def fresh(self):
        if self.runner._can_approve_group_chats(self.event) is not True:
            raise approvals.MessagingApprovalError(self.runner._group_chat_approval_denial())
        rooms = await self.read(messaging.list_messaging_rooms, self.backend, profile=self.profile)
        room = next((candidate for candidate in rooms if (
            candidate.get("room_id"), candidate.get("authority_gateway_id"), candidate.get("authority_epoch"),
        ) == self.identity), None)
        if room is None:
            raise approvals.MessagingApprovalError("This Group Chat changed. Open it again.")
        await self.read(_owner_room, self.backend, room)
        return room

    async def requests(self):
        room = await self.fresh()
        return room, await self.read(approvals.pending_approvals_for_room, self.backend, room)

    def confirmations(self):
        cache = getattr(self.runner, "_group_approval_confirmations", None)
        if not isinstance(cache, OrderedDict):
            cache = self.runner._group_approval_confirmations = OrderedDict()
        now = time.monotonic()
        for key, entry in list(cache.items()):
            if entry[0] <= now:
                cache.pop(key, None)
        while len(cache) > 128:
            cache.popitem(last=False)
        return cache

    def confirmation_key(self, reference):
        return (*_key(self.runner, self.event), *self.identity, reference.upper())

    def remember_warning(self, room, selected):
        reference = approvals.approval_reference(selected)
        if "remember" not in selected["approval"].get("choices", []):
            raise approvals.MessagingApprovalError("This request can only be approved once or denied.")
        self.confirmations()[self.confirmation_key(reference)] = (
            time.monotonic() + PAGE_TIMEOUT_SECONDS, _confirm_digest(selected), self.stamp,
        )
        bot = _label(approvals.approval_member_label(room, selected["member_id"]))
        title = (f"**Always allow this command in this chat?**\n{bot} · {_label(room['name'])}\n"
                 f"{_label(selected['approval'].get('remember_context'), 384)}\n\n"
                 f"{_label(selected['approval']['command'], 512)}\n\n"
                 "This Bot may run this command again in this Group Chat without asking. "
                 "It can affect changing files and data. You can remove this permission later.")
        return ChoicePage(title, [
            {"label": "Always allow in this chat", "value": f"confirm:{reference}:{_confirm_digest(selected)}"},
            {"label": "Go back", "value": f"request:{reference}"},
        ])

    async def approval_page(self, reference=""):
        room, pending = await self.requests()
        if not pending:
            return ChoicePage("No pending approvals.", [
                {"label": "View remembered approvals", "value": "permissions:0"},
                {"label": "‹ Back to Group Chat", "value": "group"},
            ])
        if len(pending) > 1 and not reference:
            return ChoicePage("**Approval requests**", [
                {"label": "Review request · " + _label(approvals.approval_member_label(room, action["member_id"])),
                 "value": "request:" + approvals.approval_reference(action), "full_width": True}
                for action in pending
            ] + [{"label": "View remembered approvals", "value": "permissions:0"},
                 {"label": "‹ Back to Group Chat", "value": "group"}])
        _, selected = approvals.select_pending_approval(pending, reference or approvals.approval_reference(pending[0]))
        reference = approvals.approval_reference(selected)
        choices = [{"label": "Allow once", "value": "once:" + reference},
                   {"label": "Deny", "value": "deny:" + reference}]
        if "remember" in selected["approval"].get("choices", []):
            choices.append({"label": "Always allow in this chat", "value": "remember:" + reference, "full_width": True})
        choices.extend([{"label": "View remembered approvals", "value": "permissions:0"},
                        {"label": "‹ Back to Group Chat", "value": "group"}])
        description, command = approvals._approval_display_parts(selected["approval"])
        bot = _label(approvals.approval_member_label(room, selected["member_id"]))
        title = f"**Approval needed**\n**{bot}**\n\n{description or command or 'Command'}"
        if command and command != description:
            title += f"\n{command}"
        if selected["approval"].get("remember_context"):
            title += "\n\n" + _label(selected["approval"]["remember_context"], 384)
        return ChoicePage(title, choices)

    async def permissions_page(self, offset=0):
        room = await self.fresh()
        saved = await self.read(_saved, self.backend, room)
        offset = max(0, min(offset, max(0, len(saved) - 1)))
        selected = saved[offset:offset + PAGE_SIZE]
        title = f"**Remembered approvals** · {_label(room['name'])}"
        if not saved:
            title += "\nNo commands are remembered for this Group Chat."
        choices = [{"label": "View · " + _label(approvals.approval_member_label(room, rule["member_id"]), 28)
                    + " · " + _label(rule["command_text"], 30) + " · " + _label(rule.get("context_text"), 60),
                    "value": "rule:" + rule["rule_id"], "full_width": True}
                   for rule in selected]
        if offset:
            choices.append({"label": "‹ Previous", "value": f"permissions:{max(0, offset - PAGE_SIZE)}"})
        if offset + PAGE_SIZE < len(saved):
            choices.append({"label": "Next ›", "value": f"permissions:{offset + PAGE_SIZE}"})
        choices.append({"label": "‹ Back to Group Chat", "value": "group"})
        return ChoicePage(title, choices)

    async def apply(self, choice, reference, *, confirmed=False, expected_digest=""):
        room, pending = await self.requests()
        _, selected = approvals.select_pending_approval(pending, reference)
        if choice == "remember":
            entry = self.confirmations().pop(self.confirmation_key(reference), None)
            if (not confirmed or entry is None or entry[1] != _confirm_digest(selected) or entry[2] != self.stamp
                    or (expected_digest and expected_digest != entry[1])):
                raise approvals.MessagingApprovalError("That confirmation expired. Choose Always allow in this chat again.")
        denial = self.runner._group_chat_rate_limit_denial(self.event, action="deny" if choice == "deny" else "approve")
        if denial:
            return denial
        command_id = f"approval:{messaging_event_id(self.event)}:{choice}:{reference}"
        _, selected, result = await self.read(
            approvals.submit_room_approval, self.backend, room, command_id=command_id, choice=choice,
            installation_owner_authorized=True, selection=reference, expected_request_id=selected["request_id"],
            _work_action="deny" if choice == "deny" else "approve",
        )
        bot = _label(approvals.approval_member_label(room, selected["member_id"]))
        if choice == "remember" and await self.read(_grant_saved, self.backend, result.get("command_id", command_id)):
            return ChoicePage(f"Allowed. This command is remembered for {bot} in {_label(room['name'])}.", [
                {"label": "View remembered approvals", "value": "permissions:0"},
                {"label": "‹ Back to Group Chat", "value": "group"},
            ])
        if result.get("queued"):
            return f"Decision sent for {bot}."
        if not result.get("applied", True):
            return str(result.get("result") or "Approval is no longer pending.")
        if choice == "remember":
            return "Allowed this time. The remembered permission wasn’t saved."
        return f"Allowed once for {bot}." if choice == "once" else f"Denied for {bot}."

    async def choose(self, chat_id, value):
        source = self.event.source
        destination = source.thread_id if source.platform.value == "discord" and source.thread_id else source.chat_id
        if str(chat_id) != str(destination):
            raise approvals.MessagingApprovalError("This menu belongs to another chat.")
        if self.runner._can_approve_group_chats(self.event) is not True:
            raise approvals.MessagingApprovalError(self.runner._group_chat_approval_denial())
        if self.delegate is not None:
            return await self.delegate(chat_id, value)
        await self.fresh()
        action, _, argument = str(value).partition(":")
        if action not in {"once", "deny", "confirm"}:
            denial = self.runner._group_chat_rate_limit_denial(self.event, action="approve" if action == "forget" else "read")
            if denial:
                return denial
        handler = {
            "permissions": self._choose_permissions,
            "request": self.approval_page,
            "remember": self._choose_remember,
            "confirm": self._choose_confirm,
            "once": partial(self.apply, "once"),
            "deny": partial(self.apply, "deny"),
            "rule": self._choose_rule,
            "forget": partial(self._choose_rule, forget=True),
            "group": partial(self._choose_group, chat_id=chat_id),
        }.get(action)
        if handler is None:
            raise approvals.MessagingApprovalError("This menu changed. Open it again.")
        return await handler(argument)

    async def _choose_permissions(self, argument):
        if not argument.isascii() or not argument.isdecimal() or len(argument) > 4:
            raise approvals.MessagingApprovalError("This menu changed. Open it again.")
        return await self.permissions_page(int(argument))

    async def _choose_remember(self, argument):
        room, pending = await self.requests()
        _, selected = approvals.select_pending_approval(pending, argument)
        return self.remember_warning(room, selected)

    async def _choose_confirm(self, argument):
        reference, _, fingerprint = argument.partition(":")
        if len(fingerprint) != 64 or any(char not in "0123456789abcdef" for char in fingerprint):
            raise approvals.MessagingApprovalError("That confirmation expired. Choose Always allow in this chat again.")
        return await self.apply("remember", reference, confirmed=True, expected_digest=fingerprint)

    async def _choose_rule(self, argument, *, forget=False):
        room = await self.fresh()
        saved = await self.read(_saved, self.backend, room)
        rule = next((rule for rule in saved if rule["rule_id"] == argument), None)
        if rule is None:
            return await self.permissions_page()
        if forget:
            await self.read(_forget, self.backend, room, argument, _work_action="approve")
            return ChoicePage(FORGOTTEN, [
                {"label": "View remembered approvals", "value": "permissions:0"},
                {"label": "‹ Back to Group Chat", "value": "group"},
            ])
        title = (f"**Remembered approval**\n{_label(approvals.approval_member_label(room, rule['member_id']))}"
                 f" · {_label(room['name'])}\n{_label(rule.get('context_text') or 'Connection details unavailable', 384)}"
                 f"\n\n{_label(rule['command_text'], 512)}")
        if rule["state"] == "pending":
            title += "\n\nWaiting for the first approval to be confirmed. Not active yet."
        return ChoicePage(title, [
            {"label": "Forget this permission", "value": "forget:" + rule["rule_id"], "full_width": True},
            {"label": "‹ Back to remembered approvals", "value": "permissions:0"},
        ])

    async def _choose_group(self, _argument, *, chat_id):
        room = await self.fresh()
        try:
            from gateway.hosted_room_messaging_files import room_picker_callback
        except ImportError:
            return await self.read(format_room_detail, self.backend, room, room_command=self.command)
        callback, reusable = room_picker_callback(self.runner, self.event, self.backend, self.command, None)
        if reusable:
            choices = await self.read(room_picker_choices, self.backend, [room])
            page = await callback(chat_id, choices[0]["value"])
            if isinstance(page, ChoicePage):
                self.delegate = callback
                return page
        return await self.read(format_room_detail, self.backend, room, room_command=self.command)

    async def send(self, page):
        source = await asyncio.to_thread(self.runner._normalize_source_for_session_key, self.event.source)

        @protect_group_callback(self.runner, self.event)
        async def callback(chat_id, value):
            try:
                return await self.choose(chat_id, value)
            except (approvals.MessagingApprovalError, DisclosureChanged) as exc:
                return str(exc)
            except Exception:
                logger.exception("Group Chat approval menu failed")
                return "Couldn’t apply that action. Open the Group Chat again."

        return await self.runner._try_send_group_choice_picker(
            self.event, self.runner._session_key_for_source(source), title=page.title, choices=page.choices,
            on_choice_selected=callback, reusable=True, disclosure_stamp=self.stamp,
        )

    async def text_permissions(self, page=1):
        room = await self.fresh()
        saved = await self.read(_saved, self.backend, room)
        if not saved:
            return "No commands are remembered for this Group Chat."
        pages = (len(saved) + PAGE_SIZE - 1) // PAGE_SIZE
        page = min(page, pages)
        lines = [f"**Remembered approvals** · {_label(room['name'])}", f"Page {page} of {pages}"]
        for rule in saved[(page - 1) * PAGE_SIZE:page * PAGE_SIZE]:
            length = 8
            while sum(other['rule_id'].startswith(rule['rule_id'][:length]) for other in saved) > 1:
                length += 1
            lines.append(f"`{rule['rule_id'][:length]}` · {_label(approvals.approval_member_label(room, rule['member_id']))}"
                         f" · {_label(rule['command_text'], 100)}")
            lines.append(_label(rule.get("context_text") or "Connection details unavailable", 384))
            if rule["state"] == "pending":
                lines.append("Waiting for confirmation; not active yet.")
        lines.append(f"Forget a permission: `{self.command} {self.reference} forget <permission code>`")
        if page > 1:
            lines.append(f"Previous: `{self.command} {self.reference} permissions {page - 1}`")
        if page < pages:
            lines.append(f"Next: `{self.command} {self.reference} permissions {page + 1}`")
        return "\n".join(lines)

    async def handle_command(self, words):
        handler = {
            "approvals": self._command_approvals,
            "permissions": self._command_permissions,
            "remember": self._command_remember,
            "forget": self._command_forget,
        }[words[0].casefold()]
        try:
            return await handler(words[1:])
        except (approvals.MessagingApprovalError, DisclosureChanged) as exc:
            return str(exc)

    async def _command_approvals(self, args):
        if args:
            return UNHANDLED
        adapter = self.runner._adapter_for_source(self.event.source)
        if getattr(type(adapter), "supports_choice_pages", False) is True:
            if await self.send(await self.approval_page()):
                return None
        return UNHANDLED

    async def _command_permissions(self, args):
        if len(args) > 1 or (args and (not args[0].isascii() or not args[0].isdecimal()
                                      or len(args[0]) > 4 or int(args[0]) < 1)):
            return f"Use `{self.command} {self.reference} permissions <page number>`."
        page = int(args[0]) if args else 1
        if await self.send(await self.permissions_page((page - 1) * PAGE_SIZE)):
            return None
        return await self.text_permissions(page)

    async def _command_remember(self, args):
        if len(args) not in {1, 2} or (len(args) == 2 and args[1].casefold() != "confirm"):
            return f"Use `{self.command} {self.reference} remember <approval code>`."
        room, pending = await self.requests()
        _, selected = approvals.select_pending_approval(pending, args[0])
        reference = approvals.approval_reference(selected)
        if len(args) == 2:
            result = await self.apply("remember", reference, confirmed=True)
            if isinstance(result, ChoicePage):
                return None if await self.send(result) else result.title
            return result
        page = self.remember_warning(room, selected)
        if await self.send(page):
            return None
        return page.title + f"\n\nConfirm: `{self.command} {self.reference} remember {reference} confirm`"

    async def _command_forget(self, args):
        if len(args) != 1:
            return f"Use `{self.command} {self.reference} forget <permission code>`."
        room = await self.fresh()
        denial = self.runner._group_chat_rate_limit_denial(self.event, action="approve")
        if denial:
            return denial
        await self.read(_forget, self.backend, room, args[0], _work_action="approve")
        return FORGOTTEN

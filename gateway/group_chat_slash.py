"""Messaging command surface for Bot Group Chats."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

from gateway.platforms.base import MessageEvent


logger = logging.getLogger("gateway.run")


_GROUP_CHAT_RATE_WINDOW_SECONDS = 60.0


_GROUP_CHAT_READ_RATE_LIMIT = 30


_GROUP_CHAT_MUTATION_RATE_LIMIT = 12


_GROUP_CHAT_STOP_RATE_LIMIT = 30


_GROUP_CHAT_RATE_BUCKET_CAP = 2048


_NATIVE_DISTINCT_DM_PLATFORMS = frozenset({
    "bluebubbles",
    "dingtalk",
    "email",
    "feishu",
    "mattermost",
    "qqbot",
    "sms",
    "wecom",
    "wecom_callback",
    "weixin",
    "whatsapp_cloud",
    "yuanbao",
})


class GroupChatSlashCommandsMixin:
    """Authorize, render, and mutate Group Chats from messaging clients."""

    def _home_chat_is_single_operator(self, event: MessageEvent) -> bool:
        """Recognize the configured home chat's exact authorized operator."""
        from gateway.authz_mixin import (
            _auth_env,
            _coerce_allow_set,
            _platform_authorization_env_names,
        )
        from gateway.slash_access import is_home_control_source

        source = event.source
        if getattr(source, "delivered_via_upstream_relay", False) is True:
            return False
        if not is_home_control_source(self.config, source):
            return False

        is_authorized = getattr(self, "_is_user_authorized_for_source", None)
        if not callable(is_authorized):
            return False
        try:
            if not is_authorized(source):
                return False
        except Exception:
            return False

        def _census() -> bool:
            platform_name = source.platform.value
            allowed_users_env, allow_all_env = _platform_authorization_env_names(
                source.platform
            )
            candidates: set[str] = set()
            adapter_for_source = getattr(self, "_adapter_for_source", None)
            transport_adapter = (
                adapter_for_source(source) if callable(adapter_for_source) else None
            )
            adapter_config = getattr(transport_adapter, "config", None)
            adapter_extra = getattr(adapter_config, "extra", None)
            if transport_adapter is not None:
                extra = adapter_extra if isinstance(adapter_extra, dict) else {}
            else:
                platform_config = self.config.platforms.get(source.platform)
                extra = getattr(platform_config, "extra", None) or {}
            candidates.update(_coerce_allow_set(extra.get("allow_from")))
            candidates.update(_coerce_allow_set(_auth_env(allowed_users_env)))
            candidates.update(_coerce_allow_set(_auth_env("GATEWAY_ALLOWED_USERS")))
            if _auth_env("GATEWAY_ALLOW_ALL_USERS").lower() in {
                "true",
                "1",
                "yes",
            }:
                return False
            if allow_all_env and _auth_env(allow_all_env).lower() in {
                "true",
                "1",
                "yes",
            }:
                return False

            authorization_home = getattr(
                source,
                "_authorization_profile_home",
                None,
            )
            if authorization_home is not None:
                pairing_store = getattr(self, "pairing_store", None)
            else:
                pairing_store_for = getattr(self, "_pairing_store_for", None)
                pairing_store = (
                    pairing_store_for(source) if callable(pairing_store_for) else None
                )
            if pairing_store is not None:
                try:
                    candidates.update(
                        str(row.get("user_id") or "").strip()
                        for row in pairing_store.list_approved(platform_name)
                        if str(row.get("user_id") or "").strip()
                    )
                except Exception:
                    return False
            if not candidates or "*" in candidates:
                return False

            user_id = str(source.user_id)
            matcher = getattr(pairing_store, "_user_ids_match", None)
            if callable(matcher):
                return all(
                    matcher(platform_name, candidate, user_id)
                    for candidate in candidates
                )
            return candidates == {user_id}

        authorization_home = getattr(source, "_authorization_profile_home", None)
        if authorization_home is None:
            return _census()
        from gateway.run import _profile_runtime_scope

        with _profile_runtime_scope(Path(authorization_home)):
            return _census()

    def _can_control_group_chats(self, event: MessageEvent) -> bool:
        """Authorize a trusted DM or the exact operator of an explicit home chat."""
        from gateway.slash_access import policy_for_source

        if self._home_chat_is_single_operator(event):
            return True
        # ``chat_type=dm`` is not a privacy boundary: Slack MPIMs and Matrix
        # m.direct rooms can contain several people. Adapters stamp this
        # transport-local signal only when they can prove the current surface is
        # one-to-one. Unknown and older connectors fail closed.
        platform = str(
            getattr(getattr(event.source, "platform", None), "value", "") or ""
        )
        native_distinct_dm = (
            getattr(event.source, "delivered_via_upstream_relay", False) is not True
            and platform in _NATIVE_DISTINCT_DM_PLATFORMS
            and str(getattr(event.source, "chat_type", "") or "").casefold()
            in {"dm", "direct", "private"}
        )
        if (
            getattr(event.source, "is_one_to_one", None) is not True
            and not native_distinct_dm
        ):
            return False
        chat_type = str(getattr(event.source, "chat_type", "") or "").casefold()
        if chat_type not in {"dm", "direct", "private"}:
            return False
        policy = policy_for_source(self.config, event.source)
        return policy.enabled and policy.is_admin(event.source.user_id)

    @staticmethod
    def _group_chat_control_denial(event: MessageEvent) -> str:
        chat_type = str(getattr(event.source, "chat_type", "") or "").casefold()
        platform = str(
            getattr(getattr(event.source, "platform", None), "value", "") or ""
        )
        proven_private = getattr(event.source, "is_one_to_one", None) is True or (
            getattr(event.source, "delivered_via_upstream_relay", False) is not True
            and platform in _NATIVE_DISTINCT_DM_PLATFORMS
            and chat_type in {"dm", "direct", "private"}
        )
        if chat_type not in {"dm", "direct", "private"} or not proven_private:
            return (
                "Group Chat controls are private. Use your authorized one-to-one "
                "Hermes chat."
            )
        return (
            "This chat can’t control Group Chats. Use your authorized one-to-one "
            "Hermes chat or authorize this account in settings."
        )

    def _group_chat_rate_limit_denial(
        self,
        event: MessageEvent,
        *,
        action: str,
    ) -> Optional[str]:
        """Bound authenticated Group Chat commands per person and chat."""

        normalized_action = str(action or "read").casefold()
        if normalized_action == "stop":
            limit = _GROUP_CHAT_STOP_RATE_LIMIT
            bucket_kind = "stop"
        elif normalized_action in {"send", "retry"}:
            limit = _GROUP_CHAT_MUTATION_RATE_LIMIT
            bucket_kind = "change"
        else:
            limit = _GROUP_CHAT_READ_RATE_LIMIT
            bucket_kind = "read"

        source = event.source
        from gateway.hosted_room_messaging import messaging_transport_profile

        platform = str(getattr(getattr(source, "platform", None), "value", "") or "")
        key = (
            platform,
            str(getattr(source, "profile", None) or "default"),
            messaging_transport_profile(event),
            str(getattr(source, "scope_id", None) or ""),
            str(getattr(source, "chat_id", None) or ""),
            str(
                getattr(source, "user_id_alt", None)
                or getattr(source, "user_id", None)
                or ""
            ),
            bucket_kind,
        )
        now = time.monotonic()
        buckets = getattr(self, "_group_chat_command_rate_buckets", None)
        if not isinstance(buckets, dict):
            buckets = {}
            self._group_chat_command_rate_buckets = buckets
        recent = [
            stamp
            for stamp in buckets.get(key, ())
            if now - stamp < _GROUP_CHAT_RATE_WINDOW_SECONDS
        ]
        if len(recent) >= limit:
            buckets[key] = recent
            return "Too many Group Chat commands. Wait a moment and try again."
        recent.append(now)
        buckets[key] = recent

        if len(buckets) > _GROUP_CHAT_RATE_BUCKET_CAP:
            stale_before = now - _GROUP_CHAT_RATE_WINDOW_SECONDS
            for bucket_key in list(buckets):
                if not buckets[bucket_key] or buckets[bucket_key][-1] <= stale_before:
                    buckets.pop(bucket_key, None)
            while len(buckets) > _GROUP_CHAT_RATE_BUCKET_CAP:
                buckets.pop(next(iter(buckets)))
        return None

    @staticmethod
    def _group_chat_profile(event: MessageEvent) -> str:
        """Return the profile selected by the authenticated inbound route."""

        routed = str(getattr(event.source, "profile", None) or "").strip()
        if routed:
            return routed
        from hermes_cli.profiles import get_active_profile_name

        return str(get_active_profile_name() or "default")

    def _can_approve_group_chats(self, event: MessageEvent) -> bool:
        """Keep dangerous approvals with the installation owner's main chat."""

        from gateway.hosted_room_messaging import messaging_transport_profile

        return (
            self._group_chat_profile(event) == "default"
            and messaging_transport_profile(event) == "default"
            and self._can_control_group_chats(event)
        )

    @staticmethod
    def _group_chat_approval_denial() -> str:
        return (
            "Approve or deny Bot commands from the installation owner’s "
            "authorized one-to-one Hermes chat."
        )

    @staticmethod
    def _group_chat_command_args(event: MessageEvent) -> str:
        """Return command arguments without rewriting free-form message text."""

        command_text = str(event.text or "").lstrip()
        parts = command_text.split(maxsplit=1)
        return parts[1] if len(parts) > 1 else ""

    async def _handle_rooms_command(self, event: MessageEvent) -> Optional[str]:
        """List Bot Group Chats or show one chat's recent activity."""

        from gateway import hosted_rooms
        from gateway.hosted_room_messaging import (
            RoomControlError,
            current_room_backend,
            format_room_bot_detail,
            format_room_bot_list,
            format_room_detail,
            format_room_list,
            is_message_edit,
            is_machine_authored,
            list_messaging_rooms,
            messaging_event_id,
            relay_provenance_is_unknown,
            resolve_room,
            resolve_room_picker_choice,
            room_bot_picker_choices,
            room_picker_choices,
        )

        if is_machine_authored(event):
            return "Group Chat controls are only available to people."
        if is_message_edit(event):
            return "Edited messages can’t run Group Chat commands. Send a new message."
        if relay_provenance_is_unknown(event):
            return (
                "Group Chat controls need a relay connector that reports whether the "
                "sender is a person or a bot. Update the connector and try again."
            )
        if not self._can_control_group_chats(event):
            return self._group_chat_control_denial(event)
        service = current_room_backend()
        rooms_command = f"{self._typed_command_prefix_for(event.source)}group"
        query = self._group_chat_command_args(event).strip()
        try:
            words = query.split()
            if (
                words
                and words[0].isdecimal()
                and len(words) > 1
                and words[1].casefold()
                in {
                    "approve",
                    "deny",
                    "retry",
                    "send",
                    "stop",
                }
            ):
                return await self._handle_room_command(event)
            denial = self._group_chat_rate_limit_denial(event, action="read")
            if denial:
                return denial
            profile = self._group_chat_profile(event)
            rooms = await asyncio.to_thread(
                list_messaging_rooms,
                service,
                profile=profile,
            )
            if (
                len(words) == 2
                and words[0].isdecimal()
                and words[1].casefold() == "approvals"
            ):
                if not self._can_approve_group_chats(event):
                    return self._group_chat_approval_denial()
                from gateway.hosted_room_messaging_approvals import (
                    MessagingApprovalError,
                    approval_member_label,
                    approval_picker_choices,
                    format_approval_picker_title,
                    format_pending_approvals,
                    pending_approvals_for_room,
                    resolve_approval_picker_choice,
                    submit_room_approval,
                )

                room = resolve_room(rooms, words[0])
                pending = await asyncio.to_thread(
                    pending_approvals_for_room,
                    service,
                    room,
                )
                if not pending:
                    return "This Group Chat has no pending approvals."
                choices = approval_picker_choices(room, pending)
                source = await asyncio.to_thread(
                    self._normalize_source_for_session_key,
                    event.source,
                )
                session_key = self._session_key_for_source(source)

                async def _on_approval_selected(_chat_id: str, value: str) -> str:
                    if not self._can_approve_group_chats(event):
                        return self._group_chat_approval_denial()
                    current_denial = self._group_chat_rate_limit_denial(
                        event,
                        action="approve",
                    )
                    if current_denial:
                        return current_denial
                    try:
                        current_rooms = await asyncio.to_thread(
                            list_messaging_rooms,
                            service,
                            profile=profile,
                        )
                        current_room = resolve_room(current_rooms, words[0])
                        current_pending = await asyncio.to_thread(
                            pending_approvals_for_room,
                            service,
                            current_room,
                        )
                        index, choice, request_id = resolve_approval_picker_choice(
                            current_room,
                            current_pending,
                            value,
                        )
                        _number, selected, applied = await asyncio.to_thread(
                            submit_room_approval,
                            service,
                            current_room,
                            command_id=(
                                f"approval:{messaging_event_id(event)}:"
                                f"{str(value).replace('=', '.')}"
                            ),
                            choice=choice,
                            installation_owner_authorized=(
                                self._can_approve_group_chats(event)
                            ),
                            selection=index,
                            expected_request_id=request_id,
                        )
                        bot = approval_member_label(
                            current_room,
                            str(selected["member_id"]),
                        )
                        if applied.get("applied") is False:
                            return str(applied.get("result") or "Approval expired.")
                        if applied.get("queued"):
                            return f"Decision sent for {bot}."
                        return (
                            f"Approved once for {bot}."
                            if choice == "once"
                            else f"Denied for {bot}."
                        )
                    except MessagingApprovalError as exc:
                        return str(exc)
                    except Exception:
                        logger.exception("Failed to apply Group Chat approval")
                        return (
                            "Couldn’t apply that approval. Check the Group Chat again."
                        )

                picker_sent = bool(choices) and await self._try_send_choice_picker(
                    event,
                    session_key,
                    title=format_approval_picker_title(room, pending),
                    choices=choices,
                    on_choice_selected=_on_approval_selected,
                )
                if picker_sent:
                    return None
                return format_pending_approvals(
                    service,
                    room,
                    room_reference=words[0],
                    room_command=rooms_command,
                )
            if (
                len(words) >= 2
                and words[0].isdecimal()
                and words[1].casefold() in {"bot", "bots"}
            ):
                room = resolve_room(rooms, words[0])
                if words[1].casefold() == "bot":
                    if len(words) != 3:
                        return (
                            f"Use `{rooms_command} {words[0]} bot <number or handle>`."
                        )
                    return await asyncio.to_thread(
                        format_room_bot_detail,
                        service,
                        room,
                        words[2],
                        room_command=rooms_command,
                    )
                if len(words) != 2:
                    return f"Use `{rooms_command} {words[0]} bots`."
                choices = await asyncio.to_thread(
                    room_bot_picker_choices,
                    service,
                    room,
                )
                source = await asyncio.to_thread(
                    self._normalize_source_for_session_key,
                    event.source,
                )
                session_key = self._session_key_for_source(source)

                async def _on_bot_selected(_chat_id: str, value: str) -> str:
                    if not self._can_control_group_chats(event):
                        return self._group_chat_control_denial(event)
                    current_denial = self._group_chat_rate_limit_denial(
                        event,
                        action="read",
                    )
                    if current_denial:
                        return current_denial
                    try:
                        current_rooms = await asyncio.to_thread(
                            list_messaging_rooms,
                            service,
                            profile=profile,
                        )
                        current_room = resolve_room(current_rooms, words[0])
                        return await asyncio.to_thread(
                            format_room_bot_detail,
                            service,
                            current_room,
                            value,
                            room_command=rooms_command,
                        )
                    except (RoomControlError, hosted_rooms.HostedRoomError) as exc:
                        return str(exc)
                    except Exception:
                        logger.exception("Failed to open Group Chat Bot from messaging")
                        return (
                            "Couldn’t load that Bot. "
                            f"Run `{rooms_command} {words[0]} bots` again."
                        )

                picker_sent = await self._try_send_choice_picker(
                    event,
                    session_key,
                    title=(
                        "🤖 Bots\n"
                        "Choose a Bot to see its handle and available controls."
                    ),
                    choices=choices,
                    on_choice_selected=_on_bot_selected,
                )
                if picker_sent:
                    return None
                return await asyncio.to_thread(
                    format_room_bot_list,
                    service,
                    room,
                    room_command=rooms_command,
                )
            if not query:
                choices = await asyncio.to_thread(
                    room_picker_choices,
                    service,
                    rooms,
                )
                source = await asyncio.to_thread(
                    self._normalize_source_for_session_key,
                    event.source,
                )
                session_key = self._session_key_for_source(source)

                async def _on_room_selected(_chat_id: str, value: str) -> str:
                    if not self._can_control_group_chats(event):
                        return self._group_chat_control_denial(event)
                    current_denial = self._group_chat_rate_limit_denial(
                        event,
                        action="read",
                    )
                    if current_denial:
                        return current_denial
                    try:
                        current_rooms = await asyncio.to_thread(
                            list_messaging_rooms,
                            service,
                            profile=profile,
                        )
                        selected = resolve_room_picker_choice(current_rooms, value)
                        return await asyncio.to_thread(
                            format_room_detail,
                            service,
                            selected,
                            room_command=rooms_command,
                            show_approvals=self._can_approve_group_chats(event),
                        )
                    except (RoomControlError, hosted_rooms.HostedRoomError) as exc:
                        return str(exc)
                    except Exception:
                        logger.exception(
                            "Failed to open Group Chat from messaging picker"
                        )
                        return (
                            "Couldn’t load that Group Chat. "
                            f"Run `{rooms_command}` again."
                        )

                picker_sent = await self._try_send_choice_picker(
                    event,
                    session_key,
                    title=(
                        "👥 Group Chats\n"
                        "Choose a recent Group Chat to see its status, Bots, activity, and actions. "
                        f"All: {rooms_command} list"
                    ),
                    choices=choices,
                    on_choice_selected=_on_room_selected,
                )
                if picker_sent:
                    return None
            exact_name = next(
                (
                    room
                    for room in rooms
                    if str(room.get("name") or "").casefold() == query.casefold()
                ),
                None,
            )
            if exact_name is not None:
                return await asyncio.to_thread(
                    format_room_detail,
                    service,
                    exact_name,
                    room_command=rooms_command,
                    show_approvals=self._can_approve_group_chats(event),
                )
            list_parts = query.casefold().split()
            if not query or (list_parts and list_parts[0] == "list"):
                if len(list_parts) > 2 or (
                    len(list_parts) == 2 and not list_parts[1].isdecimal()
                ):
                    return f"Use `{rooms_command} list [page]`."
                page = int(list_parts[1]) if len(list_parts) == 2 else 1
                return await asyncio.to_thread(
                    format_room_list,
                    service,
                    rooms=rooms,
                    rooms_command=rooms_command,
                    page=page,
                )

            def _detail() -> str:
                room = resolve_room(rooms, query)
                return format_room_detail(
                    service,
                    room,
                    room_command=rooms_command,
                    show_approvals=self._can_approve_group_chats(event),
                )

            return await asyncio.to_thread(_detail)
        except (RoomControlError, hosted_rooms.HostedRoomError) as exc:
            return str(exc)
        except Exception:
            logger.exception("Failed to read Bot Group Chats from messaging")
            return "Couldn’t load Group Chats. Try again in a moment."

    async def _handle_room_command(self, event: MessageEvent) -> str:
        """Send to or stop work in a Bot Group Chat."""

        from gateway import hosted_rooms
        from gateway.hosted_room_messaging import (
            RoomControlError,
            current_room_backend,
            parse_room_command,
            resolve_room,
            room_reference,
            retry_room,
            send_to_room,
            stop_room,
            is_message_edit,
            is_machine_authored,
            list_messaging_rooms,
            messaging_event_id,
            relay_provenance_is_unknown,
        )

        if is_machine_authored(event):
            return "Group Chat controls are only available to people."
        if is_message_edit(event):
            return "Edited messages can’t run Group Chat commands. Send a new message."
        if relay_provenance_is_unknown(event):
            return (
                "Group Chat controls need a relay connector that reports whether the "
                "sender is a person or a bot. Update the connector and try again."
            )
        if not self._can_control_group_chats(event):
            return self._group_chat_control_denial(event)
        service = current_room_backend()
        rooms_command = f"{self._typed_command_prefix_for(event.source)}group"
        try:
            command = parse_room_command(
                self._group_chat_command_args(event),
                command_root=rooms_command,
            )
            if command.action in {
                "approve",
                "deny",
            } and not self._can_approve_group_chats(event):
                return self._group_chat_approval_denial()
            denial = self._group_chat_rate_limit_denial(
                event,
                action=command.action,
            )
            if denial:
                return denial
            if not command.room_query.isdecimal():
                if command.action == "send":
                    raise RoomControlError(
                        f"Use `{rooms_command} <room number> send <message>`."
                    )
                raise RoomControlError(f"Use `{rooms_command} <room number> stop`.")

            def _mutate() -> str:
                profile = self._group_chat_profile(event)
                rooms = list_messaging_rooms(
                    service,
                    profile=profile,
                )
                approval_command_id = f"approval:{messaging_event_id(event)}"
                approval_receipt = None
                if command.action in {"approve", "deny"}:
                    from gateway.hosted_room_messaging_approvals import (
                        approval_command,
                        terminalize_unowned_approval_commands,
                    )

                    terminalize_unowned_approval_commands(
                        service.db_path,
                        local_gateway_id=hosted_rooms.local_authority_gateway_id(),
                    )
                    approval_receipt = approval_command(
                        service.db_path,
                        command_id=approval_command_id,
                    )
                    if (
                        approval_receipt is not None
                        and approval_receipt["state"] == "completed"
                    ):
                        return str(
                            approval_receipt.get("result_text")
                            or "Approval is no longer available."
                        )
                if approval_receipt is None:
                    room = resolve_room(rooms, command.room_query)
                else:
                    room = next(
                        (
                            candidate
                            for candidate in rooms
                            if str(candidate.get("room_id") or "")
                            == str(approval_receipt["room_id"])
                            and str(candidate.get("authority_gateway_id") or "")
                            == str(approval_receipt["authority_gateway_id"])
                            and int(candidate.get("authority_epoch") or 0)
                            == int(approval_receipt["authority_epoch"])
                        ),
                        None,
                    )
                    if room is None:
                        raise RoomControlError(
                            "That approval is no longer available. Check Group Chats again."
                        )
                if room.get("_room_mode") == "desktop" and str(
                    room.get("room_id") or ""
                ).startswith("name:"):
                    raise RoomControlError(
                        "Open this older Group Chat once in the latest Hermes Desktop "
                        "before changing it from messaging."
                    )
                if command.action == "send":
                    result = send_to_room(service, room, event, command.message)
                    return f"{result} Check: `{rooms_command} {room_reference(room)}`."
                if command.action == "retry":
                    result = retry_room(service, room, event)
                    return f"{result} Check: `{rooms_command} {room_reference(room)}`."
                if command.action in {"approve", "deny"}:
                    from gateway.hosted_room_messaging_approvals import (
                        MessagingApprovalError,
                        approval_member_label,
                        submit_room_approval,
                    )

                    try:
                        _index, pending, applied = submit_room_approval(
                            service,
                            room,
                            command_id=approval_command_id,
                            choice=("once" if command.action == "approve" else "deny"),
                            installation_owner_authorized=(
                                self._can_approve_group_chats(event)
                            ),
                            selection=command.message,
                        )
                    except MessagingApprovalError as exc:
                        raise RoomControlError(str(exc)) from exc
                    bot = approval_member_label(
                        room,
                        str(pending["member_id"]),
                    )
                    if applied.get("applied") is False:
                        result = str(applied.get("result") or "Approval expired.")
                        return (
                            f"{result} Check: `{rooms_command} {room_reference(room)}`."
                        )
                    if applied.get("queued"):
                        result = f"Decision sent for {bot}."
                    elif command.action == "approve":
                        result = f"Approved once for {bot}."
                    else:
                        result = f"Denied for {bot}."
                    return f"{result} Check: `{rooms_command} {room_reference(room)}`."
                result = stop_room(service, room, event)
                return f"{result} Check: `{rooms_command} {room_reference(room)}`."

            return await asyncio.to_thread(_mutate)
        except (RoomControlError, hosted_rooms.HostedRoomError) as exc:
            return str(exc)
        except Exception:
            logger.exception("Failed to control hosted Bot room from messaging")
            return "Couldn’t update that Bot room. Try again in a moment."

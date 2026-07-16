"""Token-owning Discord client for the public-guild connector service.

This module is a platform edge, not an agent adapter.  It never imports model
tools or prompts and never interprets message meaning.  Its only decisions are
mechanical target type, live Discord permissions, exact allowlists, bounds, and
receipt verification.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Iterable, Mapping

from gateway.discord_connector_protocol import (
    DISCORD_CONNECTOR_THREAD_TARGET_TYPES,
    MAX_CONTENT_BYTES,
    MAX_CONTENT_CHARS,
    MAX_HISTORY_CONTENT_BYTES,
    MAX_HISTORY_MESSAGES,
    MAX_NAME_CHARS,
    DiscordConnectorEvent,
    DiscordConnectorHistoryAuthority,
    DiscordConnectorHistoryAuthorityKind,
    DiscordConnectorHistoryMessage,
    DiscordConnectorHistoryPage,
    DiscordConnectorTarget,
    DiscordConnectorTargetType,
    sha256_json,
)
from gateway.discord_connector_service import DiscordConnectorAcceptedMessage

try:
    import discord
except ImportError:  # pragma: no cover - production packaging must include it
    discord = None


class DiscordPublicConnectorError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class _FreshHistoryChannelView:
    """One REST-fetched channel paired with its REST-fetched thread parent.

    ``discord.Thread.permissions_for`` resolves its parent through the gateway
    cache.  History authorization cannot use that cache because a permission
    overwrite may have changed while the request was running.  This narrow
    view delegates thread permission calculation to the separately fetched
    parent while retaining the fetched thread's history endpoint.
    """

    source: Any
    parent: Any | None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.source, name)

    def permissions_for(self, principal: Any) -> Any:
        permission_source = self.parent if self.parent is not None else self.source
        permissions_for = getattr(permission_source, "permissions_for", None)
        if not callable(permissions_for):
            raise DiscordPublicConnectorError("history_acl_unavailable")
        return permissions_for(principal)


@dataclass(frozen=True)
class DiscordPublicConnectorPolicy:
    allowed_guild_ids: frozenset[str]
    allowed_channel_ids: frozenset[str]
    allowed_user_ids: frozenset[str]
    allowed_role_ids: frozenset[str]
    free_response_channel_ids: frozenset[str] = frozenset()
    public_only: bool = True
    author_policy: str = "exact_ids_or_roles"
    allow_bot_authors: bool = False
    require_mention: bool = True
    auto_thread: bool = True
    thread_require_mention: bool = False
    reviewed_cron_history_targets: tuple[tuple[str, frozenset[str]], ...] = ()

    @classmethod
    def build(
        cls,
        *,
        allowed_guild_ids: Iterable[str],
        allowed_channel_ids: Iterable[str],
        allowed_user_ids: Iterable[str],
        allowed_role_ids: Iterable[str],
        free_response_channel_ids: Iterable[str] = (),
        public_only: bool = True,
        author_policy: str = "exact_ids_or_roles",
        allow_bot_authors: bool = False,
        require_mention: bool = True,
        auto_thread: bool = True,
        thread_require_mention: bool = False,
        reviewed_cron_history_targets: Mapping[str, Iterable[str]] | None = None,
    ) -> "DiscordPublicConnectorPolicy":
        def _ids(
            values: Iterable[str],
            label: str,
            *,
            allow_empty: bool = False,
        ) -> frozenset[str]:
            result = frozenset(str(value) for value in values)
            if (not result and not allow_empty) or any(
                not value.isdigit() or value.startswith("0") for value in result
            ):
                raise ValueError(f"{label} must be a valid snowflake set")
            return result

        if any(
            type(value) is not bool
            for value in (
                allow_bot_authors,
                require_mention,
                auto_thread,
                thread_require_mention,
                public_only,
            )
        ):
            raise ValueError("Discord connector policy flags must be boolean")
        if author_policy not in {"exact_ids_or_roles", "guild_acl"}:
            raise ValueError("Discord connector author_policy is invalid")

        cron_targets: list[tuple[str, frozenset[str]]] = []
        raw_cron_targets = (
            {}
            if reviewed_cron_history_targets is None
            else reviewed_cron_history_targets
        )
        if not isinstance(raw_cron_targets, Mapping) or any(
            not isinstance(key, str) for key in raw_cron_targets
        ):
            raise ValueError("reviewed_cron_history_targets must be an object")
        for job_id, targets in sorted(raw_cron_targets.items()):
            if (
                len(job_id) != 12
                or any(char not in "0123456789abcdef" for char in job_id)
            ):
                raise ValueError("reviewed cron history job id is invalid")
            normalized_targets = _ids(
                targets,
                f"reviewed_cron_history_targets.{job_id}",
            )
            cron_targets.append((job_id, normalized_targets))
        return cls(
            allowed_guild_ids=_ids(allowed_guild_ids, "allowed_guild_ids"),
            allowed_channel_ids=_ids(allowed_channel_ids, "allowed_channel_ids"),
            allowed_user_ids=_ids(
                allowed_user_ids,
                "allowed_user_ids",
                allow_empty=author_policy == "guild_acl",
            ),
            allowed_role_ids=_ids(
                allowed_role_ids,
                "allowed_role_ids",
                allow_empty=author_policy == "guild_acl",
            ),
            free_response_channel_ids=_ids(
                free_response_channel_ids,
                "free_response_channel_ids",
                allow_empty=True,
            ),
            public_only=public_only,
            author_policy=author_policy,
            allow_bot_authors=allow_bot_authors,
            require_mention=require_mention,
            auto_thread=auto_thread,
            thread_require_mention=thread_require_mention,
            reviewed_cron_history_targets=tuple(cron_targets),
        )

    @property
    def reviewed_cron_history_target_map(self) -> dict[str, frozenset[str]]:
        return {
            job_id: targets
            for job_id, targets in self.reviewed_cron_history_targets
        }

    @property
    def reviewed_cron_history_targets_sha256(self) -> str:
        return sha256_json(
            {
                job_id: sorted(targets)
                for job_id, targets in self.reviewed_cron_history_targets
            }
        )

    @staticmethod
    def _channel_type(channel: Any) -> tuple[int | None, str]:
        channel_type = getattr(channel, "type", None)
        raw_value = getattr(channel_type, "value", channel_type)
        value = (
            raw_value
            if isinstance(raw_value, int) and not isinstance(raw_value, bool)
            else None
        )
        name = str(getattr(channel_type, "name", "") or "").strip().casefold()
        return value, name

    @staticmethod
    def _can_view(channel: Any, role: Any) -> bool:
        permissions_for = getattr(channel, "permissions_for", None)
        if role is None or not callable(permissions_for):
            return False
        try:
            return getattr(permissions_for(role), "view_channel", None) is True
        except Exception:
            return False

    @staticmethod
    def _can_read_history(channel: Any, role: Any) -> bool:
        permissions_for = getattr(channel, "permissions_for", None)
        if role is None or not callable(permissions_for):
            return False
        try:
            return getattr(permissions_for(role), "read_message_history", None) is True
        except Exception:
            return False

    def _prove_target(
        self,
        channel: Any,
        *,
        bot_user: Any = None,
        bot_member: Any = None,
        require_history: bool,
    ) -> DiscordConnectorTarget:
        guild = getattr(channel, "guild", None)
        guild_id = str(getattr(guild, "id", "") or "")
        channel_id = str(getattr(channel, "id", "") or "")
        if not guild_id or not channel_id or guild_id not in self.allowed_guild_ids:
            raise DiscordPublicConnectorError("target_not_allowed")

        value, name = self._channel_type(channel)
        if value == 12 or name == "private_thread":
            raise DiscordPublicConnectorError("private_target_forbidden")
        if value in {10, 11} or name in {"news_thread", "public_thread"}:
            parent = getattr(channel, "parent", None)
            parent_id = str(
                getattr(channel, "parent_id", None) or getattr(parent, "id", "") or ""
            )
            if not parent_id:
                raise DiscordPublicConnectorError("public_thread_parent_missing")
            target_type = (
                DiscordConnectorTargetType.PUBLIC_GUILD_THREAD
                if self.public_only
                else DiscordConnectorTargetType.GUILD_THREAD
            )
        elif value in {0, 5} or name in {"text", "news"}:
            parent = None
            parent_id = ""
            target_type = (
                DiscordConnectorTargetType.PUBLIC_GUILD_CHANNEL
                if self.public_only
                else DiscordConnectorTargetType.GUILD_CHANNEL
            )
        else:
            # Includes DM/group-DM, forum/category/voice/stage, and unknown types.
            raise DiscordPublicConnectorError("target_type_forbidden")

        if self.public_only:
            allowed_ids = {channel_id}
            if parent_id:
                allowed_ids.add(parent_id)
            if not (allowed_ids & self.allowed_channel_ids):
                raise DiscordPublicConnectorError("target_not_allowed")
        else:
            # Production guild-ACL lanes remain a closed capability set.  A
            # root channel must itself be allowlisted; a public thread is
            # authorized only through its exact allowlisted parent.  Discord
            # ACLs prove who may use that lane, not which new lane Hermes may
            # silently add to its scope.
            allowlist_root_id = parent_id or channel_id
            if allowlist_root_id not in self.allowed_channel_ids:
                raise DiscordPublicConnectorError("target_not_allowed")

        if self.public_only:
            default_role = getattr(guild, "default_role", None)
            if not self._can_view(channel, default_role):
                raise DiscordPublicConnectorError("target_not_public")
            if parent is not None and not self._can_view(parent, default_role):
                raise DiscordPublicConnectorError("target_parent_not_public")
            if require_history and not self._can_read_history(channel, default_role):
                raise DiscordPublicConnectorError("target_history_not_public")
            if (
                require_history
                and parent is not None
                and not self._can_read_history(parent, default_role)
            ):
                raise DiscordPublicConnectorError("target_parent_history_not_public")

        if bot_user is not None or bot_member is not None:
            # A caller that supplies a freshly REST-fetched member is making
            # an explicit live-role proof.  Never replace it with cached
            # ``guild.me``.
            member = (
                bot_member
                if bot_member is not None
                else getattr(guild, "me", None) or bot_user
            )
            permissions_for = getattr(channel, "permissions_for", None)
            if not callable(permissions_for):
                raise DiscordPublicConnectorError("bot_permissions_unavailable")
            try:
                permissions = permissions_for(member)
            except Exception as exc:
                raise DiscordPublicConnectorError(
                    "bot_permissions_unavailable"
                ) from exc
            if getattr(permissions, "view_channel", None) is not True:
                raise DiscordPublicConnectorError("bot_cannot_view_target")
            if getattr(permissions, "read_message_history", None) is not True:
                raise DiscordPublicConnectorError("bot_cannot_read_target_history")
            if target_type in DISCORD_CONNECTOR_THREAD_TARGET_TYPES:
                if getattr(permissions, "send_messages_in_threads", None) is not True:
                    raise DiscordPublicConnectorError("bot_cannot_send_target")
            elif getattr(permissions, "send_messages", None) is not True:
                raise DiscordPublicConnectorError("bot_cannot_send_target")

        return DiscordConnectorTarget(
            target_type=target_type,
            guild_id=guild_id,
            channel_id=channel_id,
            parent_channel_id=parent_id or None,
        )

    def prove_target(
        self,
        channel: Any,
        *,
        bot_user: Any = None,
        bot_member: Any = None,
    ) -> DiscordConnectorTarget:
        return self._prove_target(
            channel,
            bot_user=bot_user,
            bot_member=bot_member,
            require_history=False,
        )

    def prove_history_target(
        self,
        channel: Any,
        *,
        bot_user: Any = None,
        bot_member: Any = None,
    ) -> DiscordConnectorTarget:
        return self._prove_target(
            channel,
            bot_user=bot_user,
            bot_member=bot_member,
            require_history=True,
        )

    def prove_history_authority(
        self,
        channel: Any,
        *,
        target: DiscordConnectorTarget,
        authority: DiscordConnectorHistoryAuthority,
        requester_member: Any = None,
    ) -> None:
        """Mechanically prove one internal requester/job against live ACL."""

        if not isinstance(authority, DiscordConnectorHistoryAuthority):
            raise DiscordPublicConnectorError("history_authority_invalid")
        if authority.kind is DiscordConnectorHistoryAuthorityKind.REVIEWED_CRON:
            targets = self.reviewed_cron_history_target_map.get(
                str(authority.cron_job_id or "")
            )
            if targets is None or target.channel_id not in targets:
                raise DiscordPublicConnectorError("cron_history_target_not_reviewed")
            return

        requester_id = str(authority.requester_user_id or "")
        member_id = str(getattr(requester_member, "id", "") or "")
        if not requester_id or member_id != requester_id:
            raise DiscordPublicConnectorError("history_requester_not_resolved")
        # The isolated canary is bound to one exact canary requester in its
        # config. Production guild_acl mode instead follows each member's live
        # Discord ACL without a static user allowlist.
        if self.public_only and requester_id not in self.allowed_user_ids:
            raise DiscordPublicConnectorError("history_requester_not_allowed")

        permissions_for = getattr(channel, "permissions_for", None)
        if not callable(permissions_for):
            raise DiscordPublicConnectorError("history_requester_acl_unavailable")
        try:
            permissions = permissions_for(requester_member)
        except Exception as exc:
            raise DiscordPublicConnectorError(
                "history_requester_acl_unavailable"
            ) from exc
        if (
            getattr(permissions, "view_channel", None) is not True
            or getattr(permissions, "read_message_history", None) is not True
        ):
            raise DiscordPublicConnectorError("history_requester_cannot_read")

        if target.target_type in DISCORD_CONNECTOR_THREAD_TARGET_TYPES:
            parent = getattr(channel, "parent", None)
            parent_permissions_for = getattr(parent, "permissions_for", None)
            if not callable(parent_permissions_for):
                raise DiscordPublicConnectorError(
                    "history_requester_parent_acl_unavailable"
                )
            try:
                parent_permissions = parent_permissions_for(requester_member)
            except Exception as exc:
                raise DiscordPublicConnectorError(
                    "history_requester_parent_acl_unavailable"
                ) from exc
            if (
                getattr(parent_permissions, "view_channel", None) is not True
                or getattr(parent_permissions, "read_message_history", None) is not True
            ):
                raise DiscordPublicConnectorError(
                    "history_requester_cannot_read_parent"
                )

    def event_from_message(
        self,
        message: Any,
        *,
        bot_user: Any = None,
        connector_thread_ids: frozenset[str] = frozenset(),
    ) -> DiscordConnectorEvent:
        target = self.prove_target(getattr(message, "channel", None), bot_user=bot_user)
        author = getattr(message, "author", None)
        author_id = str(getattr(author, "id", "") or "")
        author_is_bot = getattr(author, "bot", None) is True
        author_roles = getattr(author, "roles", ())
        role_ids = {
            str(getattr(role, "id", "") or "")
            for role in author_roles
            if str(getattr(role, "id", "") or "").isdigit()
        }
        if author_is_bot and not self.allow_bot_authors:
            raise DiscordPublicConnectorError("bot_author_forbidden")
        if self.author_policy == "exact_ids_or_roles":
            if (
                author_id not in self.allowed_user_ids
                and not (role_ids & self.allowed_role_ids)
            ):
                raise DiscordPublicConnectorError("author_not_allowed")
        else:
            channel = getattr(message, "channel", None)
            permissions_for = getattr(channel, "permissions_for", None)
            if not callable(permissions_for):
                raise DiscordPublicConnectorError("author_permissions_unavailable")
            try:
                author_permissions = permissions_for(author)
            except Exception as exc:
                raise DiscordPublicConnectorError(
                    "author_permissions_unavailable"
                ) from exc
            send_permission = (
                "send_messages_in_threads"
                if target.target_type in DISCORD_CONNECTOR_THREAD_TARGET_TYPES
                else "send_messages"
            )
            if (
                getattr(author_permissions, "view_channel", None) is not True
                or getattr(author_permissions, send_permission, None) is not True
            ):
                raise DiscordPublicConnectorError("author_not_allowed_by_guild_acl")

        bot_user_id = str(getattr(bot_user, "id", "") or "")
        mentioned_ids = {
            str(getattr(member, "id", "") or "")
            for member in getattr(message, "mentions", ())
        }
        mentioned_ids.update(str(value) for value in getattr(message, "raw_mentions", ()))
        target_root_id = target.parent_channel_id or target.channel_id
        free_response = target_root_id in self.free_response_channel_ids
        is_guild_thread = target.target_type in DISCORD_CONNECTOR_THREAD_TARGET_TYPES
        mention_required = self.require_mention and not free_response and not (
            is_guild_thread and not self.thread_require_mention
        )
        if mention_required and (
            not bot_user_id or bot_user_id not in mentioned_ids
        ):
            raise DiscordPublicConnectorError("mention_required")

        message_type = getattr(message, "type", None)
        type_value = getattr(message_type, "value", message_type)
        type_name = str(getattr(message_type, "name", "") or "").casefold()
        if type_value not in {0, 19} and type_name not in {"default", "reply"}:
            raise DiscordPublicConnectorError("message_type_unsupported")

        content = getattr(message, "content", None)
        if (
            not isinstance(content, str)
            or not content
            or "\x00" in content
            or len(content) > MAX_CONTENT_CHARS
            or len(content.encode("utf-8")) > MAX_CONTENT_BYTES
        ):
            raise DiscordPublicConnectorError("message_content_invalid")
        event_id = str(getattr(message, "id", "") or "")
        if not event_id.isdigit() or event_id.startswith("0"):
            raise DiscordPublicConnectorError("message_id_invalid")
        author_name = str(
            getattr(author, "display_name", None)
            or getattr(author, "name", None)
            or author_id
        )
        reference = getattr(message, "reference", None)
        reply_id_raw = getattr(reference, "message_id", None) if reference else None
        reply_id = str(reply_id_raw) if reply_id_raw is not None else None
        created_at = getattr(message, "created_at", None)
        if isinstance(created_at, datetime):
            created_at_ms = int(created_at.timestamp() * 1_000)
        else:
            created_at_ms = int(time.time() * 1_000)
        return DiscordConnectorEvent.from_mapping({
            "event_id": event_id,
            "target": target.to_mapping(),
            "author_id": author_id,
            "author_name": author_name,
            "author_is_bot": author_is_bot,
            "content": content,
            "created_at_unix_ms": created_at_ms,
            "reply_to_message_id": reply_id,
        })


class DiscordPublicConnectorClient:
    """Dedicated discord.py loop owned only by the connector service."""

    def __init__(
        self,
        token: str,
        *,
        policy: DiscordPublicConnectorPolicy,
        event_sink: Callable[[DiscordConnectorEvent], object],
        ready_timeout_seconds: float = 30,
        request_timeout_seconds: float = 15,
    ) -> None:
        if discord is None:
            raise RuntimeError("discord.py is required by the connector service")
        if (
            not isinstance(token, str)
            or not token
            or len(token) > 512
            or any(char.isspace() for char in token)
        ):
            raise ValueError("Discord connector token is invalid")
        if not isinstance(policy, DiscordPublicConnectorPolicy):
            raise TypeError("Discord connector policy is invalid")
        if not callable(event_sink):
            raise TypeError("Discord connector event sink is invalid")
        if not 0 < ready_timeout_seconds <= 120:
            raise ValueError("Discord connector ready timeout is invalid")
        if not 0 < request_timeout_seconds <= 30:
            raise ValueError("Discord connector request timeout is invalid")

        intents = discord.Intents.none()
        intents.guilds = True
        intents.guild_messages = True
        intents.message_content = True
        intents.dm_messages = False
        self._client = discord.Client(intents=intents)
        self._token = token
        self.policy = policy
        self.event_sink = event_sink
        self.ready_timeout_seconds = float(ready_timeout_seconds)
        self.request_timeout_seconds = float(request_timeout_seconds)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._closing = threading.Event()
        self._health_failed = threading.Event()
        self._created_thread_ids: set[str] = set()

        @self._client.event
        async def on_ready() -> None:
            self._ready.set()

        @self._client.event
        async def on_disconnect() -> None:
            self._mark_disconnected()

        @self._client.event
        async def on_message(message: Any) -> None:
            await self._handle_inbound_message(message)

    async def _handle_inbound_message(self, message: Any) -> None:
        """Validate and emit one inbound event, with bounded thread fallback."""

        if (
            self._client.user is not None
            and getattr(message.author, "id", None) == self._client.user.id
        ):
            return
        try:
            event = self.policy.event_from_message(
                message,
                bot_user=self._client.user,
                connector_thread_ids=frozenset(self._created_thread_ids),
            )
        except DiscordPublicConnectorError:
            return

        if (
            self.policy.auto_thread
            and event.target.target_type in {
                DiscordConnectorTargetType.PUBLIC_GUILD_CHANNEL,
                DiscordConnectorTargetType.GUILD_CHANNEL,
            }
        ):
            # Auto-threading is a delivery preference, not permission to lose
            # an already authenticated team task. Discord may reject thread
            # creation transiently (Forbidden/HTTP error) or the returned
            # thread may fail our structural proof. In either case preserve
            # the original, already-proven channel event exactly once.
            try:
                create_thread = getattr(message, "create_thread", None)
                if not callable(create_thread):
                    raise DiscordPublicConnectorError("auto_thread_unavailable")
                thread = await create_thread(
                    name=f"Muncho {event.event_id}",
                    auto_archive_duration=1440,
                    reason="Muncho guild task thread",
                )
                target = self.policy.prove_target(
                    thread,
                    bot_user=self._client.user,
                )
                self._created_thread_ids.add(target.channel_id)
                event = DiscordConnectorEvent.from_mapping(
                    {**event.to_mapping(), "target": target.to_mapping()}
                )
            except Exception:
                pass
        self.event_sink(event)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(token=<redacted>, "
            f"guilds={len(self.policy.allowed_guild_ids)}, "
            f"channels={len(self.policy.allowed_channel_ids)})"
        )

    def _mark_disconnected(self) -> None:
        was_ready = self._ready.is_set()
        self._ready.clear()
        if was_ready and not self._closing.is_set():
            self._health_failed.set()

    def readiness_identity(self) -> dict[str, Any]:
        """Return the exact non-secret identity of the live Discord session.

        This is a mechanical readiness boundary for the privileged connector
        service.  It is intentionally unavailable before Discord's real
        ``on_ready`` event and never includes token material or a token digest.
        """

        thread = self._thread
        loop = self._loop
        user = getattr(self._client, "user", None)
        user_id = str(getattr(user, "id", "") or "")
        is_ready = getattr(self._client, "is_ready", None)
        is_closed = getattr(self._client, "is_closed", None)
        intents = getattr(self._client, "intents", None)
        if (
            not self._ready.is_set()
            or self._stopped.is_set()
            or self._health_failed.is_set()
            or thread is None
            or not thread.is_alive()
            or loop is None
            or not loop.is_running()
            or not callable(is_ready)
            or is_ready() is not True
            or not callable(is_closed)
            or is_closed() is not False
            or intents is None
            or getattr(intents, "guilds", None) is not True
            or getattr(intents, "guild_messages", None) is not True
            or getattr(intents, "message_content", None) is not True
            or getattr(intents, "dm_messages", None) is not False
            or not user_id.isdigit()
            or user_id.startswith("0")
        ):
            raise DiscordPublicConnectorError("discord_not_ready")
        target_proofs = []
        for channel_id in sorted(self.policy.allowed_channel_ids):
            send_target = self.prove_public_target(channel_id)
            history_target = self.prove_public_history_target(channel_id)
            if send_target != history_target:
                raise DiscordPublicConnectorError("public_target_binding_changed")
            target_proofs.append(send_target.to_mapping())
        return {
            "discord_gateway_ready": True,
            "bot_user_id": user_id,
            "intents": ["guilds", "guild_messages", "message_content"],
            "dm_messages": False,
            "require_mention": self.policy.require_mention,
            "auto_thread": self.policy.auto_thread,
            "thread_require_mention": self.policy.thread_require_mention,
            "public_only": self.policy.public_only,
            "author_policy": self.policy.author_policy,
            "free_response_channel_ids": sorted(
                self.policy.free_response_channel_ids
            ),
            "allowed_user_ids": sorted(self.policy.allowed_user_ids),
            "allowed_role_ids": sorted(self.policy.allowed_role_ids),
            "allowed_channel_ids": sorted(self.policy.allowed_channel_ids),
            "reviewed_cron_history_targets_sha256": (
                self.policy.reviewed_cron_history_targets_sha256
            ),
            "public_target_proofs": target_proofs,
        }

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._thread_main,
            name="discord-public-connector-client",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(self.ready_timeout_seconds):
            self.stop()
            raise DiscordPublicConnectorError("discord_ready_timeout")

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._client.start(self._token))
        finally:
            self._ready.clear()
            if not self._closing.is_set():
                self._health_failed.set()
            self._stopped.set()
            loop.close()

    def stop(self) -> None:
        self._closing.set()
        self._ready.clear()
        loop = self._loop
        if loop is not None and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self._client.close(), loop)
            try:
                future.result(timeout=self.request_timeout_seconds)
            except Exception:
                pass
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self.request_timeout_seconds)
        self._thread = None

    def wait_for_health_failure(self, timeout: float | None = None) -> bool:
        """Wait for a terminal/disconnected Discord session without secrets."""

        return self._health_failed.wait(timeout)

    def _submit(self, coroutine: Any, *, deadline_unix_ms: int | None = None) -> Any:
        loop = self._loop
        if (
            loop is None
            or not loop.is_running()
            or not self._ready.is_set()
            or self._health_failed.is_set()
        ):
            if hasattr(coroutine, "close"):
                coroutine.close()
            raise DiscordPublicConnectorError("discord_not_ready")
        timeout = self.request_timeout_seconds
        if deadline_unix_ms is not None:
            remaining = (deadline_unix_ms - int(time.time() * 1_000)) / 1_000
            if remaining <= 0:
                if hasattr(coroutine, "close"):
                    coroutine.close()
                raise DiscordPublicConnectorError("discord_deadline_expired")
            timeout = min(timeout, remaining)
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise DiscordPublicConnectorError("discord_request_timeout") from exc

    async def _resolve_channel(self, channel_id: str) -> Any:
        channel = self._client.get_channel(int(channel_id))
        if channel is None:
            channel = await self._client.fetch_channel(int(channel_id))
        return channel

    async def _fetch_fresh_history_channel(
        self, channel_id: str
    ) -> _FreshHistoryChannelView:
        """Fetch channel and thread-parent ACL state from REST without cache."""

        fetch_channel = getattr(self._client, "fetch_channel", None)
        if not callable(fetch_channel):
            raise DiscordPublicConnectorError("history_live_acl_unavailable")
        try:
            source = await fetch_channel(int(channel_id))
        except Exception as exc:
            raise DiscordPublicConnectorError(
                "history_live_acl_unavailable"
            ) from exc
        if str(getattr(source, "id", "") or "") != channel_id:
            raise DiscordPublicConnectorError("history_target_binding_changed")

        value, name = self.policy._channel_type(source)
        parent = None
        if value in {10, 11} or name in {"news_thread", "public_thread"}:
            parent_id = str(getattr(source, "parent_id", "") or "")
            if not parent_id.isdigit() or parent_id.startswith("0"):
                raise DiscordPublicConnectorError("public_thread_parent_missing")
            try:
                parent = await fetch_channel(int(parent_id))
            except Exception as exc:
                raise DiscordPublicConnectorError(
                    "history_live_parent_acl_unavailable"
                ) from exc
            source_guild_id = str(
                getattr(getattr(source, "guild", None), "id", "") or ""
            )
            parent_guild_id = str(
                getattr(getattr(parent, "guild", None), "id", "") or ""
            )
            if (
                str(getattr(parent, "id", "") or "") != parent_id
                or not source_guild_id
                or parent_guild_id != source_guild_id
            ):
                raise DiscordPublicConnectorError(
                    "history_parent_target_binding_changed"
                )
        return _FreshHistoryChannelView(source=source, parent=parent)

    async def _fetch_fresh_bot_member(self, channel: Any) -> Any:
        """Fetch the current bot member/roles from the exact target guild."""

        bot_user = getattr(self._client, "user", None)
        bot_id = str(getattr(bot_user, "id", "") or "")
        guild = getattr(channel, "guild", None)
        guild_id = str(getattr(guild, "id", "") or "")
        fetch_member = getattr(guild, "fetch_member", None)
        if (
            not bot_id.isdigit()
            or bot_id.startswith("0")
            or not guild_id
            or not callable(fetch_member)
        ):
            raise DiscordPublicConnectorError("bot_live_member_unavailable")
        try:
            member = await fetch_member(int(bot_id))
        except Exception as exc:
            raise DiscordPublicConnectorError(
                "bot_live_member_unavailable"
            ) from exc
        member_id = str(getattr(member, "id", "") or "")
        member_guild = getattr(member, "guild", None)
        member_guild_id = str(getattr(member_guild, "id", "") or "")
        if member_id != bot_id or (
            member_guild is not None and member_guild_id != guild_id
        ):
            raise DiscordPublicConnectorError("bot_identity_binding_changed")
        return member

    def prove_public_target(self, channel_id: str) -> DiscordConnectorTarget:
        async def _prove() -> DiscordConnectorTarget:
            channel = await self._fetch_fresh_history_channel(channel_id)
            bot_member = await self._fetch_fresh_bot_member(channel)
            return self.policy.prove_target(
                channel,
                bot_user=self._client.user,
                bot_member=bot_member,
            )

        return self._submit(_prove())

    def prove_public_history_target(
        self, channel_id: str
    ) -> DiscordConnectorTarget:
        async def _prove() -> DiscordConnectorTarget:
            channel = await self._fetch_fresh_history_channel(channel_id)
            return self.policy.prove_history_target(
                channel,
                bot_user=self._client.user,
            )

        return self._submit(_prove())

    async def _resolve_history_requester(
        self,
        channel: Any,
        authority: DiscordConnectorHistoryAuthority,
        *,
        refresh: bool,
    ) -> Any:
        if authority.kind is DiscordConnectorHistoryAuthorityKind.REVIEWED_CRON:
            return None
        guild = getattr(channel, "guild", None)
        requester_id = str(authority.requester_user_id or "")
        if guild is None or not requester_id.isdigit() or requester_id.startswith("0"):
            raise DiscordPublicConnectorError("history_requester_not_resolved")
        fetch_member = getattr(guild, "fetch_member", None)
        get_member = getattr(guild, "get_member", None)
        member = None
        if refresh and callable(fetch_member):
            try:
                member = await fetch_member(int(requester_id))
            except Exception as exc:
                raise DiscordPublicConnectorError(
                    "history_requester_not_resolved"
                ) from exc
        if member is None and callable(get_member):
            member = get_member(int(requester_id))
        if member is None and callable(fetch_member):
            try:
                member = await fetch_member(int(requester_id))
            except Exception as exc:
                raise DiscordPublicConnectorError(
                    "history_requester_not_resolved"
                ) from exc
        if str(getattr(member, "id", "") or "") != requester_id:
            raise DiscordPublicConnectorError("history_requester_not_resolved")
        return member

    def fetch_guild_history(
        self,
        channel_id: str,
        *,
        limit: int,
        before_message_id: str | None,
        after_message_id: str | None,
        authority: DiscordConnectorHistoryAuthority,
    ) -> DiscordConnectorHistoryPage:
        async def _fetch() -> DiscordConnectorHistoryPage:
            if (
                isinstance(limit, bool)
                or not isinstance(limit, int)
                or not 1 <= limit <= MAX_HISTORY_MESSAGES
                or (before_message_id is not None and after_message_id is not None)
            ):
                raise DiscordPublicConnectorError("history_query_invalid")
            if not isinstance(authority, DiscordConnectorHistoryAuthority):
                raise DiscordPublicConnectorError("history_authority_invalid")
            channel = await self._fetch_fresh_history_channel(channel_id)
            target = self.policy.prove_history_target(
                channel,
                bot_user=self._client.user,
            )
            if target.channel_id != channel_id:
                raise DiscordPublicConnectorError("public_target_binding_changed")
            requester = await self._resolve_history_requester(
                channel,
                authority,
                refresh=True,
            )
            self.policy.prove_history_authority(
                channel,
                target=target,
                authority=authority,
                requester_member=requester,
            )
            history = getattr(channel, "history", None)
            if not callable(history):
                raise DiscordPublicConnectorError("history_unavailable")
            kwargs: dict[str, Any] = {"limit": limit + 1}
            if before_message_id is not None:
                kwargs["before"] = discord.Object(id=int(before_message_id))
            if after_message_id is not None:
                kwargs["after"] = discord.Object(id=int(after_message_id))
                kwargs["oldest_first"] = True
            observed = [item async for item in history(**kwargs)]
            has_more = len(observed) > limit
            selected = observed[:limit]
            if after_message_id is None:
                selected.reverse()

            messages: list[DiscordConnectorHistoryMessage] = []
            for item in selected:
                item_channel_id = str(
                    getattr(getattr(item, "channel", None), "id", "") or ""
                )
                if item_channel_id != target.channel_id:
                    raise DiscordPublicConnectorError("history_target_binding_changed")
                raw_content = getattr(item, "content", None)
                if not isinstance(raw_content, str) or "\x00" in raw_content:
                    raise DiscordPublicConnectorError("history_content_invalid")
                bounded = raw_content[:MAX_CONTENT_CHARS]
                encoded = bounded.encode("utf-8")
                if len(encoded) > MAX_HISTORY_CONTENT_BYTES:
                    bounded = encoded[:MAX_HISTORY_CONTENT_BYTES].decode(
                        "utf-8", errors="ignore"
                    )
                author = getattr(item, "author", None)
                author_id = str(getattr(author, "id", "") or "")
                author_name = str(
                    getattr(author, "display_name", None)
                    or getattr(author, "name", None)
                    or author_id
                )[:MAX_NAME_CHARS]
                created_at = getattr(item, "created_at", None)
                if not isinstance(created_at, datetime):
                    raise DiscordPublicConnectorError("history_timestamp_invalid")
                reference = getattr(item, "reference", None)
                reply_raw = (
                    getattr(reference, "message_id", None) if reference else None
                )
                messages.append(
                    DiscordConnectorHistoryMessage.from_mapping(
                        {
                            "message_id": str(getattr(item, "id", "") or ""),
                            "author_id": author_id,
                            "author_name": author_name,
                            "author_is_bot": getattr(author, "bot", None) is True,
                            "content": bounded,
                            "content_truncated": bounded != raw_content,
                            "created_at_unix_ms": int(created_at.timestamp() * 1_000),
                            "reply_to_message_id": (
                                str(reply_raw) if reply_raw is not None else None
                            ),
                        }
                    )
                )

            # A second REST-fetched structured proof prevents a permission or
            # parent overwrite flip during the bounded read from escaping the
            # connector boundary.  Never reuse the gateway-cached channel.
            refreshed_channel = await self._fetch_fresh_history_channel(channel_id)
            if self.policy.prove_history_target(
                refreshed_channel,
                bot_user=self._client.user,
            ) != target:
                raise DiscordPublicConnectorError("public_target_binding_changed")
            refreshed_requester = await self._resolve_history_requester(
                refreshed_channel,
                authority,
                refresh=True,
            )
            self.policy.prove_history_authority(
                refreshed_channel,
                target=target,
                authority=authority,
                requester_member=refreshed_requester,
            )
            return DiscordConnectorHistoryPage.from_mapping(
                {
                    "target": target.to_mapping(),
                    "messages": [item.to_mapping() for item in messages],
                    "query": {
                        "limit": limit,
                        "before_message_id": before_message_id,
                        "after_message_id": after_message_id,
                    },
                    "has_more": has_more,
                    "order": "oldest_to_newest",
                }
            )

        return self._submit(_fetch())

    def send_public_message(
        self,
        target: DiscordConnectorTarget,
        content: str,
        *,
        reply_to_message_id: str | None,
        deadline_unix_ms: int,
    ) -> DiscordConnectorAcceptedMessage:
        async def _send() -> DiscordConnectorAcceptedMessage:
            # Target, parent and bot permissions are REST-refreshed at the
            # last responsible moment.  Never authorize egress from the
            # gateway cache.
            channel = await self._fetch_fresh_history_channel(target.channel_id)
            bot_member = await self._fetch_fresh_bot_member(channel)
            live_target = self.policy.prove_target(
                channel,
                bot_user=self._client.user,
                bot_member=bot_member,
            )
            if live_target != target:
                raise DiscordPublicConnectorError("public_target_binding_changed")
            kwargs: dict[str, Any] = {
                "allowed_mentions": discord.AllowedMentions.none(),
            }
            if reply_to_message_id is not None:
                kwargs["reference"] = discord.Object(id=int(reply_to_message_id))
                kwargs["mention_author"] = False
            sent = await channel.source.send(content, **kwargs)
            # A second independent REST proof and exact readback close the
            # receipt boundary even if permissions changed during dispatch.
            refreshed_channel = await self._fetch_fresh_history_channel(
                target.channel_id
            )
            refreshed_bot_member = await self._fetch_fresh_bot_member(
                refreshed_channel
            )
            if (
                self.policy.prove_target(
                    refreshed_channel,
                    bot_user=self._client.user,
                    bot_member=refreshed_bot_member,
                )
                != target
            ):
                raise DiscordPublicConnectorError("public_target_binding_changed")
            readback = await refreshed_channel.source.fetch_message(int(sent.id))
            reference = getattr(readback, "reference", None)
            reply_raw = getattr(reference, "message_id", None) if reference else None
            observed_reply_to_message_id = (
                str(reply_raw) if reply_raw is not None else None
            )
            verified = (
                str(getattr(readback, "id", "")) == str(sent.id)
                and str(getattr(getattr(readback, "channel", None), "id", ""))
                == target.channel_id
                and self._client.user is not None
                and str(getattr(getattr(readback, "author", None), "id", ""))
                == str(self._client.user.id)
                and getattr(readback, "content", None) == content
                and observed_reply_to_message_id == reply_to_message_id
            )
            return DiscordConnectorAcceptedMessage(
                message_id=str(sent.id),
                readback_verified=verified,
            )

        return self._submit(_send(), deadline_unix_ms=deadline_unix_ms)


__all__ = [
    "DiscordPublicConnectorClient",
    "DiscordPublicConnectorError",
    "DiscordPublicConnectorPolicy",
]

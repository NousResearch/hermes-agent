"""Discord lifecycle methods; runtime dependencies remain on the adapter facade."""

from __future__ import annotations

from pathlib import Path as _Path

import asyncio
from typing import Any, Dict, Optional
try:
    import discord
    from discord import Message as DiscordMessage
except ImportError:
    discord = None
    DiscordMessage = Any


class DiscordLifecycleMixin:
    def _config_value(self, key: str, default: Any, *, env_key: Optional[str] = None) -> Any:
        """Resolve a liveness value from profile config, legacy env, or default."""
        from . import adapter as _adapter

        extra = self.config.extra if isinstance(getattr(self.config, "extra", None), dict) else {}
        value = extra.get(key)
        if value is None and env_key:
            value = _adapter.os.getenv(env_key)
        return default if value is None or value == "" else value

    def _finite_positive_config_float(
        self, key: str, default: float, *, env_key: Optional[str] = None
    ) -> float:
        """Resolve a finite positive liveness duration; invalid values disable it."""
        from . import adapter as _adapter

        try:
            value = float(self._config_value(key, default, env_key=env_key))
        except (TypeError, ValueError):
            return 0.0
        return value if _adapter.math.isfinite(value) and value > 0 else 0.0

    def _config_int(self, key: str, default: int, *, env_key: Optional[str] = None) -> int:
        """Resolve a positive liveness count; invalid values disable it."""
        from . import adapter as _adapter

        value = self._config_value(key, default, env_key=env_key)
        if isinstance(value, bool):
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _handle_bot_task_done(self, task: asyncio.Task) -> None:
        """Surface post-startup discord.py task exits as a retryable fatal so GatewayRunner
        re-queues us (otherwise the websocket is dead while the gateway process lives)."""
        from . import adapter as _adapter

        if getattr(self, "_disconnecting", False):
            # Intentional shutdown: drain the result to avoid "exception was never retrieved".
            with _adapter.suppress(_adapter.asyncio.CancelledError, Exception):
                task.exception()
            return
        # Ignore stale callbacks from an older client after a reconnect installed a newer task.
        if self._bot_task is not None and task is not self._bot_task:
            with _adapter.suppress(_adapter.asyncio.CancelledError, Exception):
                task.exception()
            return
        if not self._running:
            # Startup failures are handled in connect(); this is only for post-startup exits.
            with _adapter.suppress(_adapter.asyncio.CancelledError, Exception):
                task.exception()
            return
        try:
            exc = task.exception()
        except _adapter.asyncio.CancelledError:
            return
        except Exception as err:  # pragma: no cover - defensive
            exc = err
        if exc is None:
            message = "Discord gateway task exited without an exception"
        else:
            message = f"Discord gateway task exited: {exc}"
        _adapter.logger.error("[%s] %s", self.name, message, exc_info=exc if exc else False)
        self._set_fatal_error("discord_gateway_task_exited", message, retryable=True)

        async def _notify() -> None:
            try:
                await self._notify_fatal_error()
            except Exception as notify_exc:  # pragma: no cover - defensive logging
                _adapter.logger.warning(
                    "[%s] Failed to notify gateway supervisor about Discord task exit: %s",
                    self.name, notify_exc, exc_info=True,
                )
        _adapter.asyncio.create_task(_notify())

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Connect to Discord and start receiving events."""
        from . import adapter as _adapter

        if not _adapter.DISCORD_AVAILABLE:
            _adapter.logger.error("[%s] discord.py not installed. Run: pip install discord.py", self.name)
            self._set_fatal_error("missing_dependency", "discord.py not installed", retryable=False)
            return False
        if not _adapter.discord.opus.is_loaded():
            _adapter._load_opus_codec()
        if not self.config.token:
            _adapter.logger.error("[%s] No bot token configured", self.name)
            self._set_fatal_error("missing_credentials", "No bot token configured", retryable=False)
            return False
        try:
            if not self._acquire_platform_lock('discord-bot-token', self.config.token, 'Discord bot token'):
                return False
            # Snapshot gate env inside the owning profile's scope (immune to the first-writer-wins bridge).
            # Snapshot this profile's gate env vars (issue #72348): connect() runs inside the owning
            # profile's runtime scope under multiplex, so the snapshot holds THIS adapter's values, immune
            # to the first-writer-wins process-global env bridge.
            self._snapshot_gate_env()
            self._allowed_user_ids = self._get_allowed_users()
            # DISCORD_ALLOWED_ROLES: comma-separated role IDs; ANY match grants access.
            self._allowed_role_ids = self._get_allowed_roles()
            # Intents: Server Members only when usernames must be resolved — an unenabled privileged
            # intent can keep the bot offline. ``"*"`` is the open-mode wildcard, not a username.
            intents = _adapter.Intents.default()
            intents.message_content = True
            intents.dm_messages = True
            intents.guild_messages = True
            intents.members = _adapter._needs_server_members_intent(
                self._allowed_user_ids, self._allowed_role_ids,
            )
            intents.voice_states = True
            # Resolve proxy (DISCORD_PROXY > generic env vars > macOS system proxy)
            from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_bot
            proxy_url = resolve_proxy_url(platform_env_var="DISCORD_PROXY")
            if proxy_url:
                _adapter.logger.info("[%s] Using proxy for Discord: %s", self.name, proxy_url)
            # proxy= for HTTP, connector= for SOCKS; allowed_mentions per _build_allowed_mentions.
            # Close any existing client first: a zombie client also fires on_message -> double responses.
            # Without this, the old client remains connected to Discord gateway and both fire on_message,
            # causing double responses. See #18187.
            if self._client is not None:
                try:
                    if not self._client.is_closed():
                        await self._client.close()
                except Exception:
                    _adapter.logger.debug("[%s] Failed to close previous Discord client", self.name)
                finally:
                    self._client = None
                    self._ready_event.clear()
            self._client = _adapter.commands.Bot(
                command_prefix="!",  # Not really used, we handle raw messages
                intents=intents,
                allowed_mentions=_adapter._build_allowed_mentions(),
                **proxy_kwargs_for_bot(proxy_url),
            )
            adapter_self = self  # capture for closure

            @self._client.event
            async def on_ready():
                _adapter.logger.info("[%s] Connected as %s", adapter_self.name, adapter_self._client.user)
                await adapter_self._resolve_allowed_usernames()
                adapter_self._ready_event.set()
                if adapter_self._post_connect_task and not adapter_self._post_connect_task.done():
                    adapter_self._post_connect_task.cancel()
                adapter_self._post_connect_task = _adapter.asyncio.create_task(
                    adapter_self._run_post_connect_initialization()
                )
                if adapter_self._missed_message_backfill_enabled():
                    adapter_self._ensure_missed_message_backfill_task()

            @self._client.event
            async def on_message(message: DiscordMessage):
                await adapter_self._dispatch_discord_message(message)

            @self._client.event
            async def on_message_edit(before: DiscordMessage, after: DiscordMessage):
                await adapter_self._on_platform_message_edit(before, after)

            @self._client.event
            async def on_message_delete(message: DiscordMessage):
                await adapter_self._on_platform_message_delete(message)

            @self._client.event
            async def on_thread_create(thread):
                await adapter_self._on_platform_thread_create(thread)

            @self._client.event
            async def on_thread_update(before, after):
                await adapter_self._on_platform_thread_update(before, after)

            @self._client.event
            async def on_voice_state_update(member, before, after):
                """Track voice channel join/leave events."""
                bot_guild_ids = set(adapter_self._voice_clients.keys())
                if not bot_guild_ids:
                    return
                guild_id = member.guild.id
                if guild_id not in bot_guild_ids:
                    return
                if member == adapter_self._client.user:
                    return
                joined = before.channel is None and after.channel is not None
                left = before.channel is not None and after.channel is None
                switched = (
                    before.channel is not None
                    and after.channel is not None
                    and before.channel != after.channel
                )
                if joined or left or switched:
                    _adapter.logger.info(
                        "Voice state: %s (%d) %s (guild %d)",
                        member.display_name,
                        member.id,
                        "joined " + after.channel.name if joined
                        else "left " + before.channel.name if left
                        else f"moved {before.channel.name} -> {after.channel.name}",
                        guild_id,
                    )
            if self._slash_commands:
                self._register_slash_commands()
            self._disconnecting = False
            self._bot_task = _adapter.asyncio.create_task(self._client.start(self.config.token))
            self._bot_task.add_done_callback(self._handle_bot_task_done)
            ready_timeout = _adapter._discord_ready_timeout_seconds()
            # Wait for ready, failing fast if the startup task dies first (e.g. SOCKS errors).
            await _adapter._wait_for_ready_or_bot_exit(
                self._ready_event, self._bot_task,
                timeout=None if ready_timeout <= 0 else ready_timeout,
            )
            self._running = True
            self._start_liveness_probe()
            # Plugin-registered native handlers (discord.py Bot — add_listener()/event hooks).
            self._wire_plugin_handlers(self._client)
            return True
        except _adapter.asyncio.TimeoutError:
            _adapter.logger.error("[%s] Timeout waiting for connection to Discord", self.name, exc_info=True)
            # Cancel the bot task so a discarded adapter can't fire on_message (two clients answering).
            await self._cancel_bot_task()
            self._release_platform_lock()
            # Always set an explicit fatal code: a code-less failure makes the gateway guess "transient".
            self._set_fatal_error(
                "discord_connect_timeout",
                "Timed out waiting for the Discord gateway to become ready", retryable=True,
            )
            return False
        except Exception as e:  # pragma: no cover - defensive logging
            _adapter.logger.error("[%s] Failed to connect to Discord: %s", self.name, e, exc_info=True)
            # Same zombie-client hazard: client.start() may already run when a later step raises.
            await self._cancel_bot_task()
            self._release_platform_lock()
            # Classify by exception TYPE: auth/permission failures can't self-heal, so
            # retryable=False drops them from the reconnect queue and surfaces them as fatal.
            code, message, retryable = self._classify_connect_exception(e)
            self._set_fatal_error(code, message, retryable=retryable)
            return False

    def _classify_connect_exception(self, error: Exception) -> tuple:
        """Map a startup exception to ``(code, message, retryable)`` by TYPE only (never message
        text); unknown types stay retryable — a false terminal leaves a recovered platform dead."""
        from . import adapter as _adapter

        def _is(type_name: str) -> bool:
            # Class-name check covers mocked discord.py / failed imports; isinstance adds subclasses.
            if error.__class__.__name__ == type_name:
                return True
            try:
                import discord as _discord
                exc_type = getattr(_discord, type_name, None)
                return isinstance(exc_type, type) and isinstance(error, exc_type)
            except Exception:
                return False
        if _is("LoginFailure"):
            return (
                "discord_auth_error",
                f"Discord bot token rejected: {error}. The token is invalid or "
                "was revoked — regenerate it in the Discord Developer Portal "
                "and update DISCORD_BOT_TOKEN.",
                False,
            )
        if _is("PrivilegedIntentsRequired"):
            # Name the exact intents requested (Server Members only when allowlists need lookups).
            # See #79430.
            guidance = _adapter._format_privileged_intents_guidance(
                needs_members=_adapter._needs_server_members_intent(
                    getattr(self, "_allowed_user_ids", None),
                    getattr(self, "_allowed_role_ids", None),
                )
            )
            return ("discord_intents_required", guidance, False)
        return ("discord_connect_error", f"Discord startup failed: {error}", True)

    def _discord_message_admission(self, message: Any, *, claim: bool) -> tuple[bool, bool]:
        """Return ``(admitted, role_authorized)`` for one Discord event."""
        from . import adapter as _adapter

        message_id = str(getattr(message, "id", ""))
        if claim:
            if self._dedup.is_duplicate(message_id):
                return False, False
        elif self._dedup.contains(message_id):
            return False, False
        if message.author == self._client.user:
            return False, False
        if message.type not in {_adapter.discord.MessageType.default, _adapter.discord.MessageType.reply}:
            return False, False
        role_authorized = False
        if getattr(message.author, "bot", False):
            allow_bots = self._get_allow_bots()
            if allow_bots == "none":
                return False, False
            if allow_bots == "mentions" and not self._self_is_explicitly_mentioned(message):
                return False, False
            if (
                self._discord_bots_require_inline_mention()
                and not self._self_is_raw_mentioned(message)
            ):
                return False, False
        else:
            msg_guild = getattr(message, "guild", None)
            is_dm = isinstance(message.channel, _adapter.discord.DMChannel) or msg_guild is None
            msg_channel_ids = None
            if not is_dm:
                msg_channel_ids = {str(message.channel.id)}
                parent_id = self._get_parent_channel_id(message.channel)
                if parent_id:
                    msg_channel_ids.add(parent_id)
            if not self._is_allowed_user(
                str(message.author.id), message.author, guild=msg_guild, is_dm=is_dm,
                channel_ids=msg_channel_ids,
            ):
                self._warn_if_fail_closed_default()
                return False, False
            role_authorized = bool(getattr(self, "_allowed_role_ids", set()))
        raw_self_mention = self._self_is_explicitly_mentioned(message)
        if not isinstance(message.channel, _adapter.discord.DMChannel) and (
            message.mentions or raw_self_mention
        ):
            other_bots_mentioned = any(
                mentioned.bot and mentioned != self._client.user
                for mentioned in message.mentions
            )
            if other_bots_mentioned and not raw_self_mention:
                return False, False
            ignore_no_mention = _adapter.os.getenv(
                "DISCORD_IGNORE_NO_MENTION", "true"
            ).lower() in {"true", "1", "yes"}
            if ignore_no_mention and not raw_self_mention and not other_bots_mentioned:
                parent_id = None
                if hasattr(message.channel, "parent_id") and message.channel.parent_id:
                    parent_id = str(message.channel.parent_id)
                free_channels = self._discord_free_response_channels()
                channel_keys = self._discord_channel_keys(message, parent_id)
                if "*" not in free_channels and not (channel_keys & free_channels):
                    return False, False
        return True, role_authorized

    async def _dispatch_discord_message(self, message: Any) -> bool:
        """Apply Discord ingress policy and dispatch one live event."""
        from . import adapter as _adapter

        if not self._ready_event.is_set():
            try:
                await _adapter.asyncio.wait_for(self._ready_event.wait(), timeout=30.0)
            except _adapter.asyncio.TimeoutError:
                pass
        admitted, role_authorized = self._discord_message_admission(message, claim=True)
        if not admitted:
            return False
        return await self._handle_message(message, role_authorized=role_authorized)

    def _thread_id_and_chat_for_channel(self, channel) -> tuple[Optional[str], Optional[str]]:
        """Return ``(thread_id, chat_id)``; for a thread chat_id is the thread id (dispatch session key)."""
        from . import adapter as _adapter

        if channel is None:
            return None, None
        chan_id = getattr(channel, "id", None)
        if chan_id is None:
            return None, None
        is_thread = isinstance(channel, getattr(_adapter.discord, "Thread", ()))
        return (str(chan_id) if is_thread else None), str(chan_id)

    def _source_for_platform_event(
        self, *, chat_id: str, user_id: Optional[str], user_name: Optional[str],
        thread_id: Optional[str], guild_id: Optional[str], message_id: Optional[str] = None,
    ):
        """Build the SessionSource the gateway authorizes against; missing identity raises (fail closed)."""
        if not user_id or not chat_id:
            raise ValueError("gateway_platform_event requires actor and chat identities")
        return self.build_source(
            chat_id=chat_id, chat_type="thread" if thread_id else "group", user_id=user_id,
            user_name=user_name, thread_id=thread_id, guild_id=guild_id, message_id=message_id,
        )

    async def _fire_platform_event(self, event: Dict[str, Any], source) -> None:
        """Forward one envelope to the gateway boundary; no callback -> fail closed, errors never escape."""
        from . import adapter as _adapter

        handler = getattr(self, "_platform_event_handler", None)
        if handler is None:
            return
        try:
            await handler(event, source)
        except Exception:
            _adapter.logger.debug("[%s] gateway_platform_event dispatch error", self.name, exc_info=True)

    @staticmethod
    def _platform_events_subscribed() -> bool:
        """has_hook fast-path shared by every Discord fire-site."""
        try:
            from hermes_cli.lifecycle import has_hook
            return has_hook("gateway_platform_event")
        except Exception:
            return False

    async def _emit_platform_event(self, event_type: str, build) -> None:
        """Normalize one event via ``build()`` -> ``(payload, source_kwargs)`` (None drops) and dispatch."""
        from . import adapter as _adapter

        if not self._platform_events_subscribed():
            return
        try:
            built = build()
            if built is None:
                return
            payload, source_kwargs = built
            event = {"platform": "discord", "event_type": event_type, "payload": payload}
            source = self._source_for_platform_event(**source_kwargs)
        except Exception:
            _adapter.logger.debug("[%s] %s normalize error", self.name, event_type, exc_info=True)
            return
        await self._fire_platform_event(event, source)

    def _message_event_parts(self, message, extra_payload):
        """Shared normalizer for message edit/delete: (payload, source kwargs) or None."""
        from . import adapter as _adapter

        author = getattr(message, "author", None)
        if author is not None and getattr(author, "bot", False):
            return None  # bot's own progressive edits are noise, not user events
        thread_id, chat_id = self._thread_id_and_chat_for_channel(getattr(message, "channel", None))
        message_id = getattr(message, "id", None)
        if chat_id is None or message_id is None:
            return None
        guild = getattr(message, "guild", None)
        payload = {
            "chat_id": str(chat_id)[:128], "message_id": str(message_id)[:128],
            "thread_id": thread_id[:128] if thread_id else None, **extra_payload(message, author),
        }
        return payload, dict(
            chat_id=str(chat_id), user_id=str(getattr(author, "id", "") or "") or None,
            user_name=getattr(author, "display_name", None), thread_id=thread_id,
            guild_id=str(getattr(guild, "id", "")) if guild else None, message_id=str(message_id),
        )

    @staticmethod
    def _thread_event_parts(thread, extra_payload):
        """Shared normalizer for thread create/rename; the owner is the authorized actor
        because Discord's event carries none (same trade-off as ``message_deleted``)."""
        from . import adapter as _adapter

        thread_id = getattr(thread, "id", None)
        owner_id = getattr(thread, "owner_id", None)
        if thread_id is None:
            return None
        parent_id = getattr(thread, "parent_id", None)
        guild = getattr(thread, "guild", None)
        payload = {
            "thread_id": str(thread_id)[:128],
            "parent_chat_id": str(parent_id)[:128] if parent_id is not None else None,
            **extra_payload(thread, owner_id),
        }
        return payload, dict(
            chat_id=str(thread_id), user_id=str(owner_id) if owner_id is not None else None,
            user_name=None, thread_id=str(thread_id),
            guild_id=str(getattr(guild, "id", "")) if guild else None,
        )

    async def _on_platform_message_edit(self, before, after) -> None:
        """Normalize ``on_message_edit`` into event_type ``message_edited``."""
        from . import adapter as _adapter

        def _extra(message, author):
            text = getattr(message, "content", None)
            edited_at = getattr(message, "edited_at", None)
            return {
                "text": text[:8192] if isinstance(text, str) else None,
                "edited_at": (
                    str(edited_at.isoformat())[:64]
                    if edited_at is not None and hasattr(edited_at, "isoformat")
                    else None
                ),
            }
        message = after if after is not None else before
        await self._emit_platform_event("message_edited", lambda: self._message_event_parts(message, _extra))

    async def _on_platform_message_delete(self, message) -> None:
        """Normalize ``on_message_delete`` into ``message_deleted``. Discord omits the
        deleter, so the author (the only cached identity) is the source; uncached deletions never fire."""
        from . import adapter as _adapter

        def _extra(message, author):
            return {"author_id": str(getattr(author, "id", "") or "")[:128] or None}
        await self._emit_platform_event("message_deleted", lambda: self._message_event_parts(message, _extra))

    async def _on_platform_thread_create(self, thread) -> None:
        """Normalize ``on_thread_create`` into event_type ``thread_created``."""
        from . import adapter as _adapter

        def _extra(thread, owner_id):
            name = getattr(thread, "name", None)
            return {
                "name": name[:256] if isinstance(name, str) else None,
                "owner_id": str(owner_id)[:128] if owner_id is not None else None,
            }
        await self._emit_platform_event("thread_created", lambda: self._thread_event_parts(thread, _extra))

    async def _on_platform_thread_update(self, before, after) -> None:
        """Normalize ``on_thread_update`` renames into ``thread_renamed``; non-rename updates are dropped."""
        from . import adapter as _adapter

        def _build():
            old_name = getattr(before, "name", None)
            new_name = getattr(after, "name", None)
            if old_name == new_name or not isinstance(new_name, str):
                return None
            return self._thread_event_parts(after, lambda _t, _o: {
                "old_name": old_name[:256] if isinstance(old_name, str) else None,
                "new_name": new_name[:256],
            })
        await self._emit_platform_event("thread_renamed", _build)

    async def _cancel_bot_task(self) -> None:
        """Cancel and await the background client.start() task, if running."""
        from . import adapter as _adapter

        if self._bot_task and not self._bot_task.done():
            self._bot_task.cancel()
            try:
                await self._bot_task
            except (_adapter.asyncio.CancelledError, Exception):
                pass
        self._bot_task = None

    def _start_liveness_probe(self) -> None:
        """Start the periodic Gateway WS health probe (REST success doesn't prove event delivery)."""
        from . import adapter as _adapter

        if (
            self._liveness_interval_seconds <= 0
            or self._liveness_failure_threshold <= 0
            or self._heartbeat_ack_max_age_seconds <= 0
            or self._max_latency_seconds <= 0
        ):
            return
        if self._liveness_task and not self._liveness_task.done():
            return
        self._liveness_task = _adapter.asyncio.create_task(self._liveness_loop())

    def _read_websocket_health(self, client: Any) -> tuple[bool, str]:
        """Return current Discord Gateway health without making a REST request."""
        from . import adapter as _adapter

        try:
            ready = bool(client.is_ready())
        except Exception:
            return False, "not_ready"
        if not ready:
            return False, "not_ready"
        try:
            if client.is_closed():
                return False, "client_closed"
        except Exception:
            return False, "client_closed"
        websocket = getattr(client, "ws", None)
        try:
            socket_open = bool(websocket is not None and getattr(websocket, "open", False))
        except Exception:
            # A transport that can't report open state isn't a usable event stream: treat as unhealthy.
            return False, "socket_state_unavailable"
        if not socket_open:
            return False, "socket_closed"
        keep_alive = getattr(websocket, "_keep_alive", None)
        last_ack = getattr(keep_alive, "_last_ack", None)
        if not isinstance(last_ack, (int, float)):
            return False, "ack_unavailable"
        ack_age = _adapter.time.perf_counter() - last_ack
        if not _adapter.math.isfinite(ack_age) or ack_age > self._heartbeat_ack_max_age_seconds:
            return False, "ack_stale"
        latency = getattr(client, "latency", None)
        if not isinstance(latency, (int, float)) or not _adapter.math.isfinite(latency):
            return False, "latency_non_finite"
        if latency > self._max_latency_seconds:
            return False, "latency_exceeded"
        return True, "healthy"

    async def _liveness_loop(self) -> None:
        """Force a reconnect after repeated unhealthy Discord Gateway samples."""
        from . import adapter as _adapter

        interval = self._liveness_interval_seconds
        threshold = self._liveness_failure_threshold
        failures = 0
        while self._running:
            try:
                await _adapter.asyncio.sleep(interval)
            except _adapter.asyncio.CancelledError:
                return
            client = self._client
            if not self._running or client is None or self._disconnecting:
                return
            try:
                healthy, reason = self._read_websocket_health(client)
            except Exception:
                # Fail closed: a discord.py attribute change must not kill this watchdog silently.
                healthy = False
                reason = "health_check_error"
            if healthy:
                failures = 0
                continue
            failures += 1
            _adapter.logger.warning(
                "[%s] Discord Gateway WebSocket unhealthy (%s, %d/%d)", self.name, reason, failures,
                threshold,
            )
            if failures < threshold:
                continue
            # Mark recovery before closing: Bot.start()'s done callback must not overwrite this reason.
            self._disconnecting = True
            _adapter.logger.error(
                "[%s] Discord Gateway WebSocket remained unhealthy (%s); forcing reconnect",
                self.name, reason,
            )
            self._set_fatal_error(
                "discord_websocket_health_stale",
                f"Discord Gateway WebSocket health check failed: {reason}", retryable=True,
            )
            self._liveness_notification_task = _adapter.asyncio.create_task(
                self._notify_liveness_fatal_error(client)
            )
            return

    async def _notify_liveness_fatal_error(self, client: Any) -> None:
        """Close the failed client, then notify the runner outside the sampler (which must not
        await itself via ``disconnect()``); the runner owns the bounded teardown."""
        from . import adapter as _adapter

        failed_websocket = getattr(client, "ws", None)
        try:
            close_task = _adapter.asyncio.create_task(client.close())
            try:
                done, _pending = await _adapter.asyncio.wait({close_task}, timeout=1.0)
                if close_task not in done:
                    raise _adapter.asyncio.TimeoutError
                await close_task
            except _adapter.asyncio.TimeoutError:
                _adapter.logger.warning("[%s] Timed out closing unhealthy Discord client", self.name)
                close_task.cancel()
                close_task.add_done_callback(_adapter._consume_background_task_result)
                closing_task = getattr(client, "_closing_task", None)
                if isinstance(closing_task, _adapter.asyncio.Task):
                    closing_task.cancel()
                    closing_task.add_done_callback(_adapter._consume_background_task_result)
                    # Client.close() caches this task; clear it before the runner's disconnect retries.
                    client._closing_task = None
                try:
                    if _adapter._abort_discord_websocket_transport(failed_websocket):
                        _adapter.logger.warning(
                            "[%s] Aborted unresponsive Discord WebSocket transport", self.name,
                        )
                except Exception:
                    _adapter.logger.debug(
                        "[%s] Error aborting unhealthy Discord WebSocket transport", self.name,
                        exc_info=True,
                    )
            except Exception:
                _adapter.logger.debug("[%s] Error closing unhealthy Discord client", self.name, exc_info=True)
            # Runner may run disconnect() elsewhere; drop the self-ref so it can't cancel this callback.
            if self._liveness_notification_task is _adapter.asyncio.current_task():
                self._liveness_notification_task = None
            await self._notify_fatal_error()
        except Exception:
            _adapter.logger.debug("[%s] Fatal-error handler raised", self.name, exc_info=True)

    async def _cancel_liveness_task(self) -> None:
        """Cancel and await liveness tasks without awaiting the current task."""
        from . import adapter as _adapter

        current = _adapter.asyncio.current_task()
        for task_name in ("_liveness_task", "_liveness_notification_task"):
            task = getattr(self, task_name, None)
            if task is None:
                continue
            if task is current:
                continue
            if not task.done():
                task.cancel()
            try:
                await task
            except _adapter.asyncio.CancelledError:
                pass
            except Exception:
                _adapter.logger.debug("[%s] Liveness task shutdown failed", self.name, exc_info=True)
            setattr(self, task_name, None)

    async def cancel_background_tasks(self) -> None:
        """Cancel background tasks, but first flush pending text-batch sends (cancelling
        ``_pending_text_batch_tasks`` mid-send dropped replies); the flush deadline stays below the
        gateway's per-adapter disconnect budget so the outer ``wait_for`` can't hard-cancel us."""
        from . import adapter as _adapter

        pending = list(self._pending_text_batch_tasks.values())
        if pending:
            _adapter.logger.info(
                "[%s] Flushing %d pending text-batch task(s) before shutdown",
                self.name, len(pending),
            )
            try:
                await _adapter.asyncio.wait_for(
                    _adapter.asyncio.gather(*pending, return_exceptions=True),
                    timeout=self._text_batch_flush_deadline_seconds(),
                )
            except _adapter.asyncio.TimeoutError:
                _adapter.logger.warning(
                    "[%s] Text-batch flush timed out; cancelling remaining tasks", self.name,
                )
                for task in pending:
                    if not task.done():
                        task.cancel()
        self._pending_text_batch_tasks.clear()
        self._pending_text_batches.clear()
        await super().cancel_background_tasks()

    def _text_batch_flush_deadline_seconds(self) -> float:
        """Deadline for flushing pending text batches during shutdown: strictly below the gateway's
        per-adapter disconnect budget so its outer ``wait_for`` can't cancel the flush first."""
        from . import adapter as _adapter

        budget = 5.0  # mirrors gateway _ADAPTER_DISCONNECT_TIMEOUT_SECS_DEFAULT
        raw = _adapter.os.getenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "").strip()
        if raw:
            try:
                parsed = float(raw)
                if parsed > 0:
                    budget = parsed
            except ValueError:
                pass
        # Reserve ~20% (min 0.5s) headroom, hard-capped at 90% so the floor can't exceed the budget.
        headroom = max(0.5, budget * 0.2)
        deadline = max(1.0, budget - headroom)
        return min(deadline, budget * 0.9)

    async def disconnect(self) -> None:
        """Disconnect from Discord."""
        from . import adapter as _adapter

        self._disconnecting = True
        # Cancel the liveness probe first so it can't fire a spurious fatal/reconnect mid-teardown.
        await self._cancel_liveness_task()
        # Leave voice *before* cancelling the bot task: VoiceClient.disconnect() needs the main
        # gateway WS (run by the bot task) or it blocks until the timeout.
        for guild_id in list(self._voice_clients.keys()):
            try:
                await self.leave_voice_channel(guild_id)
            except Exception as e:  # pragma: no cover - defensive logging
                _adapter.logger.debug("[%s] Error leaving voice channel %s: %s", self.name, guild_id, e)
        # Cancel the bot task before closing: after a connect() timeout client.start() may still run
        # and discord.py's reconnect loop can ignore the closed flag mid-handshake.
        await self._cancel_bot_task()
        if self._client:
            try:
                await self._client.close()
            except Exception as e:  # pragma: no cover - defensive logging
                _adapter.logger.warning("[%s] Error during disconnect: %s", self.name, e, exc_info=True)
        for task in (self._post_connect_task, self._missed_message_backfill_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except _adapter.asyncio.CancelledError:
                    pass
        self._running = False
        self._client = None
        self._ready_event.clear()
        self._post_connect_task = None
        self._liveness_task = None
        self._missed_message_backfill_task = None
        self._release_platform_lock()
        _adapter.logger.info("[%s] Disconnected", self.name)

    def _command_sync_state_path(self) -> _Path:
        from . import adapter as _adapter

        from hermes_constants import get_hermes_home
        directory = get_hermes_home() / _adapter._DISCORD_COMMAND_SYNC_STATE_SUBDIR
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return directory / _adapter._DISCORD_COMMAND_SYNC_STATE_FILENAME

    def _read_command_sync_state(self) -> dict:
        from . import adapter as _adapter

        try:
            path = self._command_sync_state_path()
            if not path.exists():
                return {}
            data = _adapter.json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _write_command_sync_state(self, state: dict) -> None:
        from . import adapter as _adapter

        _adapter.atomic_json_write(
            self._command_sync_state_path(), state, indent=None, separators=(",", ":"),
        )

    def _command_sync_state_key(self, app_id: Any) -> str:
        from . import adapter as _adapter

        return str(app_id or "unknown")

    def _desired_command_sync_fingerprint(self) -> str:
        from . import adapter as _adapter

        tree = self._client.tree if self._client else None
        desired = []
        if tree is not None:
            desired = [
                self._canonicalize_app_command_payload(command.to_dict(tree))
                for command in tree.get_commands()
            ]
        desired.sort(key=lambda item: (item.get("type", 1), item.get("name", "")))
        payload = _adapter.json.dumps(desired, sort_keys=True, separators=(",", ":"))
        return _adapter.hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _command_sync_skip_reason(self, app_id: Any, fingerprint: str) -> Optional[str]:
        from . import adapter as _adapter

        entry = self._read_command_sync_state().get(self._command_sync_state_key(app_id))
        if not isinstance(entry, dict):
            return None
        now = _adapter.time.time()
        retry_after_until = float(entry.get("retry_after_until") or 0)
        if retry_after_until > now:
            remaining = max(1, int(retry_after_until - now))
            return f"Discord asked us to wait before syncing slash commands; retry in {remaining}s"
        last_success_at = float(entry.get("last_success_at") or 0)
        last_attempt_at = float(entry.get("last_attempt_at") or 0)
        if (
            entry.get("fingerprint") == fingerprint
            and last_success_at
            and last_success_at >= last_attempt_at
        ):
            return "same slash-command fingerprint already synced"
        return None

    def _update_command_sync_entry(self, app_id: Any, fingerprint: str, *, keep_existing: bool, drop=(), fields=None) -> None:
        """Rewrite this app's sync-state entry (optionally merged over the existing one).
        ``fields`` is a callable of ``now`` so timestamps derive from one clock read."""
        from . import adapter as _adapter

        state = self._read_command_sync_state()
        key = self._command_sync_state_key(app_id)
        entry = dict(state.get(key)) if keep_existing and isinstance(state.get(key), dict) else {}
        for name in drop:
            entry.pop(name, None)
        now = _adapter.time.time()
        state[key] = {**entry, "fingerprint": fingerprint, "last_attempt_at": now, **(fields(now) if fields else {})}
        self._write_command_sync_state(state)

    def _record_command_sync_attempt(self, app_id: Any, fingerprint: str) -> None:
        self._update_command_sync_entry(app_id, fingerprint, keep_existing=True, drop=("last_success_at", "summary"))

    def _record_command_sync_rate_limit(self, app_id: Any, fingerprint: str, retry_after: float) -> None:
        from . import adapter as _adapter

        retry_after = max(1.0, float(retry_after))
        self._update_command_sync_entry(
            app_id, fingerprint, keep_existing=True,
            fields=lambda now: {"retry_after_until": _adapter.time.time() + retry_after, "retry_after": retry_after},
        )

    def _record_command_sync_success(self, app_id: Any, fingerprint: str, summary: dict) -> None:
        from . import adapter as _adapter

        self._update_command_sync_entry(
            app_id, fingerprint, keep_existing=False,
            fields=lambda now: {"last_success_at": _adapter.time.time(), "summary": summary},
        )

    @staticmethod
    def _extract_discord_retry_after(exc: BaseException) -> Optional[float]:
        value = getattr(exc, "retry_after", None)
        if value is not None:
            try:
                return max(1.0, float(value))
            except (TypeError, ValueError):
                return None
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers:
            for key in ("Retry-After", "X-RateLimit-Reset-After"):
                try:
                    raw = headers.get(key)
                except Exception:
                    raw = None
                if raw is None:
                    continue
                try:
                    return max(1.0, float(raw))
                except (TypeError, ValueError):
                    continue
        return None

    @staticmethod
    def _is_discord_rate_limit(exc: BaseException) -> bool:
        """True only for Discord 429 rate-limit exceptions (``RateLimited`` or HTTPException
        status 429) — narrower than ``hasattr(exc, 'retry_after')``."""
        # isinstance-of-class guard: a mocked ``discord`` module has MagicMock attrs, not types.
        from . import adapter as _adapter

        if _adapter.DISCORD_AVAILABLE and _adapter.discord is not None:
            for attr_name in ("RateLimited", "HTTPException"):
                cls = getattr(_adapter.discord, attr_name, None)
                if not isinstance(cls, type):
                    continue
                if isinstance(exc, cls):
                    if attr_name == "RateLimited":
                        return True
                    status = getattr(exc, "status", None)
                    if status == 429:
                        return True
        # Duck-type fallback: rate-limit-ish name plus numeric retry_after (mocks, exotic transports).
        name = type(exc).__name__.lower()
        if ("ratelimit" in name or "rate_limit" in name) and getattr(exc, "retry_after", None) is not None:
            return True
        response = getattr(exc, "response", None)
        status = getattr(response, "status", None) or getattr(response, "status_code", None)
        return status == 429

    @staticmethod
    def _is_discord_unknown_interaction(exc: BaseException) -> bool:
        """True for Discord's expired interaction token error."""
        from . import adapter as _adapter

        code = getattr(exc, "code", None)
        if code is None:
            data = getattr(exc, "data", None)
            if isinstance(data, dict):
                code = data.get("code")
        try:
            code = int(code)
        except (TypeError, ValueError):
            code = None
        status = getattr(exc, "status", None)
        response = getattr(exc, "response", None)
        if status is None and response is not None:
            status = getattr(response, "status", None) or getattr(response, "status_code", None)
        try:
            status = int(status)
        except (TypeError, ValueError):
            status = None
        message = str(exc).lower()
        return code == 10062 or (status == 404 and "unknown interaction" in message)

    def _command_sync_mutation_interval_seconds(self) -> float:
        from . import adapter as _adapter

        return _adapter._DISCORD_COMMAND_SYNC_MUTATION_INTERVAL_SECONDS

    async def _sleep_between_command_sync_mutations(self) -> None:
        from . import adapter as _adapter

        interval = self._command_sync_mutation_interval_seconds()
        if interval > 0:
            await _adapter.asyncio.sleep(interval)

    async def _run_post_connect_initialization(self) -> None:
        """Finish non-critical startup work after Discord is connected."""
        from . import adapter as _adapter

        if not self._client:
            return
        try:
            sync_policy = self._get_discord_command_sync_policy()
            if sync_policy == "off":
                _adapter.logger.info("[%s] Skipping Discord slash command sync (policy=off)", self.name)
                return
            if sync_policy == "bulk":
                synced = await _adapter.asyncio.wait_for(self._client.tree.sync(), timeout=30)
                _adapter.logger.info("[%s] Synced %d slash command(s) via bulk tree sync", self.name, len(synced))
                return
            app_id = getattr(self._client, "application_id", None) or getattr(getattr(self._client, "user", None), "id", None)
            fingerprint = self._desired_command_sync_fingerprint()
            skip_reason = self._command_sync_skip_reason(app_id, fingerprint)
            if skip_reason:
                _adapter.logger.info("[%s] Skipping Discord slash command sync: %s", self.name, skip_reason)
                return
            self._record_command_sync_attempt(app_id, fingerprint)
            http = getattr(self._client, "http", None)
            has_ratelimit_timeout = http is not None and hasattr(http, "max_ratelimit_timeout")
            previous_ratelimit_timeout = getattr(http, "max_ratelimit_timeout", None) if has_ratelimit_timeout else None
            if has_ratelimit_timeout:
                http.max_ratelimit_timeout = _adapter._DISCORD_COMMAND_SYNC_MAX_RATE_LIMIT_SLEEP_SECONDS
            try:
                # The command-management bucket is small and discord.py may sleep long on a 429: bound it.
                summary = await _adapter.asyncio.wait_for(self._safe_sync_slash_commands(), timeout=600)
            except Exception as e:
                if not self._is_discord_rate_limit(e):
                    raise
                retry_after = self._extract_discord_retry_after(e)
                if retry_after is None:
                    # Rate-limited with no retry-after: back off a conservative default.
                    retry_after = _adapter._DISCORD_COMMAND_SYNC_MAX_RATE_LIMIT_SLEEP_SECONDS
                self._record_command_sync_rate_limit(app_id, fingerprint, retry_after)
                _adapter.logger.warning(
                    "[%s] Discord rate-limited slash command sync; retrying after %.0fs", self.name,
                    retry_after,
                )
                return
            finally:
                if has_ratelimit_timeout:
                    http.max_ratelimit_timeout = previous_ratelimit_timeout
            self._record_command_sync_success(app_id, fingerprint, summary)
            _adapter.logger.info(
                "[%s] Safely reconciled %d slash command(s): unchanged=%d updated=%d recreated=%d created=%d deleted=%d",
                self.name, summary["total"], summary["unchanged"], summary["updated"],
                summary["recreated"], summary["created"], summary["deleted"],
            )
        except _adapter.asyncio.TimeoutError:
            _adapter.logger.warning(
                "[%s] Slash command sync timed out — Discord rate-limit bucket "
                "may be saturated; will retry on next reconnect",
                self.name,
            )
        except _adapter.asyncio.CancelledError:
            raise
        except Exception as e:  # pragma: no cover - defensive logging
            _adapter.logger.warning("[%s] Slash command sync failed: %s", self.name, e, exc_info=True)

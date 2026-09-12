"""Discord connection and background-task lifecycle."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from contextlib import suppress
from typing import Any, Optional

from .. import adapter as _adapter

logger = _adapter.logger
discord = _adapter.discord
DISCORD_AVAILABLE = _adapter.DISCORD_AVAILABLE
Intents = _adapter.Intents
commands = _adapter.commands


def _load_opus_codec():
    return _adapter._load_opus_codec()


def _needs_server_members_intent(*args):
    return _adapter._needs_server_members_intent(*args)


def _format_privileged_intents_guidance(*args, **kwargs):
    return _adapter._format_privileged_intents_guidance(*args, **kwargs)


def _discord_ready_timeout_seconds():
    return _adapter._discord_ready_timeout_seconds()


async def _wait_for_ready_or_bot_exit(*args, **kwargs):
    return await _adapter._wait_for_ready_or_bot_exit(*args, **kwargs)


def _build_allowed_mentions(extra=None):
    return _adapter._build_allowed_mentions(extra)


def _discord_available():
    return _adapter.DISCORD_AVAILABLE

def _abort_discord_websocket_transport(websocket: Any) -> bool:
    return _adapter._abort_discord_websocket_transport(websocket)


def _consume_background_task_result(task: asyncio.Task) -> None:
    return _adapter._consume_background_task_result(task)


class LifecycleMixin:
    """Own Discord connection, liveness, and shutdown orchestration."""
    def _handle_bot_task_done(self, task: asyncio.Task) -> None:
        """Surface post-startup discord.py task exits as a retryable fatal so GatewayRunner
        re-queues us (otherwise the websocket is dead while the gateway process lives)."""
        if getattr(self, "_disconnecting", False):
            # Intentional shutdown: drain the result to avoid "exception was never retrieved".
            with suppress(asyncio.CancelledError, Exception):
                task.exception()
            return
        # Ignore stale callbacks from an older client after a reconnect installed a newer task.
        if self._bot_task is not None and task is not self._bot_task:
            with suppress(asyncio.CancelledError, Exception):
                task.exception()
            return
        if not self._running:
            # Startup failures are handled in connect(); this is only for post-startup exits.
            with suppress(asyncio.CancelledError, Exception):
                task.exception()
            return
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        except Exception as err:  # pragma: no cover - defensive
            exc = err
        if exc is None:
            message = "Discord gateway task exited without an exception"
        else:
            message = f"Discord gateway task exited: {exc}"
        logger.error("[%s] %s", self.name, message, exc_info=exc if exc else False)
        self._set_fatal_error("discord_gateway_task_exited", message, retryable=True)

        async def _notify() -> None:
            try:
                await self._notify_fatal_error()
            except Exception as notify_exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "[%s] Failed to notify gateway supervisor about Discord task exit: %s",
                    self.name, notify_exc, exc_info=True,
                )
        asyncio.create_task(_notify())
    async def _cancel_bot_task(self) -> None:
        """Cancel and await the background client.start() task, if running."""
        if self._bot_task and not self._bot_task.done():
            self._bot_task.cancel()
            try:
                await self._bot_task
            except (asyncio.CancelledError, Exception):
                pass
        self._bot_task = None

    def _start_liveness_probe(self) -> None:
        """Start the periodic Gateway WS health probe (REST success doesn't prove event delivery)."""
        if (
            self._liveness_interval_seconds <= 0
            or self._liveness_failure_threshold <= 0
            or self._heartbeat_ack_max_age_seconds <= 0
            or self._max_latency_seconds <= 0
        ):
            return
        if self._liveness_task and not self._liveness_task.done():
            return
        self._liveness_task = asyncio.create_task(self._liveness_loop())

    def _read_websocket_health(self, client: Any) -> tuple[bool, str]:
        """Return current Discord Gateway health without making a REST request."""
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
        ack_age = time.perf_counter() - last_ack
        if not math.isfinite(ack_age) or ack_age > self._heartbeat_ack_max_age_seconds:
            return False, "ack_stale"
        latency = getattr(client, "latency", None)
        if not isinstance(latency, (int, float)) or not math.isfinite(latency):
            return False, "latency_non_finite"
        if latency > self._max_latency_seconds:
            return False, "latency_exceeded"
        return True, "healthy"

    async def _liveness_loop(self) -> None:
        """Force a reconnect after repeated unhealthy Discord Gateway samples."""
        interval = self._liveness_interval_seconds
        threshold = self._liveness_failure_threshold
        failures = 0
        while self._running:
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
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
            logger.warning(
                "[%s] Discord Gateway WebSocket unhealthy (%s, %d/%d)", self.name, reason, failures,
                threshold,
            )
            if failures < threshold:
                continue
            # Mark recovery before closing: Bot.start()'s done callback must not overwrite this reason.
            self._disconnecting = True
            logger.error(
                "[%s] Discord Gateway WebSocket remained unhealthy (%s); forcing reconnect",
                self.name, reason,
            )
            self._set_fatal_error(
                "discord_websocket_health_stale",
                f"Discord Gateway WebSocket health check failed: {reason}", retryable=True,
            )
            self._liveness_notification_task = asyncio.create_task(
                self._notify_liveness_fatal_error(client)
            )
            return

    async def _notify_liveness_fatal_error(self, client: Any) -> None:
        """Close the failed client, then notify the runner outside the sampler (which must not
        await itself via ``disconnect()``); the runner owns the bounded teardown."""
        failed_websocket = getattr(client, "ws", None)
        try:
            close_task = asyncio.create_task(client.close())
            try:
                done, _pending = await asyncio.wait({close_task}, timeout=1.0)
                if close_task not in done:
                    raise asyncio.TimeoutError
                await close_task
            except asyncio.TimeoutError:
                logger.warning("[%s] Timed out closing unhealthy Discord client", self.name)
                close_task.cancel()
                close_task.add_done_callback(_consume_background_task_result)
                closing_task = getattr(client, "_closing_task", None)
                if isinstance(closing_task, asyncio.Task):
                    closing_task.cancel()
                    closing_task.add_done_callback(_consume_background_task_result)
                    # Client.close() caches this task; clear it before the runner's disconnect retries.
                    client._closing_task = None
                try:
                    if _abort_discord_websocket_transport(failed_websocket):
                        logger.warning(
                            "[%s] Aborted unresponsive Discord WebSocket transport", self.name,
                        )
                except Exception:
                    logger.debug(
                        "[%s] Error aborting unhealthy Discord WebSocket transport", self.name,
                        exc_info=True,
                    )
            except Exception:
                logger.debug("[%s] Error closing unhealthy Discord client", self.name, exc_info=True)
            # Runner may run disconnect() elsewhere; drop the self-ref so it can't cancel this callback.
            if self._liveness_notification_task is asyncio.current_task():
                self._liveness_notification_task = None
            await self._notify_fatal_error()
        except Exception:
            logger.debug("[%s] Fatal-error handler raised", self.name, exc_info=True)

    async def _cancel_liveness_task(self) -> None:
        """Cancel and await liveness tasks without awaiting the current task."""
        current = asyncio.current_task()
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
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.debug("[%s] Liveness task shutdown failed", self.name, exc_info=True)
            setattr(self, task_name, None)

    async def cancel_background_tasks(self) -> None:
        """Cancel background tasks, but first flush pending text-batch sends (cancelling
        ``_pending_text_batch_tasks`` mid-send dropped replies); the flush deadline stays below the
        gateway's per-adapter disconnect budget so the outer ``wait_for`` can't hard-cancel us."""
        pending = list(self._pending_text_batch_tasks.values())
        if pending:
            logger.info(
                "[%s] Flushing %d pending text-batch task(s) before shutdown",
                self.name, len(pending),
            )
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=self._text_batch_flush_deadline_seconds(),
                )
            except asyncio.TimeoutError:
                logger.warning(
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
        budget = 5.0  # mirrors gateway _ADAPTER_DISCONNECT_TIMEOUT_SECS_DEFAULT
        raw = os.getenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "").strip()
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
        self._disconnecting = True
        # Cancel the liveness probe first so it can't fire a spurious fatal/reconnect mid-teardown.
        await self._cancel_liveness_task()
        # Leave voice *before* cancelling the bot task: VoiceClient.disconnect() needs the main
        # gateway WS (run by the bot task) or it blocks until the timeout.
        for guild_id in list(self._voice_clients.keys()):
            try:
                await self.leave_voice_channel(guild_id)
            except Exception as e:  # pragma: no cover - defensive logging
                logger.debug("[%s] Error leaving voice channel %s: %s", self.name, guild_id, e)
        # Cancel the bot task before closing: after a connect() timeout client.start() may still run
        # and discord.py's reconnect loop can ignore the closed flag mid-handshake.
        await self._cancel_bot_task()
        if self._client:
            try:
                await self._client.close()
            except Exception as e:  # pragma: no cover - defensive logging
                logger.warning("[%s] Error during disconnect: %s", self.name, e, exc_info=True)
        for task in (self._post_connect_task, self._missed_message_backfill_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._running = False
        self._client = None
        self._ready_event.clear()
        self._post_connect_task = None
        self._liveness_task = None
        self._missed_message_backfill_task = None
        self._release_platform_lock()
        logger.info("[%s] Disconnected", self.name)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Connect to Discord and start receiving events."""
        if not _discord_available():
            logger.error("[%s] discord.py not installed. Run: pip install discord.py", self.name)
            self._set_fatal_error("missing_dependency", "discord.py not installed", retryable=False)
            return False
        if not discord.opus.is_loaded():
            _load_opus_codec()
        if not self.config.token:
            logger.error("[%s] No bot token configured", self.name)
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
            intents = Intents.default()
            intents.message_content = True
            intents.dm_messages = True
            intents.guild_messages = True
            intents.members = _needs_server_members_intent(
                self._allowed_user_ids, self._allowed_role_ids,
            )
            intents.voice_states = True
            # Resolve proxy (DISCORD_PROXY > generic env vars > macOS system proxy)
            from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_bot
            proxy_url = resolve_proxy_url(platform_env_var="DISCORD_PROXY")
            if proxy_url:
                logger.info("[%s] Using proxy for Discord: %s", self.name, proxy_url)
            # proxy= for HTTP, connector= for SOCKS; allowed_mentions per _build_allowed_mentions.
            # Close any existing client first: a zombie client also fires on_message -> double responses.
            # Without this, the old client remains connected to Discord gateway and both fire on_message,
            # causing double responses. See #18187.
            if self._client is not None:
                try:
                    if not self._client.is_closed():
                        await self._client.close()
                except Exception:
                    logger.debug("[%s] Failed to close previous Discord client", self.name)
                finally:
                    self._client = None
                    self._ready_event.clear()
            self._client = commands.Bot(
                command_prefix="!",  # Not really used, we handle raw messages
                intents=intents,
                allowed_mentions=_build_allowed_mentions(getattr(self.config, "extra", None)),
                **proxy_kwargs_for_bot(proxy_url),
            )
            from ..events import (
                message_create, message_delete, message_edit, ready,
                thread_create, thread_update, voice_state_update,
            )
            for event_module in (
                ready, message_create, message_edit, message_delete,
                thread_create, thread_update, voice_state_update,
            ):
                event_module.register(self._client, self)
            if self._slash_commands:
                self._register_slash_commands()
            self._disconnecting = False
            self._bot_task = asyncio.create_task(self._client.start(self.config.token))
            self._bot_task.add_done_callback(self._handle_bot_task_done)
            ready_timeout = _discord_ready_timeout_seconds()
            # Wait for ready, failing fast if the startup task dies first (e.g. SOCKS errors).
            await _wait_for_ready_or_bot_exit(
                self._ready_event, self._bot_task,
                timeout=None if ready_timeout <= 0 else ready_timeout,
            )
            self._running = True
            self._start_liveness_probe()
            # Plugin-registered native handlers (discord.py Bot — add_listener()/event hooks).
            self._wire_plugin_handlers(self._client)
            return True
        except asyncio.TimeoutError:
            logger.error("[%s] Timeout waiting for connection to Discord", self.name, exc_info=True)
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
            logger.error("[%s] Failed to connect to Discord: %s", self.name, e, exc_info=True)
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
            guidance = _format_privileged_intents_guidance(
                needs_members=_needs_server_members_intent(
                    getattr(self, "_allowed_user_ids", None),
                    getattr(self, "_allowed_role_ids", None),
                )
            )
            return ("discord_intents_required", guidance, False)
        return ("discord_connect_error", f"Discord startup failed: {error}", True)

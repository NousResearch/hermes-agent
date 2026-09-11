"""Native profile execution and one-attempt Slack egress for the durable router.

Handler completion is content, NEVER a delivery receipt. Only a Slack API
acknowledgement with ok=True and a nonempty ts can confirm the router outbox.
"""
from __future__ import annotations

import asyncio
import copy
import weakref
from collections.abc import Mapping

from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource, _session_key_namespace


class NativeRouterSender:
    def __init__(self, runner, *, timeout_seconds=60):
        self.runner = runner
        self.timeout_seconds = timeout_seconds
        self.adapter = None

    def bind(self, adapter):
        if adapter.platform != Platform.SLACK:
            raise RuntimeError("Work Router requires a Slack transport")
        # One runtime per receiving profile. Reconnect replaces the transport;
        # target-profile adapters are never an egress fallback.
        self.adapter = adapter

    def ready(self):
        adapter = self.adapter
        return (adapter is not None and adapter.is_connected is True
                and self.runner._owning_profile(adapter, Platform.SLACK)[0])

    def _source(self, event, target):
        adapter = self.adapter
        if adapter is None:
            raise RuntimeError("Work Router transport unavailable")
        source = SessionSource(
            platform=Platform.SLACK, chat_id=event.channel_id,
            chat_type="dm" if event.channel_type in {"im", "dm"} else "channel",
            user_id=event.author_user_id, is_bot=event.author_kind == "bot",
            thread_id=event.thread_ts,
            scope_id=event.metadata.get("slack_team_id") or None,
            profile=target,
        )
        source._transport_adapter_ref = weakref.ref(adapter)
        # Process-local ownership, deliberately not part of SessionSource wire format.
        source._work_router_owned_final = True
        if self.runner._adapter_for_source(source) is not adapter:
            raise RuntimeError("Work Router transport is no longer registered")
        return source

    async def execute(self, *, event, state, decision, operation_id):
        """Native session admission, profile scope, persistence and approval wiring.

        The returned string is only model/handler content. Busy/queue admission
        must not be mistaken for execution; refuse before entering that lane.
        """
        from hermes_cli.profiles import get_profile_dir, profile_exists
        from gateway.run import _profile_runtime_scope

        target = decision.action.target_profile
        if not target or not profile_exists(target):
            raise RuntimeError("Work Router target profile unavailable")
        source = self._source(event, target)
        session_key = self.runner._session_key_for_source(source)
        if not session_key.startswith(_session_key_namespace(target) + ":"):
            raise RuntimeError("Work Router target session namespace unavailable; requires native profile routing")
        if self.runner._is_session_running(session_key):
            raise RuntimeError("Work Router target session busy")
        native = MessageEvent(
            text=event.text, source=source, user_id=event.author_user_id,
            channel_context=decision.action.context,
            internal=True, allow_gateway_control=False,
        )
        # Scope credentials only after native session isolation has been proven.
        # Non-multiplex agent:main must never mix different staff histories.
        with _profile_runtime_scope(get_profile_dir(target)):
            text = await self.runner._handle_message(native)
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Work Router execution produced no public response")
        return text

    async def __call__(self, *, event, state, decision, operation_id):
        try:
            source = self._source(event, decision.action.target_profile)
            adapter = source._transport_adapter_ref()
            text = decision.action.content
            if decision.action.kind == "dispatch" and text is None:
                text = await self.execute(
                    event=event, state=state, decision=decision, operation_id=operation_id,
                )
            if decision.action.artifact_path:
                raise RuntimeError("Work Router artifact delivery needs a separate confirmed operation")
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError("Work Router public content missing")
            # Revalidate after execution: reconnect/shutdown must not redirect a
            # response onto another profile's adapter.
            if adapter is not self.adapter or self.runner._adapter_for_source(source) is not adapter:
                raise RuntimeError("Work Router transport changed during execution")
            blocked = adapter._outbound_blocked(event.channel_id, "Work Router send to")
            if blocked:
                return blocked
            formatted = adapter.format_message(text)
            if not formatted.strip() or len(formatted) > adapter.MAX_MESSAGE_LENGTH:
                raise RuntimeError("Work Router requires a single nonempty Slack message")
            team_id = source.scope_id or ""
            client = adapter._get_client(event.channel_id, team_id=team_id)
            if client is None:
                raise RuntimeError("Work Router Slack client unavailable")
            # SDK retries are also retries. Copy only the client wrapper so its
            # authenticated session/proxy settings survive without mutating the
            # shared native client's retry policy. No block fallback or chunks.
            single_attempt = copy.copy(client)
            single_attempt.retry_handlers = []
            async with asyncio.timeout(self.timeout_seconds):
                response = await single_attempt.chat_postMessage(
                    channel=event.channel_id, text=formatted,
                    thread_ts=event.thread_ts or None, mrkdwn=True,
                    unfurl_links=False, unfurl_media=False,
                )
            payload = response if isinstance(response, Mapping) else getattr(response, "data", None)
            ts = payload.get("ts") if isinstance(payload, Mapping) else None
            confirmed = (isinstance(payload, Mapping) and payload.get("ok") is True
                         and isinstance(ts, str) and bool(ts.strip()))
            if not confirmed:
                return SendResult(success=False, error="DELIVERY_UNCONFIRMED", retryable=False)
            # Preserve native thread-following bookkeeping after an actual post.
            adapter._bot_message_ts.add(adapter._workspace_message_marker(team_id, ts))
            if event.thread_ts:
                adapter._bot_message_ts.add(adapter._workspace_message_marker(team_id, event.thread_ts))
            adapter._trim_bot_message_timestamps()
            return SendResult(success=True, message_id=ts, raw_response=response)
        except asyncio.CancelledError:
            # Cancellation must reach the router's ambiguity/lease owner.
            raise
        except Exception:
            # Do not persist SDK exception text (may contain sensitive payloads).
            return SendResult(success=False, error="DELIVERY_UNCONFIRMED", retryable=False)
        finally:
            # Typing is ancillary, not delivery proof, and must not erase a post
            # confirmation if cleanup fails.
            adapter = locals().get("adapter")
            if adapter is not None:
                try:
                    await adapter.stop_typing(event.channel_id, metadata={
                        "thread_id": event.thread_ts,
                        "slack_team_id": event.metadata.get("slack_team_id", ""),
                    })
                except Exception:
                    pass

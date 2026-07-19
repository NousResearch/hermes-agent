"""Shared runtime helpers for busy-session follow-up handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from gateway.direct_shortcut_runtime_service import try_handle_direct_gateway_shortcuts
from gateway.platforms.base import MessageEvent, MessageType


@dataclass(slots=True)
class GatewayBusyFollowupResult:
    """Result of evaluating a message while the session already has a running agent."""

    handled: bool
    response: str | None = None


def _queue_followup_event(
    *,
    adapter: Any,
    session_key: str,
    event: Any,
) -> None:
    """Queue one follow-up event on the adapter, preserving adapter-specific behavior.

    Prefer a real ``queue_message`` implementation. MagicMock adapters (unit
    tests) auto-create that attribute, so fall back to writing
    ``_pending_messages`` when the call does not actually stage the event.
    """

    if not adapter:
        return
    pending = getattr(adapter, "_pending_messages", None)
    queue_fn = getattr(type(adapter), "queue_message", None)
    if callable(queue_fn):
        try:
            queue_fn(adapter, session_key, event)
            if isinstance(pending, dict) and session_key in pending:
                return
        except Exception:
            pass
    if isinstance(pending, dict):
        pending[session_key] = event


def _queue_followup_on_runner(
    *,
    runner: Any,
    adapter: Any,
    session_key: str,
    event: Any,
) -> None:
    """Queue a busy-session follow-up using runner FIFO when available."""

    queue_fn = getattr(runner, "_queue_or_replace_pending_event", None)
    if callable(queue_fn):
        try:
            queue_fn(session_key, event)
            return
        except Exception:
            pass
    _queue_followup_event(
        adapter=adapter,
        session_key=session_key,
        event=event,
    )


def _consume_adapter_pending_message(
    *,
    adapter: Any,
    session_key: str,
) -> None:
    """Consume one pending adapter message when the adapter exposes that hook."""

    if not adapter:
        return
    get_fn = getattr(type(adapter), "get_pending_message", None)
    if callable(get_fn):
        try:
            get_fn(adapter, session_key)
            return
        except Exception:
            pass
    pending = getattr(adapter, "_pending_messages", None)
    if isinstance(pending, dict):
        pending.pop(session_key, None)


def _queue_text_followup(
    *,
    adapter: Any,
    event: Any,
    session_key: str,
    runner: Any = None,
) -> None:
    """Queue an explicit /queue follow-up, preserving media + reply context."""

    if not adapter and runner is None:
        return
    queued_text = event.get_command_args().strip()
    media_urls = list(getattr(event, "media_urls", None) or [])
    media_types = list(getattr(event, "media_types", None) or [])
    has_media = bool(media_urls)
    message_type = getattr(event, "message_type", None)
    if has_media and message_type is not None:
        queued_type = message_type
    else:
        queued_type = MessageType.TEXT
    queued_event = MessageEvent(
        text=queued_text,
        message_type=queued_type,
        source=event.source,
        raw_message=getattr(event, "raw_message", None),
        message_id=event.message_id,
        media_urls=media_urls,
        media_types=media_types,
        metadata=dict(getattr(event, "metadata", None) or {}) or None,
        reply_to_message_id=getattr(event, "reply_to_message_id", None),
        reply_to_text=getattr(event, "reply_to_text", None),
        reply_to_author_id=getattr(event, "reply_to_author_id", None),
        reply_to_author_name=getattr(event, "reply_to_author_name", None),
        reply_to_is_own_message=bool(
            getattr(event, "reply_to_is_own_message", False)
        ),
        auto_skill=getattr(event, "auto_skill", None),
        channel_prompt=getattr(event, "channel_prompt", None),
        internal=bool(getattr(event, "internal", False)),
        timestamp=getattr(event, "timestamp", None),
    )
    if runner is not None:
        enqueue_fifo = getattr(runner, "_enqueue_fifo", None)
        if callable(enqueue_fifo) and adapter is not None:
            try:
                enqueue_fifo(session_key, queued_event, adapter)
                return
            except Exception:
                pass
        _queue_followup_on_runner(
            runner=runner,
            adapter=adapter,
            session_key=session_key,
            event=queued_event,
        )
        return
    _queue_followup_event(
        adapter=adapter,
        session_key=session_key,
        event=queued_event,
    )


def _busy_ack_or_fallback(
    *,
    adapter: Any,
    event: Any,
    interrupting: bool,
    fallback_busy_ack: Callable[[Any, str], str],
) -> str:
    """Return the adapter-specific busy ack or the generic QQ fallback.

    Only accept real ``str`` acks. MagicMock adapters auto-create
    ``_busy_followup_ack`` and would otherwise poison the return value.
    """

    if adapter is not None:
        ack_fn = getattr(type(adapter), "_busy_followup_ack", None)
        if callable(ack_fn):
            try:
                ack = ack_fn(adapter, event, interrupting=interrupting)
            except TypeError:
                try:
                    ack = ack_fn(adapter, event)
                except Exception:
                    ack = None
            except Exception:
                ack = None
            if isinstance(ack, str):
                return ack
    try:
        fallback = fallback_busy_ack(event.source, event.text)
    except Exception:
        return ""
    return fallback if isinstance(fallback, str) else ""


async def handle_gateway_busy_followup(
    *,
    runner: Any,
    event: Any,
    source: Any,
    session_key: str,
    logger: Any,
    pending_agent_sentinel: Any,
    truncate_status_preview: Callable[[Any], str],
    fallback_busy_ack: Callable[[Any, str], str],
) -> GatewayBusyFollowupResult:
    """Handle one inbound event while the gateway session already has an active agent."""

    if session_key not in runner._running_agents:
        return GatewayBusyFollowupResult(handled=False)

    if event.get_command() == "status":
        return GatewayBusyFollowupResult(
            handled=True,
            response=await runner._handle_status_command(event),
        )

    shortcut_history = None
    try:
        shortcut_session = runner.session_store.get_or_create_session(source)
        shortcut_history = runner.session_store.load_transcript(shortcut_session.session_id)
    except Exception:
        shortcut_history = None

    direct_shortcut_response = try_handle_direct_gateway_shortcuts(
        runner,
        event,
        prepare_session_env=True,
        conversation_history=list(shortcut_history or []),
        logger=logger,
    )
    if direct_shortcut_response is not None:
        return GatewayBusyFollowupResult(
            handled=True,
            response=direct_shortcut_response,
        )

    busy_input_mode = runner._get_busy_input_mode(source.platform)
    adapter = runner.adapters.get(source.platform)

    from hermes_cli.commands import resolve_command as _resolve_cmd_inner

    evt_cmd = event.get_command()
    cmd_def_inner = _resolve_cmd_inner(evt_cmd) if evt_cmd else None

    if cmd_def_inner and cmd_def_inner.name == "stop":
        # Prefer the full hard-stop path so adapter interrupt hooks, pending
        # queue drain, generation invalidation, and cache eviction stay aligned
        # with the cold-path /stop handler.
        interrupt_and_clear = getattr(runner, "_interrupt_and_clear_session", None)
        if callable(interrupt_and_clear):
            try:
                from gateway.run import _INTERRUPT_REASON_STOP
            except Exception:
                _INTERRUPT_REASON_STOP = "Stop requested"
            await interrupt_and_clear(
                session_key,
                source,
                interrupt_reason=_INTERRUPT_REASON_STOP,
                invalidation_reason="stop_command",
            )
        else:
            running_agent = runner._running_agents.get(session_key)
            if running_agent and running_agent is not pending_agent_sentinel:
                running_agent.interrupt("Stop requested")
            _consume_adapter_pending_message(
                adapter=adapter,
                session_key=session_key,
            )
            runner._pending_messages.pop(session_key, None)
            if session_key in runner._running_agents:
                del runner._running_agents[session_key]
        logger.info("HARD STOP for session %s — session lock released", session_key[:20])
        # Prefer i18n stop text when available; fall back for partial runners.
        try:
            from agent.i18n import t
            from gateway.platforms.base import EphemeralReply

            return GatewayBusyFollowupResult(
                handled=True,
                response=EphemeralReply(t("gateway.stop.stopped")),
            )
        except Exception:
            return GatewayBusyFollowupResult(
                handled=True,
                response="⚡ Force-stopped. The session is unlocked — you can send a new message.",
            )

    if cmd_def_inner and cmd_def_inner.name == "new":
        interrupt_and_clear = getattr(runner, "_interrupt_and_clear_session", None)
        if callable(interrupt_and_clear):
            try:
                from gateway.run import _INTERRUPT_REASON_RESET
            except Exception:
                _INTERRUPT_REASON_RESET = "Session reset requested"
            await interrupt_and_clear(
                session_key,
                source,
                interrupt_reason=_INTERRUPT_REASON_RESET,
                invalidation_reason="new_command",
            )
        else:
            running_agent = runner._running_agents.get(session_key)
            if running_agent and running_agent is not pending_agent_sentinel:
                running_agent.interrupt("Session reset requested")
            _consume_adapter_pending_message(
                adapter=adapter,
                session_key=session_key,
            )
            runner._pending_messages.pop(session_key, None)
            if session_key in runner._running_agents:
                del runner._running_agents[session_key]
        return GatewayBusyFollowupResult(
            handled=True,
            response=await runner._handle_reset_command(event),
        )

    if evt_cmd in ("queue", "q"):
        queued_text = event.get_command_args().strip()
        has_media = bool(getattr(event, "media_urls", None))
        if not queued_text and not has_media:
            return GatewayBusyFollowupResult(
                handled=True,
                response="Usage: /queue <prompt>",
            )
        _queue_text_followup(
            adapter=adapter,
            event=event,
            session_key=session_key,
            runner=runner,
        )
        depth_fn = getattr(runner, "_queue_depth", None)
        depth = 1
        if callable(depth_fn):
            try:
                depth = int(depth_fn(session_key, adapter=adapter) or 1)
            except Exception:
                depth = 1
        if depth <= 1:
            ack = "Queued for the next turn."
        else:
            ack = f"Queued for the next turn. ({depth} queued)"
        return GatewayBusyFollowupResult(
            handled=True,
            response=ack,
        )

    if cmd_def_inner and cmd_def_inner.name == "model":
        return GatewayBusyFollowupResult(
            handled=True,
            response="Agent is running — wait or /stop first, then switch models.",
        )

    if cmd_def_inner and cmd_def_inner.name in ("approve", "deny"):
        if cmd_def_inner.name == "approve":
            return GatewayBusyFollowupResult(
                handled=True,
                response=await runner._handle_approve_command(event),
            )
        return GatewayBusyFollowupResult(
            handled=True,
            response=await runner._handle_deny_command(event),
        )

    # /steer must be handled before the generic pending/queue path so a
    # still-booting sentinel (or queue busy mode) doesn't swallow the
    # dedicated mid-run inject / turn-boundary fallback semantics.
    if cmd_def_inner and cmd_def_inner.name == "steer":
        steer_text = event.get_command_args().strip()
        if not steer_text:
            return GatewayBusyFollowupResult(
                handled=True,
                response="Usage: /steer <prompt>",
            )
        running_agent = runner._running_agents.get(session_key)
        if running_agent is pending_agent_sentinel:
            if adapter:
                queued_event = MessageEvent(
                    text=steer_text,
                    message_type=MessageType.TEXT,
                    source=event.source,
                    message_id=event.message_id,
                    channel_prompt=getattr(event, "channel_prompt", None),
                )
                adapter._pending_messages[session_key] = queued_event
            return GatewayBusyFollowupResult(
                handled=True,
                response="Agent still starting — /steer queued for the next turn.",
            )
        if running_agent and hasattr(running_agent, "steer"):
            try:
                accepted = bool(running_agent.steer(steer_text))
            except Exception as exc:
                logger.warning("Steer failed for session %s: %s", session_key[:20], exc)
                return GatewayBusyFollowupResult(
                    handled=True,
                    response=f"⚠️ Steer failed: {exc}",
                )
            if accepted:
                preview = steer_text[:60] + ("..." if len(steer_text) > 60 else "")
                return GatewayBusyFollowupResult(
                    handled=True,
                    response=(
                        f"⏩ Steer queued — arrives after the next tool call: '{preview}'"
                    ),
                )
            return GatewayBusyFollowupResult(
                handled=True,
                response="Steer rejected (empty payload).",
            )
        # Running agent missing or lacks steer() — fall back to queue.
        if adapter:
            queued_event = MessageEvent(
                text=steer_text,
                message_type=MessageType.TEXT,
                source=event.source,
                message_id=event.message_id,
                channel_prompt=getattr(event, "channel_prompt", None),
            )
            adapter._pending_messages[session_key] = queued_event
        return GatewayBusyFollowupResult(
            handled=True,
            response="No active agent — /steer queued for the next turn.",
        )

    explicit_followup = getattr(source, "chat_type", "") == "dm"
    if (
        not explicit_followup
        and adapter
        and hasattr(adapter, "_is_explicit_busy_followup")
    ):
        try:
            explicit_followup = bool(adapter._is_explicit_busy_followup(event))
        except Exception:
            explicit_followup = False

    if event.message_type == MessageType.TEXT and not evt_cmd and explicit_followup:
        try:
            from tools.approval import (
                has_blocking_approval,
                peek_blocking_approval,
                resolve_gateway_approval,
            )

            if has_blocking_approval(session_key):
                admin_only_message = runner._admin_only_message(
                    source,
                    "deny dangerous commands",
                )
                if admin_only_message is None:
                    current_approval = peek_blocking_approval(session_key) or {}
                    resolved = resolve_gateway_approval(
                        session_key,
                        "deny",
                        resolve_all=True,
                    )
                    if resolved:
                        if adapter:
                            adapter.resume_typing_for_chat(source.chat_id)
                        running_agent = runner._running_agents.get(session_key)
                        if running_agent and running_agent is not pending_agent_sentinel:
                            running_agent.interrupt(event.text)

                        followup_text = str(event.text or "").strip()
                        if followup_text:
                            if session_key in runner._pending_messages:
                                runner._pending_messages[session_key] += "\n" + followup_text
                            else:
                                runner._pending_messages[session_key] = followup_text

                        cmd_preview = truncate_status_preview(
                            current_approval.get("command", "")
                        )
                        if resolved > 1:
                            return GatewayBusyFollowupResult(
                                handled=True,
                                response=(
                                    f"刚才挂起的 {resolved} 条危险命令我先给你拒了。"
                                    "你这条我接着处理。"
                                ),
                            )
                        if cmd_preview:
                            return GatewayBusyFollowupResult(
                                handled=True,
                                response=(
                                    f"刚才那条危险命令我先给你拒了：{cmd_preview}。"
                                    "你这条我接着处理。"
                                ),
                            )
                        return GatewayBusyFollowupResult(
                            handled=True,
                            response="刚才那条危险命令我先给你拒了。你这条我接着处理。",
                        )
        except Exception:
            pass

    if event.message_type == MessageType.PHOTO:
        logger.debug(
            "PRIORITY photo follow-up for session %s — queueing without interrupt",
            session_key[:20],
        )
        _queue_followup_on_runner(
            runner=runner,
            adapter=adapter,
            session_key=session_key,
            event=event,
        )
        return GatewayBusyFollowupResult(handled=True, response=None)

    running_agent = runner._running_agents.get(session_key)
    if running_agent is pending_agent_sentinel:
        if event.get_command() == "stop":
            if session_key in runner._running_agents:
                del runner._running_agents[session_key]
            logger.info(
                "HARD STOP (pending) for session %s — sentinel cleared",
                session_key[:20],
            )
            return GatewayBusyFollowupResult(
                handled=True,
                response="⚡ Force-stopped. The agent was still starting — session unlocked.",
            )
        _queue_followup_on_runner(
            runner=runner,
            adapter=adapter,
            session_key=session_key,
            event=event,
        )
        busy_ack = _busy_ack_or_fallback(
            adapter=adapter,
            event=event,
            interrupting=False,
            fallback_busy_ack=fallback_busy_ack,
        )
        if busy_ack:
            logger.info(
                "queued follow-up while session pending: platform=%s chat=%s session=%s",
                source.platform.value if getattr(source, "platform", None) else "unknown",
                source.chat_id or "unknown",
                session_key[:32],
            )
            return GatewayBusyFollowupResult(handled=True, response=busy_ack)
        return GatewayBusyFollowupResult(handled=True, response=None)

    force_queue_reason = runner._busy_followup_force_queue_reason(
        session_key,
        running_agent,
    )
    if force_queue_reason:
        logger.info(
            "PRIORITY force-queue for session %s — preserving active run (%s)",
            session_key[:20],
            force_queue_reason,
        )
        _queue_followup_on_runner(
            runner=runner,
            adapter=adapter,
            session_key=session_key,
            event=event,
        )
        busy_ack = _busy_ack_or_fallback(
            adapter=adapter,
            event=event,
            interrupting=False,
            fallback_busy_ack=fallback_busy_ack,
        )
        if busy_ack:
            return GatewayBusyFollowupResult(handled=True, response=busy_ack)
        return GatewayBusyFollowupResult(handled=True, response=None)

    if busy_input_mode == "queue":
        logger.debug(
            "PRIORITY queue for session %s — deferring follow-up without interrupt",
            session_key[:20],
        )
        _queue_followup_on_runner(
            runner=runner,
            adapter=adapter,
            session_key=session_key,
            event=event,
        )
        busy_ack = _busy_ack_or_fallback(
            adapter=adapter,
            event=event,
            interrupting=False,
            fallback_busy_ack=fallback_busy_ack,
        )
        if busy_ack:
            logger.info(
                "queued follow-up for active session: platform=%s chat=%s session=%s",
                source.platform.value if getattr(source, "platform", None) else "unknown",
                source.chat_id or "unknown",
                session_key[:32],
            )
            return GatewayBusyFollowupResult(handled=True, response=busy_ack)
        return GatewayBusyFollowupResult(handled=True, response=None)

    if busy_input_mode == "smart":
        should_interrupt = False
        should_fn = getattr(type(adapter), "_should_interrupt_busy_followup", None) if adapter else None
        if callable(should_fn):
            try:
                started_at = getattr(adapter, "_active_session_started_at", None)
                if (
                    isinstance(started_at, dict)
                    and session_key not in started_at
                    and session_key in runner._running_agents_ts
                ):
                    started_at[session_key] = runner._running_agents_ts[session_key]
                should_interrupt = bool(should_fn(adapter, session_key, event))
            except Exception as exc:
                logger.debug(
                    "smart busy follow-up decision failed for %s: %s",
                    session_key[:20],
                    exc,
                )
        if not should_interrupt:
            logger.debug(
                "PRIORITY smart-queue for session %s — deferring follow-up during grace window",
                session_key[:20],
            )
            _queue_followup_on_runner(
                runner=runner,
                adapter=adapter,
                session_key=session_key,
                event=event,
            )
            busy_ack = _busy_ack_or_fallback(
                adapter=adapter,
                event=event,
                interrupting=False,
                fallback_busy_ack=fallback_busy_ack,
            )
            if busy_ack:
                logger.info(
                    "smart-queued follow-up for active session: platform=%s chat=%s session=%s",
                    source.platform.value if getattr(source, "platform", None) else "unknown",
                    source.chat_id or "unknown",
                    session_key[:32],
                )
                return GatewayBusyFollowupResult(handled=True, response=busy_ack)
            return GatewayBusyFollowupResult(handled=True, response=None)

        logger.info(
            "PRIORITY smart-interrupt for session %s — switching to fresher follow-up",
            session_key[:20],
        )
        _queue_followup_on_runner(
            runner=runner,
            adapter=adapter,
            session_key=session_key,
            event=event,
        )
        running_agent.interrupt(event.text)
        busy_ack = _busy_ack_or_fallback(
            adapter=adapter,
            event=event,
            interrupting=True,
            fallback_busy_ack=fallback_busy_ack,
        )
        return GatewayBusyFollowupResult(handled=True, response=busy_ack or None)

    # Remaining slash commands (/restart, /background, /help, …) keep their
    # dedicated busy-path handlers in GatewayRunner._handle_message.
    if cmd_def_inner or evt_cmd:
        return GatewayBusyFollowupResult(handled=False)

    # Upstream interrupt / FIFO path for plain text follow-ups is also owned by
    # the runner (acks, subagent demotion, telegram grace). Fall through so
    # those behaviors stay intact; this service only owns queue/smart/QQ paths.
    if busy_input_mode in {"interrupt", "steer"}:
        return GatewayBusyFollowupResult(handled=False)

    logger.debug("PRIORITY interrupt for session %s", session_key[:20])
    running_agent.interrupt(event.text)
    if session_key in runner._pending_messages:
        runner._pending_messages[session_key] += "\n" + event.text
    else:
        runner._pending_messages[session_key] = event.text
    return GatewayBusyFollowupResult(handled=True, response=None)

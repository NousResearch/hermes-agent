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
    """Queue one follow-up event on the adapter, preserving adapter-specific behavior."""

    if not adapter:
        return
    if hasattr(adapter, "queue_message"):
        adapter.queue_message(session_key, event)
    else:
        adapter._pending_messages[session_key] = event


def _consume_adapter_pending_message(
    *,
    adapter: Any,
    session_key: str,
) -> None:
    """Consume one pending adapter message when the adapter exposes that hook."""

    if adapter and hasattr(adapter, "get_pending_message"):
        adapter.get_pending_message(session_key)


def _queue_text_followup(
    *,
    adapter: Any,
    event: Any,
    session_key: str,
) -> None:
    """Queue an explicit text follow-up via a synthetic text event."""

    if not adapter:
        return
    queued_event = MessageEvent(
        text=event.get_command_args().strip(),
        message_type=MessageType.TEXT,
        source=event.source,
        message_id=event.message_id,
    )
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
    """Return the adapter-specific busy ack or the generic QQ fallback."""

    if adapter and hasattr(adapter, "_busy_followup_ack"):
        return adapter._busy_followup_ack(event, interrupting=interrupting)
    return fallback_busy_ack(event.source, event.text)


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
        return GatewayBusyFollowupResult(
            handled=True,
            response="⚡ Force-stopped. The session is unlocked — you can send a new message.",
        )

    if cmd_def_inner and cmd_def_inner.name == "new":
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
        if not queued_text:
            return GatewayBusyFollowupResult(
                handled=True,
                response="Usage: /queue <prompt>",
            )
        _queue_text_followup(
            adapter=adapter,
            event=event,
            session_key=session_key,
        )
        return GatewayBusyFollowupResult(
            handled=True,
            response="Queued for the next turn.",
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
        _queue_followup_event(
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
        _queue_followup_event(
            adapter=adapter,
            session_key=session_key,
            event=event,
        )
        busy_ack = ""
        if adapter and hasattr(adapter, "_busy_followup_ack"):
            busy_ack = adapter._busy_followup_ack(event, interrupting=False)
        elif busy_input_mode == "queue":
            busy_ack = fallback_busy_ack(source, event.text)
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
        _queue_followup_event(
            adapter=adapter,
            session_key=session_key,
            event=event,
        )
        busy_ack = ""
        if adapter and hasattr(adapter, "_busy_followup_ack"):
            busy_ack = adapter._busy_followup_ack(event, interrupting=False)
        elif busy_input_mode == "queue":
            busy_ack = fallback_busy_ack(source, event.text)
        if busy_ack:
            return GatewayBusyFollowupResult(handled=True, response=busy_ack)
        return GatewayBusyFollowupResult(handled=True, response=None)

    if busy_input_mode == "queue":
        logger.debug(
            "PRIORITY queue for session %s — deferring follow-up without interrupt",
            session_key[:20],
        )
        _queue_followup_event(
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
        if adapter and hasattr(adapter, "_should_interrupt_busy_followup"):
            try:
                if (
                    hasattr(adapter, "_active_session_started_at")
                    and session_key not in adapter._active_session_started_at
                    and session_key in runner._running_agents_ts
                ):
                    adapter._active_session_started_at[session_key] = runner._running_agents_ts[session_key]
                should_interrupt = bool(adapter._should_interrupt_busy_followup(session_key, event))
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
            _queue_followup_event(
                adapter=adapter,
                session_key=session_key,
                event=event,
            )
            busy_ack = ""
            if adapter and hasattr(adapter, "_busy_followup_ack"):
                busy_ack = adapter._busy_followup_ack(event, interrupting=False)
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
        _queue_followup_event(
            adapter=adapter,
            session_key=session_key,
            event=event,
        )
        running_agent.interrupt(event.text)
        busy_ack = ""
        if adapter and hasattr(adapter, "_busy_followup_ack"):
            busy_ack = adapter._busy_followup_ack(event, interrupting=True)
        return GatewayBusyFollowupResult(handled=True, response=busy_ack or None)

    logger.debug("PRIORITY interrupt for session %s", session_key[:20])
    running_agent.interrupt(event.text)
    if session_key in runner._pending_messages:
        runner._pending_messages[session_key] += "\n" + event.text
    else:
        runner._pending_messages[session_key] = event.text
    return GatewayBusyFollowupResult(handled=True, response=None)

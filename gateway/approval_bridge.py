"""Approval transport for ``/background`` tasks: the card or text prompt for one exact request."""
import asyncio
import logging
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Optional

from agent.async_utils import safe_schedule_threadsafe

logger = logging.getLogger(__name__)


def _build_exec_approval_metadata(base: "dict | None", requester_user_id: "str | None") -> dict:
    """Thread routing plus the requester, which reaction-card adapters use to limit who may answer.
    The request id travels as ``send_exec_approval(request_id=...)``, never in platform metadata."""
    metadata = dict(base or {})
    if requester_user_id:
        metadata["requester_user_id"] = str(requester_user_id)
    return metadata


class _BackgroundNoticeRunner:
    """The two members ``register_timeout_notice`` reads from a TurnRunner, for a turn without one."""

    def __init__(self, *, adapter: Any, chat_id: str, session_key: str, metadata: Optional[dict[str, Any]],
                 loop: asyncio.AbstractEventLoop) -> None:
        self._ctx = SimpleNamespace(session_key=session_key, _status_adapter=adapter, _status_chat_id=chat_id,
                                    _status_thread_metadata=metadata)
        self._loop = loop

    def _schedule(self, coro, log_message: str):
        return safe_schedule_threadsafe(coro, self._loop, logger=logger, log_message=log_message)


def _make_gateway_approval_notifier(
    *,
    adapter: Any,
    chat_id: str,
    session_key: str,
    metadata: Optional[dict[str, Any]],
    requester_user_id: Optional[str],
    loop: asyncio.AbstractEventLoop,
    pause_typing: bool,
) -> Callable[[dict], None]:
    """Build the sync approval notifier for a turn without a TurnRunner (``/background``)."""
    from gateway.run import _approval_send_outcome, _format_exec_approval_fallback, _interim_metadata, _redact_approval_command
    from gateway.run_turn_runner import _renders_exec_approval_buttons
    from gateway.run_turn_runner_approval_settle import register_timeout_notice

    base_metadata = _build_exec_approval_metadata(metadata, requester_user_id)
    # A timed-out prompt must say the command did not run, as on a foreground turn.
    notice_runner = _BackgroundNoticeRunner(
        adapter=adapter, chat_id=chat_id, session_key=session_key, metadata=metadata, loop=loop)

    def _approval_notify_sync(approval_data: dict) -> None:
        if pause_typing:
            adapter.pause_typing_for_chat(chat_id)

        command = _redact_approval_command(approval_data.get("command", ""))
        description = approval_data.get("description", "dangerous command")

        if _renders_exec_approval_buttons(type(adapter)):
            try:
                future = safe_schedule_threadsafe(
                    adapter.send_exec_approval(
                        chat_id=chat_id,
                        command=command,
                        session_key=session_key,
                        description=description,
                        metadata=base_metadata,
                        request_id=approval_data.get("request_id"),
                        allow_permanent=approval_data.get("allow_permanent", True),
                        allow_session=approval_data.get("allow_session", True),
                        smart_denied=approval_data.get("smart_denied", False),
                    ),
                    loop,
                    logger=logger,
                    log_message="send_exec_approval scheduling error",
                )
                if future is None:
                    raise RuntimeError("send_exec_approval: loop unavailable")
                outcome = _approval_send_outcome(future, timeout=15)
            except Exception as exc:
                outcome = "failed"
                logger.warning("Button-based approval failed, falling back to text: %s", exc, exc_info=True)
            if outcome == "sent":
                register_timeout_notice(notice_runner, approval_data, command=command,
                                        card_message_id=getattr(future.result(timeout=0), "message_id", None))
                return
            if outcome == "ambiguous":
                # A timed-out send may already be visible; keep the waiter and NEVER duplicate it.
                return
            if outcome == "declined":
                # A destination refusal applies to every delivery lane, including text.
                raise RuntimeError("exec approval undeliverable: connector egress declined this destination")
            logger.warning("Interactive approval failed, falling back to text")

        prefix = getattr(adapter, "typed_command_prefix", "/")
        message = _format_exec_approval_fallback(
            command,
            description,
            prefix,
            allow_permanent=approval_data.get("allow_permanent", True),
            allow_session=approval_data.get("allow_session", True),
            smart_denied=approval_data.get("smart_denied", False),
            full_command=bool(getattr(adapter, "approval_fallback_single_event", False)),
        )
        fallback_metadata = {**base_metadata, "is_approval_prompt": True}
        if getattr(adapter, "approval_fallback_single_event", False):
            # Matrix approval fallbacks MUST remain one authoritative event.
            # An explicit (empty) pre-rendered boundary disables chunking while
            # retaining the adapter's normal escaped Markdown HTML.
            fallback_metadata["matrix_formatted_body"] = ""
        try:
            future = safe_schedule_threadsafe(
                adapter.send(
                    chat_id,
                    message,
                    metadata=_interim_metadata(fallback_metadata),
                ),
                loop,
                logger=logger,
                log_message="Approval text-send scheduling error",
            )
            if future is None:
                raise RuntimeError("approval fallback send: loop unavailable")
            result = future.result(timeout=15)
            if result is not None and getattr(result, "success", True) is False:
                raise RuntimeError(
                    str(getattr(result, "error", "") or "approval fallback send failed")
                )
            register_timeout_notice(notice_runner, approval_data, command=command, card_message_id=None)
        except TimeoutError:
            # A late acknowledgement does not prove the text was undelivered.
            logger.warning("Approval text send timed out; keeping the waiter armed for a late response")
        except Exception as exc:
            logger.error("Failed to send approval request: %s", exc)
            # The core approval guard catches notifier failures and denies the
            # operation immediately instead of waiting on an invisible card.
            raise

    return _approval_notify_sync


@contextmanager
def background_approval_scope(
    *, adapter: Any, source: Any, task_id: str, chat_session_key: str,
    metadata: Optional[dict[str, Any]], loop: asyncio.AbstractEventLoop,
) -> Iterator[None]:
    """Route a ``/background`` task's dangerous-command approvals to its chat. The task waits under
    its own key (so it never takes over the foreground session's queue), and typed ``/approve`` in
    ``chat_session_key`` still reaches it. Wrap the executor call that runs the agent."""
    from tools.approval import register_gateway_notify, unregister_gateway_notify
    from tools.approval_context import reset_current_session_key, set_current_session_key

    notify = _make_gateway_approval_notifier(
        adapter=adapter, chat_id=str(getattr(source, "chat_id", "") or ""), session_key=task_id,
        metadata=metadata, requester_user_id=getattr(source, "user_id", None), loop=loop, pause_typing=False,
    )
    token = set_current_session_key(task_id)
    try:
        register_gateway_notify(task_id, notify, command_session_key=chat_session_key)
        try:
            yield
        finally:
            unregister_gateway_notify(task_id)
    finally:
        reset_current_session_key(token)

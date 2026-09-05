"""Shared foreground/background approval transport with exact request identity."""
import asyncio
import logging
from typing import Any, Callable, Dict, Optional

from agent.async_utils import safe_schedule_threadsafe

logger = logging.getLogger(__name__)

def _build_exec_approval_metadata(
    base: "dict | None",
    approval_data: dict,
) -> dict:
    """Preserve transport context and attach the pending approval identity."""
    metadata = dict(base or {})
    approval_id = str(approval_data.get("approval_id") or "")
    if approval_id:
        metadata["approval_id"] = approval_id
    if "expires_at" in approval_data:
        metadata["expires_at"] = approval_data["expires_at"]
    return metadata


def _make_gateway_approval_notifier(
    *,
    adapter: Any,
    chat_id: str,
    session_key: str,
    metadata: Optional[Dict[str, Any]],
    requester_user_id: Optional[str],
    loop: asyncio.AbstractEventLoop,
    pause_typing: bool,
) -> Callable[[dict], None]:
    """Build the sync approval bridge shared by foreground and background turns."""
    from gateway.run import _approval_send_outcome, _format_exec_approval_fallback, _interim_metadata, _redact_approval_command

    base_metadata = dict(metadata or {})
    if requester_user_id:
        base_metadata["requester_user_id"] = str(requester_user_id)

    def _approval_notify_sync(approval_data: dict) -> None:
        if pause_typing:
            adapter.pause_typing_for_chat(chat_id)

        command = _redact_approval_command(approval_data.get("command", ""))
        description = approval_data.get("description", "dangerous command")

        if getattr(type(adapter), "send_exec_approval", None) is not None:
            try:
                future = safe_schedule_threadsafe(
                    adapter.send_exec_approval(
                        chat_id=chat_id,
                        command=command,
                        session_key=session_key,
                        description=description,
                        metadata=_build_exec_approval_metadata(
                            base_metadata,
                            approval_data,
                        ),
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
                if outcome in {"sent", "ambiguous"}:
                    # A timed-out send may already be visible; keep the waiter and NEVER duplicate it.
                    return
                logger.warning("Interactive approval failed, falling back to text")
            except Exception as exc:
                logger.warning(
                    "Button-based approval failed, falling back to text: %s",
                    exc,
                )

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
        except TimeoutError:
            # A late acknowledgement does not prove the text was undelivered.
            logger.warning("Approval text send timed out; keeping the waiter armed for a late response")
        except Exception as exc:
            logger.error("Failed to send approval request: %s", exc)
            # The core approval guard catches notifier failures and denies the
            # operation immediately instead of waiting on an invisible card.
            raise

    return _approval_notify_sync

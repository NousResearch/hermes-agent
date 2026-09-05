"""Send gateway approval notifications from the agent thread."""

from __future__ import annotations

from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.base_exec_approval import ea_default_reason_text


def _renders_exec_approval_buttons(adapter_cls: type) -> bool:
    """True when the adapter class renders native approval buttons. BasePlatformAdapter subclasses
    say so through ``supports_exec_approval_buttons``; anything else (test doubles, relay-style
    duck types) counts when it defines ``send_exec_approval`` itself."""
    probe = getattr(adapter_cls, "supports_exec_approval_buttons", None)
    if callable(probe) and issubclass(adapter_cls, BasePlatformAdapter):
        return bool(probe())
    return getattr(adapter_cls, "send_exec_approval", None) is not None


class _ExecApprovalDeclined(RuntimeError):
    """The connector refused the approval card's destination.

    Raised (not returned) so it propagates out of `_approval_notify_sync` to
    `_await_gateway_decision`, whose notify-failure path drops the central
    approval queue entry and unblocks the waiting tool. A plain return
    suppressed the text fallback but left that entry pending.
    """


def notify_approval(self, approval_data: dict) -> None:
    """Send the approval request from the agent thread: the adapter's interactive button
    approvals (``send_exec_approval``) when available, else plain text with ``/approve`` steps."""
    from gateway.run_turn_runner import _CARD_DESTINATION_REFUSALS, logger

    from gateway.approval_bridge import _build_exec_approval_metadata
    from gateway.run import _approval_send_outcome, _format_exec_approval_fallback, _interim_metadata, _redact_approval_command
    from gateway.run_turn_runner_approval_settle import register_timeout_notice
    ctx = self._ctx
    adapter = ctx._status_adapter
    # Slack's assistant_threads_setStatus disables the compose box, so the user can't type
    # /approve while "is thinking..." shows. Pausing stops _keep_typing re-setting it; resumed
    # in approve/deny.
    adapter.pause_typing_for_chat(ctx._status_chat_id)
    self._close_native_stream_boundary("Approval")
    # Redact credentials before display: Tirith's findings are already redacted, but the raw
    # command string still leaks secrets. Both the button and plain-text paths use this value.
    cmd = _redact_approval_command(approval_data.get("command", ""))
    desc = approval_data.get("description") or ea_default_reason_text()
    flags = {k: approval_data.get(k, d) for k, d in (("allow_permanent", True), ("allow_session", True), ("smart_denied", False))}
    # Check the *class*, not the instance — MagicMock auto-creates attributes in tests.
    if _renders_exec_approval_buttons(type(adapter)):
        try:
            fut = self._schedule(
                adapter.send_exec_approval(
                    chat_id=ctx._status_chat_id, command=cmd, session_key=ctx.session_key or "",
                    description=desc, metadata=_build_exec_approval_metadata(
                        {**(ctx._status_thread_metadata or {}), "requester_user_id": ctx.source.user_id},
                        approval_data,
                    ), **flags,
                ),
                "send_exec_approval scheduling error",
            )
            if fut is None:
                raise RuntimeError("send_exec_approval: loop unavailable")
            outcome = _approval_send_outcome(fut, timeout=15)
            if outcome == "sent":
                # Without this, a card whose timer runs out keeps live buttons and nobody
                # learns the command did NOT run (only the TUI registered a settle hook).
                register_timeout_notice(
                    self, approval_data, command=cmd,
                    card_message_id=getattr(fut.result(timeout=0), "message_id", None))
                return
            if outcome == "ambiguous":
                # Timeout ≠ failure: the card may have posted with a late ack. The prompt
                # registration stays alive so a tap still resolves; re-sending made duplicate
                # cards + orphaned "/approve: nothing pending".
                logger.warning(
                    "Button-based approval send timed out — treating "
                    "as possibly-delivered (no re-send; the prompt "
                    "stays armed for a late tap)"
                )
                return
            if outcome == "declined":
                # P5(b): the connector AUTHORIZED this destination and
                # refused it. The text fallback below re-sends the same
                # content to the same chat, which would turn a refused
                # button card into a delivered plain-text one — the exact
                # leak the egress guard exists to stop. A decline is
                # definitive, so unlike `ambiguous` the registration is
                # torn down; unlike `failed`, nothing is re-sent.
                logger.warning(
                    "Button-based approval DECLINED by the connector's "
                    "egress guard — not falling back to text (the "
                    "destination is not approved for this connection)"
                )
                # RAISE, do not return. This function is the notify_cb for
                # `_await_gateway_decision`, which already has a correct
                # undeliverable path: a raising notify drops the queue entry
                # and returns `notify_failed`, unblocking the tool. Returning
                # quietly suppressed the text fallback (right) but left the
                # CENTRAL approval entry pending (wrong) — the dangerous
                # command then blocked until the approval timeout. My earlier
                # comment claimed the registration was torn down; only the
                # adapter's private prompt map was.
                raise _ExecApprovalDeclined(
                    "exec approval undeliverable: connector egress declined "
                    "this destination"
                )
            logger.warning("Button-based approval failed (send returned error), falling back to text")
        except _ExecApprovalDeclined:
            # Must escape this handler: the fallback below is a text send to
            # the destination the connector just refused.
            raise
        except Exception as e:
            logger.warning("Button-based approval failed, falling back to text: %s", e)
    # Plain-text prompt with the adapter's typed prefix (e.g. `!approve`): typed "/" is blocked
    # in Slack threads and reserved by Matrix clients.
    msg = _format_exec_approval_fallback(
        cmd, desc, getattr(adapter, "typed_command_prefix", "/"), **flags,
        full_command=bool(getattr(adapter, "approval_fallback_single_event", False)),
    )
    try:
        # Mark as approval prompt so WeCom routes through the control lane.
        metadata = {**(ctx._status_thread_metadata or {}), "is_approval_prompt": True}
        if getattr(adapter, "approval_fallback_single_event", False):
            metadata["matrix_formatted_body"] = ""
        fut = self._schedule(
            adapter.send(ctx._status_chat_id, msg, metadata=_interim_metadata(metadata)), "Approval text-send scheduling error",
        )
        if fut is None:
            raise RuntimeError("approval fallback send: loop unavailable")
        result = fut.result(timeout=15)
        if result is not None and getattr(result, "success", True) is False:
            raise RuntimeError(str(getattr(result, "error", "") or "approval fallback send failed"))
        # No card to edit on the text path: the prompt has no buttons to drop and carries
        # the /approve instructions, so the timeout notice is posted as a new message.
        register_timeout_notice(self, approval_data, command=cmd, card_message_id=None)
    except Exception as e:
        logger.error("Failed to send approval request: %s", e)
        raise


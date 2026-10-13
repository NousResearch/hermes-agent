"""Capture approval delivery origin without retaining turn streaming/progress state."""
from gateway.turn_context import TurnContext


def approval_transport(turn):
    from gateway.run_turn_runner import TurnRunner
    ctx = turn._ctx
    origin = TurnContext(
        session_key=ctx.session_key, session_id=ctx.session_id,
        source=ctx.source, _loop_for_step=ctx._loop_for_step,
        _status_adapter=ctx._status_adapter, _status_chat_id=ctx._status_chat_id,
        _status_thread_metadata=dict(ctx._status_thread_metadata or {}),
        _run_still_current=lambda: True,
    )
    presenter = TurnRunner(turn._runner, origin)

    def notify(data):
        turn._approval_notify_sync(data)

    def background_notify(data):
        presenter._approval_notify_sync(data, background=True)

    notify.background_notify = background_notify
    return notify


def notify_approval(turn, approval_data: dict, *, background=False) -> None:
    from gateway.run_turn_runner import _ExecApprovalDeclined, _renders_exec_approval_buttons, ea_default_reason_text, logger
    from gateway.run import _approval_send_outcome, _format_exec_approval_fallback, _interim_metadata, _redact_approval_command
    from gateway.run_turn_runner_approval_settle import register_timeout_notice
    ctx = turn._ctx
    adapter = ctx._status_adapter
    # Slack's assistant_threads_setStatus disables the compose box, so the user can't type
    # /approve while "is thinking..." shows. Pausing stops _keep_typing re-setting it; resumed
    # in approve/deny.
    if not background:
        adapter.pause_typing_for_chat(ctx._status_chat_id)
        turn._close_native_stream_boundary("Approval")
    # Redact credentials before display: the raw command string can carry secrets. Both the button and plain-text paths use this value.
    cmd = _redact_approval_command(approval_data.get("command", ""))
    desc = approval_data.get("description") or ea_default_reason_text()
    flags = {k: approval_data.get(k, d) for k, d in (("allow_permanent", True), ("allow_session", True), ("smart_denied", False))}
    # Check the *class*, not the instance — MagicMock auto-creates attributes in tests.
    if _renders_exec_approval_buttons(type(adapter)):
        try:
            fut = turn._schedule(
                adapter.send_exec_approval(
                    chat_id=ctx._status_chat_id, command=cmd, session_key=ctx.session_key or "",
                    description=desc, metadata=ctx._status_thread_metadata, **flags,
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
                    turn, approval_data, command=cmd,
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
        except Exception:  # health: allow BLE001 -- adapter boundary preserves text fallback after logged send failure
            logger.exception("Button-based approval failed, falling back to text")
    # Plain-text prompt with the adapter's typed prefix (e.g. `!approve`): typed "/" is blocked
    # in Slack threads and reserved by Matrix clients.
    msg = _format_exec_approval_fallback(cmd, desc, getattr(adapter, "typed_command_prefix", "/"), **flags)
    try:
        # Mark as approval prompt: WeCom routes it through the control lane and Telegram pushes it
        # in "important" mode (#132516). Never ``notify`` — A2A reads that as the turn-final reply.
        metadata = {**(ctx._status_thread_metadata or {}), "is_approval_prompt": True}
        fut = turn._schedule(
            adapter.send(ctx._status_chat_id, msg, metadata=_interim_metadata(metadata)), "Approval text-send scheduling error",
        )
        if fut is None:
            raise RuntimeError("Approval notification loop unavailable")
        result = fut.result(timeout=15)
        if getattr(result, "success", True) is False:
            raise RuntimeError("Approval notification was not delivered")
        # No card to edit on the text path: post timeout as a new message.
        register_timeout_notice(turn, approval_data, command=cmd, card_message_id=None)
    except Exception as e:
        logger.error("Failed to send approval request: %s", e)
        raise

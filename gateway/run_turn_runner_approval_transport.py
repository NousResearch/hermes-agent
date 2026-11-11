"""The approval notifier a gateway turn registers, and the transport its background children keep.

A background ``delegate_task`` child outlives the turn that dispatched it, and that turn's ``finally``
unregisters the session's notifier (#133514). ``tools.approval_notify_lease.acquire`` takes
``background_notify`` at dispatch, so the child's later approval still reaches the chat it came from:
rendered through the adapter that answers that chat when the prompt is raised (the turn's own
resolver), never through the finished turn's stream or typing state, and never projected onto its
canonical execution generation, which has settled by then and would refuse it.
"""
from gateway.turn_context import TurnContext


def approval_transport(turn):
    from gateway.run_turn_progress import local_viewer_turn
    from gateway.session_context import NON_MESSAGING_SESSION_SURFACES
    from gateway.run_turn_runner import TurnRunner
    # The closure below must not keep the finished turn alive: capture what it needs, not the turn.
    ctx, runner = turn._ctx, getattr(turn, "_runner", None)
    origin = TurnContext(
        session_key=ctx.session_key, session_id=ctx.session_id, source=ctx.source,
        _loop_for_step=ctx._loop_for_step, _status_chat_id=ctx._status_chat_id,
        _status_thread_metadata=dict(ctx._status_thread_metadata or {}),
        _status_adapter_pinned=ctx._status_adapter_pinned,
        _status_adapter_resolver=ctx._status_adapter_resolver,
        _run_still_current=lambda: True,
    )
    # No ``TurnRunner.__init__``: it binds the turn's canonical owner (authority + generation), which a
    # background prompt must never use. ``_render_approval_sync`` reads only ``_ctx`` and ``_schedule``,
    # and its timeout notice follows this context, so it fires even after a newer turn has started.
    presenter = TurnRunner.__new__(TurnRunner)
    presenter._runner, presenter._ctx = runner, origin

    def notify(data):
        turn._approval_notify_sync(data)

    def background_notify(data):
        if origin._status_adapter is None:
            raise RuntimeError("Approval notification unavailable: no chat can show this prompt")
        presenter._render_approval_sync(data, background=True)

    # Only a chat platform can show a prompt once the turn is gone. API/webhook runs answer through
    # their own run, and a local viewer's prompts ride the turn's canonical stream: publishing no
    # transport there makes a later child request fail closed instead of parking where nobody looks.
    platform = getattr(getattr(ctx.source, "platform", None), "value", "")
    if str(platform or "") not in NON_MESSAGING_SESSION_SURFACES and not local_viewer_turn(turn):
        notify.background_notify = background_notify
    return notify

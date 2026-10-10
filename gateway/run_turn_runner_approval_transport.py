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

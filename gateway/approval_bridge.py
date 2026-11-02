"""Approval transport for ``/background`` tasks: the chat's own card or text prompt, per request.

A ``/background`` task has no ``TurnRunner``, so without this its dangerous commands could never be
approved from the chat. It waits in the chat session's own queue under a dispatch-owned notify lease
(``tools.approval_notify_lease``), the mechanism a background ``delegate_task`` child uses: the
foreground turn ending does not withdraw it, ``clear_session`` (``/new``) revokes it, and the task
ending releases it. Typed ``/approve`` / ``/deny``, card taps and ``approvals.approvers`` therefore
work exactly as for a foreground request. Prompts are rendered by ``TurnRunner._render_approval_sync``
with ``background=True`` (cards, text fallback, timeout notice), never by a parallel sender.
"""
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional


def _background_presenter(*, adapter: Any, source: Any, session_key: str, metadata: Optional[dict[str, Any]],
                          loop: Any):
    """A ``TurnRunner`` shell for ``_render_approval_sync``: it reads only ``_ctx`` and ``_schedule``
    (``run_turn_runner_approval_transport.approval_transport`` builds the same for delegate children)."""
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext

    origin = TurnContext(
        session_key=session_key, source=source, _loop_for_step=loop,
        _status_chat_id=str(getattr(source, "chat_id", "") or ""),
        _status_thread_metadata=dict(metadata or {}), _status_adapter_pinned=adapter,
        _run_still_current=lambda: True,
    )
    presenter = TurnRunner.__new__(TurnRunner)
    presenter._runner, presenter._ctx = None, origin
    return presenter


@contextmanager
def background_approval_scope(
    *, adapter: Any, source: Any, chat_session_key: str, metadata: Optional[dict[str, Any]], loop: Any,
) -> Iterator[Callable[[Callable[[], Any]], Callable[[], Any]]]:
    """Yield ``wrap(fn)``: run ``fn`` (the executor body that runs the agent) with its approvals routed to
    the chat. The lease is taken here, at dispatch, and released when the scope exits."""
    from tools import approval
    from tools.approval_context import reset_current_session_key, set_current_session_key
    from tools.approval_notify_lease import acquire, run

    presenter = _background_presenter(
        adapter=adapter, source=source, session_key=chat_session_key, metadata=metadata, loop=loop)

    def notify(data: dict) -> None:
        presenter._render_approval_sync(data, background=True)

    lease = acquire(chat_session_key)
    with approval._lock:
        lease.callback = notify  # this task's own transport, not a foreground turn's

    def wrap(fn: Callable[[], Any]) -> Callable[[], Any]:
        def _run() -> Any:
            # The lease only answers for its own session key (``approval_notify_lease.current``).
            token = set_current_session_key(chat_session_key)
            try:
                return run(lease, fn)
            finally:
                reset_current_session_key(token)
        return _run

    try:
        yield wrap
    finally:
        lease.release()

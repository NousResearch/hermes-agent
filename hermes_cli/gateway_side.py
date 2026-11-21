"""``/bg`` and ``/btw`` on the shared gateway session (``prompt.background`` / ``prompt.btw``).

On main the classic CLI built a second ``AIAgent`` (``/bg``) or made a one-shot auxiliary call
(``/btw``) in its own process. Rewired, the side work runs in the gateway, like Ink's: the verb
replies ``{task_id}`` at once and the answer arrives later on the session's event stream as
``background.complete`` / ``btw.complete`` ``{task_id, text}`` (``error: <exc>`` on failure,
``gateway/session_tui_side.py``). The connection's pump hands those frames to the waiter
registered here, so neither the turn pump nor the idle notification tick can drop them.
"""
from __future__ import annotations

import queue

SIDE_EVENTS = frozenset({"background.complete", "btw.complete"})
_POLL_SECONDS = 1.0
#: Owner refusals a re-attach can clear: the socket no longer holds the session's subscription
#: (a redial, a restarted owner, a session that moved to another id). The owner registers a
#: subscription on the connection that resumes the session, so one resume clears both.
_REATTACHABLE = frozenset({"permission_denied", "not_found"})


def connection_for(cli):
    """The CLI's gateway connection when its session is open, else None (not rewired). A hung-up
    connection is returned (its next call redials); only a final one is None."""
    from hermes_cli.gateway_tools_notify import gateway_store
    store = gateway_store(cli)
    if store is None or not getattr(cli, "_gateway_session_open", False) or store.connection.final:
        return None
    return store.connection


def _reattach(connection, session_id):
    """Re-attach this socket to the session: the owner registers a subscription on the connection
    that resumes it, which is what the side verbs require. Returns the id the owner resolved the
    session to (a compressed session's lineage root, not the id asked for)."""
    snapshot = connection.resume(connection.call, session_id, typed=connection.launch_api_key_typed)
    resolved = str((snapshot or {}).get("session_id") or session_id)
    connection.state(resolved).adopt_snapshot(snapshot)
    return resolved


def refusal_message(exc):
    """One line for a side task the owner refused: what it means and what to do. Never a traceback."""
    reason = str(exc)
    if reason == "permission_denied":
        return ("the gateway refused the side question for this session: this CLI is no longer "
                "attached to it (or the session belongs to another chat). Send a message first — "
                "that re-attaches — or resume the session.")
    if reason == "not_found":
        return "the gateway no longer knows this session; resume it (or start a new one) first."
    return f"the gateway refused the side question ({reason})."


def start(connection, session_id, verb, text):
    """Admit the side task on the owner: ``(task_id, wait)``; ``wait()`` -> the answer (raises on failure).

    A socket that lost its subscription (a redial, a restarted owner, a session that moved) is
    refused ``permission_denied``: one re-attach clears it, so the side task is admitted on the
    session the owner resolves, not abandoned to the caller."""
    from hermes_cli.gateway_client import GatewayRPCError
    try:
        result, epoch = connection.call_tagged(f"prompt.{verb}", session_id=session_id, text=text)
    except GatewayRPCError as exc:
        if str(exc) not in _REATTACHABLE:
            raise
        resolved = _reattach(connection, session_id)
        result, epoch = connection.call_tagged(f"prompt.{verb}", session_id=resolved, text=text)
    task_id = (result or {}).get("task_id")
    if not task_id:
        raise RuntimeError(f"prompt.{verb} returned no task id")
    inbox = connection.side_waiter(task_id)

    def wait():
        try:
            while True:
                try:
                    payload = inbox.get(timeout=_POLL_SECONDS)
                except queue.Empty:
                    # The task lived on the socket that admitted it: hung up, or already redialed.
                    if connection.closed or connection.epoch != epoch:
                        from hermes_cli.gateway_connection import DISCONNECTED
                        raise RuntimeError(DISCONNECTED) from None
                    continue
                answer = str((payload or {}).get("text") or "")
                if answer.startswith("error: "):
                    raise RuntimeError(answer[len("error: "):])
                return answer
        finally:
            connection.drop_side_waiter(task_id)
    return task_id, wait

"""Background-process notification turns the gateway admits for this CLI's session (W11).

On main the classic CLI drained ``tools.process_registry`` itself, queued a
``ProcessNotificationBatch`` into ``_pending_input`` and ran the ``[IMPORTANT: Background process
...]`` turn. Over the session authority the process runs in the gateway: the gateway drains the
completion, admits the notification turn (``authority.admit_automation``) and runs it. The CLI
neither drains the registry nor claims delivery (IPS-086) — it only watches its session's stream.

The claim's ``message.start`` carries the admission's async-result card (``display_kind`` /
``display_text`` / model ``text``; ``gateway.session_automation.admission_display``). The idle tick
turns it into main's ``TimelineNotification``, so main's input path paints the same ``◈ Background
Process Finished: ...`` header, and ``GatewayAgent.run_conversation`` pumps that already-admitted
turn (no ``prompt.submit``) through the same callbacks as a typed turn.
"""
from __future__ import annotations

import queue


def gateway_store(cli):
    """The CLI's gateway session store, or None (a test double / an unavailable store)."""
    from hermes_cli.gateway_session_store import GatewaySessionStore
    store = getattr(cli, "_session_db", None)
    return store if isinstance(store, GatewaySessionStore) else None


def _card(params):
    payload = params.get("payload")
    if not isinstance(payload, dict):
        return None
    kind, display, text = payload.get("display_kind"), payload.get("display_text"), payload.get("text")
    if not kind or not isinstance(display, str) or not display.strip() or not isinstance(text, str) or not text:
        return None
    return payload


def queue_admitted_turn(cli, store) -> None:
    """Idle tick: find a gateway-admitted notification turn's ``message.start`` for this session and
    queue main's ``TimelineNotification`` for it. Frames before it are the between-turn leftovers
    ``run_conversation`` would discard anyway; frames after it stay queued for the pump."""
    from tools.process_registry_notifications import TimelineNotification
    connection, session_id = store.connection, getattr(cli, "session_id", None)
    state = connection.state(session_id)
    state.admitted_turn = None  # a header queued earlier whose turn never ran is stale now
    events = connection.events
    while True:
        try:
            frame = events.get_nowait()
        except queue.Empty:
            return
        if isinstance(frame, Exception):
            from hermes_cli.gateway_connection import lost_before
            if lost_before(frame, getattr(connection, "epoch", None)):
                continue  # a connection a reconnect already replaced
            events.put(frame)  # the disconnect belongs to the next turn
            return
        params = frame.get("params") if isinstance(frame, dict) else None
        if not isinstance(params, dict) or params.get("session_id") != session_id:
            continue
        if params.get("type") == "review.summary":
            # The background review (/refine, the post-turn review) reports after its turn settled.
            from hermes_cli.gateway_agent import review_summary_line
            line = review_summary_line(params.get("payload"))
            if line:
                cli._agent_status_print(line)
            continue
        if params.get("type") != "message.start" or not params.get("admission_id"):
            continue
        card = _card(params)
        if card is None:
            continue
        state.admitted_turn = {"admission_id": params["admission_id"], "text": card["text"],
                               "execution_generation": params.get("execution_generation")}
        _put_first(cli._pending_input, TimelineNotification(
            card["text"], card["display_text"], card["display_kind"],
            card.get("notification_category") or "result"))
        return


def _put_first(pending, item):
    """Ahead of anything already queued: the gateway is running this turn NOW (typed inputs the
    post-turn path re-queued are admitted behind it), and a typed turn run first would drop it."""
    with pending.mutex:
        pending.queue.appendleft(item)
        pending.unfinished_tasks += 1
        pending.not_empty.notify()


def take_admitted_turn(agent, text):
    """The admitted turn ``run_conversation(text)`` replays instead of submitting; None (and the
    stale adoption dropped) for any other turn."""
    state = agent._state()
    adopted = getattr(state, "admitted_turn", None)
    state.admitted_turn = None
    if not adopted or adopted.get("text") != text:
        return None
    return adopted

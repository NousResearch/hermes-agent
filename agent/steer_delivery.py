"""Mid-turn /steer delivery: append a drained steer after the tool batch it waited on, and tell
a live viewer the moment it joins the conversation (the next API call is the first to read it).
``_ra()`` resolves ``run_agent`` lazily so tests patching ``run_agent.logger`` keep intercepting.
"""

from __future__ import annotations

from agent.prompt_builder import steer_user_row


def _ra():
    import run_agent
    return run_agent


def _requeue_pending_steer(agent, steer_text: str) -> None:
    """Put drained steer text back so the caller's fallback delivers it as a next-turn user message."""
    from agent.interrupt_control import requeue_drained_steer
    requeue_drained_steer(agent, steer_text)


def notify_steer_delivered(agent, steer_text: str) -> None:
    """Tell a live viewer that a drained /steer now sits in ``messages``: the next API call is the
    first one that reads it. Desktop holds each steer bubble under the live reply until then.

    The callback gets the text of each ``steer()`` submission this drain joined (oldest first), so a
    viewer releases exactly the steers it sent and keeps holding one sent after the drain; another
    client's steer or a queued prompt's transfer matches nothing it holds."""
    callback = getattr(agent, "steer_delivered_callback", None)
    if callback is None:
        return
    drained, parts = vars(agent).get("_steer_last_drain") or (None, None)
    texts = [text for _, text in parts] if parts and drained == steer_text else [steer_text]
    try:
        callback(texts)
    except Exception:  # health: allow BLE001 -- a viewer sink must never break the turn
        _ra().logger.debug("steer_delivered_callback failed", exc_info=True)


def apply_pending_steer_to_tool_results(agent, messages: list, num_tool_msgs: int) -> None:
    """Persist any pending /steer text as a standalone user message.

    Called at the end of a tool-call batch, before the next API call.

    The steer is emitted as a NEW ``role:"user"`` message appended after the
    last tool result (marker text included), so:

    - the model still sees the self-describing out-of-band marker (same text,
      same provenance semantics);
    - message-role alternation stays legal — ``assistant(tool_calls) → tool →
      user`` is the documented "user jumped in mid-run" pattern that
      ``repair_message_sequence`` deliberately keeps;
    - the appended dict carries no ``_DB_PERSISTED_MARKER`` yet, so the next
      ``_flush_messages_to_session_db`` writes it to the session store — the
      steer text finally becomes part of the durable transcript instead of
      being smeared onto an already-persisted tool row that append-only
      persistence never rewrites (replayed histories then diverge from the
      live request bytes and break the provider prompt cache).
    """
    if num_tool_msgs <= 0 or not messages:
        return
    steer_text = agent._drain_pending_steer()
    if not steer_text:
        return
    # Skip non-tool messages in the tail in case something else is appended at the boundary.
    tail = range(len(messages) - 1, max(len(messages) - num_tool_msgs - 1, -1), -1)
    target = next((messages[j] for j in tail if isinstance(messages[j], dict) and messages[j].get("role") == "tool"), None)
    if target is None:
        # No tool result in this batch (e.g. all skipped by interrupt);
        # requeue so the fallback path delivers it as a normal next-turn
        # user message (which persists like any other user turn).
        _requeue_pending_steer(agent, steer_text)
        return
    messages.append(steer_user_row(steer_text))
    notify_steer_delivered(agent, steer_text)
    _ra().logger.info(
        "Delivered /steer to agent after tool batch (%d chars) as new user message: %s", len(steer_text),
        steer_text[:120] + ("..." if len(steer_text) > 120 else ""),
    )

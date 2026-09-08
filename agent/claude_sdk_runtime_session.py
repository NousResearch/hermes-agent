"""Per-turn visibility wiring for the claude-agent-sdk runtime.

What used to be closures nested in ``run_claude_agent_sdk_turn`` are small
owner objects and module-level functions that take the agent explicitly.
Extracted from ``claude_sdk_runtime.py``.
"""

from __future__ import annotations

import logging
import threading

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("agent.claude_sdk_runtime")


class _TurnVisibility:
    """Visibility callbacks fenced to one exact Hermes turn.

    Created once per turn. The epoch / turn-id pair it records on the agent lets
    a late callback from a superseded turn (or a replaced session) recognise that
    it is stale and drop itself; the iteration counter and the stream-sink flag
    start fresh with it.
    """

    def __init__(self, agent) -> None:
        self._agent = agent
        self._turn_id = str(getattr(agent, "_current_turn_id", "") or "")
        lock = getattr(agent, "_sdk_visibility_lock", None)
        if lock is None:
            lock = threading.RLock()
            agent._sdk_visibility_lock = lock
        self._lock = lock
        with lock:
            agent._sdk_visibility_epoch = getattr(agent, "_sdk_visibility_epoch", 0) + 1
            self._epoch = agent._sdk_visibility_epoch
            agent._sdk_visibility_turn_id = self._turn_id
            agent._sdk_visibility_iteration_count = 0
            agent._sdk_stream_sink_accepted = False

    def is_current(self) -> bool:
        agent = self._agent
        with self._lock:
            return (
                getattr(agent, "_sdk_visibility_epoch", None) == self._epoch
                and getattr(agent, "_sdk_visibility_turn_id", None) == self._turn_id
                and getattr(agent, "_current_turn_id", None) == self._turn_id
                and not getattr(agent, "_interrupt_requested", False)
            )

    def on_tool_iteration(self) -> None:
        agent = self._agent
        with self._lock:
            if not self.is_current():
                return
            agent._sdk_visibility_iteration_count += 1
        try:
            agent._touch_activity("completed SDK tool iteration")
        except Exception:
            logger.debug("claude-sdk iteration activity update failed", exc_info=True)

    def relay_interim_assistant(self, text: str) -> None:
        agent = self._agent
        if not self.is_current() or not isinstance(text, str):
            return
        visible = agent._strip_think_blocks(text).strip()
        if visible:
            from agent.redact import redact_sensitive_text
            visible = redact_sensitive_text(visible)
        if not visible or visible == "(empty)" or agent._interim_text_was_delivered(visible):
            return
        callback = getattr(agent, "interim_assistant_callback", None)
        if callback is None:
            return
        # Mirror the native lane (run_agent.py::_emit_interim_assistant_
        # message): compute the flag instead of hardcoding it. With
        # `agent.claude_agent_sdk.streaming` on, this prose has already
        # been painted by the delta sink, and a False here makes the
        # surface re-render it as fresh commentary on top of the
        # streaming buffer instead of sealing the segment — the text
        # visibly appears, is dropped, then reappears.
        # Two signals, not one: the accumulator says the text was
        # released, the flag says a surface accepted it. Sealing a segment
        # nobody painted would drop the prose from the UI entirely.
        already_streamed = bool(
            getattr(agent, "_sdk_stream_sink_accepted", False)
        ) and agent._interim_content_was_streamed(visible)
        try:
            callback(visible, already_streamed=already_streamed)
            agent._record_delivered_interim_text(visible)
        except Exception:
            logger.debug("interim assistant relay raised", exc_info=True)


def _make_visibility_callbacks(agent):
    """Create visibility callbacks fenced to this exact Hermes turn."""
    visibility = _TurnVisibility(agent)
    return visibility.relay_interim_assistant, visibility.on_tool_iteration

"""Process-local frontend waits. No Hermes turn or protocol changes."""

from __future__ import annotations

import contextvars
import copy
import json
import threading
import time
from concurrent.futures import Future, TimeoutError

from ag_ui.core import ToolCallArgsEvent, ToolCallEndEvent, ToolCallStartEvent
from agui_adapter.translate import json_contract, parse_state_writer_props
from agui_adapter.translate import tool_result_content
from agui_adapter.conversations import ConversationStore, positive_seconds

CURRENT_FRONTEND = contextvars.ContextVar("agui_frontend_relay", default=None)
HANDOFF = object()


class ContinuationError(ValueError):
    pass


def _context(run_input):
    return {
        "state": copy.deepcopy(run_input.state),
        "context": [item.model_dump() for item in run_input.context],
        "forwarded_props": copy.deepcopy(run_input.forwarded_props),
    }


class FrontendRelay:
    def __init__(self, run_input, queue, emit, timeout, approvals):
        self.initial = run_input
        self.queue, self.emit, self.timeout = queue, emit, timeout
        self.approvals = approvals
        self.names = {tool.name for tool in run_input.tools}
        self.agent = self.bridge = self.run_state = None
        self.current_call = None
        self.pending = None
        self.deadline = None
        self.context = _context(run_input)
        self.run_ids = {run_input.run_id}
        self.finished = threading.Event()
        self.lock = threading.RLock()
        self.cancelled = None
        self.conversation = None
        self.history = []

    def started(self, call_id, name, args):
        if name in self.names:
            self.current_call = (call_id, name)

    def execute(self, args):
        with self.lock:
            if self.cancelled:
                raise ContinuationError(self.cancelled)
            if self.current_call is None:
                raise ContinuationError("Frontend tool has no model call identity.")
            call_id, name = self.current_call
            pending = self.pending = Future()
            self.deadline = time.monotonic() + self.timeout
        self.bridge.finish()
        self.emit(
            ToolCallStartEvent(
                tool_call_id=call_id,
                tool_call_name=name,
                parent_message_id="frontend-" + call_id,
            )
        )
        self.emit(ToolCallArgsEvent(tool_call_id=call_id, delta=json.dumps(args)))
        self.emit(ToolCallEndEvent(tool_call_id=call_id))
        self.emit(HANDOFF)
        try:
            return pending.result(timeout=self.timeout)
        except TimeoutError:
            self.cancel("Frontend result expired.")
            raise ContinuationError("Frontend result expired.") from None
        finally:
            with self.lock:
                if self.pending is pending:
                    self.pending = None

    def resolve(self, incoming):
        with self.lock:
            if self.pending is None or self.pending.done() or self.cancelled:
                raise ContinuationError(
                    "No pending frontend result (expired, unknown or already resolved)."
                )
            if time.monotonic() >= self.deadline:
                self.cancel("Frontend result expired.")
                raise ContinuationError("Frontend result expired.")
            if (
                incoming.run_id in self.run_ids
                or incoming.run_id in self.conversation.run_ids
            ):
                raise ContinuationError("Continuation requires a new run id.")
            if json_contract(incoming.tools) != json_contract(self.initial.tools):
                raise ContinuationError(
                    "Tool declarations cannot change during continuation."
                )
            if json_contract(
                parse_state_writer_props(incoming.forwarded_props)
            ) != json_contract(parse_state_writer_props(self.initial.forwarded_props)):
                raise ContinuationError(
                    "State-writer declarations cannot change during continuation."
                )
            prefix = self.initial.messages
            if json_contract(incoming.messages[: len(prefix)]) != json_contract(prefix):
                raise ContinuationError(
                    "Continuation changed the accepted message history."
                )
            suffix = incoming.messages[len(prefix) :]
            if any(message.role == "user" for message in suffix):
                raise ContinuationError(
                    "A new user turn cannot resolve a pending frontend call."
                )
            call_id, _ = self.current_call
            results = [
                message
                for message in suffix
                if message.role == "tool" and message.tool_call_id == call_id
            ]
            if len(results) != 1 or incoming.messages[-1] is not results[0]:
                raise ContinuationError(
                    "Frontend result does not match the pending call."
                )
            result = tool_result_content(results[0])
            current = _context(incoming)
            if json_contract(current) != json_contract(self.context):
                result = json.dumps(
                    {"client_tool_result": result, "adapter_context": current}
                )
                self.context = current
                self.run_state.state.clear()
                if isinstance(incoming.state, dict):
                    self.run_state.state.update(copy.deepcopy(incoming.state))
            self.run_ids.add(incoming.run_id)
            self.conversation.run_ids.add(incoming.run_id)
            self.pending.set_result(result)
            return self.queue

    def cancel(self, reason):
        with self.lock:
            if self.cancelled:
                return
            self.cancelled = reason
            if self.agent is not None:
                self.agent.interrupt(reason)
            if self.pending is not None and not self.pending.done():
                self.pending.set_exception(ContinuationError(reason))

    def disconnect(self):
        """Keep bounded handoff waits; retire a detached in-flight execution."""
        with self.lock:
            if self.finished.is_set():
                return
            if self.pending is not None and not self.pending.done():
                return
            parked = self.approvals.peek(self.initial.thread_id, queue=self.queue)
            if parked is not None and not parked.pending.decision.done():
                return
            self.cancel("AG-UI client disconnected")


class ExecutionRegistry:
    def __init__(self):
        from agui_adapter.approvals import ApprovalRegistry

        self.lock = threading.Lock()
        self.approvals = ApprovalRegistry()
        self.active = {}
        self.conversations = ConversationStore()
        self.closed = False

    def admit(self, run_input, queue, emit, timeout, *, toolsets=()):
        self.conversations.sweep()
        with self.lock:
            if self.closed:
                raise ContinuationError("AG-UI server is shutting down.")
            if run_input.thread_id in self.active:
                raise ContinuationError(
                    "An execution is already active on this thread."
                )
            relay = FrontendRelay(run_input, queue, emit, timeout, self.approvals)
            relay.conversation = self.conversations.acquire(run_input, toolsets)
            self.active[run_input.thread_id] = relay
            return relay

    def get(self, thread_id):
        with self.lock:
            return self.active.get(thread_id)

    def release(self, relay, *, successful=False):
        with self.lock:
            if self.active.get(relay.initial.thread_id) is relay:
                del self.active[relay.initial.thread_id]
        self.conversations.release(relay.conversation, relay, successful=successful)

    def close(self):
        with self.lock:
            self.closed = True
            relays = list(self.active.values())
        for relay in relays:
            relay.cancel("AG-UI server shut down.")
            parked = self.approvals.take(relay.initial.thread_id, queue=relay.queue)
            if parked is not None and not parked.pending.decision.done():
                parked.pending.decision.set_result("deny")
        for relay in relays:
            relay.finished.wait(timeout=5)
        self.conversations.sweep(all_idle=True)


def frontend_timeout():
    from agui_adapter.session import _load_agui_config

    return positive_seconds(_load_agui_config(), "frontend_tool_timeout", 60)

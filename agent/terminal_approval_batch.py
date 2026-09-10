"""Prepare desktop terminal consent without running shells ahead of their turn.

Workers keep execution middleware on its original stack. Only command approval
runs ahead; the existing sequential executor releases each worker and persists
its result before releasing the next. No terminal environment/cwd is acquired
while preparing, and the real execution still runs every command guard.
"""
from __future__ import annotations

import contextvars
import copy
import threading
import time
from contextlib import contextmanager
from typing import Any

from tools.thread_context import propagate_context_to_thread

_batch: contextvars.ContextVar[Any] = contextvars.ContextVar("terminal_approval_batch", default=None)
_slot: contextvars.ContextVar[Any] = contextvars.ContextVar("terminal_approval_slot", default=None)


class _CancelledPreparation(Exception):
    pass


class _TerminalSlot:
    def __init__(self, batch, parsed, index):
        self.batch, self.parsed, self.index = batch, parsed, index
        self.ready = threading.Event()
        self.release = threading.Event()
        self.future: Any = None
        self.tids = []
        self.preparing = False
        self.args = None
        self.decision = None
        self.guard_key = None
        self.claimed = False

    def check_cancelled(self):
        if self.batch.cancelled.is_set() or self.batch.agent._interrupt_requested:
            raise _CancelledPreparation("Terminal approval preparation cancelled; command was not started")

    def prepare(self, ref):
        from tools import terminal_tool as tt
        self.check_cancelled()
        self.args = copy.deepcopy(ref.args)
        self.preparing = True
        try:
            # Read policy only. _plan_execution/_acquire_env resolve cwd and
            # shell state later, after the previous result has been persisted.
            config = tt._get_env_config()
            if isinstance(ref.args.get("command"), str):
                from tools.approval_context import set_current_observability_context, reset_current_observability_context
                tokens = set_current_observability_context(
                    tool_call_id=ref.call_id, session_id=self.batch.agent.session_id or "",
                    turn_id=getattr(self.batch.agent, "_current_turn_id", "") or "",
                )
                try:
                    self.guard_key = (ref.args["command"], config["env_type"], tt._docker_has_host_access(config))
                    self.decision = tt._check_all_guards(*self.guard_key)
                finally:
                    reset_current_observability_context(tokens)
        finally:
            self.preparing = False
            self.ready.set()
        while not self.release.wait(0.1):
            self.check_cancelled()
        self.check_cancelled()

    def run(self):
        from agent import tool_executor as te
        token = _slot.set(self)
        pc, batch = self.parsed, self.batch
        ref = pc.ref(batch.task_id)
        try:
            with te._registered_tool_worker(batch.agent) as tid:
                self.tids.append(tid)
                self.check_cancelled()
                dispatch = te._resolve_sequential_dispatch(batch.agent, ref, batch.messages)
                return te._run_agent_tool_execution_middleware(
                    batch.agent, **ref.middleware_kwargs(), execute=dispatch.execute,
                    scope_block=pc.scope_block, display_index=self.index + 1,
                    authorization_gate=batch.authorization_gate,
                )
        finally:
            self.ready.set()
            _slot.reset(token)


class _TerminalBatch:
    def __init__(self, agent, messages, task_id, parsed):
        from agent.tool_executor import _ConcurrentToolAuthorizationGate
        from tools.daemon_pool import DaemonThreadPoolExecutor
        self.agent, self.messages, self.task_id = agent, messages, task_id
        self.cancelled = threading.Event()
        self.pending_approvals = []  # guarded by tools.approval._lock
        self.authorization_gate = _ConcurrentToolAuthorizationGate()
        self.executor = DaemonThreadPoolExecutor(max_workers=len(parsed))
        self.slots = [_TerminalSlot(self, pc, i) for i, pc in enumerate(parsed)]

    def start(self):
        from agent.tool_executor import _resolve_sequential_tool_timeout
        for slot in self.slots:
            slot.check_cancelled()
            slot.future = self.executor.submit(propagate_context_to_thread(slot.run))
            timeout = _resolve_sequential_tool_timeout()
            started = time.monotonic()
            baseline = self.authorization_gate.excluded_seconds()
            # Proceed once the worker publishes a human request OR completes
            # preparation. A wedged plugin must not hold the batch forever.
            while not slot.ready.wait(0.1):
                slot.check_cancelled()
                elapsed = time.monotonic() - started - (self.authorization_gate.excluded_seconds() - baseline)
                if timeout is not None and elapsed >= timeout:
                    raise TimeoutError("Terminal approval preparation timed out; commands were not started")

    def close(self):
        from agent.tool_executor import _interrupt_worker_tids
        from tools import approval
        # Withdraw only this batch's requests, including a worker wedged in
        # notify_cb. Thread interrupts alone leave those requests actionable.
        with approval._lock:
            self.cancelled.set()
            for session_key, entry in self.pending_approvals:
                queue = approval._gateway_queues.get(session_key, [])
                if entry in queue:
                    queue.remove(entry)
                    entry.result = "deny"
                    entry.event.set()
                if not queue:
                    approval._gateway_queues.pop(session_key, None)
            self.pending_approvals.clear()
        for slot in self.slots:
            slot.release.set()
            if slot.future is not None and not slot.future.done():
                _interrupt_worker_tids(self.agent, slot.tids)
                slot.future.cancel()
        self.executor.shutdown(wait=False, cancel_futures=True)


def prepare_current_terminal(ref):
    slot = _slot.get()
    if slot is not None and ref.name == "terminal":
        slot.prepare(ref)


def bind_prepared_dispatch(dispatch):
    """A middleware-owned thread must not lose the batch's execution barrier."""
    slot = _slot.get()
    if slot is None:
        return dispatch
    from agent.tool_executor import _registered_tool_worker

    owner_tid = threading.get_ident()

    def tracked(*args, **kwargs):
        if threading.get_ident() == owner_tid:
            return dispatch(*args, **kwargs)
        with _registered_tool_worker(slot.batch.agent) as tid:
            slot.tids.append(tid)
            slot.check_cancelled()
            return dispatch(*args, **kwargs)

    invoke = propagate_context_to_thread(tracked)
    # Each batch slot has exactly one dispatch. Reject concurrent/replayed
    # continuations before entering its captured Context on another thread.
    lock = threading.Lock()
    claimed = False

    def once(*args, **kwargs):
        nonlocal claimed
        with lock:
            if claimed:
                raise RuntimeError("Hermes tool execution callback invoked more than once")
            claimed = True
        return invoke(*args, **kwargs)

    return once


def take_prepared_call(call_id):
    batch = _batch.get()
    if batch is None:
        return None
    for slot in batch.slots:
        if slot.parsed.ref(batch.task_id).call_id == call_id and not slot.claimed:
            slot.claimed = True
            slot.check_cancelled()
            slot.release.set()
            return slot
    return None


def approval_published():
    slot = _slot.get()
    if slot is not None:
        slot.ready.set()


def register_prepared_approval(session_key, entry):
    """Called under the approval queue lock, before enqueueing the request."""
    slot = _slot.get()
    if slot is not None:
        slot.check_cancelled()
        slot.batch.pending_approvals.append((session_key, entry))


def consume_prepared_guard(command, env_type, has_host_access):
    slot = _slot.get()
    if slot is None or slot.preparing:
        return None
    slot.check_cancelled()
    from tools.approval_context import _approval_tool_call_id
    if (_approval_tool_call_id.get() != slot.parsed.ref(slot.batch.task_id).call_id
            or slot.guard_key != (command, env_type, has_host_access)):
        return None
    decision, slot.decision = slot.decision, None  # single-use, even for identical calls
    return decision


def preparing_terminal_approval():
    slot = _slot.get()
    return slot is not None and slot.preparing


def validate_prepared_terminal(args):
    slot = _slot.get()
    if slot is not None:
        slot.check_cancelled()
        # Middleware/registry coercion must not turn a prepared consent into
        # authority for different arguments, even with identical display text.
        if args != slot.args:
            slot.decision = None
            raise RuntimeError("Terminal arguments changed after approval preparation; command was not started")


def terminal_approval_runs(agent, calls):
    """Keep nonterminal barriers, but batch adjacent terminals in mixed segments."""
    from itertools import groupby
    from agent.tool_executor import _parse_tool_call

    def is_terminal(call):
        pc = _parse_tool_call(agent, call, flatten_probe=True)
        return pc.name == "terminal" and pc.parse_error is None

    for _, run in groupby(calls, key=is_terminal):
        yield list(run)


@contextmanager
def terminal_approval_batch(agent, calls, messages, task_id):
    from gateway.session_context import get_session_env
    from tools import approval
    from agent.tool_executor import _parse_tool_call
    if (len(calls) < 2 or get_session_env("HERMES_SESSION_SOURCE") != "gui"
            or approval._gateway_notify_cb(approval.get_current_session_key()) is None):
        yield
        return
    parsed = [_parse_tool_call(agent, call, flatten_probe=True) for call in calls]
    # Never prepare across a nonterminal barrier; leave mixed segments and
    # malformed calls with the established sequential path.
    ids = [pc.ref(task_id).call_id for pc in parsed]
    if (any(pc.name != "terminal" or pc.parse_error is not None for pc in parsed)
            or not all(ids) or len(set(ids)) != len(ids)):
        yield
        return
    batch = _TerminalBatch(agent, messages, task_id, parsed)
    token = _batch.set(batch)
    try:
        if not agent._interrupt_requested and not getattr(agent, "_incremental_persistence_failed", False):
            try:
                batch.start()
            except (_CancelledPreparation, TimeoutError) as exc:
                batch.close()
                agent.interrupt(str(exc))
                # The sequential path must still persist a result for every
                # assistant tool call, even if preparation never finished.
        yield
    finally:
        batch.close()
        _batch.reset(token)

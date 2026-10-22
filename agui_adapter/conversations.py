"""Bounded, process-local ownership of warm agents and canonical history."""

from __future__ import annotations

import contextvars
import copy
import math
import threading
import time
import uuid

from agui_adapter.translate import json_contract, parse_state_writer_props


def positive_seconds(config, key, default):
    message = f"agui.{key} must be positive and finite."
    raw = config.get(key, default)
    if isinstance(raw, bool):
        raise ValueError(message)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(message) from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError(message)
    return value


def conversation_limits():
    from agui_adapter.session import _load_agui_config

    config = _load_agui_config()
    capacity = config.get("max_conversations", 16)
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise ValueError("agui.max_conversations must be a positive integer.")
    timeout = positive_seconds(config, "conversation_idle_timeout", 1800)
    return capacity, timeout


class Conversation:
    def __init__(self, incoming, toolsets, now):
        self.thread_id = incoming.thread_id
        # Client-chosen thread ids are not process-wide approval identities.
        self.approval_key = "agui-" + uuid.uuid4().hex
        self.tools = copy.deepcopy(incoming.tools)
        self.writers = parse_state_writer_props(incoming.forwarded_props)
        self.toolsets = tuple(toolsets)
        self.agent = None
        self.messages = []
        self.users = []
        self.run_ids = set()
        self.context = None
        self.profile_context = None
        self.busy = False
        self.last_used = now
        self.closed = False

    def accept(self, incoming, toolsets):
        users = [message for message in incoming.messages if message.role == "user"]
        if self.busy:
            raise ValueError("An execution is already active on this thread.")
        if incoming.run_id in self.run_ids:
            raise ValueError("This run id was already accepted.")
        if (
            json_contract(incoming.tools) != json_contract(self.tools)
            or tuple(toolsets) != self.toolsets
        ):
            raise ValueError(
                "Tool declarations cannot change within a live conversation."
            )
        if json_contract(
            parse_state_writer_props(incoming.forwarded_props)
        ) != json_contract(self.writers):
            raise ValueError(
                "State-writer declarations cannot change within a live conversation."
            )
        if users:
            if incoming.messages[-1].role != "user" or json_contract(
                users[:-1]
            ) != json_contract(self.users):
                raise ValueError("The accepted user history cannot be replaced.")
            if users[-1].id in {user.id for user in self.users}:
                raise ValueError("This user message was already accepted.")
        elif self.agent is not None:
            raise ValueError("A fresh user message is required for a new turn.")
        self.users = copy.deepcopy(users)
        self.run_ids.add(incoming.run_id)
        self.busy = True

    def bind_agent(self, agent):
        self.agent = agent
        if self.profile_context is None:
            self.profile_context = contextvars.copy_context()

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.agent is not None:

            def close_resources():
                from tools.approval import clear_session
                from tools.terminal_tool import _qualify_task_key
                from tools.terminal_tool_lifecycle import cleanup_vm

                try:
                    self.agent.close()
                finally:
                    try:
                        cleanup_vm(_qualify_task_key(f"session:{self.approval_key}"))
                    finally:
                        clear_session(self.approval_key)

            self.profile_context.copy().run(close_resources)


class ConversationStore:
    def __init__(self):
        self.capacity, self.idle_timeout = conversation_limits()
        self.clock = time.monotonic
        self.lock = threading.Lock()
        self.records = {}

    def acquire(self, incoming, toolsets):
        with self.lock:
            record = self.records.get(incoming.thread_id)
            if record is None:
                users = [
                    message for message in incoming.messages if message.role == "user"
                ]
                if len(users) > 1:
                    raise ValueError(
                        "No live conversation (expired or restarted). Start a new thread with a new user request."
                    )
                if len(self.records) >= self.capacity:
                    raise ValueError(
                        "AG-UI conversation capacity reached. Wait for idle expiry before starting another thread."
                    )
                record = Conversation(incoming, toolsets, self.clock())
                record.accept(incoming, toolsets)
                self.records[incoming.thread_id] = record
            else:
                record.accept(incoming, toolsets)
            return record

    def get(self, thread_id):
        with self.lock:
            return self.records.get(thread_id)

    def release(self, record, relay, *, successful):
        with self.lock:
            if self.records.get(record.thread_id) is not record:
                return
            if successful:
                record.messages = relay.history
                record.context = copy.deepcopy(relay.context)
                record.run_ids.update(relay.run_ids)
                record.last_used = self.clock()
                record.busy = False
            else:
                del self.records[record.thread_id]
        if not successful:
            record.close()

    def sweep(self, *, all_idle=False):
        with self.lock:
            now = self.clock()
            expired = [
                record
                for record in self.records.values()
                if not record.busy
                and (all_idle or now - record.last_used >= self.idle_timeout)
            ]
            for record in expired:
                del self.records[record.thread_id]
        for record in expired:
            record.close()
        return len(expired)

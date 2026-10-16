"""Opt-in ACP v1 subagent lifecycle updates (merged ACP RFD #1992).

The pinned SDK predates this variant. Public observer/sender hooks preserve
its capability and send it through the SDK's existing ordered transport.
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Any

from acp.connection import StreamDirection
from acp.task import MessageSender

from agent.async_utils import safe_schedule_threadsafe

logger = logging.getLogger(__name__)
_STATES = {"subagent.start": "running", "subagent.tool": "running",
           "subagent.progress": "running", "subagent.complete": "idle"}


class SubagentEvents:
    """Connection-owned child identities; child sessions have no mutation rights."""

    def __init__(self) -> None:
        self.enabled = False
        self._sender: MessageSender | None = None
        self._children: dict[tuple[str, str], dict[str, Any]] = {}
        self._child_ids: set[str] = set()
        self._registry_lock = threading.Lock()
        self._emit_lock = threading.Lock()

    def connection_options(self) -> dict[str, Any]:
        return {"observers": [self.observe], "sender_factory": self.sender_factory}

    def observe(self, event) -> None:
        # Observe before SDK validation discards this newer capability. {} opts in.
        if event.direction != StreamDirection.INCOMING or event.message.get("method") != "initialize":
            return
        params = event.message.get("params")
        caps = params.get("clientCapabilities") if isinstance(params, dict) else None
        self.enabled = isinstance(caps, dict) and isinstance(caps.get("subagents"), dict)

    def sender_factory(self, writer, supervisor) -> MessageSender:
        self._sender = MessageSender(writer, supervisor)
        return self._sender

    def owns_child(self, session_id: str) -> bool:
        with self._registry_lock:
            return session_id in self._child_ids

    def _association(self, root: str, identity: dict[str, Any]):
        runtime_id = identity.get("subagent_id")
        if not isinstance(runtime_id, str) or not runtime_id:
            return None
        parent_id = identity.get("parent_id")
        with self._registry_lock:
            parent = self._children.get((root, parent_id)) if parent_id else None
            # Never guess a nested parent, or emit traffic before its announcement.
            if parent_id and (parent is None or not parent["announced"]):
                return None
            outer = parent["session_id"] if parent else root
            key = (root, runtime_id)
            child = self._children.get(key)
            if child is None:
                child = {"session_id": str(uuid.uuid4()), "parent": outer, "announced": False}
                self._children[key] = child
                self._child_ids.add(child["session_id"])
            if child["parent"] != outer:
                return None
            return outer, child

    async def _notify(self, outer: str, update: dict[str, Any]) -> None:
        await self._sender.send({"jsonrpc": "2.0", "method": "session/update",
                                 "params": {"sessionId": outer, "update": update}})

    def progress(self, root: str, loop, event: str, name=None, **identity) -> None:
        if not self.enabled or self._sender is None or event not in _STATES:
            return
        # Worker threads serialize announcements; never hold the registry lock
        # while waiting on the event loop (which also checks child identities).
        with self._emit_lock:
            association = self._association(root, identity)
            if association is None:
                return
            outer, child = association
            update = self._update(child, event, name, identity)
            future = safe_schedule_threadsafe(self._notify(outer, update), loop, logger=logger)
            if future is not None:
                future.result(timeout=5)
                child["announced"] = True

    @staticmethod
    def _update(child, event, name, identity) -> dict[str, Any]:
        state = {"state": _STATES[event]}
        if event == "subagent.complete":
            # A timed-out worker can still be unwinding; do not claim it stopped.
            if identity.get("status") == "timeout":
                state["state"] = "unknown"
            reason = {"completed": "end_turn", "interrupted": "cancelled"}.get(identity.get("status"))
            if reason:
                state["stopReason"] = reason
        metadata = {k: identity[k] for k in ("task_index", "task_count", "model", "tool_count", "status")
                    if k in identity}
        if name:
            metadata["tool"] = name
        update = {"sessionUpdate": "subagent_update", "sessionId": child["session_id"],
                  "state": state, "_meta": {"hermes": metadata}}
        if not child["announced"]:
            goal = str(identity.get("goal") or "")
            update.update(title=goal[:120] or "Subagent", description=goal, capabilities={})
        return update

    async def project(self, root, event, payload):
        """Canonical gateway events already arrive serially on the ACP loop."""
        if not self.enabled or self._sender is None or event not in _STATES:
            return
        association = self._association(root, payload)
        if association is None:
            return
        outer, child = association
        await self._notify(outer, self._update(child, event, payload.get('tool_name'), payload))
        child['announced'] = True

    def wrap_progress(self, root: str, loop, callback):
        def progress(event_type, name=None, preview=None, args=None, **kwargs):
            if isinstance(event_type, str) and event_type.startswith("subagent."):
                try:
                    self.progress(root, loop, event_type, name, **kwargs)
                except Exception:
                    logger.debug("Failed to send ACP subagent update", exc_info=True)
                return
            callback(event_type, name, preview, args, **kwargs)
        return progress

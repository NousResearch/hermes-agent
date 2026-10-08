"""Bounded, session-owned ACP subagent visibility; no control or transcript-file API."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Callable, Literal

from acp.schema import ToolCallProgress, ToolCallStart
from pydantic import BaseModel, ConfigDict, Field

from agent.redact import redact_for_egress
from utils import atomic_json_write

logger = logging.getLogger(__name__)
_ID = r"^[A-Za-z0-9_.:-]{1,128}$"
_EVENTS = {"subagent.start", "subagent.tool", "subagent.text", "subagent.complete"}
_STATUS = {"completed": "completed", "success": "completed", "cancelled": "canceled",
           "canceled": "canceled", "interrupted": "canceled"}


class SubagentSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    version: Literal[1] = 1
    id: str = Field(pattern=_ID)
    parent_id: str | None = Field(default=None, alias="parentId", pattern=_ID)
    depth: int = Field(ge=1, le=16)
    sequence: int = Field(ge=1)
    status: Literal["running", "completed", "failed", "canceled"] = "running"
    text: str = Field(default="", max_length=16384)
    tools: list[str] = Field(default_factory=list, max_length=32)


def snapshot_update(snapshot: SubagentSnapshot, *, start: bool, metadata: bool = True):
    status = {"running": "in_progress", "completed": "completed", "failed": "failed", "canceled": "failed"}
    fields = {"tool_call_id": f"hermes-subagent:{snapshot.id}", "status": status[snapshot.status]}
    if metadata:
        fields["field_meta"] = {"hermes": {"subagentProgress": snapshot.model_dump(by_alias=True)}}
    if start:
        return ToolCallStart(session_update="tool_call", title="Hermes subagent", kind="other", **fields)
    return ToolCallProgress(session_update="tool_call_update", **fields)


class SubagentProgress:
    """One journal per ACP session. Old turn callbacks retain this same object.

    The storage directory is captured in the owning profile scope, not looked up
    from child threads. The file name is a hash of the server-owned session id.
    """

    def __init__(self, session_id: str, directory: Path):
        self.session_id = session_id
        self.path = directory / (hashlib.sha256(session_id.encode()).hexdigest() + ".json")
        self._lock = threading.Lock()
        self._nodes: dict[str, SubagentSnapshot] = {}
        self._sequence = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._connection: Callable[[], Any] = lambda: None
        self._metadata = False
        self._send_lock = asyncio.Lock()
        self._pending: set[asyncio.Task] = set()
        self._outbox: dict[str, tuple[SubagentSnapshot, bool]] = {}
        self._delivery: asyncio.Task | None = None
        self._restore()

    def bind(self, loop: asyncio.AbstractEventLoop, connection: Callable[[], Any], *, metadata: bool):
        self._loop, self._connection, self._metadata = loop, connection, metadata

    def _restore(self):
        try:
            if self.path.stat().st_size > 8 * 1024 * 1024:
                return
            rows = json.loads(self.path.read_text(encoding="utf-8-sig"))
            if not isinstance(rows, list) or len(rows) > 64:
                return
            nodes = [SubagentSnapshot.model_validate(row) for row in rows]
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            logger.warning("Could not restore ACP subagent visibility journal")
            return
        self._sequence = max((node.sequence for node in nodes), default=0)
        for node in nodes:
            if node.status == "running":
                # A new ACP process cannot claim that a previous process's child is live.
                self._sequence += 1
                node = node.model_copy(update={"status": "failed", "sequence": self._sequence})
            self._nodes[node.id] = node

    def _persist(self):
        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            atomic_json_write(self.path, [node.model_dump(by_alias=True) for node in self._nodes.values()], mode=0o600)
        except OSError:
            logger.warning("Could not persist ACP subagent visibility journal")

    def record(self, event_type: str, name=None, preview=None, args=None, **kwargs):
        if event_type not in _EVENTS:
            return
        child_id = kwargs.get("subagent_id")
        if not isinstance(child_id, str) or not re.fullmatch(_ID, child_id):
            return
        with self._lock:
            previous = self._nodes.get(child_id)
            if previous is not None and previous.status != "running":
                return
            if previous is None:
                if len(self._nodes) >= 64:
                    return
                depth, parent_id = kwargs.get("depth", 1), kwargs.get("parent_id")
                if not isinstance(depth, int) or isinstance(depth, bool) or not 1 <= depth <= 16:
                    return
                # A root's parent_id may name the root Hermes session, not a child.
                if depth == 1:
                    parent_id = None
                elif (not isinstance(parent_id, str) or not re.fullmatch(_ID, parent_id)
                      or parent_id == child_id):
                    return
                previous = SubagentSnapshot(id=child_id, parentId=parent_id, depth=depth, sequence=1)
            elif event_type == "subagent.start":
                return
            changes: dict[str, Any] = {}
            if event_type == "subagent.text" and isinstance(preview, str):
                # Scrub the complete accumulated text, including secrets split across chunks.
                changes["text"] = redact_for_egress((previous.text + preview)[:16384])[:16384]
            elif event_type == "subagent.tool":
                tool = name if isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", name) else "tool"
                changes["tools"] = (previous.tools + [tool])[:32]
            elif event_type == "subagent.complete":
                changes["status"] = _STATUS.get(kwargs.get("status"), "failed")
            if changes and all(getattr(previous, key) == value for key, value in changes.items()):
                return
            self._sequence += 1
            node = previous.model_copy(update={**changes, "sequence": self._sequence})
            start = child_id not in self._nodes
            self._nodes[child_id] = node
            self._persist()
            # Scheduling under the journal lock preserves worker-thread arrival order.
            if self._loop is not None and not self._loop.is_closed():
                try:
                    self._loop.call_soon_threadsafe(self._queue, node, start)
                except RuntimeError:
                    # Shutdown may close the loop between the check and scheduling.
                    logger.debug("ACP loop closed; subagent snapshot retained for replay")

    def _queue(self, node, start):
        previous = self._outbox.get(node.id)
        self._outbox[node.id] = (node, start or (previous is not None and previous[1]))
        if self._delivery is None:
            self._delivery = asyncio.create_task(self._flush_outbox())
            self._pending.add(self._delivery)
            self._delivery.add_done_callback(self._pending.discard)

    async def _flush_outbox(self):
        try:
            while self._outbox:
                key = min(self._outbox, key=lambda key: self._outbox[key][0].sequence)
                node, start = self._outbox.pop(key)
                await self._send(node, start=start)
        finally:
            self._delivery = None

    async def _send(self, node, *, start):
        async with self._send_lock:
            conn = self._connection()
            if conn is None:
                return
            try:
                await asyncio.wait_for(
                    conn.session_update(self.session_id, snapshot_update(node, start=start, metadata=self._metadata)),
                    timeout=5,
                )
            except Exception:
                logger.debug("ACP subagent visibility delivery failed", exc_info=True)

    async def replay(self):
        with self._lock:
            nodes = sorted(self._nodes.values(), key=lambda node: (node.depth, node.id))
        for node in nodes:
            await self._send(node, start=True)

    async def drain(self):
        # Let call_soon_threadsafe submissions become tasks before taking the snapshot.
        await asyncio.sleep(0)
        if self._pending:
            await asyncio.gather(*tuple(self._pending))

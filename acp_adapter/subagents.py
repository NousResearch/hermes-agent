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
_TITLE_ACTIONS = ("count", "review", "check", "test", "compare", "summarize", "research", "inspect", "fix", "update", "write", "analyze")
_TITLE_SUBJECTS = ("files", "tests", "docs", "notes", "options", "data", "changes", "failure handling")


def public_task_title(goal: Any) -> str:
    """Explicit lossy projection: only fixed public vocabulary, never copied goal fragments."""
    safe = redact_for_egress(goal[:512] if isinstance(goal, str) else "").casefold()
    action = re.search(r"\b(" + "|".join(_TITLE_ACTIONS) + r")\b", safe)
    subject = re.search(r"\b(" + "|".join(_TITLE_SUBJECTS) + r")\b", safe)
    return redact_for_egress(f"{action[0].capitalize()} {subject[0] if subject else 'task'}" if action else "Delegated task")[:80]


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
    title: str = Field(default="Delegated task", max_length=80, pattern=r"^[^\r\n\x00-\x1f\x7f]+$")
    output_limited: bool = Field(default=False, alias="outputLimited")
    activities_limited: bool = Field(default=False, alias="activitiesLimited")


class ChildLimit(BaseModel):
    version: Literal[1] = 1
    sequence: int = Field(ge=1)
    limit: Literal[64] = 64
    children_omitted: Literal[True] = Field(default=True, alias="childrenOmitted")

    @property
    def id(self):
        return "@child-limit"


def snapshot_update(snapshot: SubagentSnapshot | ChildLimit, *, start: bool, metadata: bool = True):
    if isinstance(snapshot, ChildLimit):
        meta = {"hermes": {"subagentLimit": snapshot.model_dump(by_alias=True)}} if metadata else None
        return ToolCallStart(session_update="tool_call", tool_call_id="hermes-subagents:limit",
                             title="Additional Hermes subagents omitted: 64-child display limit reached",
                             status="completed", kind="other", field_meta=meta)
    status = {"running": "in_progress", "completed": "completed", "failed": "failed", "canceled": "failed"}
    title = snapshot.title + (" [record limited]" if snapshot.output_limited or snapshot.activities_limited else "")
    fields = {"tool_call_id": f"hermes-subagent:{snapshot.id}", "status": status[snapshot.status], "title": title}
    if metadata:
        fields["field_meta"] = {"hermes": {"subagentProgress": snapshot.model_dump(by_alias=True)}}
    if start:
        return ToolCallStart(session_update="tool_call", kind="other", **fields)
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
        self._child_limit: ChildLimit | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._connection: Callable[[], Any] = lambda: None
        self._metadata = False
        self._send_lock = asyncio.Lock()
        self._pending: set[asyncio.Task] = set()
        self._outbox: dict[str, tuple[SubagentSnapshot | ChildLimit, bool]] = {}
        self._delivery: asyncio.Task | None = None
        self._restore()

    def bind(self, loop: asyncio.AbstractEventLoop, connection: Callable[[], Any], *, metadata: bool):
        self._loop, self._connection, self._metadata = loop, connection, metadata

    def _restore(self):
        try:
            if self.path.stat().st_size > 8 * 1024 * 1024:
                return
            data = json.loads(self.path.read_text(encoding="utf-8-sig"))
            rows = data.get("children") if isinstance(data, dict) else data
            limit = data.get("childLimit") if isinstance(data, dict) else None
            child_limit = ChildLimit.model_validate(limit) if limit else None
            if not isinstance(rows, list) or len(rows) > 64:
                return
            nodes = [SubagentSnapshot.model_validate(row) for row in rows]
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            logger.warning("Could not restore ACP subagent visibility journal")
            return
        self._child_limit = child_limit
        self._sequence = max([node.sequence for node in nodes] + ([child_limit.sequence] if child_limit else [0]))
        for node in nodes:
            if node.status == "running":
                # A new ACP process cannot claim that a previous process's child is live.
                self._sequence += 1
                node = node.model_copy(update={"status": "failed", "sequence": self._sequence})
            self._nodes[node.id] = node

    def _persist(self):
        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            atomic_json_write(self.path, {"children": [node.model_dump(by_alias=True) for node in self._nodes.values()],
                                         "childLimit": self._child_limit.model_dump(by_alias=True) if self._child_limit else None}, mode=0o600)
        except OSError:
            logger.warning("Could not persist ACP subagent visibility journal")

    def _new_child(self, child_id, kwargs):
        native_depth, parent_id = kwargs.get("depth", 0), kwargs.get("parent_id")
        if not isinstance(native_depth, int) or isinstance(native_depth, bool) or not 0 <= native_depth < 16:
            return None
        depth = native_depth + 1
        if depth == 1:
            parent_id = None
        elif (not isinstance(parent_id, str) or not re.fullmatch(_ID, parent_id) or parent_id == child_id):
            return None
        return SubagentSnapshot(id=child_id, parentId=parent_id, depth=depth, sequence=1,
                                title=public_task_title(kwargs.get("goal")))

    @staticmethod
    def _event_changes(previous, event_type, name, preview, kwargs):
        if event_type == "subagent.text" and isinstance(preview, str):
            raw = previous.text + preview[:16385]
            text = redact_for_egress(raw)
            return {"text": text[:16384], "output_limited": previous.output_limited or len(raw) >= 16384}
        if event_type == "subagent.tool":
            safe_name = redact_for_egress(name[:256]) if isinstance(name, str) else ""
            tool = safe_name if re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", safe_name) else "tool"
            tools = previous.tools + [tool]
            return {"tools": tools[:32], "activities_limited": previous.activities_limited or len(tools) >= 32}
        if event_type == "subagent.complete":
            return {"status": _STATUS.get(kwargs.get("status"), "failed")}
        return {}

    def _schedule(self, node, start):
        if self._loop is not None and not self._loop.is_closed():
            try:
                self._loop.call_soon_threadsafe(self._queue, node, start)
            except RuntimeError:
                logger.debug("ACP loop closed; subagent snapshot retained for replay")

    def _note_omission(self):
        if self._child_limit is None:
            self._sequence += 1
            self._child_limit = ChildLimit(sequence=self._sequence)
            self._persist()
            self._schedule(self._child_limit, True)

    def record(self, event_type: str, name=None, preview=None, args=None, **kwargs):
        if event_type not in _EVENTS:
            return
        child_id = kwargs.get("subagent_id")
        if not isinstance(child_id, str) or not re.fullmatch(_ID, child_id):
            return
        with self._lock:
            previous = self._nodes.get(child_id)
            start = previous is None
            if previous is not None and (previous.status != "running" or event_type == "subagent.start"):
                return
            if start and len(self._nodes) >= 64:
                self._note_omission()
                return
            previous = previous or self._new_child(child_id, kwargs)
            if previous is None:
                return
            changes = self._event_changes(previous, event_type, name, preview, kwargs)
            if not start and all(getattr(previous, key) == value for key, value in changes.items()):
                return
            self._sequence += 1
            node = previous.model_copy(update={**changes, "sequence": self._sequence})
            self._nodes[child_id] = node
            self._persist()
            self._schedule(node, start)

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
            if self._child_limit is not None:
                nodes = [self._child_limit, *nodes]
        for node in nodes:
            await self._send(node, start=True)

    async def drain(self):
        # Let call_soon_threadsafe submissions become tasks before taking the snapshot.
        await asyncio.sleep(0)
        if self._pending:
            await asyncio.gather(*tuple(self._pending))

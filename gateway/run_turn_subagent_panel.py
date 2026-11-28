"""One in-place subagent panel per parent turn, fed by the ``subagent.*`` progress events.

``delegate_task`` relays ``subagent.spawn_requested/start/tool/complete`` through the parent's
``tool_progress_callback`` (``tools/delegate_tool_progress.py``). When the turn's delivery adapter
offers the optional panel capability (``subagent_panel_enabled()`` + ``send_subagent_panel`` /
``update_subagent_panel``, opted in per platform), :func:`observe` keeps one :class:`SubagentPanel`
on the turn's runner and republishes it, throttled to :data:`PUBLISH_INTERVAL_S`. Adapters without the
capability see no change.

The panel is deliberately NOT part of the turn's progress stream: a child's ``parent_cb`` was captured
when it spawned, so a background child keeps reporting here after the parent's reply, and nothing below
checks ``_run_still_current``. The panel itself is the adapter's controls object: ``authority`` (who
may press its buttons, the approval-prompt rule ``gateway.approval_actor`` applies), ``stop(id)`` and
``result(id)``.
"""
from __future__ import annotations

import asyncio
import logging
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from agent.async_utils import safe_schedule_threadsafe
from tools.approval_context import prompt_authority
from tools.delegate_tool_progress import SUBAGENT_FAILURE_STATUSES
from tools.delegate_tool_registry import interrupt_subagent

logger = logging.getLogger(__name__)

PUBLISH_INTERVAL_S = 2.5
RESULT_TEXT_MAX = 12_000  # the Result modal shows the first ~11.9k characters
_EVENTS = frozenset({"subagent.spawn_requested", "subagent.start", "subagent.tool", "subagent.complete"})


@dataclass(frozen=True)
class SubagentRow:
    """One child as the adapter renders it. ``status`` is ``running`` until ``subagent.complete``."""

    subagent_id: str
    goal: str
    status: str
    current_tool: str
    tool_count: int
    elapsed_s: float

    @property
    def running(self) -> bool:
        return self.status == "running"

    @property
    def failed(self) -> bool:
        return self.status in SUBAGENT_FAILURE_STATUSES


@dataclass
class _Child:
    goal: str
    started: float
    status: str = "running"
    current_tool: str = ""
    tool_count: int = 0
    duration_s: Optional[float] = None
    result: Optional[str] = None


class SubagentPanel:
    """Rows for one parent turn plus its publish loop and the button controls."""

    def __init__(self, adapter: Any, chat_id: str, metadata: Optional[dict], loop: Any,
                 authority: dict) -> None:
        self.panel_id = secrets.token_urlsafe(9)
        self.authority = authority
        self._adapter, self._chat_id, self._metadata, self._loop = adapter, chat_id, metadata, loop
        self._lock = threading.Lock()
        self._children: dict[str, _Child] = {}
        self._message_id: Optional[str] = None
        self._last_publish = 0.0
        self._scheduled = False
        self._failed = False

    # ── agent thread side ──
    def apply(self, event_type: str, tool_name: Optional[str], preview: Optional[str], kwargs: dict) -> bool:
        """Fold one relayed event into the rows; True when it changed what the panel shows."""
        sid = str(kwargs.get("subagent_id") or "")
        if not sid or event_type not in _EVENTS or kwargs.get("depth", 0):
            return False
        with self._lock:
            child = self._children.get(sid)
            if child is None:
                child = self._children[sid] = _Child(
                    goal=str(kwargs.get("goal") or preview or ""), started=time.monotonic())
            if event_type == "subagent.tool":
                child.current_tool = str(tool_name or "")
                child.tool_count = int(kwargs.get("tool_count") or child.tool_count + 1)
            elif event_type == "subagent.complete":
                child.status = str(kwargs.get("status") or "completed")
                child.current_tool = ""
                duration = kwargs.get("duration_seconds")
                child.duration_s = float(duration) if duration is not None else time.monotonic() - child.started
                text = kwargs.get("result_text") or kwargs.get("summary") or preview or ""
                child.result = str(text)[:RESULT_TEXT_MAX]
        return True

    def rows(self) -> tuple[SubagentRow, ...]:
        now = time.monotonic()
        with self._lock:
            return tuple(
                SubagentRow(sid, c.goal, c.status, c.current_tool, c.tool_count,
                            c.duration_s if c.duration_s is not None else now - c.started)
                for sid, c in self._children.items())

    def request_publish(self) -> None:
        """Schedule one throttled publish on the gateway loop (coalesces bursts of events)."""
        with self._lock:
            if self._scheduled or self._failed:
                return
            self._scheduled = True
            delay = max(0.0, self._last_publish + PUBLISH_INTERVAL_S - time.monotonic())
        if safe_schedule_threadsafe(self._publish_after(delay), self._loop, logger=logger,
                                    log_message="subagent panel publish scheduling error") is None:
            with self._lock:
                self._scheduled = False

    # ── gateway loop side ──
    async def _publish_after(self, delay: float) -> None:
        if delay:
            await asyncio.sleep(delay)
        with self._lock:
            self._scheduled = False
            self._last_publish = time.monotonic()
        rows = self.rows()
        if self._message_id is None:
            result = await self._adapter.send_subagent_panel(self._chat_id, self, rows, metadata=self._metadata)
            self._message_id = getattr(result, "message_id", None) if getattr(result, "success", False) else None
        else:
            result = await self._adapter.update_subagent_panel(
                self._chat_id, self._message_id, self, rows, metadata=self._metadata)
        if not getattr(result, "success", False):
            # One failed send/update ends the panel for this turn instead of re-posting on every event.
            logger.warning("Subagent panel publish failed; panel disabled for this turn: %s",
                           getattr(result, "error", None))
            with self._lock:
                self._failed = True

    # ── controls (adapter button handlers) ──
    def stop(self, subagent_id: str) -> bool:
        """Interrupt one running child of THIS panel; False when it is unknown or already finished."""
        with self._lock:
            child = self._children.get(subagent_id)
            if child is None or child.status != "running":
                return False
        return interrupt_subagent(subagent_id)

    def result(self, subagent_id: str) -> Optional[str]:
        """The finished child's result text (``""`` when it returned none); None while it runs."""
        with self._lock:
            child = self._children.get(subagent_id)
            return None if child is None or child.status == "running" else (child.result or "")


def _panel_adapter(turn) -> Any:
    adapter = turn._runner._delivery_adapter_for(turn._ctx.source)
    enabled = getattr(adapter, "subagent_panel_enabled", None)
    if not callable(enabled) or not hasattr(adapter, "send_subagent_panel"):
        return None
    return adapter if enabled() is True else None


def observe(turn, event_type: Any, tool_name: Optional[str], preview: Optional[str], kwargs: dict) -> None:
    """Route one ``subagent.*`` progress event into the turn's panel (created on the first event)."""
    if event_type not in _EVENTS:
        return
    panel = getattr(turn, "_subagent_panel", None)
    if panel is None:
        if getattr(turn, "_subagent_panel_checked", False):
            return
        turn._subagent_panel_checked = True
        adapter = _panel_adapter(turn)
        if adapter is None:
            return
        ctx = turn._ctx
        source = ctx.source
        # Read here, on the agent thread, inside the turn's profile scope (``approvals.approvers``):
        # button clicks arrive later on the adapter's loop, where it is not bound.
        authority = prompt_authority(str(source.user_id or ""), str(source.user_name or source.user_id or ""),
                                     str(source.chat_type or ""))
        panel = turn._subagent_panel = SubagentPanel(
            adapter, source.chat_id, ctx._progress_metadata, ctx._loop_for_step, authority)
    if panel.apply(event_type, tool_name, preview, kwargs):
        panel.request_publish()

"""Transport-neutral runtime telemetry projection and editable live HUD delivery.

The projector consumes Hermes' existing structured callback events; the publisher owns
one editable transport message.  Neither path talks to the model, and every failure is
contained so observability can never interrupt an agent turn.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Runtime event payloads can contain arbitrary user or tool data.  The HUD therefore
# renders only this closed vocabulary and numeric counters; it never attempts to make
# opaque payload text safe through pattern matching.
_TOOL_LABELS = {
    "terminal": "shell",
    "execute_code": "code",
    "patch": "edit",
    "write_file": "edit",
    "read_file": "file",
    "search_files": "search",
    "web_search": "web search",
    "web_extract": "web fetch",
    "browser_exec": "browser",
    "delegate_task": "subagent",
}


def _safe_usage(value: Any) -> dict[str, int]:
    """Allow only numeric accounting counters; opaque provider fields never reach the HUD."""
    if not isinstance(value, dict):
        return {}
    allowed = {
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
    }
    out: dict[str, int] = {}
    for key in allowed:
        try:
            if value.get(key) is not None:
                out[key] = max(0, int(value[key]))
        except (TypeError, ValueError):
            continue
    return out


def _format_count(value: Any) -> str:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return ""
    if count < 1000:
        return str(count)
    if count < 1_000_000:
        return f"{count / 1000:.1f}k"
    return f"{count / 1_000_000:.1f}M"


def _runtime(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return (
        f"{hours:02d}:{minutes:02d}:{secs:02d}"
        if hours
        else f"{minutes:02d}:{secs:02d}"
    )


@dataclass
class _ToolView:
    label: str
    is_error: bool = False


@dataclass
class LiveHUDProjector:
    """Pure per-turn state projector for structured Hermes runtime events."""

    started_at: Optional[float] = None
    clock: Callable[[], float] = time.monotonic
    status: str = "active"
    tool_count: int = 0
    current: Optional[_ToolView] = None
    subagents_active: int = 0
    subagents_complete: int = 0
    subagents_failed: int = 0
    api_calls: Any = None
    usage: dict[str, Any] = field(default_factory=dict)
    _pending_tools: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.started_at is None:
            self.started_at = self.clock()

    @property
    def elapsed(self) -> float:
        return max(0.0, self.clock() - float(self.started_at or 0.0))

    def observe(
        self,
        event_type: str,
        tool_name: str = None,
        preview: Any = None,
        args: Any = None,
        **kwargs: Any,
    ) -> None:
        try:
            if (
                event_type == "tool.started"
                and tool_name
                and not str(tool_name).startswith("_")
            ):
                name = str(tool_name)
                self.current = _ToolView(_TOOL_LABELS.get(name, "tool"))
                self.tool_count += 1
                self._pending_tools[name] = self._pending_tools.get(name, 0) + 1
            elif event_type == "tool.completed" and tool_name:
                name = str(tool_name)
                pending = self._pending_tools.get(name, 0)
                if pending > 1:
                    self._pending_tools[name] = pending - 1
                elif pending == 1:
                    self._pending_tools.pop(name, None)
                else:
                    self.tool_count += 1
                label = _TOOL_LABELS.get(name, "tool")
                if self.current is None or self.current.label != label:
                    self.current = _ToolView(label)
                self.current.is_error = bool(kwargs.get("is_error"))
            elif event_type == "subagent.start":
                self.subagents_active += 1
            elif event_type == "subagent.complete":
                self.subagents_active = max(0, self.subagents_active - 1)
                from tools.delegate_tool_progress import SUBAGENT_FAILURE_STATUSES

                if kwargs.get("status") in SUBAGENT_FAILURE_STATUSES:
                    self.subagents_failed += 1
                else:
                    self.subagents_complete += 1
            elif event_type == "usage":
                self.api_calls = kwargs.get("api_calls", self.api_calls)
                self.usage.update(_safe_usage(kwargs.get("usage")))
        except Exception:
            logger.debug("Live HUD projector ignored malformed event")

    def complete(self, result: Any = None) -> None:
        payload = result if isinstance(result, dict) else {}
        if payload.get("interrupted"):
            self.status = "interrupted"
        elif payload.get("failed"):
            self.status = "failed"
        elif "completed" in payload and not payload.get("completed"):
            self.status = "incomplete"
        else:
            self.status = "complete"
        self.api_calls = payload.get("api_calls", self.api_calls)
        self.usage.update(_safe_usage(payload))
        usage = payload.get("usage")
        if isinstance(usage, dict):
            self.usage.update(_safe_usage(usage))

    def _usage_line(self) -> str:
        parts: list[str] = []
        calls = _format_count(self.api_calls)
        if calls:
            parts.append(f"{calls} call{'s' if calls != '1' else ''}")
        input_tokens = _format_count(
            self.usage.get("input_tokens") or self.usage.get("prompt_tokens")
        )
        output_tokens = _format_count(
            self.usage.get("output_tokens") or self.usage.get("completion_tokens")
        )
        if input_tokens:
            parts.append(f"{input_tokens} in")
        if output_tokens:
            parts.append(f"{output_tokens} out")
        return "Usage: " + " · ".join(parts) if parts else ""

    def render(self) -> str:
        if self.status == "complete":
            header = "✓ HERMES · COMPLETE"
        elif self.status == "failed":
            header = "✗ HERMES · FAILED"
        elif self.status == "interrupted":
            header = "■ HERMES · INTERRUPTED"
        elif self.status == "incomplete":
            header = "⚠ HERMES · INCOMPLETE"
        else:
            header = "⚡ HERMES · ACTIVE"
        lines = [header, "━━━━━━━━━━━━━━━━━━", "Task: Current turn"]
        if self.current:
            lines.extend([
                "",
                f"{'✗' if self.current.is_error else '●'} {self.current.label}",
            ])
        if self.subagents_active or self.subagents_complete or self.subagents_failed:
            detail = f"{self.subagents_active} active"
            if self.subagents_complete:
                detail += f" · {self.subagents_complete} complete"
            if self.subagents_failed:
                detail += f" · {self.subagents_failed} failed"
            lines.extend(["", f"Subagents: {detail}"])
        lines.extend([
            "",
            f"Tools: {self.tool_count} · Runtime: {_runtime(self.elapsed)}",
        ])
        usage = self._usage_line()
        if usage:
            lines.append(usage)
        if self.status == "active":
            lines.append("◉ Working…")
        return "\n".join(lines)


class LiveHUDPublisher:
    """Consume structured events and maintain one editable transport message."""

    def __init__(
        self,
        *,
        transport: Any,
        chat_id: Any,
        reply_to: Any = None,
        metadata: Optional[dict] = None,
        min_edit_interval: float = 1.0,
        refresh_interval: float = 2.0,
    ) -> None:
        self.transport = transport
        self.chat_id = chat_id
        self.reply_to = reply_to
        # The HUD is never the turn-final response. Stream-sealing adapters use this marker to keep
        # an observability send from claiming final-delivery ownership.
        self.metadata = {**(metadata or {}), "_interim_send": True}
        self.min_edit_interval = max(0.0, float(min_edit_interval))
        self.refresh_interval = max(self.min_edit_interval, float(refresh_interval))
        self.projector = LiveHUDProjector()
        self.events: queue.Queue = queue.Queue(maxsize=128)
        self.message_id: Optional[str] = None
        self._last_content: Optional[str] = None
        self.failed = False
        self._complete = False
        self._lock = threading.Lock()

    def observe(
        self,
        event_type: str,
        tool_name: str = None,
        preview: Any = None,
        args: Any = None,
        **kwargs: Any,
    ) -> None:
        if not self._complete and not self.failed:
            self._put_latest((event_type, tool_name, preview, args, kwargs))

    def _put_latest(self, item: tuple[Any, ...]) -> None:
        """Offer without blocking the agent thread, evicting one stale event under bursts."""
        try:
            self.events.put_nowait(item)
            return
        except queue.Full:
            pass
        try:
            self.events.get_nowait()
        except queue.Empty:
            pass
        try:
            self.events.put_nowait(item)
        except queue.Full:
            pass

    def complete(self, result: Any = None) -> None:
        with self._lock:
            if self._complete:
                return
            self._complete = True
        self._put_latest(("turn.complete", None, None, None, {"result": result}))

    async def _publish(self, *, final: bool) -> bool:
        content = self.projector.render()
        try:
            if self.message_id is None:
                result = await self.transport.send(
                    self.chat_id,
                    content,
                    reply_to=self.reply_to,
                    metadata=self.metadata,
                )
                if getattr(result, "success", False) and getattr(
                    result, "message_id", None
                ):
                    self.message_id = str(result.message_id)
                    self._last_content = content
                    return True
                self.failed = True
                return False
            kwargs = {
                "chat_id": self.chat_id,
                "message_id": self.message_id,
                "content": content,
                "finalize": final,
            }
            try:
                if (
                    self.metadata
                    and "metadata"
                    in inspect.signature(self.transport.edit_message).parameters
                ):
                    kwargs["metadata"] = self.metadata
            except (TypeError, ValueError):
                pass
            if content == self._last_content and not final:
                return True
            result = await self.transport.edit_message(**kwargs)
            if not getattr(result, "success", False):
                self.failed = True
                return False
            self._last_content = content
            return True
        except asyncio.CancelledError:
            raise
        except Exception:
            self.failed = True
            logger.debug("Live HUD transport update failed")
            return False

    def _drain(self) -> tuple[bool, bool]:
        changed = False
        final = False
        while True:
            try:
                event_type, tool_name, preview, args, kwargs = self.events.get_nowait()
            except queue.Empty:
                break
            changed = True
            if event_type == "turn.complete":
                self.projector.complete(kwargs.get("result"))
                final = True
            else:
                self.projector.observe(event_type, tool_name, preview, args, **kwargs)
        return changed, final

    async def run(self) -> None:
        """Publish until completion. Failures are swallowed and never spawn replacement messages."""
        last_publish = 0.0
        try:
            if not await self._publish(final=False):
                return
            last_publish = time.monotonic()
            while True:
                changed, final = self._drain()
                now = time.monotonic()
                due = changed or (now - last_publish >= self.refresh_interval)
                if due:
                    remaining = self.min_edit_interval - (now - last_publish)
                    if remaining > 0:
                        await asyncio.sleep(remaining)
                    if not await self._publish(final=final):
                        return
                    last_publish = time.monotonic()
                if final:
                    return
                await asyncio.sleep(min(0.1, self.refresh_interval))
        except asyncio.CancelledError:
            raise
        except Exception:
            self.failed = True
            logger.debug("Live HUD publisher failed")

"""Compact in-chat status cards for tool execution progress."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class ToolStatusRow:
    call_id: str
    tool_name: str
    display_name: str
    status: str = "running"
    started_at: str | None = None
    duration: float | None = None
    summary: str | None = None
    error: str | None = None


class ToolStatusCards:
    """Track ordered tool-call rows and render a compact editable card."""

    def __init__(self, *, app_name: str = "Hermes") -> None:
        self.app_name = app_name or "Hermes"
        self._rows: list[ToolStatusRow] = []
        self._counter = 0

    def started(
        self,
        *,
        tool_name: str,
        display_name: str | None = None,
        call_id: str | None = None,
        started_at: str | None = None,
        summary: str | None = None,
    ) -> str:
        row_id = call_id or self._next_id(tool_name)
        existing = self._find_by_id(row_id)
        if existing is None:
            self._rows.append(
                ToolStatusRow(
                    call_id=row_id,
                    tool_name=tool_name,
                    display_name=display_name or self._display_name(tool_name),
                    started_at=started_at,
                    summary=summary,
                )
            )
        else:
            existing.status = "running"
            existing.display_name = display_name or existing.display_name
            existing.started_at = started_at or existing.started_at
            existing.summary = summary or existing.summary
        return row_id

    def completed(
        self,
        *,
        tool_name: str,
        call_id: str | None = None,
        duration: float | None = None,
        summary: str | None = None,
    ) -> None:
        row = self._find_for_finish(call_id=call_id, tool_name=tool_name)
        if row is None:
            row_id = self.started(tool_name=tool_name, call_id=call_id)
            row = self._find_by_id(row_id)
        if row is None:
            return
        row.status = "completed"
        row.duration = duration
        row.summary = summary or row.summary
        row.error = None

    def failed(
        self,
        *,
        tool_name: str,
        call_id: str | None = None,
        duration: float | None = None,
        error: str | None = None,
    ) -> None:
        row = self._find_for_finish(call_id=call_id, tool_name=tool_name)
        if row is None:
            row_id = self.started(tool_name=tool_name, call_id=call_id)
            row = self._find_by_id(row_id)
        if row is None:
            return
        row.status = "failed"
        row.duration = duration
        row.error = self._first_line(error)

    def render(self) -> str:
        return "\n".join(self.render_lines())

    def render_lines(self) -> list[str]:
        lines = [f"*{self.app_name}* · Tool execution"]
        for row in self._rows:
            lines.append(self._render_row(row))
            lines.extend(self._render_detail_lines(row))
        return lines

    @classmethod
    def event_payload(
        cls,
        event_type: str,
        *,
        tool_name: str,
        display_name: str | None = None,
        call_id: str | None = None,
        started_at: str | None = None,
        duration: float | None = None,
        summary: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        return {
            "type": event_type,
            "id": call_id,
            "tool_name": tool_name,
            "display_name": display_name or cls._display_name(tool_name),
            "status": cls._status_for_event(event_type),
            "started_at": started_at,
            "completed_at": (
                datetime.utcnow().isoformat(timespec="seconds") + "Z"
                if event_type in {"tool_call_completed", "tool_call_failed"}
                else None
            ),
            "duration_ms": int(duration * 1000) if duration is not None else None,
            "summary": summary,
            "error": error,
        }

    def apply_event(self, payload: dict[str, Any]) -> None:
        event_type = str(payload.get("type") or "")
        tool_name = str(payload.get("tool_name") or "tool")
        call_id = payload.get("id")
        duration_ms = payload.get("duration_ms")
        duration = None
        if isinstance(duration_ms, (int, float)):
            duration = max(0.0, float(duration_ms) / 1000.0)

        if event_type == "tool_call_started":
            self.started(
                tool_name=tool_name,
                display_name=payload.get("display_name"),
                call_id=str(call_id) if call_id else None,
                started_at=payload.get("started_at"),
                summary=payload.get("summary"),
            )
        elif event_type == "tool_call_completed":
            self.completed(
                tool_name=tool_name,
                call_id=str(call_id) if call_id else None,
                duration=duration,
                summary=payload.get("summary"),
            )
        elif event_type == "tool_call_failed":
            self.failed(
                tool_name=tool_name,
                call_id=str(call_id) if call_id else None,
                duration=duration,
                error=payload.get("error") or payload.get("summary"),
            )

    def _next_id(self, tool_name: str) -> str:
        self._counter += 1
        return f"{tool_name}:{self._counter}"

    def _find_by_id(self, call_id: str) -> ToolStatusRow | None:
        return next((row for row in self._rows if row.call_id == call_id), None)

    def _find_for_finish(
        self,
        *,
        call_id: str | None,
        tool_name: str,
    ) -> ToolStatusRow | None:
        if call_id:
            found = self._find_by_id(call_id)
            if found is not None:
                return found
        return next(
            (
                row
                for row in self._rows
                if row.tool_name == tool_name and row.status == "running"
            ),
            None,
        )

    @staticmethod
    def _display_name(tool_name: str) -> str:
        return f"Run {tool_name}"

    @staticmethod
    def _status_for_event(event_type: str) -> str:
        return {
            "tool_call_started": "running",
            "tool_call_completed": "completed",
            "tool_call_failed": "failed",
        }.get(event_type, "running")

    @staticmethod
    def _first_line(value: str | None) -> str | None:
        if not value:
            return None
        return str(value).strip().splitlines()[0][:180]

    @staticmethod
    def _format_duration(duration: float | None) -> str:
        if duration is None:
            return ""
        return f" in {duration:.1f}s"

    def _render_row(self, row: ToolStatusRow) -> str:
        if row.status == "completed":
            status = f"Completed{self._format_duration(row.duration)}"
            return f"  ✓ {row.display_name} · {status}"
        if row.status == "failed":
            status = f"Failed{self._format_duration(row.duration)}"
            line = f"  ✕ {row.display_name} · {status}"
            if row.error:
                line = f"{line} · {row.error}"
            return line
        started = f" · {row.started_at}" if row.started_at else ""
        return f"  ⏳ {row.display_name} · Running{started}"

    @staticmethod
    def _render_detail_lines(row: ToolStatusRow) -> list[str]:
        if not row.summary:
            return []
        detail = str(row.summary).rstrip()
        if not detail:
            return []
        return [f"    {line}" if line else "    " for line in detail.splitlines()]


__all__ = ["ToolStatusCards", "ToolStatusRow"]

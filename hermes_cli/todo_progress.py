"""Pure state and rendering helpers for the interactive todo progress tray.

The model-facing todo schema stays unchanged. This module projects that one
canonical list into compact, expanded, and pipe-safe text views and owns only
presentation state (selection, collapse and confirmation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import unicodedata

_VALID_STATUSES = {"pending", "in_progress", "completed", "cancelled"}
_GLYPHS = {
    "pending": "[ ]",
    "in_progress": "[>]",
    "completed": "[x]",
    "cancelled": "[-]",
}
_STYLES = {
    "pending": "class:todo-pending",
    "in_progress": "class:todo-active",
    "completed": "class:todo-dim",
    "cancelled": "class:todo-dim",
}


def _one_line(value: Any) -> str:
    flattened = " ".join(str(value or "").split())
    return "".join(
        char
        for char in flattened
        # Strip control/format/unpaired-surrogate chars AND line/paragraph
        # separators (U+2028/U+2029) — the latter render as hard breaks in
        # prompt_toolkit and would split a single panel row.
        if unicodedata.category(char) not in {"Cc", "Cf", "Cs", "Zl", "Zp"}
    )


def _clean_items(items: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        # Keep the canonical id byte-for-byte (apart from outer whitespace) so
        # selection still maps back to TodoStore. Sanitize it only when shown.
        item_id = str(raw.get("id") or "").strip()
        content = _one_line(raw.get("content"))
        status = str(raw.get("status") or "pending").strip().lower()
        if not item_id or not content or status not in _VALID_STATUSES:
            continue
        cleaned.append({"id": item_id, "content": content, "status": status})
    return cleaned


def _char_width(char: str) -> int:
    if not char or unicodedata.combining(char):
        return 0
    if unicodedata.category(char).startswith("C"):
        return 0
    return 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1


def _display_width(text: str) -> int:
    return sum(_char_width(char) for char in text)


def _truncate(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if _display_width(text) <= width:
        return text
    if width == 1:
        return "…"
    out: list[str] = []
    used = 0
    for char in text:
        size = _char_width(char)
        if used + size > width - 1:
            break
        out.append(char)
        used += size
    return "".join(out) + "…"


def _counted(items: list[dict[str, str]]) -> list[dict[str, str]]:
    return [item for item in items if item["status"] != "cancelled"]


def _focus_item(items: list[dict[str, str]]) -> dict[str, str] | None:
    return (
        next((item for item in items if item["status"] == "in_progress"), None)
        or next((item for item in items if item["status"] == "pending"), None)
        or (items[-1] if items else None)
    )


@dataclass
class TodoPanelState:
    expanded: bool = False
    selected_id: str | None = None
    show_completed: bool = True
    confirmation: tuple[str, str, int | None] | None = None
    notice: str = ""

    def visible_items(self, items: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
        rows = _clean_items(items)
        if self.show_completed:
            return rows
        return [row for row in rows if row["status"] not in {"completed", "cancelled"}]

    def _ensure_selection(self, items: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
        rows = self.visible_items(items)
        if not rows:
            self.selected_id = None
            return rows
        if not any(row["id"] == self.selected_id for row in rows):
            focus = _focus_item(rows)
            self.selected_id = focus["id"] if focus else rows[0]["id"]
        return rows

    def toggle(self, items: Iterable[dict[str, Any]]) -> bool:
        rows = _clean_items(items)
        if not rows:
            return False
        self.expanded = not self.expanded
        self.confirmation = None
        self.notice = ""
        if self.expanded:
            self._ensure_selection(rows)
        return True

    def close(self) -> None:
        self.expanded = False
        self.confirmation = None
        self.notice = ""

    def move(self, items: Iterable[dict[str, Any]], delta: int) -> None:
        rows = self._ensure_selection(items)
        if not rows:
            return
        index = next((i for i, row in enumerate(rows) if row["id"] == self.selected_id), 0)
        index = max(0, min(len(rows) - 1, index + delta))
        self.selected_id = rows[index]["id"]
        self.confirmation = None
        self.notice = ""

    def toggle_completed(self, items: Iterable[dict[str, Any]]) -> None:
        self.show_completed = not self.show_completed
        self.confirmation = None
        self.notice = ""
        self._ensure_selection(items)

    def request_status(
        self,
        items: Iterable[dict[str, Any]],
        status: str,
        *,
        expected_revision: int | None = None,
    ) -> bool:
        normalized = str(status or "").strip().lower()
        if normalized not in _VALID_STATUSES:
            return False
        rows = self._ensure_selection(items)
        selected = next((row for row in rows if row["id"] == self.selected_id), None)
        if selected is None or selected["status"] == normalized:
            return False
        self.confirmation = (selected["id"], normalized, expected_revision)
        verb = "Mark" if normalized == "completed" else "Reopen" if normalized == "pending" else "Set"
        self.notice = f'{verb} "{selected["content"]}" {normalized}? Enter confirm · Esc cancel'
        return True

    def cancel_confirmation(self) -> bool:
        if self.confirmation is None:
            return False
        self.confirmation = None
        self.notice = ""
        return True

    def confirm(self, store: Any) -> bool:
        if self.confirmation is None:
            return False
        item_id, status, expected_revision = self.confirmation
        items = store.read()
        selected = next((row for row in items if row.get("id") == item_id), None)
        changed = bool(
            store.update_status(
                item_id,
                status,
                actor="user",
                expected_revision=expected_revision,
            )
        )
        self.confirmation = None
        if changed and selected:
            self.notice = f'Task changed by you: {selected["content"]} → {status}'
        else:
            self.notice = "Task changed before confirmation; refreshed."
        return changed


def _fragments(lines: list[tuple[str, str]]) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for index, (style, text) in enumerate(lines):
        result.append((style, text + ("\n" if index < len(lines) - 1 else "")))
    return result


def format_todo_panel_fragments(
    items: Iterable[dict[str, Any]],
    state: TodoPanelState,
    *,
    width: int = 80,
    max_rows: int = 8,
) -> list[tuple[str, str]]:
    rows = _clean_items(items)
    if not rows:
        return []

    counted = _counted(rows)
    completed = sum(row["status"] == "completed" for row in counted)
    header = f"Tasks {completed}/{len(counted)}"
    focus = _focus_item(rows)

    if not state.expanded:
        active = f'{_GLYPHS[focus["status"]]} {focus["content"]}' if focus else ""
        if width < 40:
            return _fragments(
                [
                    ("class:todo-header", _truncate(header, width)),
                    (_STYLES.get(focus["status"], "class:todo-pending") if focus else "", _truncate(active, width)),
                    ("class:todo-hint", _truncate("Ctrl+T details", width)),
                ]
            )
        suffix = "Ctrl+T details"
        available = max(1, width - _display_width(header) - _display_width(suffix) - 6)
        active = _truncate(active, available)
        line = f"{header} · {active} · {suffix}"
        if width < 60:
            return _fragments(
                [
                    ("class:todo-header", _truncate(f"{header} · {active}", width)),
                    ("class:todo-hint", suffix),
                ]
            )
        return [("class:todo-header", _truncate(line, width))]

    visible = state._ensure_selection(rows)
    # Header + footer + optional overflow consume two or three rows. Leave the
    # rest for tasks, preserving authored order and keeping the active task in
    # the natural viewport when it appears near the start of a normal plan.
    task_budget = max(1, max_rows - 3)
    selected_index = next(
        (index for index, row in enumerate(visible) if row["id"] == state.selected_id),
        0,
    )
    start = max(0, selected_index - task_budget // 2)
    end = min(len(visible), start + task_budget)
    start = max(0, end - task_budget)
    shown = visible[start:end]
    overflow = max(0, len(visible) - len(shown))
    lines: list[tuple[str, str]] = [("class:todo-header", _truncate(header, width))]
    for row in shown:
        selected = row["id"] == state.selected_id
        prefix = "> " if selected else "  "
        label = f'{prefix}{_GLYPHS[row["status"]]} {row["content"]}'
        style = "class:todo-selected" if selected else _STYLES[row["status"]]
        lines.append((style, _truncate(label, width)))
    if overflow:
        lines.append(("class:todo-hint", f"... {overflow} more tasks"))
    completed_action = "hide done" if state.show_completed else "show done"
    footer = state.notice or f"↑↓ select · m done · u reopen · c {completed_action} · Esc close"
    lines.append(("class:todo-hint", _truncate(footer, width)))
    return _fragments(lines[:max_rows])


def format_todo_plain_snapshot(items: Iterable[dict[str, Any]]) -> list[str]:
    rows = _clean_items(items)
    counts = {status: 0 for status in ("completed", "in_progress", "pending", "cancelled")}
    output: list[str] = []
    for row in rows:
        counts[row["status"]] += 1
        item_id = _one_line(row["id"]) or "?"
        output.append(f'TASK {item_id} {row["status"].upper()} {row["content"]}')
    output.append(
        "SUMMARY "
        + " ".join(f"{status}={counts[status]}" for status in ("completed", "in_progress", "pending", "cancelled"))
    )
    return output

"""Hermes CLI Kanban View - TUI board for todo tasks.

A curses-based three-column kanban board displaying todo tasks.
Supports keyboard navigation, priority colors, and task detail viewing.
Integrates with TodoStore and skin engine for consistent styling.

USAGE
=====
    from hermes_cli.kanban_view import run_kanban_view
    from tools.todo_tool import TodoStore
    
    store = TodoStore()
    # ... populate store ...
    run_kanban_view(store)

KEYBOARD NAVIGATION
-------------------
    Left/Right (h/l or arrows): Switch columns
    Up/Down (k/j or arrows):    Select task within column
    Enter:                     View task details
    d:                         Toggle task status (pending -> in_progress -> completed)
    r:                         Refresh view
    q/ESC:                     Exit

COLUMNS
-------
    Pending      - Tasks waiting to start
    In Progress  - Tasks currently being worked on
    Completed    - Finished tasks (also shows cancelled)
"""

import sys
import curses
from typing import Dict, List, Optional, Any, Tuple

from hermes_cli.skin_engine import get_active_skin, SkinConfig
from hermes_cli.curses_ui import flush_stdin


# Priority color mapping (fallback ANSI colors for terminals without hex support)
PRIORITY_ANSI_COLORS = {
    "critical": 1,   # Red
    "high": 2,       # Yellow/Orange
    "medium": 3,     # Green
    "low": 4,        # Blue/Cyan
}

# Priority markers for display
PRIORITY_MARKERS = {
    "critical": "!!!",
    "high": "!!",
    "medium": "!",
    "low": ".",
}

# Status markers
STATUS_MARKERS = {
    "pending": "[ ]",
    "in_progress": "[>]",
    "completed": "[x]",
    "cancelled": "[~]",
}

# Type markers
TYPE_MARKERS = {
    "task": "",
    "milestone": "[M]",
    "blocked": "[B]",
}


def _hex_to_curses_color(hex_color: str) -> int:
    """Convert hex color to curses RGB components (0-1000 scale)."""
    try:
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        # Scale to curses 0-1000 range
        return (r * 1000 // 255, g * 1000 // 255, b * 1000 // 255)
    except (ValueError, IndexError):
        return (500, 500, 500)  # Fallback gray


def _init_colors(stdscr, skin: SkinConfig) -> Dict[str, int]:
    """Initialize curses color pairs from skin config.
    
    Returns a dict mapping color names to curses color pair numbers.
    """
    colors: Dict[str, int] = {}
    
    if not curses.has_colors():
        return colors
    
    curses.start_color()
    curses.use_default_colors()
    
    # Priority colors (pairs 1-4)
    priority_hex = {
        "critical": skin.get_color("ui_error", "#ef5350"),
        "high": skin.get_color("ui_warn", "#ffa726"),
        "medium": skin.get_color("ui_ok", "#4caf50"),
        "low": skin.get_color("ui_label", "#4dd0e1"),
    }
    
    for i, (priority, hex_val) in enumerate(priority_hex.items(), start=1):
        r, g, b = _hex_to_curses_color(hex_val)
        try:
            curses.init_color(i + 10, r, g, b)  # Custom color definition
            curses.init_pair(i, i + 10, -1)  # Pair with transparent background
        except curses.error:
            # Fallback to standard ANSI colors if custom colors fail
            curses.init_pair(i, PRIORITY_ANSI_COLORS[priority], -1)
        colors[priority] = i
    
    # UI accent colors (pairs 5-8)
    ui_colors = {
        "accent": skin.get_color("ui_accent", "#FFBF00"),
        "header": skin.get_color("banner_title", "#FFD700"),
        "dim": skin.get_color("banner_dim", "#B8860B"),
        "selected": skin.get_color("ui_label", "#4dd0e1"),
    }
    
    for i, (name, hex_val) in enumerate(ui_colors.items(), start=5):
        r, g, b = _hex_to_curses_color(hex_val)
        try:
            curses.init_color(i + 10, r, g, b)
            curses.init_pair(i, i + 10, -1)
        except curses.error:
            # Fallback
            curses.init_pair(i, curses.COLOR_YELLOW if name == "accent" else curses.COLOR_WHITE, -1)
        colors[name] = i
    
    # Column header colors (pairs 9-11)
    column_colors = {
        "pending": skin.get_color("ui_warn", "#ffa726"),
        "in_progress": skin.get_color("ui_accent", "#FFBF00"),
        "completed": skin.get_color("ui_ok", "#4caf50"),
    }
    
    for i, (col, hex_val) in enumerate(column_colors.items(), start=9):
        r, g, b = _hex_to_curses_color(hex_val)
        try:
            curses.init_color(i + 10, r, g, b)
            curses.init_pair(i, i + 10, -1)
        except curses.error:
            curses.init_pair(i, curses.COLOR_YELLOW if col == "pending" else 
                           curses.COLOR_GREEN if col == "completed" else curses.COLOR_CYAN, -1)
        colors[f"col_{col}"] = i
    
    return colors


class KanbanBoard:
    """Curses-based three-column kanban board for todo tasks."""
    
    COLUMNS = ["pending", "in_progress", "completed"]
    COLUMN_LABELS = {
        "pending": "Pending",
        "in_progress": "In Progress",
        "completed": "Completed",
    }
    
    def __init__(self, store, skin: Optional[SkinConfig] = None):
        self.store = store
        self.skin = skin or get_active_skin()
        self.colors: Dict[str, int] = {}
        
        # Navigation state
        self.current_col = 0  # 0=pending, 1=in_progress, 2=completed
        self.cursor_pos = 0   # Position within current column
        self.scroll_offset = 0
        
        # View state
        self.show_details = False
        self.selected_task: Optional[Dict[str, Any]] = None
        self.detail_scroll = 0
        
        # Tasks by column
        self.tasks_by_column: Dict[str, List[Dict[str, Any]]] = {}
        self._refresh_tasks()
    
    def _refresh_tasks(self):
        """Reload tasks from store and organize by status."""
        items = self.store.read()
        self.tasks_by_column = {
            "pending": [],
            "in_progress": [],
            "completed": [],
        }
        
        # Include cancelled tasks in completed column
        for item in items:
            status = item.get("status", "pending")
            if status == "cancelled":
                self.tasks_by_column["completed"].append(item)
            elif status in self.tasks_by_column:
                self.tasks_by_column[status].append(item)
        
        # Sort by priority within each column
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        for col in self.tasks_by_column:
            self.tasks_by_column[col].sort(
                key=lambda t: (priority_order.get(t.get("priority", "medium"), 2), t.get("id", ""))
            )
        
        # Reset cursor if needed
        current_tasks = self._get_current_tasks()
        if self.cursor_pos >= len(current_tasks):
            self.cursor_pos = max(0, len(current_tasks) - 1)
    
    def _get_current_tasks(self) -> List[Dict[str, Any]]:
        """Get tasks for the currently selected column."""
        col_name = self.COLUMNS[self.current_col]
        return self.tasks_by_column.get(col_name, [])
    
    def _get_current_task(self) -> Optional[Dict[str, Any]]:
        """Get the currently selected task."""
        tasks = self._get_current_tasks()
        if 0 <= self.cursor_pos < len(tasks):
            return tasks[self.cursor_pos]
        return None
    
    def _truncate_text(self, text: str, max_len: int) -> str:
        """Truncate text to max length with ellipsis."""
        if len(text) <= max_len:
            return text
        return text[:max_len - 3] + "..."
    
    def _format_task_line(self, task: Dict[str, Any], is_selected: bool, max_width: int) -> Tuple[str, int]:
        """Format a single task line for display.
        
        Returns (formatted_line, color_pair_number).
        """
        priority = task.get("priority", "medium")
        task_type = task.get("type", "task")
        
        # Build the display string
        priority_mark = PRIORITY_MARKERS.get(priority, "!")
        type_mark = TYPE_MARKERS.get(task_type, "")
        
        # ID and truncated content
        task_id = task.get("id", "?")
        content = self._truncate_text(task.get("content", ""), max_width - 15)
        
        # Assigned to
        assigned = task.get("assigned_to", "")
        assigned_str = f"@{assigned}" if assigned else ""
        
        # Format: [priority] id. content [type] @assigned
        parts = [f"{priority_mark}", f"{task_id}.", content]
        if type_mark:
            parts.append(type_mark)
        if assigned_str:
            parts.append(assigned_str)
        
        line = " ".join(parts)
        
        # Get color for priority
        color_pair = self.colors.get(priority, 0)
        
        return line, color_pair
    
    def _draw_header(self, stdscr, max_x: int, max_y: int) -> int:
        """Draw the header row. Returns the y position after header."""
        agent_name = self.skin.get_branding("agent_name", "Hermes Agent")
        header_attr = curses.A_BOLD
        if "header" in self.colors:
            header_attr |= curses.color_pair(self.colors["header"])
        
        try:
            stdscr.addnstr(0, 0, f"{agent_name} - Kanban Board", max_x - 1, header_attr)
        except curses.error:
            pass
        
        # Help line
        help_text = "h/l: cols  j/k: tasks  Enter: details  d: toggle  r: refresh  q: quit"
        try:
            stdscr.addnstr(1, 0, help_text, max_x - 1, curses.A_DIM)
        except curses.error:
            pass
        
        return 2
    
    def _draw_columns(self, stdscr, start_y: int, max_x: int, max_y: int):
        """Draw the three column headers and task lists."""
        # Calculate column widths
        col_width = max_x // 3
        if col_width < 20:
            col_width = 20  # Minimum width
        
        # Draw column headers
        for i, col_name in enumerate(self.COLUMNS):
            x_offset = i * col_width
            label = self.COLUMN_LABELS[col_name]
            count = len(self.tasks_by_column.get(col_name, []))
            
            header_text = f"{label} ({count})"
            attr = curses.A_BOLD
            
            # Use column-specific color
            col_color_key = f"col_{col_name}"
            if col_color_key in self.colors:
                attr |= curses.color_pair(self.colors[col_color_key])
            elif "accent" in self.colors:
                attr |= curses.color_pair(self.colors["accent"])
            
            try:
                stdscr.addnstr(start_y, x_offset, header_text, col_width - 1, attr)
                # Draw separator line
                stdscr.addnstr(start_y + 1, x_offset, "-" * (col_width - 2), col_width - 1, curses.A_DIM)
            except curses.error:
                pass
        
        # Draw tasks in each column
        task_start_y = start_y + 2
        visible_rows = max_y - task_start_y - 2  # Leave room for status bar
        
        # Calculate scroll for current column
        current_tasks = self._get_current_tasks()
        if self.cursor_pos < self.scroll_offset:
            self.scroll_offset = self.cursor_pos
        elif self.cursor_pos >= self.scroll_offset + visible_rows:
            self.scroll_offset = self.cursor_pos - visible_rows + 1
        
        for col_idx, col_name in enumerate(self.COLUMNS):
            x_offset = col_idx * col_width
            tasks = self.tasks_by_column.get(col_name, [])
            
            # Draw tasks
            for draw_i, task_i in enumerate(range(self.scroll_offset, min(len(tasks), self.scroll_offset + visible_rows))):
                y = task_start_y + draw_i
                if y >= max_y - 2:
                    break
                
                task = tasks[task_i]
                is_selected = (col_idx == self.current_col and task_i == self.cursor_pos)
                
                line, color_pair = self._format_task_line(task, is_selected, col_width - 4)
                
                # Add selection indicator
                prefix = ">" if is_selected else " "
                full_line = f"{prefix} {line}"
                
                attr = curses.A_NORMAL
                if is_selected:
                    attr = curses.A_BOLD | curses.A_REVERSE
                elif color_pair:
                    attr |= curses.color_pair(color_pair)
                
                try:
                    stdscr.addnstr(y, x_offset, full_line, col_width - 1, attr)
                except curses.error:
                    pass
            
            # Show count if more tasks than visible
            if len(tasks) > visible_rows:
                more_count = len(tasks) - visible_rows - self.scroll_offset
                if more_count > 0:
                    try:
                        y = max_y - 2
                        stdscr.addnstr(y, x_offset, f"... +{more_count} more", col_width - 1, curses.A_DIM)
                    except curses.error:
                        pass
    
    def _draw_status_bar(self, stdscr, max_x: int, max_y: int):
        """Draw the bottom status bar."""
        status_y = max_y - 1
        
        # Count totals
        total = sum(len(self.tasks_by_column.get(col, [])) for col in self.COLUMNS)
        pending = len(self.tasks_by_column.get("pending", []))
        in_progress = len(self.tasks_by_column.get("in_progress", []))
        completed = len(self.tasks_by_column.get("completed", []))
        
        # Blocked count
        blocked = len(self.store.get_blocked_tasks())
        
        status_text = f"Total: {total} | Pending: {pending} | In Progress: {in_progress} | Completed: {completed}"
        if blocked > 0:
            status_text += f" | Blocked: {blocked}"
        
        attr = curses.A_DIM
        if "dim" in self.colors:
            attr |= curses.color_pair(self.colors["dim"])
        
        try:
            stdscr.addnstr(status_y, 0, status_text, max_x - 1, attr)
        except curses.error:
            pass
    
    def _draw_detail_view(self, stdscr, max_x: int, max_y: int):
        """Draw the task detail overlay."""
        task = self.selected_task
        if not task:
            return
        
        # Draw a bordered detail panel
        panel_width = min(60, max_x - 4)
        panel_height = min(20, max_y - 4)
        panel_x = (max_x - panel_width) // 2
        panel_y = (max_y - panel_height) // 2
        
        # Draw border
        try:
            # Top border
            stdscr.addnstr(panel_y, panel_x, "+" + "-" * (panel_width - 2) + "+", panel_width, curses.A_BOLD)
            # Side borders
            for y in range(panel_y + 1, panel_y + panel_height - 1):
                stdscr.addnstr(y, panel_x, "|", 1, curses.A_BOLD)
                stdscr.addnstr(y, panel_x + panel_width - 1, "|", 1, curses.A_BOLD)
            # Bottom border
            stdscr.addnstr(panel_y + panel_height - 1, panel_x, "+" + "-" * (panel_width - 2) + "+", panel_width, curses.A_BOLD)
        except curses.error:
            pass
        
        # Draw content
        content_x = panel_x + 2
        content_width = panel_width - 4
        
        lines = []
        lines.append(("Task Details", curses.A_BOLD))
        lines.append(("", curses.A_NORMAL))
        
        # Task info
        task_id = task.get("id", "?")
        content = task.get("content", "(no description)")
        status = task.get("status", "pending")
        priority = task.get("priority", "medium")
        task_type = task.get("type", "task")
        assigned = task.get("assigned_to", "")
        created = task.get("created_at", "")
        depends_on = task.get("depends_on", [])
        
        lines.append((f"ID: {task_id}", curses.A_NORMAL))
        lines.append((f"Status: {status}", curses.A_NORMAL))
        lines.append((f"Priority: {priority}", curses.color_pair(self.colors.get(priority, 0)) if priority in self.colors else curses.A_NORMAL))
        lines.append((f"Type: {task_type}", curses.A_NORMAL))
        lines.append(("", curses.A_NORMAL))
        
        # Content (wrap if needed)
        content_lines = []
        words = content.split()
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 <= content_width:
                current_line += (word if not current_line else " " + word)
            else:
                if current_line:
                    content_lines.append(current_line)
                current_line = word
        if current_line:
            content_lines.append(current_line)
        
        lines.append(("Content:", curses.A_BOLD))
        for cl in content_lines[:3]:  # Limit to 3 lines
            lines.append((cl, curses.A_NORMAL))
        
        if assigned:
            lines.append(("", curses.A_NORMAL))
            lines.append((f"Assigned to: @{assigned}", curses.A_NORMAL))
        
        if depends_on:
            lines.append(("", curses.A_NORMAL))
            lines.append((f"Depends on: {', '.join(depends_on)}", curses.A_NORMAL))
        
        if created:
            lines.append(("", curses.A_NORMAL))
            lines.append((f"Created: {created[:19]}", curses.A_DIM))  # Truncate timestamp
        
        lines.append(("", curses.A_NORMAL))
        lines.append(("Press Enter/ESC to close", curses.A_DIM))
        
        # Draw lines with scrolling
        visible_content_height = panel_height - 4
        for i, (text, attr) in enumerate(lines[self.detail_scroll:self.detail_scroll + visible_content_height]):
            y = panel_y + 2 + i
            if y >= panel_y + panel_height - 2:
                break
            try:
                stdscr.addnstr(y, content_x, text, content_width, attr)
            except curses.error:
                pass
    
    def _draw(self, stdscr):
        """Main draw function."""
        stdscr.clear()
        max_y, max_x = stdscr.getmaxyx()
        
        y = self._draw_header(stdscr, max_x, max_y)
        self._draw_columns(stdscr, y, max_x, max_y)
        self._draw_status_bar(stdscr, max_x, max_y)
        
        if self.show_details and self.selected_task:
            self._draw_detail_view(stdscr, max_x, max_y)
        
        stdscr.refresh()
    
    def _handle_key(self, key) -> bool:
        """Handle keyboard input. Returns True to continue, False to exit."""
        # Exit keys
        if key in (ord("q"), 27):  # q or ESC
            return False
        
        # Detail view mode
        if self.show_details:
            if key in (curses.KEY_ENTER, 10, 13, 27, ord("q")):
                self.show_details = False
                self.selected_task = None
                self.detail_scroll = 0
            elif key == curses.KEY_UP or key == ord("k"):
                self.detail_scroll = max(0, self.detail_scroll - 1)
            elif key == curses.KEY_DOWN or key == ord("j"):
                self.detail_scroll += 1
            return True
        
        # Column navigation
        if key == curses.KEY_LEFT or key == ord("h"):
            self.current_col = (self.current_col - 1) % 3
            self.cursor_pos = 0
            self.scroll_offset = 0
        elif key == curses.KEY_RIGHT or key == ord("l"):
            self.current_col = (self.current_col + 1) % 3
            self.cursor_pos = 0
            self.scroll_offset = 0
        
        # Task navigation within column
        elif key == curses.KEY_UP or key == ord("k"):
            tasks = self._get_current_tasks()
            if tasks:
                self.cursor_pos = (self.cursor_pos - 1) % len(tasks)
        elif key == curses.KEY_DOWN or key == ord("j"):
            tasks = self._get_current_tasks()
            if tasks:
                self.cursor_pos = (self.cursor_pos + 1) % len(tasks)
        
        # View task details
        elif key in (curses.KEY_ENTER, 10, 13):
            task = self._get_current_task()
            if task:
                self.selected_task = task
                self.show_details = True
                self.detail_scroll = 0
        
        # Toggle task status
        elif key == ord("d"):
            task = self._get_current_task()
            if task:
                current_status = task.get("status", "pending")
                # Cycle: pending -> in_progress -> completed -> pending
                status_cycle = ["pending", "in_progress", "completed"]
                current_idx = status_cycle.index(current_status) if current_status in status_cycle else 0
                new_status = status_cycle[(current_idx + 1) % 3]
                
                # Update via store
                self.store.write([{"id": task["id"], "status": new_status}], merge=True)
                self._refresh_tasks()
        
        # Refresh
        elif key == ord("r"):
            self._refresh_tasks()
        
        return True
    
    def run(self, stdscr):
        """Main loop."""
        self.colors = _init_colors(stdscr, self.skin)
        curses.curs_set(0)  # Hide cursor
        
        while True:
            self._draw(stdscr)
            key = stdscr.getch()
            if not self._handle_key(key):
                break


def run_kanban_view(store, skin: Optional[SkinConfig] = None) -> None:
    """Run the kanban board TUI.
    
    Args:
        store: TodoStore instance containing tasks
        skin: Optional SkinConfig for styling (defaults to active skin)
    
    This function handles the curses wrapper and terminal cleanup.
    Safe to call even when stdin is not a TTY - will just print a message.
    """
    if not sys.stdin.isatty():
        print("Kanban view requires an interactive terminal.")
        # Print a simple text summary instead
        items = store.read()
        if not items:
            print("No tasks in todo list.")
            return
        
        print("\n=== Kanban Summary ===\n")
        for status in ["pending", "in_progress", "completed"]:
            tasks = [t for t in items if t.get("status") == status]
            if tasks:
                print(f"{status.upper()} ({len(tasks)}):")
                for t in tasks:
                    priority = t.get("priority", "medium")
                    marker = PRIORITY_MARKERS.get(priority, "!")
                    print(f"  {marker} {t.get('id')}. {t.get('content')}")
                print()
        return
    
    try:
        board = KanbanBoard(store, skin)
        curses.wrapper(board.run)
        flush_stdin()
    except Exception as e:
        print(f"Error running kanban view: {e}")
        flush_stdin()


def kanban_view_from_store(store) -> None:
    """Convenience function to run kanban view from a TodoStore.
    
    This is the primary entry point for the kanban TUI.
    """
    run_kanban_view(store)


# For standalone testing
if __name__ == "__main__":
    from tools.todo_tool import TodoStore
    
    # Create a demo store with sample tasks
    store = TodoStore()
    store.write([
        {"id": "1", "content": "Design API endpoints", "status": "completed", "priority": "high"},
        {"id": "2", "content": "Implement authentication", "status": "in_progress", "priority": "critical", "assigned_to": "Alice"},
        {"id": "3", "content": "Write unit tests", "status": "pending", "priority": "medium", "depends_on": ["2"]},
        {"id": "4", "content": "Create documentation", "status": "pending", "priority": "low"},
        {"id": "5", "content": "Deploy to staging", "status": "pending", "priority": "high", "depends_on": ["1", "2"]},
        {"id": "6", "content": "Milestone: Beta release", "status": "pending", "priority": "critical", "type": "milestone"},
    ])
    
    run_kanban_view(store)
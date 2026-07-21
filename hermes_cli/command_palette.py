"""Command palette for quick command access (Ctrl+P).

Provides a filterable overlay listing all available commands with fuzzy search.
"""

from __future__ import annotations

from typing import Callable

from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import VSplit
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.widgets.dialogs import Dialog
from prompt_toolkit.widgets import TextArea


def fuzzy_match(text: str, query: str) -> bool:
    """Check if query matches text with fuzzy matching."""
    if not query:
        return True
    text_lower = text.lower()
    query_lower = query.lower()
    if query_lower in text_lower:
        return True
    query_idx = 0
    for char in text_lower:
        if query_idx < len(query_lower) and char == query_lower[query_idx]:
            query_idx += 1
    return query_idx == len(query_lower)


class CommandPalette:
    """Command palette dialog with fuzzy search."""

    def __init__(
        self,
        commands: list[dict],
        on_select: Callable[[str], None],
        on_close: Callable[[], None],
    ):
        self._commands = commands
        self._on_select = on_select
        self._on_close = on_close
        self._query = ""
        self._selected_index = 0
        self._kb = KeyBindings()
        self._filtered_commands: list[dict] = []
        self._setup_key_bindings()
        self._filter_commands()
        self._dialog = self._create_dialog()
        self._search_area: TextArea | None = None
        self._list_control: FormattedTextControl | None = None

    def _filter_commands(self):
        """Filter commands based on current query."""
        if not self._query:
            self._filtered_commands = self._commands.copy()
        else:
            self._filtered_commands = [
                cmd
                for cmd in self._commands
                if fuzzy_match(cmd["name"], self._query)
                or fuzzy_match(cmd["description"], self._query)
                or any(
                    fuzzy_match(alias, self._query) for alias in cmd.get("aliases", [])
                )
            ]
        self._selected_index = min(
            self._selected_index, max(0, len(self._filtered_commands) - 1)
        )

    def _setup_key_bindings(self):
        """Setup key bindings for the command palette."""

        @self._kb.add(Keys.Up)
        def _up(event):
            self._selected_index = max(0, self._selected_index - 1)
            self._refresh_list()

        @self._kb.add(Keys.Down)
        def _down(event):
            self._selected_index = min(
                len(self._filtered_commands) - 1, self._selected_index + 1
            )
            self._refresh_list()

        @self._kb.add(Keys.Enter)
        def _enter(event):
            if self._filtered_commands:
                cmd = self._filtered_commands[self._selected_index]
                self._on_select(cmd["name"])

        @self._kb.add(Keys.Escape)
        def _escape(event):
            self._on_close()

    def _create_dialog(self) -> Dialog:
        """Create the command palette dialog."""
        self._search_area = TextArea(
            multiline=False,
            accept_handler=self._on_search_submit,
            get_line_prefix=self._get_search_prompt,
        )

        def on_text_changed(buf):
            self._query = buf.text
            self._filter_commands()
            self._refresh_list()

        self._search_area.buffer.on_text_changed = on_text_changed

        self._list_control = FormattedTextControl(
            text=self._get_list_text,
            focusable=True,
            key_bindings=self._kb,
        )
        list_window = Window(self._list_control, width=70, height=18)

        self._dialog = Dialog(
            title=FormattedText(
                [
                    ("", "Command Palette "),
                    ("fg:ansicyan", "Ctrl+P"),
                    ("", " (↑↓ navigate, Enter select, Esc close)"),
                ]
            ),
            body=VSplit(
                [
                    self._search_area,
                    list_window,
                ]
            ),
            with_background=True,
        )
        return self._dialog

    def _get_search_prompt(self, line_no, other):
        return FormattedText([("fg:ansiyellow", "> ")])

    def _on_search_submit(self):
        """Handle search input submission."""
        if self._filtered_commands:
            cmd = self._filtered_commands[self._selected_index]
            self._on_select(cmd["name"])

    def _refresh_list(self):
        """Refresh the command list display."""
        if self._list_control:
            self._list_control.text = self._get_list_text()

    def _get_list_text(self) -> FormattedText:
        """Get formatted text for the command list."""
        lines: list[tuple[str, str]] = []
        categories: dict[str, list[dict]] = {}
        for cmd in self._filtered_commands:
            cat = cmd.get("category", "Other")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(cmd)

        category_order = ["Session", "Configuration", "Tools & Skills", "Info", "Exit"]
        for cat in category_order:
            if cat in categories:
                lines.append(("", f"\n━━ {cat} ━━\n"))
                for cmd in categories[cat]:
                    global_idx = self._filtered_commands.index(cmd)
                    prefix = "» " if global_idx == self._selected_index else "  "
                    cmd_name = cmd["name"]
                    desc = cmd.get("description", "")
                    aliases = cmd.get("aliases", ())
                    alias_str = f" ({', '.join(aliases)})" if aliases else ""
                    if global_idx == self._selected_index:
                        lines.append(
                            (
                                "fg:ansiyellow bold",
                                f"{prefix}/{cmd_name}{alias_str} - {desc}\n",
                            )
                        )
                    else:
                        lines.append(("", f"{prefix}/{cmd_name}{alias_str} - {desc}\n"))

        for cat, cmds in categories.items():
            if cat not in category_order:
                lines.append(("", f"\n━━ {cat} ━━\n"))
                for cmd in cmds:
                    global_idx = self._filtered_commands.index(cmd)
                    prefix = "» " if global_idx == self._selected_index else "  "
                    cmd_name = cmd["name"]
                    desc = cmd.get("description", "")
                    aliases = cmd.get("aliases", ())
                    alias_str = f" ({', '.join(aliases)})" if aliases else ""
                    if global_idx == self._selected_index:
                        lines.append(
                            (
                                "fg:ansiyellow bold",
                                f"{prefix}/{cmd_name}{alias_str} - {desc}\n",
                            )
                        )
                    else:
                        lines.append(("", f"{prefix}/{cmd_name}{alias_str} - {desc}\n"))

        if not lines:
            lines.append(("", "No commands found"))

        return lines

    @property
    def dialog(self) -> Dialog:
        """Get the dialog widget."""
        return self._dialog

    @property
    def key_bindings(self) -> KeyBindings:
        """Get the key bindings."""
        return self._kb

    def show(self):
        """Show and focus the palette."""
        self._query = ""
        self._selected_index = 0
        self._filter_commands()
        self._refresh_list()
        if self._search_area:
            self._search_area.buffer.text = ""

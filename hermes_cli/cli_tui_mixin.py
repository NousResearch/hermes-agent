"""prompt_toolkit TUI facade; behavior lives in topic siblings."""

from __future__ import annotations

import errno

import json

import os

import queue

import shutil

import string

import sys

import threading

import time

from agent.interrupt_compat import request_hard_interrupt

from hermes_cli.commands_completion import SlashCommandAutoSuggest, SlashCommandCompleter

from pathlib import Path

from prompt_toolkit.filters import Condition

from prompt_toolkit.history import FileHistory

from prompt_toolkit.key_binding import KeyBindings

from prompt_toolkit.layout import (
    ConditionalContainer,
    FormattedTextControl,
    HSplit,
    Layout,
    Window,
    WindowAlign)

from prompt_toolkit.layout.dimension import Dimension

from prompt_toolkit.layout.menus import CompletionsMenu

from prompt_toolkit.layout.processors import (
    ConditionalProcessor,
    PasswordProcessor,
    Processor,
    Transformation)

from prompt_toolkit.styles import Style as PTStyle

from prompt_toolkit.widgets import TextArea

from typing import Optional

_PANEL_RESERVED_BELOW = 6

_TYPING_CHARS = string.digits + string.ascii_letters + "-_.:/ "

_APPROVAL_CHOICE_LABELS = {
    "once": "Allow once",
    "session": "Allow for this session",
    "always": "Add to permanent allowlist",
    "deny": "Deny",
    "view": "Show full command"}

def _num_prefix(i: int) -> str:
    """Quick-select key label: 1-9 for items 1-9, 0 for the 10th, blank beyond."""
    return str(i + 1) if i < 9 else ("0" if i == 9 else " ")

def _term_rows() -> int:
    return shutil.get_terminal_size((100, 24)).lines

class _Panel:
    """Fragment accumulator for one bordered overlay panel (``(style, text)`` tuples)."""

    def __init__(self, border: str, box_width: int, title: str = "", title_style: str = ""):
        from cli import _append_blank_panel_line, _append_panel_line
        self.lines, self.border, self.width = [], border, box_width
        self._row, self._blank = _append_panel_line, _append_blank_panel_line
        if title:
            # Title inlined into the top rule: ``╭─ Title ───╮``.
            self.lines.append((border, "╭─ "))
            self.lines.append((title_style, title))
            self.lines.append((border, " " + ("─" * max(0, box_width - len(title) - 3)) + "╮\n"))
        else:
            self.lines.append((border, "╭" + ("─" * box_width) + "╮\n"))

    def row(self, style: str, text: str) -> None:
        self._row(self.lines, self.border, style, text, self.width)

    def blank(self) -> None:
        self._blank(self.lines, self.border, self.width)

    def close(self) -> list:
        self.lines.append((self.border, "╰" + ("─" * self.width) + "╯\n"))
        return self.lines

def _wrap_rows(wrap, items, width, indent) -> list[tuple[int, str]]:
    """``(index, wrapped_line)`` pairs so selected styling can be re-applied per row."""
    return [(i, w) for i, label in enumerate(items) for w in wrap(label, width, subsequent_indent=indent)]

from hermes_cli.cli_tui_panels import CLITuiPanelsMixin
from hermes_cli.cli_tui_chrome import CLITuiChromeMixin
from hermes_cli.cli_tui_handlers import CLITuiHandlersMixin
from hermes_cli.cli_tui_submission import CLITuiSubmissionMixin
from hermes_cli.cli_tui_construction import CLITuiConstructionMixin

class CLITuiMixin(CLITuiPanelsMixin, CLITuiChromeMixin, CLITuiHandlersMixin, CLITuiSubmissionMixin, CLITuiConstructionMixin):
    """prompt_toolkit TUI construction, key-binding handlers, and overlay display fragments."""

    _TUI_MODAL_HINTS = (
            ("_sudo_state", "_sudo_deadline", '  password hidden · Enter to skip'),
            ("_secret_state", "_secret_deadline", '  secret hidden · Enter to skip'),
            ("_approval_state", "_approval_deadline", '  ↑/↓ to select, Enter to confirm'),
            ("_slash_confirm_state", "_slash_confirm_deadline", '  type 1/2/3, or ↑/↓ to select, Enter to confirm'),
        )

    def _get_extra_tui_widgets(self) -> list:
        """Extension hook: wrapper CLIs return widgets inserted between the spacer and status bar."""
        return []

    def _register_extra_tui_keybindings(self, kb, *, input_area) -> None:
        """Extension hook: wrapper CLIs add bindings to ``kb`` (``input_area`` is the main TextArea)."""

    def _build_tui_layout_children(
        self,
        *,
        sudo_widget,
        secret_widget,
        approval_widget,
        slash_confirm_widget=None,
        clarify_widget,
        model_picker_widget=None,
        command_palette_widget=None,
        spinner_widget=None,
        spacer,
        status_bar,
        input_rule_top,
        image_bar,
        input_area,
        input_rule_bot,
        voice_status_bar,
        completions_menu) -> list:
        """Ordered children of the root ``HSplit``; override only for full control over ordering
        (wrappers normally override ``_get_extra_tui_widgets`` instead)."""
        ordered = [
            Window(height=0),
            sudo_widget,
            secret_widget,
            approval_widget,
            slash_confirm_widget,
            clarify_widget,
            model_picker_widget,
            command_palette_widget,
            spinner_widget,
            spacer,
            *self._get_extra_tui_widgets(),
            getattr(self, "_pet_widget", None),
            getattr(self, "_stash_panel_widget", None),
            getattr(self, "_subagent_dock_widget", None),
            status_bar,
            input_rule_top,
            image_bar,
            input_area,
            input_rule_bot,
            voice_status_bar,
            completions_menu]
        return [item for item in ordered if item is not None]

    def _tui_suppress_closed_loop_errors(self, loop, context):
        exc = context.get("exception")
        if isinstance(exc, RuntimeError) and "Event loop is closed" in str(exc):
            return
        if isinstance(exc, KeyError) and "is not registered" in str(exc):
            return  # selector registration failures (#6393)
        if isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.EIO:
            return  # broken stdout on interrupt (#13710)
        loop.default_exception_handler(context)

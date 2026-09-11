"""prompt_toolkit TUI topic sibling; cli.py-internal symbols remain lazy."""

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


class CLITuiChromeMixin:
    def _tui_input_rule_height(self, position: str, width: Optional[int] = None) -> int:
        """Visible height for the top/bottom input separator rules."""
        if position not in {"top", "bottom"}:
            raise ValueError(f"Unknown input rule position: {position}")
        if getattr(self, "_status_bar_suppressed_after_resize", False):
            return 0
        if position == "top":
            return 1
        return 0 if self._use_minimal_tui_chrome(width=width) else 1

    def _get_tui_prompt_symbols(self) -> tuple[str, str]:
        """Return ``(normal_prompt, state_suffix)`` for the active skin.

        ``state_suffix`` is what special states (sudo/secret/approval/agent) render after their
        leading icon. A non-default profile name is prepended (``coder ❯``).
        """
        try:
            from hermes_cli.skin_engine import get_active_prompt_symbol
            symbol = get_active_prompt_symbol("❯ ")
        except Exception:
            symbol = "❯ "
        symbol = (symbol or "❯ ").rstrip() + " "
        try:
            from hermes_cli.profiles import get_active_profile_name
            profile = get_active_profile_name()
            if profile not in {"default", "custom"}:
                symbol = f"{profile} {symbol}"
        except Exception:
            pass
        stripped = symbol.rstrip()
        if not stripped:
            return "❯ ", "❯ "
        parts = stripped.split()
        candidate = parts[-1] if parts else ""
        if any(ch in candidate for ch in ("❯", ">", "$", "#", "›", "»", "→")):
            return symbol, candidate.rstrip() + " "
        # Icon-only custom prompts should still remain visible in special states.
        return symbol, symbol

    def _audio_level_bar(self) -> str:
        """One-char audio level indicator from the recorder's current RMS."""
        rec = getattr(self, "_voice_recorder", None)
        if rec is None:
            return ""
        # RMS 0-32767 → index 0-7; typical speech is 500-5000, display caps at ~8000.
        return " ▁▂▃▄▅▆▇"[min(rec.current_rms, 8000) * 7 // 8000]

    def _get_tui_prompt_fragments(self):
        """prompt_toolkit fragments for the current interactive state."""
        symbol, state_suffix = self._get_tui_prompt_symbols()
        compact = self._use_minimal_tui_chrome(width=self._get_tui_terminal_width())

        def _state_fragment(style: str, icon: str, extra: str = ""):
            if compact:
                text = icon
                if extra:
                    text = f"{text} {extra.strip()}".rstrip()
                return [(style, text + " ")]
            if extra:
                return [(style, f"{icon} {extra} {state_suffix}")]
            return [(style, f"{icon} {state_suffix}")]

        if self._voice_recording:
            return _state_fragment("class:voice-recording", "●", self._audio_level_bar())
        if self._voice_processing:
            return _state_fragment("class:voice-processing", "◉")
        if self._sudo_state:
            return _state_fragment("class:sudo-prompt", "🔐")
        if self._secret_state:
            return _state_fragment("class:sudo-prompt", "🔑")
        if self._approval_state or getattr(self, "_slash_confirm_state", None):
            return _state_fragment("class:prompt-working", "⚠")
        if self._clarify_freetext:
            return _state_fragment("class:clarify-selected", "✎")
        if self._clarify_state:
            return _state_fragment("class:prompt-working", "?")
        if self._command_running:
            return _state_fragment("class:prompt-working", self._command_spinner_frame())
        if self._agent_running:
            return _state_fragment("class:prompt-working", "⚕")
        if self._voice_mode:
            return _state_fragment("class:voice-prompt", "🎤")
        return [("class:prompt", symbol)]

    def _get_tui_prompt_text(self) -> str:
        """Visible prompt text for width calculations."""
        return "".join(text for _, text in self._get_tui_prompt_fragments())

    def _build_tui_style_dict(self) -> dict[str, str]:
        """Layer the active skin's prompt_toolkit colors over the base TUI style.

        On a light terminal, hex tokens in each style string are rewritten through the light-mode
        remap so the chrome stays readable on cream Terminal.app backgrounds. CRITICAL: a style
        that paints its own ``bg:`` (status bar, completion menu) is left alone — its fg was tuned
        for that dark bg and remapping would give dark-on-dark; the terminal's mode is irrelevant.
        """
        from cli import _detect_light_mode, _maybe_remap_for_light_mode
        style_dict = dict(getattr(self, "_tui_style_base", {}) or {})
        try:
            from hermes_cli.skin_engine import get_prompt_toolkit_style_overrides
            style_dict.update(get_prompt_toolkit_style_overrides())
        except Exception:
            pass
        try:
            if _detect_light_mode():
                def _remap_value(v: str) -> str:
                    if not v:
                        return v
                    tokens = v.split()
                    if any(t.startswith("bg:") for t in tokens):
                        return v
                    return " ".join(_maybe_remap_for_light_mode(t) if t.startswith("#") else t for t in tokens)
                style_dict = {k: _remap_value(v or "") for k, v in style_dict.items()}
        except Exception:
            pass
        return style_dict

    def _apply_tui_skin_style(self) -> bool:
        """Refresh prompt_toolkit styling for a running interactive TUI."""
        if not getattr(self, "_app", None) or not getattr(self, "_tui_style_base", None):
            return False
        self._app.style = PTStyle.from_dict(self._build_tui_style_dict())
        self._invalidate(min_interval=0.0)
        return True

    def _tui_hint_text(self):
        for state_attr, deadline_attr, hint in self._TUI_MODAL_HINTS:
            if getattr(self, state_attr):
                if state_attr == "_sudo_state" and (self._sudo_state.get("vault_save") or {}).get("step") == "identifier":
                    hint = '  shown as you type · Enter to continue'
                remaining = max(0, int(getattr(self, deadline_attr) - time.monotonic()))
                return [('class:hint', hint), ('class:clarify-countdown', f'  ({remaining}s)')]
        if self._clarify_state:
            # None deadline = unlimited wait → hide the countdown entirely.
            if self._clarify_deadline is None:
                countdown = ''
            else:
                countdown = f'  ({max(0, int(self._clarify_deadline - time.monotonic()))}s)'
            if self._clarify_freetext:
                hint = '  type your answer and press Enter'
            elif self._clarify_state.get("questions"):
                hint = '  ↑/↓ to select, Enter to lock, Tab next question'
            else:
                hint = '  ↑/↓ to select, Enter to confirm'
            return [('class:hint', hint), ('class:clarify-countdown', countdown)]
        if self._command_running:
            frame = self._command_spinner_frame()
            if self._command_blocks_input:
                detail = "input temporarily disabled"
            else:
                detail = "input stays active; Enter queues"
            return [('class:hint', f'  {frame} command in progress · {detail}')]
        return []

    def _tui_placeholder_text(self):
        if self._voice_recording:
            return f"recording... {self._voice_record_key_label()} to stop, Ctrl+C to cancel"
        if self._voice_processing:
            return "transcribing..."
        if self._sudo_state:
            if (self._sudo_state.get("vault_save") or {}).get("step") == "identifier":
                return "type your email / username, Enter to continue · ESC to skip"
            return "type password (hidden), Enter to submit · ESC to skip"
        if self._secret_state:
            return "type secret (hidden), Enter to submit · ESC to skip"
        if self._approval_state:
            return ""
        if self._slash_confirm_state:
            return "type 1/2/3, or use ↑/↓ then Enter"
        if self._clarify_freetext:
            return "type your answer here and press Enter"
        if self._clarify_state:
            return ""
        if self._command_running:
            return f"{self._command_spinner_frame()} {self._command_status or 'Processing command...'}"
        if self._agent_running:
            return "msg=interrupt · /queue · /bg · /steer · Ctrl+C cancel"
        if self._voice_mode:
            return f"type or {self._voice_record_key_label()} to record"
        # Advertise a parked draft so the stash can never be silently forgotten.
        try:
            _stash_hint = self._prompt_stash.placeholder_hint()
        except Exception:
            _stash_hint = ""
        if _stash_hint:
            return _stash_hint
        # Idle + empty composer: a task-oriented example chosen once per session
        # (self._composer_placeholder) so it stays stable while being read, not flickering.
        return getattr(self, "_composer_placeholder", "") or ""

    def _get_stash_panel_display_fragments(self):
        try:
            _stash = self._prompt_stash
            return self._render_stash_panel(
                _stash.panel_rows(), _stash.panel_cursor, self._get_tui_terminal_width())
        except Exception:
            return []

    def _tui_image_bar_fragments(self):
        from cli import _format_image_attachment_badges
        if not self._attached_images:
            return []
        badges = _format_image_attachment_badges(self._attached_images, self._image_counter)
        return [("class:image-badge", f" {badges} ")]

    def _tui_voice_status_fragments(self):
        return self._get_voice_status_fragments()

    def _tui_spinner_text(self):
        spinner_line = self._render_spinner_text()
        return [('class:hint', spinner_line)] if spinner_line else []

    def _tui_spinner_height(self):
        return self._spinner_widget_height()

    def _tui_hint_height(self):
        if (
            self._sudo_state or self._secret_state or self._approval_state
            or self._slash_confirm_state or self._clarify_state or self._command_running):
            return 1
        # Keep a spacer while the agent runs on roomy terminals; reclaim the row on narrow screens.
        return self._agent_spacer_height()

    def _tui_set_base_style(self):
        """Populate ``self._tui_style_base`` (skin-aware defaults the style dict is built from)."""
        self._tui_style_base = {
            # Empty input/prompt styles inherit the terminal's own fg/bg so typed text is readable
            # in both light and dark schemes (a hardcoded near-white was invisible on light).
            'input-area': '',
            'placeholder': '#888888 italic',
            'prompt': '',
            'prompt-working': '#888888 italic',
            'hint': '#888888 italic',
            'status-bar': 'bg:#1a1a2e #C0C0C0',
            'status-bar-strong': 'bg:#1a1a2e #FFD700 bold',
            'status-bar-dim': 'bg:#1a1a2e #8B8682',
            'status-bar-good': 'bg:#1a1a2e #8FBC8F bold',
            'status-bar-warn': 'bg:#1a1a2e #FFD700 bold',
            'status-bar-bad': 'bg:#1a1a2e #FF8C00 bold',
            'status-bar-critical': 'bg:#1a1a2e #FF6B6B bold',
            'status-bar-yolo': 'bg:#1a1a2e #FF4444 bold',
            'status-bar-session-title': 'bg:#FFD700 #1a1a2e bold',
            'input-rule': '#CD7F32',
            'image-badge': '#87CEEB bold',
            'completion-menu': 'bg:#1a1a2e #FFF8DC',
            'completion-menu.completion': 'bg:#1a1a2e #FFF8DC',
            'completion-menu.completion.current': 'bg:#333355 #FFD700',
            'completion-menu.meta.completion': 'bg:#1a1a2e #888888',
            'completion-menu.meta.completion.current': 'bg:#333355 #FFBF00',
            'clarify-border': '#CD7F32',
            'clarify-title': '#FFD700 bold',
            'clarify-question': '#FFF8DC bold',
            'clarify-choice': '#AAAAAA',
            'clarify-selected': '#FFD700 bold',
            'clarify-active-other': '#FFD700 italic',
            'clarify-answer': '#98FB98',
            'clarify-countdown': '#CD7F32',
            'sudo-prompt': '#FF6B6B bold',
            'sudo-border': '#CD7F32',
            'sudo-title': '#FF6B6B bold',
            'sudo-text': '#FFF8DC',
            'approval-border': '#CD7F32',
            'approval-title': '#FF8C00 bold',
            'approval-desc': '#FFF8DC bold',
            'approval-cmd': '#AAAAAA italic',
            'approval-choice': '#AAAAAA',
            'approval-selected': '#FFD700 bold',
            'voice-prompt': '#87CEEB',
            'voice-recording': '#FF4444 bold',
            'voice-processing': '#FFA500 italic',
            'voice-status': 'bg:#1a1a2e #87CEEB',
            'voice-status-recording': 'bg:#1a1a2e #FF4444 bold'}



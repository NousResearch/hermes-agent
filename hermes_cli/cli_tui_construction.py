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


class CLITuiConstructionMixin:
    def _tui_spinner_loop(self):
        while not self._should_exit:
            if not self._app:
                time.sleep(0.1)
                continue
            monitor = getattr(self, "_subagent_monitor", None)
            if monitor is not None:
                monitor.tick()
            if self._command_running:
                self._invalidate(min_interval=0.1)
                time.sleep(0.1)
            else:
                # Never repaint the idle prompt on a timer: in non-full-screen mode background
                # redraws fight tmux/Ghostty/cmux viewport restoration after focus changes and
                # visually move the input area. Input/agent events invalidate explicitly.
                time.sleep(0.2)

    def _tui_approval_up(self, event):
        st = self._approval_state
        if st:
            st["selected"] = max(0, st["selected"] - 1)
            event.app.invalidate()

    def _tui_wake_startup(self):
        from cli import logger
        try:
            self._maybe_start_wake_word()
        except Exception as e:
            logger.debug("wake-word startup skipped: %s", e)

    def _tui_hint_height(self):
        if (
            self._sudo_state or self._secret_state or self._approval_state
            or self._slash_confirm_state or self._clarify_state or self._command_running):
            return 1
        # Keep a spacer while the agent runs on roomy terminals; reclaim the row on narrow screens.
        return self._agent_spacer_height()

    def _tui_init_run_state(self):
        """Reset the per-run REPL state (queues, modal states, voice state, config watcher)."""
        self._agent_running = False
        self._pending_input = queue.Queue()     # normal input (commands + new queries)
        self._interrupt_queue = queue.Queue()   # messages typed while the agent is running
        # Seeded -q handoff: main() can't put directly into _pending_input (this reinit would
        # discard it), so the seeded first message rides in on an attribute and is enqueued here.
        _seed_msg = getattr(self, "_seeded_first_message", None)
        if _seed_msg is not None:
            self._seeded_first_message = None
            self._pending_input.put(_seed_msg)
        # See constructor note; mirrored for the run() path that skips the earlier __init__ branch.
        self._last_turn_interrupted = False
        self._should_exit = False
        self._last_ctrl_c_time = 0  # double Ctrl+C force-exit tracking

        # Plugins get a CLI reference so they can inject messages.
        from hermes_cli.plugins import get_plugin_manager
        get_plugin_manager()._cli_ref = self

        # Config file watcher — detect mcp_servers changes and auto-reload.
        from hermes_cli.config import get_config_path as _get_config_path
        _cfg_path = _get_config_path()
        self._config_mtime: float = _cfg_path.stat().st_mtime if _cfg_path.exists() else 0.0
        self._config_mcp_servers: dict = self.config.get("mcp_servers") or {}
        self._last_config_check: float = 0.0  # monotonic time of last check

        # Modal overlay states: each is a dict (with a response_queue) while active, else None.
        # The prompt_toolkit UI switches to the matching selection/input mode.
        self._clarify_state = None
        self._clarify_freetext = False  # True when the user chose "Other" and is typing
        self._clarify_deadline = 0      # monotonic timeout
        self._sudo_state = None
        self._sudo_deadline = 0
        self._modal_input_snapshot = None
        self._approval_state = None
        self._approval_deadline = 0
        self._approval_lock = threading.Lock()  # serialize concurrent approval prompts (delegation race)
        # Destructive slash-command confirmations (/new, /clear, /undo) are answered through the
        # composer, not raw input(), so the labels stay visible and Enter can't EOF the app.
        self._slash_confirm_state = None
        self._slash_confirm_deadline = 0
        self._command_running = False
        self._command_blocks_input = False
        self._command_status = ""
        self._secret_state = None       # skill-setup secret capture
        self._secret_deadline = 0

        self._attached_images: list[Path] = []  # clipboard image attachments
        self._image_counter = 0

        # Voice mode state (protected by _voice_lock for cross-thread access).
        self._voice_lock = threading.Lock()
        self._voice_mode = False
        self._voice_tts = False
        self._voice_recorder = None     # AudioRecorder (lazy init)
        self._voice_recording = False
        self._voice_processing = False  # STT in progress
        self._voice_continuous = False  # auto-restart after the agent responds
        self._voice_tts_done = threading.Event()  # TTS playback finished
        self._voice_tts_done.set()  # initially "done" (no TTS pending)
        self._voice_tts_stop = None  # active streaming pipeline's stop event
        self._voice_barge_capture = threading.Event()  # barge monitor is capturing the interruption
        self._voice_last_tts_text = ""  # most recently spoken TTS text (echo guard, #75780)
        self._voice_barge_phase = None  # "generation" or "playback" phase of the last barge trip

        if os.environ.get("HERMES_DEFER_AGENT_STARTUP") != "1":
            self._install_tool_callbacks()
            self._ensure_tirith_security()

    def _tui_build_key_bindings(self):
        """Build the prompt_toolkit KeyBindings for the REPL input area.

        Registration ORDER matters: for the same key, prompt_toolkit picks the last matching
        binding, so the generic handlers (Tab, history Up/Down) are registered before or after
        their filtered modal overrides deliberately.
        """
        from cli import (
            CLI_CONFIG,
            _bind_prompt_submit_keys,
            _cli_multiline_shortcuts_enabled,
            _preserve_ctrl_enter_newline)
        from prompt_toolkit.keys import Keys
        kb = KeyBindings()
        _multiline_shortcuts_enabled = _cli_multiline_shortcuts_enabled(self.config or CLI_CONFIG)
        self._tui_multiline_shortcuts = _multiline_shortcuts_enabled

        kb.add(Keys.Ignore, eager=True)(self._tui_handle_ignored_terminal_sequence)
        _bind_prompt_submit_keys(
            kb, self._tui_handle_enter, multiline_shortcuts_enabled=_multiline_shortcuts_enabled)
        kb.add('escape', 'enter')(self._tui_insert_newline)
        # Ctrl+J inserts a newline (Claude Code / Codex / OpenCode). Windows Terminal delivers
        # Ctrl+Enter as the same c-j code. display.cli_multiline_shortcuts: false restores legacy
        # c-j submit on unusual POSIX PTYs where Enter is LF.
        if _multiline_shortcuts_enabled or _preserve_ctrl_enter_newline():
            kb.add('c-j')(self._tui_insert_newline)

        self._tui_bind_editor_and_stash(kb)
        self._tui_bind_overlay_navigation(kb)

        # History: the TextArea is multiline so Up/Down alone only move the cursor;
        # Buffer.auto_up/auto_down browse history when on the first/last line.
        _normal_input = Condition(
            lambda: not self._clarify_state and not self._approval_state and not self._slash_confirm_state
            and not self._sudo_state and not self._secret_state and not self._model_picker_state
            and not self._command_palette_state)
        kb.add('up', filter=_normal_input)(self._tui_history_up)
        kb.add('down', filter=_normal_input)(self._tui_history_down)
        kb.add('c-l')(self._tui_handle_ctrl_l)
        kb.add('c-c')(self._tui_handle_ctrl_c)
        # No Ctrl+Shift+C binding: terminal emulators intercept it before stdin, and
        # prompt_toolkit's key parser doesn't recognise 'c-S-c' anyway (#19884/#19895).
        kb.add('c-q')(self._tui_handle_ctrl_q)
        kb.add('c-d')(self._tui_handle_ctrl_d)
        _modal_prompt_active = Condition(
            lambda: bool(self._secret_state or self._sudo_state or self._slash_confirm_state))
        kb.add('escape', filter=_modal_prompt_active, eager=True)(self._tui_handle_escape_modal)
        kb.add('escape', 'escape', filter=~_modal_prompt_active)(self._tui_handle_double_escape)
        kb.add('c-z')(self._tui_handle_ctrl_z)

        kb.add(*self._tui_voice_record_key_sequence())(self._tui_handle_voice_record)
        kb.add(Keys.BracketedPaste, eager=True)(self._tui_handle_paste)
        kb.add('c-v')(self._tui_handle_ctrl_v)
        kb.add('escape', 'v')(self._tui_handle_alt_v)
        from hermes_cli.cli_subagent_monitor import modal_prompt_active, open_monitor, toggle_dock
        for key in ('c-t', 'f6'):
            kb.add(key, filter=Condition(lambda: not modal_prompt_active(self)))(
                lambda event: open_monitor(self))
        kb.add('f7', filter=Condition(lambda: not modal_prompt_active(self)))(
            lambda event: toggle_dock(self))
        return kb

    def _tui_bind_editor_and_stash(self, kb) -> None:
        # VSCode/Cursor bind Ctrl+G to "Find Next" so it never reaches the terminal; Alt+G is
        # unbound there and arrives as ('escape', 'g') — register it as a fallback.
        _editor_filter = Condition(
            lambda: not self._clarify_state and not self._approval_state
            and not self._sudo_state and not self._secret_state)
        kb.add('c-g', filter=_editor_filter)(
            kb.add('escape', 'g', filter=_editor_filter)(self._tui_handle_open_in_editor))
        # Ctrl+S prompt stash: park a draft, send something else, bring it back. Suppressed while
        # a modal prompt owns the composer so Ctrl+S can't stash a password.
        _stash_filter = Condition(
            lambda: not self._clarify_state and not self._approval_state and not self._sudo_state
            and not self._secret_state and not self._slash_confirm_state and not self._model_picker_state
        )
        _stash_panel_filter = Condition(lambda: self._prompt_stash.panel_open and bool(len(self._prompt_stash)))
        kb.add('c-s', filter=_stash_filter)(self._tui_handle_prompt_stash)
        kb.add('up', filter=_stash_panel_filter, eager=True)(self._tui_handle_stash_panel_up)
        kb.add('down', filter=_stash_panel_filter, eager=True)(self._tui_handle_stash_panel_down)
        kb.add('enter', filter=_stash_panel_filter, eager=True)(self._tui_handle_stash_panel_restore)
        kb.add('d', filter=_stash_panel_filter, eager=True)(
            kb.add('D', filter=_stash_panel_filter, eager=True)(self._tui_handle_stash_panel_delete)
        )
        kb.add('escape', filter=_stash_panel_filter, eager=True)(self._tui_handle_stash_panel_close)
        kb.add('tab', eager=True)(self._tui_handle_tab)

    def _tui_bind_overlay_navigation(self, kb) -> None:
        """Clarify / approval / slash-confirm / model picker / command palette navigation keys."""
        _clarify_nav = Condition(lambda: bool(self._clarify_state) and not self._clarify_freetext)
        _clarify_batch = Condition(
            lambda: bool(self._clarify_state) and bool(self._clarify_state.get("questions"))
            and not self._clarify_freetext)
        kb.add('up', filter=_clarify_nav)(self._tui_clarify_up)
        kb.add('down', filter=_clarify_nav)(self._tui_clarify_down)
        # Multi-select: Space toggles the checkbox under the cursor.
        kb.add('space', filter=Condition(
            lambda: bool(self._clarify_state) and not self._clarify_freetext
            and self._clarify_state.get("multi_select")
        ))(self._tui_clarify_toggle)
        # Batch clarify: Tab / Shift-Tab cycle the active question (any-order answering; moving
        # onto an answered question lets the user re-answer it). Registered after the generic
        # tab handler so this filtered binding wins while the batch panel is open.
        kb.add('tab', filter=_clarify_batch, eager=True)(self._tui_clarify_batch_tab)
        kb.add('s-tab', filter=_clarify_batch, eager=True)(self._tui_clarify_batch_backtab)
        # Number keys: 1-9 select items 0-8, 0 selects item 9 (10th).
        for _num in range(10):
            _idx = 9 if _num == 0 else _num - 1
            kb.add(str(_num), filter=_clarify_nav)(self._tui_make_clarify_number_handler(_idx))

        _approval = Condition(lambda: bool(self._approval_state))
        _slash_confirm = Condition(lambda: bool(self._slash_confirm_state))
        _picker = Condition(lambda: bool(self._model_picker_state))
        kb.add('up', filter=_approval)(self._tui_approval_up)
        kb.add('down', filter=_approval)(self._tui_approval_down)
        kb.add('up', filter=_slash_confirm)(self._tui_slash_confirm_up)
        kb.add('down', filter=_slash_confirm)(self._tui_slash_confirm_down)
        kb.add('up', filter=_picker)(self._tui_model_picker_up)
        kb.add('down', filter=_picker)(self._tui_model_picker_down)

        def _model_picker_typing_active() -> bool:
            # Type-to-filter is only live on the model stage (concrete list).
            st = self._model_picker_state
            return bool(st) and st.get("stage") == "model"

        _picker_typing = Condition(_model_picker_typing_active)
        for _ch in _TYPING_CHARS:
            kb.add(_ch, filter=_picker_typing)(self._tui_make_model_filter_char_handler(_ch))
        kb.add('backspace', filter=_picker_typing)(self._tui_model_picker_filter_backspace)
        kb.add('escape', filter=_picker, eager=True)(self._tui_model_picker_escape)

        _palette = Condition(lambda: bool(self._command_palette_state))
        kb.add('c-p', filter=Condition(
            lambda: not self._command_palette_state and not self._model_picker_state and not self._clarify_state
            and not self._approval_state and not self._slash_confirm_state and not self._sudo_state
            and not self._secret_state
        ))(self._tui_open_command_palette)
        kb.add('up', filter=_palette)(self._tui_command_palette_up)
        kb.add('down', filter=_palette)(self._tui_command_palette_down)
        kb.add('enter', filter=_palette)(self._tui_command_palette_enter)
        kb.add('backspace', filter=_palette)(self._tui_command_palette_backspace)
        kb.add('escape', filter=_palette, eager=True)(self._tui_command_palette_escape)
        for _pch in _TYPING_CHARS:
            kb.add(_pch, filter=_palette)(self._tui_make_palette_char_handler(_pch))

        for _num in range(10):
            _idx = 9 if _num == 0 else _num - 1
            kb.add(str(_num), filter=_approval)(self._tui_make_approval_number_handler(_idx))
        for _num in range(10):
            _idx = 9 if _num == 0 else _num - 1
            kb.add(str(_num), filter=_slash_confirm)(self._tui_make_slash_confirm_number_handler(_idx))

    def _tui_voice_record_key_sequence(self) -> tuple:
        """Resolve the push-to-talk key (voice.record_key, default Ctrl+B) to a prompt_toolkit
        key sequence and cache the UI label.

        Config spellings (ctrl/control/alt/option/opt) are normalized to c-x / a-x so the same
        value binds identically in TUI and CLI. super/win/windows silently fall back to the
        default (prompt_toolkit has no super modifier) — warn so users notice the split. The
        label cache uses the same ``_raw_key`` that drives the binding, so status/placeholder/
        recording-hint renders can never drift from the live key even if the config is edited
        mid-session.
        """
        from cli import logger
        # Voice push-to-talk key: configurable via config.yaml (voice.record_key) Default: Ctrl+B (avoids
        # conflict with Ctrl+R readline reverse-search). Config spellings (ctrl/control/alt/option/opt) are
        # normalized to prompt_toolkit's c-x / a-x format via
        # ``normalize_voice_record_key_for_prompt_toolkit`` so the same config value binds identically in
        # the TUI and CLI (Copilot round-9 review on #19835). ``super``/``win``/``windows`` configs silently
        # fall back to the default here since prompt_toolkit has no super modifier — log a warning so users
        # notice the TUI/CLI split instead of a silent mismatch (round-11).
        _raw_key: object = "ctrl+b"
        try:
            from hermes_cli.config import load_config
            from hermes_cli.voice import (
                normalize_voice_record_key_for_prompt_toolkit,
                pt_key_to_sequence,
                voice_record_key_from_config)
            _raw_key = voice_record_key_from_config(load_config())
            _voice_key = normalize_voice_record_key_for_prompt_toolkit(_raw_key)
            if (
                isinstance(_raw_key, str)
                and _raw_key.strip().lower().split("+", 1)[0].strip() in {"super", "win", "windows"}
                and _voice_key == "c-b"):
                logger.warning(
                    "voice.record_key %r uses a TUI-only modifier (super/win); "
                    "CLI fell back to Ctrl+B. Use ctrl+<key> or alt+<key> for "
                    "cross-runtime parity.",
                    _raw_key)
        except Exception:
            _voice_key = "c-b"
        # Cache the UI label here — same ``_raw_key`` that drives the prompt_toolkit binding below. Every
        # status / placeholder / recording-hint render reads this cached value so display can never drift
        # from the live keybinding even if the user edits voice.record_key mid-session (Copilot round-13 on
        # #19835).
        self.set_voice_record_key_cache(_raw_key)
        return pt_key_to_sequence(_voice_key)

    def _tui_overlay_widget(self, fragments_fn, state_attr: str):
        """Wrapped, auto-sized panel shown while ``self.<state_attr>`` is not None."""
        return ConditionalContainer(
            Window(FormattedTextControl(fragments_fn), wrap_lines=True),
            filter=Condition(lambda: getattr(self, state_attr) is not None))

    def _tui_build_layout(self, kb):
        """Build the TUI widgets, Layout and Style; registers wrapper keybindings on ``kb``."""
        cli_ref = self
        from hermes_cli.cli_subagent_monitor import install_dock
        install_dock(self)
        input_area = self._tui_build_input_area()
        spinner_widget = Window(
            content=FormattedTextControl(self._tui_spinner_text),
            height=self._tui_spinner_height,
            wrap_lines=True)
        # Petdex mascot — right-aligned Kitty placeholder or half-block sprite above the prompt;
        # height 0 when no pet is enabled. The animation thread queues virtual Kitty frames;
        # after_render writes them out-of-band while prompt_toolkit owns the placeholder grid.
        self._pet_widget = Window(
            content=FormattedTextControl(self._pet_fragments),
            height=self._pet_widget_height,
            align=WindowAlign.RIGHT)
        # Hint line above the input: only for interactive prompts that need extra instructions
        # (sudo countdown, approval navigation, clarify); the agent-running hint is the placeholder.
        spacer = Window(content=FormattedTextControl(self._tui_hint_text), height=self._tui_hint_height)
        clarify_widget = self._tui_overlay_widget(self._get_clarify_display_fragments, "_clarify_state")
        sudo_widget = self._tui_overlay_widget(self._get_sudo_display_fragments, "_sudo_state")
        secret_widget = self._tui_overlay_widget(self._get_secret_display_fragments, "_secret_state")
        approval_widget = self._tui_overlay_widget(self._get_approval_display_fragments, "_approval_state")
        slash_confirm_widget = self._tui_overlay_widget(
            self._get_slash_confirm_display_fragments, "_slash_confirm_state")
        model_picker_widget = self._tui_overlay_widget(
            self._get_model_picker_display_fragments, "_model_picker_state")
        command_palette_widget = self._tui_overlay_widget(
            self._get_command_palette_display_fragments, "_command_palette_state")
        # Rules above/below the input; narrow terminals hide the bottom one to recover a row.
        input_rule_top = Window(
            char='─', height=lambda: cli_ref._tui_input_rule_height("top"), style='class:input-rule',
        )
        input_rule_bot = Window(
            char='─', height=lambda: cli_ref._tui_input_rule_height("bottom"), style='class:input-rule',
        )
        image_bar = Window(
            content=FormattedTextControl(self._tui_image_bar_fragments),
            height=Condition(lambda: bool(cli_ref._attached_images)))
        voice_status_bar = ConditionalContainer(
            Window(FormattedTextControl(self._tui_voice_status_fragments), height=1),
            filter=Condition(lambda: cli_ref._voice_mode))
        status_bar = ConditionalContainer(
            Window(
                content=FormattedTextControl(lambda: cli_ref._get_status_bar_fragments()),
                height=1,
                # wrap_lines=False: fragments overflowing the width must never wrap onto a second
                # row (looked like a duplicated status bar on long SSH sessions with stale
                # shutil sizes). _get_status_bar_fragments reads prompt_toolkit's own width, so
                # this is the belt-and-suspenders guard.
                wrap_lines=False),
            filter=Condition(
                lambda: cli_ref._status_bar_visible
                and not getattr(cli_ref, "_status_bar_suppressed_after_resize", False)))
        # Stash browse panel — just above the status bar, Ctrl+S on an empty composer with 2+ drafts.
        self._stash_panel_widget = ConditionalContainer(
            Window(FormattedTextControl(self._get_stash_panel_display_fragments), wrap_lines=False),
            filter=Condition(lambda: cli_ref._prompt_stash.panel_open and bool(len(cli_ref._prompt_stash))),
        )
        self._register_extra_tui_keybindings(kb, input_area=input_area)
        layout = Layout(HSplit(self._build_tui_layout_children(
            sudo_widget=sudo_widget,
            secret_widget=secret_widget,
            approval_widget=approval_widget,
            slash_confirm_widget=slash_confirm_widget,
            clarify_widget=clarify_widget,
            model_picker_widget=model_picker_widget,
            command_palette_widget=command_palette_widget,
            spinner_widget=spinner_widget,
            spacer=spacer,
            status_bar=status_bar,
            input_rule_top=input_rule_top,
            image_bar=image_bar,
            input_area=input_area,
            input_rule_bot=input_rule_bot,
            voice_status_bar=voice_status_bar,
            completions_menu=CompletionsMenu(max_height=12, scroll_offset=1))))
        self._tui_set_base_style()
        return layout, PTStyle.from_dict(self._build_tui_style_dict())

    def _tui_build_input_area(self):
        """Multi-line prompt TextArea with slash completion, paste-collapse tracking and
        placeholder/password processors."""
        from cli import _estimate_tui_input_height, get_skill_bundles, get_skill_commands
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.completion import ThreadedCompleter
        cli_ref = self

        def get_prompt():
            return cli_ref._get_tui_prompt_fragments()

        _completer = SlashCommandCompleter(
            skill_commands_provider=lambda: get_skill_commands(),
            command_filter=cli_ref._command_available,
            skill_bundles_provider=lambda: get_skill_bundles())
        input_area = TextArea(
            height=Dimension(min=1, max=8, preferred=1),
            prompt=get_prompt,
            style='class:input-area',
            multiline=True,
            wrap_lines=True,
            read_only=Condition(lambda: bool(cli_ref._command_blocks_input)),
            history=FileHistory(str(self._history_file)),
            # The completer does blocking work (fuzzy @-file indexing shells out to rg/fd with a
            # 2s timeout; path completion hits os.listdir/stat), so complete_while_typing inline
            # would stall the render loop per keystroke (WSL2/slow FS). ThreadedCompleter moves
            # it off the UI event loop.
            completer=ThreadedCompleter(_completer),
            complete_while_typing=True,
            auto_suggest=SlashCommandAutoSuggest(history_suggest=AutoSuggestFromHistory(), completer=_completer),
        )
        # Keep prompt_toolkit on its simple tempfile path: buffer.tempfile = "prompt.md" takes
        # the complex-tempfile branch that re-mkdir()s the mkdtemp() dir and raises EEXIST.
        input_area.buffer.tempfile_suffix = '.md'

        def _input_height():
            # Accounts for explicit newlines AND visual wrapping so the area fits its content.
            try:
                from prompt_toolkit.application import get_app
                doc = input_area.buffer.document
                try:
                    terminal_columns = get_app().output.get_size().columns
                except Exception:
                    terminal_columns = shutil.get_terminal_size((80, 24)).columns
                return _estimate_tui_input_height(doc.lines, self._get_tui_prompt_text(), terminal_columns)
            except Exception:
                return 1

        input_area.window.height = _input_height
        # Paste collapsing state (large pastes are saved to a file and replaced by a placeholder).
        self._tui_paste_counter = 0
        self._tui_prev_text_len = 0
        self._tui_prev_newline_count = 0
        self._tui_paste_just_collapsed = False
        self._skip_paste_collapse = False
        input_area.buffer.on_text_changed += self._tui_on_text_changed
        # Mask input with '*' while a sudo/secret prompt is active.
        input_area.control.input_processors.append(ConditionalProcessor(
            PasswordProcessor(),
            filter=Condition(lambda: (bool(cli_ref._sudo_state)
                                      and (cli_ref._sudo_state.get("vault_save") or {}).get("step") != "identifier"
                                      and not cli_ref._sudo_state.get("vault_code"))
                             or bool(cli_ref._secret_state))))

        class _PlaceholderProcessor(Processor):
            """Render grayed-out placeholder text inside the input when empty."""
            def __init__(self, get_text):
                self._get_text = get_text

            def apply_transformation(self, ti):
                if not ti.document.text and ti.lineno == 0:
                    text = self._get_text()
                    if text:
                        # Append after existing fragments (preserves the ❯ prompt).
                        return Transformation(fragments=ti.fragments + [('class:placeholder', text)])
                return Transformation(fragments=ti.fragments)

        input_area.control.input_processors.append(_PlaceholderProcessor(self._tui_placeholder_text))
        return input_area



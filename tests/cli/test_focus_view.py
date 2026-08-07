"""Tests for ``/focus`` — the display-only reduced-output view.

Focus view composes with the existing ``/verbose`` tool-progress machinery
rather than adding a second suppression mechanism.  These tests cover:

* the on/off/status toggle state machine (``resolve_focus_arg``);
* the hidden-line counter and its formatter (which must respect the
  pre-focus ``/verbose`` mode so it never over-claims);
* status-bar segment composition in both CLI renderers;
* the CLI command handler's stash/restore of ``tool_progress_mode``;
* **the prompt-cache invariant** — a real fake turn is dispatched through
  ``agent.tool_executor`` with focus on and with focus off, and the resulting
  model-facing ``messages`` lists must be byte-identical.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.focus_view import (
    FOCUS_CONFIG_KEY,
    FOCUS_STATUSBAR_LABEL,
    FOCUS_TOOL_PROGRESS_MODE,
    effective_tool_progress_mode,
    focus_statusbar_segment,
    format_focus_status,
    format_focus_toggle_message,
    format_hidden_line,
    normalize_tool_progress_mode,
    resolve_focus_arg,
    would_display_tool_line,
)
from hermes_cli.cli_commands_mixin import CLICommandsMixin


# =========================================================================
# Toggle state machine — on | off | status | bare | garbage
# =========================================================================


class TestToggleStateMachine:
    def test_bare_toggles_from_off_to_on(self):
        assert resolve_focus_arg("", False) == ("set", True)





    @pytest.mark.parametrize("word", ["status", "show", "?", "STATUS"])
    def test_status_words_never_mutate(self, word):
        action, target = resolve_focus_arg(word, True)
        assert action == "status"
        assert target is None

    @pytest.mark.parametrize("word", ["sideways", "onn", "--global", "2"])
    def test_garbage_reports_usage(self, word):
        assert resolve_focus_arg(word, False) == ("usage", None)



# =========================================================================
# Suppression respects the existing /verbose modes
# =========================================================================


class TestComposesWithVerboseModes:
    def test_focus_on_snaps_to_the_existing_off_mode(self):
        # Focus view must reuse the tool_progress "off" path, not invent a mode.
        assert FOCUS_TOOL_PROGRESS_MODE == "off"
        for configured in ("off", "new", "all", "verbose"):
            assert effective_tool_progress_mode(True, configured) == "off"

    @pytest.mark.parametrize("configured", ["off", "new", "all", "verbose"])
    def test_focus_off_leaves_the_configured_verbose_mode_untouched(self, configured):
        assert effective_tool_progress_mode(False, configured) == configured




    def test_new_mode_skips_consecutive_repeats_like_the_renderer(self):
        assert would_display_tool_line("new", "terminal", "terminal") is False
        assert would_display_tool_line("new", "read_file", "terminal") is True
        # "all" always counts, even repeats.
        assert would_display_tool_line("all", "terminal", "terminal") is True



# =========================================================================
# Hidden-count formatter + recovery line
# =========================================================================


class TestHiddenCountFormatter:
    def test_zero_and_negative_produce_no_line(self):
        assert format_hidden_line(0) is None
        assert format_hidden_line(-3) is None



    def test_line_always_names_the_recovery_command(self):
        assert "/focus off" in format_hidden_line(2)



class _FocusHost(CLICommandsMixin):
    """Minimal host exposing only the attributes the focus helpers read."""

    def __init__(self, *, enabled=False, saved="all", tool_progress="all"):
        self._focus_view_enabled = enabled
        self._focus_saved_tool_progress = saved
        self._focus_hidden_lines = 0
        self._focus_last_counted_tool = None
        self.tool_progress_mode = tool_progress
        self.agent = None


class TestHiddenCounterAccumulation:
    def test_counts_each_suppressed_tool_line(self):
        host = _FocusHost(enabled=True, saved="all")
        for name in ("terminal", "read_file", "web_search"):
            host._note_focus_hidden_line(name)
        assert host._focus_hidden_lines == 3




    def test_recovery_line_is_emitted_then_counter_resets(self):
        host = _FocusHost(enabled=True, saved="all")
        for name in ("terminal", "read_file"):
            host._note_focus_hidden_line(name)

        with patch("cli._cprint") as printer:
            host._emit_focus_recovery_line()

        assert printer.call_count == 1
        assert "2 tool lines hidden" in printer.call_args[0][0]
        assert "/focus off" in printer.call_args[0][0]
        # Reset so the next turn starts from zero.
        assert host._focus_hidden_lines == 0
        assert host._focus_last_counted_tool is None




# =========================================================================
# CLI command handler — stash / restore / persistence
# =========================================================================


class TestFocusCommandHandler:
    def test_on_stashes_the_verbose_mode_and_snaps_to_off(self):
        host = _FocusHost(enabled=False, saved=None, tool_progress="verbose")
        with patch("cli.save_config_value", return_value=True) as saver, \
             patch("cli._cprint"):
            host._handle_focus_command("/focus on")

        assert host._focus_view_enabled is True
        assert host.tool_progress_mode == "off"
        assert host._focus_saved_tool_progress == "verbose"
        saver.assert_called_once_with(FOCUS_CONFIG_KEY, True)





    def test_idempotent_on_does_not_reclobber_the_stash(self):
        host = _FocusHost(enabled=True, saved="verbose", tool_progress="off")
        with patch("cli.save_config_value") as saver, patch("cli._cprint"):
            host._handle_focus_command("/focus on")
        saver.assert_not_called()
        # The stash still points at the real pre-focus mode, not "off".
        assert host._focus_saved_tool_progress == "verbose"

    def test_live_agent_mode_is_synced(self):
        host = _FocusHost(enabled=False, saved=None, tool_progress="all")
        host.agent = SimpleNamespace(tool_progress_mode="all")
        with patch("cli.save_config_value", return_value=True), patch("cli._cprint"):
            host._handle_focus_command("/focus on")
        # tool_executor gates on the AGENT copy — syncing it is what makes the
        # suppression take effect this turn instead of after an agent rebuild.
        assert host.agent.tool_progress_mode == "off"




# =========================================================================
# Status-bar segment composition
# =========================================================================


class TestStatusBarSegment:
    def test_segment_present_only_when_enabled(self):
        assert focus_statusbar_segment(True) == FOCUS_STATUSBAR_LABEL
        assert focus_statusbar_segment(False) == ""


    @pytest.mark.parametrize("width", [40, 60, 120])
    def test_text_renderer_includes_the_badge_at_every_width_tier(self, width):
        from cli import HermesCLI

        host = HermesCLI.__new__(HermesCLI)
        host.model = "opus"
        host._focus_view_enabled = True

        snapshot = {
            "model_name": "opus",
            "model_short": "opus",
            "duration": "1m",
            "context_percent": 12,
            "context_tokens": 1000,
            "context_length": 200000,
            "compressions": 0,
            "active_background_tasks": 0,
            "active_background_processes": 0,
            "active_background_subagents": 0,
            "battery_label": "",
            "battery_category": "dim",
            "focus_label": FOCUS_STATUSBAR_LABEL,
            "prompt_elapsed": "",
            "idle_since": "",
        }

        with patch.object(HermesCLI, "_get_status_bar_snapshot", return_value=snapshot), \
             patch.object(HermesCLI, "_is_session_yolo_active", return_value=False):
            text = HermesCLI._build_status_bar_text(host, width=width)

        assert "focus" in text




# =========================================================================
# PROMPT-CACHE INVARIANT: model-facing messages are identical either way
# =========================================================================


def _make_agent(tool_progress_mode: str):
    """Build a real AIAgent whose display mode is the only difference."""
    from run_agent import AIAgent

    tool_defs = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "search",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    with (
        patch("run_agent.get_tool_definitions", return_value=tool_defs),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            # quiet_mode False so the display gate is genuinely exercised —
            # with quiet_mode True the tool_progress gate would be moot.
            quiet_mode=False,
            skip_context_files=True,
            skip_memory=True,
            tool_progress_mode=tool_progress_mode,
        )
    agent.client = MagicMock()
    agent.tool_delay = 0
    agent._flush_messages_to_session_db = MagicMock()
    return agent


def _tool_call(call_id: str, query: str):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name="web_search", arguments=json.dumps({"query": query})
        ),
    )


def _run_fake_turn(tool_progress_mode: str, dispatch_mode: str = "sequential"):
    """Dispatch an identical fake turn and return the model-facing messages."""
    agent = _make_agent(tool_progress_mode)
    assistant_message = SimpleNamespace(
        content="",
        tool_calls=[
            _tool_call("call-1", "alpha"),
            _tool_call("call-2", "beta"),
            _tool_call("call-3", "gamma"),
        ],
    )
    messages: list = [
        {"role": "system", "content": "you are hermes"},
        {"role": "user", "content": "find three things"},
    ]

    def fake_dispatch(name, args, task_id, *positional, **kwargs):
        return json.dumps({"ok": args["query"]})

    with (
        patch("run_agent.handle_function_call", side_effect=fake_dispatch),
        patch.object(agent, "_invoke_tool", side_effect=fake_dispatch),
        patch(
            "agent.tool_executor.maybe_persist_tool_result",
            side_effect=lambda **kwargs: kwargs["content"],
        ),
        # Swallow display writes so the test doesn't spam stdout; the point is
        # what lands in `messages`, not what prints.
        patch("builtins.print"),
    ):
        execute = getattr(agent, f"_execute_tool_calls_{dispatch_mode}")
        execute(assistant_message, messages, "task-focus")

    return messages


class TestModelFacingMessagesUnchanged:
    """Focus view is display-only: the request payload must not shift a byte."""

    @pytest.mark.parametrize("dispatch_mode", ["sequential", "concurrent"])
    def test_model_facing_messages_identical_with_focus_on_vs_off(self, dispatch_mode):
        # Focus ON == the existing tool_progress "off" suppression path.
        focus_on = _run_fake_turn(FOCUS_TOOL_PROGRESS_MODE, dispatch_mode)
        # Focus OFF == the default noisy display mode.
        focus_off = _run_fake_turn("all", dispatch_mode)

        assert focus_on == focus_off, (
            "focus view altered the model-facing messages — display-only "
            "invariant violated (prompt cache would break)"
        )
        # Sanity: the turn really did produce tool results to compare.
        assert [m["role"] for m in focus_on].count("tool") == 3
        assert json.loads(focus_on[-1]["content"]) == {"ok": "gamma"}


    def test_toggling_focus_does_not_touch_conversation_history(self):
        host = _FocusHost(enabled=False, saved=None, tool_progress="all")
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        host.conversation_history = history
        snapshot = json.dumps(history)

        with patch("cli.save_config_value", return_value=True), patch("cli._cprint"):
            host._handle_focus_command("/focus on")
            host._note_focus_hidden_line("terminal")
            host._emit_focus_recovery_line()
            host._handle_focus_command("/focus off")

        assert json.dumps(host.conversation_history) == snapshot


# =========================================================================
# Registry wiring
# =========================================================================


class TestCommandRegistration:
    def test_focus_is_registered_with_the_sibling_toggle_convention(self):
        from hermes_cli.commands import resolve_command

        cmd = resolve_command("focus")
        assert cmd is not None
        assert cmd.category == "Configuration"
        assert cmd.args_hint == "[on|off|status]"
        assert set(cmd.subcommands) == {"on", "off", "status"}

    def test_verbose_cycle_releases_focus_view(self):
        # /verbose is the explicit tool-progress control; cycling it must clear
        # the focus badge so the indicator can never contradict the display.
        from cli import HermesCLI

        host = HermesCLI.__new__(HermesCLI)
        host.tool_progress_mode = "off"
        host._focus_view_enabled = True
        host._focus_saved_tool_progress = "all"
        host._focus_hidden_lines = 3
        host._focus_last_counted_tool = "terminal"
        host.agent = None

        with patch("cli.save_config_value", return_value=True), patch("cli._cprint"):
            HermesCLI._toggle_verbose(host)

        assert host._focus_view_enabled is False
        assert host._focus_saved_tool_progress is None
        assert host._focus_hidden_lines == 0
        assert host.tool_progress_mode == "new"

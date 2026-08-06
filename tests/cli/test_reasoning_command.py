"""Tests for the combined /reasoning command.

Covers both reasoning effort level management and reasoning display toggle,
plus the reasoning extraction and display pipeline from run_agent through CLI.

Combines functionality from:
- PR #789 (Aum08Desai): reasoning effort level management
- PR #790 (0xbyt4): reasoning display toggle and rendering
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import re


# ---------------------------------------------------------------------------
# Effort level parsing
# ---------------------------------------------------------------------------

class TestParseReasoningConfig(unittest.TestCase):
    """Verify _parse_reasoning_config handles all effort levels."""

    def _parse(self, effort):
        from cli import _parse_reasoning_config
        return _parse_reasoning_config(effort)

    def test_none_disables(self):
        result = self._parse("none")
        self.assertEqual(result, {"enabled": False})

    def test_valid_levels(self):
        for level in ("low", "medium", "high", "xhigh", "max", "ultra", "minimal"):
            result = self._parse(level)
            self.assertIsNotNone(result)
            self.assertTrue(result.get("enabled"))
            self.assertEqual(result["effort"], level)


    def test_unknown_returns_none(self):
        self.assertIsNone(self._parse("turbo"))



# ---------------------------------------------------------------------------
# /reasoning command handler (combined effort + display)
# ---------------------------------------------------------------------------

class TestHandleReasoningCommand(unittest.TestCase):
    """Test the combined _handle_reasoning_command method."""

    def _make_cli(self, reasoning_config=None, show_reasoning=False):
        """Create a minimal CLI stub with the reasoning attributes."""
        stub = SimpleNamespace(
            reasoning_config=reasoning_config,
            show_reasoning=show_reasoning,
            agent=MagicMock(),
        )
        return stub

    def test_show_enables_display(self):
        stub = self._make_cli(show_reasoning=False)
        # Simulate /reasoning show
        arg = "show"
        if arg in {"show", "on"}:
            stub.show_reasoning = True
            stub.agent.reasoning_callback = lambda x: None
        self.assertTrue(stub.show_reasoning)

    def test_hide_disables_display(self):
        stub = self._make_cli(show_reasoning=True)
        # Simulate /reasoning hide
        arg = "hide"
        if arg in {"hide", "off"}:
            stub.show_reasoning = False
            stub.agent.reasoning_callback = None
        self.assertFalse(stub.show_reasoning)
        self.assertIsNone(stub.agent.reasoning_callback)




    def test_effort_none_disables_reasoning(self):
        from cli import _parse_reasoning_config
        stub = self._make_cli()
        parsed = _parse_reasoning_config("none")
        stub.reasoning_config = parsed
        self.assertEqual(stub.reasoning_config, {"enabled": False})





    def test_effort_defaults_to_session_only(self):
        """Plain /reasoning <level> is session-scoped — no config write."""
        from hermes_cli.cli_commands_mixin import CLICommandsMixin

        stub = self._make_cli(reasoning_config={"enabled": True, "effort": "medium"})
        with patch("cli.save_config_value") as save_config, patch("cli._cprint"):
            CLICommandsMixin._handle_reasoning_command(stub, "/reasoning high")

        save_config.assert_not_called()
        self.assertEqual(stub.reasoning_config, {"enabled": True, "effort": "high"})
        self.assertIsNone(stub.agent)



    def test_new_session_clears_session_reasoning_override(self):
        """/new and /clear must not carry a session-only effort override forward."""
        from cli import CLI_CONFIG, HermesCLI

        agent = SimpleNamespace(
            reasoning_config={"enabled": True, "effort": "high"},
            reset_session_state=MagicMock(),
        )
        stub = SimpleNamespace(
            agent=agent,
            conversation_history=[],
            session_id="old-session",
            _session_db=None,
            _pending_title=None,
            _resumed=False,
            reasoning_config={"enabled": True, "effort": "high"},
            _notify_session_boundary=MagicMock(),
        )

        with patch.dict(CLI_CONFIG.setdefault("agent", {}), {"reasoning_effort": "medium"}):
            HermesCLI.new_session(stub, silent=True)

        self.assertEqual(stub.reasoning_config, {"enabled": True, "effort": "medium"})
        self.assertEqual(agent.reasoning_config, {"enabled": True, "effort": "medium"})
        agent.reset_session_state.assert_called_once()

    def test_new_session_resets_service_tier_and_model_from_config(self):
        """/new re-derives service tier and model from config.yaml — session
        /fast and /model switches do not carry forward (#48055, #23131)."""
        from cli import CLI_CONFIG, HermesCLI

        agent = SimpleNamespace(
            reasoning_config=None,
            reset_session_state=MagicMock(),
            switch_model=MagicMock(),
        )
        stub = SimpleNamespace(
            agent=agent,
            conversation_history=[],
            session_id="old-session",
            _session_db=None,
            _pending_title=None,
            _resumed=False,
            reasoning_config=None,
            _notify_session_boundary=MagicMock(),
            # Session had switched to fast + a session-only model.
            service_tier="priority",
            _pending_one_turn_model_restore={"model": "stale"},
            model="session-switched-model",
            provider="openrouter",
            requested_provider="openrouter",
            api_key="k",
            base_url="",
            api_mode="",
        )

        fake_result = SimpleNamespace(
            success=True,
            new_model="config-default-model",
            target_provider="openrouter",
            api_key="k2",
            base_url="https://openrouter.ai/api/v1",
            api_mode="chat_completions",
        )
        with patch.dict(
            CLI_CONFIG.setdefault("agent", {}),
            {"reasoning_effort": "medium", "service_tier": "normal"},
        ), patch.dict(
            CLI_CONFIG,
            {"model": {"default": "config-default-model", "provider": "openrouter"}},
        ), patch(
            "hermes_cli.model_switch.switch_model", return_value=fake_result
        ):
            HermesCLI.new_session(stub, silent=True)

        # Fast override cleared back to config default (normal → None).
        self.assertIsNone(stub.service_tier)
        # One-turn restore snapshot cleared.
        self.assertIsNone(stub._pending_one_turn_model_restore)
        # Model reset to the config default via the live agent swap.
        self.assertEqual(stub.model, "config-default-model")
        agent.switch_model.assert_called_once()


# ---------------------------------------------------------------------------
# Reasoning extraction and result dict
# ---------------------------------------------------------------------------

class TestLastReasoningInResult(unittest.TestCase):
    """Verify reasoning extraction from the messages list."""

    def _build_messages(self, reasoning=None):
        return [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "Hi there!",
                "reasoning": reasoning,
                "finish_reason": "stop",
            },
        ]

    def test_reasoning_present(self):
        messages = self._build_messages(reasoning="Let me think...")
        last_reasoning = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                break
            if msg.get("role") == "assistant" and msg.get("reasoning"):
                last_reasoning = msg["reasoning"]
                break
        self.assertEqual(last_reasoning, "Let me think...")


    def test_picks_last_assistant(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "...", "reasoning": "first thought"},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "done!", "reasoning": "final thought"},
        ]
        last_reasoning = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                break
            if msg.get("role") == "assistant" and msg.get("reasoning"):
                last_reasoning = msg["reasoning"]
                break
        self.assertEqual(last_reasoning, "final thought")



# ---------------------------------------------------------------------------
# Reasoning display collapse
# ---------------------------------------------------------------------------

class TestReasoningCollapse(unittest.TestCase):
    """Verify long reasoning is collapsed to 10 lines in the box."""

    def test_short_reasoning_not_collapsed(self):
        reasoning = "\n".join(f"Line {i}" for i in range(5))
        lines = reasoning.strip().splitlines()
        self.assertLessEqual(len(lines), 10)



    def test_intermediate_callback_collapses_to_5(self):
        """_on_reasoning shows max 5 lines."""
        reasoning = "\n".join(f"Step {i}" for i in range(12))
        lines = reasoning.strip().splitlines()
        if len(lines) > 5:
            preview = "\n".join(lines[:5])
            preview += f"\n  ... ({len(lines) - 5} more lines)"
        else:
            preview = reasoning.strip()
        preview_lines = preview.splitlines()
        self.assertEqual(len(preview_lines), 6)
        self.assertIn("7 more lines", preview_lines[-1])


# ---------------------------------------------------------------------------
# Reasoning callback
# ---------------------------------------------------------------------------

class TestReasoningCallback(unittest.TestCase):
    """Verify reasoning_callback invocation."""

    def test_callback_invoked_with_reasoning(self):
        captured = []
        agent = MagicMock()
        agent.reasoning_callback = lambda t: captured.append(t)
        agent._extract_reasoning = MagicMock(return_value="deep thought")

        reasoning_text = agent._extract_reasoning(MagicMock())
        if reasoning_text and agent.reasoning_callback:
            agent.reasoning_callback(reasoning_text)
        self.assertEqual(captured, ["deep thought"])


        # No exception = pass


class TestReasoningPreviewBuffering(unittest.TestCase):
    def _make_cli(self):
        from cli import HermesCLI

        cli = HermesCLI.__new__(HermesCLI)
        cli.verbose = True
        cli._spinner_text = ""
        cli._reasoning_preview_buf = ""
        cli._invalidate = lambda *args, **kwargs: None
        return cli

    @patch("cli._cprint")
    def test_streamed_reasoning_chunks_wait_for_boundary(self, mock_cprint):
        cli = self._make_cli()

        cli._on_reasoning("Let")
        cli._on_reasoning(" me")
        cli._on_reasoning(" think")

        self.assertEqual(mock_cprint.call_count, 0)

        cli._on_reasoning(" about this.\n")

        self.assertEqual(mock_cprint.call_count, 1)
        rendered = mock_cprint.call_args[0][0]
        self.assertIn("[thinking] Let me think about this.", rendered)

    @patch("cli._cprint")
    def test_pending_reasoning_flushes_when_thinking_stops(self, mock_cprint):
        cli = self._make_cli()

        cli._on_reasoning("see")
        cli._on_reasoning(" how")
        cli._on_reasoning(" this")
        cli._on_reasoning(" plays")
        cli._on_reasoning(" out")

        self.assertEqual(mock_cprint.call_count, 0)

        cli._on_thinking("")

        self.assertEqual(mock_cprint.call_count, 1)
        rendered = mock_cprint.call_args[0][0]
        self.assertIn("[thinking] see how this plays out", rendered)


    @patch("cli.shutil.get_terminal_size", return_value=SimpleNamespace(columns=60))
    def test_reasoning_flush_threshold_tracks_terminal_width(self, _mock_term):
        cli = self._make_cli()

        cli._reasoning_preview_buf = "a" * 30
        cli._flush_reasoning_preview(force=False)
        self.assertEqual(cli._reasoning_preview_buf, "a" * 30)


class TestReasoningDisplayModeSelection(unittest.TestCase):
    def _make_cli(self, *, show_reasoning=False, streaming_enabled=False, verbose=False):
        from cli import HermesCLI

        cli = HermesCLI.__new__(HermesCLI)
        cli.show_reasoning = show_reasoning
        cli.streaming_enabled = streaming_enabled
        cli.verbose = verbose
        cli._stream_reasoning_delta = lambda text: ("stream", text)
        cli._on_reasoning = lambda text: ("preview", text)
        return cli

    def test_show_reasoning_non_streaming_uses_final_box_only(self):
        cli = self._make_cli(show_reasoning=True, streaming_enabled=False, verbose=False)

        self.assertIsNone(cli._current_reasoning_callback())

    def test_show_reasoning_streaming_uses_live_reasoning_box(self):
        cli = self._make_cli(show_reasoning=True, streaming_enabled=True, verbose=False)

        callback = cli._current_reasoning_callback()
        self.assertIsNotNone(callback)
        self.assertEqual(callback("x"), ("stream", "x"))

    def test_verbose_without_show_reasoning_uses_preview_callback(self):
        cli = self._make_cli(show_reasoning=False, streaming_enabled=False, verbose=True)

        callback = cli._current_reasoning_callback()
        self.assertIsNotNone(callback)
        self.assertEqual(callback("x"), ("preview", "x"))


# ---------------------------------------------------------------------------
# Real provider format extraction
# ---------------------------------------------------------------------------

class TestExtractReasoningFormats(unittest.TestCase):
    """Test _extract_reasoning with real provider response formats."""

    def _get_extractor(self):
        from run_agent import AIAgent
        return AIAgent._extract_reasoning

    def test_openrouter_reasoning_details(self):
        extract = self._get_extractor()
        msg = SimpleNamespace(
            reasoning=None,
            reasoning_content=None,
            reasoning_details=[
                {"type": "reasoning.summary", "summary": "Analyzing Python lists."},
            ],
        )
        result = extract(None, msg)
        self.assertIn("Python lists", result)

    def test_deepseek_reasoning_field(self):
        extract = self._get_extractor()
        msg = SimpleNamespace(
            reasoning="Solving step by step.\nx + y = 8.",
            reasoning_content=None,
        )
        result = extract(None, msg)
        self.assertIn("x + y = 8", result)

    def test_moonshot_reasoning_content(self):
        extract = self._get_extractor()
        msg = SimpleNamespace(
            reasoning_content="Explaining async/await.",
        )
        result = extract(None, msg)
        self.assertIn("async/await", result)



# ---------------------------------------------------------------------------
# Inline <think> block extraction fallback
# ---------------------------------------------------------------------------

class TestInlineThinkBlockExtraction(unittest.TestCase):
    """Test _build_assistant_message extracts inline <think> blocks as reasoning
    when no structured API-level reasoning fields are present."""

    def _build_msg(self, content, reasoning=None, reasoning_content=None, reasoning_details=None, tool_calls=None):
        """Create a mock API response message."""
        msg = SimpleNamespace(content=content, tool_calls=tool_calls)
        if reasoning is not None:
            msg.reasoning = reasoning
        if reasoning_content is not None:
            msg.reasoning_content = reasoning_content
        if reasoning_details is not None:
            msg.reasoning_details = reasoning_details
        return msg

    def _make_agent(self):
        """Create a minimal agent with _build_assistant_message."""
        from run_agent import AIAgent
        agent = MagicMock(spec=AIAgent)
        agent._build_assistant_message = AIAgent._build_assistant_message.__get__(agent)
        agent._extract_reasoning = AIAgent._extract_reasoning.__get__(agent)
        agent.verbose_logging = False
        agent.reasoning_callback = None
        agent.stream_delta_callback = None  # non-streaming by default
        agent._stream_callback = None  # non-streaming by default
        return agent

    def test_single_think_block_extracted(self):
        agent = self._make_agent()
        api_msg = self._build_msg("<think>Let me calculate 2+2=4.</think>The answer is 4.")
        result = agent._build_assistant_message(api_msg, "stop")
        self.assertEqual(result["reasoning"], "Let me calculate 2+2=4.")



    def test_structured_reasoning_takes_priority(self):
        """When structured API reasoning exists, inline think blocks should NOT override."""
        agent = self._make_agent()
        api_msg = self._build_msg(
            "<think>Inline thought.</think>Response text.",
            reasoning="Structured reasoning from API.",
        )
        result = agent._build_assistant_message(api_msg, "stop")
        self.assertEqual(result["reasoning"], "Structured reasoning from API.")



    def test_callback_fires_for_inline_think(self):
        """Reasoning callback should fire when reasoning is extracted from inline think blocks."""
        agent = self._make_agent()
        captured = []
        agent.reasoning_callback = lambda t: captured.append(t)
        api_msg = self._build_msg("<think>Deep analysis here.</think>Answer.")
        agent._build_assistant_message(api_msg, "stop")
        self.assertEqual(len(captured), 1)
        self.assertIn("Deep analysis", captured[0])


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

class TestConfigDefault(unittest.TestCase):
    """Verify config default for show_reasoning."""

    def test_default_config_has_show_reasoning(self):
        from hermes_cli.config import DEFAULT_CONFIG
        display = DEFAULT_CONFIG.get("display", {})
        self.assertIn("show_reasoning", display)
        # Default ON (July 2026 TTFT-perception change): thinking models
        # stream reasoning for tens of seconds; hiding it left users staring
        # at a spinner. The key must exist and be a bool.
        self.assertTrue(display["show_reasoning"])


class TestCommandRegistered(unittest.TestCase):
    """Verify /reasoning is in the COMMANDS dict."""

    def test_reasoning_in_commands(self):
        from hermes_cli.commands import COMMANDS
        self.assertIn("/reasoning", COMMANDS)


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

class TestEndToEndPipeline(unittest.TestCase):
    """Simulate the full pipeline: extraction -> result dict -> display."""

    def test_openrouter_claude_pipeline(self):
        from run_agent import AIAgent

        api_message = SimpleNamespace(
            role="assistant",
            content="Lists support append().",
            tool_calls=None,
            reasoning=None,
            reasoning_content=None,
            reasoning_details=[
                {"type": "reasoning.summary", "summary": "Python list methods."},
            ],
        )

        reasoning = AIAgent._extract_reasoning(None, api_message)
        self.assertIsNotNone(reasoning)

        messages = [
            {"role": "user", "content": "How do I add items?"},
            {"role": "assistant", "content": api_message.content, "reasoning": reasoning},
        ]

        last_reasoning = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                break
            if msg.get("role") == "assistant" and msg.get("reasoning"):
                last_reasoning = msg["reasoning"]
                break

        result = {
            "final_response": api_message.content,
            "last_reasoning": last_reasoning,
        }

        self.assertIn("last_reasoning", result)
        self.assertIn("Python list methods", result["last_reasoning"])



# ---------------------------------------------------------------------------
# Duplicate reasoning box prevention (Bug fix: 3 boxes for 1 reasoning)
# ---------------------------------------------------------------------------

class TestReasoningDeltasFiredFlag(unittest.TestCase):
    """_build_assistant_message should not re-fire reasoning_callback when
    reasoning was already streamed via _fire_reasoning_delta."""

    def _make_agent(self):
        from run_agent import AIAgent
        agent = AIAgent.__new__(AIAgent)
        agent.reasoning_callback = None
        agent.stream_delta_callback = None
        agent._stream_callback = None
        agent.verbose_logging = False
        return agent

    def test_fire_reasoning_delta_calls_callback(self):
        agent = self._make_agent()
        captured = []
        agent.reasoning_callback = lambda t: captured.append(t)
        agent._fire_reasoning_delta("thinking...")
        self.assertEqual(captured, ["thinking..."])



    def test_build_assistant_message_fires_callback_without_streaming(self):
        """When no streaming is active, callback always fires for structured
        reasoning."""
        agent = self._make_agent()
        captured = []
        agent.reasoning_callback = lambda t: captured.append(t)
        # No streaming
        agent.stream_delta_callback = None

        msg = SimpleNamespace(
            content="I'll merge that.",
            tool_calls=None,
            reasoning_content="Let me merge the PR.",
            reasoning=None,
            reasoning_details=None,
        )
        agent._build_assistant_message(msg, "stop")

        self.assertEqual(captured, ["Let me merge the PR."])


class TestReasoningShownThisTurnFlag(unittest.TestCase):
    """Post-response reasoning display should be suppressed when reasoning
    was already shown during streaming in a tool-calling loop."""

    def _make_cli(self):
        from cli import HermesCLI
        cli = HermesCLI.__new__(HermesCLI)
        cli.show_reasoning = True
        cli.streaming_enabled = True
        cli._stream_box_opened = False
        cli._reasoning_box_opened = False
        cli._reasoning_stream_started = False
        cli._reasoning_shown_this_turn = False
        cli._reasoning_buf = ""
        cli._stream_buf = ""
        cli._stream_started = False
        cli._stream_text_ansi = ""
        cli._stream_prefilt = ""
        cli._in_reasoning_block = False
        cli._reasoning_preview_buf = ""
        return cli

    @patch("cli._cprint")
    def test_streaming_reasoning_sets_turn_flag(self, mock_cprint):
        cli = self._make_cli()
        self.assertFalse(cli._reasoning_shown_this_turn)
        cli._stream_reasoning_delta("Thinking about it...")
        self.assertTrue(cli._reasoning_shown_this_turn)


    @patch("cli._cprint")
    def test_turn_flag_cleared_before_new_turn(self, mock_cprint):
        """The turn flag should be reset at the start of a new user turn.
        This happens outside _reset_stream_state, at the call site."""
        cli = self._make_cli()
        cli._reasoning_shown_this_turn = True

        # Simulate new user turn setup
        cli._reset_stream_state()
        cli._reasoning_shown_this_turn = False  # done by process_input

        self.assertFalse(cli._reasoning_shown_this_turn)


if __name__ == "__main__":
    unittest.main()

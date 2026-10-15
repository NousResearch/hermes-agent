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




# ---------------------------------------------------------------------------
# Reasoning display collapse
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Reasoning callback
# ---------------------------------------------------------------------------



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





# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------




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
        agent.context_compressor = None
        return agent




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

    def test_native_codex_summary_entries_keep_separate_reasoning_ids(self):
        agent = self._make_agent()
        structured = []
        agent.reasoning_event_callback = lambda phase, item_id, text="": structured.append((phase, item_id, text))
        msg = SimpleNamespace(
            content="Done.", tool_calls=None,
            reasoning_content="Inspecting\nChecking\n\nVerifying",
            reasoning=None, reasoning_details=None,
            codex_reasoning_items=[
                {"type": "reasoning", "id": "rs_live", "summary": [
                    {"type": "summary_text", "text": "Inspecting"},
                    {"type": "summary_text", "text": "Checking"},
                ]},
                {"type": "reasoning", "id": "rs_verify", "summary": [
                    {"type": "summary_text", "text": "Verifying"}
                ]},
            ],
        )
        agent._build_assistant_message(msg, "stop")
        self.assertEqual(structured, [
            ("start", "rs_live:summary:0", ""), ("delta", "rs_live:summary:0", "Inspecting"), ("end", "rs_live:summary:0", ""),
            ("start", "rs_live:summary:1", ""), ("delta", "rs_live:summary:1", "Checking"), ("end", "rs_live:summary:1", ""),
            ("start", "rs_verify:summary:0", ""), ("delta", "rs_verify:summary:0", "Verifying"), ("end", "rs_verify:summary:0", ""),
        ])


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




if __name__ == "__main__":
    unittest.main()

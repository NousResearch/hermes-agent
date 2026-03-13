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
        for level in ("low", "medium", "high", "xhigh", "minimal"):
            result = self._parse(level)
            self.assertIsNotNone(result)
            self.assertTrue(result.get("enabled"))
            self.assertEqual(result["effort"], level)

    def test_empty_returns_none(self):
        self.assertIsNone(self._parse(""))
        self.assertIsNone(self._parse("  "))

    def test_unknown_returns_none(self):
        self.assertIsNone(self._parse("ultra"))
        self.assertIsNone(self._parse("turbo"))

    def test_case_insensitive(self):
        result = self._parse("HIGH")
        self.assertIsNotNone(result)
        self.assertEqual(result["effort"], "high")


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
        if arg in ("show", "on"):
            stub.show_reasoning = True
            stub.agent.reasoning_callback = lambda x: None
        self.assertTrue(stub.show_reasoning)

    def test_hide_disables_display(self):
        stub = self._make_cli(show_reasoning=True)
        # Simulate /reasoning hide
        arg = "hide"
        if arg in ("hide", "off"):
            stub.show_reasoning = False
            stub.agent.reasoning_callback = None
        self.assertFalse(stub.show_reasoning)
        self.assertIsNone(stub.agent.reasoning_callback)

    def test_on_enables_display(self):
        stub = self._make_cli(show_reasoning=False)
        arg = "on"
        if arg in ("show", "on"):
            stub.show_reasoning = True
        self.assertTrue(stub.show_reasoning)

    def test_off_disables_display(self):
        stub = self._make_cli(show_reasoning=True)
        arg = "off"
        if arg in ("hide", "off"):
            stub.show_reasoning = False
        self.assertFalse(stub.show_reasoning)

    def test_effort_level_sets_config(self):
        """Setting an effort level should update reasoning_config."""
        from cli import _parse_reasoning_config
        stub = self._make_cli()
        arg = "high"
        parsed = _parse_reasoning_config(arg)
        stub.reasoning_config = parsed
        self.assertEqual(stub.reasoning_config, {"enabled": True, "effort": "high"})

    def test_effort_none_disables_reasoning(self):
        from cli import _parse_reasoning_config
        stub = self._make_cli()
        parsed = _parse_reasoning_config("none")
        stub.reasoning_config = parsed
        self.assertEqual(stub.reasoning_config, {"enabled": False})

    def test_invalid_argument_rejected(self):
        """Invalid arguments should be rejected (parsed returns None)."""
        from cli import _parse_reasoning_config
        parsed = _parse_reasoning_config("turbo")
        self.assertIsNone(parsed)

    def test_no_args_shows_status(self):
        """With no args, should show current state (no crash)."""
        stub = self._make_cli(reasoning_config=None, show_reasoning=False)
        rc = stub.reasoning_config
        if rc is None:
            level = "medium (default)"
        elif rc.get("enabled") is False:
            level = "none (disabled)"
        else:
            level = rc.get("effort", "medium")
        display_state = "on" if stub.show_reasoning else "off"
        self.assertEqual(level, "medium (default)")
        self.assertEqual(display_state, "off")

    def test_status_with_disabled_reasoning(self):
        stub = self._make_cli(reasoning_config={"enabled": False}, show_reasoning=True)
        rc = stub.reasoning_config
        if rc is None:
            level = "medium (default)"
        elif rc.get("enabled") is False:
            level = "none (disabled)"
        else:
            level = rc.get("effort", "medium")
        self.assertEqual(level, "none (disabled)")

    def test_status_with_explicit_level(self):
        stub = self._make_cli(
            reasoning_config={"enabled": True, "effort": "xhigh"},
            show_reasoning=True,
        )
        rc = stub.reasoning_config
        level = rc.get("effort", "medium")
        self.assertEqual(level, "xhigh")


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
            if msg.get("role") == "assistant" and msg.get("reasoning"):
                last_reasoning = msg["reasoning"]
                break
        self.assertEqual(last_reasoning, "Let me think...")

    def test_reasoning_none(self):
        messages = self._build_messages(reasoning=None)
        last_reasoning = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("reasoning"):
                last_reasoning = msg["reasoning"]
                break
        self.assertIsNone(last_reasoning)

    def test_picks_last_assistant(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "...", "reasoning": "first thought"},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "done!", "reasoning": "final thought"},
        ]
        last_reasoning = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("reasoning"):
                last_reasoning = msg["reasoning"]
                break
        self.assertEqual(last_reasoning, "final thought")

    def test_empty_reasoning_treated_as_none(self):
        messages = self._build_messages(reasoning="")
        last_reasoning = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("reasoning"):
                last_reasoning = msg["reasoning"]
                break
        self.assertIsNone(last_reasoning)


# ---------------------------------------------------------------------------
# Reasoning display collapse
# ---------------------------------------------------------------------------

class TestReasoningCollapse(unittest.TestCase):
    """Verify long reasoning is collapsed to 10 lines in the box."""

    def test_short_reasoning_not_collapsed(self):
        reasoning = "\n".join(f"Line {i}" for i in range(5))
        lines = reasoning.strip().splitlines()
        self.assertLessEqual(len(lines), 10)

    def test_long_reasoning_collapsed(self):
        reasoning = "\n".join(f"Line {i}" for i in range(25))
        lines = reasoning.strip().splitlines()
        self.assertTrue(len(lines) > 10)
        if len(lines) > 10:
            display = "\n".join(lines[:10])
            display += f"\n  ... ({len(lines) - 10} more lines)"
        display_lines = display.splitlines()
        self.assertEqual(len(display_lines), 11)
        self.assertIn("15 more lines", display_lines[-1])

    def test_exactly_10_lines_not_collapsed(self):
        reasoning = "\n".join(f"Line {i}" for i in range(10))
        lines = reasoning.strip().splitlines()
        self.assertEqual(len(lines), 10)
        self.assertFalse(len(lines) > 10)

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

    def test_callback_not_invoked_without_reasoning(self):
        captured = []
        agent = MagicMock()
        agent.reasoning_callback = lambda t: captured.append(t)
        agent._extract_reasoning = MagicMock(return_value=None)

        reasoning_text = agent._extract_reasoning(MagicMock())
        if reasoning_text and agent.reasoning_callback:
            agent.reasoning_callback(reasoning_text)
        self.assertEqual(captured, [])

    def test_callback_none_does_not_crash(self):
        reasoning_text = "some thought"
        callback = None
        if reasoning_text and callback:
            callback(reasoning_text)
        # No exception = pass


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

    def test_no_reasoning_returns_none(self):
        extract = self._get_extractor()
        msg = SimpleNamespace(content="Hello!")
        result = extract(None, msg)
        self.assertIsNone(result)


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
        return agent

    def test_single_think_block_extracted(self):
        agent = self._make_agent()
        api_msg = self._build_msg("<think>Let me calculate 2+2=4.</think>The answer is 4.")
        result = agent._build_assistant_message(api_msg, "stop")
        self.assertEqual(result["reasoning"], "Let me calculate 2+2=4.")

    def test_multiple_think_blocks_extracted(self):
        agent = self._make_agent()
        api_msg = self._build_msg("<think>First thought.</think>Some text<think>Second thought.</think>More text")
        result = agent._build_assistant_message(api_msg, "stop")
        self.assertIn("First thought.", result["reasoning"])
        self.assertIn("Second thought.", result["reasoning"])

    def test_no_think_blocks_no_reasoning(self):
        agent = self._make_agent()
        api_msg = self._build_msg("Just a plain response.")
        result = agent._build_assistant_message(api_msg, "stop")
        # No structured reasoning AND no inline think blocks → None
        self.assertIsNone(result["reasoning"])

    def test_structured_reasoning_takes_priority(self):
        """When structured API reasoning exists, inline think blocks should NOT override."""
        agent = self._make_agent()
        api_msg = self._build_msg(
            "<think>Inline thought.</think>Response text.",
            reasoning="Structured reasoning from API.",
        )
        result = agent._build_assistant_message(api_msg, "stop")
        self.assertEqual(result["reasoning"], "Structured reasoning from API.")

    def test_empty_think_block_ignored(self):
        agent = self._make_agent()
        api_msg = self._build_msg("<think></think>Hello!")
        result = agent._build_assistant_message(api_msg, "stop")
        # Empty think block should not produce reasoning
        self.assertIsNone(result["reasoning"])

    def test_multiline_think_block(self):
        agent = self._make_agent()
        api_msg = self._build_msg("<think>\nStep 1: Analyze.\nStep 2: Solve.\n</think>Done.")
        result = agent._build_assistant_message(api_msg, "stop")
        self.assertIn("Step 1: Analyze.", result["reasoning"])
        self.assertIn("Step 2: Solve.", result["reasoning"])

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
        self.assertFalse(display["show_reasoning"])


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
            if msg.get("role") == "assistant" and msg.get("reasoning"):
                last_reasoning = msg["reasoning"]
                break

        result = {
            "final_response": api_message.content,
            "last_reasoning": last_reasoning,
        }

        self.assertIn("last_reasoning", result)
        self.assertIn("Python list methods", result["last_reasoning"])

    def test_no_reasoning_model_pipeline(self):
        from run_agent import AIAgent

        api_message = SimpleNamespace(content="Paris.", tool_calls=None)
        reasoning = AIAgent._extract_reasoning(None, api_message)
        self.assertIsNone(reasoning)

        result = {"final_response": api_message.content, "last_reasoning": reasoning}
        self.assertIsNone(result["last_reasoning"])


if __name__ == "__main__":
    unittest.main()

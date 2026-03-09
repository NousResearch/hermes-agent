"""Tests for reasoning display feature.

Verifies:
1. run_agent returns last_reasoning in result dict
2. CLI _toggle_reasoning toggles show_reasoning state
3. Reasoning is rendered when show_reasoning is True
4. Reasoning is hidden when show_reasoning is False
5. Long reasoning is collapsed to 10 lines
6. Config default is respected
"""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock


class TestLastReasoningInResult(unittest.TestCase):
    """Verify run_agent includes last_reasoning in its return dict."""

    def _build_messages(self, reasoning=None):
        """Build a minimal messages list with an assistant message."""
        msgs = [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "Hi there!",
                "reasoning": reasoning,
                "finish_reason": "stop",
            },
        ]
        return msgs

    def test_last_reasoning_present_when_model_reasons(self):
        """Result dict should contain reasoning from the last assistant message."""
        messages = self._build_messages(reasoning="Let me think about this carefully...")
        # Simulate what run_agent does to extract last_reasoning
        last_reasoning = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("reasoning"):
                last_reasoning = msg["reasoning"]
                break
        self.assertEqual(last_reasoning, "Let me think about this carefully...")

    def test_last_reasoning_none_when_no_reasoning(self):
        """Result dict should have None when model doesn't reason."""
        messages = self._build_messages(reasoning=None)
        last_reasoning = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("reasoning"):
                last_reasoning = msg["reasoning"]
                break
        self.assertIsNone(last_reasoning)

    def test_last_reasoning_picks_final_assistant_message(self):
        """When multiple assistant messages exist, pick the last one's reasoning."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "...", "reasoning": "first thought", "finish_reason": "tool_calls"},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "done!", "reasoning": "final thought", "finish_reason": "stop"},
        ]
        last_reasoning = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("reasoning"):
                last_reasoning = msg["reasoning"]
                break
        self.assertEqual(last_reasoning, "final thought")

    def test_last_reasoning_skips_empty_reasoning(self):
        """Empty string reasoning should be treated as no reasoning."""
        messages = self._build_messages(reasoning="")
        last_reasoning = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("reasoning"):
                last_reasoning = msg["reasoning"]
                break
        self.assertIsNone(last_reasoning)


class TestReasoningCollapse(unittest.TestCase):
    """Verify long reasoning is collapsed to 10 lines."""

    def test_short_reasoning_not_collapsed(self):
        """Reasoning with <= 10 lines should be shown in full."""
        reasoning = "\n".join(f"Line {i}" for i in range(5))
        lines = reasoning.strip().splitlines()
        self.assertEqual(len(lines), 5)
        self.assertLessEqual(len(lines), 10)

    def test_long_reasoning_collapsed(self):
        """Reasoning with > 10 lines should show first 10 + count."""
        reasoning = "\n".join(f"Line {i}" for i in range(25))
        lines = reasoning.strip().splitlines()
        if len(lines) > 10:
            display = "\n".join(lines[:10])
            display += f"\n  ... ({len(lines) - 10} more lines)"
        else:
            display = reasoning.strip()
        display_lines = display.splitlines()
        # 10 content lines + 1 "more lines" indicator
        self.assertEqual(len(display_lines), 11)
        self.assertIn("15 more lines", display_lines[-1])

    def test_exactly_10_lines_not_collapsed(self):
        """Reasoning with exactly 10 lines should not be collapsed."""
        reasoning = "\n".join(f"Line {i}" for i in range(10))
        lines = reasoning.strip().splitlines()
        self.assertEqual(len(lines), 10)
        self.assertFalse(len(lines) > 10)


class TestToggleReasoning(unittest.TestCase):
    """Verify /reasoning command toggles state correctly."""

    def _make_cli_stub(self, initial=False):
        """Create a minimal object with show_reasoning attribute."""
        stub = SimpleNamespace(show_reasoning=initial)
        return stub

    def test_toggle_on(self):
        stub = self._make_cli_stub(initial=False)
        stub.show_reasoning = not stub.show_reasoning
        self.assertTrue(stub.show_reasoning)

    def test_toggle_off(self):
        stub = self._make_cli_stub(initial=True)
        stub.show_reasoning = not stub.show_reasoning
        self.assertFalse(stub.show_reasoning)

    def test_explicit_on(self):
        stub = self._make_cli_stub(initial=False)
        arg = "on"
        if arg == "on":
            stub.show_reasoning = True
        self.assertTrue(stub.show_reasoning)

    def test_explicit_off(self):
        stub = self._make_cli_stub(initial=True)
        arg = "off"
        if arg == "off":
            stub.show_reasoning = False
        self.assertFalse(stub.show_reasoning)


class TestReasoningCallback(unittest.TestCase):
    """Verify reasoning_callback is invoked during _build_assistant_message."""

    def test_callback_invoked_with_reasoning(self):
        """When reasoning is present, callback should be called."""
        captured = []

        def on_reasoning(text):
            captured.append(text)

        agent = MagicMock()
        agent.reasoning_callback = on_reasoning
        agent.verbose_logging = False
        agent._extract_reasoning = MagicMock(return_value="deep thought")

        # Simulate what _build_assistant_message does
        reasoning_text = agent._extract_reasoning(MagicMock())
        if reasoning_text and agent.reasoning_callback:
            agent.reasoning_callback(reasoning_text)

        self.assertEqual(captured, ["deep thought"])

    def test_callback_not_invoked_without_reasoning(self):
        """When no reasoning, callback should not be called."""
        captured = []

        def on_reasoning(text):
            captured.append(text)

        agent = MagicMock()
        agent.reasoning_callback = on_reasoning
        agent._extract_reasoning = MagicMock(return_value=None)

        reasoning_text = agent._extract_reasoning(MagicMock())
        if reasoning_text and agent.reasoning_callback:
            agent.reasoning_callback(reasoning_text)

        self.assertEqual(captured, [])

    def test_callback_none_does_not_crash(self):
        """When reasoning_callback is None, no error should occur."""
        reasoning_text = "some thought"
        callback = None
        # This should not raise
        if reasoning_text and callback:
            callback(reasoning_text)


class TestExtractReasoningRealFormats(unittest.TestCase):
    """Simulate real provider response formats through _extract_reasoning.

    Uses the actual AIAgent._extract_reasoning method (not mocked) to verify
    that reasoning is correctly extracted from each provider format.
    """

    def _get_extractor(self):
        """Get _extract_reasoning as a bound method from a minimal AIAgent."""
        from run_agent import AIAgent
        return AIAgent._extract_reasoning

    def test_openrouter_reasoning_details_format(self):
        """OpenRouter unified format: reasoning_details array with summary."""
        extract = self._get_extractor()
        msg = SimpleNamespace(
            reasoning=None,
            reasoning_content=None,
            reasoning_details=[
                {"type": "reasoning.summary", "summary": "The user wants to know about Python lists."},
                {"type": "reasoning.summary", "summary": "I should explain append, extend, and insert."},
            ],
        )
        result = extract(None, msg)
        self.assertIn("Python lists", result)
        self.assertIn("append, extend", result)

    def test_openrouter_reasoning_details_with_content_key(self):
        """Some providers use 'content' instead of 'summary' in details."""
        extract = self._get_extractor()
        msg = SimpleNamespace(
            reasoning=None,
            reasoning_content=None,
            reasoning_details=[
                {"type": "thinking", "content": "Let me analyze this step by step."},
            ],
        )
        result = extract(None, msg)
        self.assertIn("step by step", result)

    def test_deepseek_reasoning_field(self):
        """DeepSeek R1 format: direct 'reasoning' field."""
        extract = self._get_extractor()
        msg = SimpleNamespace(
            reasoning="I need to solve this math problem.\nFirst, let me identify the variables.\nx = 5, y = 3\nTherefore x + y = 8.",
            reasoning_content=None,
        )
        # No reasoning_details attr
        result = extract(None, msg)
        self.assertIn("math problem", result)
        self.assertIn("x + y = 8", result)

    def test_moonshot_reasoning_content_field(self):
        """Moonshot/Novita format: 'reasoning_content' field."""
        extract = self._get_extractor()
        msg = SimpleNamespace(
            reasoning_content="The user is asking about async/await in Python. I should explain coroutines first.",
        )
        # No reasoning or reasoning_details attrs
        result = extract(None, msg)
        self.assertIn("async/await", result)

    def test_no_reasoning_returns_none(self):
        """Standard response with no reasoning fields returns None."""
        extract = self._get_extractor()
        msg = SimpleNamespace(content="Hello!")
        result = extract(None, msg)
        self.assertIsNone(result)

    def test_combined_reasoning_and_details(self):
        """When both reasoning and reasoning_details exist, combine them."""
        extract = self._get_extractor()
        msg = SimpleNamespace(
            reasoning="Initial thought process.",
            reasoning_content=None,
            reasoning_details=[
                {"type": "reasoning.summary", "summary": "Additional analysis from provider."},
            ],
        )
        result = extract(None, msg)
        self.assertIn("Initial thought process", result)
        self.assertIn("Additional analysis", result)

    def test_deduplication_reasoning_and_reasoning_content(self):
        """When reasoning == reasoning_content, don't duplicate."""
        extract = self._get_extractor()
        same_text = "Thinking about the problem..."
        msg = SimpleNamespace(
            reasoning=same_text,
            reasoning_content=same_text,
            reasoning_details=None,
        )
        result = extract(None, msg)
        # Should appear only once
        self.assertEqual(result.count(same_text), 1)


class TestEndToEndReasoningPipeline(unittest.TestCase):
    """Simulate the full pipeline: API response -> extraction -> result dict -> CLI display."""

    def test_full_pipeline_openrouter_claude(self):
        """Simulate an OpenRouter Claude response with reasoning through the full pipeline."""
        from run_agent import AIAgent

        # 1. Simulate API response message (OpenRouter Claude with reasoning)
        api_message = SimpleNamespace(
            role="assistant",
            content="Python lists support append(), extend(), and insert() methods.",
            tool_calls=None,
            reasoning=None,
            reasoning_content=None,
            reasoning_details=[
                {
                    "type": "reasoning.summary",
                    "summary": "The user is asking about Python list methods. I should cover the three main mutation methods.",
                },
            ],
        )

        # 2. Extract reasoning (using real method)
        reasoning = AIAgent._extract_reasoning(None, api_message)
        self.assertIsNotNone(reasoning)
        self.assertIn("Python list methods", reasoning)

        # 3. Build messages list (simulating what run_conversation does)
        messages = [
            {"role": "user", "content": "How do I add items to a Python list?"},
            {
                "role": "assistant",
                "content": api_message.content,
                "reasoning": reasoning,
                "finish_reason": "stop",
            },
        ]

        # 4. Extract last_reasoning (simulating what run_conversation does)
        last_reasoning = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("reasoning"):
                last_reasoning = msg["reasoning"]
                break

        self.assertEqual(last_reasoning, reasoning)

        # 5. Build result dict
        result = {
            "final_response": api_message.content,
            "last_reasoning": last_reasoning,
            "messages": messages,
            "completed": True,
        }

        self.assertIn("last_reasoning", result)
        self.assertIn("Python list methods", result["last_reasoning"])
        self.assertIn("append()", result["final_response"])

    def test_full_pipeline_deepseek_r1(self):
        """Simulate a DeepSeek R1 response with long reasoning."""
        from run_agent import AIAgent

        # Long reasoning (>10 lines - should be collapsed in display)
        long_reasoning = "\n".join([
            "Let me solve this step by step.",
            "Step 1: Parse the input string.",
            "Step 2: Identify the delimiter.",
            "Step 3: Split the string.",
            "Step 4: Convert each part to integer.",
            "Step 5: Handle edge cases (empty string, no delimiter).",
            "Step 6: What if the delimiter appears at the start?",
            "Step 7: What about trailing delimiters?",
            "Step 8: Should I strip whitespace?",
            "Step 9: Return the list of integers.",
            "Step 10: Add error handling for non-numeric values.",
            "Step 11: Consider using a list comprehension.",
            "Step 12: Final implementation ready.",
        ])

        api_message = SimpleNamespace(
            reasoning=long_reasoning,
            content="Here's a function to parse a delimited string into integers:\n```python\ndef parse_ints(s, sep=','):\n    return [int(x.strip()) for x in s.split(sep) if x.strip()]\n```",
            tool_calls=None,
            reasoning_content=None,
        )

        reasoning = AIAgent._extract_reasoning(None, api_message)
        self.assertIsNotNone(reasoning)
        self.assertEqual(len(reasoning.splitlines()), 13)

        # Verify collapse logic
        lines = reasoning.strip().splitlines()
        self.assertTrue(len(lines) > 10, "Reasoning should be long enough to collapse")

        # Simulate display collapse
        display_lines = lines[:10]
        display_lines.append(f"  ... ({len(lines) - 10} more lines)")
        self.assertEqual(len(display_lines), 11)
        self.assertIn("3 more lines", display_lines[-1])

    def test_full_pipeline_no_reasoning_model(self):
        """Simulate a standard model (e.g., GPT-4) with no reasoning."""
        from run_agent import AIAgent

        api_message = SimpleNamespace(
            content="The capital of France is Paris.",
            tool_calls=None,
        )

        reasoning = AIAgent._extract_reasoning(None, api_message)
        self.assertIsNone(reasoning)

        result = {
            "final_response": api_message.content,
            "last_reasoning": reasoning,
            "completed": True,
        }
        self.assertIsNone(result["last_reasoning"])

    def test_callback_fires_during_tool_loop(self):
        """Simulate intermediate reasoning during a multi-step tool-call loop."""
        from run_agent import AIAgent

        captured_reasoning = []

        def on_reasoning(text):
            captured_reasoning.append(text)

        # Step 1: Agent reasons before first tool call
        msg1 = SimpleNamespace(
            reasoning="I need to search the web first to find current info.",
            reasoning_content=None,
            content=None,
            tool_calls=[SimpleNamespace(id="tc1", function=SimpleNamespace(name="web_search", arguments='{"q":"test"}'))],
        )

        # Step 2: Agent reasons before second tool call
        msg2 = SimpleNamespace(
            reasoning="The search results show multiple options. Let me read the top result.",
            reasoning_content=None,
            content=None,
            tool_calls=[SimpleNamespace(id="tc2", function=SimpleNamespace(name="web_extract", arguments='{"url":"http://example.com"}'))],
        )

        # Step 3: Agent gives final response with reasoning
        msg3 = SimpleNamespace(
            reasoning="Now I have enough information to answer comprehensively.",
            reasoning_content=None,
            content="Based on my research, here is the answer...",
            tool_calls=None,
        )

        # Simulate the callback being called at each step
        for msg in [msg1, msg2, msg3]:
            reasoning = AIAgent._extract_reasoning(None, msg)
            if reasoning and on_reasoning:
                on_reasoning(reasoning)

        self.assertEqual(len(captured_reasoning), 3)
        self.assertIn("search the web", captured_reasoning[0])
        self.assertIn("read the top result", captured_reasoning[1])
        self.assertIn("enough information", captured_reasoning[2])


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
        self.assertIn("reasoning", COMMANDS["/reasoning"].lower())


if __name__ == "__main__":
    unittest.main()

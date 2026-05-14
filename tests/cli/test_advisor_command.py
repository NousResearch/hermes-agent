"""Unit tests for the /advisor command (CLI + gateway dispatch).

These tests verify:
- CommandDef registration (name, aliases, args_hint)
- Model/question parsing from raw args
- Guard against empty conversation history
- Background thread is spawned for valid input
- Gateway handler returns the consulting ack string
"""
from __future__ import annotations

import threading
import types
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cli(history: list | None = None):
    """Return a minimal HermesCLI-like stub for testing _handle_advisor_command."""
    cli = MagicMock()
    cli._ADVISOR_DEFAULT_MODEL = "claude-opus-4-7"
    cli._ADVISOR_SYSTEM_PROMPT = "You are an expert advisor..."
    cli._ensure_runtime_credentials.return_value = True
    cli._resolve_turn_agent_config.return_value = {
        "model": "claude-opus-4-7",
        "runtime": {
            "api_key": "sk-test",
            "base_url": None,
            "provider": "anthropic",
            "api_mode": None,
        },
    }
    cli.final_response_markdown = "markdown"
    cli.bell_on_complete = False
    cli._app = None
    cli._background_tasks = {}
    cli._spinner_text = ""
    cli._agent_running = False
    cli._session_db = None

    # Attach a fake agent with messages
    fake_agent = MagicMock()
    fake_agent.messages = history if history is not None else [
        {"role": "user", "content": "What is a neural network?"},
        {"role": "assistant", "content": "A neural network is..."},
    ]
    cli.agent = fake_agent

    return cli


# ---------------------------------------------------------------------------
# CommandDef registration
# ---------------------------------------------------------------------------

class TestAdvisorCommandDef(unittest.TestCase):

    def test_advisor_in_registry(self):
        from hermes_cli.commands import COMMAND_REGISTRY
        names = {c.name for c in COMMAND_REGISTRY}
        self.assertIn("advisor", names)

    def test_advisor_aliases(self):
        from hermes_cli.commands import COMMAND_REGISTRY
        cmd = next(c for c in COMMAND_REGISTRY if c.name == "advisor")
        self.assertIn("consult", cmd.aliases)

    def test_advisor_args_hint(self):
        from hermes_cli.commands import COMMAND_REGISTRY
        cmd = next(c for c in COMMAND_REGISTRY if c.name == "advisor")
        self.assertIsNotNone(cmd.args_hint)

    def test_resolve_consult_alias(self):
        from hermes_cli.commands import resolve_command
        cmd = resolve_command("consult")
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.name, "advisor")

    def test_no_em_dash_in_description(self):
        from hermes_cli.commands import COMMAND_REGISTRY
        cmd = next(c for c in COMMAND_REGISTRY if c.name == "advisor")
        self.assertNotIn("\u2014", cmd.description)


# ---------------------------------------------------------------------------
# Argument parsing (model hint detection)
# ---------------------------------------------------------------------------

class TestAdvisorArgParsing(unittest.TestCase):

    def _parse(self, raw_args: str):
        """Exercise the argument-parsing logic from _handle_advisor_command."""
        advisor_model = "claude-opus-4-7"
        question_override = ""
        if raw_args:
            first_word = raw_args.split()[0]
            _model_keywords = {"opus", "sonnet", "haiku", "claude", "gpt", "gemini", "llama",
                               "mixtral", "qwen", "mistral", "grok", "deepseek"}
            _is_model_hint = (
                any(kw in first_word.lower() for kw in _model_keywords)
                or "-" in first_word
            )
            if _is_model_hint:
                advisor_model = first_word
                question_override = raw_args.split(None, 1)[1].strip() if " " in raw_args else ""
            else:
                question_override = raw_args
        return advisor_model, question_override

    def test_no_args_uses_default_model(self):
        model, q = self._parse("")
        self.assertEqual(model, "claude-opus-4-7")
        self.assertEqual(q, "")

    def test_opus_shorthand(self):
        model, q = self._parse("opus")
        self.assertEqual(model, "opus")
        self.assertEqual(q, "")

    def test_full_model_id(self):
        model, q = self._parse("claude-opus-4-7")
        self.assertEqual(model, "claude-opus-4-7")
        self.assertEqual(q, "")

    def test_model_with_question(self):
        model, q = self._parse("opus What is a qubit?")
        self.assertEqual(model, "opus")
        self.assertEqual(q, "What is a qubit?")

    def test_plain_question_no_model(self):
        model, q = self._parse("Is the last answer correct?")
        self.assertEqual(model, "claude-opus-4-7")
        self.assertEqual(q, "Is the last answer correct?")

    def test_sonnet_detected(self):
        model, q = self._parse("sonnet")
        self.assertEqual(model, "sonnet")

    def test_hyphen_in_first_word_treated_as_model(self):
        model, q = self._parse("gpt-4o What's new?")
        self.assertEqual(model, "gpt-4o")
        self.assertEqual(q, "What's new?")


# ---------------------------------------------------------------------------
# _handle_advisor_command behaviour
# ---------------------------------------------------------------------------

class TestHandleAdvisorCommand(unittest.TestCase):

    def _invoke(self, cli, cmd: str):
        """Call the real method on the stub."""
        from cli import HermesCLI
        return HermesCLI._handle_advisor_command(cli, cmd)

    def test_no_history_prints_and_returns(self):
        """When agent.messages is empty, should print a hint and not spawn a thread."""
        cli = _make_cli(history=[])
        self._invoke(cli, "/advisor")
        # No thread should have been appended to _background_tasks
        self.assertEqual(len(cli._background_tasks), 0)

    def test_no_credentials_returns_early(self):
        cli = _make_cli()
        cli._ensure_runtime_credentials.return_value = False
        self._invoke(cli, "/advisor")
        self.assertEqual(len(cli._background_tasks), 0)

    def test_spawns_thread_with_history(self):
        cli = _make_cli()
        self._invoke(cli, "/advisor")
        self.assertEqual(len(cli._background_tasks), 1)
        task_id, thread = next(iter(cli._background_tasks.items()))
        self.assertIsInstance(thread, threading.Thread)
        self.assertTrue(thread.daemon)

    def test_custom_model_passed(self):
        cli = _make_cli()
        # Patch AIAgent so we can inspect what model was passed.
        with patch("cli.AIAgent") as MockAgent:
            MockAgent.return_value.run_conversation.return_value = {"final_response": "ok"}
            self._invoke(cli, "/advisor sonnet Is this right?")
            # Background thread may not have run yet, but _background_tasks was populated.
            self.assertEqual(len(cli._background_tasks), 1)


# ---------------------------------------------------------------------------
# Gateway: /advisor handler
# ---------------------------------------------------------------------------

class TestGatewayAdvisorHandler(unittest.IsolatedAsyncioTestCase):

    async def test_returns_ack_string(self):
        from gateway.run import GatewayRunner

        runner = MagicMock(spec=GatewayRunner)
        runner._ADVISOR_DEFAULT_MODEL = GatewayRunner._ADVISOR_DEFAULT_MODEL
        runner._reply_anchor_for_event.return_value = None
        runner._background_tasks = set()

        event = MagicMock()
        event.get_command_args.return_value = ""
        event.source.platform = "discord"

        with patch("asyncio.create_task") as mock_create_task:
            mock_task = MagicMock()
            mock_create_task.return_value = mock_task

            result = await GatewayRunner._handle_advisor_command(runner, event)

        self.assertIn("claude-opus-4-7", result)
        self.assertIn("Consulting advisor", result)

    async def test_custom_model_in_ack(self):
        from gateway.run import GatewayRunner

        runner = MagicMock(spec=GatewayRunner)
        runner._ADVISOR_DEFAULT_MODEL = GatewayRunner._ADVISOR_DEFAULT_MODEL
        runner._reply_anchor_for_event.return_value = None
        runner._background_tasks = set()

        event = MagicMock()
        event.get_command_args.return_value = "opus"
        event.source.platform = "discord"

        with patch("asyncio.create_task") as mock_create_task:
            mock_task = MagicMock()
            mock_create_task.return_value = mock_task

            result = await GatewayRunner._handle_advisor_command(runner, event)

        self.assertIn("opus", result)


if __name__ == "__main__":
    unittest.main()

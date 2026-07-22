"""Unit tests for the Cursor Agent OpenAI-compat shim."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.cursor_agent_client import CURSOR_CURATED_MODELS, CursorAgentClient, list_cursor_model_ids


class CursorAgentClientTests(unittest.TestCase):
    def test_list_cursor_model_ids_falls_back_to_curated(self) -> None:
        with patch.dict("os.environ", {"CURSOR_API_KEY": ""}, clear=False):
            models = list_cursor_model_ids(api_key="")
        self.assertEqual(models[:3], ["auto", "default", "composer-2.5"])
        self.assertEqual(models, list(CURSOR_CURATED_MODELS))

    def test_list_cursor_model_ids_merges_live_catalog(self) -> None:
        live = [
            SimpleNamespace(id="auto"),
            SimpleNamespace(id="composer-2.5"),
            SimpleNamespace(id="grok-4.5"),
        ]
        with patch("cursor_sdk.Cursor") as mock_cursor:
            mock_cursor.models.list.return_value = live
            models = list_cursor_model_ids(api_key="test-key")
        self.assertEqual(models[0], "auto")
        self.assertIn("composer-2.5", models)
        self.assertIn("grok-4.5", models)

    def test_missing_api_key_raises(self) -> None:
        client = CursorAgentClient(api_key="")
        with self.assertRaises(RuntimeError) as ctx:
            client._create_chat_completion(
                model="auto",
                messages=[{"role": "user", "content": "hi"}],
            )
        self.assertIn("CURSOR_API_KEY", str(ctx.exception))

    def test_prompt_maps_to_openai_completion(self) -> None:
        client = CursorAgentClient(api_key="test-key", cwd="/tmp")
        fake_result = SimpleNamespace(status="finished", result="Hello from Cursor")

        with patch("agent.cursor_agent_client._import_cursor_sdk") as mock_import:
            Agent = MagicMock()
            Agent.prompt = MagicMock(return_value=fake_result)
            AgentOptions = MagicMock(return_value=SimpleNamespace())
            LocalAgentOptions = MagicMock(return_value=SimpleNamespace(cwd="/tmp"))
            mock_import.return_value = (Agent, AgentOptions, LocalAgentOptions)

            response = client._create_chat_completion(
                model="auto",
                messages=[{"role": "user", "content": "hi"}],
            )

        self.assertEqual(response.choices[0].message.content, "Hello from Cursor")
        self.assertEqual(response.choices[0].finish_reason, "stop")
        self.assertGreater(response.usage.total_tokens, 0)
        Agent.prompt.assert_called_once()
        args, _kwargs = Agent.prompt.call_args
        self.assertEqual(len(args), 2)  # prompt text + AgentOptions
        self.assertIn("hi", str(args[0]))
        AgentOptions.assert_called_once()
        opt_kwargs = AgentOptions.call_args.kwargs
        self.assertEqual(opt_kwargs["api_key"], "test-key")
        self.assertEqual(opt_kwargs["model"], "auto")


if __name__ == "__main__":
    unittest.main()

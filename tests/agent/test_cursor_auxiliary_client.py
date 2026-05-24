"""Auxiliary routing for the Cursor SDK provider."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class CursorAuxiliaryClientTests(unittest.TestCase):
    def test_resolve_provider_client_cursor_returns_sdk_shim(self):
        """Cursor must not fall through to OpenAI HTTP with base_url cursor://sdk."""
        from agent.auxiliary_client import resolve_provider_client
        from agent.cursor_auxiliary_client import CursorAuxiliaryClient

        with patch.dict(os.environ, {"CURSOR_API_KEY": "cursor_test_key"}, clear=False):
            client, model = resolve_provider_client("cursor", model="composer-2.5")
        self.assertIsNotNone(client)
        self.assertIsInstance(client, CursorAuxiliaryClient)
        self.assertEqual(model, "composer-2.5")
        self.assertEqual(str(client.base_url), "cursor://sdk")

    def test_resolve_provider_client_cursor_without_key_returns_none(self):
        from agent.auxiliary_client import resolve_provider_client

        with patch(
            "agent.cursor_auxiliary_client.build_cursor_auxiliary_client",
            return_value=(None, None),
        ):
            client, model = resolve_provider_client("cursor", model="composer-2.5")
        self.assertIsNone(client)
        self.assertIsNone(model)

    def test_cursor_completions_adapter_maps_prompt_result(self):
        from agent.cursor_auxiliary_client import CursorAuxiliaryClient

        run_result = SimpleNamespace(
            status="finished", result='{"title": "T", "body": "B"}'
        )
        with patch.dict(os.environ, {"CURSOR_API_KEY": "cursor_test_key"}, clear=False), patch(
            "cursor_sdk.Agent.prompt",
            return_value=run_result,
        ) as prompt_mock, patch(
            "agent.cursor_auxiliary_client.get_cursor_sdk_client",
            return_value=MagicMock(),
        ):
            client = CursorAuxiliaryClient(model="composer-2.5")
            resp = client.chat.completions.create(
                model="composer-2.5",
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "user"},
                ],
                timeout=30,
            )

        self.assertEqual(
            resp.choices[0].message.content, '{"title": "T", "body": "B"}'
        )
        prompt_mock.assert_called_once()
        prompt_text = prompt_mock.call_args[0][0]
        self.assertIn("sys", prompt_text)
        self.assertIn("user", prompt_text)

    def test_messages_to_prompt_flattens_roles(self):
        from agent.cursor_auxiliary_client import _messages_to_prompt

        text = _messages_to_prompt(
            [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hello"},
            ]
        )
        self.assertIn("Be concise.", text)
        self.assertIn("Hello", text)

    def test_cursor_stream_logger_emits_assistant_deltas(self):
        from hermes_cli.kanban_worker_log import CursorStreamLogger

        chunks: list[str] = []
        logger = CursorStreamLogger(chunks.append)
        logger.handle(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Hello"}]},
            }
        )
        logger.handle(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Hello world"}]},
            }
        )
        self.assertEqual("".join(chunks), "Hello world")

    def test_prepare_reload_only_never_resets_clients(self):
        from agent import cursor_auxiliary_client as cac

        with patch.object(cac, "reset_cursor_sdk_client") as reset_mock, patch(
            "agent.auxiliary_client.evict_cached_auxiliary_clients"
        ) as evict_mock:
            cac.prepare_cursor_auxiliary_credentials(reload_only=True)
        reset_mock.assert_not_called()
        evict_mock.assert_not_called()

    def test_per_task_sdk_clients_are_isolated(self):
        from agent import cursor_auxiliary_client as cac

        created = []

        def _fake_from_bridge(**kwargs):
            client = MagicMock(name=f"client-{len(created)}")
            created.append(client)
            return client

        with patch.object(cac, "_client_from_shared_bridge", side_effect=_fake_from_bridge):
            a = cac.get_cursor_sdk_client(kanban_isolation_key="t_a")
            b = cac.get_cursor_sdk_client(kanban_isolation_key="t_b")
        self.assertIsNot(a, b)
        self.assertEqual(len(created), 2)
        cac.release_cursor_sdk_client("t_a")
        a.close.assert_called_once()
        b.close.assert_not_called()

    def test_effective_cursor_auxiliary_timeout_floor(self, monkeypatch):
        from agent.cursor_auxiliary_client import effective_cursor_auxiliary_timeout

        monkeypatch.delenv("HERMES_KANBAN_CURSOR_AUX_TIMEOUT", raising=False)
        # Reload module constant after env change
        import agent.cursor_auxiliary_client as cac

        monkeypatch.setattr(cac, "_CURSOR_KANBAN_AUX_TIMEOUT_FLOOR", 600.0)
        self.assertEqual(cac.effective_cursor_auxiliary_timeout(120), 600.0)
        self.assertEqual(cac.effective_cursor_auxiliary_timeout(900), 900.0)

    def test_prepare_skips_client_reset_while_other_aux_ops_in_flight(self):
        from agent import cursor_auxiliary_client as cac

        sentinel = object()
        with patch.object(cac, "reset_cursor_sdk_client") as reset_mock, patch(
            "agent.auxiliary_client.evict_cached_auxiliary_clients"
        ) as evict_mock, patch.object(
            cac, "_active_auxiliary_ops", 2
        ):
            cac.prepare_cursor_auxiliary_credentials()
        reset_mock.assert_not_called()
        evict_mock.assert_not_called()

        with patch.object(cac, "reset_cursor_sdk_client") as reset_mock, patch(
            "agent.auxiliary_client.evict_cached_auxiliary_clients"
        ) as evict_mock, patch.object(
            cac, "_active_auxiliary_ops", 1
        ):
            cac.prepare_cursor_auxiliary_credentials()
        reset_mock.assert_called_once()
        evict_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()

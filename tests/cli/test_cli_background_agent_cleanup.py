"""Regression tests: /background tasks must tear down their AIAgent (#50197).

_handle_background_command's run_background() worker spawns a fresh AIAgent
per invocation. Its finally block used to reset callbacks and clear the
spinner/task-tracking bookkeeping but never called shutdown_memory_provider()
or close() on the agent itself, leaking a terminal sandbox, browser daemon,
httpx client, and memory-provider session per /background task.
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cli as cli_module
from cli import HermesCLI


def _make_background_cli_stub():
    cli = HermesCLI.__new__(HermesCLI)
    cli._approval_state = None
    cli._approval_deadline = 0
    cli._approval_lock = threading.Lock()
    cli._sudo_state = None
    cli._sudo_deadline = 0
    cli._modal_input_snapshot = None
    cli._invalidate = MagicMock()
    cli._app = None
    cli._background_task_counter = 0
    cli._background_tasks = {}
    cli._ensure_runtime_credentials = MagicMock(return_value=True)
    cli._resolve_turn_agent_config = MagicMock(return_value={
        "model": "test-model",
        "runtime": {
            "api_key": "test-key",
            "base_url": "https://example.test/v1",
            "provider": "test",
            "api_mode": "chat_completions",
        },
        "request_overrides": None,
    })
    cli.max_turns = 90
    cli.enabled_toolsets = []
    cli._session_db = None
    cli.reasoning_config = {}
    cli.service_tier = None
    cli._providers_only = None
    cli._providers_ignore = None
    cli._providers_order = None
    cli._provider_sort = None
    cli._provider_require_params = None
    cli._provider_data_collection = None
    cli._openrouter_min_coding_score = None
    cli._fallback_model = None
    cli._agent_running = False
    cli._spinner_text = ""
    cli.bell_on_complete = False
    cli.final_response_markdown = "strip"
    return cli


class TestBackgroundCommandAgentCleanup:
    def _run_and_join(self, cli, cmd):
        cli._handle_background_command(cmd)
        for thread in list(cli._background_tasks.values()):
            thread.join(timeout=10)

    def test_agent_shutdown_and_close_called_on_success(self):
        cli = _make_background_cli_stub()
        calls = []

        class FakeAgent:
            def __init__(self, **_kwargs):
                self._print_fn = None
                self.thinking_callback = None

            def run_conversation(self, **_kwargs):
                return {"final_response": "done", "messages": []}

            def shutdown_memory_provider(self):
                calls.append("shutdown_memory_provider")

            def close(self):
                calls.append("close")

        with patch.object(cli_module, "AIAgent", FakeAgent), \
             patch.object(cli_module, "_cprint"), \
             patch.object(cli_module, "ChatConsole") as chat_console, \
             patch("agent.auxiliary_client.cleanup_stale_async_clients") as mock_cleanup:
            chat_console.return_value.print = MagicMock()
            self._run_and_join(cli, "/background do something")

        assert calls == ["shutdown_memory_provider", "close"]
        mock_cleanup.assert_called_once_with()
        assert not cli._background_tasks

    def test_agent_shutdown_and_close_called_on_failure(self):
        """Cleanup must still run when run_conversation() raises."""
        cli = _make_background_cli_stub()
        calls = []

        class FakeAgent:
            def __init__(self, **_kwargs):
                self._print_fn = None
                self.thinking_callback = None

            def run_conversation(self, **_kwargs):
                raise RuntimeError("boom")

            def shutdown_memory_provider(self):
                calls.append("shutdown_memory_provider")

            def close(self):
                calls.append("close")

        with patch.object(cli_module, "AIAgent", FakeAgent), \
             patch.object(cli_module, "_cprint"):
            self._run_and_join(cli, "/background do something")

        assert calls == ["shutdown_memory_provider", "close"]

    def test_no_crash_when_agent_construction_fails(self):
        """bg_agent stays None if AIAgent() itself raises; finally must not
        try to close a nonexistent agent."""
        cli = _make_background_cli_stub()

        with patch.object(cli_module, "AIAgent", side_effect=RuntimeError("no creds")), \
             patch.object(cli_module, "_cprint"):
            self._run_and_join(cli, "/background do something")

        assert not cli._background_tasks

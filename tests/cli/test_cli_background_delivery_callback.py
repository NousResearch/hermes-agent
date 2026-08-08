"""Regression tests for the /background delivery hook (``_background_complete_callback``).

Background
----------
``_handle_background_command`` spawns the background task in a daemon thread
and returns immediately. The final response used to be printed to the console
only — visible in a classic terminal, but lost in the TUI/Desktop route where
the slash worker's stdout is a protocol channel nobody renders. The fix adds an
optional ``_background_complete_callback(task_id, text)`` hook that the TUI
slash-worker sets; the classic CLI keeps ``None`` and the console behaviour.

These tests exercise the hook without running a real model: ``AIAgent`` is
replaced by a fake whose ``run_conversation`` returns a canned response.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch


def _make_cli():
    """Create a HermesCLI instance with prompt_toolkit stubbed out."""
    _clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        "display": {"compact": False, "tool_progress": "all"},
        "agent": {},
        "terminal": {"env_type": "local"},
    }
    clean_env = {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}
    prompt_toolkit_stubs = {
        "prompt_toolkit": MagicMock(),
        "prompt_toolkit.history": MagicMock(),
        "prompt_toolkit.styles": MagicMock(),
        "prompt_toolkit.patch_stdout": MagicMock(),
        "prompt_toolkit.application": MagicMock(),
        "prompt_toolkit.layout": MagicMock(),
        "prompt_toolkit.layout.processors": MagicMock(),
        "prompt_toolkit.filters": MagicMock(),
        "prompt_toolkit.layout.dimension": MagicMock(),
        "prompt_toolkit.layout.menus": MagicMock(),
        "prompt_toolkit.widgets": MagicMock(),
        "prompt_toolkit.key_binding": MagicMock(),
        "prompt_toolkit.completion": MagicMock(),
        "prompt_toolkit.formatted_text": MagicMock(),
        "prompt_toolkit.auto_suggest": MagicMock(),
    }
    with patch.dict(sys.modules, prompt_toolkit_stubs), patch.dict(
        "os.environ", clean_env, clear=False
    ):
        import cli as _cli_mod

        _cli_mod = importlib.reload(_cli_mod)
        with patch.object(_cli_mod, "get_tool_definitions", return_value=[]), patch.dict(
            _cli_mod.__dict__, {"CLI_CONFIG": _clean_config}
        ):
            return _cli_mod.HermesCLI()


class TestBackgroundCompleteCallback:
    def test_callback_receives_final_response(self):
        import cli as cli_module

        seen = {}

        def fake_callback(task_id, text):
            seen["task_id"] = task_id
            seen["text"] = text

        cli = _make_cli()
        cli._background_complete_callback = fake_callback

        class FakeAgent:
            def __init__(self, **kwargs):
                self._print_fn = None
                self.thinking_callback = None

            def run_conversation(self, **kwargs):
                return {"final_response": "het onderzoeksresultaat", "messages": [], "completed": True}

        with patch.object(cli_module, "AIAgent", FakeAgent), \
             patch.object(cli_module, "_cprint"), \
             patch.object(cli_module, "ChatConsole") as chat_console, \
             patch.object(cli, "_ensure_runtime_credentials", return_value=True):
            chat_console.return_value.print = MagicMock()
            cli._handle_background_command("/btw onderzoek de specs")

            for _thread in list(cli._background_tasks.values()):
                _thread.join(timeout=10)

        assert seen["text"] == "het onderzoeksresultaat"
        assert seen["task_id"].startswith("bg_")
        assert not cli._background_tasks

    def test_callback_receives_error(self):
        import cli as cli_module

        seen = {}
        cli = _make_cli()
        cli._background_complete_callback = lambda task_id, text: seen.update(task_id=task_id, text=text)

        class ExplodingAgent:
            def __init__(self, **kwargs):
                self._print_fn = None
                self.thinking_callback = None

            def run_conversation(self, **kwargs):
                raise RuntimeError("model kapot")

        with patch.object(cli_module, "AIAgent", ExplodingAgent), \
             patch.object(cli_module, "_cprint"), \
             patch.object(cli_module, "ChatConsole") as chat_console, \
             patch.object(cli, "_ensure_runtime_credentials", return_value=True):
            chat_console.return_value.print = MagicMock()
            cli._handle_background_command("/btw dit gaat mis")

            for _thread in list(cli._background_tasks.values()):
                _thread.join(timeout=10)

        assert seen["text"].startswith("Error:")
        assert seen["task_id"].startswith("bg_")

    def test_without_callback_console_still_works(self):
        """The classic CLI path (callback None) must keep working."""
        import cli as cli_module

        cli = _make_cli()
        assert cli._background_complete_callback is None

        class FakeAgent:
            def __init__(self, **kwargs):
                self._print_fn = None
                self.thinking_callback = None

            def run_conversation(self, **kwargs):
                return {"final_response": "ok", "messages": [], "completed": True}

        with patch.object(cli_module, "AIAgent", FakeAgent), \
             patch.object(cli_module, "_cprint"), \
             patch.object(cli_module, "ChatConsole") as chat_console, \
             patch.object(cli, "_ensure_runtime_credentials", return_value=True):
            chat_console.return_value.print = MagicMock()
            cli._handle_background_command("/btw gewone cli")

            for _thread in list(cli._background_tasks.values()):
                _thread.join(timeout=10)

        assert not cli._background_tasks
        # No callback registered → nothing to assert beyond the thread finishing.

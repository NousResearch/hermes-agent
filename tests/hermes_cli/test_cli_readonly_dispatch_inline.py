"""Tests for _should_handle_readonly_dispatch_inline — #75835."""

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest


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


class TestReadonlyDispatchDetector:
    """_should_handle_readonly_dispatch_inline gates inline dispatch."""

    # ── Commands that should dispatch inline ──────────────────────────

    @pytest.mark.parametrize("text", [
        "/status",
        "/agents",
        "/tasks",       # alias for /agents
        "/context",
        "/ctx",         # alias for /context
        "/profile",
        "/version",
        "/v",           # alias for /version
        "/help",
        "/egress",
    ])
    def test_detects_readonly_dispatch_commands(self, text):
        cli = _make_cli()
        cli._agent_running = True
        assert cli._should_handle_readonly_dispatch_inline(text) is True

    # ── Commands that must NOT dispatch inline ────────────────────────

    @pytest.mark.parametrize("text", [
        "/queue hello",     # busy_handler=queue — mutates state
        "/goal do x",       # busy_handler=goal — mutates state
        "/subgoal check",   # modifies goal criteria
        "/yolo",            # changes security mode
        "/update",          # modifies installation
        "/model gpt-5",     # handled by dedicated inline handler
        "/steer text",      # handled by dedicated inline handler
        "/background task", # handled by dedicated inline handler
    ])
    def test_rejects_state_mutating_commands(self, text):
        cli = _make_cli()
        assert cli._should_handle_readonly_dispatch_inline(text) is False

    # ── Edge cases ─────────────────────────────────────────────────────

    def test_ignores_non_slash_input(self):
        cli = _make_cli()
        assert cli._should_handle_readonly_dispatch_inline("status") is False
        assert cli._should_handle_readonly_dispatch_inline("") is False

    def test_ignores_with_attached_images(self):
        cli = _make_cli()
        assert cli._should_handle_readonly_dispatch_inline("/status", has_images=True) is False

    def test_ignores_unknown_commands(self):
        cli = _make_cli()
        assert cli._should_handle_readonly_dispatch_inline("/nonexistent") is False

    def test_ignores_when_agent_idle(self):
        """Read-only inline dispatch must only fire while the agent is running.
        When idle, commands follow the normal process_loop path."""
        cli = _make_cli()
        cli._agent_running = False
        assert cli._should_handle_readonly_dispatch_inline("/status") is False
        assert cli._should_handle_readonly_dispatch_inline("/egress") is False

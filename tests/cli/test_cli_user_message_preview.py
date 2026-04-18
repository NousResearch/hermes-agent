import importlib
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


_cli_mod = None


def _make_cli(user_message_preview=None):
    global _cli_mod
    clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        "display": {
            "compact": False,
            "tool_progress": "all",
            "user_message_preview": user_message_preview or {"first_lines": 2, "last_lines": 2},
        },
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
    with patch.dict(sys.modules, prompt_toolkit_stubs), patch.dict("os.environ", clean_env, clear=False):
        import cli as mod

        mod = importlib.reload(mod)
        _cli_mod = mod
        with patch.object(mod, "get_tool_definitions", return_value=[]), patch.dict(mod.__dict__, {"CLI_CONFIG": clean_config}):
            return mod.HermesCLI()


class TestSubmittedUserMessagePreview:
    def test_default_preview_shows_first_two_lines_and_last_two_lines(self):
        cli = _make_cli()

        rendered = cli._format_submitted_user_message_preview(
            "line1\nline2\nline3\nline4\nline5\nline6"
        )

        assert "line1" in rendered
        assert "line2" in rendered
        assert "line5" in rendered
        assert "line6" in rendered
        assert "line3" not in rendered
        assert "line4" not in rendered
        assert "(+2 more lines)" in rendered

    def test_preview_can_hide_last_lines(self):
        cli = _make_cli({"first_lines": 2, "last_lines": 0})

        rendered = cli._format_submitted_user_message_preview(
            "line1\nline2\nline3\nline4\nline5\nline6"
        )

        assert "line1" in rendered
        assert "line2" in rendered
        assert "line5" not in rendered
        assert "line6" not in rendered
        assert "(+4 more lines)" in rendered

    def test_invalid_first_lines_value_falls_back_to_one(self):
        cli = _make_cli({"first_lines": 0, "last_lines": 2})

        rendered = cli._format_submitted_user_message_preview("line1\nline2\nline3\nline4")

        assert "line1" in rendered
        assert "line3" in rendered
        assert "line4" in rendered
        assert "(+1 more line)" in rendered

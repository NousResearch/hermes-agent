from types import SimpleNamespace
from unittest.mock import patch

import cli
from cli import HermesCLI


class DummyConsole:
    def print(self, *args, **kwargs):
        pass

    def clear(self):
        pass


def _make_cli():
    clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        "display": {"compact": True, "tool_progress": "all"},
        "agent": {},
        "terminal": {"env_type": "local"},
    }
    with patch("cli.get_tool_definitions", return_value=[]), \
         patch.dict(cli.__dict__, {"CLI_CONFIG": clean_config}):
        shell = HermesCLI(compact=True, max_turns=1)
        shell.console = DummyConsole()
        shell.agent = SimpleNamespace(context_compressor=None)
        shell.session_id = "session-123"
        shell._session_db = None
        return shell


def test_stats_command_dispatches_to_show_stats():
    shell = _make_cli()

    with patch.object(shell, "_show_stats") as show_stats:
        assert shell.process_command("/stats") is True

    show_stats.assert_called_once_with()

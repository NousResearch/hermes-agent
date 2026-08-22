"""Runtime parity guard between COMMAND_REGISTRY and the classic CLI dispatcher.

``COMMAND_REGISTRY`` (hermes_cli/commands.py) is the single source of truth for
``/help``, tab completion, and the gateway command menus. The classic CLI
dispatches slash commands from ``HermesCLI.process_command``. Nothing keeps the
two in sync, so a command can be registered — and therefore advertised in
``/help`` and offered by autocomplete — while the CLI has no branch for it. The
user types it and gets "Unknown command".

This test walks the registry and exercises every CLI-reachable command through
``process_command`` at runtime, asserting it never falls through to the
"Unknown command" path. No source inspection: a command is either dispatched
(its handler runs) or it is not, and the spy on ``cli._cprint`` sees the
difference.
"""

from __future__ import annotations

from datetime import datetime
import threading
from unittest.mock import MagicMock, patch

import pytest

from cli import HermesCLI
from hermes_cli.commands import COMMAND_REGISTRY

# Registered and CLI-reachable today, but process_command has no dispatch
# branch. Each entry is a live bug tracked in #74594; drop it when the
# dispatch lands and this test will hold the line from then on.
KNOWN_MISSING_DISPATCH = {
    # https://github.com/NousResearch/hermes-agent/issues/74594
    "whoami",
}


def _cli_reachable(cmd) -> bool:
    """True when the classic CLI is expected to handle ``cmd`` itself."""
    if cmd.gateway_only:
        return False
    if cmd.name in KNOWN_MISSING_DISPATCH:
        return False
    return True


def _make_cli() -> HermesCLI:
    """Lightweight HermesCLI instance for dispatch tests (no REPL init)."""
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.config = {}
    cli_obj.console = MagicMock()
    cli_obj.agent = None
    cli_obj.conversation_history = []
    cli_obj.session_id = None
    cli_obj._pending_input = MagicMock()
    cli_obj._app = None
    cli_obj._pending_resume_sessions = None
    cli_obj._delete_session_on_exit = False
    cli_obj._compact = False
    cli_obj.enabled_toolsets = []
    cli_obj._session_db = MagicMock()
    cli_obj.api_key = "test-key"
    cli_obj.model = "test-model"
    cli_obj.provider = "test-provider"
    cli_obj.base_url = "https://test"
    cli_obj.session_start = datetime.now()
    cli_obj.personalities = {}
    cli_obj.reasoning_config = {}
    cli_obj.busy_input_mode = "reject"
    cli_obj.tool_progress_mode = "classic"
    cli_obj._voice_mode = False
    cli_obj._battery_visible = False
    cli_obj._status_bar_visible = False
    cli_obj.max_turns = 10
    cli_obj._modal_input_snapshot = None
    cli_obj.show_reasoning = False
    cli_obj._voice_lock = threading.Lock()
    cli_obj.verbose = False
    cli_obj._voice_tts = False
    return cli_obj


CLI_COMMANDS = [c for c in COMMAND_REGISTRY if _cli_reachable(c)]


@pytest.mark.parametrize("cmd", CLI_COMMANDS, ids=lambda c: c.name)
def test_registered_cli_command_has_runtime_dispatch(cmd):
    """A registered CLI command must never fall through to "Unknown command"."""
    cli_obj = _make_cli()
    with (
        patch("cli._cprint") as mock_cprint,
        patch("builtins.input", return_value=""),
    ):
        result = cli_obj.process_command(f"/{cmd.name}")

    printed = " ".join(str(c) for c in mock_cprint.call_args_list)
    assert "Unknown command" not in printed
    # /quit and its alias return False to signal REPL exit; every other
    # registered CLI command must keep the REPL alive.
    expected = cmd.name not in {"quit", "exit"}
    assert result is expected


def test_unknown_command_is_detected():
    """Negative control: the guard actually fires for an unregistered command."""
    cli_obj = _make_cli()
    with patch("cli._cprint") as mock_cprint:
        cli_obj.process_command("/definitely-not-a-registered-command")

    printed = " ".join(str(c) for c in mock_cprint.call_args_list)
    assert "Unknown command" in printed


ALIASED_COMMANDS = [
    (cmd.name, alias)
    for cmd in COMMAND_REGISTRY
    if _cli_reachable(cmd)
    for alias in cmd.aliases
]


@pytest.mark.parametrize("cmd,alias", ALIASED_COMMANDS, ids=lambda v: v)
def test_registered_cli_alias_has_runtime_dispatch(cmd, alias):
    """Aliases resolve through the registry and must dispatch, not go Unknown."""
    cli_obj = _make_cli()
    with (
        patch("cli._cprint") as mock_cprint,
        patch("builtins.input", return_value=""),
    ):
        result = cli_obj.process_command(f"/{alias}")

    printed = " ".join(str(c) for c in mock_cprint.call_args_list)
    assert "Unknown command" not in printed
    expected = cmd not in {"quit", "exit"}
    assert result is expected

"""Cross-surface contract for the persistent /approvals mode command."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import yaml

from cli import HermesCLI
from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, SUBCOMMANDS, gateway_help_lines, resolve_command
from hermes_cli.commands_completion import SlashCommandCompleter
from hermes_cli.commands_platforms import telegram_bot_commands
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document


def _completions(text: str) -> set[str]:
    return {
        item.text
        for item in SlashCommandCompleter().get_completions(
            Document(text=text), CompleteEvent(completion_requested=True)
        )
    }


def test_approvals_registry_drives_help_menu_and_autocomplete():
    command = resolve_command("approvals")
    assert command is not None
    assert command.category == "Configuration"
    assert command.args_hint == "[manual|smart|off]"
    assert SUBCOMMANDS["/approvals"] == ["manual", "smart", "off"]
    assert "approvals" in GATEWAY_KNOWN_COMMANDS
    assert any("/approvals" in line for line in gateway_help_lines())
    assert "approvals" in {name for name, _ in telegram_bot_commands()}
    assert _completions("/approvals ") == {"manual", "smart", "off"}


def _isolate_config(monkeypatch, home):
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(home / "missing-managed"))
    from hermes_cli import managed_scope
    from hermes_cli.config import _LOAD_CONFIG_CACHE, _RAW_CONFIG_CACHE

    _LOAD_CONFIG_CACHE.clear()
    _RAW_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()






def test_shared_command_refuses_managed_mode_override(tmp_path, monkeypatch):
    from hermes_cli import managed_scope
    from hermes_cli.approval_mode import run_approval_mode_command

    home = tmp_path / "home"
    managed = tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    (managed / "config.yaml").write_text("approvals:\n  mode: manual\n", encoding="utf-8")
    managed_scope.invalidate_managed_cache()

    result = run_approval_mode_command("off")

    assert result.ok is False
    assert result.mode == "manual"
    assert result.changed is False
    assert "managed" in result.message.lower()
    assert not (home / "config.yaml").exists()


def test_unbound_approvals_handler_refused(tmp_path, monkeypatch):
    """An agent or adversary calling CLICommandsMixin._handle_approvals_command
    directly (unbound, outside process_command dispatch) must NOT persist changes
    to security policy (#104697 P1-A)."""
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    config_file = home / "config.yaml"
    config_file.write_text("approvals:\n  mode: smart\n", encoding="utf-8")
    initial_bytes = config_file.read_bytes()

    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli.config import _LOAD_CONFIG_CACHE, _RAW_CONFIG_CACHE
    _LOAD_CONFIG_CACHE.clear()
    _RAW_CONFIG_CACHE.clear()

    from hermes_cli.cli_commands_mixin import CLICommandsMixin

    # Plain context: no process_command frame on the stack.
    # Must cleanly report error or exit without mutating config.yaml.
    try:
        CLICommandsMixin._handle_approvals_command(object(), "/approvals off")
    except (SystemExit, RuntimeError):
        pass

    assert config_file.read_bytes() == initial_bytes
    cfg = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert (cfg.get("approvals") or {}).get("mode") == "smart"


def test_sanctioned_repl_process_command_persists(tmp_path, monkeypatch):
    """The sanctioned REPL path (via HermesCLI.process_command) carries the
    real process_command frame from cli.py, satisfying the stamp check and
    persisting the change (#104697)."""
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    config_file = home / "config.yaml"
    config_file.write_text("approvals:\n  mode: smart\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli.config import _LOAD_CONFIG_CACHE, _RAW_CONFIG_CACHE
    _LOAD_CONFIG_CACHE.clear()
    _RAW_CONFIG_CACHE.clear()

    cli = HermesCLI.__new__(HermesCLI)
    cli.config = {}
    cli.console = MagicMock()
    cli.agent = None
    cli.conversation_history = []
    cli.session_id = "test-session"

    result = cli.process_command("/approvals off")
    assert result is True

    cfg = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert (cfg.get("approvals") or {}).get("mode") == "off"








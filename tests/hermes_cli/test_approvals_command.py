"""Cross-surface contract for the persistent /approvals mode command."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import yaml

from cli import HermesCLI
from hermes_cli.commands import (
    GATEWAY_KNOWN_COMMANDS,
    SUBCOMMANDS,
    SlashCommandCompleter,
    gateway_help_lines,
    resolve_command,
    telegram_bot_commands,
)
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


def test_shared_approval_mode_command_reports_effective_default_without_writing(tmp_path, monkeypatch):
    from hermes_cli.approval_mode import run_approval_mode_command
    from tools.approval import _get_approval_mode

    _isolate_config(monkeypatch, tmp_path)
    result = run_approval_mode_command(None)

    assert result.ok is True
    assert result.mode == _get_approval_mode()
    assert result.changed is False
    assert result.mode in result.message
    assert not (tmp_path / "config.yaml").exists()


def test_shared_approval_mode_command_persists_profile_setting(tmp_path, monkeypatch):
    from hermes_cli.approval_mode import run_approval_mode_command

    _isolate_config(monkeypatch, tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text("model:\n  default: test-model\n", encoding="utf-8")

    result = run_approval_mode_command("off")

    assert result.ok is True
    assert result.mode == "off"
    assert result.changed is True
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["approvals"]["mode"] in {"off", False}
    assert "persistent" in result.message.lower()


def test_shared_approval_mode_command_rejects_unknown_mode_without_writing(tmp_path, monkeypatch):
    from hermes_cli.approval_mode import run_approval_mode_command

    _isolate_config(monkeypatch, tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text("approvals:\n  mode: smart\n", encoding="utf-8")
    before = path.read_bytes()

    result = run_approval_mode_command("auto")

    assert result.ok is False
    assert result.mode == "smart"
    assert path.read_bytes() == before
    assert "manual|smart|off" in result.message


def test_shared_status_matches_runtime_normalization_for_all_stored_shapes():
    from hermes_cli.approval_mode import run_approval_mode_command
    from tools.approval import _get_approval_mode

    for stored in (None, "manual", "smart", "off", False, True, "", "auto"):
        config = {"approvals": {}} if stored is None else {"approvals": {"mode": stored}}
        with patch("hermes_cli.config.load_config", return_value=config):
            result = run_approval_mode_command(None)
            assert result.mode == _get_approval_mode(), stored


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


def test_cli_dispatch_uses_shared_handler_without_rebuilding_agent():
    cli = HermesCLI.__new__(HermesCLI)
    cli.config = {}
    cli.console = MagicMock()
    cli.agent = object()
    cli._agent_running = False
    cli._pending_input = MagicMock()

    with patch.object(cli, "_handle_approvals_command", create=True) as handler:
        assert cli.process_command("/approvals manual") is True

    handler.assert_called_once_with("/approvals manual")
    assert cli.agent is not None


def test_cli_handler_prints_shared_result_and_preserves_agent_cache():
    cli = HermesCLI.__new__(HermesCLI)
    cached_agent = object()
    cli.agent = cached_agent
    result = SimpleNamespace(message="Approval mode: smart (persistent profile setting).")

    with (
        patch("hermes_cli.approval_mode.run_approval_mode_command", return_value=result) as run,
        patch("cli._cprint") as output,
    ):
        cli._handle_approvals_command("/approvals smart")

    run.assert_called_once_with("smart")
    output.assert_called_once_with("  Approval mode: smart (persistent profile setting).")
    assert cli.agent is cached_agent


def test_cli_live_process_command_persists_mode(tmp_path, monkeypatch):
    cli = HermesCLI.__new__(HermesCLI)
    cached_agent = object()
    cli.config = {}
    cli.console = MagicMock()
    cli.agent = cached_agent
    cli._agent_running = False
    cli._pending_input = MagicMock()
    _isolate_config(monkeypatch, tmp_path)

    with patch("cli._cprint") as output:
        assert cli.process_command("/approvals off") is True

    stored = yaml.safe_load((tmp_path / "config.yaml").read_text())["approvals"]["mode"]
    assert stored in {"off", False}
    assert "persistent profile setting" in output.call_args.args[0]
    assert cli.agent is cached_agent

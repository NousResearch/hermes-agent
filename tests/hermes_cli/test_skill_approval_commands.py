"""Skill approval scope commands at the shared slash-command seam."""

import pytest

from hermes_cli.write_approval_commands import handle_pending_subcommand


@pytest.mark.parametrize("command", ["approval", "mode"])
@pytest.mark.parametrize("scope", ["create", "all"])
def test_scope_command_enables_gate_and_selects_scope(command, scope):
    calls = []

    def setter(enabled, scope=None):
        calls.append((enabled, scope))

    output = handle_pending_subcommand("skills", [command, scope], set_mode_fn=setter)
    assert calls == [(True, scope)]
    assert "scope: " + scope in output


@pytest.mark.parametrize("arg", ["on", "off", "enabled", "disabled"])
def test_legacy_boolean_only_setters_remain_supported(arg):
    calls = []
    output = handle_pending_subcommand("skills", ["approval", arg], set_mode_fn=calls.append)
    assert calls == [arg in {"on", "enabled"}]
    assert "set to" in output


@pytest.mark.parametrize("subsystem, args", [
    ("memory", ["approval", "create"]), ("memory", ["mode", "all"]),
    ("skills", ["approval", "typo"]), ("skills", ["approval", "create", "all"]),
])
def test_invalid_values_do_not_call_setter(subsystem, args):
    calls = []
    assert "Invalid value" in handle_pending_subcommand(subsystem, args, set_mode_fn=calls.append)
    assert calls == []


@pytest.mark.parametrize("args", [["approval"], ["mode"], ["approval", "status"],
                                  ["approval", "current"], ["approval", "help"], []])
def test_help_explains_selected_scope_when_off(tmp_path, monkeypatch, args):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    path.write_text("skills:\n  write_approval: false\n  write_approval_mode: create\n")
    before = path.read_bytes()
    out = handle_pending_subcommand("skills", args)
    assert "scope: create" in out
    assert "on|off|create|all" in out
    assert "without changing the selected scope" in out
    assert "pending writes stay pending" in out
    assert path.read_bytes() == before


def test_registry_help_and_completion_advertise_scopes(tmp_path, monkeypatch):
    from hermes_cli.commands import resolve_command, gateway_help_lines
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("skills:\n  write_approval: true\n")
    command = resolve_command("skills")
    assert command is not None
    assert "approval" in command.args_hint and "create|all" in command.args_hint
    assert "mode" in command.subcommands and "mode" in (command.desktop_subcommands or ())
    assert any("create|all" in line for line in gateway_help_lines(["skills"]))

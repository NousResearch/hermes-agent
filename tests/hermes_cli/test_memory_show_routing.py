"""Routing tests for `/memory show` on the CLI.

Mirror of tests/gateway/test_memory_show_command.py for the CLI surface: a
leading `show` token reaches the readout helper, everything else falls through
to the write-approval handler. Readout formatting/parsing is covered in
tests/tools/test_memory_{readout,format}.py.
"""

from hermes_cli.cli_commands_mixin import CLICommandsMixin


class _FakeCLI(CLICommandsMixin):
    """Minimal stand-in: records whether the readout helper was reached."""

    def __init__(self):
        self.agent = None
        self.shown = "UNSET"

    def _handle_memory_show(self, target_args):
        self.shown = target_args


def test_show_routes_to_readout_helper():
    cli = _FakeCLI()
    cli._handle_memory_command("/memory show user")
    assert cli.shown == ["user"]


def test_bare_show_routes_with_empty_target():
    cli = _FakeCLI()
    cli._handle_memory_command("/memory show")
    assert cli.shown == []


def test_pending_falls_through_to_approval(monkeypatch, capsys):
    import hermes_cli.write_approval_commands as wac

    seen = {}

    def _fake_pending(subsystem, args, **kwargs):
        seen["args"] = args
        return "APPROVAL-FLOW"

    monkeypatch.setattr(wac, "handle_pending_subcommand", _fake_pending)

    cli = _FakeCLI()
    cli._handle_memory_command("/memory pending")

    assert cli.shown == "UNSET"  # readout NOT invoked
    assert seen["args"] == ["pending"]
    assert "APPROVAL-FLOW" in capsys.readouterr().out


def test_bare_memory_falls_through_to_approval(monkeypatch, capsys):
    import hermes_cli.write_approval_commands as wac

    seen = {}

    def _fake_pending(subsystem, args, **kwargs):
        seen["args"] = args
        return "APPROVAL-FLOW"

    monkeypatch.setattr(wac, "handle_pending_subcommand", _fake_pending)

    cli = _FakeCLI()
    cli._handle_memory_command("/memory")

    assert cli.shown == "UNSET"
    assert seen["args"] == []

"""Termux startup hints must not become unsupported execution options."""
import sys


def test_termux_fast_launch_preserves_gateway_execution_options(monkeypatch):
    from hermes_cli import main
    from hermes_cli.gateway_chat import validate_options

    monkeypatch.setenv("TERMUX_VERSION", "fixture")
    monkeypatch.delenv("HERMES_TERMUX_DISABLE_FAST_CLI", raising=False)
    monkeypatch.setattr(sys, "argv", ["hermes", "--cli"])
    launches = []

    def launch(args):
        validate_options(args)
        launches.append(args)

    monkeypatch.setattr(main, "cmd_chat", launch)
    assert main._try_termux_fast_cli_launch()
    assert len(launches) == 1
    assert not getattr(launches[0], "compact", False)

from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli.cli_commands_mixin import CLICommandsMixin


def test_plugin_slash_command_uses_cli_route_identity():
    captured = {}

    def invoke(handler, raw_args, **context):
        captured.update(context)
        return handler(raw_args)

    shell = SimpleNamespace(session_id="durable-session", console=SimpleNamespace(print=lambda *_a, **_k: None))
    with (
        patch("hermes_cli.plugins.get_plugin_command_handler", lambda _name: lambda _args: None),
        patch("hermes_cli.plugins.invoke_plugin_command", invoke),
        patch("hermes_cli.plugins.resolve_plugin_command_result", lambda value: value),
        patch("cli._cprint", lambda *_args, **_kwargs: None),
    ):
        CLICommandsMixin._run_plugin_slash_command(shell, "/control", "payload")

    assert captured == {
        "session_id": "durable-session",
        "session_key": "cli:durable-session",
        "platform": "cli",
    }

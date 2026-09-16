"""Container sessions must not expose the host checkpoint store through /rollback."""

from types import SimpleNamespace

from hermes_cli.cli_commands_mixin import CLICommandsMixin


def test_rollback_refuses_container_session_before_store_access(monkeypatch, capsys):
    class _Manager:
        enabled = True

        def list_checkpoints(self, *_args, **_kwargs):
            raise AssertionError("container session read the host checkpoint store")

    monkeypatch.setenv("TERMINAL_ENV", "docker")
    cli = object.__new__(CLICommandsMixin)
    cli.agent = SimpleNamespace(_checkpoint_mgr=_Manager())
    cli.session_id = "container-session"

    cli._handle_rollback_command("/rollback")

    output = capsys.readouterr().out
    assert "unavailable for terminal.backend=docker" in output

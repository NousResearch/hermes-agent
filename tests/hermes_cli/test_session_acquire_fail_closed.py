"""Registry failures refuse CLI entry before either interactive or one-shot work."""
import builtins
from unittest.mock import Mock

import pytest

import cli as cli_mod
from cli import HermesCLI
from hermes_cli import active_sessions


@pytest.fixture
def cli(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    instance = object.__new__(HermesCLI)
    instance.session_id = "cli-session"
    instance.config = {}  # No capacity limit still requires exclusive ownership.
    instance._active_session_lease = None
    instance._console_print = Mock()
    instance._tui_print_startup = Mock(side_effect=AssertionError("unleased CLI entered runtime"))
    return instance


def _fail_registry(monkeypatch, failure):
    def unavailable(*args, **kwargs):
        raise OSError("private/path credential=synthetic-secret")

    if failure == "import":
        real_import = builtins.__import__

        def registry_import(name, *args, **kwargs):
            if name == "hermes_cli.active_sessions":
                raise ImportError("private/path credential=synthetic-secret")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", registry_import)
    else:
        monkeypatch.setattr(active_sessions, "_write_entries" if failure == "write" else "_FileLock", unavailable)


@pytest.mark.parametrize("failure", ["write", "lock", "import"])
def test_registry_write_failure_does_not_start_interactive_cli(cli, monkeypatch, caplog, failure):
    _fail_registry(monkeypatch, failure)
    cli.run()

    cli._tui_print_startup.assert_not_called()
    assert cli._active_session_lease is None
    text = cli._console_print.call_args.args[0]
    assert "session" in text.lower() and "try again" in text.lower()
    assert "private/path" not in text and "synthetic-secret" not in text
    assert "synthetic-secret" not in caplog.text


@pytest.mark.parametrize("failure", ["write", "lock", "import"])
def test_quiet_query_exits_with_coordination_reason_before_work(cli, monkeypatch, capsys, failure):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "test")
    collect = Mock(side_effect=AssertionError("unleased CLI began query processing"))
    monkeypatch.setattr(cli_mod, "_collect_query_images", collect)
    _fail_registry(monkeypatch, failure)

    with pytest.raises(SystemExit) as stopped:
        cli_mod._run_single_query_mode(cli, "hello", image=None, quiet=True, oneshot=True)

    assert stopped.value.code == 1
    collect.assert_not_called()
    cli._console_print.assert_not_called()
    assert cli._active_session_lease is None
    output = capsys.readouterr()
    assert output.out == ""
    assert "hermes-refusal-reason: SESSION_COORDINATION_UNAVAILABLE" in output.err
    assert "try again" in output.err.lower()
    assert "private/path" not in output.err and "synthetic-secret" not in output.err


@pytest.mark.parametrize("reason", ["MAX_CONCURRENT_SESSIONS", "SESSION_NOT_OWNED"])
@pytest.mark.parametrize("stderr", [False, True])
def test_known_refusal_is_not_replaced_with_coordination_error(cli, monkeypatch, capsys, reason, stderr):
    refusal = active_sessions.ActiveSessionRefusal("Existing controlled refusal", reason)
    monkeypatch.setattr(active_sessions, "try_acquire_active_session", lambda **kwargs: (None, refusal))

    assert cli._claim_active_session(stderr=stderr) is False
    assert cli._active_session_lease is None
    output = capsys.readouterr()
    if stderr:
        assert output.err == f"hermes-refusal-reason: {reason}\nExisting controlled refusal\n"
        cli._console_print.assert_not_called()
    else:
        assert cli._console_print.call_args.args[0] == "[bold red]Existing controlled refusal[/]"
    assert "SESSION_COORDINATION_UNAVAILABLE" not in output.err

"""The PID-1-with-no-init startup warning (#111577): a Compose ``entrypoint:`` override
makes hermes PID 1 with no reaper above it, so orphaned children pile up as zombies."""

from hermes_cli.main import _warn_if_unsupervised_pid1


def test_warns_when_process_is_pid_1(capsys):
    _warn_if_unsupervised_pid1(pid=1)
    err = capsys.readouterr().err
    assert "PID 1" in err and "init: true" in err


def test_silent_when_not_pid_1(capsys):
    _warn_if_unsupervised_pid1(pid=4242)
    captured = capsys.readouterr()
    assert captured.err == "" and captured.out == ""

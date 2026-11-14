"""The cwd-persistence hook reads only the terminal backend name (#41128).

It runs on every messaging turn, terminal or not, so a typo in an unrelated terminal setting
(a timeout, an SSH port) must not fail the turn; the terminal tool reports those itself.
"""
from gateway.session_terminal_cwd import _local_backend


def test_an_unrelated_terminal_typo_does_not_fail_the_check(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_TIMEOUT", "not-a-number")
    monkeypatch.setenv("TERMINAL_SSH_PORT", "nope")
    assert _local_backend(None) is True


def test_a_remote_backend_is_not_local(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    assert _local_backend(None) is False

"""Regression coverage for HOME-independent host path helpers."""

import os

from hermes_constants import expand_user_path, get_os_user_home


def test_get_os_user_home_ignores_corrupt_home_env(monkeypatch):
    monkeypatch.setenv("HOME", "/tmp/\ufffcbad")

    home = get_os_user_home()

    assert "\ufffc" not in str(home)
    if os.name != "nt":
        assert str(home).startswith("/")


def test_expand_user_path_ignores_corrupt_home_env(monkeypatch):
    monkeypatch.setenv("HOME", "/tmp/\ufffcbad")

    expanded = expand_user_path("~/.hermes/hermes-agent")

    assert expanded == str(get_os_user_home() / ".hermes" / "hermes-agent")
    assert "\ufffc" not in expanded

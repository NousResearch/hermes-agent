"""Tests for Desktop backend ownership predicates."""

from __future__ import annotations

import sys

from runtime.desktop_identity import is_desktop_owned_backend, is_desktop_ssh_backend_argv


def test_ssh_backend_argv_requires_token_file_switch():
    assert is_desktop_ssh_backend_argv(["serve", "--ssh-session-token-file", "token"])
    assert not is_desktop_ssh_backend_argv(["serve", "--host", "127.0.0.1"])


def test_desktop_marker_alone_is_not_ownership():
    assert not is_desktop_owned_backend(["serve"], environ={"HERMES_DESKTOP": "1"})


def test_local_dashboard_token_proves_desktop_ownership():
    env = {"HERMES_DESKTOP": "1", "HERMES_DASHBOARD_SESSION_TOKEN": "secret"}
    assert is_desktop_owned_backend(["serve"], environ=env)


def test_ssh_token_file_proves_desktop_ownership(monkeypatch):
    env = {"HERMES_DESKTOP": "1"}
    argv = ["serve", "--isolated", "--ssh-session-token-file", "/tmp/session.token"]
    assert is_desktop_owned_backend(argv, environ=env)

    monkeypatch.setenv("HERMES_DESKTOP", "1")
    monkeypatch.delenv("HERMES_DASHBOARD_SESSION_TOKEN", raising=False)
    monkeypatch.setattr(sys, "argv", ["hermes", *argv])
    assert is_desktop_owned_backend()

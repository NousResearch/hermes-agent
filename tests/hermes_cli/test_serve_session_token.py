"""Tests for hermes serve --print-session-token."""

from __future__ import annotations

import argparse
import json
from urllib.error import URLError


import pytest

from hermes_cli.serve_session_token import (
    ServeTokenError,
    fetch_loopback_session_token,
    print_session_token,
    probe_host,
)
from hermes_cli.subcommands.dashboard import build_dashboard_parser


def dashboard_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    build_dashboard_parser(sub, cmd_dashboard=lambda args: None, cmd_dashboard_register=lambda args: None)
    return parser


def test_serve_help_advertises_print_session_token(capsys):
    parser = dashboard_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "--help"])
    output = capsys.readouterr().out
    assert "--print-session-token" in output


def test_probe_host_maps_wildcards_to_loopback():
    assert probe_host("0.0.0.0") == "127.0.0.1"
    assert probe_host("::") == "127.0.0.1"
    assert probe_host("10.0.0.5") == "10.0.0.5"


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_fetch_returns_loopback_token():
    def urlopen(request, timeout=3.0):
        assert request.full_url == "http://127.0.0.1:9119/api/status"
        return _FakeResponse(json.dumps({"auth_required": False, "session_token": "abc123"}).encode())

    assert fetch_loopback_session_token("127.0.0.1", 9119, urlopen=urlopen) == "abc123"


def test_fetch_missing_serve_is_typed():
    def urlopen(request, timeout=3.0):
        raise URLError("connection refused")

    with pytest.raises(ServeTokenError, match="listening") as info:
        fetch_loopback_session_token("127.0.0.1", 9119, urlopen=urlopen)
    assert info.value.kind == "missing"


def test_fetch_gated_serve_is_typed():
    def urlopen(request, timeout=3.0):
        return _FakeResponse(json.dumps({"auth_required": True, "session_token": "nope"}).encode())

    with pytest.raises(ServeTokenError, match="sign-in") as info:
        fetch_loopback_session_token("127.0.0.1", 9119, urlopen=urlopen)
    assert info.value.kind == "gated"


def test_fetch_old_serve_without_token_is_typed():
    def urlopen(request, timeout=3.0):
        return _FakeResponse(json.dumps({"auth_required": False}).encode())

    with pytest.raises(ServeTokenError, match="publish") as info:
        fetch_loopback_session_token("127.0.0.1", 9119, urlopen=urlopen)
    assert info.value.kind == "old"


def test_print_session_token_writes_only_the_token(monkeypatch, capsys):
    monkeypatch.setattr(
        "hermes_cli.serve_session_token.fetch_loopback_session_token",
        lambda host, port: "tok-live",
    )
    assert print_session_token("127.0.0.1", 9119) == 0
    out = capsys.readouterr()
    assert out.out == "tok-live\n"
    assert out.err == ""


def test_print_session_token_errors_go_to_stderr(monkeypatch, capsys):
    def boom(host, port):
        raise ServeTokenError("missing", "No hermes serve/dashboard is listening on 127.0.0.1:9119.")

    monkeypatch.setattr("hermes_cli.serve_session_token.fetch_loopback_session_token", boom)
    assert print_session_token("127.0.0.1", 9119) == 1
    out = capsys.readouterr()
    assert out.out == ""
    assert "listening" in out.err

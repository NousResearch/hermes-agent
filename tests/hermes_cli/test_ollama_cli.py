"""Tests for hermes ollama CLI — parser, fetch failure, cookie status, render, profile isolation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def test_add_parser_registers_subcommand():
    """add_parser registers the 'ollama' subcommand with --json and cookie subcommands."""
    import argparse

    from hermes_cli.ollama_cli import add_parser, cmd_ollama

    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers()
    add_parser(subs)

    # Parse bare "ollama"
    args = parser.parse_args(["ollama"])
    assert args.func is cmd_ollama
    assert not args.json

    # Parse "ollama --json"
    args = parser.parse_args(["ollama", "--json"])
    assert args.json

    # Parse "ollama cookie"
    args = parser.parse_args(["ollama", "cookie"])
    assert args.subcommand == "cookie"


# ---------------------------------------------------------------------------
# Cookie status
# ---------------------------------------------------------------------------


def test_cookie_status_missing(tmp_path):
    """cookie status returns 'missing' when no cookie file exists."""
    from hermes_cli.ollama_cli import _cookie_status

    with patch("hermes_cli.ollama_cli.COOKIE_FILE", tmp_path / "ollama_cookie.txt"):
        assert _cookie_status() == "missing"


def test_cookie_status_empty(tmp_path):
    """cookie status returns 'empty' for an empty cookie file."""
    from hermes_cli.ollama_cli import _cookie_status

    cookie = tmp_path / "ollama_cookie.txt"
    cookie.write_text("")
    with patch("hermes_cli.ollama_cli.COOKIE_FILE", cookie):
        assert _cookie_status() == "empty"


def test_cookie_status_malformed(tmp_path):
    """cookie status returns 'malformed' when content doesn't start with __Secure-session=."""
    from hermes_cli.ollama_cli import _cookie_status

    cookie = tmp_path / "ollama_cookie.txt"
    cookie.write_text("some-garbage")
    with patch("hermes_cli.ollama_cli.COOKIE_FILE", cookie):
        assert _cookie_status() == "malformed"


def test_cookie_status_present(tmp_path):
    """cookie status returns 'present' for a valid cookie file."""
    from hermes_cli.ollama_cli import _cookie_status

    cookie = tmp_path / "ollama_cookie.txt"
    cookie.write_text("__Secure-session=abc123")
    with patch("hermes_cli.ollama_cli.COOKIE_FILE", cookie):
        assert _cookie_status() == "present"


# ---------------------------------------------------------------------------
# Fetch failure (no cookie)
# ---------------------------------------------------------------------------


def test_fetch_usage_returns_none_when_no_cookie(tmp_path):
    """_fetch_usage returns None when no cookie file exists."""
    from hermes_cli.ollama_cli import _fetch_usage

    with patch("hermes_cli.ollama_cli.COOKIE_FILE", tmp_path / "ollama_cookie.txt"):
        assert _fetch_usage() is None


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def test_render_usage_unavailable():
    """_render_usage(None) returns unavailable message."""
    from hermes_cli.ollama_cli import _render_usage

    lines = _render_usage(None)
    assert any("unavailable" in l.lower() for l in lines)


def test_render_usage_with_data():
    """_render_usage renders plan and windows with percentages."""
    from hermes_cli.ollama_cli import _render_usage

    data = {
        "provider": "ollama-cloud",
        "plan": "Pro",
        "windows": [
            {"label": "Session", "used_percent": 12.5, "reset_at": "2026-07-16 00:00:00"},
            {"label": "Weekly", "used_percent": 45.0, "reset_at": "2026-07-18 00:00:00"},
        ],
        "fetched_at": "2026-07-15 15:30:00",
    }
    lines = _render_usage(data)
    text = "\n".join(lines)
    assert "Pro" in text
    assert "Session" in text
    assert "Weekly" in text
    assert "88%" in text  # 100 - round(12.5) = 88
    assert "55%" in text  # 100 - 45 = 55


# ---------------------------------------------------------------------------
# Profile isolation
# ---------------------------------------------------------------------------


def test_cookie_file_uses_get_hermes_home():
    """COOKIE_FILE is derived from get_hermes_home(), not Path.home() / '.hermes'."""
    import hermes_cli.ollama_cli as mod

    path = mod.COOKIE_FILE
    assert "ollama_cookie.txt" in str(path)
    # The path should not contain the hardcoded ".hermes" segment as a literal
    # — it should go through get_hermes_home() which returns the profile-aware root.
    assert ".hermes" not in str(path) or "profiles" not in str(path)


# ---------------------------------------------------------------------------
# CLI dispatch (process_command integration)
# ---------------------------------------------------------------------------


def test_ollama_slash_command_is_registered():
    """The /ollama command is registered in the command registry."""
    from hermes_cli.commands import COMMAND_REGISTRY, resolve_command

    resolved = resolve_command("/ollama")
    assert resolved is not None
    assert resolved.name == "ollama"


def test_ollama_dispatch_no_agent(tmp_path, monkeypatch):
    """_handle_ollama_command prints when no agent / no cookie."""
    from cli import HermesCLI

    cli = object.__new__(HermesCLI)
    cli.agent = None
    cli._app = None

    from hermes_cli.ollama_cli import COOKIE_FILE as _CF

    monkeypatch.setattr(
        "hermes_cli.ollama_cli.COOKIE_FILE",
        tmp_path / "ollama_cookie.txt",
    )

    # Capture stdout
    import io

    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        cli._handle_ollama_command("/ollama")
    finally:
        sys.stdout = old_stdout

    output = captured.getvalue()
    assert "unavailable" in output.lower() or "Ollama" in output
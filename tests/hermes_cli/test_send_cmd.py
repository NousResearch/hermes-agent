"""Tests for the ``hermes send`` CLI subcommand.

Covers the argument parsing / stdin / file / list behavior of
``hermes_cli.send_cmd``. The underlying ``send_message_tool`` is stubbed so
no network I/O or gateway is required.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from hermes_cli import send_cmd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse(argv):
    """Build the top-level parser and return the parsed args for ``argv``."""
    import argparse

    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    send_cmd.register_send_subparser(subparsers)
    return parser.parse_args(["send", *argv])


class _FakeTool:
    """Replacement for ``tools.send_message_tool.send_message_tool``."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def __call__(self, args, **_kw):
        self.calls.append(dict(args))
        return json.dumps(self.payload)


@pytest.fixture
def fake_tool(monkeypatch):
    """Install a fake send_message_tool and return the stub for inspection."""
    import sys
    import types

    fake = _FakeTool({"success": True, "message_id": "m123"})

    mod = types.ModuleType("tools.send_message_tool")
    mod.send_message_tool = fake
    # Register the stub so ``from tools.send_message_tool import ...`` inside
    # cmd_send resolves to our fake. Also patch the parent ``tools`` package
    # entry so attribute lookup works.
    monkeypatch.setitem(sys.modules, "tools.send_message_tool", mod)
    return fake


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_positional_message_success(fake_tool, capsys):
    args = _parse(["--to", "telegram", "hello world"])
    with pytest.raises(SystemExit) as exc:
        send_cmd.cmd_send(args)
    assert exc.value.code == 0
    assert fake_tool.calls == [
        {"action": "send", "target": "telegram", "message": "hello world"}
    ]
    out = capsys.readouterr()
    assert "sent" in out.out or out.out == ""  # "sent" is the default success banner


def test_stdin_message(fake_tool, monkeypatch, capsys):
    # Piped stdin (not a tty) should be consumed as the message body.
    monkeypatch.setattr("sys.stdin", io.StringIO("piped body\n"))
    # Force isatty to return False so the CLI reads from stdin.
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    args = _parse(["--to", "discord:#ops"])
    with pytest.raises(SystemExit) as exc:
        send_cmd.cmd_send(args)
    assert exc.value.code == 0
    assert fake_tool.calls[0]["message"] == "piped body\n"
    assert fake_tool.calls[0]["target"] == "discord:#ops"


def test_file_message(fake_tool, tmp_path):
    body = tmp_path / "msg.txt"
    body.write_text("from a file\n")
    args = _parse(["--to", "slack:#eng", "--file", str(body)])
    with pytest.raises(SystemExit) as exc:
        send_cmd.cmd_send(args)
    assert exc.value.code == 0
    assert fake_tool.calls[0]["message"] == "from a file\n"


def test_file_dash_means_stdin(fake_tool, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("dash body"))
    args = _parse(["--to", "telegram", "--file", "-"])
    with pytest.raises(SystemExit) as exc:
        send_cmd.cmd_send(args)
    assert exc.value.code == 0
    assert fake_tool.calls[0]["message"] == "dash body"


def test_subject_prepends_header(fake_tool):
    args = _parse(["--to", "telegram", "--subject", "[CI]", "body text"])
    with pytest.raises(SystemExit) as exc:
        send_cmd.cmd_send(args)
    assert exc.value.code == 0
    assert fake_tool.calls[0]["message"] == "[CI]\n\nbody text"


def test_json_mode_emits_payload(fake_tool, capsys):
    args = _parse(["--to", "telegram", "--json", "hi"])
    with pytest.raises(SystemExit) as exc:
        send_cmd.cmd_send(args)
    assert exc.value.code == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload.get("success") is True
    assert payload.get("message_id") == "m123"


def test_quiet_suppresses_stdout(fake_tool, capsys):
    args = _parse(["--to", "telegram", "--quiet", "shh"])
    with pytest.raises(SystemExit) as exc:
        send_cmd.cmd_send(args)
    assert exc.value.code == 0
    out = capsys.readouterr()
    assert out.out == ""


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_missing_target(fake_tool, capsys, monkeypatch):
    # Ensure stdin is a tty so the CLI does not try to consume it as a body.
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    args = _parse(["hello"])
    with pytest.raises(SystemExit) as exc:
        send_cmd.cmd_send(args)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--to" in err


def test_missing_message(fake_tool, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    args = _parse(["--to", "telegram"])
    with pytest.raises(SystemExit) as exc:
        send_cmd.cmd_send(args)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "no message" in err.lower()


def test_file_not_found_is_usage_error(fake_tool, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    args = _parse(["--to", "telegram", "--file", "/nonexistent/does-not-exist.txt"])
    with pytest.raises(SystemExit) as exc:
        send_cmd.cmd_send(args)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "cannot read" in err.lower()


def test_tool_error_returns_failure_exit(monkeypatch, capsys):
    import sys as _sys
    import types as _types

    fake_mod = _types.ModuleType("tools.send_message_tool")

    def _bad_tool(args, **_kw):
        return json.dumps({"error": "platform blew up"})

    fake_mod.send_message_tool = _bad_tool
    monkeypatch.setitem(_sys.modules, "tools.send_message_tool", fake_mod)

    args = _parse(["--to", "telegram", "nope"])
    with pytest.raises(SystemExit) as exc:
        send_cmd.cmd_send(args)
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "platform blew up" in err


def test_skipped_result_is_success(monkeypatch):
    import sys as _sys
    import types as _types

    fake_mod = _types.ModuleType("tools.send_message_tool")
    fake_mod.send_message_tool = lambda args, **_kw: json.dumps(
        {"success": True, "skipped": True, "reason": "duplicate"}
    )
    monkeypatch.setitem(_sys.modules, "tools.send_message_tool", fake_mod)

    args = _parse(["--to", "telegram", "dup"])
    with pytest.raises(SystemExit) as exc:
        send_cmd.cmd_send(args)
    assert exc.value.code == 0


# ---------------------------------------------------------------------------
# --list
# ---------------------------------------------------------------------------


def test_list_human_output(monkeypatch, capsys):
    import sys as _sys
    import types as _types

    fake_dir = _types.ModuleType("gateway.channel_directory")
    fake_dir.format_directory_for_display = lambda: "Available messaging targets:\n\nTelegram:\n  telegram:-100123\n"
    fake_dir.load_directory = lambda: {
        "platforms": {"telegram": [{"id": "-100123", "name": "Test Group"}]}
    }
    monkeypatch.setitem(_sys.modules, "gateway.channel_directory", fake_dir)

    args = _parse(["--list"])
    with pytest.raises(SystemExit) as exc:
        send_cmd.cmd_send(args)
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Telegram" in out


def test_list_json(monkeypatch, capsys):
    import sys as _sys
    import types as _types

    fake_dir = _types.ModuleType("gateway.channel_directory")
    fake_dir.format_directory_for_display = lambda: "(ignored in json mode)"
    fake_dir.load_directory = lambda: {
        "platforms": {"telegram": [{"id": "-100123", "name": "Test Group"}]}
    }
    monkeypatch.setitem(_sys.modules, "gateway.channel_directory", fake_dir)

    args = _parse(["--list", "--json"])
    with pytest.raises(SystemExit) as exc:
        send_cmd.cmd_send(args)
    assert exc.value.code == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["platforms"]["telegram"][0]["name"] == "Test Group"


def test_list_filter_platform(monkeypatch, capsys):
    import sys as _sys
    import types as _types

    fake_dir = _types.ModuleType("gateway.channel_directory")
    fake_dir.format_directory_for_display = lambda: "(should not be called when filter set)"
    fake_dir.load_directory = lambda: {
        "platforms": {
            "telegram": [{"id": "-100123", "name": "TG Chat"}],
            "discord": [{"id": "555", "name": "bot-home"}],
        }
    }
    monkeypatch.setitem(_sys.modules, "gateway.channel_directory", fake_dir)

    # When --list is set, argparse puts the optional bareword in the
    # `message` positional slot (where the send-mode body would go).
    args = _parse(["--list", "telegram"])
    with pytest.raises(SystemExit) as exc:
        send_cmd.cmd_send(args)
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "telegram" in out.lower()
    assert "discord" not in out.lower()


def test_list_unknown_platform_fails(monkeypatch, capsys):
    import sys as _sys
    import types as _types

    fake_dir = _types.ModuleType("gateway.channel_directory")
    fake_dir.format_directory_for_display = lambda: ""
    fake_dir.load_directory = lambda: {"platforms": {"telegram": []}}
    monkeypatch.setitem(_sys.modules, "gateway.channel_directory", fake_dir)

    args = _parse(["--list", "pigeon-post"])
    with pytest.raises(SystemExit) as exc:
        send_cmd.cmd_send(args)
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "pigeon-post" in err


# ---------------------------------------------------------------------------
# Parser registration contract
# ---------------------------------------------------------------------------


def test_register_send_subparser_is_reusable():
    """Sanity check: the registrar returns a parser and wires ``cmd_send``."""
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    send_parser = send_cmd.register_send_subparser(subparsers)
    assert send_parser is not None
    args = parser.parse_args(["send", "--to", "telegram", "hi"])
    assert args.func is send_cmd.cmd_send
    assert args.to == "telegram"
    assert args.message == "hi"


# ---------------------------------------------------------------------------
# Env loader
# ---------------------------------------------------------------------------


def test_load_hermes_env_bridges_config_yaml_scalars(tmp_path, monkeypatch):
    """Top-level config.yaml scalars should be bridged into os.environ.

    This mirrors the gateway/run.py bootstrap behavior: without this, running
    ``hermes send`` from a fresh shell cannot resolve the home channel
    because ``TELEGRAM_HOME_CHANNEL`` (saved by ``hermes config set``) lives
    in config.yaml, not in .env — and the gateway's config loader reads via
    ``os.getenv(...)``.
    """
    import os

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / ".env").write_text("SOME_TOKEN=abc123\n")
    (hermes_home / "config.yaml").write_text(
        "TELEGRAM_HOME_CHANNEL: '5550001111'\nnested:\n  ignored: true\n"
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)
    monkeypatch.delenv("SOME_TOKEN", raising=False)

    # Force get_hermes_home() to re-resolve under the patched env.
    from importlib import reload

    import hermes_cli.config as _hc_config
    reload(_hc_config)

    send_cmd._load_hermes_env()

    assert os.environ.get("SOME_TOKEN") == "abc123"
    assert os.environ.get("TELEGRAM_HOME_CHANNEL") == "5550001111"


def test_load_hermes_env_does_not_override_existing(tmp_path, monkeypatch):
    """Existing env vars must not be clobbered by config.yaml values."""
    import os

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("TELEGRAM_HOME_CHANNEL: yaml_value\n")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "env_value")

    from importlib import reload
    import hermes_cli.config as _hc_config
    reload(_hc_config)

    send_cmd._load_hermes_env()

    assert os.environ.get("TELEGRAM_HOME_CHANNEL") == "env_value"


def test_load_hermes_env_handles_missing_files(tmp_path, monkeypatch):
    """No .env or config.yaml should be a silent no-op, not an exception."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from importlib import reload
    import hermes_cli.config as _hc_config
    reload(_hc_config)

    # Should not raise.
    send_cmd._load_hermes_env()

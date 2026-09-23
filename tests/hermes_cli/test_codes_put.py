"""``hermes codes put`` — model-blind verification-code handoff (#119683)."""

from __future__ import annotations

import io
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent import code_registry  # noqa: E402
from hermes_cli.codes import _cmd_put, codes_command, register_cli  # noqa: E402


@pytest.fixture(autouse=True)
def _clean():
    code_registry.clear_codes()
    yield
    code_registry.clear_codes()


def _args(**overrides):
    base = dict(
        code=None,
        origin="",
        source="",
        ttl=None,
        extract=False,
        _codes_handler=_cmd_put,
    )
    base.update(overrides)
    return Namespace(**base)


def test_register_cli_builds_put_subparser():
    import argparse

    p = argparse.ArgumentParser()
    register_cli(p)
    args = p.parse_args(["put", "123456", "--source", "sms", "--origin", "https://example.com"])
    assert args.codes_action == "put"
    assert args.code == "123456"
    assert args.source == "sms"
    assert args.origin == "https://example.com"


def test_put_prints_handle_json_only(capsys):
    _cmd_put(_args(code="424242", source="sms"))
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["code_handle"].startswith(code_registry.HANDLE_PREFIX)
    assert "424242" not in captured.out
    assert payload["source"] == "sms"
    # parked and consumable
    assert code_registry.consume(payload["code_handle"]) == "424242"


def test_put_reads_code_from_stdin_dash(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO("555666\n"))
    _cmd_put(_args(code="-", source="email"))
    payload = json.loads(capsys.readouterr().out)
    assert code_registry.consume(payload["code_handle"]) == "555666"


def test_put_bare_reads_stdin_when_not_tty(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO("777888"))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    _cmd_put(_args(code=None))
    payload = json.loads(capsys.readouterr().out)
    assert code_registry.consume(payload["code_handle"]) == "777888"


def test_put_empty_code_exits():
    with pytest.raises(SystemExit) as excinfo:
        _cmd_put(_args(code="   "))
    assert "code is required" in str(excinfo.value)


def test_put_forwards_ttl_and_origin(monkeypatch, capsys):
    _cmd_put(_args(code="121212", origin="https://example.com", ttl=120.0))
    payload = json.loads(capsys.readouterr().out)
    assert payload["origin"] == "https://example.com"
    # consume requires matching origin
    assert code_registry.consume(payload["code_handle"], origin="https://evil.test") is None


def test_codes_command_dispatches_handler(capsys):
    codes_command(_args(code="909090"))
    payload = json.loads(capsys.readouterr().out)
    assert payload["code_handle"].startswith(code_registry.HANDLE_PREFIX)


def test_extract_from_piped_body_mints_without_printing_body(monkeypatch, capsys):
    body = "GitHub: your verification code is 424242. Do not share it."
    monkeypatch.setattr(sys, "stdin", io.StringIO(body))
    _cmd_put(_args(code="-", extract=True, source="email"))
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["code_handle"].startswith(code_registry.HANDLE_PREFIX)
    assert "424242" not in captured.out
    assert "verification code" not in captured.out
    assert code_registry.consume(payload["code_handle"]) == "424242"


def test_extract_with_no_code_exits(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("no digits here at all"))
    with pytest.raises(SystemExit) as excinfo:
        _cmd_put(_args(code="-", extract=True))
    assert "no verification code found" in str(excinfo.value)

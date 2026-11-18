"""``hermes send`` exit contract for partial MEDIA delivery: when the platform sender delivered the
text but reported the requested attachment as undelivered, the command exits non-zero and says so
on stderr instead of printing "sent" (#115908 sibling). Drives the real ``cmd_send`` ->
``send_message_tool`` -> ``_send_to_platform`` chain; only the env/config loaders, the target
resolver and the Telegram transport are faked."""

import argparse
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import Platform
from hermes_cli.send_cmd import _emit_result, cmd_send


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(home / ".hermes"))
    monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "0")


def _run_cmd_send(message, sender_result):
    calls = []

    async def fake_sender(*args, **kwargs):
        calls.append(kwargs)
        return dict(sender_result)

    pconfig = SimpleNamespace(enabled=True, token="***", extra={})
    config = SimpleNamespace(platforms={Platform.TELEGRAM: pconfig}, get_home_channel=lambda _platform: None)
    args = argparse.Namespace(to="telegram:12345", message=message, file=None, subject=None, mentions=None,
                              list_targets=False, json=False, quiet=False)
    with patch("hermes_cli.send_cmd._load_hermes_env"), \
         patch("gateway.config.load_gateway_config", return_value=config), \
         patch("gateway.outbound.send._resolve_tool_target", return_value=("telegram", "12345", None, None)), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=asyncio.run), \
         patch("gateway.mirror.mirror_to_session", return_value=True), \
         patch("gateway.outbound.send._plugin_standalone_sender", return_value=(fake_sender, None)), \
         pytest.raises(SystemExit) as exit_info:
        cmd_send(args)
    return exit_info.value.code, calls


def test_sender_media_warning_exits_nonzero_with_count_only_error_line(tmp_path, capsys):
    report = tmp_path / "report.pdf"
    report.write_bytes(b"%PDF report")
    code, calls = _run_cmd_send(
        f"report\nMEDIA:{report}",
        {"success": True, "message_id": "77", "warnings": [f"Failed to send media {report}: rejected"]})

    out, err = capsys.readouterr()
    assert code == 1
    assert len(calls) == 1  # partial delivery is reported, never re-sent
    assert out == ""
    # The detail is the sender's warning line (``_emit_result`` prints every warning); the error
    # line itself carries counts only.
    lines = err.splitlines()
    assert f"hermes send: warning: Failed to send media {report}: rejected" in lines
    error_lines = [line for line in lines if line.startswith("hermes send: Delivery incomplete:")]
    assert len(error_lines) == 1
    assert str(report) not in error_lines[0] and "rejected" not in error_lines[0]


def test_complete_media_delivery_exits_zero(tmp_path, capsys):
    report = tmp_path / "report.pdf"
    report.write_bytes(b"%PDF report")
    code, _calls = _run_cmd_send(f"report\nMEDIA:{report}", {"success": True, "message_id": "78"})

    out, err = capsys.readouterr()
    assert code == 0
    assert out.strip() == "sent"
    assert err == ""


def test_unconfirmed_media_delivery_exits_zero_and_says_so_on_stderr(tmp_path, capsys):
    report = tmp_path / "report.pdf"
    report.write_bytes(b"%PDF report")
    note = f"Telegram never confirmed media {report} (Timed out); it may already be in the chat, so do not resend it"
    code, calls = _run_cmd_send(f"report\nMEDIA:{report}",
                                {"success": True, "message_id": "79", "delivery_unconfirmed": note})

    out, err = capsys.readouterr()
    assert code == 0
    assert len(calls) == 1
    assert out.strip() == "sent"
    assert "hermes send: warning: Telegram never confirmed media" in err


def test_plain_emit_result_prints_delivery_unconfirmed_without_failing(capsys):
    payload = {"success": True, "delivery_unconfirmed": "Telegram never confirmed a photo album; do not resend it"}
    assert _emit_result(json.dumps(payload), json_mode=False, quiet=False) == 0
    out, err = capsys.readouterr()
    assert out.strip() == "sent"
    assert err.strip() == "hermes send: warning: Telegram never confirmed a photo album; do not resend it"

"""``send_message`` partial MEDIA delivery: a send that delivered the text but reported a requested
attachment as undelivered (sender ``warnings``) is a partial success, never ``success: True``
(#115908 sibling). Runs the real ``_handle_send`` -> ``_send_to_platform`` -> ``_emit_result``
chain; only the gateway config loader, the target resolver and the native transport are faked."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import Platform
from gateway.platforms.send_targets import StandaloneSendPolicy
from hermes_cli.send_cmd import _emit_result
from gateway.outbound.send import send_message_tool

_SECRET_TOKEN = "tok-partial-delivery-secret"


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(home / ".hermes"))
    monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "0")
    return home


def _config(platform):
    pconfig = SimpleNamespace(enabled=True, token=_SECRET_TOKEN, extra={})
    return SimpleNamespace(platforms={platform: pconfig}, get_home_channel=lambda _platform: None)


def _send(platform, platform_name, chat_id, message, *, sender_result, sender_patches):
    """Run the send through the real post-processing with only config, target and transport faked."""
    calls = []

    async def fake_sender(*args, **kwargs):
        calls.append((args, kwargs))
        return dict(sender_result)

    with patch("gateway.config.load_gateway_config", return_value=_config(platform)), \
         patch("gateway.outbound.send._resolve_tool_target", return_value=(platform_name, chat_id, None, None)), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=asyncio.run), \
         patch("gateway.mirror.mirror_to_session", return_value=True), \
         sender_patches(fake_sender):
        raw = send_message_tool({"action": "send", "target": f"{platform_name}:{chat_id}", "message": message})
    return raw, json.loads(raw), calls


def _telegram_transport(fake_sender):
    return patch("gateway.outbound.send._plugin_standalone_sender", return_value=(fake_sender, None))


class _TextDirectTransport:
    """Email-style text-only plugin: the generic path omits MEDIA and adds its own warning."""

    def __init__(self, fake_sender):
        self._patches = [
            patch("gateway.outbound.send._standalone_send_policy",
                  return_value=StandaloneSendPolicy(label="Email", text_direct=True)),
            patch("gateway.outbound.send._registry_standalone_send", new=fake_sender),
        ]

    def __enter__(self):
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()
        return False


def _attachment(tmp_path, name="report.pdf"):
    path = tmp_path / name
    path.write_bytes(b"%PDF report")
    return path


def _assert_counts_only(error, *leaks):
    assert error.startswith("Delivery incomplete:")
    for leak in leaks:
        assert str(leak) not in error


def test_generic_text_direct_path_omitting_media_is_partial_and_exits_nonzero(tmp_path):
    report = _attachment(tmp_path)
    raw, result, calls = _send(
        Platform.EMAIL, "email", "ops@example.invalid", f"nightly report\nMEDIA:{report}",
        sender_result={"success": True, "message_id": "m-1"}, sender_patches=_TextDirectTransport)

    assert len(calls) == 1  # the text went out exactly once; no automatic resend
    assert result["success"] is False
    assert result["partial_success"] is True
    assert result["message_id"] == "m-1"
    assert len(result["warnings"]) == 1 and "MEDIA attachments were omitted" in result["warnings"][0]
    assert result["mirrored"] is True
    _assert_counts_only(result["error"], report, _SECRET_TOKEN, "omitted")
    assert _emit_result(raw, json_mode=True, quiet=True) == 1


def test_sender_reported_media_failure_is_partial_and_keeps_warnings(tmp_path):
    report = _attachment(tmp_path)
    warning = f"Failed to send media {report}: upload rejected token={_SECRET_TOKEN}"
    raw, result, calls = _send(
        Platform.TELEGRAM, "telegram", "12345", f"report\nMEDIA:{report}",
        sender_result={"success": True, "message_id": "77", "warnings": [warning]},
        sender_patches=_telegram_transport)

    assert len(calls) == 1
    assert calls[0][1]["media_files"] == [(str(report.resolve()), False)]
    assert result["success"] is False
    assert result["partial_success"] is True
    assert result["message_id"] == "77"
    assert result["warnings"] == [warning]
    assert "1 MEDIA delivery warning(s)" in result["error"]
    _assert_counts_only(result["error"], report, _SECRET_TOKEN, "upload rejected")
    assert _emit_result(raw, json_mode=True, quiet=True) == 1


def test_complete_media_delivery_stays_success(tmp_path):
    report = _attachment(tmp_path)
    raw, result, _calls = _send(
        Platform.TELEGRAM, "telegram", "12345", f"report\nMEDIA:{report}",
        sender_result={"success": True, "message_id": "78"}, sender_patches=_telegram_transport)

    assert result["success"] is True
    assert "partial_success" not in result and "error" not in result
    assert _emit_result(raw, json_mode=True, quiet=True) == 0


def test_text_only_send_with_advisory_warning_stays_success():
    raw, result, _calls = _send(
        Platform.TELEGRAM, "telegram", "12345", "plain text",
        sender_result={"success": True, "message_id": "79", "warnings": ["advisory"]},
        sender_patches=_telegram_transport)

    assert result["success"] is True
    assert result["warnings"] == ["advisory"]
    assert "partial_success" not in result
    assert _emit_result(raw, json_mode=True, quiet=True) == 0


def test_failed_sender_stays_a_plain_failure(tmp_path):
    report = _attachment(tmp_path)
    raw, result, calls = _send(
        Platform.TELEGRAM, "telegram", "12345", f"report\nMEDIA:{report}",
        sender_result={"error": "Telegram send failed: chat not found"}, sender_patches=_telegram_transport)

    assert len(calls) == 1
    assert "success" not in result and "partial_success" not in result
    assert result["error"] == "Telegram send failed: chat not found"
    assert _emit_result(raw, json_mode=True, quiet=True) == 1


def test_filter_drop_and_sender_warning_are_both_reported(tmp_path):
    report = _attachment(tmp_path)
    missing = tmp_path / "missing.pdf"
    warning = f"Failed to send media {report}: too large"
    raw, result, calls = _send(
        Platform.TELEGRAM, "telegram", "12345", f"report\nMEDIA:{report}\nMEDIA:{missing}",
        sender_result={"success": True, "message_id": "80", "warnings": [warning]},
        sender_patches=_telegram_transport)

    assert calls[0][1]["media_files"] == [(str(report.resolve()), False)]
    assert result["success"] is False
    assert result["partial_success"] is True
    assert result["media_dropped"] == [{"path": str(missing), "reason": "not found on this host"}]
    assert result["warnings"] == [warning]
    assert "1 requested MEDIA attachment(s) dropped" in result["error"]
    assert "1 MEDIA delivery warning(s)" in result["error"]
    _assert_counts_only(result["error"], report, missing, "too large")
    assert _emit_result(raw, json_mode=True, quiet=True) == 1


_UNCONFIRMED = "Telegram never confirmed a photo album; it may already be in the chat, so do not resend it"


def test_unconfirmed_album_alone_stays_success_and_keeps_the_typed_field(tmp_path):
    report = _attachment(tmp_path)
    raw, result, calls = _send(
        Platform.TELEGRAM, "telegram", "12345", f"report\nMEDIA:{report}",
        sender_result={"success": True, "delivery_unconfirmed": _UNCONFIRMED},
        sender_patches=_telegram_transport)

    assert len(calls) == 1  # no automatic resend
    assert result["success"] is True
    assert "partial_success" not in result and "error" not in result
    assert result["delivery_unconfirmed"] == _UNCONFIRMED
    assert _emit_result(raw, json_mode=True, quiet=True) == 0


def test_unconfirmed_album_does_not_mask_a_real_failure(tmp_path):
    report = _attachment(tmp_path)
    warning = f"Failed to send Telegram photo {report}: file too big"
    raw, result, calls = _send(
        Platform.TELEGRAM, "telegram", "12345", f"report\nMEDIA:{report}",
        sender_result={"success": True, "warnings": [warning], "delivery_unconfirmed": _UNCONFIRMED},
        sender_patches=_telegram_transport)

    assert len(calls) == 1
    assert result["success"] is False
    assert result["partial_success"] is True
    assert result["warnings"] == [warning]
    assert result["delivery_unconfirmed"] == _UNCONFIRMED
    assert "1 MEDIA delivery warning(s)" in result["error"]
    assert _emit_result(raw, json_mode=True, quiet=True) == 1

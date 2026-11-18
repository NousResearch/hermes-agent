"""Telegram standalone sender: a photo album whose reply was lost (``raw_response["ambiguous"]``) is
reported in the typed ``delivery_unconfirmed`` field, never as a failure ``warning`` — a warning makes
``hermes send`` a partial (exit 1) and invites a duplicate re-send of an album that may already be in
the chat. Same contract as the live adapter path (``gateway.outbound.send._send_live_adapter_media``).
A single (non-album) upload whose request may have landed (``request_may_have_delivered``) follows the
same contract. Runs the real ``_send_telegram``; only the bot factory, the upload transports and the
lost-request classifier are faked."""

import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import Platform
from gateway.outbound.send import send_message_tool
from gateway.platforms.base import SendResult
from plugins.platforms.telegram.standalone import _send_telegram


@pytest.fixture(autouse=True)
def _telegram_package_stub(monkeypatch):
    """``python-telegram-bot`` is an optional extra: without it ``_send_telegram`` returns its
    "not installed" error before any upload, so every case here would test the import, not the
    contract. Stub the package the way the other ``_send_telegram`` tests do
    (``tests/tools/test_telegram_send_message_chunk_mdv2.py``) so the run is the same with or
    without the extra. The adapter is imported first, under the real environment, so the stub never
    becomes its cached ``telegram`` binding."""
    import plugins.platforms.telegram.adapter  # noqa: F401
    constants = SimpleNamespace(ParseMode=SimpleNamespace(MARKDOWN_V2="MarkdownV2", HTML="HTML"))
    monkeypatch.setitem(sys.modules, "telegram", SimpleNamespace(constants=constants))
    monkeypatch.setitem(sys.modules, "telegram.constants", constants)


def _photos(tmp_path, count=2):
    paths = []
    for i in range(count):
        path = tmp_path / f"shot{i}.png"
        path.write_bytes(b"\x89PNG fake")
        paths.append(str(path))
    return paths


def _send_album(paths, album_result):
    calls = []

    async def fake_album(bot, extra, chat_id, album, **kwargs):
        calls.append(list(album))
        return album_result

    with patch("plugins.platforms.telegram.standalone._telegram_bot", return_value=SimpleNamespace()), \
         patch("plugins.platforms.telegram.telegram_outbound.send_album_standalone", new=fake_album):
        result = asyncio.run(_send_telegram("tok", "12345", "", media_files=[(p, False) for p in paths]))
    return result, calls


def test_ambiguous_album_is_typed_unconfirmed_success_not_a_warning(tmp_path):
    paths = _photos(tmp_path)
    result, calls = _send_album(paths, SendResult(
        success=True, raw_response={"failed": [], "refused": [], "ambiguous": True}))

    assert len(calls) == 1  # one album attempt; nothing resent
    assert result["success"] is True
    assert "warnings" not in result
    assert "do not resend" in result["delivery_unconfirmed"]


def test_ambiguous_album_keeps_real_failures_as_warnings(tmp_path):
    paths = _photos(tmp_path, count=3)
    result, calls = _send_album(paths, SendResult(
        success=True, raw_response={"failed": [(paths[2], "file too big")], "refused": [], "ambiguous": True}))

    assert len(calls) == 1
    assert result["success"] is True
    assert len(result["warnings"]) == 1 and paths[2] in result["warnings"][0]
    assert "do not resend" in result["delivery_unconfirmed"]


@pytest.fixture
def _isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(home / ".hermes"))
    monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "0")


def _upload_patches(may_have_delivered, album_result=None):
    """Fake bot, a per-file upload whose request raised, the lost-request verdict and (optionally) the album."""
    async def failing_upload(*args, **kwargs):
        raise TimeoutError("Timed out")

    async def fake_album(bot, extra, chat_id, album, **kwargs):
        return album_result

    patches = [patch("plugins.platforms.telegram.standalone._telegram_bot", return_value=SimpleNamespace()),
               patch("plugins.platforms.telegram.standalone._telegram_send_one_media", new=failing_upload),
               patch("plugins.platforms.telegram.telegram_media_retry.request_may_have_delivered",
                     return_value=may_have_delivered)]
    if album_result is not None:
        patches.append(patch("plugins.platforms.telegram.telegram_outbound.send_album_standalone", new=fake_album))
    return patches


def _run_with(patches, fn):
    for p in patches:
        p.start()
    try:
        return fn()
    finally:
        for p in reversed(patches):
            p.stop()


def _document(tmp_path):
    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF report")
    return str(path)


def _send_tool(path, text=""):
    """``send_message`` through the real ``_handle_send`` -> ``_send_to_platform`` -> ``_send_telegram``."""
    async def sender(pconfig, chat_id, message, *, media_files=None, **_kw):
        return await _send_telegram("tok", chat_id, message, media_files=media_files)

    pconfig = SimpleNamespace(enabled=True, token="tok", extra={})
    config = SimpleNamespace(platforms={Platform.TELEGRAM: pconfig}, get_home_channel=lambda _platform: None)
    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("gateway.outbound.send._resolve_tool_target", return_value=("telegram", "12345", None, None)), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=asyncio.run), \
         patch("gateway.mirror.mirror_to_session", return_value=True), \
         patch("gateway.outbound.send._plugin_standalone_sender", return_value=(sender, None)):
        return json.loads(send_message_tool(
            {"action": "send", "target": "telegram:12345", "message": f"{text}\nMEDIA:{path}".lstrip()}))


def test_unconfirmed_single_upload_is_typed_unconfirmed_success_not_a_warning(tmp_path):
    path = _document(tmp_path)
    result = _run_with(_upload_patches(True), lambda: asyncio.run(
        _send_telegram("tok", "12345", "", media_files=[(path, False)])))

    assert result["success"] is True
    assert "warnings" not in result
    assert "do not resend" in result["delivery_unconfirmed"] and path in result["delivery_unconfirmed"]


def test_unconfirmed_single_upload_through_send_message_is_not_partial(tmp_path, _isolated_home):
    path = _document(tmp_path)
    result = _run_with(_upload_patches(True), lambda: _send_tool(path))

    assert result["success"] is True
    assert "partial_success" not in result and "error" not in result
    assert "do not resend" in result["delivery_unconfirmed"]


def test_definite_single_upload_failure_stays_a_warning_and_partial(tmp_path, _isolated_home):
    path = _document(tmp_path)
    text = "report " * 200  # past the caption cap: the text goes as its own message and lands

    async def text_lands(*args, **kwargs):
        return SimpleNamespace(message_id=5)

    def patches():
        return [*_upload_patches(False),
                patch("plugins.platforms.telegram.standalone._telegram_send_text_chunk", new=text_lands)]

    direct = _run_with(patches(), lambda: asyncio.run(
        _send_telegram("tok", "12345", text, media_files=[(path, False)])))
    assert direct["success"] is True
    assert "delivery_unconfirmed" not in direct
    assert len(direct["warnings"]) == 1 and path in direct["warnings"][0]

    routed = _run_with(patches(), lambda: _send_tool(path, text))
    assert routed["success"] is False and routed["partial_success"] is True
    assert "delivery_unconfirmed" not in routed


def test_unconfirmed_album_and_single_upload_share_one_unconfirmed_field(tmp_path):
    paths = _photos(tmp_path)
    album = SendResult(success=True, raw_response={"failed": [], "refused": [paths[1]], "ambiguous": True})
    result = _run_with(_upload_patches(True, album), lambda: asyncio.run(
        _send_telegram("tok", "12345", "", media_files=[(p, False) for p in paths])))

    assert result["success"] is True
    assert "warnings" not in result
    unconfirmed = result["delivery_unconfirmed"]
    assert isinstance(unconfirmed, str)
    assert "photo album" in unconfirmed and paths[1] in unconfirmed

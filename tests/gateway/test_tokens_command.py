"""/tokens [on|off|always|status] — three-state per-session/global toggle."""
import asyncio
import importlib.util as _ilu
import json

import pytest
from unittest.mock import MagicMock

# gateway.run pulls heavy optional deps (rich, aiohttp, …); skip cleanly where
# they're absent (e.g. minimal CI/VPS pytest env).
if _ilu.find_spec("rich") is None or _ilu.find_spec("aiohttp") is None:
    pytest.skip("gateway runtime deps not installed", allow_module_level=True)

from tests.gateway.test_voice_command import _ensure_discord_mock

_ensure_discord_mock()

from gateway.platforms.base import MessageEvent, MessageType, SessionSource


def _event(text, chat_id="123"):
    source = SessionSource(chat_id=chat_id, user_id="u1", platform=MagicMock())
    source.platform.value = "telegram"
    source.thread_id = None
    ev = MessageEvent(text=text, message_type=MessageType.TEXT, source=source)
    ev.message_id = "m1"
    return ev


@pytest.fixture
def runner(tmp_path):
    from gateway.run import GatewayRunner
    r = object.__new__(GatewayRunner)
    r._tokens_display = {}
    r._tokens_display_global = False
    r._TOKENS_DISPLAY_PATH = tmp_path / "gateway_tokens_display.json"
    return r


def _run(runner, text):
    return asyncio.run(runner._handle_tokens_command(_event(text)))


def test_on_is_per_session(runner):
    out = _run(runner, "/tokens on")
    assert "session" in out.lower()
    assert runner._tokens_display["telegram:123"] is True
    assert runner._tokens_display_global is False
    assert runner._tokens_enabled_for(_event("").source.platform, "123") is True
    # other chats unaffected
    assert runner._tokens_enabled_for(_event("", chat_id="999").source.platform, "999") is False


def test_always_is_global(runner):
    out = _run(runner, "/tokens always")
    assert "global" in out.lower()
    assert runner._tokens_display_global is True
    # any chat without an override now shows
    assert runner._tokens_enabled_for(_event("", chat_id="999").source.platform, "999") is True


def test_per_session_override_wins_over_global(runner):
    _run(runner, "/tokens always")
    _run(runner, "/tokens on")  # redundant but explicit
    # mute just this session while global stays for others
    runner._tokens_display["telegram:123"] = False
    assert runner._tokens_enabled_for(_event("").source.platform, "123") is False
    assert runner._tokens_enabled_for(_event("", chat_id="999").source.platform, "999") is True


def test_off_clears_global(runner):
    _run(runner, "/tokens always")
    out = _run(runner, "/tokens off")
    assert "off" in out.lower()
    assert runner._tokens_display["telegram:123"] is False
    assert runner._tokens_display_global is False


def test_status_reports_both(runner):
    _run(runner, "/tokens always")
    out = _run(runner, "/tokens status")
    assert "global" in out.lower() and "on" in out.lower()


def test_persistence_roundtrip_and_migration(runner):
    _run(runner, "/tokens always")
    _run(runner, "/tokens on")
    # saved file is the new format
    saved = json.loads(runner._TOKENS_DISPLAY_PATH.read_text())
    assert saved["global"] is True and saved["chats"]["telegram:123"] is True
    # reload restores both global + chats
    runner._tokens_display = runner._load_tokens_display()
    assert runner._tokens_display_global is True
    assert runner._tokens_display["telegram:123"] is True
    # legacy flat format migrates as per-session overrides
    runner._TOKENS_DISPLAY_PATH.write_text(json.dumps({"telegram:5": True}))
    chats = runner._load_tokens_display()
    assert chats["telegram:5"] is True and runner._tokens_display_global is False

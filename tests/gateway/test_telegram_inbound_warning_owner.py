"""Inbound cache diagnostics use the routed owner before agent admission."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, PlatformConfig
from gateway.profile_routing import ProfileRoute
from gateway.run import GatewayRunner
from hermes_constants import get_hermes_home
from plugins.platforms.telegram.adapter import TelegramAdapter
from tests.gateway.test_telegram_documents import _make_document, _make_message, _make_update


@pytest.mark.asyncio
@pytest.mark.parametrize("ambient", [False, True])
@pytest.mark.parametrize("setting", [None, False, True])
@pytest.mark.parametrize("kind", ["document", "photo", "voice", "audio", "video"])
async def test_inbound_cache_failure_keeps_owner_policy_and_context(
        tmp_path, monkeypatch, caplog, ambient, setting, kind):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
    (root / "config.yaml").write_text(json.dumps({"display": {"suppress_warning_notifications": ambient}}))
    for name, value in (("a", setting), ("b", not bool(setting))):
        home = root / "profiles" / name
        home.mkdir(parents=True)
        config = {} if value is None else {"display": {"suppress_warning_notifications": value}}
        (home / "config.yaml").write_text(json.dumps(config))
    adapter = TelegramAdapter(PlatformConfig(enabled=True))
    adapter._is_callback_user_authorized = lambda *a, **k: True
    runner = GatewayRunner(GatewayConfig(multiplex_profiles=True))
    runner.config.profile_routes = [
        ProfileRoute(name=name, platform="telegram", profile=name, chat_id=chat)
        for name, chat in (("a", "100"), ("b", "200"))]
    adapter.gateway_runner = runner
    adapter.handle_message = AsyncMock()
    for name, chat_id in (("a", 100), ("b", 200), ("a", 100)):
        media = _make_document()
        media.get_file = AsyncMock(side_effect=RuntimeError("download failure retained"))
        msg = _make_message()
        setattr(msg, kind, [media] if kind == "photo" else media)
        msg.chat.id = chat_id
        msg.reply_to_message = msg.quote = None
        sent_homes = []
        async def reply(*args, **kwargs):
            sent_homes.append(get_hermes_home())
        msg.reply_text = AsyncMock(side_effect=reply)
        await adapter._handle_media_message(_make_update(msg), SimpleNamespace())
        event = adapter.handle_message.call_args.args[0]
        suppressed = setting is True if name == "a" else not bool(setting)
        assert event.source.profile == name
        assert msg.reply_text.await_count == (0 if suppressed else 1)
        assert sent_homes == ([] if suppressed else [root / "profiles" / name])
        assert "could not be downloaded" in event.text
        assert ("asked to retry" in event.text) is not suppressed
        assert not event.media_urls
        assert get_hermes_home() == root
    assert "download failure retained" in caplog.text
    assert adapter.handle_message.await_count == 3

"""Actual adapter producers; only vendor transport and pacing clocks replaced."""
from types import SimpleNamespace
from unittest.mock import AsyncMock
import json

import pytest
from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult


@pytest.fixture(params=[None, False, True, "override"])
def policy(request, monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    value = request.param
    config: dict = {} if value is None else {"display": {"suppress_warning_notifications": value is True}}
    if value == "override":
        config["display"]["platforms"] = {p: {"suppress_warning_notifications": True} for p in ("slack", "signal")}
    (tmp_path / "config.yaml").write_text(json.dumps(config))
    return value in (True, "override")


@pytest.mark.asyncio
async def test_signal_pacing_producer_keeps_actual_image_delivery(policy, monkeypatch, tmp_path):
    import gateway.platforms.signal as module
    adapter = module.SignalAdapter(PlatformConfig())
    scheduler = SimpleNamespace(state=lambda: {}, estimate_wait=lambda n: 120,
        acquire=AsyncMock(), report_rpc_duration=AsyncMock())
    monkeypatch.setattr(module, "get_scheduler", lambda: scheduler)
    adapter._stop_typing_indicator = AsyncMock()
    path = tmp_path / "image.png"
    path.write_bytes(b"image")
    adapter._resolve_image_path = AsyncMock(return_value=(str(path), None, None))
    adapter._with_target = AsyncMock(side_effect=lambda params, chat: params)
    adapter._rpc = AsyncMock(return_value={"timestamp": 123})
    adapter.send = AsyncMock(return_value=SendResult(success=True))
    result = await adapter.send_multiple_images("recipient", [(path.as_uri(), "caption")])
    assert result.success
    assert adapter.send.await_count == (0 if policy else 1)
    if not policy:
        assert "rate limit" in adapter.send.call_args.args[1]
    assert adapter._rpc.call_args.args[1]["attachments"] == [str(path)]
    scheduler.acquire.assert_awaited_once_with(1)
    scheduler.report_rpc_duration.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("caption", [None, "Warning: requested caption"])
async def test_slack_actual_upload_failure_keeps_caption_and_source_log(policy, tmp_path, caption, caplog):
    from plugins.platforms.slack.adapter import SlackAdapter
    adapter = SlackAdapter(PlatformConfig())
    client = SimpleNamespace(files_upload_v2=AsyncMock(side_effect=RuntimeError("upload rejected")))
    adapter._app = SimpleNamespace(client=client)
    adapter._client_for = lambda chat_id, metadata: client
    adapter._dm_target = AsyncMock(return_value="C123")
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="fallback"))
    path = tmp_path / "report.pdf"
    path.write_bytes(b"report")
    result = await adapter.send_document("C123", str(path), caption=caption, metadata={"thread_id": "123.456"})
    client.files_upload_v2.assert_awaited_once()
    assert "upload rejected" in caplog.text
    if policy:
        assert not result.success and result.message_id is None
        assert adapter.send.await_count == bool(caption)
        if caption:
            assert adapter.send.call_args.args[1] == caption
    else:
        assert result.success and result.message_id == "fallback"  # legacy return semantics
        assert "Couldn't deliver" in adapter.send.call_args.args[1]
    if adapter.send.await_count:
        assert adapter.send.call_args.kwargs["metadata"] == {"thread_id": "123.456"}


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", [None, False, True])
async def test_discord_admin_alert_uses_owner_and_logical_destination(tmp_path, monkeypatch, setting):
    from pathlib import Path
    from gateway.config import Platform
    from plugins.platforms.discord.adapter import DiscordAdapter
    from hermes_constants import get_hermes_home
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    (root / "config.yaml").write_text("display: {suppress_warning_notifications: false}")
    for name, muted in [("a", setting), ("b", not bool(setting))]:
        home = root / "profiles" / name
        home.mkdir(parents=True)
        display = {} if muted is None else {"suppress_warning_notifications": muted}
        # Discord's own setting is deliberately the opposite of the alert target.
        display["platforms"] = {"discord": {"suppress_warning_notifications": not bool(muted)}}
        config = {"display": display, "platforms": {"telegram": {"enabled": True,
                  "home_channel": {"platform": "telegram", "chat_id": "admin-" + name, "name": "test"}}}}
        (home / "config.yaml").write_text(json.dumps(config))
    wire = []
    async def send(chat_id, text):
        wire.append((chat_id, text, get_hermes_home()))
        return SendResult(success=True)
    maps = {n: {Platform.TELEGRAM: SimpleNamespace(send=send),
                Platform.SLACK: SimpleNamespace(send=AsyncMock())} for n in ("a", "b")}
    runner = SimpleNamespace(adapters={}, _adapters_for_profile=maps.get)
    for name in ("a", "b", "a"):
        adapter = DiscordAdapter(PlatformConfig())
        adapter._owner_profile = name
        adapter.gateway_runner = runner
        before = len(wire)
        await adapter._notify_unauthorized_slash("user", "123", "chan", "guild", "/test", "denied")
        suppressed = setting is True if name == "a" else not bool(setting)
        assert len(wire) - before == (0 if suppressed else 1)
        if not suppressed:
            assert wire[-1][0] == "admin-" + name
            assert wire[-1][2] == root / "profiles" / name
        maps[name][Platform.SLACK].send.assert_not_awaited()
        assert get_hermes_home() == root

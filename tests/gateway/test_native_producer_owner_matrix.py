"""Native media producers keep routed owner decisions across alternating homes."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from gateway.config import GatewayConfig, PlatformConfig
from gateway.run import GatewayRunner
from gateway.platforms.base import SendResult
from hermes_constants import get_hermes_home


@pytest.fixture(params=[None, False, True, "override", "managed", "null"])
def owners(request, tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    (root / "config.yaml").write_text('display: {suppress_warning_notifications: true}')
    mode = request.param
    if mode == "managed":
        (managed / "config.yaml").write_text('display: {suppress_warning_notifications: true}')
    for name in ("a", "b"):
        home = root / "profiles" / name
        home.mkdir(parents=True)
        value = mode if name == "a" else not bool(mode)
        cfg = {} if value is None else {"display": {"suppress_warning_notifications": value is True}}
        if name == "a" and mode in ("override", "null"):
            cfg = {"display": {"suppress_warning_notifications": True, "platforms": {
                p: {"suppress_warning_notifications": False if mode == "override" else None}
                for p in ("slack", "signal")}}}
        (home / "config.yaml").write_text(json.dumps(cfg))
    def suppressed(name):
        return mode == "managed" or (mode in (True, "null") if name == "a" else not bool(mode))
    return root, suppressed


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["document"])
async def test_slack_upload_failure_owner_scope_keeps_caption_and_result(owners, tmp_path, caplog, kind):
    from plugins.platforms.slack.adapter import SlackAdapter
    root, suppressed = owners
    adapter = SlackAdapter(PlatformConfig())
    adapter.gateway_runner = GatewayRunner(GatewayConfig(multiplex_profiles=True))
    attempts, wire = [], []
    async def upload(**kwargs):
        attempts.append((get_hermes_home(), kwargs))
        raise RuntimeError("upload rejected")
    client = SimpleNamespace(files_upload_v2=upload)
    adapter._app = SimpleNamespace(client=client)
    adapter._client_for = lambda chat_id, metadata: client
    adapter._dm_target = AsyncMock(return_value="C123")
    async def send(chat_id, content, **kwargs):
        wire.append((get_hermes_home(), content, kwargs))
        return SendResult(success=True, message_id="fallback")
    adapter.send = send
    path = tmp_path / "media.bin"
    path.write_bytes(b"local bytes")
    for name in ("a", "b", "a"):
        source = adapter.build_source(chat_id="C123", user_id="U123")
        source.profile = name
        before = len(attempts)
        with adapter._media_delivery_scope(source):
            result = await getattr(adapter, "send_" + kind)("C123", str(path), caption="Warning: requested caption", metadata={"thread_id": "123.456"})
        assert len(attempts) > before
        assert all(home == root / "profiles" / name for home, _ in attempts[before:])
        assert wire[-1][0] == root / "profiles" / name
        assert wire[-1][2]["metadata"] == {"thread_id": "123.456"}
        assert ("Couldn't deliver" in wire[-1][1]) is not suppressed(name)
        assert "Warning: requested caption" in wire[-1][1]
        assert result.success is not suppressed(name)
        assert result.message_id == (None if suppressed(name) else "fallback")
        assert get_hermes_home() == root
    assert "upload rejected" in caplog.text


@pytest.mark.asyncio
async def test_signal_pacing_owner_scope_keeps_acquisition_and_upload(owners, tmp_path, monkeypatch):
    import gateway.platforms.signal as module
    root, suppressed = owners
    adapter = module.SignalAdapter(PlatformConfig())
    adapter.gateway_runner = GatewayRunner(GatewayConfig(multiplex_profiles=True))
    scheduler = SimpleNamespace(state=lambda: {}, estimate_wait=lambda n: 120,
        acquire=AsyncMock(), report_rpc_duration=AsyncMock())
    monkeypatch.setattr(module, "get_scheduler", lambda: scheduler)
    adapter._stop_typing_indicator = AsyncMock()
    path = tmp_path / "image.png"
    path.write_bytes(b"image")
    adapter._resolve_image_path = AsyncMock(return_value=(str(path), None, None))
    adapter._with_target = AsyncMock(side_effect=lambda params, chat: params)
    uploads, warnings = [], []
    async def rpc(method, params, **kwargs):
        uploads.append((get_hermes_home(), params))
        return {"timestamp": 123}
    async def send(*args, **kwargs):
        warnings.append(get_hermes_home())
        return SendResult(success=True)
    adapter._rpc, adapter.send = rpc, send
    for name in ("a", "b", "a"):
        source = adapter.build_source(chat_id="recipient", user_id="human")
        source.profile = name
        before = len(warnings)
        with adapter._media_delivery_scope(source):
            result = await adapter.send_multiple_images("recipient", [(path.as_uri(), "caption")])
        assert result.success
        assert len(warnings) - before == (0 if suppressed(name) else 1)
        if not suppressed(name):
            assert warnings[-1] == root / "profiles" / name
        assert uploads[-1][0] == root / "profiles" / name
        assert uploads[-1][1]["attachments"] == [str(path)]
        assert get_hermes_home() == root
    assert scheduler.acquire.await_count == 3
    assert scheduler.report_rpc_duration.await_count == 3

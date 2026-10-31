"""Real TCP HTTP/WS coverage for the gateway-owned API lifecycle."""

import asyncio
import json
import signal

import httpx
import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus


@pytest.mark.asyncio
async def test_http_lifespan_does_not_start_or_stop_runtime_services(monkeypatch):
    from hermes_cli import web_server as web
    from tui_gateway import methods_groups

    calls = []
    monkeypatch.setenv("HERMES_DESKTOP", "1")
    for name in ("_warm_gateway_module", "_eager_reconcile_own_session_db",
                 "_start_desktop_cron_ticker", "_terminate_desktop_managed_gateway"):
        monkeypatch.setattr(web, name, lambda *a, _name=name, **k: calls.append(_name))
    for name in ("start_hosted_room_service", "stop_hosted_room_service"):
        monkeypatch.setattr(methods_groups, name, lambda *a, _name=name, **k: calls.append(_name))
    # A real gateway owns these services; HTTP must not acquire or release them.
    monkeypatch.setattr(web.app.state, "gateway_runner", object(), raising=False)
    async with web.app.router.lifespan_context(web.app):
        await asyncio.sleep(0)
        assert calls == []
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize('shared', [False, True])
async def test_owner_serves_existing_routes_and_ws_without_signal_ownership(monkeypatch, tmp_path, shared):
    from gateway.run_api import start_gateway_api, stop_gateway_api
    from hermes_cli import web_server as web

    from gateway.run import GatewayRunner
    from gateway.config import GatewayConfig

    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "gateway-sessions"))
    assert runner.adapters == {}
    request = {"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {}}
    if shared:
        from gateway.config import Platform
        from gateway.session import SessionSource
        from gateway.session_authority import initialize_session_authority
        authority = await initialize_session_authority(runner, profile_id='fixture', instance_id='api-owner')
        ref = authority.register(SessionSource(platform=Platform.TELEGRAM, chat_id='api-test', user_id='fixture'))
        request.update(method='session.resume', params={'session_id': ref.session_id})
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    handle = await start_gateway_api(runner)
    try:
        assert handle.api_origin == f"http://127.0.0.1:{web.app.state.bound_port}"
        assert web.app.state.bound_port > 0
        assert web.app.state.gateway_runner is runner
        assert before == {sig: signal.getsignal(sig) for sig in before}
        auth = {"Authorization": f"Bearer {web._SESSION_TOKEN}"}
        async with httpx.AsyncClient(base_url=handle.api_origin) as client:
            assert (await client.get("/api/config/raw")).status_code == 401
            assert (await client.get("/api/config/raw", headers=auth)).status_code == 200
            assert (await client.post("/api/ws-ticket")).status_code in (401, 403)
            assert (await client.get("/api/config/raw", headers={**auth, "Host": "attacker.invalid"})).status_code == 400
        url = handle.api_origin.replace("http:", "ws:") + "/api/ws"
        with pytest.raises(InvalidStatus) as rejected:
            async with connect(url):
                pass
        assert rejected.value.response.status_code == 403
        async with connect(url + "?token=" + web._SESSION_TOKEN) as ws:
            await ws.send(json.dumps(request))
            while True:
                reply = json.loads(await asyncio.wait_for(ws.recv(), 10))
                if reply.get("id") == 1:
                    break
            assert "result" in reply, reply
            if shared:
                assert reply['result'].get('authority_epoch') == authority.epoch, reply
                assert reply['result']['session_id'] == ref.session_id
                assert len(authority.sessions[ref.session_id].subscribers) == 1
        with pytest.raises(RuntimeError, match="already"):
            await start_gateway_api(runner)
    finally:
        await stop_gateway_api(handle)
        if runner._session_db is not None:
            await runner._session_db.close()
    await stop_gateway_api(handle)
    assert web.app.state.session_authority is None
    assert before == {sig: signal.getsignal(sig) for sig in before}
    with pytest.raises(httpx.ConnectError):
        async with httpx.AsyncClient() as client:
            await client.get(handle.api_origin + "/api/config/raw")

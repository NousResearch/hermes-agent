"""The pasted redirect is control traffic, never a conversation turn."""
import asyncio
from dataclasses import replace
from urllib.parse import urlencode

import pytest

from gateway.config import Platform
from gateway.session import SessionSource


def source(**changes):
    return replace(SessionSource(platform=Platform.TELEGRAM, chat_id="chat", user_id="alice"), **changes)


def test_oauth_parameters_are_redacted_before_log_handlers(caplog):
    import logging
    import hermes_logging
    with caplog.at_level(logging.WARNING):
        logging.getLogger("transport").warning("inbound %s", "http://localhost/callback?code=PRIVATECODE&state=PRIVATESTATE")
        logging.getLogger("transport").warning("inbound %s", "http://localhost/callback?%63ode=PRIVATECODE&amp;%73tate=PRIVATESTATE")
        logging.getLogger("transport").warning("inbound %s", "code=PRIVATECODE&state=PRIVATESTATE")
        logging.getLogger("transport").warning("inbound %s", "https://invalid.example/%23code=PRIVATECODE%26state=PRIVATESTATE")
    assert "PRIVATECODE" not in caplog.text
    assert "PRIVATESTATE" not in caplog.text


def test_control_socket_forwards_structured_request_without_exposing_payload(tmp_path):
    import json
    from gateway.control_socket import GatewayControlServer
    received = []
    server = GatewayControlServer(tmp_path, request_handlers={
        "mcp-oauth": lambda data: received.append(data) or {"status": "started"}})
    result = json.loads(server.handle_request_line(json.dumps({
        "verb": "mcp-oauth", "params": {"server": "reports", "session_id": "s"}}).encode()))
    assert result["result"] == {"status": "started"}
    assert received == [{"server": "reports", "session_id": "s"}]
    rejected = json.loads(server.handle_request_line(b'{"verb":"unknown"}'))
    assert "mcp-oauth" in rejected["supported_verbs"]


def flow(tmp_path):
    from gateway.mcp_oauth import MessagingOAuthFlow
    return MessagingOAuthFlow(source(), str(tmp_path), "reports", "http://127.0.0.1:8765/callback")


async def publish(attempt):
    await attempt.publish_authorization_url("https://idp.example/authorize?" + urlencode({
        "state": "secret-state", "redirect_uri": attempt.redirect_uri,
    }))


@pytest.mark.asyncio
async def test_exact_callback_reaches_same_attempt_once(tmp_path):
    attempt = flow(tmp_path)
    await publish(attempt)
    url = attempt.redirect_uri + "?code=secret-code&state=secret-state"
    assert attempt.deliver_url(source(), str(tmp_path), url)
    assert not attempt.deliver_url(source(), str(tmp_path), url)
    assert await attempt.wait_for_callback() == ("secret-code", "secret-state")
    assert "secret" not in repr(attempt)


@pytest.mark.asyncio
async def test_transport_link_envelopes_preserve_exact_callback(tmp_path):
    from gateway.mcp_oauth import callback_url
    valid = "http://127.0.0.1:8765/callback?code=secret-code&state=secret-state"
    assert callback_url("<" + valid + ">") == valid
    assert callback_url("<" + valid + "|" + valid + ">") == valid
    assert callback_url("<" + valid + "|different>") != valid


@pytest.mark.asyncio
async def test_rejects_foreign_principals_and_malformed_redirects(tmp_path):
    attempt = flow(tmp_path)
    await publish(attempt)
    valid = attempt.redirect_uri + "?code=secret-code&state=secret-state"
    for other in (source(user_id="mallory"), source(chat_id="other"),
                  source(profile="other"), source(thread_id="other"), source(scope_id="other")):
        assert not attempt.deliver_url(other, str(tmp_path), valid)
    assert not attempt.deliver_url(source(), str(tmp_path / "other"), valid)
    for bad in (valid.replace("secret-state", "wrong"), valid + "&state=secret-state",
                valid + "#fragment", valid.replace("127.0.0.1", "evil.example"),
                valid.replace("8765", "9999"), valid.replace("/callback", "/elsewhere"),
                "prefix " + valid, valid + "&code=other", valid + "&error=denied",
                valid.replace("secret-code", "%ZZ"), valid.replace("secret-code", "%0a"),
                valid.replace("secret-state", "%C3%A9"), valid.replace("secret-code", "")):
        assert not attempt.deliver_url(source(), str(tmp_path), bad)
    assert attempt.deliver_url(source(), str(tmp_path), valid)


@pytest.mark.asyncio
async def test_expiry_and_cancellation_reject_late_callback(tmp_path):
    attempt = flow(tmp_path)
    await publish(attempt)
    attempt.deadline = 0
    assert not attempt.deliver_url(source(), str(tmp_path), attempt.redirect_uri + "?code=c&state=secret-state")
    with pytest.raises((TimeoutError, RuntimeError)):
        await attempt.wait_for_callback()
    attempt = flow(tmp_path)
    await publish(attempt)
    attempt.cancel()
    assert not attempt.deliver_url(source(), str(tmp_path), attempt.redirect_uri + "?code=c&state=secret-state")
    with pytest.raises(RuntimeError):
        await attempt.wait_for_callback()


@pytest.mark.asyncio
async def test_base_ingress_drops_orphan_callback_before_busy_queue(tmp_path):
    from unittest.mock import AsyncMock
    from gateway.platforms.base import BasePlatformAdapter, MessageEvent
    # Exercise the shared ingress with a minimal adapter, including an active agent.
    class Adapter(BasePlatformAdapter):
        async def connect(self): return True
        async def disconnect(self): pass
        async def send(self, *args, **kwargs): pass
        async def get_chat_info(self, chat_id): return {}
    from gateway.config import PlatformConfig
    adapter = Adapter(PlatformConfig(), Platform.TELEGRAM)
    adapter._message_handler = AsyncMock()
    adapter._start_session_processing = lambda *args: pytest.fail("callback became a turn")
    event = MessageEvent(text="http://127.0.0.1:8765/callback?code=SECRET&state=STATE", source=source())
    await adapter.handle_message(event)
    assert event.text == "[OAuth callback redacted]"
    adapter._message_handler.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["success", "cancel", "restart", "expiry"])
async def test_relay_runs_real_oauth_exchange_and_discovery(tmp_path, monkeypatch, finish):
    import base64
    import hashlib
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from urllib.parse import parse_qs, urlsplit
    from gateway.mcp_oauth import MessagingOAuthRelay
    from tools.mcp_oauth import HermesTokenStorage
    from hermes_cli.mcp_config import _save_mcp_server, _get_mcp_servers
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    observed = {}
    token_entered = threading.Event()
    token_release = threading.Event()

    class Provider(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def reply(self, value, status=200, **headers):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for key, value in headers.items(): self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)
        def do_GET(self):
            if "oauth-protected-resource" in self.path:
                self.reply({"resource": endpoint, "authorization_servers": [issuer]})
            elif "well-known" in self.path:
                self.reply({"issuer": issuer, "authorization_endpoint": "https://idp.example/authorize",
                            "token_endpoint": issuer + "/token", "registration_endpoint": issuer + "/register",
                            "response_types_supported": ["code"], "code_challenge_methods_supported": ["S256"],
                            "grant_types_supported": ["authorization_code", "refresh_token"]})
            else: self.reply({}, 405)
        def do_DELETE(self): self.reply({})
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.path == "/register":
                registration = json.loads(body)
                observed["registration"] = registration
                self.reply({**registration, "client_id": "hermes-test"}, 201)
            elif self.path == "/token":
                observed["token"] = parse_qs(body.decode())
                token_entered.set()
                assert token_release.wait(15)
                self.reply({"access_token": "private-token", "token_type": "Bearer", "expires_in": 3600})
            elif self.headers.get("Authorization") != "Bearer private-token":
                self.reply({}, 401, **{"WWW-Authenticate": 'Bearer resource_metadata="' + issuer + '/.well-known/oauth-protected-resource"'})
            else:
                req = json.loads(body)
                if "id" not in req: self.reply({}, 202); return
                result = ({"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                           "serverInfo": {"name": "fixture", "version": "1"}} if req["method"] == "initialize"
                          else {"tools": [{"name": "hello", "description": "Hello", "inputSchema": {"type": "object"}}]})
                self.reply({"jsonrpc": "2.0", "id": req["id"], "result": result})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    issuer = f"http://127.0.0.1:{server.server_port}"
    endpoint = issuer + "/mcp"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    from gateway.platforms.base import BasePlatformAdapter
    from gateway.config import PlatformConfig
    class Adapter(BasePlatformAdapter):
        async def connect(self): return True
        async def disconnect(self): pass
        async def send(self, *args, **kwargs): pass
        async def get_chat_info(self, chat_id): return {}
    adapter = Adapter(PlatformConfig(), Platform.TELEGRAM)
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True))
    adapter._message_handler = AsyncMock()
    runner = SimpleNamespace(_adapter_for_source=lambda src: adapter,
                             _thread_metadata_for_source=lambda src: {},
                             _is_user_authorized_for_source=lambda src: True,
                             _resolve_profile_home_for_source=lambda src: tmp_path)
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    runner.config = GatewayConfig()
    runner.session_store = SessionStore(tmp_path / "sessions", runner.config)
    entry = runner.session_store.get_or_create_session(source())
    adapter.gateway_runner = runner
    relay = MessagingOAuthRelay(runner)
    runner._mcp_oauth_relay = relay
    cfg = {"url": endpoint, "auth": "oauth"}
    from tools.mcp_tool_discovery import _note_connect_failure
    _note_connect_failure("reports", RuntimeError("OAuth required"))
    if finish != "success":
        assert _save_mcp_server("reports", cfg)
        from mcp.shared.auth import OAuthToken
        await HermesTokenStorage("reports").set_tokens(OAuthToken(access_token="old-token", token_type="Bearer"))
    from gateway.control_socket import GatewayControlServer
    control = GatewayControlServer(tmp_path, request_handlers={"mcp-oauth": relay.request})
    assert await control.start()
    try:
        from gateway.control_socket import query_gateway_control
        params = {"session_id": entry.session_id, "user_id": "alice", "home": str(tmp_path), "server": "reports"}
        for key, value in (("session_id", "foreign-session"), ("user_id", "mallory"), ("home", str(tmp_path / "foreign"))):
            assert await asyncio.to_thread(query_gateway_control, tmp_path, "mcp-oauth", params={**params, key: value}) is None
        assert not relay.attempts
        from gateway.session_context import set_session_vars
        from hermes_cli.mcp_gateway_oauth import gateway_oauth_login
        set_session_vars(platform="telegram", user_id="alice", session_id=entry.session_id)
        if finish == "success":
            import os
            import subprocess
            import sys
            result = await asyncio.to_thread(subprocess.run,
                [sys.executable, "-m", "hermes_cli.main", "mcp", "login", "reports", "--gateway", "--url", endpoint],
                env={**os.environ, "HERMES_HOME": str(tmp_path), "HERMES_SESSION_ID": entry.session_id,
                     "HERMES_SESSION_USER_ID": "alice"}, capture_output=True, text=True, timeout=20)
            assert result.returncode == 0, result.stderr
            assert "queued" in result.stdout
        else:
            await asyncio.to_thread(gateway_oauth_login, SimpleNamespace(name="reports"))
        await asyncio.sleep(0)
        attempt = relay.attempts[(str(tmp_path), "reports")]
        with pytest.raises(RuntimeError): relay.start(source(), tmp_path, "reports", cfg)
        authorization_url = await attempt.wait_for_authorization_url(15)
        query = parse_qs(urlsplit(authorization_url).query)
        assert query["code_challenge_method"] == ["S256"]
        assert query["resource"] == [endpoint]
        callback = query["redirect_uri"][0] + "?" + urlencode({"code": "private-code", "state": query["state"][0]})
        from gateway.platforms.base import MessageEvent
        from gateway.mcp_oauth import intercept_callback
        event = MessageEvent(text=callback, source=source())
        adapter._active_sessions[adapter._event_session_key(event)] = object()
        await adapter.handle_message(event)
        assert event.text == "[OAuth callback redacted]"
        adapter._message_handler.assert_not_called()
        assert not adapter._pending_messages
        assert await asyncio.to_thread(token_entered.wait, 10)
        if finish == "cancel": attempt.cancel()
        if finish == "restart": await relay.close()
        if finish == "expiry": attempt.deadline = 0
        token_release.set()
        if finish == "restart":
            await asyncio.gather(*tuple(relay.tasks), return_exceptions=True)
            for _ in range(200):
                if attempt.worker_done: break
                await asyncio.sleep(0.05)
            assert attempt.worker_done
        else:
            await asyncio.wait_for(asyncio.gather(*tuple(relay.tasks)), 30)
        if finish != "success":
            assert (await HermesTokenStorage("reports").get_tokens()).access_token == "old-token"
            assert attempt.status == "error"
            restarted = MessagingOAuthRelay(runner)
            runner._mcp_oauth_relay = restarted
            assert await intercept_callback(runner, MessageEvent(text=callback, source=source()))
            assert "private-code" not in str(adapter.send.call_args_list)
            return
        assert attempt.status == "approved", attempt.error
        token = observed["token"]
        challenge = base64.urlsafe_b64encode(hashlib.sha256(token["code_verifier"][0].encode()).digest()).rstrip(b"=").decode()
        assert query["code_challenge"] == [challenge]
        assert token["redirect_uri"] == query["redirect_uri"]
        assert token["resource"] == [endpoint]
        assert token["code"] == ["private-code"]
        assert query["redirect_uri"][0] in observed["registration"]["redirect_uris"]
        assert (await HermesTokenStorage("reports").get_tokens()).access_token == "private-token"
        from tools import mcp_tool
        assert mcp_tool._servers["reports"].session is not None
        assert _get_mcp_servers()["reports"] == cfg
        notices = [call.args[1] for call in adapter.send.call_args_list]
        assert all(call.kwargs["metadata"]["_interim_send"] for call in adapter.send.call_args_list)
        assert any(authorization_url in text for text in notices)
        assert any("connected" in text for text in notices)
        assert all("private-code" not in text for text in notices)
        assert await intercept_callback(runner, MessageEvent(text=callback, source=source()))
    finally:
        token_release.set()
        await relay.close()
        await control.stop()
        from tools.mcp_tool_lifecycle import shutdown_mcp_servers
        await asyncio.to_thread(shutdown_mcp_servers)
        server.shutdown()
        server.server_close()

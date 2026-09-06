"""Keyless Slack uses the normal Bolt lifecycle; only API authentication/routing changes."""

import asyncio
import copy
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import pytest

# Import real SDKs before the adapter (other Slack test files use module stubs).
pytest.importorskip("slack_bolt")
from slack_bolt.request.async_request import AsyncBoltRequest
from slack_sdk.web.async_client import AsyncWebClient

from gateway.config import Platform, PlatformConfig, load_gateway_config
from plugins.platforms.slack import adapter as slack


@pytest.fixture
def slack_http(monkeypatch, tmp_path):
    requests = []
    updated = threading.Event()
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            method = urlsplit(self.path).path.rsplit("/", 1)[-1]
            requests.append((self.path, self.headers.get("Authorization"), data))
            if method == "auth.test":
                result = {"ok": True, "team_id": "T1", "user_id": "UBOT",
                          "bot_id": "B1", "user": "hermes", "team": "test"}
            elif method == "apps.connections.open":
                result = {"ok": True, "url": "wss://socket.invalid/ticket"}
            elif method == "conversations.open":
                result = {"ok": True, "channel": {"id": "D1"}}
            else:
                result = {"ok": True, "channel": "C1", "ts": "123.456"}
            body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            if method == "chat.update":
                updated.set()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    # SDKs re-read proxy env on construction. Exercise NO_PROXY re-pinning rather
    # than merely removing proxy env, and fail closed on any accidental live I/O.
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(key, endpoint)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.delenv("no_proxy", raising=False)
    # Signature-tolerant: callers pass NO_PROXY target_hosts (PR #73433) or nothing.
    monkeypatch.setattr(
        slack, "resolve_proxy_url", lambda *args, **kwargs: endpoint)
    real_request = slack.aiohttp.ClientSession._request

    async def local_only(self, method, url, **kwargs):
        assert urlsplit(str(url)).hostname in {"127.0.0.1", "edge.invalid"}, f"Unexpected API: {url}"
        target = kwargs.get("proxy") or str(url)
        assert urlsplit(str(target)).hostname == "127.0.0.1", f"Nonlocal I/O: {url}"
        return await real_request(self, method, url, **kwargs)

    monkeypatch.setattr(slack.aiohttp.ClientSession, "_request", local_only)
    # Never open even a test websocket: still obtain the ticket through the real
    # SocketModeClient WebClient. All SDK handler/task teardown remains real.
    tickets = asyncio.Queue()

    async def ticket_only(handler):
        ticket = await handler.client.issue_new_wss_url()
        await tickets.put(ticket)
        await asyncio.Event().wait()

    monkeypatch.setattr(slack.AsyncSocketModeHandler, "start_async", ticket_only)
    try:
        yield endpoint, requests, tickets, updated
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.asyncio
async def test_keyless_bolt_routes_auth_tickets_and_authorized_approvals(tmp_path, monkeypatch, slack_http):
    endpoint, requests, tickets, updated = slack_http
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
    for key in ("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "SLACK_KEYLESS_PROXY"):
        monkeypatch.delenv(key, raising=False)
    (tmp_path / "config.yaml").write_text(
        f"platforms:\n  slack:\n    enabled: true\n    extra:\n      keyless_api_base_url: {endpoint}/api\n"
    )
    # Saved OAuth installations are not credentials for the configured edge.
    (tmp_path / "slack_tokens.json").write_text(json.dumps({"TLOCAL": {"token": "xoxb-local-only"}}))
    config = load_gateway_config()
    assert Platform.SLACK in config.get_connected_platforms()
    pconfig = config.platforms[Platform.SLACK]
    from gateway.run import _platform_has_bot_credential
    assert _platform_has_bot_credential(Platform.SLACK, pconfig), "keyless reconnect must remain eligible"
    adapter = slack.SlackAdapter(pconfig)
    adapter.set_authorization_check(lambda user, *_: user == "UOWNER")
    try:
        assert await adapter.connect()
        assert await asyncio.wait_for(tickets.get(), 5) == "wss://socket.invalid/ticket"
        first_handler = adapter._handler
        assert isinstance(adapter._app.client, AsyncWebClient)
        assert adapter._handler.client.web_client is adapter._app.client
        assert adapter._app.client.proxy is None
        assert adapter._handler.client.proxy == endpoint  # WSS does not share API NO_PROXY
        assert adapter._get_client("C1").base_url == f"{endpoint}/api/"

        # The scoped lock is process-owned (same-PID reacquisition permits
        # reconnect). A separate process must not claim this edge identity.
        def probe_lock():
            result = subprocess.run([sys.executable, "-c",
                "import sys; from gateway import status; "
                "status._looks_like_gateway_process = lambda pid: True; "
                "print(status.acquire_scoped_lock(sys.argv[1], sys.argv[2])[0])",
                adapter._platform_lock_scope, adapter._platform_lock_identity],
                check=True, capture_output=True, text=True, timeout=10)
            return result.stdout.strip() == "True"
        assert not probe_lock()

        # Exercise the native inbound path too, not merely outbound API access.
        delivered = asyncio.Queue()

        async def receive(event):
            await delivered.put(event)
            return None

        adapter.set_message_handler(receive)
        message = {"type": "event_callback", "team_id": "T1", "event_id": "Ev1",
                   "event": {"type": "app_mention", "team": "T1", "channel": "C1",
                             "user": "UOWNER", "text": "<@UBOT> transport probe",
                             "ts": "100.2", "client_msg_id": "probe-1"}}
        inbound = AsyncBoltRequest(body=message, mode="socket_mode")
        assert (await adapter._app.async_dispatch(inbound)).status == 200
        event = await asyncio.wait_for(delivered.get(), 5)
        assert event.text == "transport probe"
        assert event.source.chat_id == "C1"
        assert event.source.user_id == "UOWNER"

        from tools import approval
        from tools.approval_gateway_wait import _ApprovalEntry
        session_key = "agent:main:slack:group:C1:100.1"
        entry = _ApprovalEntry({"command": "example"})
        monkeypatch.setitem(approval._gateway_queues, session_key, [entry])
        sent = await adapter.send_exec_approval(
            "C1", "example", session_key, metadata={"thread_id": "100.1", "team_id": "T1"})
        assert sent.success
        post = next(json.loads(body) for path, _, body in requests if path.endswith("chat.postMessage"))
        action = post["blocks"][1]["elements"][0]
        body = {"type": "block_actions", "team": {"id": "T1"}, "user": {"id": "UOTHER"},
                "channel": {"id": "C1"}, "actions": [action],
                "message": {"ts": sent.message_id, "thread_ts": "100.1", "blocks": post["blocks"]}}
        # Bolt creates a new SDK client before its authorization middleware.
        denied = AsyncBoltRequest(body=copy.deepcopy(body), mode="socket_mode")
        assert (await adapter._app.async_dispatch(denied)).status == 200
        assert isinstance(denied.context.client, AsyncWebClient)
        assert denied.context.client is not adapter._app.client
        assert denied.context.client.base_url == adapter._app.client.base_url
        assert denied.context.client.proxy is None
        await asyncio.sleep(0)  # Bolt runs listeners after ack in ordinary token mode
        assert entry.result is None
        body["user"]["id"] = "UOWNER"
        allowed = AsyncBoltRequest(body=body, mode="socket_mode")
        assert (await adapter._app.async_dispatch(allowed)).status == 200
        assert await asyncio.to_thread(updated.wait, 5)
        assert entry.result == "once"
        assert entry.event.is_set()
        assert any(path.endswith("chat.update") for path, _, _ in requests)
        assert sum(urlsplit(path).path.endswith("auth.test") for path, _, _ in requests) >= 2
        for path, auth, _ in requests:
            token = "xapp-keyless" if urlsplit(path).path.endswith("apps.connections.open") else "xoxb-keyless"
            assert auth == f"Bearer {token}"
        assert allowed.context.bot_user_id == "UBOT"  # Bolt recognizes the placeholder as a bot token
        # Out-of-process notifications route DM resolution, JSON posts, and SDK
        # media captions through the edge too (without opening a socket).
        notification = await slack._standalone_send(pconfig, "UOWNER", "notification")
        assert notification["success"]
        caption = await slack._standalone_send(pconfig, "C1", "", caption="caption", media_files=[
            (str(tmp_path / "missing.txt"), False)])
        assert caption["success"]
        assert caption["warnings"]

        # Reconnect replaces (and closes) old Socket Mode state, preserving the
        # endpoint without sharing it with an ordinary-token adapter.
        assert await adapter.connect(is_reconnect=True)
        await asyncio.wait_for(tickets.get(), 5)
        assert first_handler.client.closed
        assert adapter._handler is not first_handler
    finally:
        await adapter.disconnect()
    assert not adapter._running
    assert not adapter._team_clients
    # Same endpoint is usable again after teardown releases the ordinary lock.
    other = slack.SlackAdapter(pconfig)
    assert await other.connect()
    await other.disconnect()


@pytest.mark.asyncio
async def test_keyless_requires_explicit_valid_endpoint_and_does_not_reconfigure_token_mode(monkeypatch, slack_http):
    endpoint, requests, tickets, updated = slack_http
    monkeypatch.setenv("SLACK_KEYLESS_PROXY", "1")  # obsolete flag cannot enable keyless auth
    monkeypatch.delenv("SLACK_APP_TOKEN", raising=False)
    missing = slack.SlackAdapter(PlatformConfig(enabled=True))
    assert not await missing.connect()
    assert missing._fatal_error_code == "missing_slack_bot_token"
    assert not requests
    ordinary = slack.SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test"))
    client = ordinary._new_web_client("xoxb-test", None)
    assert client.base_url == "https://slack.com/api/"
    assert client.token == "xoxb-test"
    assert client.proxy is None
    assert not await ordinary.connect(), "a real app token remains mandatory without endpoint config"
    assert ordinary._fatal_error_code == "missing_slack_app_token"
    # A secondary profile cannot borrow the default profile's real app token.
    from agent import secret_scope
    monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-default-profile")
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    scope = secret_scope.set_secret_scope({})
    try:
        assert not await ordinary.connect()
        assert ordinary._fatal_error_code == "missing_slack_app_token"
    finally:
        secret_scope.reset_secret_scope(scope)

    for value in ("not-a-url", "ftp://example.test/api/", "https://slack.com/api/", "https://user:password@example.test/api/"):
        bad = slack.SlackAdapter(PlatformConfig(enabled=True, extra={"keyless_api_base_url": value}))
        assert not await bad.connect()
    assert not requests

    # A non-NO_PROXY endpoint must use the resolved HTTP proxy, including the
    # per-request Bolt auth client and socket ticket client (no global rewrite).
    routed = slack.SlackAdapter(PlatformConfig(enabled=True, extra={
        "keyless_api_base_url": "http://edge.invalid/api/"}))
    try:
        assert await routed.connect()
        await asyncio.wait_for(tickets.get(), 5)
        request = AsyncBoltRequest(body={"type": "event_callback", "team_id": "T1",
            "event": {"type": "unhandled_for_test", "user": "UOWNER"}}, mode="socket_mode")
        assert (await routed._app.async_dispatch(request)).status == 200
        assert request.context.client.proxy == endpoint
        assert all(path.startswith("http://edge.invalid/api/") for path, _, _ in requests)
        assert client.base_url == "https://slack.com/api/"  # other instances untouched
    finally:
        await routed.disconnect()

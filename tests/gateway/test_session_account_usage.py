"""Account limits cross the registered RPC with real policy, authorization and storage.

Only the provider HTTP transport is substituted. Reads must use the session's
credential, preserve profile ownership and expose normalized quota facts only.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import threading
from typing import Any

import httpx
import pytest
import pytest_asyncio


OPENROUTER_URL = "https://openrouter.ai/api/v1"
CODEX_URL = "https://chatgpt.com/backend-api/codex"
ANTHROPIC_URL = "https://api.anthropic.com"


async def _rpc(connection, method, **params):
    return await connection.dispatch({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})


@dataclass
class UsageOwner:
    home: Path
    runner: Any
    authority: Any
    connection: Any
    connections: list = field(default_factory=list)

    async def create(self, provider="openrouter", base_url=OPENROUTER_URL, api_key="session-key", **params):
        request = {"request_id": "usage-" + str(len(self.authority.sessions)), "source": "gui",
                   "cwd": str(self.home), "model": "quota-model", "toolsets": [],
                   "provider": provider, "base_url": base_url, **params}
        if api_key is not None:
            request["api_key"] = api_key
        reply = await _rpc(self.connection, "session.create", **request)
        assert "result" in reply, reply
        return reply["result"]["session_id"]

    def connect(self, **identity):
        from gateway.session_controls import AuthorityConnection

        connection = AuthorityConnection(self.authority, object(), {"user_id": "owner", **identity})
        self.connections.append(connection)
        return connection


@pytest_asyncio.fixture
async def usage_owner():
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner, _profile_runtime_scope
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    (home / "config.yaml").write_text(json.dumps({
        "model": {"provider": "openrouter", "default": "profile-model", "base_url": OPENROUTER_URL},
        "terminal": {"backend": "local", "cwd": str(home)},
        "auxiliary": {"title_generation": {"enabled": False}},
        "display": {"auto_prune": False},
    }), encoding="utf-8")
    (home / ".env").write_text("OPENROUTER_API_KEY=profile-account-key\n", encoding="utf-8")
    (home / "auth.json").write_text(json.dumps({"version": 1, "credential_pool": {
        "openrouter": [{"id": "another-login", "auth_type": "api_key", "source": "manual",
                        "priority": 0, "access_token": "pool-account-key"}],
    }}), encoding="utf-8")
    with _profile_runtime_scope(home, hydrate_secrets=False):
        runner = GatewayRunner(GatewayConfig(sessions_dir=home / "sessions", multiplex_profiles=True))
        authority = await initialize_session_authority(runner, profile_id=str(home), instance_id="usage-owner")
    connection = AuthorityConnection(authority, object(), {"user_id": "owner"})
    owner = UsageOwner(home, runner, authority, connection, [connection])
    yield owner
    for peer in owner.connections:
        for sid, subscription in list(peer.subscriptions.items()):
            await _rpc(peer, "session.detach", session_id=sid, subscription_id=subscription,
                       end_reason="test_cleanup")
        await peer.close()
        await asyncio.gather(*peer.__dict__.get("_notice_tasks", ()), return_exceptions=True)
    runner.close_all_session_db_handles()


def _http(monkeypatch, handler):
    client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client(transport=httpx.MockTransport(handler), **kwargs))


def _openrouter_payload(request):
    if request.url.path.endswith("/credits"):
        return {"data": {"total_credits": 300, "total_usage": 10.92, "private": "provider-private-data"}}
    if request.url.path.endswith("/key"):
        return {"data": {"limit": 100, "limit_remaining": 70, "limit_reset": "monthly", "usage": 12.5,
                         "usage_daily": 0.5, "usage_weekly": 2, "usage_monthly": 8,
                         "label": "provider-private-data"}}
    raise AssertionError(request.url)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider, base_url, token", [
    ("openrouter", OPENROUTER_URL, "session-router-key"),
    ("openai-codex", CODEX_URL, "session-codex-key"),
    ("anthropic", ANTHROPIC_URL, "sk-ant-oat01-session-oauth"),
])
async def test_registered_usage_projects_the_exact_launch_account(usage_owner, monkeypatch, provider, base_url, token):
    """Catch unwired RPCs, substituted pool logins, lost reset facts and raw credential disclosure."""
    from gateway.session_contract import SessionRef
    from hermes_state_runtime import list_session_admissions

    owner = usage_owner
    sid = await owner.create(provider, base_url, token)
    before = {name: (owner.home / name).read_bytes() for name in ("auth.json", "config.yaml", ".env")}
    requests = []

    def respond(request):
        requests.append(request)
        if provider == "openrouter":
            payload = _openrouter_payload(request)
        elif provider == "openai-codex":
            payload = {"plan_type": "pro", "email": "provider-private-data", "rate_limit": {
                "primary_window": {"used_percent": 38, "reset_at": 1_900_000_000,
                                   "limit_window_seconds": 604800}},
                "rate_limit_reset_credits": {"available_count": 2}}
        else:
            payload = {"five_hour": {"utilization": 25, "resets_at": "2030-01-02T03:04:05Z"},
                       "seven_day": {"utilization": 42}, "seven_day_opus": {"utilization": 100},
                       "extra_usage": {"is_enabled": True, "used_credits": 12.5,
                                       "monthly_limit": 50, "currency": "USD"},
                       "email": "provider-private-data"}
        return httpx.Response(200, json=payload)

    _http(monkeypatch, respond)
    reply = await _rpc(owner.connection, "session.account_usage", session_id=sid)
    assert "result" in reply, reply
    result = reply["result"]
    assert result["status"] == "ok", result
    snapshot = result["snapshot"]
    assert snapshot["provider"] == provider
    assert datetime.fromisoformat(snapshot["fetched_at"]).tzinfo is not None
    windows = {window["label_key"]: window for window in snapshot["windows"]}
    rows = {row["key"]: row["args"] for row in snapshot["rows"]}
    if provider == "openrouter":
        assert snapshot["credits_balance"] == pytest.approx(289.08)
        assert windows["api_key_quota"]["remaining_percent"] == 70
        assert windows["api_key_quota"]["limit_remaining"] == 70
        assert windows["api_key_quota"]["reset_interval"] == "monthly"
        assert rows["api_key_usage"] == {"total": 12.5, "daily": 0.5, "weekly": 2, "monthly": 8}
        assert {request.url.path for request in requests} == {"/api/v1/credits", "/api/v1/key"}
    elif provider == "openai-codex":
        assert windows["weekly"]["remaining_percent"] == 62
        assert windows["weekly"]["scope"] == "account"
        assert datetime.fromisoformat(windows["weekly"]["reset_at"]) == datetime(2030, 3, 17, 17, 46, 40, tzinfo=UTC)
        assert rows["banked_resets"]["count"] == 2
        assert [request.url.path for request in requests] == ["/backend-api/wham/usage"]
    else:
        assert windows["current_session"]["remaining_percent"] == 75
        assert windows["current_week"]["remaining_percent"] == 58
        assert windows["opus_week"]["remaining_percent"] == 0
        assert windows["opus_week"]["scope"] == "model"
        assert datetime.fromisoformat(windows["current_session"]["reset_at"]) == datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
        assert rows["extra_usage"] == {"used": 12.5, "limit": 50, "currency": "USD"}
        assert [request.url.path for request in requests] == ["/api/oauth/usage"]
    assert requests and all(request.method == "GET" and request.headers["Authorization"] == "Bearer " + token
                            for request in requests)
    wire = json.dumps(reply)
    assert all(private not in wire for private in (token, "pool-account-key", "profile-account-key",
                                                   "provider-private-data", base_url))
    assert {name: (owner.home / name).read_bytes() for name in before} == before
    assert owner.authority.agent(SessionRef(str(owner.home), sid)) is None
    assert not list_session_admissions(owner.authority.db, session_id=sid, pending_only=False)


@pytest.mark.asyncio
async def test_failed_codex_poll_cannot_reuse_a_success_or_refresh_another_login(usage_owner, monkeypatch):
    """Catch stale success after 401 and refresh/rotation performed by a read-only quota poll."""
    owner = usage_owner
    sid = await owner.create("openai-codex", CODEX_URL, "frozen-codex-key")
    before = (owner.home / "auth.json").read_bytes()
    requests = []

    def respond(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(200, json={"rate_limit": {"primary_window": {"used_percent": 10}}})
        return httpx.Response(401, json={"error": "provider-private-data"})

    _http(monkeypatch, respond)
    first = await _rpc(owner.connection, "session.account_usage", session_id=sid)
    assert first["result"]["snapshot"]["windows"][0]["remaining_percent"] == 90
    failed = await _rpc(owner.connection, "session.account_usage", session_id=sid)
    assert failed["result"]["status"] == "unavailable"
    assert failed["result"]["snapshot"] is None
    assert failed["result"]["reason"] == "fetch_failed"
    assert len(requests) == 2 and all(request.method == "GET" and
                                    request.headers["Authorization"] == "Bearer frozen-codex-key" for request in requests)
    assert (owner.home / "auth.json").read_bytes() == before


@pytest.mark.asyncio
async def test_usage_rejects_unauthorized_and_invalid_requests_before_http(usage_owner, monkeypatch):
    """Catch cross-user/profile credential probing and caller-supplied routing overrides."""
    owner = usage_owner
    sid = await owner.create()
    requests = []
    _http(monkeypatch, lambda request: requests.append(request) or httpx.Response(500))
    denied = owner.connect(capabilities=[])
    foreign = owner.connect(user_id="different-owner")
    cases = [(denied, {"session_id": sid}, "permission_denied"),
             (foreign, {"session_id": sid}, "permission_denied"),
             (owner.connection, {"session_id": sid, "profile": "different-profile"}, "profile_mismatch"),
             (owner.connection, {"session_id": sid, "api_key": "untrusted-key"}, "invalid_params"),
             (owner.connection, {"session_id": "missing-session"}, "not_found"),
             (owner.connection, {"session_id": ""}, "invalid_params"),
             (owner.connection, {"session_id": 42}, "invalid_params"),
             (owner.connection, {}, "invalid_params")]
    for connection, params, reason in cases:
        reply = await _rpc(connection, "session.account_usage", **params)
        assert reply.get("error", {}).get("data", {}).get("reason") == reason, reply
    assert not requests


@pytest.mark.asyncio
async def test_unattributable_credentials_and_unsupported_routes_never_probe_another_account(usage_owner, monkeypatch):
    """Catch ambient-login fallback for keyless sessions, plain API keys and proxy routes."""
    owner = usage_owner
    sessions = []
    for provider, url, token, status, reason in [
        ("openrouter", OPENROUTER_URL, None, "unavailable", "credentials_unavailable"),
        ("anthropic", ANTHROPIC_URL, "plain-api-key", "unsupported", "oauth_required"),
        ("anthropic", "https://proxy.example.invalid/anthropic", "sk-ant-oat01-session-oauth",
         "unsupported", "route_unsupported"),
        ("custom", "https://proxy.example.invalid/v1", "custom-key", "unsupported", "provider_unsupported"),
    ]:
        sessions.append((await owner.create(provider, url, token), status, reason))
    requests = []
    _http(monkeypatch, lambda request: requests.append(request) or httpx.Response(500))

    for sid, status, reason in sessions:
        reply = await _rpc(owner.connection, "session.account_usage", session_id=sid)
        assert reply.get("result", {}).get("status") == status, reply
        assert reply["result"]["reason"] == reason and reply["result"]["snapshot"] is None
    assert not requests


@pytest.mark.asyncio
@pytest.mark.parametrize("mode_config", [
    {"api_mode": "codex_app_server"}, {"api_mode": "codex_responses", "openai_runtime": "codex_app_server"},
])
async def test_app_server_policy_cannot_report_its_unrelated_http_accounts_quota(usage_owner, monkeypatch, mode_config):
    """Guard the decided app-server fix: its private login is not identified by a launch HTTP key."""
    owner = usage_owner
    config_path = owner.home / "config.yaml"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["model"] = {"provider": "openai-codex", "default": "quota-model", "base_url": CODEX_URL, **mode_config}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    sid = await owner.create("openai-codex", CODEX_URL, "unrelated-http-key")
    requests = []
    _http(monkeypatch, lambda request: requests.append(request) or httpx.Response(500))

    reply = await _rpc(owner.connection, "session.account_usage", session_id=sid)

    assert reply.get("result", {}).get("status") == "unsupported", reply
    assert reply["result"]["reason"] == "route_unsupported"
    assert reply["result"]["snapshot"] is None
    assert not requests


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["route", "runtime"])
@pytest.mark.parametrize("timing", ["before_poll", "during_fetch"])
async def test_regression_followed_policy_changes_do_not_report_the_old_account(usage_owner, monkeypatch, change, timing):
    """Catch a followed profile's superseded launch account surviving route or app-server changes."""
    owner = usage_owner
    config_path = owner.home / "config.yaml"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["model"] = {"provider": "openai-codex", "default": "quota-model", "base_url": CODEX_URL,
                       "api_mode": "codex_responses"}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    sid = await owner.create("openai-codex", CODEX_URL, "original-launch-key", follow_profile_config=True)
    started = asyncio.Event()
    pause, release = threading.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    requests = []

    def respond(request):
        requests.append(request)
        if pause.is_set():
            loop.call_soon_threadsafe(started.set)
            if not release.wait(timeout=10):
                raise TimeoutError("the test did not release the followed-profile quota response")
        return httpx.Response(200, json={"rate_limit": {"primary_window": {"used_percent": 13}}})

    def change_profile():
        if change == "route":
            config["model"] = {"provider": "openrouter", "default": "changed-profile-model",
                               "base_url": OPENROUTER_URL, "api_mode": "chat_completions"}
        else:
            config["model"]["openai_runtime"] = "codex_app_server"
        config_path.write_text(json.dumps(config), encoding="utf-8")

    _http(monkeypatch, respond)
    first = await _rpc(owner.connection, "session.account_usage", session_id=sid)
    assert first.get("result", {}).get("status") == "ok", first
    assert first["result"]["snapshot"]["windows"][0]["remaining_percent"] == 87
    pending = None
    try:
        if timing == "during_fetch":
            pause.set()
            pending = asyncio.create_task(_rpc(owner.connection, "session.account_usage", session_id=sid))
            await asyncio.wait_for(started.wait(), 10)
            change_profile()
            release.set()
            reply = await asyncio.wait_for(pending, 10)
            assert reply.get("result", {}).get("status") == "unavailable", reply
            assert reply["result"]["reason"] == "credentials_changed"
        else:
            change_profile()
            reply = await _rpc(owner.connection, "session.account_usage", session_id=sid)
            expected = ("unavailable", "credentials_unavailable") if change == "route" else ("unsupported", "route_unsupported")
            assert reply.get("result", {}).get("status") == expected[0], reply
            assert reply["result"]["reason"] == expected[1]
        assert reply["result"]["snapshot"] is None
        assert len(requests) == (2 if timing == "during_fetch" else 1)
        assert all(request.headers["Authorization"] == "Bearer original-launch-key" for request in requests)
    finally:
        release.set()
        if pending is not None:
            await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("changed, value", [
    ("api_key", "rotated-session-key"), ("base_url", "https://codex-route.example.invalid"),
    ("api_mode", "codex_app_server"),
])
async def test_inflight_usage_is_discarded_when_the_retained_execution_changes(usage_owner, monkeypatch, changed, value):
    """Catch old-account results surviving a key, route or independently authenticated runtime switch."""
    from gateway.session_authorities import owner_scope
    from run_agent import AIAgent

    owner = usage_owner
    sid = await owner.create("openai-codex", CODEX_URL, "launch-key")
    with owner_scope(owner.authority):
        agent = AIAgent(provider="openai-codex", base_url=CODEX_URL, api_key="retained-session-key",
                        api_mode="codex_responses", model="quota-model", enabled_toolsets=[],
                        skip_memory=True, skip_context_files=True, quiet_mode=True, session_id=sid)
    route = owner.authority.sessions[sid].route
    owner.runner._running_agents[route] = agent
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    requests = []

    def respond(request):
        requests.append(request)
        loop.call_soon_threadsafe(started.set)
        if not release.wait(timeout=10):
            raise TimeoutError("the test did not release the quota response")
        return httpx.Response(200, json={"rate_limit": {"primary_window": {"used_percent": 10}}})

    _http(monkeypatch, respond)
    pending = asyncio.create_task(_rpc(owner.connection, "session.account_usage", session_id=sid))
    try:
        await asyncio.wait_for(started.wait(), 10)
        setattr(agent, changed, value)
        release.set()
        reply = await asyncio.wait_for(pending, 10)
        assert reply.get("result", {}).get("status") == "unavailable", reply
        assert reply["result"]["reason"] == "credentials_changed"
        assert reply["result"]["snapshot"] is None
        assert len(requests) == 1 and requests[0].headers["Authorization"] == "Bearer retained-session-key"
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
        owner.runner._running_agents.pop(route, None)
        agent.close()


@pytest.mark.asyncio
async def test_host_profile_reads_keep_home_secrets_and_terminal_scope_across_http_threads(usage_owner, monkeypatch):
    """Catch one profile's quota request borrowing another profile's key, secret scope or terminal policy."""
    from agent.secret_scope import get_secret
    from gateway.run import _profile_runtime_scope
    from gateway.session_authorities import SessionAuthorities
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from hermes_cli.profiles import get_profile_dir
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB
    from tools.terminal_scope import get_terminal_scope

    owner = usage_owner
    monkeypatch.setattr(Path, "home", lambda: owner.home.parent)
    secondary = get_profile_dir("usage-secondary")
    secondary.mkdir(parents=True)
    secondary_config = json.loads((owner.home / "config.yaml").read_text(encoding="utf-8"))
    secondary_config["terminal"]["cwd"] = str(secondary)
    (secondary / "config.yaml").write_text(json.dumps(secondary_config), encoding="utf-8")
    with (owner.home / ".env").open("a", encoding="utf-8") as env_file:
        env_file.write("ACCOUNT_USAGE_SCOPE_PROBE=primary\n")
    (secondary / ".env").write_text("ACCOUNT_USAGE_SCOPE_PROBE=secondary\n", encoding="utf-8")
    registry = owner.runner.session_authorities = SessionAuthorities(owner.home, multiplexed=True)
    registry.add(owner.home, owner.authority)
    with _profile_runtime_scope(secondary, hydrate_secrets=False):
        authority = await initialize_session_authority(owner.runner, profile_id=str(secondary),
                                                       instance_id="secondary-owner", register=False,
                                                       db=SessionDB(secondary / "state.db"))
    registry.add(secondary, authority, name="usage-secondary")
    peer = AuthorityConnection(authority, object(), {"user_id": "owner"})
    owner.connections.append(peer)
    secondary_owner = UsageOwner(secondary, owner.runner, authority, peer)
    sid = await owner.create(api_key="primary-key")
    secondary_sid = await secondary_owner.create(api_key="secondary-key")
    host = owner.connect(profile_scope="host")
    monkeypatch.setenv("ACCOUNT_USAGE_SCOPE_PROBE", "ambient-launch-value")
    before = dict(os.environ)
    seen = {}
    both_started = threading.Barrier(2)

    def respond(request):
        token = request.headers["Authorization"]
        if request.url.path.endswith("/credits"):
            seen[token] = (get_hermes_home(), get_secret("ACCOUNT_USAGE_SCOPE_PROBE"),
                           get_terminal_scope().get("TERMINAL_CWD"))
            both_started.wait(timeout=10)
        total = 120 if token == "Bearer primary-key" else 220
        return httpx.Response(200, json={"data": {"total_credits": total, "total_usage": 20}})

    _http(monkeypatch, respond)
    primary_reply, secondary_reply = await asyncio.gather(
        _rpc(owner.connection, "session.account_usage", session_id=sid),
        _rpc(host, "session.account_usage", session_id=secondary_sid, profile="usage-secondary"))

    assert primary_reply["result"]["snapshot"]["credits_balance"] == 100
    assert secondary_reply["result"]["snapshot"]["credits_balance"] == 200
    assert seen["Bearer primary-key"][:2] == (owner.home, "primary")
    assert seen["Bearer secondary-key"][:2] == (secondary, "secondary")
    assert Path(seen["Bearer primary-key"][2]) == owner.home
    assert Path(seen["Bearer secondary-key"][2]) == secondary
    assert dict(os.environ) == before


@pytest.mark.asyncio
async def test_slow_account_http_cannot_hold_stop_on_the_websocket_lane(usage_owner, monkeypatch):
    """Catch account polls accidentally joining the serial WS lane ahead of Stop."""
    from tests.tui_gateway.test_ws_concurrent_reads import Socket
    from tui_gateway import ws as ws_mod

    class ReplySocket(Socket):
        def __init__(self):
            super().__init__()
            self.scope = {"hermes.session_authority": usage_owner.authority}
            self.replies = asyncio.Queue()

        async def send_text(self, line):
            await super().send_text(line)
            self.replies.put_nowait(json.loads(line))

        async def reply(self, request_id):
            while True:
                frame = await asyncio.wait_for(self.replies.get(), 10)
                if frame.get("id") == request_id:
                    return frame

    owner = usage_owner
    sid = await owner.create()
    row = owner.authority.db.get_session(sid)
    sock = ReplySocket()
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()

    def respond(request):
        if request.url.path.endswith("/credits"):
            loop.call_soon_threadsafe(started.set)
            if not release.wait(timeout=15):
                raise TimeoutError("the test did not release the quota response")
        return httpx.Response(200, json=_openrouter_payload(request))

    _http(monkeypatch, respond)
    task = asyncio.create_task(ws_mod.handle_ws(sock, auth_identity={"user_id": "owner"}))
    sock.inbound.put_nowait({"id": "quota", "method": "session.account_usage", "params": {"session_id": sid}})
    try:
        await asyncio.wait_for(started.wait(), 10)
        sock.inbound.put_nowait({"id": "stop", "method": "session.interrupt", "params": {
            "session_id": sid, "execution_generation": row["runtime_generation"]}})
        stopped = await sock.reply("stop")
        assert "result" in stopped, stopped
        assert stopped["result"]["ref"]["session_id"] == sid
        assert not release.is_set() and not any(frame.get("id") == "quota" for frame in sock.sent)
        release.set()
        quota = await sock.reply("quota")
        assert quota["result"]["snapshot"]["credits_balance"] == pytest.approx(289.08)
    finally:
        release.set()
        sock.inbound.put_nowait(None)
        await asyncio.wait_for(task, 30)

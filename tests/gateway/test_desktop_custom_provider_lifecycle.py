"""Desktop row identity survives creation, committed model picks and next-turn resolution.

Real authority RPCs, SQLite receipts, policy restoration, model switching and runtime routing;
only external model metadata is held offline. Custom rows send ``custom:<key>``; a bare native
provider remains native.

Known, SEPARATE resolver ambiguity (not covered here, not fixed by this contract): an explicit
``custom:<key>`` can still be intercepted by an EARLIER row whose display name normalizes to
that key (a legacy ``custom_providers:`` entry named ``DeepSeek`` shadows a later ``deepseek``
row). These cases use one row per name so the identity under test is the one that resolves; the
shadow edge is reported separately rather than pinned as behaviour.
"""

from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from types import SimpleNamespace

import pytest


MODEL = "desktop-contract-model"
START_MODEL = "desktop-start-model"


@pytest.fixture
def catalog(monkeypatch):
    from agent import models_dev

    metadata = {"custom": {"id": "custom", "models": {
        name: {"id": name, "name": name, "tool_call": True,
               "cost": {"input": 0, "output": 0},
               "limit": {"context": 100000, "output": 4000}}
        for name in (MODEL, START_MODEL)
    }}}
    monkeypatch.setattr(models_dev, "fetch_models_dev", lambda *args, **kwargs: metadata)

    class Catalog(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            self.server.received.append((self.path, self.headers.get("Authorization")))
            body = json.dumps({"data": [
                {"id": name, "context_length": 100000} for name in (MODEL, START_MODEL)]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Catalog)
    server.received = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _config(kind, base, provider="deepseek"):
    start = {"name": "Start endpoint", "base_url": base + "/start/v1",
             "key_env": "DESKTOP_START_KEY", "models": {START_MODEL: {}}}
    picked = {"base_url": base + "/correct/v1", "key_env": "DESKTOP_PICK_KEY",
              "models": {MODEL: {}}, "api_mode": "chat_completions"}
    config = {"model": {"provider": "custom:start", "default": START_MODEL},
              "providers": {"start": start}, "platform_toolsets": {"cli": []}}
    if kind == "keyed":
        config["providers"][provider] = {"name": "Collision endpoint", **picked}
    else:
        config["custom_providers"] = [{"name": provider, **picked}]
    return config


@asynccontextmanager
async def _owner(home, monkeypatch, config):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.run_turn_prepare import GatewayTurnPrepareMixin
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    from gateway.session_controls import AuthorityConnection
    from hermes_state_runtime import begin_runtime_epoch

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(run, "_hermes_home", home)
    monkeypatch.delenv("CUSTOM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    (home / ".env").write_text(
        "DESKTOP_START_KEY=dummy-start-key\nDESKTOP_PICK_KEY=dummy-picked-key\n"
        "DEEPSEEK_API_KEY=dummy-native-key\nMINIMAX_CN_API_KEY=dummy-minimax-native-key\n",
        encoding="utf-8")
    store = SessionStore(home / "sessions", GatewayConfig())
    db = store._db
    runner = SimpleNamespace(
        session_store=store, _session_db=db, adapters={}, _draining=False,
        _evict_cached_agent=lambda route: None, _cached_agent_for=lambda route: None,
        _peek_session_state=lambda route: None, _handle_message=None,
        _set_session_reasoning_override=lambda *args: None,
    )
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = SessionAuthority(
        runner, profile_id="owned", instance_id="desktop-contract", db=db,
        epoch=begin_runtime_epoch(db, instance_id="desktop-contract"))
    runner.session_authority = authority
    runner._resolve_session_agent_runtime = (
        GatewayTurnPrepareMixin._resolve_session_agent_runtime.__get__(runner))
    connection = AuthorityConnection(authority, object(), {"user_id": "human"})
    try:
        yield authority, connection
    finally:
        await connection.close()
        store.close_all_db_handles()


async def _rpc(connection, method, **params):
    response = await connection.dispatch({"jsonrpc": "2.0", "id": method,
                                          "method": method, "params": params})
    assert "result" in response, response
    return response["result"]


async def _create(connection, home, **selection):
    return await _rpc(connection, "session.create", request_id="desktop-create", source="gui",
                      cwd=str(home), toolsets=[], **selection)


def _restored_route(authority, sid):
    from gateway.session_policy import profile_read_scope, restore_policy, route_launch_key
    from gateway.session_worker_construct import route_runtime
    from hermes_state_local import local_receipt

    receipt = local_receipt(authority.db, sid)
    policy = restore_policy(receipt["policy"])
    # Do not supply an expected provider to the resolver: it must consume the stored selection.
    with profile_read_scope(policy, authority):
        frozen, model = policy.effective(authority)
        runtime, fallback = route_runtime(
            policy, frozen, model, route_launch_key(authority, policy, frozen))
    assert fallback is None
    assert "dummy-picked-key" not in json.dumps(receipt)
    return frozen["model"], model, runtime


@pytest.mark.asyncio
@pytest.mark.parametrize("kind, provider", [
    ("keyed", "deepseek"), ("legacy", "deepseek"), ("keyed", "minimax-cn")])
async def test_explicit_custom_create_survives_receipt_restoration(tmp_path, monkeypatch, kind, provider):
    config = _config(kind, "https://correct.example", provider)
    async with _owner(tmp_path, monkeypatch, config) as (authority, connection):
        created = await _create(connection, tmp_path, model=MODEL, provider=f"custom:{provider}")
        block, model, runtime = _restored_route(authority, created["session_id"])
        assert (block["provider"], model) == (f"custom:{provider}", MODEL)
        assert (runtime["provider"], runtime["base_url"], runtime["api_key"]) == (
            "custom", "https://correct.example/correct/v1", "dummy-picked-key")
        assert runtime["api_mode"] == "chat_completions"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind, provider", [
    ("keyed", "deepseek"), ("legacy", "deepseek"), ("keyed", "minimax-cn")])
async def test_live_custom_pick_commits_identity_for_the_next_runtime(
        tmp_path, monkeypatch, catalog, kind, provider):
    server, base = catalog
    async with _owner(tmp_path, monkeypatch, _config(kind, base, provider)) as (authority, connection):
        created = await _create(connection, tmp_path)
        sid = created["session_id"]
        _, _, initial = _restored_route(authority, sid)
        assert (initial["base_url"], initial["api_key"]) == (base + "/start/v1", "dummy-start-key")
        row = authority.db.get_session(sid)
        result = await _rpc(
            connection, "session.mutate", session_id=sid, request_id="desktop-pick", operation="model",
            payload={"model": MODEL, "provider": f"custom:{provider}"},
            expected_revision=row["runtime_revision"], expected_generation=row["runtime_generation"])
        assert result["model"] == MODEL
        # Eviction/restart cannot rescue a wrong identity with a warm switch result's API key.
        authority._local_config_secrets = {}
        block, model, runtime = _restored_route(authority, sid)
        assert (block["provider"], model) == (f"custom:{provider}", MODEL)
        assert (runtime["provider"], runtime["base_url"], runtime["api_key"]) == (
            "custom", base + "/correct/v1", "dummy-picked-key")
        assert runtime["api_mode"] == "chat_completions"
        live = authority.sessions[sid]
        next_model, next_runtime = authority.runner._resolve_session_agent_runtime(source=live.source)
        assert (next_model, next_runtime["base_url"], next_runtime["api_key"]) == (
            model, runtime["base_url"], runtime["api_key"])
        assert ("/correct/v1/models", "Bearer dummy-picked-key") in server.received
        # Creation may discover context metadata on the start endpoint. Neither endpoint
        # may receive the other's credential, including the real switch's validation probe.
        assert all(bearer == ("Bearer dummy-start-key" if path.startswith("/start/")
                              else "Bearer dummy-picked-key") for path, bearer in server.received)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind, provider, native_url, mode, key", [
    ("keyed", "deepseek", "https://api.deepseek.com/v1", "chat_completions", "dummy-native-key"),
    ("legacy", "deepseek", "https://api.deepseek.com/v1", "chat_completions", "dummy-native-key"),
    ("keyed", "minimax-cn", "https://api.minimaxi.com/anthropic", "anthropic_messages",
     "dummy-minimax-native-key")])
async def test_bare_native_create_is_not_reinterpreted_as_the_custom_row(
        tmp_path, monkeypatch, kind, provider, native_url, mode, key):
    config = _config(kind, "https://correct.example", provider)
    async with _owner(tmp_path, monkeypatch, config) as (authority, connection):
        created = await _create(connection, tmp_path, model=MODEL, provider=provider)
        block, model, runtime = _restored_route(authority, created["session_id"])
        assert (block["provider"], model, runtime["provider"]) == (provider, MODEL, provider)
        assert (runtime["base_url"].rstrip("/"), runtime["api_mode"], runtime["api_key"]) == (
            native_url, mode, key)


@pytest.mark.asyncio
async def test_endpointless_builtin_tuning_does_not_displace_a_held_legacy_route(tmp_path, monkeypatch):
    config = _config("legacy", "https://correct.example")
    config["providers"]["deepseek"] = {"stale_timeout_seconds": 600}
    config["model"] = {"provider": "custom:deepseek", "default": MODEL,
                       "base_url": "https://correct.example/correct/v1",
                       "key_env": "DESKTOP_PICK_KEY", "context_length": 32768}
    async with _owner(tmp_path, monkeypatch, config) as (authority, connection):
        created = await _create(connection, tmp_path, model=MODEL, provider="deepseek")
        block, model, runtime = _restored_route(authority, created["session_id"])
        assert (block["provider"], block["context_length"], model) == ("custom:deepseek", 32768, MODEL)
        assert (runtime["provider"], runtime["base_url"], runtime["api_key"]) == (
            "custom", "https://correct.example/correct/v1", "dummy-picked-key")

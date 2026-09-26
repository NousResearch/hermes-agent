"""Catalog-driven reasoning works on first use and stays isolated across clients/profiles."""
import asyncio
import contextlib
import importlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
import yaml

from agent.provider_reasoning import prepare_client_reasoning
from agent.transports.codex import ResponsesApiTransport
from hermes_constants import set_hermes_home_override, reset_hermes_home_override
from providers import get_provider_profile


@pytest.fixture
def catalog_module():
    profile = get_provider_profile("xai")
    module = importlib.import_module(type(profile).__module__ + ".reasoning")
    module._catalogs.clear()
    yield module
    module._catalogs.clear()


@pytest.fixture
def endpoint():
    state = SimpleNamespace(calls=[], responses=[], payload={"data": []}, status=200,
                            entered=threading.Event(), release=threading.Event())
    state.release.set()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            state.calls.append((self.path, self.headers.get("Authorization")))
            state.entered.set()
            state.release.wait(10)
            payload = state.payload(self.headers) if callable(state.payload) else state.payload
            data = json.dumps(payload).encode()
            self.send_response(state.status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path != "/v1/responses":
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")
                return
            state.responses.append(body)
            response = {"id": "resp_test", "object": "response", "created_at": 1, "status": "completed",
                "model": body["model"], "output": [{"id": "msg_test", "type": "message", "role": "assistant",
                "status": "completed", "content": [{"type": "output_text", "text": "OK", "annotations": []}]}],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}
            data = ("event: response.completed\ndata: " + json.dumps({"type": "response.completed", "response": response}) + "\n\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_port}/v1"
    yield state
    state.release.set()
    server.shutdown()
    server.server_close()
    thread.join(5)


def build(client, model, effort="xhigh", warnings=None):
    return ResponsesApiTransport().build_kwargs(
        model=model, messages=[{"role": "user", "content": "Hello"}], tools=[],
        is_xai_responses=True, base_url=str(client.base_url),
        reasoning_profile=getattr(client, "_hermes_reasoning_profile", None),
        reasoning_warning_state=warnings,
        reasoning_config=effort if isinstance(effort, dict) else {"effort": effort},
    )


@pytest.mark.parametrize("model", ["grok-4.7", "x-ai/grok-4.7", "grok-4.7-latest", "grok-unreleased"])
@pytest.mark.parametrize("levels,requested,expected", [
    (["low", "medium", "high", "xhigh"], "low", "low"),
    (["low", "medium", "high", "xhigh"], "medium", "medium"),
    (["low", "medium", "high", "xhigh"], "high", "high"),
    (["low", "high", "xhigh"], "xhigh", "xhigh"),
    (["low", "high", "xhigh"], "ultra", "xhigh"),
    (["low", "high"], "xhigh", "high"),
    (["none", "low", "high"], "none", "none"),
    (["none", "low"], {"enabled": False}, "none"),
    (["low"], {"enabled": False}, None),
    ([], "xhigh", None), (None, "xhigh", None), ("xhigh", "xhigh", None),
    (["invented"], "xhigh", None), (["ultra"], "ultra", None),
])
def test_catalog_controls_first_primary_and_sync_async_auxiliary_requests(
        catalog_module, endpoint, model, levels, requested, expected, monkeypatch):
    from run_agent import AIAgent
    from agent.auxiliary_client import resolve_provider_client

    canonical = "grok-4.7" if "4.7" in model else model
    endpoint.payload = {"data": [{"id": canonical, "aliases": ["grok-private-alias", "grok-4.7-latest"],
                                 "capabilities": {"reasoning_effort": levels}}]}
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **kwargs: [])
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda: {})
    config = requested if isinstance(requested, dict) else {"effort": requested}
    agent = AIAgent(model=model, provider="xai", api_mode="codex_responses",
                    api_key="test-key", base_url=endpoint.url, quiet_mode=True,
                    skip_memory=True, skip_context_files=True, enabled_toolsets=[],
                    reasoning_config=config)
    initial_calls = len(endpoint.calls)
    assert initial_calls > 0
    main_request = agent._build_api_kwargs([{"role": "user", "content": "Hello"}])
    assert main_request.get("reasoning") == ({"effort": expected} if expected is not None else None)
    route = dict(provider="xai", model=model, explicit_api_key="test-key",
                 explicit_base_url=endpoint.url, api_mode="codex_responses")
    wrapper, _ = resolve_provider_client(**route)
    async_wrapper, _ = resolve_provider_client(**route, async_mode=True)
    try:
        assert len(endpoint.calls) == initial_calls  # initialization fetched, no model picker required
        for selected_model in (model, "grok-private-alias"):
            main = build(agent.client, selected_model, requested)
            assert main.get("reasoning") == ({"effort": expected} if expected is not None else None)
            assert main["model"] == selected_model
        kwargs = {"model": model, "messages": [{"role": "user", "content": "Hello"}],
                  "extra_body": {"reasoning": config}}
        wrapper.chat.completions.create(**kwargs)
        asyncio.run(async_wrapper.chat.completions.create(**kwargs))
        assert len(endpoint.calls) == initial_calls
        for sent in endpoint.responses:
            assert sent.get("reasoning") == main.get("reasoning")
            assert "service_tier" not in sent
            assert ("reasoning.encrypted_content" in sent.get("include", [])) == (
                config.get("enabled") is not False or expected == "none")
        assert len(endpoint.responses) == 2
    finally:
        wrapper._real_client.close()
        async_wrapper._real_client.close()
        agent.client.close()


@contextlib.contextmanager
def home_scope(home):
    token = set_hermes_home_override(str(home))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def test_catalog_cache_credentials_profiles_refresh_and_config_overrides(catalog_module, endpoint, tmp_path, monkeypatch, caplog):
    homes = [tmp_path / "A", tmp_path / "B"]
    for home in homes:
        home.mkdir()
        (home / "config.yaml").write_text("{}\n")
    (homes[1] / "config.yaml").write_text(yaml.safe_dump({"model_overrides": {"xai": {
        "grok-future": {"supported_reasoning_efforts": ["low", "medium"]}}}}))
    endpoint.payload = lambda headers: {"data": [{"id": "grok-future", "capabilities": {
        "reasoning_effort": ["low", "xhigh"] if headers["Authorization"] == "Bearer account-A" else ["low", "high"]}}]}
    now = [100.0]
    monkeypatch.setattr(catalog_module.time, "monotonic", lambda: now[0])
    # Concurrent cold starts share a GET. Real HTTP, config and scope readers are used.
    with home_scope(homes[0]):
        from contextvars import copy_context
        endpoint.release.clear()
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(copy_context().run, prepare_client_reasoning,
                SimpleNamespace(base_url=endpoint.url, api_key="account-A"), provider="xai") for _ in range(6)]
            assert endpoint.entered.wait(5)
            endpoint.release.set()
            profiles = [f.result(10) for f in futures]
        assert len(endpoint.calls) == 1
        assert all(p.supported_reasoning_efforts("grok-future") == ("low", "xhigh") for p in profiles)
    for home, key, expected in [(homes[1], "account-B", "medium"), (homes[0], "account-A", "xhigh"),
                                 (homes[0], "account-B", "high")]:
        with home_scope(home):
            client = SimpleNamespace(base_url=endpoint.url, api_key=key)
            prepare_client_reasoning(client, provider="xai")
            assert build(client, "grok-future")["reasoning"]["effort"] == expected
    assert len(endpoint.calls) == 3
    with home_scope(homes[0]):
        client = SimpleNamespace(base_url=endpoint.url, api_key="account-A")
        prepare_client_reasoning(client, provider="xai")
        (homes[0] / "config.yaml").write_text(yaml.safe_dump({"model_overrides": {"xai": {
            "GROK-FUTURE": {"supported_reasoning_efforts": []}}}}))
        assert "reasoning" not in build(client, "grok-future")
        (homes[0] / "config.yaml").write_text(yaml.safe_dump({"model_overrides": {"xai": {
            "grok-future": {"supported_reasoning_efforts": "bad"}}}}))
        assert build(client, "grok-future")["reasoning"]["effort"] == "xhigh"
        assert "invalid supported_reasoning_efforts" in caplog.text
        (homes[0] / "config.yaml").write_text("{}\n")
        endpoint.status = 503
        now[0] += 1201
        prepare_client_reasoning(client, provider="xai")
        entry = catalog_module._catalogs[client._hermes_reasoning_profile._catalog_key]
        assert entry.ready.wait(5)
        assert build(client, "grok-future")["reasoning"]["effort"] == "xhigh"
        count = len(endpoint.calls)
        prepare_client_reasoning(client, provider="xai")
        assert len(endpoint.calls) == count
        endpoint.status = 200
        endpoint.payload = {"data": [{"id": "grok-future", "capabilities": {"reasoning_effort": ["low"]}}]}
        now[0] += 301
        prepare_client_reasoning(client, provider="xai")
        assert entry.ready.wait(5)
        assert build(client, "grok-future")["reasoning"]["effort"] == "low"
        # Unknown-model warning is once per caller/session, not once globally.
        warnings = set()
        for _ in range(2):
            assert "reasoning" not in build(client, "unknown-grok", warnings=warnings)
        assert caplog.text.count("Cannot confirm the requested reasoning effort") == 1
        assert "reasoning" not in build(client, "grok-4.70", warnings=warnings)

        # Known catalog models without effort metadata do not inherit an unknown-model default.
        (homes[0] / "config.yaml").write_text(yaml.safe_dump({"model_overrides": {"xai": {
            "_default": {"supported_reasoning_efforts": ["low"]}}}}))
        endpoint.payload = {"data": [{"id": "grok-4.6", "aliases": ["known-alias"]}]}
        now[0] += 1201
        prepare_client_reasoning(client, provider="xai")
        assert entry.ready.wait(5)
        assert build(client, "grok-4.6")["reasoning"]["effort"] == "xhigh"
        assert "reasoning" not in build(client, "known-alias")
        assert build(client, "unknown-grok")["reasoning"]["effort"] == "low"
        # A callable credential must use its effective key and a separate cache slot.
        (homes[0] / "config.yaml").write_text("{}\n")
        endpoint.payload = {"data": [{"id": "grok-future", "capabilities": {"reasoning_effort": ["high"]}}]}
        client._api_key_provider = lambda: "account-C"
        client.api_key = ""
        prepare_client_reasoning(client, provider="xai")
        assert endpoint.calls[-1][1] == "Bearer account-C"
        assert build(client, "grok-future")["reasoning"]["effort"] == "high"
        # Cold failures are negative-cached and preserve historical compatibility.
        client._api_key_provider = lambda: "unavailable-account"
        endpoint.status = 503
        prepare_client_reasoning(client, provider="xai")
        calls = len(endpoint.calls)
        assert build(client, "grok-4.6")["reasoning"]["effort"] == "xhigh"
        prepare_client_reasoning(client, provider="xai")
        assert len(endpoint.calls) == calls
        # Changing endpoints creates a separate slot even with the same credentials.
        client.base_url = endpoint.url.replace("127.0.0.1", "localhost")
        prepare_client_reasoning(client, provider="xai")
        assert len(endpoint.calls) == calls + 1

        # A stalled cold load is bounded and subsequent turns do not wait again.
        # Hold the fetch worker explicitly to avoid relying on OS socket timing.
        entered, release = threading.Event(), threading.Event()
        def stalled_fetch(*args):
            entered.set()
            release.wait(5)
            raise TimeoutError()
        monkeypatch.setattr(catalog_module, "_fetch", stalled_fetch)
        monkeypatch.setattr(catalog_module, "_TIMEOUT", 0.05)
        client._api_key_provider = lambda: "stalled-account"
        try:
            prepare_client_reasoning(client, provider="xai")
            assert entered.is_set()
            stalled = catalog_module._catalogs[client._hermes_reasoning_profile._catalog_key]
            assert stalled.loading and not stalled.ready.is_set()
            assert stalled.refresh_after == now[0] + 300
            monkeypatch.setattr(stalled.ready, "wait", lambda *args: pytest.fail("repeated cold wait"))
            prepare_client_reasoning(client, provider="xai")
        finally:
            release.set()

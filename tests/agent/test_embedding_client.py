"""Focused transport coverage for profile-scoped OpenAI-compatible embeddings."""
from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from openai import AsyncOpenAI, OpenAI

from agent import embedding_client


@pytest.fixture
def embedding_server():
    state = {"requests": [], "statuses": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            state["requests"].append((self.path, self.headers.get("Authorization"), body))
            status = state["statuses"].pop(0) if state["statuses"] else 200
            if status != 200:
                data = json.dumps({"error": {"message": "credential test-key was rejected"}}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            count = len(body["input"])
            response = {
                "object": "list", "model": "actual-test-embedding-model",
                "data": [{"object": "embedding", "index": i, "embedding": [float(i), 0.5]} for i in range(count)],
                "usage": {"prompt_tokens": 2 * count, "total_tokens": 2 * count},
            }
            data = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _patch_resolver(monkeypatch, base_url, *, async_mode=False):
    client_type = AsyncOpenAI if async_mode else OpenAI
    client = client_type(api_key="test-key", base_url=base_url, max_retries=0)
    seen = []

    def resolve(provider, model, *, async_mode=False):
        seen.append((provider, model, async_mode))
        return client, model

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {
        "auxiliary": {"embedding": {"provider": "auto", "model": "", "input_type_handling": "shared"}},
    })
    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", resolve)
    return client, seen


def test_call_embeddings_uses_loopback_transport_and_actual_model(monkeypatch, embedding_server):
    base_url, state = embedding_server
    client, seen = _patch_resolver(monkeypatch, base_url)
    try:
        provider, model, response = embedding_client.call_embeddings(
            ["synthetic one", "synthetic two"], input_type="query", provider="openai-api",
            model="requested-model", dimensions=2, timeout=8,
        )
    finally:
        client.close()

    assert (provider, model) == ("openai-api", "actual-test-embedding-model")
    assert response._hermes_input_type_handling == "shared"
    assert [row.embedding for row in response.data] == [[0.0, 0.5], [1.0, 0.5]]
    assert seen == [("openai-api", "requested-model", False)]
    assert state["requests"][0][0] == "/v1/embeddings"
    assert state["requests"][0][1] == "Bearer test-key"
    assert state["requests"][0][2]["input"] == ["synthetic one", "synthetic two"]
    assert state["requests"][0][2]["dimensions"] == 2
    # OpenAI's API uses a shared query/document embedding endpoint; semantics are preserved as
    # caller metadata and are never rewritten into a chat completion request.
    assert state["requests"][0][2].get("input_type") is None


def test_async_call_embeddings_supports_cancellation_and_loopback(monkeypatch, embedding_server):
    base_url, state = embedding_server
    client, seen = _patch_resolver(monkeypatch, base_url, async_mode=True)

    async def run():
        try:
            return await embedding_client.async_call_embeddings(
                ["synthetic query"], input_type="query", provider="openrouter", model="embed-model", timeout=8,
            )
        finally:
            await client.close()

    provider, model, response = asyncio.run(run())
    assert (provider, model) == ("openrouter", "actual-test-embedding-model")
    assert response.data[0].index == 0
    assert seen == [("openrouter", "embed-model", True)]
    assert state["requests"][0][2]["input"] == ["synthetic query"]


def test_auto_selection_uses_embedding_default_not_chat_route(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {
        "model": {"provider": "openai-codex", "model": "gpt-chat-model"},
        "auxiliary": {"embedding": {"provider": "auto", "model": "", "timeout": 17}},
    })
    assert embedding_client._selection(None, None, None, None) == (
        "openai-api", "text-embedding-3-small", None, 17.0, None,
    )


def test_oauth_route_fails_before_credentials_are_resolved(monkeypatch):
    called = []
    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", lambda *a, **k: called.append(True))
    with pytest.raises(embedding_client.EmbeddingError) as caught:
        embedding_client.call_embeddings(["synthetic"], provider="openai-codex", model="gpt-5")
    assert caught.value.code == "embedding_unsupported"
    assert "test-key" not in str(caught.value)
    assert called == []


def test_auth_failure_refreshes_same_provider_and_sanitizes_error(monkeypatch, embedding_server):
    base_url, state = embedding_server
    state["statuses"].append(401)
    client, seen = _patch_resolver(monkeypatch, base_url)
    refreshes = []
    monkeypatch.setattr(embedding_client, "_recover_same_provider", lambda provider, exc: refreshes.append(provider) or True)
    try:
        provider, model, response = embedding_client.call_embeddings(
            ["synthetic"], provider="openai-api", model="embed-model",
        )
    finally:
        client.close()
    assert provider == "openai-api"
    assert model == "actual-test-embedding-model"
    assert response.data[0].embedding == [0.0, 0.5]
    assert refreshes == ["openai-api"]
    assert [row[0] for row in seen] == ["openai-api", "openai-api"]


def test_failure_message_does_not_include_provider_body_or_exception_chain(monkeypatch, embedding_server):
    base_url, state = embedding_server
    state["statuses"].append(401)
    client, _seen = _patch_resolver(monkeypatch, base_url)
    monkeypatch.setattr(embedding_client, "_recover_same_provider", lambda *_args: False)
    try:
        with pytest.raises(embedding_client.EmbeddingError) as caught:
            embedding_client.call_embeddings(["synthetic"], provider="openai-api", model="embed-model")
    finally:
        client.close()
    assert caught.value.code == "embedding_failed"
    assert "test-key" not in str(caught.value)
    assert "credential test-key" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_configured_dimensions_must_match_provider_response(monkeypatch, embedding_server):
    base_url, _state = embedding_server
    client, _seen = _patch_resolver(monkeypatch, base_url)
    try:
        with pytest.raises(embedding_client.EmbeddingError) as caught:
            embedding_client.call_embeddings(
                ["synthetic"], provider="openai-api", model="text-embedding-3-small", dimensions=3,
            )
    finally:
        client.close()
    assert caught.value.code == "embedding_invalid_response"
    assert caught.value.__context__ is None


def test_rejects_clients_without_embeddings_before_chat_fallback(monkeypatch):
    class ChatOnly:
        chat = object()

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {
        "auxiliary": {"embedding": {"provider": "auto", "model": "", "input_type_handling": "shared"}},
    })
    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", lambda *a, **k: (ChatOnly(), "chat-model"))
    with pytest.raises(embedding_client.EmbeddingError) as caught:
        embedding_client.call_embeddings(["synthetic"], provider="openai-api", model="chat-model")
    assert caught.value.code == "embedding_unsupported"


def test_auth_recovery_uses_existing_pool_then_provider_refresh(monkeypatch):
    from agent import auxiliary_client

    pool_calls = []
    refresh_calls = []
    monkeypatch.setattr(embedding_client, "_is_auth_failure", lambda _provider, _exc: True)
    monkeypatch.setattr(auxiliary_client, "_recover_provider_pool",
                        lambda provider, exc: pool_calls.append((provider, exc)) or True)
    monkeypatch.setattr(auxiliary_client, "_refresh_provider_credentials",
                        lambda provider: refresh_calls.append(provider) or True)

    failure = RuntimeError("synthetic auth rejection")
    assert embedding_client._recover_same_provider("openai", failure)
    assert pool_calls == [("openai", failure)]
    assert refresh_calls == []

    monkeypatch.setattr(auxiliary_client, "_recover_provider_pool", lambda *_args: False)
    assert embedding_client._recover_same_provider("openai", failure)
    assert refresh_calls == ["openai"]


def test_profile_override_binds_existing_profile_scope(monkeypatch):
    from agent import runtime_scope
    from hermes_cli import profiles

    entered = []

    @contextlib.contextmanager
    def scoped(profile):
        entered.append(profile)
        yield

    monkeypatch.setattr(profiles, "validate_profile_name", lambda _profile: None)
    monkeypatch.setattr(profiles, "profile_exists", lambda _profile: True)
    monkeypatch.setattr(profiles, "get_profile_dir", lambda _profile: "profile-home")
    monkeypatch.setattr(runtime_scope, "bind_profile_scope", scoped)
    with embedding_client._profile_scope("work"):
        pass
    assert entered == ["profile-home"]


def _write_profile(home, base_url, *, model, keys, timeout=8):
    import hermes_yaml as yaml

    home.mkdir(parents=True, exist_ok=True)
    provider_name = "embed-loop"
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "openai-codex", "default": "chat-only-default"},
        "providers": {provider_name: {
            "name": provider_name,
            "provider_key": provider_name,
            "base_url": base_url,
            "api_mode": "chat_completions",
        }},
        "auxiliary": {"embedding": {
            "provider": f"custom:{provider_name}",
            "model": model,
            "dimensions": 2,
            "timeout": timeout,
            "input_type_handling": "shared",
        }},
    }))
    (home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {},
        "credential_pool": {provider_name: [
            {
                "id": f"key-{index}", "label": f"test key {index}", "auth_type": "api_key",
                "priority": index, "source": "manual", "access_token": key,
            }
            for index, key in enumerate(keys)
        ]},
    }))


def test_real_resolver_rotates_profile_credential_pool_on_loopback(monkeypatch, tmp_path, embedding_server):
    base_url, state = embedding_server
    state["statuses"].append(401)
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_constants
    hermes_constants._default_hermes_root_memo = None
    _write_profile(home, base_url, model="embed-loopback-model", keys=["synthetic-key-1", "synthetic-key-2"])

    provider, model, response = embedding_client.call_embeddings(["fixture input"])

    assert provider == "custom:embed-loop"
    assert model == "actual-test-embedding-model"
    assert response.data[0].embedding == [0.0, 0.5]
    assert [row[1] for row in state["requests"]] == [
        "Bearer synthetic-key-1", "Bearer synthetic-key-2",
    ]
    assert [row[2]["model"] for row in state["requests"]] == [
        "embed-loopback-model", "embed-loopback-model",
    ]


def test_actual_profile_scope_selects_isolated_config_and_credentials(monkeypatch, tmp_path, embedding_server):
    base_url, state = embedding_server
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_constants
    hermes_constants._default_hermes_root_memo = None
    _write_profile(home, base_url, model="default-profile-model", keys=["default-synthetic-key"])
    work = home / "profiles" / "work"
    _write_profile(work, base_url, model="work-profile-model", keys=["work-synthetic-key"])
    from hermes_cli.profiles import get_profile_dir, profile_exists
    assert get_profile_dir("work") == work
    assert profile_exists("work")

    embedding_client.call_embeddings(["default input"])
    embedding_client.call_embeddings(["work input"], profile="work")

    async def invoke(profile, content):
        await embedding_client.async_call_embeddings([content], profile=profile)

    async def concurrent_calls():
        await asyncio.gather(invoke(None, "default async"), invoke("work", "work async"))

    asyncio.run(concurrent_calls())

    assert sorted((row[1], row[2]["model"]) for row in state["requests"]) == sorted([
        ("Bearer default-synthetic-key", "default-profile-model"),
        ("Bearer work-synthetic-key", "work-profile-model"),
        ("Bearer default-synthetic-key", "default-profile-model"),
        ("Bearer work-synthetic-key", "work-profile-model"),
    ])


def test_custom_provider_requires_configured_shared_handling_before_resolution(monkeypatch):
    resolved = []
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {
        "providers": {"embed-loop": {"base_url": "http://127.0.0.1:1/v1", "api_mode": "chat_completions"}},
        "auxiliary": {"embedding": {"provider": "custom:embed-loop", "model": "unknown-task-model"}},
    })
    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", lambda *a, **k: resolved.append(True))
    assert embedding_client._selection(None, None, None, None)[0:2] == ("custom:embed-loop", "unknown-task-model")
    with pytest.raises(embedding_client.EmbeddingError) as caught:
        embedding_client.call_embeddings(["synthetic"])
    assert caught.value.code == "embedding_unsupported"
    assert resolved == []


def test_custom_messages_route_is_rejected_before_credentials(monkeypatch):
    resolved = []
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {
        "providers": {"messages-only": {"base_url": "https://api.anthropic.com/v1"}},
    })
    monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", lambda *a, **k: resolved.append(True))
    assert not embedding_client._is_supported_route("custom:messages-only")
    with pytest.raises(embedding_client.EmbeddingError) as caught:
        embedding_client.call_embeddings(["synthetic"], provider="custom:messages-only", model="claude")
    assert caught.value.code == "embedding_unsupported"
    assert resolved == []


def test_embedding_error_uses_fixed_message_for_single_code_argument():
    error = embedding_client.EmbeddingError("embedding_invalid_response")
    assert error.code == "embedding_invalid_response"
    assert str(error) == "The embedding provider returned an invalid response."


@pytest.mark.parametrize("dimensions", [True, 0, 65537, "2"])
def test_malformed_configured_dimensions_fail_closed(monkeypatch, dimensions):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {
        "auxiliary": {"embedding": {"provider": "openai-api", "model": "text-embedding-3-small", "dimensions": dimensions}},
    })
    with pytest.raises(embedding_client.EmbeddingError) as caught:
        embedding_client._selection(None, None, None, None)
    assert caught.value.code == "embedding_unsupported"


def test_config_failure_is_sanitized_without_exception_context(monkeypatch):
    def fail():
        raise RuntimeError("synthetic-key from bad config")

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", fail)
    with pytest.raises(embedding_client.EmbeddingError) as caught:
        embedding_client.call_embeddings(["synthetic"])
    assert caught.value.code == "embedding_unconfigured"
    assert "synthetic-key" not in str(caught.value)
    assert caught.value.__context__ is None

"""Actual background tasks exercise the same configured route as foreground chat."""

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading
import time

import pytest
import yaml


@pytest.fixture
def actual_endpoint(monkeypatch):
    from agent.auxiliary_client import (
        shutdown_cached_clients,
        _reset_aux_unhealthy_cache,
    )

    shutdown_cached_clients()
    _reset_aux_unhealthy_cache()
    requests = []
    resolve_address = socket.getaddrinfo

    def local_actual_address(host, *args, **kwargs):
        if host in ("api.actual.inc", b"api.actual.inc"):
            host = "127.0.0.1"
        return resolve_address(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", local_actual_address)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost,api.actual.inc")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, payload))
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            if payload["model"] == "unavailable-model":
                body = json.dumps({
                    "error": {
                        "message": "The model is not supported with this account",
                        "type": "invalid_request_error",
                        "code": "unsupported_model",
                    }
                }).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            content = (
                '{"title":"Actual background routing"}'
                if "response_format" in payload
                else "The task is complete."
            )
            response = {
                "id": "chatcmpl-background",
                "created": 1,
                "model": payload["model"],
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
            if payload.get("stream"):
                response["object"] = "chat.completion.chunk"
                response["choices"][0]["delta"] = response["choices"][0].pop("message")
                body = f"data: {json.dumps(response)}\n\ndata: [DONE]\n\n".encode()
                content_type = "text/event-stream"
            else:
                body = json.dumps(response).encode()
                content_type = "application/json"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        shutdown_cached_clients()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize(
    "aux_provider",
    [
        "auto",
        "actual",
        "actual-computer",
        "actualcomputer",
        "aci",
        "custom",
        "custom:actual-relay",
    ],
)
@pytest.mark.parametrize(
    "hosted,use_api_key", [(False, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("stale_mode", [None, "codex_responses"])
def test_actual_background_tasks_reach_chat_completions(
    tmp_path,
    monkeypatch,
    actual_endpoint,
    aux_provider,
    hosted,
    use_api_key,
    stale_mode,
):
    from agent.auxiliary_client import async_call_llm
    from agent.context_compressor import ContextCompressor
    from agent.title_generator import generate_title
    from hermes_cli.runtime_provider import resolve_runtime_provider

    base_url, requests = actual_endpoint
    if hosted:
        base_url = base_url.replace("127.0.0.1", "api.actual.inc")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    if use_api_key:
        monkeypatch.setenv("ACTUAL_API_KEY", "actual-test-key")
        monkeypatch.setenv("ACTUAL_BASE_URL", "http://127.0.0.1:1")
    aux_model = "override-model" if aux_provider.startswith("custom") else "test-model"
    config = {
        "model": {
            "provider": "aci" if aux_provider == "aci" else "actual",
            "default": "test-model",
            "base_url": base_url,
        },
        "auxiliary": {
            task: {
                "provider": aux_provider,
                "model": aux_model,
                "api_mode": stale_mode,
                **(
                    {"base_url": base_url if stale_mode else base_url + "/v1"}
                    if aux_provider == "custom"
                    else {}
                ),
            }
            for task in (
                "compression",
                "title_generation",
                "session_search",
            )
        },
    }
    config["providers"] = {
        "actual-relay": {
            "base_url": base_url if stale_mode else base_url + "/v1",
            "key_env": "ACTUAL_API_KEY",
            "transport": stale_mode or "chat_completions",
        }
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    runtime = resolve_runtime_provider(requested="actual")
    runtime["model"] = "test-model"
    assert runtime["api_mode"] == "chat_completions"
    if stale_mode:
        runtime["api_mode"] = stale_mode
    assert (
        generate_title(
            "Stale conversation", main_runtime=runtime, runtime_validator=lambda: False
        )
        is None
    )
    assert requests == []
    assert (
        generate_title("Check the background routing", timeout=5, main_runtime=runtime)
        == "Actual background routing"
    )
    compressor = ContextCompressor(
        model=runtime["model"],
        provider=runtime["provider"],
        api_key=runtime["api_key"],
        base_url=runtime["base_url"],
        api_mode=runtime["api_mode"],
        config_context_length=32768,
        quiet_mode=True,
    )
    assert (
        compressor._call_summary_llm("Summarize this conversation.", time.monotonic())
        == "The task is complete."
    )
    response = asyncio.run(
        async_call_llm(
            task="session_search",
            messages=[{"role": "user", "content": "Summarize the results."}],
            main_runtime=runtime,
            timeout=5,
        )
    )
    assert response.choices[0].message.content == "The task is complete."
    assert len(requests) == 3
    assert [(path, payload["model"]) for path, payload in requests] == [
        ("/v1/chat/completions", aux_model)
    ] * 3


@pytest.mark.parametrize(
    "provider,hosted",
    [
        ("actual", False),
        ("aci", False),
        ("actual-computer", False),
        ("actualcomputer", False),
        ("custom", True),
        ("custom:actual-relay", True),
    ],
)
@pytest.mark.parametrize(
    "entrypoint",
    [
        "init",
        "auto",
        "switch",
        "fallback",
        "restore",
        "rotation",
        "init_fallback",
        "init_auto",
    ],
)
def test_actual_runtime_transitions_reach_chat_completions(
    tmp_path, monkeypatch, actual_endpoint, provider, hosted, entrypoint
):
    from agent.error_classifier import FailoverReason
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from run_agent import AIAgent

    base_url, requests = actual_endpoint
    if hosted:
        base_url = base_url.replace("127.0.0.1", "api.actual.inc")
    base_url += "/v1"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("ACTUAL_API_KEY", "actual-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "actual-test-key")
    config = {
        "model": {
            "provider": provider,
            "default": "gpt-5.4",
            "base_url": base_url,
            "api_mode": "codex_responses",
        },
        "providers": {
            "actual-relay": {
                "base_url": base_url,
                "key_env": "ACTUAL_API_KEY",
                "transport": "codex_responses",
            }
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    runtime = resolve_runtime_provider(requested=provider)
    assert runtime["api_mode"] == "chat_completions"
    initial_base = base_url.removesuffix("/v1")
    initial_provider = provider
    initial_key = "actual-test-key"
    if entrypoint in {"init_fallback", "init_auto"}:
        initial_base = initial_key = None
        initial_provider = (
            "missing-provider" if entrypoint == "init_fallback" else "auto"
        )
    elif entrypoint == "rotation":
        initial_base = base_url.replace("api.actual.inc", "127.0.0.1")
    agent = AIAgent(
        provider=initial_provider,
        base_url=initial_base,
        api_key=initial_key,
        api_mode=None if entrypoint == "auto" else "codex_responses",
        model="primary-model" if entrypoint == "fallback" else "gpt-5.4",
        enabled_toolsets=[],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=False,
        fallback_model={
            "provider": provider,
            "model": "gpt-5.4",
            "base_url": base_url.removesuffix("/v1"),
            "api_key": "actual-test-key",
            "api_mode": "codex_responses",
        },
    )
    try:
        if entrypoint == "switch":
            agent.switch_model(
                "gpt-5.4",
                provider,
                api_key="actual-test-key",
                base_url=base_url.removesuffix("/v1"),
                api_mode="codex_responses",
            )
        elif entrypoint == "fallback":
            assert agent._try_activate_fallback(FailoverReason.rate_limit)
        elif entrypoint == "restore":
            agent._primary_runtime["api_mode"] = "codex_responses"
            agent._fallback_activated = True
            assert agent._restore_primary_runtime()
        elif entrypoint == "rotation":
            from agent.credential_pool import PooledCredential

            agent._swap_credential(
                PooledCredential.from_dict(
                    provider,
                    {
                        "id": "rotated",
                        "access_token": "actual-rotated-key",
                        "base_url": base_url.removesuffix("/v1"),
                    },
                )
            )
        assert agent.api_mode == "chat_completions"
        response = agent._interruptible_api_call(
            agent._build_api_kwargs([{"role": "user", "content": "Reply briefly."}], [])
        )
        assert response.choices[0].message.content == "The task is complete."
        inference_requests = [
            (path, body) for path, body in requests if path != "/api/show"
        ]
        assert len(inference_requests) == 1
        assert inference_requests[0][0] == "/v1/chat/completions"
        assert inference_requests[0][1]["model"] == "gpt-5.4"
    finally:
        agent.client.close()


@pytest.mark.parametrize(
    "provider,hosted",
    [
        ("actual", False),
        ("aci", False),
        ("actual-computer", False),
        ("actualcomputer", False),
        ("custom", True),
        ("custom:actual-relay", True),
    ],
)
@pytest.mark.parametrize("send_site", ["main", "auxiliary", "async_auxiliary"])
def test_actual_rejects_forced_responses_before_http(
    tmp_path, monkeypatch, actual_endpoint, provider, hosted, send_site
):
    from agent.auxiliary_client import (
        AsyncCodexAuxiliaryClient,
        CodexAuxiliaryClient,
        resolve_provider_client,
    )
    from run_agent import AIAgent

    base_url, requests = actual_endpoint
    if hosted:
        base_url = base_url.replace("127.0.0.1", "api.actual.inc")
    base_url += "/v1"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = {
        "model": {"provider": provider, "base_url": base_url, "default": "test-model"},
        "providers": {
            "actual-relay": {"base_url": base_url, "transport": "codex_responses"}
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    if send_site == "main":
        agent = AIAgent(
            provider=provider,
            base_url=base_url,
            api_key="actual-test-key",
            model="test-model",
            enabled_toolsets=[],
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            save_trajectories=False,
        )
        client = agent.client
        agent.api_mode = "codex_responses"

        def force_responses():
            return agent._run_codex_stream(
                {"model": "test-model", "input": "Reply briefly."}, client=client
            )

    else:
        client, model = resolve_provider_client(
            provider,
            model="test-model",
            explicit_base_url=base_url,
            explicit_api_key="actual-test-key",
            api_mode="codex_responses",
        )
        wrapper = CodexAuxiliaryClient(client, model)
        if send_site == "async_auxiliary":
            wrapper = AsyncCodexAuxiliaryClient(wrapper)

        def force_responses():
            result = wrapper.chat.completions.create(
                model=model, messages=[{"role": "user", "content": "Reply briefly."}]
            )
            return asyncio.run(result) if send_site == "async_auxiliary" else result

    try:
        initial_requests = list(requests)
        with pytest.raises(ValueError, match="Actual.*Chat Completions"):
            force_responses()
        assert requests == initial_requests
    finally:
        client.close()


@pytest.mark.parametrize("provider", ["actual", "aci", "custom:actual-relay"])
@pytest.mark.parametrize("async_mode", [False, True])
def test_actual_auxiliary_fallback_reaches_chat_completions(
    tmp_path, monkeypatch, actual_endpoint, provider, async_mode
):
    from agent.auxiliary_client import call_llm, async_call_llm

    local_url, requests = actual_endpoint
    actual_url = local_url.replace("127.0.0.1", "api.actual.inc") + "/v1"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("ACTUAL_API_KEY", "actual-test-key")
    config = {
        "model": {
            "provider": "actual",
            "default": "test-model",
            "base_url": actual_url,
        },
        "providers": {
            "actual-relay": {
                "base_url": actual_url,
                "key_env": "ACTUAL_API_KEY",
                "transport": "codex_responses",
            }
        },
        "auxiliary": {
            "session_search": {
                "provider": "custom",
                "model": "unavailable-model",
                "base_url": local_url + "/v1",
                "fallback_chain": [
                    {
                        "provider": provider,
                        "model": "test-model",
                        "api_mode": "codex_responses",
                        "base_url": actual_url,
                    }
                ],
            }
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    kwargs = {
        "task": "session_search",
        "messages": [{"role": "user", "content": "Summarize the results."}],
        "timeout": 5,
    }
    response = (
        asyncio.run(async_call_llm(**kwargs)) if async_mode else call_llm(**kwargs)
    )
    assert response.choices[0].message.content == "The task is complete."
    assert requests[0][1]["model"] == "unavailable-model"
    assert requests[-1][1]["model"] == "test-model"
    assert all(path == "/v1/chat/completions" for path, _payload in requests)


@pytest.mark.parametrize("override", ["", "http://127.0.0.1:8081", "invalid-url"])
@pytest.mark.parametrize("configured_provider", ["actual", "aci"])
def test_actual_setup_keeps_provider_settings_in_yaml(
    tmp_path, monkeypatch, override, configured_provider
):
    from hermes_cli import config as config_module
    from hermes_cli import model_setup_flows as setup
    from hermes_cli.auth import resolve_api_key_provider_credentials
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from providers import get_provider_profile

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("ACTUAL_API_KEY", "actual-test-key")
    monkeypatch.setenv("ACTUAL_BASE_URL", "http://127.0.0.1:8089")
    configured_url = "http://127.0.0.1:8080"
    raw = {
        "model": {
            "provider": configured_provider,
            "default": "old-model",
            "base_url": configured_url,
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    env_path = tmp_path / ".env"
    env_path.write_text("ACTUAL_API_KEY=actual-test-key\n", encoding="utf-8")
    monkeypatch.setattr(
        setup,
        "_ensure_flow_api_key",
        lambda *_a, **_kw: ("actual-test-key", "actual-test-key", False),
    )
    monkeypatch.setattr(setup, "_ask", lambda *_a, **_kw: override)
    monkeypatch.setattr(setup, "_api_key_provider_model_list", lambda *_a: [])
    monkeypatch.setattr(setup, "_pick_model_or_prompt", lambda *_a, **_kw: "test-model")
    setup._model_flow_api_key_provider(config_module.load_config(), "actual")

    expected_url = override if override.startswith("http://") else configured_url
    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved["model"]["provider"] == "actual"
    assert saved["model"]["base_url"] == expected_url
    assert saved["model"]["default"] == "test-model"
    assert env_path.read_text(encoding="utf-8") == "ACTUAL_API_KEY=actual-test-key\n"
    assert get_provider_profile("actual").env_vars == ("ACTUAL_API_KEY",)
    assert "ACTUAL_BASE_URL" not in config_module.OPTIONAL_ENV_VARS
    assert "ACTUAL_API_MODE" not in config_module.OPTIONAL_ENV_VARS
    assert (
        resolve_api_key_provider_credentials("actual")["base_url"]
        == expected_url + "/v1"
    )
    for kwargs in ({}, {"explicit_api_key": "actual-test-key"}):
        runtime = resolve_runtime_provider(requested="actual", **kwargs)
        assert runtime["base_url"] == expected_url + "/v1"
        assert runtime["api_mode"] == "chat_completions"


def test_actual_key_reload_keeps_yaml_endpoint(tmp_path, monkeypatch, actual_endpoint):
    from run_agent import AIAgent

    base_url, requests = actual_endpoint
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("ACTUAL_API_KEY", "actual-test-key")
    monkeypatch.setenv("ACTUAL_BASE_URL", "http://127.0.0.1:1")
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({
            "model": {"provider": "aci", "default": "test-model", "base_url": base_url},
        }),
        encoding="utf-8",
    )
    env_path = tmp_path / ".env"
    env_path.write_text("ACTUAL_API_KEY=actual-test-key\n", encoding="utf-8")
    agent = AIAgent(
        provider="actual",
        base_url=base_url,
        api_key="actual-test-key",
        model="test-model",
        enabled_toolsets=[],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=False,
    )
    try:
        assert not agent._try_refresh_env_client_credentials()
        env_path.write_text("ACTUAL_API_KEY=actual-rotated-key\n", encoding="utf-8")
        assert agent._try_refresh_env_client_credentials()
        assert agent.api_key == "actual-rotated-key"
        assert agent.base_url == base_url + "/v1"
        assert agent.api_mode == "chat_completions"
        response = agent._interruptible_api_call(
            agent._build_api_kwargs([{"role": "user", "content": "Reply briefly."}], [])
        )
        assert response.choices[0].message.content == "The task is complete."
        assert [
            path for path, body in requests if body.get("model") == "test-model"
        ] == ["/v1/chat/completions"]
    finally:
        agent.client.close()

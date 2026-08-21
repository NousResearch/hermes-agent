"""Contracts for request-scoped, model-locked ``/v1/runs`` execution."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.request_model_runtime import (
    MODEL_API_KEY_HEADER,
    RequestModelRuntimeError,
    parse_request_model_runtime,
)


_SERVER_KEY = "server-auth-key-that-is-long-enough"
_UPSTREAM_KEY = "tenant-upstream-secret"


def _runtime_body(*, model: str = "tenant-model", provider: str = "custom"):
    return {
        "input": "hello",
        "model": model,
        "provider": provider,
        "model_options": {"reasoning_effort": "high"},
        "runtime_model": {
            "base_url": "https://models.example.test/v1",
            "api_mode": "chat_completions",
            "max_tokens": 4096,
            "request_overrides": {
                "temperature": 0.2,
                "top_p": 0.9,
                "parallel_tool_calls": False,
            },
        },
    }


def _headers(*, upstream_key: str = _UPSTREAM_KEY):
    return {
        "Authorization": f"Bearer {_SERVER_KEY}",
        MODEL_API_KEY_HEADER: upstream_key,
    }


def _adapter(*, enabled: bool = True):
    return APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "key": _SERVER_KEY,
                "allow_request_model_runtime": enabled,
            },
        )
    )


def _runs_app(adapter: APIServerAdapter):
    app = web.Application()
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    return app


class TestRequestModelRuntimeParsing:
    def test_valid_runtime_is_immutable_and_repr_redacts_transport(self):
        runtime = parse_request_model_runtime(
            _runtime_body(),
            api_key_header=f"Bearer {_UPSTREAM_KEY}",
        )

        assert runtime is not None
        assert runtime.model == "tenant-model"
        assert runtime.provider == "custom"
        assert runtime.api_key == _UPSTREAM_KEY
        assert runtime.agent_kwargs() == {
            "provider": "custom",
            "base_url": "https://models.example.test/v1",
            "api_key": _UPSTREAM_KEY,
            "api_mode": "chat_completions",
            "max_tokens": 4096,
            "request_overrides": {
                "temperature": 0.2,
                "top_p": 0.9,
                "parallel_tool_calls": False,
            },
        }
        assert _UPSTREAM_KEY not in repr(runtime)
        assert "models.example.test" not in repr(runtime)
        with pytest.raises(TypeError):
            runtime.request_overrides["temperature"] = 0.9

    @pytest.mark.parametrize(
        ("mutation", "expected_param"),
        [
            (lambda body: body["runtime_model"].update(api_key="in-body"), "runtime_model"),
            (lambda body: body["runtime_model"].update(base_url="file:///etc/passwd"), "runtime_model.base_url"),
            (lambda body: body["runtime_model"].update(api_mode="unknown"), "runtime_model.api_mode"),
            (
                lambda body: body["runtime_model"]["request_overrides"].update(messages=[]),
                "runtime_model.request_overrides",
            ),
        ],
    )
    def test_unsafe_or_unknown_fields_fail_closed(self, mutation, expected_param):
        body = _runtime_body()
        mutation(body)
        with pytest.raises(RequestModelRuntimeError) as exc_info:
            parse_request_model_runtime(body, api_key_header=_UPSTREAM_KEY)
        assert exc_info.value.param == expected_param


class TestRequestModelRuntimeRunsContract:
    @pytest.mark.asyncio
    async def test_disabled_feature_rejects_before_agent_creation(self):
        adapter = _adapter(enabled=False)
        async with TestClient(TestServer(_runs_app(adapter))) as client:
            with patch.object(adapter, "_create_agent") as create_agent:
                response = await client.post(
                    "/v1/runs",
                    json=_runtime_body(),
                    headers=_headers(),
                )
                payload = await response.json()

        assert response.status == 403
        assert payload["error"]["code"] == "request_model_runtime_disabled"
        create_agent.assert_not_called()

    @pytest.mark.asyncio
    async def test_feature_requires_api_server_auth_even_when_handler_is_embedded(self):
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={"allow_request_model_runtime": True},
            )
        )
        async with TestClient(TestServer(_runs_app(adapter))) as client:
            response = await client.post(
                "/v1/runs",
                json=_runtime_body(),
                headers={MODEL_API_KEY_HEADER: _UPSTREAM_KEY},
            )
            payload = await response.json()

        assert response.status == 403
        assert payload["error"]["code"] == "request_model_runtime_requires_auth"

    @pytest.mark.asyncio
    async def test_runtime_lock_cannot_be_disabled(self):
        body = _runtime_body()
        body["require_model_lock"] = False
        async with TestClient(TestServer(_runs_app(_adapter()))) as client:
            response = await client.post(
                "/v1/runs",
                json=body,
                headers=_headers(),
            )
            payload = await response.json()

        assert response.status == 400
        assert payload["error"]["code"] == "request_model_runtime_requires_lock"

    @pytest.mark.asyncio
    async def test_virtual_model_alias_is_rejected(self):
        adapter = _adapter()
        body = _runtime_body(model=adapter._model_name)
        async with TestClient(TestServer(_runs_app(adapter))) as client:
            response = await client.post(
                "/v1/runs",
                json=body,
                headers=_headers(),
            )
            payload = await response.json()

        assert response.status == 400
        assert payload["error"]["code"] == "request_model_runtime_alias_conflict"

    @pytest.mark.asyncio
    async def test_run_receives_complete_locked_runtime_without_secret_egress(self):
        adapter = _adapter()
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "done"}
        mock_agent.session_prompt_tokens = 1
        mock_agent.session_completion_tokens = 1
        mock_agent.session_total_tokens = 2

        async with TestClient(TestServer(_runs_app(adapter))) as client:
            with patch.object(adapter, "_create_agent", return_value=mock_agent) as create_agent:
                response = await client.post(
                    "/v1/runs",
                    json=_runtime_body(),
                    headers=_headers(),
                )
                assert response.status == 202
                run_id = (await response.json())["run_id"]

                for _ in range(40):
                    if create_agent.call_args is not None:
                        break
                    await asyncio.sleep(0.05)

                status_response = await client.get(
                    f"/v1/runs/{run_id}",
                    headers={"Authorization": f"Bearer {_SERVER_KEY}"},
                )
                status_payload = await status_response.json()

        kwargs = create_agent.call_args.kwargs
        runtime = kwargs["request_model_runtime"]
        assert kwargs["confirmed_runtime_lock"] is True
        assert kwargs["route"] is None
        assert runtime.api_key == _UPSTREAM_KEY
        assert runtime.model == "tenant-model"
        assert runtime.request_overrides["temperature"] == 0.2
        assert _UPSTREAM_KEY not in json.dumps(status_payload)

    @pytest.mark.asyncio
    async def test_capabilities_advertise_opt_in_and_credential_header(self):
        adapter = _adapter()
        async with TestClient(TestServer(_runs_app(adapter))) as client:
            response = await client.get(
                "/v1/capabilities",
                headers={"Authorization": f"Bearer {_SERVER_KEY}"},
            )
            payload = await response.json()

        assert payload["features"]["run_request_model_runtime"] is True
        assert (
            payload["features"]["run_request_model_credential_header"]
            == MODEL_API_KEY_HEADER
        )


class TestRequestModelRuntimeAgentCreation:
    def test_request_runtime_bypasses_global_resolution_and_disables_fallback(
        self,
        monkeypatch,
    ):
        adapter = _adapter()
        runtime = parse_request_model_runtime(
            _runtime_body(),
            api_key_header=_UPSTREAM_KEY,
        )
        captured = {}

        class FakeAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.__dict__.update(kwargs)

        monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
        monkeypatch.setattr(
            "gateway.run._resolve_runtime_agent_kwargs",
            lambda: (_ for _ in ()).throw(AssertionError("global runtime was resolved")),
        )
        monkeypatch.setattr(
            "gateway.run._resolve_gateway_model",
            lambda: (_ for _ in ()).throw(AssertionError("global model was resolved")),
        )
        monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
        monkeypatch.setattr("gateway.run._checkpoint_agent_kwargs", lambda _cfg: {})
        monkeypatch.setattr("gateway.run._current_max_iterations", lambda: 90)
        monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_: set())
        monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)
        monkeypatch.setattr(
            adapter,
            "_session_model_override_for",
            lambda *_: (_ for _ in ()).throw(AssertionError("session override was read")),
        )

        agent = adapter._create_agent(
            session_id="tenant-session",
            gateway_session_key="tenant-memory-scope",
            requested_model=runtime.model,
            requested_provider=runtime.provider,
            model_options={"reasoning_effort": "high"},
            confirmed_runtime_lock=True,
            request_model_runtime=runtime,
        )

        assert captured["model"] == "tenant-model"
        assert captured["provider"] == "custom"
        assert captured["base_url"] == "https://models.example.test/v1"
        assert captured["api_key"] == _UPSTREAM_KEY
        assert captured["api_mode"] == "chat_completions"
        assert captured["max_tokens"] == 4096
        assert captured["request_overrides"]["temperature"] == 0.2
        assert captured["fallback_model"] is None
        assert captured["reasoning_config"] == {"enabled": True, "effort": "high"}
        assert agent._runtime_model_locked is True
        assert agent._runtime_model_lock_source == "request_runtime"
        assert agent._hermes_api_runtime == {
            "provider": "custom",
            "model": "tenant-model",
            "route_source": "request_runtime",
        }
        assert adapter._last_resolved_model == {}

    def test_locked_auxiliary_failure_never_uses_profile_fallback(self, monkeypatch):
        from agent import auxiliary_client as auxiliary

        invalid_response = MagicMock()
        invalid_response.choices = []
        client = MagicMock()

        monkeypatch.setattr(
            auxiliary,
            "_get_cached_client",
            lambda provider, model=None, **kwargs: (client, model),
        )
        monkeypatch.setattr(
            auxiliary,
            "_relay_sync_completion",
            lambda *_args, **_kwargs: invalid_response,
        )
        monkeypatch.setattr(
            auxiliary,
            "_resolve_task_provider_model",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("auxiliary task route was resolved")
            ),
        )
        for fallback_name in (
            "_try_configured_fallback_chain",
            "_try_main_fallback_chain",
            "_try_payment_fallback",
            "_try_main_agent_model_fallback",
        ):
            monkeypatch.setattr(
                auxiliary,
                fallback_name,
                lambda *_args, _name=fallback_name, **_kwargs: (
                    _ for _ in ()
                ).throw(AssertionError(f"{_name} was called")),
            )

        with pytest.raises(RuntimeError):
            auxiliary.call_llm(
                task="title_generation",
                messages=[{"role": "user", "content": "hello"}],
                main_runtime={
                    "provider": "custom",
                    "model": "locked-model",
                    "base_url": "https://locked.example.test/v1",
                    "api_key": "locked-secret",
                    "api_mode": "chat_completions",
                    "model_locked": True,
                },
            )

    @pytest.mark.asyncio
    async def test_locked_async_auxiliary_failure_never_uses_profile_fallback(
        self, monkeypatch
    ):
        from agent import auxiliary_client as auxiliary

        invalid_response = MagicMock()
        invalid_response.choices = []
        client = MagicMock()
        monkeypatch.setattr(
            auxiliary,
            "_get_cached_client",
            lambda provider, model=None, **kwargs: (client, model),
        )
        monkeypatch.setattr(
            auxiliary,
            "_relay_async_completion",
            AsyncMock(return_value=invalid_response),
        )
        monkeypatch.setattr(
            auxiliary,
            "_resolve_task_provider_model",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("auxiliary task route was resolved")
            ),
        )
        for fallback_name in (
            "_try_configured_fallback_chain",
            "_try_main_fallback_chain",
            "_try_payment_fallback",
            "_try_main_agent_model_fallback",
        ):
            monkeypatch.setattr(
                auxiliary,
                fallback_name,
                lambda *_args, _name=fallback_name, **_kwargs: (
                    _ for _ in ()
                ).throw(AssertionError(f"{_name} was called")),
            )

        with pytest.raises(RuntimeError):
            await auxiliary.async_call_llm(
                task="compression",
                messages=[{"role": "user", "content": "hello"}],
                main_runtime={
                    "provider": "custom",
                    "model": "locked-model",
                    "base_url": "https://locked.example.test/v1",
                    "api_key": "locked-secret",
                    "api_mode": "chat_completions",
                    "model_locked": True,
                },
            )


class TestRequestModelRuntimeEndToEnd:
    @pytest.mark.asyncio
    async def test_concurrent_runs_reach_only_their_locked_upstream(self, monkeypatch):
        """Exercise the real APIServerAdapter -> AIAgent -> OpenAI transport.

        Two simultaneous callers intentionally use the same provider slug but
        different endpoints, keys, model ids, and generation parameters.  A
        process-global mutation or fallback would make the recorded requests
        cross; the invariant is that each fake upstream sees exactly its own
        complete runtime snapshot.
        """

        def start_upstream(name: str, expected_key: str, calls: list[dict]):
            # AIAgent construction performs a synchronous endpoint probe.  A
            # stdlib thread-backed server keeps that real request independent
            # of the aiohttp event loop serving /v1/runs.
            class Handler(BaseHTTPRequestHandler):
                protocol_version = "HTTP/1.0"

                def do_POST(self):
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length))
                    if self.path != "/v1/chat/completions":
                        self.send_response(404)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return

                    calls.append(
                        {
                            "authorization": self.headers.get("Authorization"),
                            "payload": payload,
                        }
                    )
                    if self.headers.get("Authorization") != f"Bearer {expected_key}":
                        raw = b'{"error":"wrong key"}'
                        self.send_response(401)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(raw)))
                        self.end_headers()
                        self.wfile.write(raw)
                        return

                    if not payload.get("stream"):
                        response_payload = {
                            "id": f"chatcmpl-{name}-aux",
                            "object": "chat.completion",
                            "created": 1,
                            "model": payload.get("model"),
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": json.dumps(
                                            {"title": f"Tenant {name.upper()} task"}
                                        ),
                                    },
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 3,
                                "completion_tokens": 2,
                                "total_tokens": 5,
                            },
                        }
                        raw = json.dumps(response_payload).encode()
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(raw)))
                        self.end_headers()
                        self.wfile.write(raw)
                        return

                    chunks = [
                        {
                            "id": f"chatcmpl-{name}",
                            "object": "chat.completion.chunk",
                            "created": 1,
                            "model": payload.get("model"),
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "content": f"served-{name}",
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        },
                        {
                            "id": f"chatcmpl-{name}",
                            "object": "chat.completion.chunk",
                            "created": 1,
                            "model": payload.get("model"),
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 3,
                                "completion_tokens": 2,
                                "total_tokens": 5,
                            },
                        },
                    ]
                    raw = (
                        "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
                        + "data: [DONE]\n\n"
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)

                def log_message(self, _format, *args):
                    return

            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            return server

        calls_a: list[dict] = []
        calls_b: list[dict] = []
        key_a = "tenant-a-runtime-key"
        key_b = "tenant-b-runtime-key"

        # The request-runtime path must remain independent of global provider
        # resolution even under real AIAgent construction.
        monkeypatch.setattr(
            "gateway.run._resolve_runtime_agent_kwargs",
            lambda: (_ for _ in ()).throw(AssertionError("global runtime was resolved")),
        )

        upstream_a = start_upstream("a", key_a, calls_a)
        upstream_b = start_upstream("b", key_b, calls_b)
        agents = []
        try:
            adapter = _adapter()
            real_create_agent = adapter._create_agent

            def capture_agent(**kwargs):
                agent = real_create_agent(**kwargs)
                agents.append(agent)
                return agent

            async with TestClient(TestServer(_runs_app(adapter))) as client:
                def body(name: str, base_url: str, temperature: float):
                    payload = _runtime_body(model=f"tenant-{name}-model")
                    payload["runtime_model"]["base_url"] = base_url
                    payload["runtime_model"]["request_overrides"]["temperature"] = temperature
                    return payload

                base_a = f"http://127.0.0.1:{upstream_a.server_port}/v1"
                base_b = f"http://127.0.0.1:{upstream_b.server_port}/v1"
                with patch.object(adapter, "_create_agent", side_effect=capture_agent):
                    response_a, response_b = await asyncio.gather(
                        client.post(
                            "/v1/runs",
                            json=body("a", base_a, 0.1),
                            headers=_headers(upstream_key=key_a),
                        ),
                        client.post(
                            "/v1/runs",
                            json=body("b", base_b, 0.8),
                            headers=_headers(upstream_key=key_b),
                        ),
                    )
                    assert response_a.status == 202
                    assert response_b.status == 202
                    run_a = (await response_a.json())["run_id"]
                    run_b = (await response_b.json())["run_id"]

                    statuses = {}
                    for _ in range(100):
                        for name, run_id in (("a", run_a), ("b", run_b)):
                            response = await client.get(
                                f"/v1/runs/{run_id}",
                                headers={"Authorization": f"Bearer {_SERVER_KEY}"},
                            )
                            statuses[name] = await response.json()
                        if all(
                            statuses[name].get("status") == "completed"
                            for name in ("a", "b")
                        ):
                            break
                        await asyncio.sleep(0.05)

                    # Auto-title runs in a background worker. Wait for its
                    # auxiliary request so the E2E proves nested LLM calls are
                    # locked to the same tenant runtime as the main stream.
                    for _ in range(40):
                        if all(
                            any(not call["payload"].get("stream") for call in calls)
                            for calls in (calls_a, calls_b)
                        ):
                            break
                        await asyncio.sleep(0.05)
        finally:
            for agent in agents:
                agent.close()
            upstream_a.shutdown()
            upstream_a.server_close()
            upstream_b.shutdown()
            upstream_b.server_close()

        assert statuses["a"]["status"] == "completed", statuses["a"]
        assert statuses["b"]["status"] == "completed", statuses["b"]
        assert statuses["a"]["output"] == "served-a"
        assert statuses["b"]["output"] == "served-b"
        assert calls_a
        assert calls_b
        assert all(call["authorization"] == f"Bearer {key_a}" for call in calls_a)
        assert all(call["authorization"] == f"Bearer {key_b}" for call in calls_b)
        assert all(call["payload"]["model"] == "tenant-a-model" for call in calls_a)
        assert all(call["payload"]["model"] == "tenant-b-model" for call in calls_b)
        main_calls_a = [call for call in calls_a if call["payload"].get("stream")]
        main_calls_b = [call for call in calls_b if call["payload"].get("stream")]
        aux_calls_a = [call for call in calls_a if not call["payload"].get("stream")]
        aux_calls_b = [call for call in calls_b if not call["payload"].get("stream")]
        assert len(main_calls_a) == 1
        assert len(main_calls_b) == 1
        assert aux_calls_a
        assert aux_calls_b
        assert main_calls_a[0]["payload"]["temperature"] == 0.1
        assert main_calls_b[0]["payload"]["temperature"] == 0.8
        serialized_status = json.dumps(statuses)
        assert key_a not in serialized_status
        assert key_b not in serialized_status

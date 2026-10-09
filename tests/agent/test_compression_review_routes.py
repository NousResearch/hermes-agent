"""PR #72637: effective routes and actionable abort diagnostics after fallback."""

import json
import time
from types import SimpleNamespace

import httpx
import pytest

from agent import auxiliary_client as aux
from agent.context_compressor import ContextCompressor, pin_summary_route
from agent.conversation_compression_diagnostics import _emit_compression_auth_hint


@pytest.fixture
def compression_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "auxiliary:\n  compression:\n    provider: custom\n    model: aux-A\n"
        "    base_url: https://aux.example/v1\n    api_key: aux-key\n"
    )
    aux._client_cache.clear()
    yield
    aux._client_cache.clear()


def compressor():
    return ContextCompressor(
        model="main-B", provider="custom", base_url="https://main.example/v1",
        api_key="main-key", quiet_mode=True,
    )


def diagnostic(comp):
    warnings = []
    _emit_compression_auth_hint(SimpleNamespace(
        context_compressor=comp, _emit_warning=warnings.append,
    ))
    return "\n".join(warnings)


def install_wire(monkeypatch, handler):
    from openai import OpenAI

    class TransportClient(OpenAI):
        def __init__(self, **kwargs):
            kwargs["http_client"] = httpx.Client(transport=httpx.MockTransport(handler))
            super().__init__(**kwargs)

    monkeypatch.setattr(aux, "OpenAI", TransportClient)


def response(request, content="", finish_reason="stop"):
    if json.loads(request.content).get("stream"):
        chunk = {"id": "summary", "object": "chat.completion.chunk", "created": 0,
                 "model": json.loads(request.content)["model"],
                 "choices": [{"index": 0, "delta": {"role": "assistant", "content": content},
                              "finish_reason": finish_reason}]}
        return httpx.Response(200, request=request, headers={"content-type": "text/event-stream"},
                              text="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n")
    return httpx.Response(200, request=request, json={
        "id": "summary", "object": "chat.completion", "created": 0,
        "model": json.loads(request.content)["model"],
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": finish_reason}],
    })


def test_pinned_missing_key_names_effective_route(compression_config, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    comp = compressor()
    comp.api_key = ""
    install_wire(monkeypatch, lambda request: pytest.fail("missing credential must prevent dispatch"))
    with pin_summary_route({"provider": "groq", "model": "pinned-B"}):
        assert comp._generate_summary([{"role": "user", "content": "summarize"}]) is None
    assert comp._last_attempt_failure_class == "auth"
    rendered = diagnostic(comp)
    assert "groq" in rendered and "pinned-B" in rendered
    assert "no request was dispatched" in rendered
    assert "credential missing" in rendered
    assert "aux-A" not in rendered and "aux.example" not in rendered
    assert not comp._last_aux_call_model


def test_aux_failure_retries_main_at_wire(compression_config, monkeypatch):
    wire = []

    def handle(request):
        wire.append((request.url.host, json.loads(request.content)["model"],
                     request.headers["authorization"]))
        return response(request, "" if request.url.host == "aux.example" else "valid main summary")

    install_wire(monkeypatch, handle)
    comp = compressor()
    result = comp._generate_summary([
        {"role": "user", "content": "Please summarize this work"},
        {"role": "assistant", "content": "Work completed"},
    ])
    assert result and "valid main summary" in result
    assert wire == [("aux.example", "aux-A", "Bearer aux-key"),
                    ("main.example", "main-B", "Bearer main-key")]
    assert comp._last_aux_call_model == "main-B"


@pytest.mark.parametrize("content,finish_reason", [("", "stop"), ("partial", "length")])
def test_http_200_invalid_summary_is_not_unreachable(
    compression_config, monkeypatch, content, finish_reason,
):
    wire = []

    def handle(request):
        wire.append(request.url.host)
        return response(request, content, finish_reason)

    install_wire(monkeypatch, handle)
    comp = compressor()
    # Keep the same runtime identity so this exercises the terminal abort instead of main fallback.
    comp.model = "aux-A"
    assert comp._generate_summary([{"role": "user", "content": "summarize"}]) is None
    assert wire == ["aux.example"]
    rendered = diagnostic(comp)
    assert "aux-A" in rendered
    assert "could not be reached" not in rendered


@pytest.mark.parametrize("rewrite", [False, True])
@pytest.mark.parametrize("mode", ["sync", "stream", "async"])
def test_managed_relay_reports_final_wire_model(tmp_path, monkeypatch, rewrite, mode):
    import asyncio

    pytest.importorskip("nemo_relay")
    from agent import auxiliary_relay, relay_runtime

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    relay_runtime._reset_for_tests()
    lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(), session_id="route-review", platform="cli",
    )
    turn = relay_runtime.SESSION_COORDINATOR.begin_turn(lease, turn_id="turn", task_id="task")
    lease.host.retain_managed_execution("route-review")
    relay = lease.host.relay
    comp = compressor()
    wire = []
    final_model = "aux-B" if rewrite else "aux-A"
    route_info = {}

    def rewrite_request(name, request, annotated):
        annotated.model = final_model
        return relay.LLMRequestInterceptOutcome(request, annotated)

    def create(**request):
        wire.append(request["model"])
        raise ValueError("invalid provider response")

    async def acreate(request):
        return create(**request)

    client = SimpleNamespace(
        base_url="https://aux.example/v1", chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)),
    )
    relay.intercepts.register_llm_request("route-review", 1, False, rewrite_request)
    try:
        with auxiliary_relay._relay_aux_call_scope((), {"task": "compression"}):
            auxiliary_relay._set_relay_auxiliary_route(
                "custom", "aux-A", "chat_completions", route_callback=comp._record_aux_route,
                route_info=route_info,
            )
            kwargs = {"model": "aux-A", "messages": [{"role": "user", "content": "summarize"}]}
            with pytest.raises(ValueError, match="invalid provider response"):
                if mode == "stream":
                    stream = auxiliary_relay._relay_sync_stream(client, {**kwargs, "stream": True})
                    list(stream)
                elif mode == "async":
                    asyncio.run(auxiliary_relay._relay_async_completion(client, kwargs, create=acreate))
                else:
                    auxiliary_relay._relay_sync_completion(client, kwargs, create=lambda r: create(**r))
        assert wire == [final_model]
        assert route_info == {"provider": "custom", "model": final_model}
        comp._last_attempt_failure_class = "other"
        rendered = diagnostic(comp)
        assert f"Model: {final_model}" in rendered
        if rewrite:
            assert "aux-A" not in rendered
    finally:
        relay.intercepts.deregister_llm_request("route-review")
        lease.host.release_managed_execution("route-review")
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
        relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
        relay_runtime._reset_for_tests()


@pytest.mark.parametrize("status,message,is_quota", [
    (402, "Payment required", True),
    (429, "insufficient_quota", True),
    (403, "out of credits", True),
    (401, "Invalid API key", False),
    (403, "Permission denied", False),
])
def test_access_failure_guidance_preserves_quota_semantics(status, message, is_quota):
    comp = compressor()
    comp._record_aux_route("custom", comp.model, comp.base_url)
    error = RuntimeError(message)
    error.status_code = status
    assert comp._on_summary_failure(error, [], None, "") is None
    rendered = diagnostic(comp)
    assert comp._last_summary_auth_failure  # Existing preservation/cooldown contract stays intact.
    if is_quota:
        assert comp._last_attempt_failure_class == "quota"
        assert "billing" in rendered and "quota" in rendered
        assert "credential" not in rendered
    else:
        assert comp._last_attempt_failure_class == "auth"
        assert "auth/permission" in rendered and "credential" in rendered


@pytest.mark.parametrize("content,finish_reason", [
    ("", "stop"), ("partial", "length"), ("I cannot summarize this conversation.", "stop"),
])
def test_nous_self_heal_updates_all_summary_identity_consumers(tmp_path, monkeypatch, content, finish_reason):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "auxiliary:\n  compression:\n    provider: custom\n    model: stale-model\n"
        "    base_url: https://inference-api.nousresearch.com/v1\n    api_key: test-key\n"
    )
    aux._client_cache.clear()
    wire = []

    def handle(request):
        model = json.loads(request.content)["model"]
        wire.append(model)
        if model == "stale-model":
            return httpx.Response(404, request=request, json={"error": {
                "message": "model not found", "type": "not_found_error", "code": "model_not_found",
            }})
        return response(request, content, finish_reason)

    install_wire(monkeypatch, handle)
    monkeypatch.setattr("hermes_cli.models.get_nous_recommended_aux_model", lambda **kw: "healed-model")
    comp = compressor()
    comp._active_compression_telemetry = {"effective_aux_context": None}
    # Inspect the failing summary attempt before the ordinary main-runtime retry can replace it.
    try:
        with pytest.raises(RuntimeError) as raised:
            comp._call_summary_llm("summarize", time.monotonic())
        assert wire[-1] == "healed-model"
        assert wire[:-1] and set(wire[:-1]) == {"stale-model"}
        assert comp._last_aux_call_model == "healed-model"
        assert comp._last_aux_resolved_model == "healed-model"
        assert comp._active_compression_telemetry["aux_model"] == "healed-model"
        assert "model=healed-model" in str(raised.value)
        assert "stale-model" not in str(raised.value)
        comp._last_attempt_failure_class = "other"
        assert "Model: healed-model" in diagnostic(comp)
    finally:
        aux._client_cache.clear()

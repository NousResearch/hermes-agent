"""Native auxiliary events keep physical requests and host-owned credentials."""

import asyncio
import json
from copy import deepcopy

import httpx
import pytest
from openai import OpenAI

from agent import auxiliary_client as ac
from agent.auxiliary_native import NativeRouteError
from hermes_cli import plugins as plugins_mod
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

MODEL = "gpt-6-luna"
MESSAGES = [{"role": "system", "content": "Keep the frozen rules."}, {"role": "user", "content": "Synthetic task."}]


def _final():
    return {
        "id": "resp_test", "object": "response", "created_at": 1, "model": MODEL, "status": "completed",
        "output": [
            {"id": "reason_test", "type": "reasoning", "summary": [], "encrypted_content": "synthetic-reasoning"},
            {"id": "msg_test", "type": "message", "role": "assistant", "status": "completed", "phase": "final_answer",
             "content": [{"type": "output_text", "text": "Synthetic answer.", "annotations": []}]},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13,
                  "input_tokens_details": {"cached_tokens": 8}, "output_tokens_details": {"reasoning_tokens": 1}},
    }


@pytest.fixture
def native_route(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text("model:\n  provider: openai-codex\n  model: gpt-6-luna\n", encoding="utf-8")
    manager = PluginManager()
    manager._discovered = True
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)
    monkeypatch.setattr(plugins_mod, "_plugin_managers_by_home", {})
    ctx = PluginContext(PluginManifest(name="native-consumer", source="user"), manager)
    ctx.register_auxiliary_task("native_checkpoint", display_name="Native checkpoint", description="Synthetic native request")
    calls = []

    def receive(request):
        body = json.loads(request.content)
        calls.append((body, dict(request.headers)))
        final = _final()
        frames = [{"type": "response.output_item.done", "item": item, "output_index": index}
                  for index, item in enumerate(final["output"])]
        frames.append({"type": "response.completed", "response": final})
        data = "".join("data: " + json.dumps(frame) + "\n\n" for frame in frames)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=data)

    client = OpenAI(api_key="synthetic-token", base_url="https://chatgpt.com/backend-api/codex",
                    max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(receive)))
    wrapped = ac.CodexAuxiliaryClient(client, MODEL)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **kw: {
        "provider": "openai-codex", "model": MODEL, "api_mode": "codex_responses",
        "base_url": str(client.base_url), "api_key": client.api_key,
    })
    ctx._test_cached_resolver = ac._get_cached_client
    monkeypatch.setattr(ac, "_get_cached_client", lambda *a, **kw: (wrapped, MODEL))
    monkeypatch.setattr(ac, "_resolve_task_provider_model", lambda *a, **kw: ("openai-codex", MODEL, None, None, "codex_responses"))
    token = ac.set_runtime_main("moa", "synthetic", session_id="session-test", cache_scope="scope-test")
    from agent.plugin_llm import _TrustPolicy
    ctx.llm._policy_loader = lambda _: _TrustPolicy(
        plugin_id="native-consumer", allow_provider_override=True, allowed_providers=frozenset({"openai-codex"}),
        allow_model_override=True, allowed_models=frozenset({MODEL}),
    )
    yield ctx, wrapped, calls, home
    ac.reset_runtime_main(token)
    client.close()


def _capture(ctx):
    events = []
    for name in ("pre_auxiliary_call", "post_auxiliary_call", "pre_auxiliary_native_request", "post_auxiliary_native_request"):
        ctx.register_hook(name, lambda _name=name, **kw: events.append((_name, kw)))
    return events


def _send(task="moa_reference", **kwargs):
    return ac.call_llm(task=task, provider="openai-codex", model=MODEL, messages=deepcopy(MESSAGES), **kwargs)


@pytest.mark.parametrize("stream", [False, True])
def test_native_reference_and_streamed_aggregator_are_observed_once(native_route, stream):
    ctx, _wrapped, calls, _home = native_route
    events = _capture(ctx)
    response = _send(task="moa_aggregator" if stream else "moa_reference", stream=stream)
    assert response.choices[0].message.content == "Synthetic answer."
    assert [name for name, _ in events] == [
        "pre_auxiliary_call", "pre_auxiliary_native_request", "post_auxiliary_native_request", "post_auxiliary_call",
    ]
    pre = events[1][1]
    post = events[2][1]
    assert pre["native_request"] == calls[0][0]
    assert pre["native_request"]["stream"] is True
    assert pre["native_request"]["instructions"] == MESSAGES[0]["content"]
    assert pre["api_request_id"] == events[0][1]["api_request_id"] == post["api_request_id"]
    assert pre["retry_count"] == 0
    assert events[0][1]["request"]["body"]["messages"] == MESSAGES
    assert post["native_response"]["output"][0]["encrypted_content"] == "synthetic-reasoning"
    assert post["native_response"]["output"][1]["phase"] == "final_answer"
    assert post["usage"]["input_tokens_details"]["cached_tokens"] == 8
    assert "synthetic-token" not in json.dumps(pre)
    assert str(_home) not in json.dumps(pre["route_context"])


def test_native_observer_mutation_and_failure_do_not_change_request_or_other_observer(native_route):
    ctx, _wrapped, calls, _home = native_route
    seen = []

    def mutate(**kw):
        kw["native_request"]["input"].clear()
        kw["route_context"]["model"] = "changed"
        raise RuntimeError("Synthetic observer failure")

    ctx.register_hook("pre_auxiliary_native_request", mutate)
    ctx.register_hook("pre_auxiliary_native_request", lambda **kw: seen.append(kw))
    _send()
    assert seen[0]["native_request"] == calls[0][0]
    assert seen[0]["route_context"]["model"] == MODEL


def test_no_native_hooks_make_no_fingerprints(native_route, monkeypatch):
    _ctx, _wrapped, calls, _home = native_route
    monkeypatch.setattr("agent.auxiliary_native._fingerprint", lambda _value: pytest.fail("Unexpected native capture"))
    _send()
    assert len(calls) == 1


def test_complete_native_preserves_prefix_settings_headers_and_terminal_payload(native_route):
    ctx, _wrapped, calls, _home = native_route
    events = _capture(ctx)
    _send(extra_headers={"session_id": "synthetic-affinity"})
    pre = events[1][1]
    body = deepcopy(pre["native_request"])
    body["input"].append({"role": "user", "content": "Make a synthetic checkpoint."})
    result = ctx.llm.complete_native(native_request=body, route_context=pre["route_context"],
                                    expected_session_id="session-test", task="native_checkpoint")
    assert calls[1][0] == body
    assert calls[1][1]["authorization"] == "Bearer synthetic-token"
    assert calls[1][1]["session_id"] == "synthetic-affinity"
    assert result.native_response["output"][0]["encrypted_content"] == "synthetic-reasoning"
    assert result.native_response["output"][1]["phase"] == "final_answer"
    assert result.audit["task"] == "native_checkpoint"
    assert len(calls) == 2


@pytest.mark.parametrize("change", ["context", "model", "prefix", "settings", "body_options", "session", "profile", "credential", "headers", "query", "sdk_timeout", "sdk_headers"])
def test_complete_native_refuses_changes_before_network(native_route, monkeypatch, change):
    ctx, wrapped, calls, home = native_route
    events = _capture(ctx)
    _send()
    pre = events[1][1]
    body = deepcopy(pre["native_request"])
    route = deepcopy(pre["route_context"])
    session = "session-test"
    def change_profile():
        other = home.parent / "other"
        other.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(other))

    changes = {
        "context": lambda: route.__setitem__("provider", "custom"),
        "model": lambda: body.__setitem__("model", "gpt-other"),
        "prefix": lambda: body["input"].__setitem__(0, {"role": "user", "content": "Changed prefix."}),
        "settings": lambda: body.__setitem__("instructions", "Changed rules."),
        "body_options": lambda: body.__setitem__("extra_body", {"model": "gpt-other"}),
        "session": lambda: None,
        "profile": change_profile,
        "credential": lambda: setattr(wrapped._real_client, "api_key", "changed-synthetic-token"),
        "headers": lambda: wrapped._real_client._custom_headers.__setitem__("x-initiator", "changed"),
        "query": lambda: wrapped._real_client._custom_query.__setitem__("unsupported", "changed"),
        "sdk_timeout": lambda: body.__setitem__("timeout", 1),
        "sdk_headers": lambda: body.__setitem__("extra_headers", {"x-initiator": "changed"}),
    }
    changes[change]()
    session = "other-session" if change == "session" else session
    with pytest.raises((NativeRouteError, PermissionError)):
        ctx.llm.complete_native(native_request=body, route_context=route, expected_session_id=session, task="native_checkpoint")
    assert len(calls) == 1


def test_complete_native_applies_task_and_provider_model_trust(native_route):
    from agent.plugin_llm import PluginLlmTrustError, _TrustPolicy

    ctx, _wrapped, calls, _home = native_route
    events = _capture(ctx)
    _send()
    pre = events[1][1]
    kwargs = dict(native_request=pre["native_request"], route_context=pre["route_context"], expected_session_id="session-test")
    with pytest.raises(PluginLlmTrustError):
        ctx.llm.complete_native(**kwargs, task="foreign_task")
    ctx.llm._policy_loader = lambda _: _TrustPolicy(plugin_id="native-consumer")
    with pytest.raises(PluginLlmTrustError):
        ctx.llm.complete_native(**kwargs, task="native_checkpoint")
    assert len(calls) == 1


def test_async_native_reference_keeps_attempt_identity(native_route):
    ctx, wrapped, _calls, _home = native_route
    events = _capture(ctx)
    async_wrapped = ac.AsyncCodexAuxiliaryClient(wrapped)
    from unittest.mock import patch

    async def send():
        with patch.object(ac, "_get_cached_client", return_value=(async_wrapped, MODEL)):
            return await ac.async_call_llm(task="moa_reference", provider="openai-codex", model=MODEL, messages=MESSAGES)

    response = asyncio.run(send())
    assert response.choices[0].message.content == "Synthetic answer."
    native = [(name, event) for name, event in events if "native_request" in name]
    assert [name for name, _ in native] == ["pre_auxiliary_native_request", "post_auxiliary_native_request"]
    assert native[0][1]["api_request_id"] == native[1][1]["api_request_id"]


def test_native_retry_has_one_event_pair_per_physical_attempt(native_route, monkeypatch):
    ctx, wrapped, calls, _home = native_route
    events = _capture(ctx)
    original = wrapped._real_client.responses.create
    attempts = []

    def retry_once(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            from openai import APIConnectionError
            raise APIConnectionError(request=httpx.Request("POST", "https://example.com/responses"))
        return original(**kwargs)

    monkeypatch.setattr(wrapped._real_client.responses, "create", retry_once)
    monkeypatch.setattr(ac, "_TRANSIENT_RETRY_BACKOFF_BASE", 0.0)
    _send()
    native = [(name, event) for name, event in events if "native_request" in name]
    assert len(native) == 4
    assert [event["retry_count"] for _name, event in native] == [0, 0, 1, 1]
    assert len({event["api_request_id"] for _name, event in native}) == 1
    assert native[1][1]["error_type"] == "APIConnectionError"
    assert len(attempts) == 2 and len(calls) == 1


def test_native_cancel_posts_error_without_a_completed_response(native_route, monkeypatch):
    ctx, wrapped, _calls, _home = native_route
    events = _capture(ctx)

    def cancel(**_kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(wrapped._real_client.responses, "create", cancel)
    with pytest.raises(KeyboardInterrupt):
        _send(task="moa_aggregator", stream=True)
    native = [(name, event) for name, event in events if "native_request" in name]
    assert len(native) == 2
    assert native[1][1]["error_type"] == "KeyboardInterrupt"
    assert native[1][1]["native_response"] is None


def test_complete_native_records_usage_once(native_route):
    from agent.aux_accounting import reset_accounting_context, set_accounting_context
    from unittest.mock import Mock

    ctx, _wrapped, _calls, _home = native_route
    events = _capture(ctx)
    _send()
    pre = events[1][1]
    db = Mock()
    token = set_accounting_context(db, "session-test")
    try:
        ctx.llm.complete_native(native_request=pre["native_request"], route_context=pre["route_context"],
                                expected_session_id="session-test", task="native_checkpoint")
    finally:
        reset_accounting_context(token)
    assert db.record_auxiliary_usage.call_count == 1
    kwargs = db.record_auxiliary_usage.call_args.kwargs
    assert kwargs["input_tokens"] == 2 and kwargs["output_tokens"] == 3
    assert kwargs["cache_read_tokens"] == 8 and kwargs["reasoning_tokens"] == 1


def test_complete_native_rechecks_credentials_after_response(native_route, monkeypatch):
    from unittest.mock import patch

    ctx, wrapped, calls, _home = native_route
    events = _capture(ctx)
    _send()
    pre = events[1][1]
    original = wrapped._real_client.responses.create

    def change_after_response(**kwargs):
        result = original(**kwargs)
        wrapped._real_client.api_key = "changed-after-send"
        return result

    with patch.object(wrapped._real_client, "with_options", return_value=wrapped._real_client):
        monkeypatch.setattr(wrapped._real_client.responses, "create", change_after_response)
        with pytest.raises(NativeRouteError, match="credential changed"):
            ctx.llm.complete_native(native_request=pre["native_request"], route_context=pre["route_context"],
                                    expected_session_id="session-test", task="native_checkpoint")
    assert len(calls) == 2


def test_cached_resolution_detects_changed_credential_source(native_route, monkeypatch):
    ctx, wrapped, calls, home = native_route
    source = home / "synthetic-credential.txt"
    source.write_text("synthetic-source-one", encoding="utf-8")
    base_url = str(wrapped._real_client.base_url)
    transport = wrapped._real_client._client._transport
    monkeypatch.setattr(ac, "_client_cache", {})
    monkeypatch.setattr(ac, "_get_cached_client", ctx._test_cached_resolver)
    monkeypatch.setattr(ac, "_resolve_codex_credential_and_base", lambda: (source.read_text(encoding="utf-8"), base_url))
    monkeypatch.setattr(ac, "_create_openai_client", lambda **kw: OpenAI(
        **kw, max_retries=0, http_client=httpx.Client(transport=transport),
    ))
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **kw: {
        "provider": "openai-codex", "model": MODEL, "api_mode": "codex_responses",
        "base_url": base_url, "api_key": source.read_text(encoding="utf-8"),
    })
    events = _capture(ctx)
    _send()
    pre = events[1][1]
    source.write_text("synthetic-source-two", encoding="utf-8")
    with pytest.raises(NativeRouteError, match="credential changed"):
        ctx.llm.complete_native(native_request=pre["native_request"], route_context=pre["route_context"],
                                expected_session_id="session-test", task="native_checkpoint")
    assert len(calls) == 1


def test_real_plugin_discovery_registers_native_hooks_and_task(native_route):
    from agent.plugin_llm import _TrustPolicy

    _ctx, _wrapped, _calls, home = native_route
    plugin = home / "plugins" / "discovered-native"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: discovered-native\nversion: 0.1.0\n", encoding="utf-8")
    (plugin / "__init__.py").write_text(
        "EVENTS = []\nLLM = None\ndef register(ctx):\n"
        "    global LLM\n    LLM = ctx.llm\n"
        "    ctx.register_auxiliary_task('discovered_checkpoint', display_name='Checkpoint', description='Synthetic checkpoint')\n"
        "    ctx.register_hook('pre_auxiliary_native_request', lambda **kw: EVENTS.append(kw))\n", encoding="utf-8",
    )
    (home / "config.yaml").write_text("plugins:\n  enabled: [discovered-native]\n", encoding="utf-8")
    manager = plugins_mod.get_plugin_manager()
    manager._discovered = False
    manager.discover_and_load()
    loaded = manager._plugins["discovered-native"]
    assert loaded.error is None and loaded.module is not None
    _send()
    pre = loaded.module.EVENTS[0]
    loaded.module.LLM._policy_loader = lambda _: _TrustPolicy(
        plugin_id="discovered-native", allow_provider_override=True, allow_model_override=True,
    )
    result = loaded.module.LLM.complete_native(
        native_request=pre["native_request"], route_context=pre["route_context"],
        expected_session_id="session-test", task="discovered_checkpoint",
    )
    assert result.audit["plugin_id"] == "discovered-native"


@pytest.mark.parametrize("headers", [{"session_id": "bad\nvalue"}, {"session_id": "one", "Session_Id": "two"}, {"x-api-key": "secret"}])
def test_unsupported_native_headers_do_not_make_route_context(native_route, headers):
    ctx, _wrapped, _calls, _home = native_route
    events = _capture(ctx)
    # The SDK may reject a malformed header. Observation must not introduce a failure.
    from agent.auxiliary_native import _native_body
    with pytest.raises(NativeRouteError):
        _native_body({"model": MODEL, "input": [], "stream": True, "extra_headers": headers})
    assert events == []


def test_raw_auxiliary_tools_keep_full_schema_and_cannot_change_provider_input(native_route):
    ctx, _wrapped, calls, _home = native_route
    tools = [{"type": "function", "function": {
        "name": "synthetic_tool", "description": "Exact tool text. " * 2000,
        "parameters": {"type": "object", "properties": {"rows": {"type": "array", "items": {
            "type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"],
        }}}, "required": ["rows"]},
    }}]
    snapshots = []
    ctx.register_hook("pre_auxiliary_call", lambda **kw: kw["request_tools"].clear())
    ctx.register_hook("pre_auxiliary_call", lambda **kw: snapshots.append(kw["request_tools"]))
    _send(tools=deepcopy(tools))
    assert snapshots[0] == tools
    assert calls[0][0]["tools"][0]["description"] == tools[0]["function"]["description"]
    assert calls[0][0]["tools"][0]["parameters"]["properties"]["rows"]["items"] == tools[0]["function"]["parameters"]["properties"]["rows"]["items"]


def test_native_query_options_leave_ordinary_call_unchanged_without_a_capture(native_route):
    ctx, wrapped, calls, _home = native_route
    events = _capture(ctx)
    wrapped._real_client._custom_query["unsupported"] = "value"
    response = _send()
    assert response.choices[0].message.content == "Synthetic answer."
    assert [name for name, _event in events] == ["pre_auxiliary_call", "post_auxiliary_call"]
    assert len(calls) == 1


def test_native_timeout_evicts_only_original_cached_transport(native_route, monkeypatch):
    ctx, wrapped, calls, _home = native_route
    events = _capture(ctx)
    _send()
    pre = events[1][1]
    unrelated = object()
    monkeypatch.setattr(ac, "_client_cache", {
        "native-owner": (wrapped, MODEL, None), "unrelated": (unrelated, MODEL, None),
    })

    def timeout_at_start(guard):
        guard._close_client_on_timeout()
        raise TimeoutError("Synthetic timeout")

    monkeypatch.setattr(ac._CodexStreamGuard, "start", timeout_at_start)
    with pytest.raises(TimeoutError):
        ctx.llm.complete_native(native_request=pre["native_request"], route_context=pre["route_context"],
                                expected_session_id="session-test", task="native_checkpoint")
    assert "native-owner" not in ac._client_cache
    assert ac._client_cache["unrelated"][0] is unrelated
    assert len(calls) == 1


@pytest.mark.parametrize("terminal", [False, True])
def test_native_cancelled_create_closes_late_result_without_read(native_route, monkeypatch, terminal):
    from types import SimpleNamespace

    ctx, _wrapped, calls, _home = native_route
    events = _capture(ctx)
    _send()
    pre = events[1][1]
    guards = []
    closed = []

    class LateStream:
        def __iter__(self):
            pytest.fail("Cancelled native stream must not be read")

        def close(self):
            closed.append(True)

    result = SimpleNamespace(output=[], close=lambda: closed.append(True)) if terminal else LateStream()
    monkeypatch.setattr(ac._CodexStreamGuard, "start", lambda guard: guards.append(guard))

    def cancelled_create(_self, **kwargs):
        guards[0]._protected_cancel_check = lambda: True
        guards[0].timed_out.set()
        return result

    from openai.resources.responses import Responses
    monkeypatch.setattr(Responses, "create", cancelled_create)
    with pytest.raises(TimeoutError):
        ctx.llm.complete_native(native_request=pre["native_request"], route_context=pre["route_context"],
                                expected_session_id="session-test", task="native_checkpoint")
    assert closed
    assert len(calls) == 1

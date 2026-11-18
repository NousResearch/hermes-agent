"""Provider-reported cache evidence on the main request observer (#157124)."""
import json
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI

from agent.api_request_hooks import ApiRequestHooksMixin
from agent.codex_runtime import _consume_codex_event_stream
from agent.turn_response_intake import _fire_post_api_request_hook
from hermes_cli import plugins


class _HookAgent(ApiRequestHooksMixin):
    api_mode = "codex_responses"
    provider = "openai-codex"
    base_url = "https://chatgpt.com/backend-api/codex"
    model = "gpt-6.2-sol"
    session_id = "private-session"
    platform = "cli"


@pytest.fixture
def observe(monkeypatch):
    manager = plugins.PluginManager()
    monkeypatch.setattr(plugins, "_delivery_manager", lambda: manager)
    ctx = plugins.PluginContext(plugins.PluginManifest(name="cache-counter", source="user"), manager)
    records = []
    ctx.register_hook("post_api_request", lambda responses_cache=None: records.append(responses_cache))

    def deliver(response, mode="codex_responses"):
        agent = _HookAgent()
        agent.api_mode = mode
        assistant = SimpleNamespace(role="assistant", content="private reply", tool_calls=[])
        _fire_post_api_request_hook(agent, response, assistant, "stop", api_messages=[], api_call_count=1,
                                    api_duration=1, api_start_time=1, api_request_id="private-request",
                                    effective_task_id="private-task", turn_id="private-turn")
        return records.pop()

    return deliver


def test_sdk_completed_frame_to_narrow_plugin_callback(observe):
    response = {
        "id": "private-response", "status": "completed", "instructions": "private prompt", "output": [],
        "usage": {
            "input_tokens": 4446, "output_tokens": 5, "total_tokens": 4451,
            "input_tokens_details": {"cached_tokens": 2944, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
            "attribution": {"request_fields": {
                "instructions": {"input_tokens": 1222, "cached_tokens": 0, "cache_write_tokens": 0,
                                 "output_tokens": 0, "text": "private instruction"},
                "private tool name": {"input_tokens": 8},
            }},
        },
        "prompt_cache_diagnostics": {"type": "unavailable", "message": "private diagnostic"},
        "input_transformations": [],
    }
    frame = {"type": "response.completed", "response": response, "sequence_number": 1}
    sse = f'data: {json.dumps(frame)}\n\ndata: [DONE]\n\n'
    transport = httpx.MockTransport(lambda _: httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"}))
    with OpenAI(api_key="unused", base_url="https://fixture.invalid", http_client=httpx.Client(transport=transport)) as client:
        with client.responses.create(model=_HookAgent.model, input="private input", stream=True) as events:
            result = _consume_codex_event_stream(events, model=_HookAgent.model)

    assert observe(result) == {
        "input_tokens": 4446, "cached_tokens": 2944, "cache_write_tokens": 0,
        "request_fields": {"instructions": {"input_tokens": 1222, "cached_tokens": 0, "cache_write_tokens": 0, "output_tokens": 0}},
        "prompt_cache_diagnostics": {"type": "unavailable"}, "input_transformation_count": 0,
    }


@pytest.mark.parametrize(("diag", "expected"), [
    ({"type": "cache_miss", "reason": "input_changed", "cache_missed_tokens": 512, "comparison_reusable_tokens": 1024},
     {"type": "cache_miss", "reason": "input_changed", "cache_missed_tokens": 512, "comparison_reusable_tokens": 1024}),
    ({"type": "cache_miss", "reason": "private text", "cache_missed_tokens": True, "comparison_reusable_tokens": -1},
     {"type": "cache_miss"}),
    ({"type": ["private diagnostic"]}, None),
])
def test_sparse_or_malformed_terminal_metadata(observe, diag, expected):
    def assemble(fields):
        return _consume_codex_event_stream([{"type": "response.completed", "response": fields}], model=_HookAgent.model)

    result = assemble({"prompt_cache_diagnostics": diag, "input_transformations": [{"id": "private-id"}]})
    evidence = {"input_transformation_count": 1}
    if expected:
        evidence["prompt_cache_diagnostics"] = expected
    assert observe(result) == evidence
    assert observe(assemble({})) is None
    assert observe(result, mode="chat_completions") is None


def test_request_field_counts_are_sparse_not_coerced_or_accumulated(observe):
    def record(fields):
        usage = {"attribution": {"request_fields": fields}}
        event = {"type": "response.completed", "response": {"usage": usage}}
        return observe(_consume_codex_event_stream([event], model=_HookAgent.model))

    assert record({"instructions": {"input_tokens": 4500, "cached_tokens": 4472}}) == {
        "request_fields": {"instructions": {"input_tokens": 4500, "cached_tokens": 4472}}
    }
    assert record({"instructions": {"input_tokens": "private text", "cached_tokens": False},
                   "tools": {"cached_tokens": -1, "cache_write_tokens": 256}}) == {
        "request_fields": {"tools": {"cache_write_tokens": 256}}
    }

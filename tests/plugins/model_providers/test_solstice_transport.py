"""Solstice inference goes through the gemini plugin's Interactions wire on ``interactions:createPerUserQuota``:
stateless (``store: false``), one request per turn, bearer-only headers, and an interrupted stream is an error.

Every test drives the profile's real client (``create_client(...).chat.completions.create``) against a fake
httpx transport that records what went on the wire."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

import providers
from providers.base import ProviderProfile
from agent.gemini_native_adapter import _FREE_TIER_GUIDANCE, _STANDARD_KEY_GUIDANCE, GeminiAPIError

URL = "https://generativelanguage.googleapis.com/v1beta/interactions:createPerUserQuota"
TOKEN = "ya29.test-access-token"
SIG = "c2lnLXNvbHN0aWNl"
USER = [{"role": "user", "content": "hi"}]
COMPLETED = {"event_type": "interaction.completed", "interaction": {
    "id": "i1", "status": "completed", "usage": {"total_input_tokens": 3, "total_output_tokens": 2, "total_tokens": 5}}}
UNARY = {"id": "i1", "status": "completed", "steps": [{"type": "model_output", "content": [{"type": "text", "text": "hello"}]}],
         "usage": {"total_input_tokens": 3, "total_output_tokens": 2, "total_tokens": 5}}


def _sse(*events: dict) -> bytes:
    return "".join(f"event: {e['event_type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def _client(handler, *, base_url="https://generativelanguage.googleapis.com/v1alpha", **kwargs):
    profile = providers.get_provider_profile("solstice")
    assert profile is not None
    return profile.create_client(api_key=TOKEN, base_url=base_url,
                                 http_client=httpx.Client(transport=httpx.MockTransport(handler)), **kwargs)


def _recorder(*, unary: dict = UNARY, events: tuple = (COMPLETED,), status: int = 200, body: bytes | None = None):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if status != 200:
            return httpx.Response(status, content=body or b"", headers={"content-type": "application/json"})
        if json.loads(request.content).get("stream"):
            return httpx.Response(200, content=_sse(*events), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=unary)

    return seen, handler


def _call(client, *, stream: bool, **kwargs):
    result = client.chat.completions.create(model="gemini-4-hydrogen", messages=kwargs.pop("messages", USER), stream=stream, **kwargs)
    return list(result) if stream else result


@pytest.mark.parametrize("max_tokens", [100_000, "100000"])  # a quoted request override counts too
@pytest.mark.parametrize("stream", [False, True])
def test_every_request_is_stateless_single_request_interactions_on_the_per_user_quota_method(stream, max_tokens):
    seen, handler = _recorder()
    # A long-decode budget, a cap above one request's and sampling params: none may reach the wire.
    _call(_client(handler), stream=stream, temperature=0.7, top_p=0.9, max_tokens=max_tokens,
          extra_body={"max_output_tokens": 262_144, "thinking_config": {"includeThoughts": True, "thinkingLevel": "high"}})

    assert len(seen) == 1
    assert str(seen[0].url) == URL
    body = json.loads(seen[0].content)
    assert body["store"] is False and body["stream"] is stream
    assert "previous_interaction_id" not in body and "background" not in body
    generation = body["generation_config"]
    assert "temperature" not in generation and "top_p" not in generation
    assert 0 < generation["max_output_tokens"] <= 65_536


@pytest.mark.parametrize("stream", [False, True])
def test_a_request_the_shared_builder_would_store_still_goes_out_stateless_at_one_requests_cap(stream):
    seen, handler = _recorder()
    client = _client(handler)
    thinking = {"includeThoughts": True, "thinkingLevel": "xhigh"}
    # The shared builder stores a long-decode request (continuation hops resume only from a stored one).
    wire = importlib.import_module("plugins.model_providers.gemini.interactions")
    assert wire.build_interactions_request(model="gemini-4-hydrogen", messages=USER, max_tokens=262_144,
                                           thinking_config=thinking)["store"] is True

    _call(client, stream=stream, max_tokens=262_144, extra_body={"max_output_tokens": 1_048_576, "thinking_config": thinking})

    assert len(seen) == 1
    body = json.loads(seen[0].content)
    assert body["store"] is False
    assert body["generation_config"]["max_output_tokens"] == 65_536


@pytest.mark.parametrize("stream", [False, True])
def test_headers_are_bearer_and_hermes_user_agent_only(stream):
    from hermes_cli.version_info import hermes_user_agent

    seen, handler = _recorder()
    _call(_client(handler, default_headers={"X-Goog-User-Project": "p", "x-goog-api-key": "AIza-x", "DPoP": "proof"}),
          stream=stream)

    request = seen[0]
    assert request.headers["authorization"] == f"Bearer {TOKEN}"
    assert request.headers["user-agent"] == hermes_user_agent("hermes-agent")
    assert request.headers["user-agent"].startswith("hermes-agent/")
    assert request.headers["content-type"] == "application/json"
    for prohibited in ("x-goog-api-key", "x-goog-user-project", "dpop"):
        assert prohibited not in request.headers
    assert "key" not in request.url.params


@pytest.mark.parametrize("stream", [False, True])
def test_read_timeout_is_an_hour_unless_the_caller_sets_one(stream):
    seen, handler = _recorder()
    client = _client(handler)
    _call(client, stream=stream)
    assert seen[0].extensions["timeout"]["read"] == 3600.0

    _call(client, stream=stream, timeout=httpx.Timeout(connect=5.0, read=42.0, write=5.0, pool=5.0))
    assert seen[1].extensions["timeout"]["read"] == 42.0


@pytest.mark.parametrize("stream", [False, True])
def test_a_zero_per_call_timeout_is_passed_through_not_replaced_by_the_default(stream):
    seen, handler = _recorder()
    _call(_client(handler), stream=stream, timeout=0.0)
    assert seen[0].extensions["timeout"] == {"connect": 0.0, "read": 0.0, "write": 0.0, "pool": 0.0}


@pytest.mark.parametrize("base_url", [
    "https://generativelanguage.googleapis.com/v1alpha",  # rows minted with the old base
    "https://generativelanguage.googleapis.com/v1alpha/",
    "https://generativelanguage.googleapis.com/v1beta",
    None,
])
def test_stored_or_missing_base_still_posts_to_the_v1beta_method(base_url):
    seen, handler = _recorder()
    _call(_client(handler, base_url=base_url), stream=False)
    assert str(seen[0].url) == URL


@pytest.mark.parametrize(("include", "summaries"), [(True, "auto"), (False, "none")])
def test_thinking_config_becomes_interactions_thinking_fields(include, summaries):
    seen, handler = _recorder()
    _call(_client(handler), stream=True, extra_body={"thinking_config": {"includeThoughts": include, "thinkingLevel": "xhigh"}})
    generation = json.loads(seen[0].content)["generation_config"]
    assert generation["thinking_level"] == "xhigh" and generation["thinking_summaries"] == summaries


def test_streamed_thought_summary_is_reasoning_and_its_signature_is_replayed_next_turn():
    turn1 = (
        {"event_type": "interaction.created", "interaction": {"id": "i1", "status": "in_progress"}},
        {"event_type": "step.start", "index": 0, "step": {"type": "thought"}},
        {"event_type": "step.delta", "index": 0, "delta": {"type": "thought_summary", "content": {"type": "text", "text": "Reading it first."}}},
        {"event_type": "step.delta", "index": 0, "delta": {"type": "thought_signature", "signature": SIG}},
        {"event_type": "step.stop", "index": 0},
        {"event_type": "step.start", "index": 1, "step": {"type": "function_call", "id": "call_1", "name": "read_file", "arguments": {}}},
        {"event_type": "step.delta", "index": 1, "delta": {"type": "arguments_delta", "arguments": '{"path": "a.py"}'}},
        {"event_type": "step.stop", "index": 1},
        {"event_type": "interaction.completed", "interaction": {"id": "i1", "status": "requires_action", "usage": {}}},
    )
    seen, handler = _recorder(events=turn1)
    client = _client(handler)
    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]
    chunks = _call(client, stream=True, tools=tools, extra_body={"thinking_config": {"includeThoughts": True, "thinkingLevel": "high"}})

    deltas = [c.choices[0].delta for c in chunks]
    assert "".join(d.reasoning or "" for d in deltas) == "Reading it first."
    calls = [tc for d in deltas for tc in (d.tool_calls or [])]
    first = calls[0]
    assert (first.id, first.function.name) == ("call_1", "read_file")
    assert chunks[-1].choices[0].finish_reason == "tool_calls"
    arguments = "".join(tc.function.arguments for tc in calls)

    # Turn 2: the history as the agent stores it, the signature riding the tool call it preceded.
    history = [*USER, {"role": "assistant", "content": "", "tool_calls": [{
        "id": first.id, "type": "function", "function": {"name": "read_file", "arguments": arguments},
        "extra_content": first.extra_content}]},
        {"role": "tool", "tool_call_id": first.id, "content": "print(1)"}]
    _call(client, stream=True, tools=tools, messages=history)
    replayed = json.loads(seen[1].content)["input"]
    assert replayed[1:3] == [{"type": "thought", "signature": SIG},
                             {"type": "function_call", "id": "call_1", "name": "read_file", "arguments": {"path": "a.py"}}]
    assert replayed[3]["type"] == "function_result" and replayed[3]["call_id"] == "call_1"


_TEXT = ({"event_type": "interaction.created", "interaction": {"id": "i1", "status": "in_progress"}},
         {"event_type": "step.start", "index": 0, "step": {"type": "model_output"}},
         {"event_type": "step.delta", "index": 0, "delta": {"type": "text", "text": "partial"}})


@pytest.mark.parametrize("tail", [
    (),  # closed before interaction.completed, no HTTP error
    ({"event_type": "interaction.status_update", "interaction_id": "i1", "status": "failed"},),
    ({"event_type": "interaction.completed", "interaction": {"id": "i1", "status": "failed",
                                                             "errors": [{"code": "internal", "message": "boom"}]}},),
    ({"event_type": "error", "error": {"code": "service_unavailable", "message": "overloaded"}},),
], ids=["closed-early", "status-failed-then-closed", "completed-failed", "error-event"])
def test_interrupted_stream_raises_instead_of_returning_the_partial_turn(tail):
    seen, handler = _recorder(events=(*_TEXT, *tail))
    with pytest.raises(GeminiAPIError) as raised:
        _call(_client(handler), stream=True)
    assert len(seen) == 1  # never a continuation request: nothing is stored to continue from
    assert raised.value.code in ("gemini_stream_error", "gemini_http_500", "gemini_http_503")


def test_http_error_on_the_method_uses_the_interactions_mapper_without_api_key_remedies():
    body = json.dumps({"error": {"code": "too_many_requests", "message": "Quota exceeded on Free Tier"}}).encode()
    for stream in (False, True):
        _, handler = _recorder(status=429, body=body)
        with pytest.raises(GeminiAPIError) as raised:
            _call(_client(handler), stream=stream)
        error = raised.value
        assert error.status_code == 429 and error.code == "gemini_rate_limited"
        assert "too_many_requests" in str(error)  # the snake_case Interactions code, normalized by its mapper
        assert _FREE_TIER_GUIDANCE not in str(error) and _STANDARD_KEY_GUIDANCE not in str(error)


def test_profile_imports_without_httpx_and_loads_the_transport_only_in_create_client():
    root = Path(__file__).resolve().parents[3]
    code = (
        "import sys; sys.modules['httpx'] = None\n"
        "import importlib.util\n"
        "d = 'plugins/model-providers/solstice'\n"
        "spec = importlib.util.spec_from_file_location('plugins.model_providers.solstice', d + '/__init__.py',"
        " submodule_search_locations=[d])\n"
        "m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)\n"
        "assert 'plugins.model_providers.solstice.transport' not in sys.modules\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], check=False, cwd=root, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0 and result.stdout.strip() == "ok", result.stderr[-2000:]


class _ReplacementGeminiProfile(ProviderProfile):
    """A gemini profile from a package with no ``interactions`` module (a replacement plugin)."""


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("gemini", ["absent", "replaced"])
def test_without_the_gemini_plugins_wire_a_call_raises_a_clear_error_and_sends_nothing(monkeypatch, gemini, stream):
    providers.get_provider_profile("gemini")  # discovery has run, so the registry entry exists to remove
    if gemini == "absent":
        monkeypatch.delitem(providers._REGISTRY, "gemini")
    else:
        monkeypatch.setitem(providers._REGISTRY, "gemini", _ReplacementGeminiProfile(name="gemini"))
    seen, handler = _recorder()
    client = _client(handler)  # creating the client must not fail: that would fall back to the generic client
    with pytest.raises(GeminiAPIError) as raised:
        _call(client, stream=stream)
    assert raised.value.code == "gemini_wire_unavailable"
    assert "Solstice sends its requests through the gemini provider plugin's Interactions wire" in str(raised.value)
    assert ("no gemini provider plugin is loaded" if gemini == "absent" else "has no interactions module") in str(raised.value)
    assert seen == []


@pytest.mark.parametrize("stream", [False, True])
def test_an_older_gemini_wire_without_key_guidance_is_refused_before_any_request(monkeypatch, stream):
    """A profile-local gemini plugin copy from before ``http_error(key_guidance=...)`` existed: refused up
    front with the reason, instead of a ``TypeError`` when the first error response comes back."""
    from plugins.model_providers.solstice.transport import interactions_wire

    def old_http_error(response, *, body_text, api_key, base_url):  # the signature before key_guidance
        raise AssertionError("never reached")

    monkeypatch.setattr(interactions_wire.__globals__["gemini_plugin_module"]("interactions"), "http_error", old_http_error)
    seen, handler = _recorder(status=429, body=b'{"error": {"code": "too_many_requests", "message": "slow down"}}')
    client = _client(handler)
    with pytest.raises(GeminiAPIError) as raised:
        _call(client, stream=stream)
    assert raised.value.code == "gemini_wire_unavailable"
    assert "http_error(key_guidance=...)" in str(raised.value)
    assert seen == []

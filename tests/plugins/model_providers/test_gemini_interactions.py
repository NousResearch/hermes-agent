"""Gemini plugin: the Interactions API wire and the per-model choice between it and generateContent.

Payload shapes follow Google's OpenAPI spec for ``/v1beta/interactions`` (CreateModelInteractionParams,
Interaction, InteractionSseEvent)."""

from __future__ import annotations

import importlib
import json

import httpx
import pytest

from hermes_agent import providers
from hermes_agent.agent.gemini_native_adapter import GeminiAPIError

SIG_A, SIG_B = "c2lnLWE=", "c2lnLWI="
PNG = "data:image/png;base64,iVBORw0KGgo="


@pytest.fixture()
def gemini():
    providers.get_provider_profile("gemini")  # discovery imports the bundled plugin
    return (importlib.import_module("plugins.model_providers.gemini.client"),
            importlib.import_module("plugins.model_providers.gemini.interactions"))


@pytest.fixture()
def config(monkeypatch):
    holder: dict = {}
    import hermes_agent.hermes_cli.config
    monkeypatch.setattr(hermes_agent.hermes_cli.config, "load_config_readonly", lambda: holder)
    return holder


def _client(gemini, handler, base_url="https://generativelanguage.googleapis.com/v1beta"):
    client_mod, _ = gemini
    return client_mod.GeminiClient(api_key="k", base_url=base_url, http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def _sse(*events: dict) -> bytes:
    return "".join(f"event: {e['event_type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


HISTORY = [
    {"role": "system", "content": "be brief"},
    {"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": PNG}}]},
    {"role": "assistant", "content": "checking", "extra_content": {"google": {"thought_signature": SIG_A}},
     "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": '{"p": 1}'},
                     "extra_content": {"google": {"thought_signature": SIG_B}}}]},
    {"role": "tool", "tool_call_id": "c1", "content": '{"ok": true}'},
    {"role": "user", "content": "and?"},
]


def test_xhigh_thinking_level_reaches_the_request(gemini):
    _, wire = gemini
    request = wire.build_interactions_request(
        model="gemini-4-hydrogen", messages=[{"role": "user", "content": "hi"}],
        thinking_config={"thinkingLevel": "xhigh", "includeThoughts": True})
    assert request["generation_config"]["thinking_level"] == "xhigh"


def test_request_is_stateless_typed_steps_with_signatures_replayed_as_thought_steps(gemini):
    _, wire = gemini
    request = wire.build_interactions_request(
        model="gemini-3.8-flash", messages=HISTORY, max_tokens=1000,
        tools=[{"type": "function", "function": {"name": "read", "description": "Read", "parameters": {"type": "object", "properties": {"p": {"type": "integer"}}}}}],
        tool_choice="required", thinking_config={"thinkingLevel": "high", "includeThoughts": True},
        response_format={"type": "json_object"},
    )
    assert request["store"] is False and request["model"] == "gemini-3.8-flash"
    assert request["system_instruction"] == "be brief"
    assert request["input"] == [
        {"type": "user_input", "content": [{"type": "text", "text": "look"}, {"type": "image", "mime_type": "image/png", "data": "iVBORw0KGgo="}]},
        {"type": "thought", "signature": SIG_A},
        {"type": "model_output", "content": [{"type": "text", "text": "checking"}]},
        {"type": "thought", "signature": SIG_B},
        {"type": "function_call", "id": "c1", "name": "read", "arguments": {"p": 1}},
        {"type": "function_result", "call_id": "c1", "result": {"ok": True}, "name": "read"},
        {"type": "user_input", "content": [{"type": "text", "text": "and?"}]},
    ]
    assert request["tools"][0]["type"] == "function" and request["tools"][0]["name"] == "read"
    assert request["tools"][0]["parameters"]["properties"]["p"]["type"] == "integer"
    generation = request["generation_config"]
    assert generation["thinking_level"] == "high" and generation["thinking_summaries"] == "auto"
    assert generation["tool_choice"] == "any"
    # Thinking raises a small cap to the model ceiling, as on generateContent.
    assert generation["max_output_tokens"] == 65535
    assert "response_format" not in request  # forced tool calls win over JSON output


def test_tool_results_schema_text_images_and_foreign_turns(gemini):
    _, wire = gemini
    steps, _ = wire.build_input_steps([
        {"role": "assistant", "content": None, "tool_calls": [  # another provider's turn: no signature
            {"id": "a", "type": "function", "function": {"name": "describe", "arguments": "{}"}},
            {"id": "b", "type": "function", "function": {"name": "shot", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": '{"$ref": "#/$defs/x", "$defs": {"x": {}}}'},
        {"role": "tool", "tool_call_id": "b", "content": [{"type": "text", "text": "here"}, {"type": "image_url", "image_url": {"url": PNG}}]},
    ])
    # An unsigned tool-calling turn opens with one placeholder thought: Google rejects it otherwise.
    assert [s["type"] for s in steps] == ["thought", "function_call", "function_call", "function_result", "function_result"]
    assert steps[0]["signature"] == "skip_thought_signature_validator"
    assert isinstance(steps[3]["result"], str)  # a JSON Schema result goes as text
    assert steps[4]["result"] == [{"type": "text", "text": "here"}, {"type": "image", "mime_type": "image/png", "data": "iVBORw0KGgo="}]
    assert steps[4]["name"] == "shot"


def test_tool_calling_turn_always_opens_with_a_thought_step(gemini):
    _, wire = gemini
    call = {"id": "c", "type": "function", "function": {"name": "read", "arguments": "{}"}}
    signed = {**call, "extra_content": {"google": {"thought_signature": SIG_B}}}

    def shape(message):
        steps, _ = wire.build_input_steps([message])
        return [(s["type"], s.get("signature")) for s in steps]

    placeholder = ("thought", "skip_thought_signature_validator")
    # Unsigned text ahead of the call: the placeholder leads the turn, not the call.
    assert shape({"role": "assistant", "content": "checking", "tool_calls": [call]}) == [placeholder, ("model_output", None), ("function_call", None)]
    # Unsigned text ahead of a signed call: the real signature stays with its call.
    assert shape({"role": "assistant", "content": "checking", "tool_calls": [signed]}) == [
        placeholder, ("model_output", None), ("thought", SIG_B), ("function_call", None)]
    # A turn that already opens with its own signature, and a text-only turn, get no placeholder.
    assert shape({"role": "assistant", "content": None, "tool_calls": [signed]}) == [("thought", SIG_B), ("function_call", None)]
    assert shape({"role": "assistant", "content": "hello"}) == [("model_output", None)]


def test_named_tool_choice_json_schema_and_long_decode_store(gemini):
    _, wire = gemini
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
    named = wire.build_interactions_request(model="m", messages=[{"role": "user", "content": "x"}], tools=tools,
                                            tool_choice={"type": "function", "function": {"name": "f"}})
    assert named["generation_config"]["tool_choice"] == {"allowed_tools": {"mode": "any", "tools": ["f"]}}
    structured = wire.build_interactions_request(
        model="m", messages=[{"role": "user", "content": "x"}],
        response_format={"type": "json_schema", "json_schema": {"schema": {"type": "object", "properties": {"a": {"type": "string"}}}}})
    assert structured["response_format"]["mime_type"] == "application/json"
    assert structured["response_format"]["schema"]["properties"]["a"]["type"] == "string"
    long_decode = wire.build_interactions_request(model="gemini-4-argon", messages=[{"role": "user", "content": "x"}], max_tokens=262_144)
    assert long_decode["store"] is True and long_decode["generation_config"]["max_output_tokens"] == 262_144


INTERACTION = {
    "id": "int_1", "status": "requires_action", "created": "t", "updated": "t",
    "steps": [
        {"type": "thought", "signature": SIG_A, "summary": [{"type": "text", "text": "plan"}]},
        {"type": "model_output", "content": [{"type": "text", "text": "Let me read."}]},
        {"type": "function_call", "id": "fc_1", "name": "read", "arguments": {"p": 2}},
    ],
    "usage": {"total_input_tokens": 100, "total_cached_tokens": 40, "total_output_tokens": 20, "total_thought_tokens": 30, "total_tokens": 150},
}


def test_unary_round_trip_through_routing(gemini, config):
    config["providers"] = {"gemini": {"gemini_api": "interactions"}}
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, request.headers.get("x-goog-api-key"), json.loads(request.content)))
        return httpx.Response(200, json=INTERACTION)

    response = _client(gemini, handler).chat.completions.create(model="google/gemini-3.8-flash", messages=[{"role": "user", "content": "hi"}])
    path, key, body = seen[0]
    assert path == "/v1beta/interactions" and key == "k" and body["model"] == "gemini-3.8-flash" and "stream" not in body
    choice = response.choices[0]
    assert choice.finish_reason == "tool_calls" and choice.message.content == "Let me read." and choice.message.reasoning == "plan"
    call = choice.message.tool_calls[0]
    assert (call.id, call.function.name, json.loads(call.function.arguments)) == ("fc_1", "read", {"p": 2})
    assert call.extra_content == {"google": {"thought_signature": SIG_A}}
    usage = response.usage
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (100, 50, 150)
    assert usage.prompt_tokens_details.cached_tokens == 40 and usage.completion_tokens_details.reasoning_tokens == 30

    # What Hermes stores for that turn replays the signature ahead of the call.
    stored = {"role": "assistant", "content": choice.message.content, "extra_content": choice.message.extra_content,
              "tool_calls": [{"id": call.id, "type": "function", "function": {"name": "read", "arguments": call.function.arguments},
                              "extra_content": call.extra_content}]}
    steps, _ = gemini[1].build_input_steps([{"role": "user", "content": "hi"}, stored])
    assert [s["type"] for s in steps] == ["user_input", "thought", "model_output", "function_call"]
    assert steps[1]["signature"] == SIG_A


@pytest.mark.parametrize(("status", "finish"), [("completed", "stop"), ("incomplete", "length"), ("budget_exceeded", "length")])
def test_status_maps_to_finish_reason(gemini, status, finish):
    _, wire = gemini
    out = wire.translate_interaction([{"id": "i", "status": status, "steps": [{"type": "model_output", "content": [{"type": "text", "text": "x"}]}]}], "m")
    assert out.choices[0].finish_reason == finish


def test_failed_interaction_raises_a_classified_error(gemini):
    _, wire = gemini
    with pytest.raises(GeminiAPIError) as raised:
        wire.translate_interaction([{"id": "i", "status": "failed", "errors": [{"code": "RESOURCE_EXHAUSTED", "message": "quota"}]}], "m")
    assert raised.value.status_code == 429 and raised.value.code == "gemini_rate_limited"


def test_http_errors_keep_the_generate_content_mapping(gemini, config):
    config["providers"] = {"gemini": {"gemini_api": "interactions"}}
    body = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded for metric: generate_content_free_tier_requests"}}
    client = _client(gemini, lambda request: httpx.Response(429, json=body, headers={"retry-after": "7"}))
    with pytest.raises(GeminiAPIError) as raised:
        client.chat.completions.create(model="gemini-3.8-flash", messages=[{"role": "user", "content": "hi"}])
    assert raised.value.status_code == 429 and raised.value.code == "gemini_rate_limited"


STREAM = [
    {"event_type": "interaction.created", "interaction": {"id": "int_s", "status": "in_progress"}},
    {"event_type": "step.start", "index": 0, "step": {"type": "thought"}},
    {"event_type": "step.delta", "index": 0, "delta": {"type": "thought_summary", "content": {"type": "text", "text": "thi"}}},
    # A summary item without ``type`` still reads as text.
    {"event_type": "step.delta", "index": 0, "delta": {"type": "thought_summary", "content": {"text": "nk"}}},
    {"event_type": "step.delta", "index": 0, "delta": {"type": "thought_signature", "signature": SIG_B}},
    {"event_type": "step.stop", "index": 0},
    {"event_type": "step.start", "index": 1, "step": {"type": "model_output"}},
    {"event_type": "step.delta", "index": 1, "delta": {"type": "text", "text": "Hel"}},
    {"event_type": "step.delta", "index": 1, "delta": {"type": "text", "text": "lo"}},
    {"event_type": "step.stop", "index": 1},
    {"event_type": "step.start", "index": 2, "step": {"type": "function_call", "id": "fc_9", "name": "run", "arguments": {}}},
    {"event_type": "step.delta", "index": 2, "delta": {"type": "arguments_delta", "arguments": '{"cmd": '}},
    {"event_type": "step.delta", "index": 2, "delta": {"type": "arguments_delta", "arguments": '"ls"}'}},
    {"event_type": "step.stop", "index": 2},
    {"event_type": "step.start", "index": 3, "step": {"type": "function_call", "id": "fc_10", "name": "noargs", "arguments": {}}},
    {"event_type": "step.stop", "index": 3},
    {"event_type": "interaction.completed", "interaction": {"id": "int_s", "status": "requires_action",
                                                           "usage": {"total_input_tokens": 9, "total_output_tokens": 4, "total_thought_tokens": 2, "total_tokens": 15}}},
]


def _collect(chunks):
    text, reasoning, calls, finish, usage = "", "", {}, None, None
    for chunk in chunks:
        delta = chunk.choices[0].delta
        text += delta.content or ""
        reasoning += delta.reasoning or ""
        for tc in delta.tool_calls or []:
            slot = calls.setdefault(tc.index, {"id": tc.id, "name": "", "args": "", "extra": None})
            slot["name"] += tc.function.name or ""
            slot["args"] += tc.function.arguments or ""
            slot["extra"] = slot["extra"] or getattr(tc, "extra_content", None)
        finish = chunk.choices[0].finish_reason or finish
        usage = getattr(chunk, "usage", None) or usage
    return text, reasoning, calls, finish, usage


def test_stream_translates_steps_to_chat_chunks(gemini, config):
    config["providers"] = {"gemini": {"models": {"gemini-3.8-flash": {"gemini_api": "interactions"}}}}
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, content=_sse(*STREAM), headers={"content-type": "text/event-stream"})

    stream = _client(gemini, handler).chat.completions.create(model="gemini-3.8-flash", messages=[{"role": "user", "content": "hi"}], stream=True)
    text, reasoning, calls, finish, usage = _collect(stream)
    assert bodies[0]["stream"] is True
    assert (text, reasoning, finish) == ("Hello", "think", "tool_calls")
    assert calls[0]["id"] == "fc_9" and calls[0]["name"] == "run" and json.loads(calls[0]["args"]) == {"cmd": "ls"}
    assert calls[0]["extra"] == {"google": {"thought_signature": SIG_B}}
    assert calls[1]["name"] == "noargs" and calls[1]["args"] == "{}"
    assert (usage.prompt_tokens, usage.completion_tokens) == (9, 6)


def test_stream_error_event_raises(gemini, config):
    config["providers"] = {"gemini": {"gemini_api": "interactions"}}
    events = [STREAM[0], {"event_type": "error", "error": {"code": "UNAVAILABLE", "message": "overloaded"}}]
    client = _client(gemini, lambda r: httpx.Response(200, content=_sse(*events)))
    with pytest.raises(GeminiAPIError) as raised:
        list(client.chat.completions.create(model="gemini-3.8-flash", messages=[{"role": "user", "content": "hi"}], stream=True))
    assert raised.value.status_code == 503


def _message_signatures(chunks):
    """Message-level (text) signatures a stream carried, in order."""
    return [c.choices[0].delta.extra_content["google"]["thought_signature"] for c in chunks if c.choices[0].delta.extra_content]


THOUGHT_TEXT_THOUGHT_CALL = [
    {"type": "thought", "signature": SIG_A}, {"type": "model_output", "content": [{"type": "text", "text": "checking"}]},
    {"type": "thought", "signature": SIG_B}, {"type": "function_call", "id": "c1", "name": "read", "arguments": {"p": 1}},
]


def test_each_signature_stays_with_the_output_it_preceded(gemini, config):
    """A thought before the text and another before the call: the text keeps the first, the call the second."""
    _, wire = gemini
    out = wire.translate_interaction([{"id": "i", "status": "requires_action", "steps": THOUGHT_TEXT_THOUGHT_CALL}], "m").choices[0].message
    assert out.extra_content == {"google": {"thought_signature": SIG_A}}
    assert out.tool_calls[0].extra_content == {"google": {"thought_signature": SIG_B}}
    stored = {"role": "assistant", "content": out.content, "extra_content": out.extra_content, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "read", "arguments": '{"p": 1}'}, "extra_content": out.tool_calls[0].extra_content}]}
    steps, _ = wire.build_input_steps([stored])
    assert [(s["type"], s.get("signature")) for s in steps] == [("thought", SIG_A), ("model_output", None), ("thought", SIG_B), ("function_call", None)]

    # The same turn streamed: only the text's signature is sent at message level.
    config["providers"] = {"gemini": {"gemini_api": "interactions"}}
    events = [{"event_type": "interaction.created", "interaction": {}},
              {"event_type": "step.start", "index": 0, "step": {"type": "thought"}},
              {"event_type": "step.delta", "index": 0, "delta": {"type": "thought_signature", "signature": SIG_A}},
              {"event_type": "step.start", "index": 1, "step": {"type": "model_output"}},
              {"event_type": "step.delta", "index": 1, "delta": {"type": "text", "text": "checking"}},
              {"event_type": "step.start", "index": 2, "step": {"type": "thought", "signature": SIG_B}},
              {"event_type": "step.start", "index": 3, "step": {"type": "function_call", "id": "c1", "name": "read", "arguments": {"p": 1}}},
              {"event_type": "interaction.completed", "interaction": {"status": "requires_action", "usage": {}}}]
    chunks = list(_client(gemini, lambda r: httpx.Response(200, content=_sse(*events))).chat.completions.create(
        model="gemini-3.8-flash", messages=[{"role": "user", "content": "hi"}], stream=True))
    assert _message_signatures(chunks) == [SIG_A]
    assert _collect(chunks)[2][0]["extra"] == {"google": {"thought_signature": SIG_B}}


def test_signature_on_the_thought_step_itself_reaches_a_text_only_stream(gemini, config):
    config["providers"] = {"gemini": {"gemini_api": "interactions"}}
    events = [{"event_type": "step.start", "index": 0, "step": {"type": "thought", "signature": SIG_A}},
              {"event_type": "step.start", "index": 1, "step": {"type": "model_output", "content": [{"type": "text", "text": "hi"}]}},
              {"event_type": "interaction.completed", "interaction": {"status": "completed", "usage": {}}}]
    chunks = list(_client(gemini, lambda r: httpx.Response(200, content=_sse(*events))).chat.completions.create(
        model="gemini-3.8-flash", messages=[{"role": "user", "content": "hi"}], stream=True))
    assert _message_signatures(chunks) == [SIG_A] and _collect(chunks)[0] == "hi"


def test_context_size_fields_are_set_only_for_a_multi_hop_usage(gemini):
    """Google reports ``initial_input_tokens`` on every response; alone it would make ``turn_usage`` read the
    turn's context as uncached. A multi-hop usage carries the first hop's prompt and its cached part."""
    _, wire = gemini
    single = wire.usage_namespace({"total_input_tokens": 100, "total_cached_tokens": 80, "initial_input_tokens": 100})
    assert not hasattr(single, "initial_prompt_tokens") and single.prompt_tokens_details.cached_tokens == 80
    merged = wire.usage_namespace(wire.merge_usage([{"total_input_tokens": 100, "total_cached_tokens": 80},
                                                    {"total_input_tokens": 110, "total_cached_tokens": 90}]))
    assert (merged.prompt_tokens, merged.prompt_tokens_details.cached_tokens) == (210, 170)
    assert (merged.initial_prompt_tokens, merged.initial_cached_tokens) == (100, 80)


def _hop(i, status, text, token=None):
    return {"id": f"int_{i}", "status": status, **({"continuation_token": token} if token else {}),
            "steps": [{"type": "model_output", "content": [{"type": "text", "text": text}]}],
            "usage": {"total_input_tokens": 10 * i, "total_output_tokens": 5, "total_tokens": 10 * i + 5}}


def test_unary_long_decode_continues_from_the_stored_hop(gemini, config):
    config["providers"] = {"gemini": {"gemini_api": "interactions"}}
    hops = iter([_hop(1, "requires_continuation", "part1 ", "tok1"), _hop(2, "completed", "part2")])
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=next(hops))

    out = _client(gemini, handler).chat.completions.create(
        model="gemini-4-argon", messages=[{"role": "user", "content": "prove it"}], extra_body={"max_output_tokens": 262_144})
    assert bodies[0]["store"] is True and "continuation_token" not in bodies[0]
    assert bodies[1]["previous_interaction_id"] == "int_1" and bodies[1]["continuation_token"] == "tok1"
    assert bodies[1]["input"] == bodies[0]["input"]  # the same request, never the partial output
    assert out.choices[0].message.content == "part1 part2" and out.choices[0].finish_reason == "stop"
    assert out.usage.prompt_tokens == 30 and out.usage.initial_prompt_tokens == 10


def test_stream_long_decode_opens_a_new_hop(gemini, config):
    config["providers"] = {"gemini": {"gemini_api": "interactions"}}
    hop1 = [{"event_type": "interaction.created", "interaction": {"id": "h1"}},
            {"event_type": "step.start", "index": 0, "step": {"type": "model_output"}},
            {"event_type": "step.delta", "index": 0, "delta": {"type": "text", "text": "a"}},
            {"event_type": "interaction.status_update", "interaction_id": "h1", "status": "requires_continuation", "continuation_token": "t1"},
            {"event_type": "interaction.completed", "interaction": {"id": "h1", "status": "requires_continuation", "usage": {"total_input_tokens": 7, "total_tokens": 9}}}]
    hop2 = [{"event_type": "interaction.created", "interaction": {"id": "h2"}},
            {"event_type": "step.start", "index": 0, "step": {"type": "model_output"}},
            {"event_type": "step.delta", "index": 0, "delta": {"type": "text", "text": "b"}},
            {"event_type": "interaction.completed", "interaction": {"id": "h2", "status": "completed", "usage": {"total_input_tokens": 9, "total_tokens": 12}}}]
    streams = iter([hop1, hop2])
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, content=_sse(*next(streams)))

    chunks = _client(gemini, handler).chat.completions.create(
        model="gemini-4-argon", messages=[{"role": "user", "content": "x"}], stream=True, extra_body={"max_output_tokens": 262_144})
    text, _, _, finish, usage = _collect(chunks)
    assert (text, finish) == ("ab", "stop")
    assert bodies[1]["previous_interaction_id"] == "h1" and bodies[1]["continuation_token"] == "t1" and bodies[1]["stream"] is True
    assert usage.prompt_tokens == 16 and usage.initial_prompt_tokens == 7


def _dropping(*parts):
    """A streamed body that sends each part, then drops the connection where a part is an exception."""
    def body():
        for part in parts:
            if isinstance(part, BaseException):
                raise part
            yield part
    return httpx.Response(200, content=body(), headers={"content-type": "text/event-stream"})


def _text_hop(interaction_id, *events):
    return [{"event_type": "interaction.created", "interaction": {"id": interaction_id}},
            {"event_type": "step.start", "index": 0, "step": {"type": "model_output"}}, *events]


def _delta(text):
    return {"event_type": "step.delta", "index": 0, "delta": {"type": "text", "text": text}}


def _checkpoint(interaction_id, token, input_tokens, output_tokens):
    return {"event_type": "interaction.status_update", "interaction_id": interaction_id, "status": "in_progress", "continuation_token": token,
            "usage": {"total_input_tokens": input_tokens, "total_output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens}}


def _done(interaction_id, input_tokens, output_tokens):
    return {"event_type": "interaction.completed", "interaction": {"id": interaction_id, "status": "completed", "usage": {
        "total_input_tokens": input_tokens, "total_output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens}}}


LONG = {"model": "gemini-4-argon", "messages": [{"role": "user", "content": "x"}], "stream": True, "extra_body": {"max_output_tokens": 262_144}}


def test_dropped_stream_resumes_from_its_last_checkpoint(gemini, config):
    config["providers"] = {"gemini": {"gemini_api": "interactions"}}
    first = _dropping(_sse(*_text_hop("h1", _delta("a"), _checkpoint("h1", "cp1", 7, 5), _delta("b"), _checkpoint("h1", "cp2", 7, 9))),
                      httpx.ReadError("connection reset"))
    second = _dropping(_sse(*_text_hop("h2", _delta("c"), _done("h2", 16, 3))))
    responses, bodies = iter([first, second]), []

    def handler(request):
        bodies.append(json.loads(request.content))
        return next(responses)

    text, _, _, finish, usage = _collect(_client(gemini, handler).chat.completions.create(**LONG))
    assert (text, finish) == ("abc", "stop")  # nothing repeated, nothing lost
    assert (bodies[1]["previous_interaction_id"], bodies[1]["continuation_token"]) == ("h1", "cp2")
    assert bodies[1]["input"] == bodies[0]["input"] and bodies[1]["stream"] is True
    # The dropped hop counts up to its last checkpoint; the first hop's input stays the context size.
    assert (usage.prompt_tokens, usage.completion_tokens, usage.initial_prompt_tokens) == (23, 12, 7)


def test_next_hop_rides_the_token_on_the_completed_event(gemini, config):
    """Output can follow a hop's last checkpoint, so the hop continues from ``interaction.completed``'s token."""
    config["providers"] = {"gemini": {"gemini_api": "interactions"}}
    hop_end = {"event_type": "interaction.completed", "interaction": {"id": "h1", "status": "requires_continuation", "continuation_token": "end1",
                                                                      "usage": {"total_input_tokens": 7, "total_output_tokens": 9, "total_tokens": 16}}}
    first = _dropping(_sse(*_text_hop("h1", _delta("a"), _checkpoint("h1", "cp1", 7, 5), _delta("b"), hop_end)))
    second = _dropping(_sse(*_text_hop("h2", _delta("c"), _done("h2", 16, 3))))
    responses, bodies = iter([first, second]), []

    def handler(request):
        bodies.append(json.loads(request.content))
        return next(responses)

    text, _, _, finish, _ = _collect(_client(gemini, handler).chat.completions.create(**LONG))
    assert (text, finish) == ("abc", "stop")
    assert (bodies[1]["previous_interaction_id"], bodies[1]["continuation_token"]) == ("h1", "end1")


def test_drop_after_output_past_the_checkpoint_is_not_resumed(gemini, config):
    config["providers"] = {"gemini": {"gemini_api": "interactions"}}
    first = _dropping(_sse(*_text_hop("h1", _delta("a"), _checkpoint("h1", "cp1", 7, 5), _delta("b"))), httpx.ReadError("connection reset"))
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return first

    with pytest.raises(GeminiAPIError) as raised:  # resuming from cp1 would repeat "b"
        list(_client(gemini, handler).chat.completions.create(**LONG))
    assert raised.value.code == "gemini_stream_error" and len(bodies) == 1


def test_a_checkpoint_that_keeps_dropping_ends_in_an_error(gemini, config):
    config["providers"] = {"gemini": {"gemini_api": "interactions"}}
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return _dropping(_sse(*_text_hop("h1", _delta("a"), _checkpoint("h1", "cp1", 7, 5))), httpx.ReadError("reset"))
        raise httpx.ConnectError("unreachable")

    with pytest.raises(GeminiAPIError) as raised:
        list(_client(gemini, handler).chat.completions.create(**LONG))
    assert raised.value.code == "gemini_stream_error"
    assert [b.get("continuation_token") for b in bodies] == [None, "cp1", "cp1", "cp1"]  # one request + three resends
    assert {b.get("previous_interaction_id") for b in bodies[1:]} == {"h1"}


AI_STUDIO, VERTEX = "https://generativelanguage.googleapis.com/v1beta", "https://aiplatform.googleapis.com/v1beta1/publishers/google"


@pytest.mark.parametrize(("settings", "model", "base", "expected"), [
    # auto (the default): Gemini 3 and later, and the -latest aliases, use Interactions.
    ({}, "gemini-3.8-flash", AI_STUDIO, "interactions"),
    ({}, "google/gemini-4-argon", AI_STUDIO, "interactions"),
    ({}, "gemini-3-flash-preview", AI_STUDIO, "interactions"),
    ({}, "gemini-flash-latest", AI_STUDIO, "interactions"),
    ({}, "gemini-2.5-flash", AI_STUDIO, "generate"),  # thinkingBudget has no Interactions field
    ({}, "gemma-4-31b-it", AI_STUDIO, "generate"),
    # An explicit setting wins, per model over provider-wide; a per-model auto restores the default.
    ({"gemini_api": "generate_content"}, "gemini-3.8-flash", AI_STUDIO, "generate"),
    ({"gemini_api": "interactions"}, "gemini-2.5-flash", AI_STUDIO, "interactions"),
    ({"models": {"gemini-3.8-flash": {"gemini_api": "generate_content"}}}, "gemini-3.8-flash", AI_STUDIO, "generate"),
    ({"gemini_api": "generate_content", "models": {"gemini-3.8-flash": {"gemini_api": "auto"}}}, "gemini-3.8-flash", AI_STUDIO, "interactions"),
    # A per-model setting matches the id as configured or its bare form.
    ({"models": {"google/gemini-3.8-flash": {"gemini_api": "generate_content"}}}, "google/gemini-3.8-flash", AI_STUDIO, "generate"),
    ({"models": {"gemini-3.8-flash": {"gemini_api": "generate_content"}}}, "google/gemini-3.8-flash", AI_STUDIO, "generate"),
    # Interactions exists only on the AI Studio host.
    ({}, "gemini-3.8-flash", VERTEX, "generate"),
    ({"gemini_api": "interactions"}, "gemini-3.8-flash", VERTEX, "generate"),
])
def test_per_model_routing(gemini, config, settings, model, base, expected):
    config["providers"] = {"gemini": settings}
    client_mod, _ = gemini
    assert client_mod.uses_interactions(model, base) is (expected == "interactions")


@pytest.mark.parametrize(("settings", "model", "path"), [
    ({}, "gemini-3.8-flash", "/v1beta/interactions"),
    ({}, "gemini-2.5-flash", "/v1beta/models/gemini-2.5-flash:generateContent"),
    ({"gemini_api": "generate_content"}, "gemini-3.8-flash", "/v1beta/models/gemini-3.8-flash:generateContent"),
])
def test_default_api_per_model_on_the_wire(gemini, config, settings, model, path):
    config["providers"] = {"gemini": settings}
    seen = []

    def handler(request):
        seen.append(request.url.path)
        if request.url.path.endswith("/interactions"):
            return httpx.Response(200, json={"id": "i", "status": "completed", "steps": [{"type": "model_output", "content": [{"type": "text", "text": "ok"}]}]})
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}]})

    out = _client(gemini, handler).chat.completions.create(model=model, messages=[{"role": "user", "content": "hi"}])
    assert seen == [path] and out.choices[0].message.content == "ok"


def test_profile_supplies_the_client_only_on_the_native_surface(gemini):
    client_mod, _ = gemini
    profile = providers.get_provider_profile("gemini")
    native = profile.create_client(api_key="k", base_url="https://generativelanguage.googleapis.com/v1beta")
    try:
        assert isinstance(native, client_mod.GeminiClient)
    finally:
        native.close()
    assert profile.create_client(api_key="k", base_url="https://generativelanguage.googleapis.com/v1beta/openai") is None


# Error bodies captured from the live Interactions endpoint (free-tier key).
@pytest.mark.parametrize(("status", "body", "headers", "code", "guidance"), [
    (429, {"error": {"message": "Rate limit exceeded for model gemini-4-argon (limit: 0 requests per day on Free Tier). "
                                "Please upgrade your tier at https://ai.dev/rate-limit.", "code": "too_many_requests"}},
     {"retry-after": "57563"}, "gemini_rate_limited", True),
    (400, {"error": {"message": "The value 'bogus' is not supported for 'generation_config.thinking_level'.", "code": "invalid_request"}},
     {}, "gemini_http_400", False),
    (400, [{"error": {"code": 400, "message": "API key not valid. Please pass a valid API key.", "status": "INVALID_ARGUMENT",
                      "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID"}]}}],
     {}, "gemini_http_400", False),
])
def test_interactions_http_error_bodies(gemini, config, status, body, headers, code, guidance):
    config["providers"] = {"gemini": {"gemini_api": "interactions"}}
    client = _client(gemini, lambda request: httpx.Response(status, json=body, headers=headers))
    with pytest.raises(GeminiAPIError) as raised:
        client.chat.completions.create(model="gemini-3.8-flash", messages=[{"role": "user", "content": "hi"}])
    error = raised.value
    assert error.status_code == status and error.code == code
    first = body[0] if isinstance(body, list) else body
    assert first["error"]["message"] in str(error) and "returned HTTP" not in str(error)
    assert ("billing" in str(error)) is guidance
    if headers:
        assert error.retry_after == 57563
    if isinstance(body, list):
        assert error.details["reason"] == "API_KEY_INVALID"


def test_in_stream_snake_case_codes_map_to_http_status(gemini):
    _, wire = gemini
    for rpc, status in (("too_many_requests", 429), ("service_unavailable", 503), ("invalid_request", 400)):
        assert wire.interaction_error({"code": rpc, "message": "m"}).status_code == status


def test_gemini_api_is_a_known_providers_key(caplog):
    from hermes_agent.hermes_cli.config_providers import _normalize_custom_provider_entry

    with caplog.at_level("WARNING"):
        _normalize_custom_provider_entry({"base_url": "https://generativelanguage.googleapis.com/v1beta", "gemini_api": "interactions"},
                                         provider_key="gemini-known-key-probe")
    assert "unknown config keys" not in caplog.text

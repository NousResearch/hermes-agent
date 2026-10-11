"""One same-prefix request for compaction: capture, request shape, and refusals, with a real SDK sink."""

import copy
import json
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI

from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
from agent.conversation_compression import CompressionCommitFence
from agent.prefix_request import (
    PrefixRequest, PrefixRequestError, begin_capture, capture_response, publish_response,
)

REPLY = "Completed synthetic reply."


def make_agent(usage=None, reply_content="Synthetic handoff.", finish="stop"):
    calls = []

    def sink(request):
        calls.append(json.loads(request.content))
        content = REPLY if len(calls) == 1 else reply_content
        return httpx.Response(200, request=request, json={
            "id": "synthetic", "object": "chat.completion", "created": 0, "model": "synthetic",
            "choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": content}}],
            "usage": usage or {"prompt_tokens": 5000, "completion_tokens": 5, "total_tokens": 5005},
        })

    client = OpenAI(api_key="synthetic-no-secret", base_url="https://sink.invalid/v1", max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(sink), trust_env=False))
    history = []
    for index in range(4):
        history.extend([{"role": "user", "content": f"{index} data" * 50},
                        {"role": "assistant", "content": " result" * 50}])
    history.append({"role": "user", "content": "Continue."})
    tools = [{"type": "function", "function": {"name": "synthetic_tool", "parameters": {"type": "object"}}}]
    agent = SimpleNamespace(
        client=client, context_compressor=SimpleNamespace(wants_prefix_request=True, context_length=65536),
        session_id="synthetic", model="synthetic", provider="custom", base_url="https://sink.invalid/v1",
        api_mode="chat_completions", tools=tools, reasoning_config={"effort": "high"},
        _cached_system_prompt="Keep the synthetic task.",
        _client_kwargs={"api_key": "synthetic-no-secret", "base_url": "https://sink.invalid/v1"},
        _prefix_source_messages=history, _session_messages=history,
        _create_request_openai_client=lambda **kwargs: client,
        _max_tokens_param=lambda value: {"max_tokens": value},
        _close_request_openai_client=lambda *args, **kwargs: None)
    ordinary = {"model": "synthetic", "messages": [
        {"role": "system", "content": agent._cached_system_prompt}, *copy.deepcopy(history)],
        "tools": tools, "tool_choice": "auto", "max_tokens": 2048, "stream": False,
        "extra_body": {"draft": True, "chat_template_kwargs": {"enable_thinking": True}, "seed": 7}}
    return agent, calls, ordinary, client, history


def ordinary_turn(agent, ordinary, history):
    begin_capture(agent, ordinary)
    response = _dispatch_nonstreaming_api_request(agent, ordinary, make_client=lambda *a, **k: agent.client)
    response = capture_response(agent, ordinary, response)
    publish_response(agent, response)
    history.append({"role": "assistant", "content": response.choices[0].message.content})
    return response


def test_request_keeps_the_prefix_and_settings_and_appends_reply_and_instruction():
    agent, calls, ordinary, client, history = make_agent()
    try:
        ordinary_turn(agent, ordinary, history)
        before = copy.deepcopy(history)
        result = PrefixRequest(agent, history, CompressionCommitFence())("Write the handoff.", timeout_s=30)
        assert len(calls) == 2
        first, second = calls
        assert second["messages"][:len(first["messages"])] == first["messages"]
        assert second["messages"][len(first["messages"]):] == [
            {"role": "assistant", "content": REPLY}, {"role": "user", "content": "Write the handoff."}]
        changed = {"messages", "stream", "stream_options"}
        assert {k: v for k, v in second.items() if k not in changed} == {
            k: v for k, v in first.items() if k not in changed}
        assert second["draft"] is True and second["chat_template_kwargs"] == {"enable_thinking": True}
        assert second["stream"] is False
        assert history == before
        assert result["content"] == "Synthetic handoff."
        assert result["finish_reason"] == "stop"
        assert result["tool_calls"] is False and result["refusal"] is False
        assert result["usage"] == {"prompt_tokens": 5000, "completion_tokens": 5, "cache_read_tokens": None}
        assert result["elapsed_s"] >= 0
    finally:
        client.close()


def test_cached_tokens_are_reported_when_the_server_sends_them():
    usage = {"prompt_tokens": 5000, "completion_tokens": 5, "total_tokens": 5005,
             "prompt_tokens_details": {"cached_tokens": 4864}}
    agent, calls, ordinary, client, history = make_agent(usage=usage)
    try:
        ordinary_turn(agent, ordinary, history)
        result = PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert result["usage"]["cache_read_tokens"] == 4864
    finally:
        client.close()


def test_a_streamed_turn_is_captured():
    agent, calls, ordinary, client, history = make_agent()
    try:
        streamed = {**ordinary, "stream": True, "stream_options": {"include_usage": True}}
        begin_capture(agent, streamed)
        message = SimpleNamespace(role="assistant", content=REPLY, tool_calls=None, refusal=None,
                                  reasoning_content="hidden reasoning")
        response = SimpleNamespace(id="stream-1", model="synthetic", usage=None,
                                   choices=[SimpleNamespace(index=0, message=message, finish_reason="stop")])
        publish_response(agent, capture_response(agent, streamed, response))
        history.append({"role": "assistant", "content": REPLY})
        PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert len(calls) == 1
        assert calls[0]["stream"] is False and "stream_options" not in calls[0]
        assert calls[0]["messages"][-2:] == [{"role": "assistant", "content": REPLY},
                                             {"role": "user", "content": "Write the handoff."}]
    finally:
        client.close()


def test_only_one_attempt():
    agent, calls, ordinary, client, history = make_agent()
    try:
        ordinary_turn(agent, ordinary, history)
        request = PrefixRequest(agent, history)
        request("Write the handoff.", timeout_s=30)
        with pytest.raises(PrefixRequestError, match="already_used"):
            request("Again.", timeout_s=30)
        assert len(calls) == 2
    finally:
        client.close()


@pytest.mark.parametrize("change, reason", [
    (lambda agent, history: history.__setitem__(0, {"role": "user", "content": "Edited."}), "history_changed"),
    (lambda agent, history: history.__setitem__(slice(-3, -1), []), "history_changed"),
    (lambda agent, history: setattr(agent, "model", "other"), "route_changed"),
    (lambda agent, history: setattr(agent, "api_mode", "codex_responses"), "api_mode_unsupported"),
    (lambda agent, history: setattr(agent.context_compressor, "context_length", 1000), "capacity"),
])
def test_refuses_before_any_request(change, reason):
    agent, calls, ordinary, client, history = make_agent()
    try:
        ordinary_turn(agent, ordinary, history)
        change(agent, history)
        with pytest.raises(PrefixRequestError, match=reason):
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert len(calls) == 1
    finally:
        client.close()


def test_refuses_without_a_successful_capture():
    agent, calls, ordinary, client, history = make_agent()
    try:
        history.append({"role": "assistant", "content": REPLY})
        with pytest.raises(PrefixRequestError, match="no_capture"):
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert calls == []
    finally:
        client.close()


def test_rows_after_the_captured_request_are_appended_in_order():
    """Automatic compaction before a turn: the reply and the new user message follow the captured prefix."""
    agent, calls, ordinary, client, history = make_agent()
    try:
        ordinary_turn(agent, ordinary, history)
        history.append({"role": "user", "content": "Next question.", "timestamp": 5})
        PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        first, second = calls
        n = len(first["messages"])
        assert second["messages"][:n] == first["messages"]
        # The instruction joins the new user message: no two adjacent user rows.
        assert second["messages"][n:] == [{"role": "assistant", "content": REPLY},
                                          {"role": "user", "content": "Next question.\n\nWrite the handoff."}]
    finally:
        client.close()


def test_a_tool_call_reply_is_published_and_its_tool_rows_follow():
    """Automatic compaction inside a turn: the tool call and its result follow the captured prefix."""
    agent, calls, ordinary, client, history = make_agent()
    try:
        begin_capture(agent, ordinary)
        call = SimpleNamespace(id="call-1", type="function", function=SimpleNamespace(name="synthetic_tool",
                                                                                      arguments="{}"))
        message = SimpleNamespace(role="assistant", content=None, tool_calls=[call], refusal=None)
        response = SimpleNamespace(usage={"prompt_tokens": 5000, "prompt_tokens_details": {"cached_tokens": 4000}},
                                   choices=[SimpleNamespace(index=0, message=message, finish_reason="tool_calls")])
        publish_response(agent, capture_response(agent, ordinary, response))
        assert agent._prefix_capsule["usage"]["cache_read_tokens"] == 4000
        wire_call = {"id": "call-1", "type": "function", "function": {"name": "synthetic_tool", "arguments": "{}"}}
        history.extend([{"role": "assistant", "content": None, "reasoning": "hidden", "tool_calls": [wire_call]},
                        {"role": "tool", "tool_call_id": "call-1", "name": "synthetic_tool", "content": "result"}])
        request = PrefixRequest(agent, history)
        assert request.cache_read_tokens == 4000
        request("Write the handoff.", timeout_s=30)
        assert calls[0]["messages"][-3:] == [
            {"role": "assistant", "content": None, "tool_calls": [wire_call]},
            {"role": "tool", "tool_call_id": "call-1", "content": "result"},  # The transport drops a tool name.
            {"role": "user", "content": "Write the handoff."}]
    finally:
        client.close()


def test_the_capacity_check_uses_the_reported_prompt_count():
    agent, calls, ordinary, client, history = make_agent(
        usage={"prompt_tokens": 64000, "completion_tokens": 5, "total_tokens": 64005})
    try:
        ordinary_turn(agent, ordinary, history)
        with pytest.raises(PrefixRequestError, match="capacity"):
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert len(calls) == 1
    finally:
        client.close()


def test_capture_age_is_known_after_a_capture():
    agent, calls, ordinary, client, history = make_agent()
    try:
        ordinary_turn(agent, ordinary, history)
        assert 0 <= PrefixRequest(agent, history).capture_age_s < 60
    finally:
        client.close()


def test_cache_read_tokens_is_unknown_without_a_capture():
    agent, calls, ordinary, client, history = make_agent()
    try:
        assert PrefixRequest(agent, history).cache_read_tokens is None
        assert PrefixRequest(agent, history).capture_age_s is None
    finally:
        client.close()


def test_a_cancelled_fence_refuses():
    agent, calls, ordinary, client, history = make_agent()
    try:
        ordinary_turn(agent, ordinary, history)
        fence = CompressionCommitFence()
        fence.cancel_before_commit()
        with pytest.raises(PrefixRequestError, match="cancelled"):
            PrefixRequest(agent, history, fence)("Write the handoff.", timeout_s=30)
        assert len(calls) == 1
    finally:
        client.close()


def test_no_capture_unless_the_engine_asks_for_it():
    agent, calls, ordinary, client, history = make_agent()
    try:
        agent.context_compressor.wants_prefix_request = False
        begin_capture(agent, ordinary)
        assert getattr(agent, "_prefix_capture", None) is None
    finally:
        client.close()


def test_request_local_headers_are_not_captured():
    agent, calls, ordinary, client, history = make_agent()
    try:
        begin_capture(agent, {**ordinary, "extra_headers": {"x-initiator": "user"}})
        assert getattr(agent, "_prefix_capture", None) is None
    finally:
        client.close()


def _tool_round(history, ordinary):
    """One tool round, stored the way Hermes stores it, and the wire copy that Hermes sends for it."""
    history[:] = [
        {"role": "user", "content": "Read the file.",
         "api_content": "Read the file.\n\n[recalled context: the user wants short answers]"},
        {"role": "assistant", "content": "", "finish_reason": "tool_calls", "reasoning": "Look first.",
         "tool_calls": [{"id": "call_1", "call_id": "call_1", "response_item_id": "fc_1", "type": "function",
                         "function": {"name": "synthetic_tool", "arguments": '{"path": "a.txt"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "name": "synthetic_tool", "content": "file text " * 50},
        {"role": "user", "content": "Continue."},
    ]
    ordinary["messages"] = [ordinary["messages"][0],
        {"role": "user", "content": "Read the file.\n\n[recalled context: the user wants short answers]"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "synthetic_tool", "arguments": '{"path":"a.txt"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "file text " * 50},
        {"role": "user", "content": "Continue."}]


def test_host_wire_transforms_of_the_same_rows_are_accepted():
    agent, calls, ordinary, client, history = make_agent()
    try:
        _tool_round(history, ordinary)
        ordinary_turn(agent, ordinary, history)
        PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert len(calls) == 2
        assert calls[1]["messages"][:len(calls[0]["messages"])] == calls[0]["messages"]
    finally:
        client.close()


@pytest.mark.parametrize("change", [
    lambda messages: messages[1].__setitem__("content", "A different request."),
    lambda messages: messages[2]["tool_calls"][0]["function"].__setitem__("arguments", '{"path":"b.txt"}'),
    lambda messages: messages.__delitem__(slice(1, 4)),
    lambda messages: messages.append({"role": "user", "content": "Request-time note."}),
    lambda messages: messages[1].__setitem__("name", "another_user"),
    lambda messages: messages[2]["tool_calls"][0].__setitem__("id", "call_9") or messages[3].__setitem__(
        "tool_call_id", "call_9"),
])
def test_a_request_with_other_rows_is_refused(change):
    agent, calls, ordinary, client, history = make_agent()
    try:
        _tool_round(history, ordinary)
        change(ordinary["messages"])
        ordinary_turn(agent, ordinary, history)
        with pytest.raises(PrefixRequestError, match="source_transform_unsupported"):
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert len(calls) == 1
    finally:
        client.close()


def _execution_middleware(monkeypatch, *callbacks):
    import hermes_cli.plugins as plugins
    manager = SimpleNamespace(_middleware={"llm_execution": list(callbacks)},
                              _report_hook_failure=lambda *args, **kwargs: None)
    monkeypatch.setattr(plugins, "_delivery_manager", lambda: manager)


def test_the_request_runs_through_the_execution_middleware(monkeypatch):
    seen = []

    def audit(request=None, next_call=None, **context):
        seen.append((len(request["messages"]), context.get("purpose"), context.get("session_id")))
        return next_call()
    _execution_middleware(monkeypatch, audit)
    agent, calls, ordinary, client, history = make_agent()
    try:
        ordinary_turn(agent, ordinary, history)
        result = PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert result["content"] == "Synthetic handoff."
        assert seen == [(len(calls[1]["messages"]), "context_prefix_request", "synthetic")]
    finally:
        client.close()


def test_a_blocking_execution_middleware_stops_the_request(monkeypatch):
    def block(request=None, next_call=None, **context):
        return None  # A policy middleware blocks by not calling next_call.
    agent, calls, ordinary, client, history = make_agent()
    try:
        ordinary_turn(agent, ordinary, history)
        _execution_middleware(monkeypatch, block)
        with pytest.raises(PrefixRequestError, match="incomplete_response"):
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert len(calls) == 1
    finally:
        client.close()


def test_a_media_part_in_the_sent_rows_is_refused():
    agent, calls, ordinary, client, history = make_agent()
    try:
        ordinary["messages"][1]["content"] = [
            {"type": "text", "text": ordinary["messages"][1]["content"]},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
        ordinary_turn(agent, ordinary, history)
        with pytest.raises(PrefixRequestError, match="messages_unsupported"):
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert len(calls) == 1
    finally:
        client.close()


def test_a_named_user_row_sent_without_its_name_is_refused():
    # The transport drops ``name`` from tool rows only; a user row without its name lost its speaker.
    agent, calls, ordinary, client, history = make_agent()
    try:
        _tool_round(history, ordinary)
        history[0]["name"] = "alice"
        ordinary_turn(agent, ordinary, history)
        with pytest.raises(PrefixRequestError, match="source_transform_unsupported"):
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert len(calls) == 1
    finally:
        client.close()


def test_rows_with_an_api_content_sidecar_are_sent_as_the_main_loop_sends_them():
    agent, calls, ordinary, client, history = make_agent()
    try:
        history[0]["api_content"] = "[recalled: short answers]\n\n" + history[0]["content"]
        ordinary["messages"][1]["content"] = history[0]["api_content"]
        ordinary_turn(agent, ordinary, history)
        history.append({"role": "user", "content": "Next.", "api_content": "[note]\n\nNext."})
        PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        # The instruction joins the trailing user row: strict chat templates refuse two adjacent user rows.
        assert calls[1]["messages"][-1] == {"role": "user", "content": "[note]\n\nNext.\n\nWrite the handoff."}
    finally:
        client.close()


def test_a_sent_row_that_is_not_the_stored_api_content_is_refused():
    agent, calls, ordinary, client, history = make_agent()
    try:
        history[1]["api_content"] = "The answer is 4."
        ordinary["messages"][2]["content"] = "The answer is 5." + history[1]["content"]
        ordinary_turn(agent, ordinary, history)
        with pytest.raises(PrefixRequestError, match="source_transform_unsupported"):
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
    finally:
        client.close()


def test_the_request_runs_through_the_request_middleware(monkeypatch):
    seen = []

    def redact(request=None, **context):
        seen.append(context.get("purpose"))
        return {"request": json.loads(json.dumps(request).replace("SECRET", "[redacted]"))}
    import hermes_cli.plugins as plugins
    manager = SimpleNamespace(
        _middleware={"llm_request": [redact]}, has_middleware=lambda kind: kind == "llm_request",
        invoke_middleware=lambda kind, **kwargs: [redact(**kwargs)] if kind == "llm_request" else [],
        _report_hook_failure=lambda *args, **kwargs: None)
    agent, calls, ordinary, client, history = make_agent()
    try:
        ordinary_turn(agent, ordinary, history)
        monkeypatch.setattr(plugins, "_delivery_manager", lambda: manager)
        history.append({"role": "user", "content": "The key is SECRET."})
        PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert seen == ["context_prefix_request"]
        assert "SECRET" not in json.dumps(calls[1]) and "The key is [redacted]." in json.dumps(calls[1])
    finally:
        client.close()


def _pad_assistant_rows(ordinary):
    for row in ordinary["messages"]:
        if row["role"] == "assistant":
            row["reasoning_content"] = " "


def test_appended_assistant_rows_get_reasoning_content_like_a_main_request():
    # A thinking-mode route (DeepSeek, Kimi) rejects an assistant row without reasoning_content.
    from agent.message_sanitization import apply_reasoning_content_policy
    agent, calls, ordinary, client, history = make_agent()
    agent._copy_reasoning_content_for_api = lambda source, target: apply_reasoning_content_policy(source, target, True)
    _pad_assistant_rows(ordinary)
    try:
        ordinary_turn(agent, ordinary, history)
        PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert calls[1]["messages"][-2] == {"role": "assistant", "content": REPLY, "reasoning_content": " "}
    finally:
        client.close()


def test_changed_white_space_inside_a_sent_row_is_refused():
    # White space is API-visible: it can change code, tables, or commands.
    agent, calls, ordinary, client, history = make_agent()
    try:
        history[0]["content"] = "def f():\n    return 1"
        ordinary["messages"][1]["content"] = "def f():\n  return 1"
        ordinary_turn(agent, ordinary, history)
        with pytest.raises(PrefixRequestError, match="source_transform_unsupported"):
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
    finally:
        client.close()


def test_a_replacement_reply_from_execution_middleware_is_refused(monkeypatch):
    agent, calls, ordinary, client, history = make_agent()

    def replace(request=None, next_call=None, **context):
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
            content="## Goal\nA handoff that the server did not write.", tool_calls=None, refusal=None))], usage=None)

    def change(request=None, next_call=None, **context):
        response = next_call()
        response.choices[0].message.content = "## Goal\nChanged."
        return response
    try:
        ordinary_turn(agent, ordinary, history)
        for middleware in (replace, change):
            _execution_middleware(monkeypatch, middleware)
            with pytest.raises(PrefixRequestError, match="middleware_changed_reply"):
                PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
    finally:
        client.close()


def test_appended_tool_calls_are_sanitized_like_a_main_request():
    # Strict providers reject call_id and response_item_id; only a Gemini model reads extra_content.
    agent, calls, ordinary, client, history = make_agent()
    try:
        ordinary_turn(agent, ordinary, history)
        history.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "call_id": "call_1", "response_item_id": "fc_1", "type": "function",
             "extra_content": {"google": {"thought_signature": "sig"}},
             "function": {"name": "synthetic_tool", "arguments": "{}"}}]})
        history.append({"role": "tool", "tool_call_id": "call_1", "content": "ok"})
        PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert calls[1]["messages"][-3]["tool_calls"] == [
            {"id": "call_1", "type": "function", "function": {"name": "synthetic_tool", "arguments": "{}"}}]
    finally:
        client.close()


def test_a_request_middleware_that_changes_the_captured_part_is_refused(monkeypatch):
    # The captured request already went through llm_request middleware; a second pass would apply it twice.
    def prepend(request=None, **context):
        return {"request": {**request, "messages": [{"role": "system", "content": "policy"}, *request["messages"]]}}
    import hermes_cli.plugins as plugins
    manager = SimpleNamespace(
        _middleware={"llm_request": [prepend]}, has_middleware=lambda kind: kind == "llm_request",
        invoke_middleware=lambda kind, **kwargs: [prepend(**kwargs)] if kind == "llm_request" else [],
        _report_hook_failure=lambda *args, **kwargs: None)
    agent, calls, ordinary, client, history = make_agent()
    try:
        ordinary_turn(agent, ordinary, history)
        monkeypatch.setattr(plugins, "_delivery_manager", lambda: manager)
        with pytest.raises(PrefixRequestError, match="middleware_rewrite"):
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert len(calls) == 1
    finally:
        client.close()


def test_a_changed_api_content_sidecar_is_a_changed_history():
    agent, calls, ordinary, client, history = make_agent()
    try:
        history[0]["api_content"] = "[a]" + history[0]["content"]
        ordinary["messages"][1]["content"] = history[0]["api_content"]
        ordinary_turn(agent, ordinary, history)
        history[0]["api_content"] = "[b]" + history[0]["content"]
        with pytest.raises(PrefixRequestError, match="history_changed"):
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
    finally:
        client.close()


def test_the_stop_setting_is_not_sent():
    # A stop sequence of the main request could cut the handoff after its headings.
    agent, calls, ordinary, client, history = make_agent()
    try:
        ordinary["stop"] = ["## Next"]
        ordinary_turn(agent, ordinary, history)
        PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert "stop" in calls[0] and "stop" not in calls[1]
    finally:
        client.close()


def test_capacity_is_checked_again_after_the_request_middleware(monkeypatch):
    def expand(request=None, **context):
        # The rows after the capture can change; the instruction (the last row) cannot.
        request["messages"][-2]["content"] += " context" * 100_000
        return {"request": request}
    import hermes_cli.plugins as plugins
    manager = SimpleNamespace(
        _middleware={"llm_request": [expand]}, has_middleware=lambda kind: kind == "llm_request",
        invoke_middleware=lambda kind, **kwargs: [expand(**kwargs)] if kind == "llm_request" else [],
        _report_hook_failure=lambda *args, **kwargs: None)
    agent, calls, ordinary, client, history = make_agent()
    try:
        ordinary_turn(agent, ordinary, history)
        monkeypatch.setattr(plugins, "_delivery_manager", lambda: manager)
        with pytest.raises(PrefixRequestError, match="capacity"):
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert len(calls) == 1
    finally:
        client.close()


def test_a_changed_json_type_in_arguments_is_refused():
    # true and 1 are different on the wire; Python == does not see it.
    agent, calls, ordinary, client, history = make_agent()
    try:
        _tool_round(history, ordinary)
        history[1]["tool_calls"][0]["function"]["arguments"] = '{"path": "a.txt", "force": true}'
        ordinary["messages"][2]["tool_calls"][0]["function"]["arguments"] = '{"path":"a.txt","force":1}'
        ordinary_turn(agent, ordinary, history)
        with pytest.raises(PrefixRequestError, match="source_transform_unsupported"):
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
    finally:
        client.close()


@pytest.mark.parametrize("base_url, kept", [("https://openrouter.ai/api/v1", True), ("https://sink.invalid/v1", False)])
def test_appended_rows_keep_reasoning_details_on_a_replaying_route(base_url, kept):
    agent, calls, ordinary, client, history = make_agent()
    agent.base_url = base_url
    details = [{"type": "reasoning.encrypted", "data": "abc"}]
    try:
        ordinary_turn(agent, ordinary, history)
        history[-1]["reasoning_details"] = details
        PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert calls[1]["messages"][-2].get("reasoning_details") == (details if kept else None)
    finally:
        client.close()


def test_the_native_carrier_of_the_provider_profile_is_replayed(monkeypatch):
    # The main transport replays the <provider>.native_assistant carrier that the provider profile declares,
    # also on a route that does not replay other reasoning_details.
    import providers
    monkeypatch.setattr(providers, "get_provider_profile",
                        lambda name: SimpleNamespace(native_reasoning_details_type="acme.native_assistant"))
    agent, calls, ordinary, client, history = make_agent()
    carrier = [{"type": "acme.native_assistant", "data": "n"}]
    history[1]["reasoning_details"] = carrier
    ordinary["messages"][2]["reasoning_details"] = carrier
    try:
        ordinary_turn(agent, ordinary, history)
        history[-1]["reasoning_details"] = carrier
        PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert len(calls) == 2
        assert calls[1]["messages"][-2].get("reasoning_details") == carrier
    finally:
        client.close()


def _request_middleware(monkeypatch, callback):
    import hermes_cli.plugins as plugins
    manager = SimpleNamespace(
        _middleware={"llm_request": [callback]}, has_middleware=lambda kind: kind == "llm_request",
        invoke_middleware=lambda kind, **kwargs: [callback(**kwargs)] if kind == "llm_request" else [],
        _report_hook_failure=lambda *args, **kwargs: None)
    monkeypatch.setattr(plugins, "_delivery_manager", lambda: manager)


def _rewrite_prefix_row(request):
    request["messages"][1]["content"] = "A different captured request."


def _rewrite_tools(request):
    request["tools"] = []


def _rewrite_model(request):
    request["model"] = "another-model"


def _drop_instruction(request):
    del request["messages"][-1]


@pytest.mark.parametrize("rewrite", [_rewrite_prefix_row, _rewrite_tools, _rewrite_model, _drop_instruction])
def test_a_request_middleware_rewrite_of_the_captured_part_falls_back(monkeypatch, rewrite):
    def middleware(request=None, **context):
        rewrite(request)
        return {"request": request}
    agent, calls, ordinary, client, history = make_agent()
    try:
        ordinary_turn(agent, ordinary, history)
        _request_middleware(monkeypatch, middleware)
        with pytest.raises(PrefixRequestError, match="middleware_rewrite"):
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert len(calls) == 1
    finally:
        client.close()


@pytest.mark.parametrize("in_place", [True, False])
@pytest.mark.parametrize("rewrite", [_rewrite_prefix_row, _rewrite_tools, _rewrite_model, _drop_instruction])
def test_an_execution_middleware_rewrite_of_the_captured_part_falls_back(monkeypatch, rewrite, in_place):
    # The identity check runs inside _send, the last seam before the provider.
    def middleware(request=None, next_call=None, **context):
        if in_place:
            rewrite(request)
            return next_call()
        changed = copy.deepcopy(request)
        rewrite(changed)
        return next_call(changed)
    agent, calls, ordinary, client, history = make_agent()
    try:
        ordinary_turn(agent, ordinary, history)
        _execution_middleware(monkeypatch, middleware)
        with pytest.raises(PrefixRequestError, match="middleware_rewrite"):
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert len(calls) == 1
    finally:
        client.close()


@pytest.mark.parametrize("rewrite", [_rewrite_prefix_row, _rewrite_tools, _rewrite_model])
def test_execution_middleware_cannot_change_the_validation_baseline(monkeypatch, rewrite):
    def middleware(request=None, original_request=None, next_call=None, **context):
        rewrite(request)
        rewrite(original_request)
        return next_call()

    agent, calls, ordinary, client, history = make_agent()
    try:
        ordinary_turn(agent, ordinary, history)
        _execution_middleware(monkeypatch, middleware)
        with pytest.raises(PrefixRequestError, match="middleware_rewrite"):
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert len(calls) == 1
    finally:
        client.close()


def test_an_execution_middleware_can_redact_the_rows_after_the_capture(monkeypatch):
    def redact(request=None, next_call=None, **context):
        changed = copy.deepcopy(request)
        changed["messages"][-2]["content"] = "[redacted]"
        return next_call(changed)
    agent, calls, ordinary, client, history = make_agent()
    try:
        ordinary_turn(agent, ordinary, history)
        _execution_middleware(monkeypatch, redact)
        PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert calls[1]["messages"][-2]["content"] == "[redacted]"
        assert calls[1]["messages"][:len(calls[0]["messages"])] == calls[0]["messages"]
    finally:
        client.close()


@pytest.mark.parametrize("change", [lambda row: row.update(reasoning_content="Other thinking."),
                                    lambda row: row.pop("reasoning_content")])
def test_changed_reasoning_content_in_a_captured_row_is_refused(change):
    # The provider reads reasoning_content. A changed field makes a prefix that the history does not have.
    from agent.message_sanitization import apply_reasoning_content_policy
    agent, calls, ordinary, client, history = make_agent()
    agent._copy_reasoning_content_for_api = lambda source, target: apply_reasoning_content_policy(source, target, True)
    _pad_assistant_rows(ordinary)
    change(ordinary["messages"][2])
    try:
        ordinary_turn(agent, ordinary, history)
        with pytest.raises(PrefixRequestError, match="source_transform_unsupported"):
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert len(calls) == 1
    finally:
        client.close()


@pytest.mark.parametrize("sent, accepted", [
    ([{"type": "reasoning.encrypted", "data": "abc"}], True),
    (None, False),
    ([{"type": "reasoning.encrypted", "data": "other"}], False),
])
def test_reasoning_details_of_a_captured_row_must_be_the_stored_ones(sent, accepted):
    agent, calls, ordinary, client, history = make_agent()
    agent.base_url = "https://openrouter.ai/api/v1"
    history[1]["reasoning_details"] = [{"type": "reasoning.encrypted", "data": "abc"}]
    if sent is not None:
        ordinary["messages"][2]["reasoning_details"] = sent
    try:
        ordinary_turn(agent, ordinary, history)
        if accepted:
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
            assert len(calls) == 2
        else:
            with pytest.raises(PrefixRequestError, match="source_transform_unsupported"):
                PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
            assert len(calls) == 1
    finally:
        client.close()


@pytest.mark.parametrize("main, sent", [
    ({"max_tokens": 128}, {"max_tokens": 2048}),
    ({"max_tokens": 3000}, {"max_tokens": 3000}),
    ({"max_tokens": 60_000}, {"max_tokens": 8192}),
    ({"max_completion_tokens": 500}, {"max_completion_tokens": 2048}),
    # Without a limit the server default applies: a small one cuts the handoff, a large one can take more space
    # than the capacity check reserves. The handoff gets 8,192 tokens in the field of the route.
    ({}, {"max_tokens": 8192}),
])
def test_the_handoff_has_its_own_reply_limit(main, sent):
    # The reply limit of the main request is for another task: a small one cuts the handoff, a large one
    # reserves space that the handoff does not need (and can refuse a request that fits).
    agent, calls, ordinary, client, history = make_agent()
    ordinary.pop("max_tokens")
    ordinary.update(main)
    try:
        ordinary_turn(agent, ordinary, history)
        PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        limits = ("max_tokens", "max_completion_tokens")
        assert {key: calls[1][key] for key in limits if key in calls[1]} == sent
    finally:
        client.close()


def test_a_provider_spelling_of_the_finish_reason_is_normalized():
    # Some OpenAI-compatible servers send STOP. The transport normalizes it for a main request; the capture and
    # the warm reply do not go through the transport.
    agent, calls, ordinary, client, history = make_agent(finish="STOP")
    try:
        ordinary_turn(agent, ordinary, history)
        result = PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert result["finish_reason"] == "stop"
    finally:
        client.close()


@pytest.mark.parametrize("sent, accepted", [
    ("Do not 0 data", False), ("0 data now", False), ("0 data\n\n[context]", False),
    ("Ignore the next line.\n0 data", False), ("0 data", True),
])
def test_text_added_to_a_stored_row_is_refused(sent, accepted):
    # The host stores the text that it sends (api_content). Added text, on the same line or on its own line,
    # can change the meaning.
    agent, calls, ordinary, client, history = make_agent()
    history[0]["content"] = "0 data"
    ordinary["messages"][1]["content"] = sent
    try:
        ordinary_turn(agent, ordinary, history)
        if accepted:
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
            assert len(calls) == 2
        else:
            with pytest.raises(PrefixRequestError, match="source_transform_unsupported"):
                PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
    finally:
        client.close()


def test_web_search_is_not_sent_with_the_handoff():
    agent, calls, ordinary, client, history = make_agent()
    ordinary["web_search_options"] = {}
    try:
        ordinary_turn(agent, ordinary, history)
        PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert "web_search_options" in calls[0] and "web_search_options" not in calls[1]
    finally:
        client.close()


def test_the_capacity_check_reserves_the_larger_reply_limit():
    # 60,000 measured prompt tokens in a 65,536 window: the 6,000-token max_completion_tokens does not fit, even
    # with the smaller max_tokens (2,048) next to it. A server can honor either limit.
    agent, calls, ordinary, client, history = make_agent(usage={"prompt_tokens": 60_000, "completion_tokens": 5,
                                                                "total_tokens": 60_005})
    ordinary["max_completion_tokens"] = 6_000
    try:
        ordinary_turn(agent, ordinary, history)
        with pytest.raises(PrefixRequestError, match="capacity"):
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert len(calls) == 1
    finally:
        client.close()


@pytest.mark.parametrize("sent, accepted", [("0 data", False), ("    0 data", True)])
def test_changed_white_space_around_a_stored_row_is_refused(sent, accepted):
    # A middleware that dedents the first line of a code fragment changes its meaning.
    agent, calls, ordinary, client, history = make_agent()
    history[0]["content"] = "    0 data"
    ordinary["messages"][1]["content"] = sent
    try:
        ordinary_turn(agent, ordinary, history)
        if accepted:
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
            assert len(calls) == 2
        else:
            with pytest.raises(PrefixRequestError, match="source_transform_unsupported"):
                PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
    finally:
        client.close()


def test_no_two_adjacent_user_rows_are_sent():
    # Trailing user rows join as the main loop joins them; the host instruction is the last block.
    agent, calls, ordinary, client, history = make_agent()
    try:
        ordinary_turn(agent, ordinary, history)
        history.extend([{"role": "user", "content": "First."}, {"role": "user", "content": "Second."}])
        PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert calls[1]["messages"][-1] == {"role": "user", "content": "First.\n\nSecond.\n\nWrite the handoff."}
        roles = [row["role"] for row in calls[1]["messages"]]
        assert not any(a == b == "user" for a, b in zip(roles, roles[1:]))
    finally:
        client.close()


def test_a_middleware_can_redact_a_trailing_user_row_joined_with_the_instruction(monkeypatch):
    def redact(request=None, next_call=None, **context):
        changed = copy.deepcopy(request)
        changed["messages"][-1]["content"] = changed["messages"][-1]["content"].replace("SECRET", "[redacted]")
        return next_call(changed)
    agent, calls, ordinary, client, history = make_agent()
    try:
        ordinary_turn(agent, ordinary, history)
        history.append({"role": "user", "content": "The key is SECRET."})
        _execution_middleware(monkeypatch, redact)
        PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert calls[1]["messages"][-1]["content"] == "The key is [redacted].\n\nWrite the handoff."
    finally:
        client.close()


def test_a_middleware_that_changes_the_instruction_block_is_refused(monkeypatch):
    def change(request=None, next_call=None, **context):
        changed = copy.deepcopy(request)
        changed["messages"][-1]["content"] = changed["messages"][-1]["content"].replace("handoff", "poem")
        return next_call(changed)
    agent, calls, ordinary, client, history = make_agent()
    try:
        ordinary_turn(agent, ordinary, history)
        history.append({"role": "user", "content": "Next."})
        _execution_middleware(monkeypatch, change)
        with pytest.raises(PrefixRequestError, match="middleware_rewrite"):
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert len(calls) == 1
    finally:
        client.close()


def test_the_instruction_joins_a_named_last_user_row():
    # The ordinary request also ended with that row; strict chat templates refuse two adjacent user rows.
    agent, calls, ordinary, client, history = make_agent()
    try:
        ordinary_turn(agent, ordinary, history)
        history.append({"role": "user", "content": "Next.", "name": "alice"})
        PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert calls[1]["messages"][-1] == {"role": "user", "name": "alice", "content": "Next.\n\nWrite the handoff."}
    finally:
        client.close()


def test_a_route_without_a_limit_gets_the_handoff_limit_in_its_own_field():
    # AIAgent._max_tokens_param chooses the field: max_completion_tokens for the newer OpenAI families.
    agent, calls, ordinary, client, history = make_agent()
    agent._max_tokens_param = lambda value: {"max_completion_tokens": value}
    ordinary.pop("max_tokens")
    try:
        ordinary_turn(agent, ordinary, history)
        PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert calls[1].get("max_completion_tokens") == 8192 and "max_tokens" not in calls[1]
    finally:
        client.close()


@pytest.mark.parametrize("change, reason", [
    (lambda agent, fence: fence.cancel_before_commit(), "cancelled"),
    (lambda agent, fence: setattr(agent, "model", "another-model"), "route_changed"),
])
def test_the_attempt_is_checked_again_at_the_send_seam(monkeypatch, change, reason):
    # Compression runs on a pooled thread and can outlive a host timeout: an execution middleware can call
    # next_call after a cancel or a model switch.
    agent, calls, ordinary, client, history = make_agent()
    fence = CompressionCommitFence()

    def late(request=None, next_call=None, **context):
        change(agent, fence)
        return next_call()
    try:
        ordinary_turn(agent, ordinary, history)
        _execution_middleware(monkeypatch, late)
        with pytest.raises(PrefixRequestError, match=reason):
            PrefixRequest(agent, history, fence)("Write the handoff.", timeout_s=30)
        assert len(calls) == 1
    finally:
        client.close()


@pytest.mark.parametrize("field, accepted", [
    ({"recipient": "x"}, False), ({"prefix": True}, False),
    # Prompt caching marks rows with cache_control on some routes.
    ({"cache_control": {"type": "ephemeral"}}, True),
])
def test_a_captured_row_with_an_extra_field_is_refused(field, accepted):
    # A provider control that a middleware added changes what the model reads, without a change to the history.
    agent, calls, ordinary, client, history = make_agent()
    ordinary["messages"][1].update(field)
    try:
        ordinary_turn(agent, ordinary, history)
        if accepted:
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
            assert len(calls) == 2
        else:
            with pytest.raises(PrefixRequestError, match="source_transform_unsupported"):
                PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
            assert len(calls) == 1
    finally:
        client.close()


def test_a_changed_tool_call_type_is_refused():
    # The same keys with another type value: the provider reads another kind of call than the stored one.
    from agent.prefix_request import _no_extra_fields
    want = {"role": "assistant", "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}
    changed = copy.deepcopy(want)
    changed["tool_calls"][0]["type"] = "custom"
    assert _no_extra_fields(want, want)
    assert not _no_extra_fields(changed, want)


def test_execution_middleware_growth_is_checked_before_the_provider_call(monkeypatch):
    def grow(request=None, next_call=None, **context):
        request["messages"][-2]["content"] = "large text " * 20_000
        return next_call()

    agent, calls, ordinary, client, history = make_agent()
    agent.context_compressor.context_length = 10_000
    try:
        ordinary_turn(agent, ordinary, history)
        _execution_middleware(monkeypatch, grow)
        with pytest.raises(PrefixRequestError, match="^capacity$"):
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert len(calls) == 1
    finally:
        client.close()


def test_execution_middleware_cannot_send_after_the_attempt_returns(monkeypatch):
    from hermes_cli.middleware import _DownstreamExecutionError

    stored = []

    def defer(request=None, next_call=None, **context):
        stored.append(next_call)
        return None

    agent, calls, ordinary, client, history = make_agent()
    try:
        ordinary_turn(agent, ordinary, history)
        _execution_middleware(monkeypatch, defer)
        with pytest.raises(PrefixRequestError, match="^incomplete_response$"):
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        # The middleware callback wraps errors until an active chain can unwrap them.
        with pytest.raises(_DownstreamExecutionError) as caught:
            stored[0]()
        assert isinstance(caught.value.original, PrefixRequestError)
        assert str(caught.value.original) == "attempt_finished"
        assert len(calls) == 1
    finally:
        client.close()


def test_execution_middleware_can_wait_for_a_worker_callback(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    def worker(request=None, next_call=None, **context):
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(next_call).result()

    agent, calls, ordinary, client, history = make_agent()
    try:
        ordinary_turn(agent, ordinary, history)
        _execution_middleware(monkeypatch, worker)
        result = PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert result["content"] == "Synthetic handoff."
        assert len(calls) == 2
    finally:
        client.close()


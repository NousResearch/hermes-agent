"""Keep the warm request on the final provider body and the same Relay policy."""

import copy
import json
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI

from agent.prefix_request import PrefixRequest, PrefixRequestError, begin_capture, publish_response
from agent.prefix_request_capture import bind_stream_capture, main_capture_scope, open_main_chat_stream
from agent.turn_api_call import perform_api_call
from tests.agent.test_prefix_request import REPLY, make_agent, ordinary_turn
from tests.agent.test_relay_llm import relay_turn as _relay_turn_fixture


@pytest.fixture
def relay_turn(tmp_path, monkeypatch):
    yield from _relay_turn_fixture.__wrapped__(tmp_path, monkeypatch)


def _turn(agent, ordinary):
    agent.platform = "cli"
    agent._has_pending_redirect = lambda: False
    return perform_api_call(
        agent, api_kwargs=ordinary, _original_api_kwargs=ordinary, _llm_middleware_trace=[],
        _moa_prepared_request=None, _retry=SimpleNamespace(), thinking_spinner=None,
        retry_count=0, api_call_count=0, api_request_id="synthetic", effective_task_id="task-1",
        turn_id="turn-1", interrupted=False).response


@pytest.mark.parametrize("change_setting", [False, True])
def test_managed_main_request_keeps_the_final_setting(relay_turn, change_setting):
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request

    relay, _ = relay_turn
    agent, calls, ordinary, client, history = make_agent()
    agent.session_id = "session-1"
    agent._disable_streaming = True
    agent._interruptible_api_call = lambda request: _dispatch_nonstreaming_api_request(
        agent, request, make_client=lambda *a, **k: client)
    ordinary["temperature"] = 0.75
    seen = []

    def rewrite(name, request, annotated):
        seen.append(name)
        if change_setting:
            annotated.params = {**(annotated.params or {}), "temperature": 0.25}
        return relay.LLMRequestInterceptOutcome(request, annotated)

    relay.intercepts.register_llm_request("test.warm.settings", 1, False, rewrite)
    try:
        response = _turn(agent, ordinary)
        history.append({"role": "assistant", "content": response.choices[0].message.content})
        result = PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert result["content"] == "Synthetic handoff."
        assert len(calls) == 2 and len(seen) == 2
        assert calls[1]["temperature"] == calls[0]["temperature"] == (0.25 if change_setting else 0.75)
        assert "extra_headers" not in calls[0] and "extra_headers" not in calls[1]
    finally:
        relay.intercepts.deregister_llm_request("test.warm.settings")
        client.close()


def test_a_relay_rewrite_of_the_captured_prefix_is_refused(relay_turn):
    relay, _ = relay_turn
    agent, calls, ordinary, client, history = make_agent()
    agent.session_id = "session-1"
    try:
        ordinary_turn(agent, ordinary, history)

        def rewrite(name, request, annotated):
            rows = copy.deepcopy(annotated.messages)
            rows[0]["content"] = "A changed system prompt."
            annotated.messages = rows
            return relay.LLMRequestInterceptOutcome(request, annotated)

        relay.intercepts.register_llm_request("test.warm.prefix", 1, False, rewrite)
        try:
            with pytest.raises(PrefixRequestError, match="^middleware_rewrite$"):
                PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
            assert len(calls) == 1
        finally:
            relay.intercepts.deregister_llm_request("test.warm.prefix")
    finally:
        client.close()


def test_rewritten_main_history_cannot_replace_the_stored_history(relay_turn):
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request

    relay, _ = relay_turn
    agent, calls, ordinary, client, history = make_agent()
    agent.session_id = "session-1"
    agent._disable_streaming = True
    agent._interruptible_api_call = lambda request: _dispatch_nonstreaming_api_request(
        agent, request, make_client=lambda *a, **k: client)

    def rewrite(name, request, annotated):
        rows = copy.deepcopy(annotated.messages)
        rows[1]["content"] = "Text that is not in the stored history."
        annotated.messages = rows
        return relay.LLMRequestInterceptOutcome(request, annotated)

    relay.intercepts.register_llm_request("test.warm.main-history", 1, False, rewrite)
    try:
        response = _turn(agent, ordinary)
        history.append({"role": "assistant", "content": response.choices[0].message.content})
        with pytest.raises(PrefixRequestError, match="^source_transform_unsupported$"):
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert len(calls) == 1
    finally:
        relay.intercepts.deregister_llm_request("test.warm.main-history")
        client.close()


def test_relay_execution_cannot_send_a_warm_request_twice(relay_turn):
    relay, _ = relay_turn
    agent, calls, ordinary, client, history = make_agent()
    agent.session_id = "session-1"
    errors = []

    async def duplicate(name, request, context, next_call):
        first = await next_call(request)
        try:
            await next_call(request)
        except Exception as error:
            errors.append(error)
        return first

    try:
        ordinary_turn(agent, ordinary, history)
        relay.intercepts.register_llm_execution("test.warm.once", 1, duplicate)
        try:
            result = PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
            assert result["content"] == "Synthetic handoff."
            assert len(calls) == 2
            assert len(errors) == 1 and "already_used" in str(errors[0])
        finally:
            relay.intercepts.deregister_llm_execution("test.warm.once")
    finally:
        client.close()


@pytest.mark.parametrize("headers", [{"authorization": "synthetic"},
                                    {"traceparent": "synthetic-trace", "authorization": "synthetic"}])
def test_non_trace_request_headers_remain_unsupported(headers):
    agent, calls, ordinary, client, history = make_agent()
    try:
        begin_capture(agent, {**ordinary, "extra_headers": headers})
        assert agent._prefix_capture is None
        with pytest.raises(PrefixRequestError, match="^no_capture$"):
            PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
        assert calls == []
    finally:
        client.close()


def test_stream_capture_uses_its_own_physical_token(relay_turn):
    from agent import relay_llm

    relay, _ = relay_turn
    agent, calls, ordinary, client, history = make_agent()
    agent.session_id = "session-1"
    stores = []
    driver = SimpleNamespace(agent=agent, last_chunk_time={},
                             clients=SimpleNamespace(set_client=lambda value: value))
    agent._touch_activity = lambda text: None

    def rewrite(name, request, annotated):
        annotated.params = {**(annotated.params or {}), "temperature": 0.25}
        return relay.LLMRequestInterceptOutcome(request, annotated)

    relay.intercepts.register_llm_request("test.warm.stream", 1, False, rewrite)
    try:
        # Use the real Relay provider callback and SDK stream entry, with a completed response sink.
        with main_capture_scope(agent):
            for _ in range(2):
                store = {}
                stores.append(store)
                response = relay_llm.execute(
                    ordinary, lambda kwargs: open_main_chat_stream(driver, kwargs, store),
                    session_id="session-1", name="custom", model_name="synthetic",
                    metadata={"api_mode": "chat_completions"})
                store["response"] = response
                bind_stream_capture(agent, store, response)
        assert stores[0]["token"] is not stores[1]["token"]
        first = bind_stream_capture(agent, stores[0], stores[0]["response"])
        assert first._hermes_prefix_capture is stores[0]["token"]
        publish_response(agent, first)
        assert agent._prefix_capsule is None
        second = bind_stream_capture(agent, stores[1], stores[1]["response"])
        publish_response(agent, second)
        assert agent._prefix_capsule["body"]["temperature"] == 0.25
    finally:
        relay.intercepts.deregister_llm_request("test.warm.stream")
        client.close()


def test_capture_scope_is_off_when_the_engine_does_not_ask_for_it():
    from agent.prefix_request_capture import capture_main_request

    agent, calls, ordinary, client, history = make_agent()
    agent.context_compressor.wants_prefix_request = False
    try:
        with main_capture_scope(agent):
            assert capture_main_request(agent, ordinary) is None
        assert getattr(agent, "_prefix_capture", None) is None
        assert getattr(agent, "_prefix_capsule", None) is None
        assert calls == []
    finally:
        client.close()


@pytest.mark.parametrize("warm_enabled", [True, False])
def test_managed_stream_driver_captures_the_final_sdk_body(relay_turn, warm_enabled):
    from run_agent import AIAgent

    relay, _ = relay_turn
    template, _, ordinary, unused_client, history = make_agent()
    unused_client.close()
    calls, headers = [], []

    def sink(request):
        body = json.loads(request.content)
        calls.append(body)
        headers.append(dict(request.headers))
        message = {"role": "assistant", "content": REPLY if body.get("stream") else "Synthetic handoff."}
        usage = {"prompt_tokens": 5000, "completion_tokens": 5, "total_tokens": 5005}
        if body.get("stream"):
            chunk = {"id": "synthetic-stream", "object": "chat.completion.chunk", "created": 0,
                     "model": "synthetic", "choices": [{"index": 0, "delta": message, "finish_reason": "stop"}],
                     "usage": usage}
            return httpx.Response(200, request=request, headers={"content-type": "text/event-stream"},
                                  content="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n")
        return httpx.Response(200, request=request, json={
            "id": "synthetic", "object": "chat.completion", "created": 0, "model": "synthetic",
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}], "usage": usage})

    client = OpenAI(api_key="synthetic-no-secret", base_url=template.base_url, max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(sink), trust_env=False))
    agent = AIAgent(api_key="synthetic-no-secret", base_url=template.base_url, model="synthetic", provider="custom",
                    quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=[], max_iterations=1,
                    session_id="session-1")
    agent.client = client
    agent._client_kwargs = template._client_kwargs
    agent.tools = template.tools
    agent._cached_system_prompt = template._cached_system_prompt
    agent._prefix_source_messages = agent._session_messages = history
    agent.context_compressor.warm_handoff = "on" if warm_enabled else "off"
    agent.context_compressor.context_length = 65536
    agent._create_request_openai_client = lambda **kwargs: client
    agent._close_request_openai_client = lambda *args, **kwargs: None
    ordinary["temperature"] = 0.75

    def rewrite(name, request, annotated):
        annotated.params = {**(annotated.params or {}), "temperature": 0.25}
        return relay.LLMRequestInterceptOutcome(request, annotated)

    relay.intercepts.register_llm_request("test.warm.real-stream", 1, False, rewrite)
    try:
        response = _turn(agent, ordinary)
        assert response.choices[0].message.content == REPLY
        history.append({"role": "assistant", "content": REPLY})
        if warm_enabled:
            result = PrefixRequest(agent, history)("Write the handoff.", timeout_s=30)
            assert result["content"] == "Synthetic handoff."
            assert len(calls) == 2
            assert calls[1]["temperature"] == calls[0]["temperature"] == 0.25
            assert calls[1]["messages"][:len(calls[0]["messages"])] == calls[0]["messages"]
            assert headers[0]["traceparent"] != headers[1]["traceparent"]
            assert "extra_headers" not in calls[1]
        else:
            assert getattr(agent, "_prefix_capture", None) is None
            assert getattr(agent, "_prefix_capsule", None) is None
            assert len(calls) == 1
    finally:
        relay.intercepts.deregister_llm_request("test.warm.real-stream")
        client.close()

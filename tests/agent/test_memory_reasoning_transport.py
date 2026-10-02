"""Restricted requests cross the actual SDK/HTTP boundary once per reservation."""

import json
import threading
import time

import httpx
from openai import OpenAI
import pytest

from agent.memory_reasoning import CoreSingleAttemptTransport, PriceQuote, Route


class Agent:
    def __init__(self, mode, handler, provider="openai"):
        self.api_mode = mode
        self.provider = provider
        self.parent = object()
        self.parent_closed = False
        self.sends = 0
        self.aborts = 0
        self.releases = 0

        def counted(request):
            self.sends += 1
            return handler(request)

        self.http = httpx.Client(transport=httpx.MockTransport(counted))
        self.client = OpenAI(api_key="unit-test", base_url="https://example.test/v1",
                             http_client=self.http, max_retries=0)

    def _create_request_openai_client(self, **kwargs):
        assert self.client.max_retries == 0
        return self.client

    def _abort_request_openai_client(self, client, *, reason):
        assert client is self.client
        self.aborts += 1

    def _close_request_openai_client(self, client, *, reason):
        assert client is self.client
        self.releases += 1

    def _is_codex_backend(self):
        return False


def transport(*, cancelled=lambda: False, deadline=None, price=True, provider="openai"):
    return CoreSingleAttemptTransport(
        input_bound=lambda request, route: 100,
        price=PriceQuote("2026-10-test", provider, "gpt-5", 2, 8) if price else None,
        cancelled=cancelled, deadline_monotonic=deadline or time.monotonic() + 5,
    )


def test_chat_sdk_one_http_send_effort_usage_price_and_parent_preserved():
    def handler(request):
        body = json.loads(request.content)
        assert request.url.path.endswith("/chat/completions")
        assert body["reasoning_effort"] == "high"
        return httpx.Response(200, json={"id": "chatcmpl-1", "object": "chat.completion",
            "created": 1, "model": "gpt-5", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}})

    agent = Agent("chat_completions", handler)
    route = Route("openai", "gpt-5", "high")
    wire = {"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "high"}
    single = transport()
    try:
        assert single.effective_effort(wire, route) == "high"
        response = single.complete(wire, agent)
        assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (11, 3)
        assert single.actual_cost(response, route) == pytest.approx((11 * 2 + 3 * 8) / 1_000_000)
        assert agent.sends == agent.releases == 1
        assert agent.aborts == 0 and not agent.parent_closed
    finally:
        agent.client.close()


def test_codex_responses_sdk_stream_core_parser_one_send():
    def handler(request):
        body = json.loads(request.content)
        assert request.url.path.endswith("/responses")
        assert body["stream"] is True
        assert body["reasoning"]["effort"] == "high"
        events = [
            {"type": "response.output_item.done", "output_index": 0, "item": {
                "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": "ok"}]}},
            {"type": "response.completed", "response": {"id": "resp_1", "status": "completed",
                "usage": {"input_tokens": 12, "output_tokens": 4, "total_tokens": 16}}},
        ]
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text="".join("data: " + json.dumps(e) + "\n\n" for e in events))

    agent = Agent("codex_responses", handler, provider="openai-codex")
    route = Route("openai-codex", "gpt-5", "high", api_mode="codex_responses")
    wire = {"model": "gpt-5", "instructions": "inspect", "input": [{"role": "user", "content": "hi"}],
            "reasoning": {"effort": "high"}, "store": False}
    single = transport(provider="openai-codex")
    try:
        assert single.effective_effort(wire, route) == "high"
        response = single.complete(wire, agent)
        assert response.output[0].content[0].text == "ok"
        assert (response.usage.input_tokens, response.usage.output_tokens) == (12, 4)
        assert single.actual_cost(response, route) == pytest.approx((12 * 2 + 4 * 8) / 1_000_000)
        assert agent.sends == agent.releases == 1
    finally:
        agent.client.close()


@pytest.mark.parametrize("mode", ["chat_completions", "codex_responses"])
def test_failed_http_send_has_no_hidden_retry(mode):
    agent = Agent(mode, lambda request: httpx.Response(503, json={"error": {"message": "unavailable"}}))
    wire = ({"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]}
            if mode == "chat_completions" else
            {"model": "gpt-5", "input": [{"role": "user", "content": "hi"}], "store": False})
    try:
        with pytest.raises(Exception):
            transport().complete(wire, agent)
        assert agent.sends == agent.releases == 1
    finally:
        agent.client.close()


def test_missing_versioned_price_fails_closed():
    route = Route("openai", "gpt-5", "high")
    assert transport(price=False).cost_upper_bound(100, 20, route) is None
    assert transport().cost_upper_bound(100, 20, Route("openai", "other", "high")) is None
    single = transport()
    single.effective_effort({"reasoning_effort": "high", "service_tier": "priority"}, route)
    assert single.cost_upper_bound(100, 20, route) is None


@pytest.mark.parametrize("stop", ["cancel", "deadline"])
def test_cancel_and_deadline_abort_only_request_client(stop):
    entered, release = threading.Event(), threading.Event()
    cancelled = threading.Event()

    def handler(request):
        entered.set()
        release.wait(2)
        raise httpx.ReadError("socket retired")

    agent = Agent("chat_completions", handler)
    original_abort = agent._abort_request_openai_client

    def abort(client, *, reason):
        original_abort(client, reason=reason)
        release.set()

    agent._abort_request_openai_client = abort
    deadline = time.monotonic() + (0.15 if stop == "deadline" else 5)
    single = transport(cancelled=cancelled.is_set, deadline=deadline)
    result = []
    worker = threading.Thread(target=lambda: result.append(_capture(lambda: single.complete(
        {"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]}, agent))))
    try:
        worker.start()
        assert entered.wait(1)
        if stop == "cancel":
            cancelled.set()
        worker.join(2)
        assert not worker.is_alive()
        assert isinstance(result[0], Exception)
        assert agent.aborts == agent.releases == agent.sends == 1
        assert not agent.parent_closed
    finally:
        release.set()
        agent.client.close()


def _capture(fn):
    try:
        return fn()
    except Exception as exc:
        return exc

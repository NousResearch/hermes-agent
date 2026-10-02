"""Quota cascades retain conversation state and isolate benches between profiles.

The HTTP transport boundary is simulated; agents, provider resolution, tool
execution, error classification, and turn recovery use their production paths.
The free route's fixture model is explicitly configured, never catalogued as free.
"""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from email.utils import format_datetime
import json
from pathlib import Path
import time

import httpx
import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from agent.secret_scope import reset_secret_scope, set_multiplex_active, set_secret_scope
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from run_agent import AIAgent


@contextmanager
def _profile(home):
    home_token = set_hermes_home_override(home)
    secret_token = set_secret_scope({}, profile_home=str(home))
    try:
        yield
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    # Even optional auth discovery must stay in the disposable home.
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    homes = [tmp_path / name for name in ("a", "b")]
    chain = [
        {"provider": "subscription-two", "model": "second-model"},
        {"provider": "opencode-zen-free", "model": "configured-tool-model"},
    ]
    for home in homes:
        home.mkdir()
        (home / "config.yaml").write_text(json.dumps({
            "providers": {"subscription-two": {
                "base_url": "https://second.test/v1", "api_key": "fixture",
                "api_mode": "chat_completions"},
                "opencode-zen-free": {"base_url": "https://free.test/v1",
                    "api_key": "no-key-required", "transport": "chat_completions",
                    "extra_headers": {"Authorization": ""}}},
            "fallback_providers": chain,
            "model": {"context_length": 65536, "streaming": False},
            "tools": {"tool_search": {"enabled": "off"}},
            "compression": {"enabled": False},
            "agent": {"reasoning_effort": "xhigh", "auto_recovery_cycles": 0, "api_max_retries": 3,
                      "reasoning_overrides": {"configured-tool-model": False}},
        }))
    requests, notices = [], []
    scripts = {}

    def respond(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": []})
        if not request.url.path.endswith("/chat/completions"):
            return httpx.Response(404, json={"error": "No metadata in fixture"})
        assert request.url.host in {"primary.test", "primary-other.test", "second.test", "free.test"}, request.url
        if request.url.host == "free.test":
            assert request.headers.get("Authorization") == ""
        body = json.loads(request.content)
        requests.append((request.url.host, deepcopy(body)))
        queue = scripts.get(request.url.host, [])
        item = queue.pop(0) if queue else "Answered successfully."
        if isinstance(item, tuple):
            status, payload, headers = item
            return httpx.Response(status, json={"error": payload}, headers=headers)
        message = ({"role": "assistant", "content": None, "tool_calls": [{
            "id": "write-once", "type": "function",
            "function": {"name": "quota_fixture_write", "arguments": "{}"},
        }], "reasoning": "fixture private reasoning"} if item == "tool" else
                   {"role": "assistant", "content": item})
        return httpx.Response(200, json={
            "id": "fixture", "object": "chat.completion", "created": 1,
            "model": body["model"], "choices": [{"index": 0, "message": message,
            "finish_reason": "tool_calls" if item == "tool" else "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda self, req: transport.handle_request(req))
    # Register a real tool and drive its real dispatcher; the receipt catches write replay.
    import toolsets
    from tools.registry import registry
    schema = {"name": "quota_fixture_write", "description": "Write a local test receipt.",
              "parameters": {"type": "object", "properties": {}}}
    receipt = tmp_path / "writes"

    def write(args, **kwargs):
        with receipt.open("a") as stream:
            stream.write("written\n")
        return json.dumps({"written": True})

    registry.register("quota_fixture_write", "quota_fixture", schema, write)
    monkeypatch.setitem(toolsets.TOOLSETS, "quota_fixture", {"tools": ["quota_fixture_write"]})
    agents = []

    def make(base_url="https://primary.test/v1"):
        agent = AIAgent(
            api_key="fixture", base_url=base_url, provider="subscription-one",
            model="primary-model", api_mode="chat_completions", fallback_model=deepcopy(chain),
            enabled_toolsets=["quota_fixture"], quiet_mode=True, skip_memory=True,
            skip_context_files=True, save_trajectories=False, max_iterations=8,
            status_callback=lambda kind, text: notices.append(str(text)),
        )
        agent._disable_streaming = True
        agent._auto_recovery_cycles = 0
        agents.append(agent)
        return agent

    set_multiplex_active(True)
    try:
        yield homes, make, scripts, requests, notices, receipt
    finally:
        set_multiplex_active(False)
        for agent in agents:
            agent.client.close()


def _error(code, message, *, status=429, headers=None, **fields):
    return status, {"code": code, "message": message, **fields}, headers or {}


@pytest.mark.parametrize("reset_source", ["absolute", "retry-after", "http-date"])
def test_quota_cascade_shared_benches_reset_recovery_and_write_once(runtime, monkeypatch, reset_source):
    homes, make, scripts, requests, notices, receipt = runtime
    clock = [time.time()]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    start = clock[0]
    original_monotonic = time.monotonic
    monkeypatch.setattr(time, "monotonic", lambda: original_monotonic() + clock[0] - start)
    primary_error = _error("usage_limit_reached", "Subscription usage limit reached", reset_at=start + 120)
    second_error = _error("insufficient_quota", "Subscription quota exhausted", reset_at=start + 60)
    if reset_source != "absolute":
        second_error = _error("insufficient_quota", "Subscription quota exhausted", headers={
            "Retry-After": "60" if reset_source == "retry-after" else format_datetime(
                datetime.fromtimestamp(start + 60, timezone.utc), usegmt=True)})
    scripts.update({"primary.test": ["tool", primary_error], "second.test": [second_error]})
    with _profile(homes[0]):
        agent = make()
        primary = deepcopy(agent._primary_runtime)
        tools = deepcopy(agent.tools)
        first = agent.run_conversation("Write the receipt once, then answer.", system_message="Keep these instructions stable.")
        assert first.get("final_response") == "Answered successfully.", first
        assert [host for host, _ in requests] == ["primary.test", "primary.test", "second.test", "free.test"]
        assert receipt.read_text() == "written\n"
        assert agent.tools == tools and agent._primary_runtime == primary
        after_write = [body for _, body in requests[1:]]
        # Wire sanitization may remove private reasoning but keeps the conversation/tools.
        for body in after_write:
            assert body["tools"] == after_write[0]["tools"]
            assert [(m["role"], m.get("content"), m.get("tool_calls"), m.get("tool_call_id"))
                    for m in body["messages"]] == [
                        (m["role"], m.get("content"), m.get("tool_calls"), m.get("tool_call_id"))
                        for m in after_write[0]["messages"]]
        assert "reasoning_effort" not in after_write[-1]
        assert all("reasoning" not in m and "reasoning_content" not in m for m in after_write[-1]["messages"])
        emitted = len(notices)
        before = len(requests)
        second = agent.run_conversation("Continue.", conversation_history=first["messages"])
        assert second.get("final_response") == "Answered successfully."
        assert [host for host, _ in requests[before:]] == ["free.test"]
        assert len(notices) == emitted  # no repeat downgrade on every user message
        before = len(requests)
        fresh = make()
        assert fresh.run_conversation("Another conversation.").get("final_response") == "Answered successfully."
        assert [host for host, _ in requests[before:]] == ["free.test"]
    with _profile(homes[1]):
        before = len(requests)
        assert make().run_conversation("Profile B.").get("final_response") == "Answered successfully."
        assert [host for host, _ in requests[before:]] == ["primary.test"]
    with _profile(homes[0]):
        before = len(requests)
        assert make().run_conversation("Profile A again.").get("final_response") == "Answered successfully."
        assert [host for host, _ in requests[before:]] == ["free.test"]
        before = len(requests)
        assert make("https://primary-other.test/v1").run_conversation("Independent endpoint.").get("final_response") == "Answered successfully."
        assert [host for host, _ in requests[before:]] == ["primary-other.test"]
        clock[0] = start + 61
        before = len(requests)
        assert agent.run_conversation("Secondary window reopened.").get("final_response") == "Answered successfully."
        assert [host for host, _ in requests[before:]] == ["second.test"]
        clock[0] = start + 121
        before = len(requests)
        notices.clear()
        # A failed primary probe must not announce recovery; its new quota extends the bench.
        scripts["primary.test"] = [_error("insufficient_quota", "Quota exhausted", reset_at=start + 180)]
        assert agent.run_conversation("Try the primary window.").get("final_response") == "Answered successfully."
        assert [host for host, _ in requests[before:]] == ["primary.test", "second.test"]
        assert not any("Primary model restored" in text for text in notices)
        clock[0] = start + 181
        notices.clear()
        # At the wire boundary no successful response has arrived yet.
        seen = []
        original = httpx.HTTPTransport.handle_request
        def verify_notice(self, req):
            seen.append(list(notices))
            return original(self, req)
        monkeypatch.setattr(httpx.HTTPTransport, "handle_request", verify_notice)
        assert agent.run_conversation("Recover.").get("final_response") == "Answered successfully."
        assert not any("Primary model restored" in text for texts in seen for text in texts)
        assert sum("Primary model restored" in text for text in notices) == 1
        assert receipt.read_text() == "written\n"
        for host in ("primary.test", "second.test", "free.test"):
            scripts[host] = [_error("insufficient_quota", "Quota exhausted", reset_at=start + 260)]
        before = len(requests)
        assert make().run_conversation("Every allowance is empty.").get("error")
        assert [host for host, _ in requests[before:]] == ["primary.test", "second.test", "free.test"]
        before = len(requests)
        assert make().run_conversation("A peer must not probe the empty chain.").get("error")
        assert len(requests) == before


@pytest.mark.parametrize("error,reason,shared", [
    (_error("terminal_quota_exhausted", "Quota exhausted"), FailoverReason.billing, True),
    (_error("usage_limit_reached", "Subscription usage limit reached; try again in 2 minutes", resets_in_seconds=120), FailoverReason.rate_limit, True),
    (_error("payment_required", "Payment required", status=402), FailoverReason.billing, True),
    (_error("billing_error", "Out of extra usage", status=400), FailoverReason.billing, False),
    (_error("rate_limit_exceeded", "Too many requests", headers={"Retry-After": "30"}), FailoverReason.rate_limit, False),
    (_error("rate_limit_exceeded", "Requests per minute exceeded"), FailoverReason.rate_limit, False),
    (_error("invalid_api_key", "Invalid API key", status=401), FailoverReason.auth, False),
    (_error("context_length_exceeded", "Maximum context length exceeded", status=400), FailoverReason.context_overflow, False),
    (_error("invalid_request_error", "Unsupported parameter: fixture", status=400), FailoverReason.format_error, False),
])
def test_only_exhaustion_benches_peers_and_nonquota_errors_keep_their_recovery(runtime, error, reason, shared):
    homes, make, scripts, requests, notices, _receipt = runtime
    status, body, headers = error
    response = httpx.Response(status, json={"error": body}, headers=headers,
                              request=httpx.Request("POST", "https://primary.test/v1/chat/completions"))
    from openai import APIStatusError
    exception = APIStatusError(body["message"], response=response, body={"error": body})
    assert classify_api_error(exception, provider="subscription-one", model="primary-model").reason == reason
    scripts["primary.test"] = [error]
    with _profile(homes[0]):
        agent = make()
        result = agent.run_conversation("Answer.")
        if reason == FailoverReason.context_overflow:
            assert [host for host, _ in requests] == ["primary.test"]
            assert result.get("error")
        else:
            assert result.get("final_response") == "Answered successfully.", result
            assert [host for host, _ in requests] == ["primary.test", "second.test"]
        before = len(requests)
        assert make().run_conversation("Peer.").get("final_response") == "Answered successfully."
        assert [host for host, _ in requests[before:]] == (["second.test"] if shared else ["primary.test"])

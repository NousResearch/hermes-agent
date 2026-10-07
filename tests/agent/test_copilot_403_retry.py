"""Copilot's opt-in 403 budget is independent of generic retries and preserves the payload."""

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pytest
from openai import OpenAI

from run_agent import AIAgent


@contextmanager
def _peer(statuses, error_message="Forbidden", response_overrides=None):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            status = statuses[min(len(requests) - 1, len(statuses) - 1)]
            body = ({"error": {"message": error_message, "type": "permission_error"}}
                    if status != 200 else {
                        "id": "local-completion", "object": "chat.completion", "created": 0,
                        "model": "test-model", "choices": [{"index": 0, "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Recovered"}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
                    })
            if response_overrides and len(requests) - 1 in response_overrides:
                body.update(response_overrides[len(requests) - 1])
            streaming = status == 200 and requests[-1].get("stream")
            if streaming:
                body["object"] = "chat.completion.chunk"
                for choice in body.get("choices") or []:
                    if "message" in choice:
                        choice["delta"] = choice.pop("message")
                payload = f"data: {json.dumps(body)}\n\ndata: [DONE]\n\n".encode()
            else:
                payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/event-stream" if streaming else "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _agent(monkeypatch, limit, url, provider):
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    option = "" if limit is None else f"  copilot_403_max_retries: {limit}\n"
    (home / "config.yaml").write_text(
        "model:\n  context_length: 128000\nagent:\n  api_max_retries: 1\n  auto_recovery_cycles: 5\n"
        "  environment_probe: false\n" + option + "compression:\n  enabled: false\n",
        encoding="utf-8",
    )
    # Initialize the real client on loopback without invoking Copilot credential resolution.
    agent = AIAgent(api_key="local-test-key", base_url=url, provider="custom", model="test-model",
                    enabled_toolsets=[], quiet_mode=True, skip_context_files=True,
                    skip_memory=True, save_trajectories=False)
    agent.provider = provider
    agent._disable_streaming = True
    monkeypatch.setattr("agent.retry_utils.jittered_backoff", lambda *a, **kw: 0)
    return agent


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("provider,limit,statuses,expected", [
    ("copilot", None, [403], 1),
    ("copilot", 0, [403], 1),
    ("copilot", 2, [403], 3),
    ("github-copilot", 2, [403, 403, 200], 3),
    ("github", 1, [403, 200], 2),
    ("custom", 2, [403], 1),
    ("copilot", 2, [401], 1),
    ("copilot", -1, [403], 1),
    ("copilot", "invalid", [403], 1),
    ("copilot", True, [403], 1),
    ("copilot", 1.5, [403], 1),
    ("GitHub-Copilot", 1, [403, 200], 2),
])
def test_copilot_403_request_budget(monkeypatch, provider, limit, statuses, expected, streaming):
    with _peer(statuses) as (url, requests):
        agent = _agent(monkeypatch, limit, url, provider)
        agent._disable_streaming = not streaming
        with patch.object(agent, "_try_refresh_copilot_client_credentials", return_value=False) as refresh:
            result = agent.run_conversation("Keep this prompt unchanged", system_message="Stable system")
        assert len(requests) == expected
        assert all(request == requests[0] for request in requests)
        assert result["completed"] is (statuses[-1] == 200)
        if statuses[0] == 403:
            refresh.assert_not_called()
        elif provider == "copilot":
            refresh.assert_called_once()
        agent.client.close()


@pytest.mark.parametrize("statuses,api_limit,counts,budgets,streaming", [
    (statuses, api_limit, counts, budgets, streaming)
    for statuses, api_limit, counts, budgets in [
        ([403, 403, 200], 1, [0, 1, 2], [3, 3, 3]),
        ([403, 403, 403], 1, [0, 1, 2], [3, 3, 3]),
        ([500, 403, 403, 200], 2, [0, 1, 1, 2], [3, 2, 3, 3]),
    ]
    for streaming in ([False, True] if api_limit == 1 else [False])
])
def test_copilot_403_attempt_observability(monkeypatch, caplog, statuses, api_limit, counts, budgets, streaming):
    from agent import relay_llm
    from agent.turn_api_error import handle_api_error

    hooks, relays, generic_budgets = [], [], []
    original_execute = relay_llm.execute
    original_stream = relay_llm.stream

    def observe_stream(*args, **kwargs):
        relays.append(dict(kwargs["metadata"]))
        return original_stream(*args, **kwargs)

    def observe_relay(*args, **kwargs):
        relays.append(dict(kwargs["metadata"]))
        return original_execute(*args, **kwargs)

    def observe_error(agent, **kwargs):
        verdict = handle_api_error(agent, **kwargs)
        generic_budgets.append((verdict.retry_count, verdict.max_retries))
        return verdict

    with _peer(statuses) as (url, requests):
        agent = _agent(monkeypatch, 2, url, "copilot")
        agent._api_max_retries = api_limit
        agent._disable_streaming = not streaming
        monkeypatch.setattr("hermes_cli.lifecycle.has_hook",
                            lambda name: name in {"pre_api_request", "api_request_error"})
        def observe_hook(name, **kwargs):
            hooks.append((name, kwargs))
            return []
        monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", observe_hook)
        # Keep the phase's real signature: the loop discovers named locals from it.
        from functools import wraps
        monkeypatch.setattr("agent.conversation_loop.handle_api_error", wraps(handle_api_error)(observe_error))
        monkeypatch.setattr(relay_llm, "execute", observe_relay)
        monkeypatch.setattr(relay_llm, "stream", observe_stream)
        result = agent.run_conversation("Unchanged prompt", system_message="Stable system")
        agent.client.close()

    pre = [data for name, data in hooks if name == "pre_api_request"]
    errors = [data for name, data in hooks if name == "api_request_error"]
    failed = 2 if statuses[-1] == 200 else 3
    assert len(requests) == len(statuses)
    assert requests == [requests[0]] * len(requests)
    assert result["completed"] is (statuses[-1] == 200)
    assert generic_budgets == ([(0, 1)] * failed if api_limit == 1 else [(1, 2)] * 3)
    # Like generic hooks, retry_count is zero-based (prior retries); max_retries
    # is the total attempt budget. Log lines use the one-based failed attempt.
    assert [(item["retry_count"], item["max_retries"]) for item in pre] == list(zip(counts, budgets))
    assert [(item["retry_count"], item["max_retries"]) for item in relays] == list(zip(counts, budgets))
    if api_limit == 2:
        assert (errors[0]["retry_count"], errors[0]["max_retries"], errors[0]["retryable"]) == (0, 2, True)
        errors = errors[1:]
    assert [(item["retry_count"], item["max_retries"], item["retryable"], item["reason"])
            for item in errors] == [(i, 3, i < 2, "auth") for i in range(failed)]
    lines = [record.getMessage() for record in caplog.records
             if record.getMessage().startswith("API call failed (")]
    if api_limit == 2:
        assert "attempt 1/2" in lines.pop(0)
    assert len(lines) == failed
    for attempt, line in enumerate(lines, 1):
        assert f"attempt {attempt}/3" in line
        assert ("not retryable" in line) is (attempt == 3)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("statuses,expected_requests,expected_errors", [
    ([403, 500, 500, 200], [(0, 3), (1, 3), (1, 4), (2, 4)],
     [(0, 3, True), (0, 4, True), (1, 4, True)]),
    ([403, 500, 403, 500, 403], [(0, 3), (1, 3), (1, 4), (2, 3), (2, 4)],
     [(0, 3, True), (0, 4, True), (1, 3, True), (1, 4, True), (2, 3, False)]),
])
def test_copilot_403_reverse_mixed_cycle_metadata(
        monkeypatch, statuses, expected_requests, expected_errors, streaming):
    from agent import relay_llm

    hooks, relays = [], []
    original_execute, original_stream = relay_llm.execute, relay_llm.stream

    def observe_relay(original):
        def observed(*args, **kwargs):
            metadata = kwargs["metadata"]
            relays.append((metadata["retry_count"], metadata["max_retries"]))
            return original(*args, **kwargs)
        return observed

    def observe_hook(name, **kwargs):
        hooks.append((name, kwargs))
        return []

    monkeypatch.setattr("hermes_cli.lifecycle.has_hook",
                        lambda name: name in {"pre_api_request", "api_request_error"})
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", observe_hook)
    monkeypatch.setattr(relay_llm, "execute", observe_relay(original_execute))
    monkeypatch.setattr(relay_llm, "stream", observe_relay(original_stream))
    with _peer(statuses) as (url, requests):
        agent = _agent(monkeypatch, 2, url, "copilot")
        agent._api_max_retries = 4
        agent._disable_streaming = not streaming
        # Exercise outer retry scheduling, not the separate one-off 5xx unmask probe.
        import time
        agent._stream_5xx_probe_ts = time.monotonic()
        result = agent.run_conversation("Unchanged prompt", system_message="Stable system")
        agent.client.close()

    pre = [data for name, data in hooks if name == "pre_api_request"]
    errors = [data for name, data in hooks if name == "api_request_error"]
    assert len(requests) == len(statuses)
    assert requests == [requests[0]] * len(requests)
    assert result["completed"] is (statuses[-1] == 200)
    # Error hooks classify each failure's own policy; scheduling that policy must
    # govern the next request even after the independent Copilot counter is nonzero.
    assert [(item["retry_count"], item["max_retries"], item["retryable"])
            for item in errors] == expected_errors
    assert [(item["retry_count"], item["max_retries"]) for item in pre] == expected_requests
    assert relays == expected_requests


@pytest.mark.parametrize("response_kind,pool_recovery,streaming", [
    ("empty", False, False), ("empty", True, False),
    ("malformed", False, False), ("malformed", True, False),
    ("truncated", False, False), ("truncated", False, True),
])
def test_copilot_403_invalid_response_schedules_generic_metadata(
        monkeypatch, response_kind, pool_recovery, streaming):
    from types import SimpleNamespace
    from agent import relay_llm
    from agent.error_classifier import FailoverReason

    hooks, relays = [], []
    original_execute, original_stream = relay_llm.execute, relay_llm.stream

    def observe_relay(original):
        def observed(*args, **kwargs):
            metadata = kwargs["metadata"]
            relays.append((metadata["retry_count"], metadata["max_retries"]))
            return original(*args, **kwargs)
        return observed

    def observe_hook(name, **kwargs):
        hooks.append((name, kwargs))
        return []

    monkeypatch.setattr("hermes_cli.lifecycle.has_hook",
                        lambda name: name in {"pre_api_request", "api_request_error"})
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", observe_hook)
    monkeypatch.setattr(relay_llm, "execute", observe_relay(original_execute))
    monkeypatch.setattr(relay_llm, "stream", observe_relay(original_stream))
    override = {"choices": [] if response_kind == "empty" else None}
    if response_kind == "truncated":
        override = {"choices": [{"index": 0, "finish_reason": "length", "message": {
            "role": "assistant", "content": None, "tool_calls": [{
                "index": 0, "id": "local-tool", "type": "function",
                "function": {"name": "terminal", "arguments": '{"command":'},
            }],
        }}]}
    with _peer([403, 200, 200], response_overrides={1: override}) as (url, requests):
        agent = _agent(monkeypatch, 2, url, "copilot")
        agent._api_max_retries = 4
        agent._disable_streaming = not streaming
        # Empty SSE choices raise EmptyStreamError inside the transport instead of
        # reaching response-shape validation; truncation reaches the response scheduler.
        if pool_recovery:
            # Copilot does not report Codex soft failures; isolate the shared helper's
            # early pool exit while keeping the HTTP client and response validation real.
            monkeypatch.setattr("agent.turn_recovery.classify_codex_soft_failure",
                                lambda *args: (SimpleNamespace(reason=FailoverReason.rate_limit,
                                                               billing_unverified=False), {}))
        with patch.object(agent, "_recover_with_credential_pool",
                          return_value=(pool_recovery, False)) as pool:
            result = agent.run_conversation("Unchanged prompt", system_message="Stable system")
        agent.client.close()

    pre = [data for name, data in hooks if name == "pre_api_request"]
    errors = [data for name, data in hooks if name == "api_request_error"]
    expected = [(0, 3), (1, 3), (0 if pool_recovery or response_kind == "truncated" else 1, 4)]
    assert len(requests) == 3
    assert all(request["messages"] == requests[0]["messages"] for request in requests)
    if response_kind != "truncated":
        assert requests == [requests[0]] * 3
    assert result["completed"] is True
    assert result["final_response"] == "Recovered"
    expected_errors = [(0, 3, "auth")]
    if response_kind != "truncated":
        expected_errors.append((0, 4, "invalid_response"))
    assert [(item["retry_count"], item["max_retries"], item["reason"])
            for item in errors] == expected_errors
    if pool_recovery:
        pool.assert_called_once()
    else:
        pool.assert_not_called()
    assert [(item["retry_count"], item["max_retries"]) for item in pre] == expected
    assert relays == expected


@pytest.mark.parametrize("error_message", ["Forbidden", "insufficient credits", "content policy violation"])
@pytest.mark.parametrize("mode", ["disabled", "exhausted", "fallback", "interrupt", "redirect", "interrupt-exhausted"])
def test_copilot_403_exit_contract(monkeypatch, mode, error_message):
    with _peer([403, 200] if mode == "redirect" else [403], error_message) as (url, requests), \
            _peer([200]) as (fallback_url, fallback_requests):
        agent = _agent(monkeypatch, 0 if mode == "disabled" else 2, url, "copilot")
        if mode == "fallback":
            agent._fallback_chain = [{"provider": "custom", "model": "test-model",
                                      "base_url": fallback_url}]
        if mode in {"interrupt", "redirect"}:
            monkeypatch.setattr("agent.retry_utils.jittered_backoff", lambda *a, **kw: 1)
            def steer_or_stop(_seconds):
                if mode == "redirect":
                    with agent._pending_redirect_lock:
                        agent._pending_redirect = "Use the corrected prompt"
                        agent._interrupt_requested = True
                else:
                    agent.interrupt()
            monkeypatch.setattr("agent.turn_recovery.time.sleep", steer_or_stop)
        if mode == "interrupt-exhausted":
            original_hook = agent._invoke_api_request_error_hook
            def interrupt_last_attempt(**kwargs):
                original_hook(**kwargs)
                if len(requests) == 3:
                    agent.interrupt()
            monkeypatch.setattr(agent, "_invoke_api_request_error_hook", interrupt_last_attempt)
        fallback_client = OpenAI(api_key="local-fallback-key", base_url=fallback_url)
        with patch("agent.model_metadata.get_model_context_length", return_value=128000), \
                patch("agent.auxiliary_client.resolve_provider_client",
                   return_value=(fallback_client, "test-model")) as resolve, \
                patch.object(agent, "_recover_with_credential_pool",
                             return_value=(True, False)) as pool, \
                patch.object(agent, "_try_refresh_copilot_client_credentials", return_value=True) as refresh:
            # Disabled keeps the existing recovery path; do not simulate a successful pool
            # rotation there (the opt-in budget is not supposed to govern old behavior).
            if mode == "disabled":
                pool.return_value = (False, False)
            result = agent.run_conversation("Original prompt", system_message="Stable system")
        if mode == "fallback":
            assert len(requests) == 3
            assert len(fallback_requests) == 1
            assert all(request == requests[0] for request in requests)
            assert requests[0]["messages"][1:] == fallback_requests[0]["messages"][1:]
            assert result["final_response"] == "Recovered"
            resolve.assert_called_once()
        elif mode in {"interrupt", "interrupt-exhausted"}:
            assert len(requests) == (3 if mode == "interrupt-exhausted" else 1)
            assert result["interrupted"] is True
        elif mode == "redirect":
            assert len(requests) == 2
            assert requests[-1]["messages"][-1]["content"].endswith("Use the corrected prompt")
            assert result["completed"] is True
        else:
            assert len(requests) == (1 if mode == "disabled" else 3)
            assert "hermes config set agent.copilot_403_max_retries 3" in result["final_response"]
            assert "restart" in result["final_response"].lower()
            assert "entitlement" in result["final_response"].lower()
        if mode != "disabled":
            pool.assert_not_called()
        refresh.assert_not_called()
        agent.client.close()
        fallback_client.close()

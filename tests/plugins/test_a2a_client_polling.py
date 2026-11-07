"""Client-side tests for immediate return and polling via a2a_call."""

from __future__ import annotations

import pytest

from plugins.platforms.a2a import tools


class TestA2aClientPolling:
    def test_a2a_call_rejects_both_message_and_task_id(self):
        res = tools.a2a_call({"agent": "peer", "message": "msg", "task_id": "t-1"})
        assert "Error: cannot provide both 'message' and 'task_id'" in res

    def test_a2a_call_polls_to_completion(self, monkeypatch):
        calls = []

        def fake_fetch_card(base_url, headers, timeout):
            return {"url": base_url, "supportedInterfaces": [{"protocolBinding": "JSONRPC", "url": base_url + "/rpc"}]}

        def fake_http_post_json(url, body, headers, timeout):
            calls.append(body)
            method = body["method"]
            if method == "SendMessage":
                return {
                    "result": {
                        "task": {
                            "id": "task-xyz",
                            "contextId": "ctx-1",
                            "status": {"state": "TASK_STATE_WORKING"},
                        }
                    }
                }
            if method == "GetTask":
                return {
                    "result": {
                        "task": {
                            "id": "task-xyz",
                            "contextId": "ctx-1",
                            "status": {"state": "TASK_STATE_COMPLETED"},
                            "artifacts": [{"artifactId": "a1", "parts": [{"text": "polled result"}]}],
                        }
                    }
                }
            raise AssertionError(f"unexpected method {method}")

        monkeypatch.setattr(tools, "_fetch_card", fake_fetch_card)
        monkeypatch.setattr(tools, "_http_post_json", fake_http_post_json)
        monkeypatch.setattr(tools, "_configured_peers", lambda: {"peer-vps": {"url": "http://peer.internal"}})

        # Fast forward sleep
        monkeypatch.setattr(tools, "_sleep", lambda s: None)

        out = tools.a2a_call({"agent": "peer-vps", "message": "compute something"})
        assert "[peer-vps · context ctx-1 · task task-xyz · completed]" in out
        assert "polled result" in out
        assert len(calls) == 2

    def test_a2a_orchestrate_with_return_immediately_peer(self, monkeypatch):
        monkeypatch.setattr(tools, "_configured_peers", lambda: {
            "p1": {"url": "http://p1.internal", "capabilities": ["test"]},
        })
        # Simulate a2a_call answering working first then completed on poll
        calls = 0

        def fake_call(args):
            nonlocal calls
            calls += 1
            if calls == 1:
                return "[p1 · context ctx-1 · task t-1 · working]\nThe task is currently in progress"
            return "[p1 · context ctx-1 · task t-1 · completed]\nfinal result"

        monkeypatch.setattr(tools, "a2a_call", fake_call)
        res = tools.a2a_orchestrate({"capability": "test", "message": "ping", "mode": "all"})
        assert "final result" not in res  # first call was working, so not treated as success

    def test_a2a_call_stops_polling_on_nonretryable_http_error(self, monkeypatch):
        calls = []
        clock = {"now": 0.0}

        def fake_fetch_card(base_url, headers, timeout):
            return {"url": base_url, "supportedInterfaces": [{"protocolBinding": "JSONRPC", "url": base_url + "/rpc"}]}

        def fake_http_post_json(url, body, headers, timeout):
            calls.append(body["method"])
            if body["method"] == "SendMessage":
                return {"result": {"task": {"id": "task-xyz", "contextId": "ctx-1",
                                               "status": {"state": "TASK_STATE_WORKING"}}}}
            raise tools.urllib.error.HTTPError(url, 400, "bad request", {}, None)

        monkeypatch.setattr(tools, "_fetch_card", fake_fetch_card)
        monkeypatch.setattr(tools, "_http_post_json", fake_http_post_json)
        monkeypatch.setattr(tools, "_configured_peers", lambda: {
            "peer-vps": {"url": "http://peer.internal", "timeout": 1},
        })
        monkeypatch.setattr(tools, "_monotonic", lambda: clock["now"])
        monkeypatch.setattr(tools, "_sleep", lambda seconds: clock.__setitem__("now", clock["now"] + seconds))

        out = tools.a2a_call({"agent": "peer-vps", "message": "continue task"})

        assert out.startswith("[peer-vps · context ctx-1 · task task-xyz · unknown]")
        assert "HTTP 400" in out
        assert "Do not resubmit it" in out
        assert "task_id='task-xyz'" in out
        assert calls == ["SendMessage", "GetTask"]

    def test_a2a_call_retries_transient_http_error(self, monkeypatch):
        calls = []

        def fake_fetch_card(base_url, headers, timeout):
            return {"url": base_url, "supportedInterfaces": [{"protocolBinding": "JSONRPC", "url": base_url + "/rpc"}]}

        def fake_http_post_json(url, body, headers, timeout):
            calls.append(body["method"])
            if body["method"] == "SendMessage":
                return {"result": {"task": {"id": "task-xyz", "contextId": "ctx-1",
                                               "status": {"state": "TASK_STATE_WORKING"}}}}
            if len(calls) == 2:
                raise tools.urllib.error.HTTPError(url, 503, "temporarily unavailable", {}, None)
            return {"result": {"task": {"id": "task-xyz", "contextId": "ctx-1",
                                           "status": {"state": "TASK_STATE_COMPLETED"},
                                           "artifacts": [{"parts": [{"text": "done"}]}]}}}

        monkeypatch.setattr(tools, "_fetch_card", fake_fetch_card)
        monkeypatch.setattr(tools, "_http_post_json", fake_http_post_json)
        monkeypatch.setattr(tools, "_configured_peers", lambda: {
            "peer-vps": {"url": "http://peer.internal", "timeout": 10},
        })
        monkeypatch.setattr(tools, "_sleep", lambda seconds: None)

        out = tools.a2a_call({"agent": "peer-vps", "message": "continue task"})

        assert "[peer-vps · context ctx-1 · task task-xyz · completed]" in out
        assert "done" in out
        assert calls == ["SendMessage", "GetTask", "GetTask"]

    def test_a2a_call_preserves_task_id_when_direct_poll_fails(self, monkeypatch):
        def fake_fetch_card(base_url, headers, timeout):
            return {"url": base_url, "supportedInterfaces": [{"protocolBinding": "JSONRPC", "url": base_url + "/rpc"}]}

        def fake_http_post_json(url, body, headers, timeout):
            raise tools.urllib.error.HTTPError(url, 404, "not found", {}, None)

        monkeypatch.setattr(tools, "_fetch_card", fake_fetch_card)
        monkeypatch.setattr(tools, "_http_post_json", fake_http_post_json)
        monkeypatch.setattr(tools, "_configured_peers", lambda: {
            "peer-vps": {"url": "http://peer.internal", "timeout": 10},
        })

        out = tools.a2a_call({"agent": "peer-vps", "task_id": "task-xyz", "context_id": "ctx-1"})

        assert out.startswith("[peer-vps · context ctx-1 · task task-xyz · unknown]")
        assert "HTTP 404" in out
        assert "task_id='task-xyz'" in out

    def test_a2a_call_preserves_task_id_on_unexpected_poll_error(self, monkeypatch):
        def fake_fetch_card(base_url, headers, timeout):
            return {"url": base_url, "supportedInterfaces": [{"protocolBinding": "JSONRPC", "url": base_url + "/rpc"}]}

        def fake_http_post_json(url, body, headers, timeout):
            if body["method"] == "SendMessage":
                return {"result": {"task": {"id": "task-xyz", "contextId": "ctx-1",
                                               "status": {"state": "TASK_STATE_WORKING"}}}}
            raise RuntimeError("poll implementation error")

        monkeypatch.setattr(tools, "_fetch_card", fake_fetch_card)
        monkeypatch.setattr(tools, "_http_post_json", fake_http_post_json)
        monkeypatch.setattr(tools, "_configured_peers", lambda: {
            "peer-vps": {"url": "http://peer.internal", "timeout": 10},
        })
        monkeypatch.setattr(tools, "_sleep", lambda seconds: None)

        out = tools.a2a_call({"agent": "peer-vps", "message": "continue task"})

        assert out.startswith("[peer-vps · context ctx-1 · task task-xyz · unknown]")
        assert "Do not resubmit it" in out
        assert "task_id='task-xyz'" in out

    def test_a2a_call_receipt_persists_correctly_on_final_poll(self, monkeypatch):
        calls = []

        def fake_fetch_card(base_url, headers, timeout):
            return {"url": base_url, "supportedInterfaces": [{"protocolBinding": "JSONRPC", "url": base_url + "/rpc"}]}

        def fake_http_post_json(url, body, headers, timeout):
            calls.append(body["method"])
            if body["method"] == "SendMessage":
                return {"result": {"task": {"id": "task-receipt", "contextId": "ctx-receipt",
                                               "status": {"state": "TASK_STATE_WORKING"}}}}
            return {"result": {"task": {"id": "task-receipt", "contextId": "ctx-receipt",
                                           "status": {"state": "TASK_STATE_COMPLETED"},
                                           "artifacts": [{"parts": [{"text": "receipt finalized"}]}]}}}

        monkeypatch.setattr(tools, "_fetch_card", fake_fetch_card)
        monkeypatch.setattr(tools, "_http_post_json", fake_http_post_json)
        monkeypatch.setattr(tools, "_configured_peers", lambda: {
            "peer-vps": {"url": "http://peer.internal", "timeout": 10},
        })
        monkeypatch.setattr(tools, "_sleep", lambda seconds: None)

        persisted = []
        monkeypatch.setattr(tools.protocol, "persist_message", lambda ctx, role, text, tid: persisted.append((ctx, role, text, tid)))

        out = tools.a2a_call({"agent": "peer-vps", "message": "generate receipt"})

        assert "[peer-vps · context ctx-receipt · task task-receipt · completed]" in out
        assert "receipt finalized" in out
        assert any(p[1] == "agent" and p[2] == "receipt finalized" and p[3] == "task-receipt" for p in persisted)

    def test_a2a_orchestrate_best_mode_selects_successful_peer(self, monkeypatch):
        monkeypatch.setattr(tools, "_configured_peers", lambda: {
            "p1": {"url": "http://p1.internal", "capabilities": ["math"]},
            "p2": {"url": "http://p2.internal", "capabilities": ["math"]},
        })
        monkeypatch.setattr(tools, "_fetch_card", lambda url, h, t: {"url": url, "supportedInterfaces": [{"protocolBinding": "JSONRPC", "url": url + "/rpc"}]})

        def fake_post(url, body, h, t):
            ans = "short" if "p1" in url else "very long detailed correct answer"
            return {"result": {"task": {"id": "t-ok", "contextId": "ctx-ok", "status": {"state": "TASK_STATE_COMPLETED"},
                                           "artifacts": [{"parts": [{"text": ans}]}]}}}

        monkeypatch.setattr(tools, "_http_post_json", fake_post)
        res = tools.a2a_orchestrate({"capability": "math", "message": "solve", "mode": "best"})
        assert "[best: p2]" in res
        assert "very long detailed correct answer" in res

    def test_a2a_orchestrate_classifies_polling_error_as_failure(self, monkeypatch):
        monkeypatch.setattr(tools, "_configured_peers", lambda: {
            "p-error": {"url": "http://p-error.internal", "capabilities": ["calc"]},
        })
        monkeypatch.setattr(tools, "_fetch_card", lambda url, h, t: {"url": url, "supportedInterfaces": [{"protocolBinding": "JSONRPC", "url": url + "/rpc"}]})

        def fake_post(url, body, h, t):
            if body["method"] == "SendMessage":
                return {"result": {"task": {"id": "t-err", "contextId": "ctx-err", "status": {"state": "TASK_STATE_WORKING"}}}}
            raise tools.urllib.error.HTTPError(url, 500, "internal server error", {}, None)

        monkeypatch.setattr(tools, "_http_post_json", fake_post)
        monkeypatch.setattr(tools, "_sleep", lambda s: None)

        res = tools.a2a_orchestrate({"capability": "calc", "message": "sum", "mode": "best"})
        # Must be treated as failed peers, not a success:
        assert "All peers failed:" in res
        assert "p-error" in res

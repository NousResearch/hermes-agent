"""A2A long-job path: returnImmediately, GetTask poll, live-task watchdog."""

from __future__ import annotations

import time

import pytest
from concurrent.futures import Future

from plugins.platforms.a2a import protocol, tools
from plugins.platforms.a2a import adapter as a2a_adapter
from plugins.platforms.a2a.adapter import A2AAdapter


def _bare_adapter():
    from gateway.config import PlatformConfig
    return A2AAdapter(PlatformConfig(enabled=True))


def _peer_cfg():
    return {"a2a_agents": {"fpga": {"url": "http://fpga.local:9900"}}}


class TestReturnImmediatelyFlag:
    def test_v1_flag(self):
        assert A2AAdapter._config_return_immediately(
            {"configuration": {"returnImmediately": True}}
        ) is True

    def test_snake_case_flag(self):
        assert A2AAdapter._config_return_immediately(
            {"configuration": {"return_immediately": True}}
        ) is True

    def test_old_blocking_false(self):
        assert A2AAdapter._config_return_immediately(
            {"configuration": {"blocking": False}}
        ) is True

    def test_default_is_blocking(self):
        assert A2AAdapter._config_return_immediately({}) is False
        assert A2AAdapter._config_return_immediately(
            {"configuration": {"blocking": True}}
        ) is False
        assert A2AAdapter._config_return_immediately(
            {"configuration": {"returnImmediately": False}}
        ) is False


class TestWatchdogSkipsLiveTasks:
    def test_skip_keeps_running_task(self):
        store = protocol.TaskStore()
        store.create("live", "ctx", "peer")
        store.create("dead", "ctx", "peer")
        failed = store.fail_orphans(-1, exclude={"live"})
        assert failed == ["dead"]
        assert store.get("live")["state"] == protocol.STATE_SUBMITTED
        assert store.get("dead")["state"] == protocol.STATE_FAILED

    def test_no_skip_fails_old_task(self):
        store = protocol.TaskStore()
        store.create("alone", "ctx", "peer")
        failed = store.fail_orphans(-1)
        assert failed == ["alone"]
        assert store.get("alone")["state"] == protocol.STATE_FAILED

    def test_adapter_skips_pending_waiters(self):
        adapter = _bare_adapter()
        adapter.tasks.create("live", "ctx", "peer")
        adapter._add_pending("live", "ctx")
        try:
            failed = adapter.tasks.fail_orphans(-1, exclude=set(adapter._active_tasks))
            assert failed == []
            assert adapter.tasks.get("live")["state"] == protocol.STATE_SUBMITTED
        finally:
            adapter._pop_pending("live")


class TestInboundReturnImmediately:
    def _pending(self, adapter, task_id="t-imm", ctx="c-imm"):
        adapter.tasks.create(task_id, ctx, "alice")
        adapter.tasks.set_state(task_id, protocol.STATE_WORKING)
        fut = adapter._add_pending(task_id, ctx)
        pending = {
            "task_id": task_id,
            "context_id": ctx,
            "peer": "alice",
            "future": fut,
            "created_iso": protocol.now_iso(),
            "started": time.time(),
        }
        return pending, fut

    def test_send_returns_working_without_waiting(self):
        adapter = _bare_adapter()
        pending, fut = self._pending(adapter)
        adapter._prepare_task = lambda params, peer, agent=None: (None, pending)

        resp = adapter._rpc_message_send(
            1,
            {"configuration": {"returnImmediately": True}, "message": {}},
            "alice",
        )
        task = resp["result"]
        assert task["id"] == "t-imm"
        assert task["status"]["state"] == protocol.STATE_WORKING
        assert fut.done() is False
        assert "t-imm" in set(adapter._active_tasks)

        fut.set_result((protocol.STATE_COMPLETED, "bitfile ready"))
        deadline = time.time() + 2
        rec = adapter.tasks.get("t-imm")
        while rec["state"] != protocol.STATE_COMPLETED and time.time() < deadline:
            time.sleep(0.02)
            rec = adapter.tasks.get("t-imm")
        assert rec["state"] == protocol.STATE_COMPLETED
        assert rec["reply"] == "bitfile ready"
        assert adapter.tasks.get("t-imm") is not None

    def test_v1_send_wraps_working_task(self):
        adapter = _bare_adapter()
        pending, fut = self._pending(adapter, task_id="t-v1", ctx="c-v1")
        adapter._prepare_task = lambda params, peer, agent=None: (None, pending)
        try:
            resp = adapter._rpc_message_send(
                7,
                {"configuration": {"returnImmediately": True}, "message": {}},
                "alice",
                v1_response=True,
            )
            wrapped = resp["result"]
            assert "task" in wrapped
            assert wrapped["task"]["status"]["state"] == protocol.STATE_WORKING
            assert wrapped["task"]["id"] == "t-v1"
        finally:
            if not fut.done():
                fut.set_result((protocol.STATE_FAILED, "test teardown"))
            deadline = time.time() + 2
            while "t-v1" in set(adapter._active_tasks) and time.time() < deadline:
                time.sleep(0.02)

    def test_blocking_send_still_waits(self):
        adapter = _bare_adapter()
        pending, fut = self._pending(adapter, task_id="t-block", ctx="c-block")
        adapter._prepare_task = lambda params, peer, agent=None: (None, pending)
        awaited = []

        def fake_await(pending_arg, keepalive=None):
            awaited.append(True)
            return protocol.STATE_COMPLETED, "done now"

        adapter._await_reply = fake_await
        resp = adapter._rpc_message_send(1, {"message": {}}, "alice")
        assert awaited == [True]
        assert resp["result"]["status"]["state"] == protocol.STATE_COMPLETED
        assert protocol.extract_text(resp["result"]["artifacts"][0]) == "done now"

    def test_get_task_sees_working_then_completed(self):
        adapter = _bare_adapter()
        pending, fut = self._pending(adapter, task_id="t-poll", ctx="c-poll")
        adapter._prepare_task = lambda params, peer, agent=None: (None, pending)
        adapter._rpc_message_send(
            1,
            {"configuration": {"returnImmediately": True}, "message": {}},
            "alice",
        )
        got = adapter._rpc_tasks_get(2, {"id": "t-poll"})
        assert got["result"]["status"]["state"] == protocol.STATE_WORKING

        fut.set_result((protocol.STATE_COMPLETED, "synth ok"))
        deadline = time.time() + 2
        rec = adapter.tasks.get("t-poll")
        while rec["state"] != protocol.STATE_COMPLETED and time.time() < deadline:
            time.sleep(0.02)
            rec = adapter.tasks.get("t-poll")
        got = adapter._rpc_tasks_get(3, {"taskId": "t-poll"})
        assert got["result"]["status"]["state"] == protocol.STATE_COMPLETED
        assert protocol.extract_text(got["result"]["artifacts"][0]) == "synth ok"

    def test_already_closed_task_is_not_finalized_twice(self):
        adapter = _bare_adapter()
        pending, _fut = self._pending(adapter, task_id="t-dup", ctx="c-dup")
        adapter.tasks.complete("t-dup", protocol.STATE_CANCELED, "stopped")
        state, reply = adapter._finalize_task(
            pending, protocol.STATE_COMPLETED, "should ignore")
        assert state == protocol.STATE_CANCELED
        assert reply == "stopped"
        rec = adapter.tasks.get("t-dup")
        assert rec["state"] == protocol.STATE_CANCELED
        assert rec["reply"] == "stopped"

    def test_background_waiter_fails_after_ceiling(self, monkeypatch):
        monkeypatch.setattr(a2a_adapter, "_BACKGROUND_WAIT_SECONDS", 0.05)
        adapter = _bare_adapter()
        pending, fut = self._pending(adapter, task_id="t-ceil", ctx="c-ceil")
        adapter._wait_in_background(pending)
        deadline = time.time() + 2
        rec = adapter.tasks.get("t-ceil")
        while rec["state"] != protocol.STATE_FAILED and time.time() < deadline:
            time.sleep(0.02)
            rec = adapter.tasks.get("t-ceil")
        assert rec["state"] == protocol.STATE_FAILED
        assert "did not reply in time" in (rec.get("reply") or "")
        assert "t-ceil" not in set(adapter._active_tasks)
        fut.set_result((protocol.STATE_COMPLETED, "too late"))
        rec = adapter.tasks.get("t-ceil")
        assert rec["state"] == protocol.STATE_FAILED


class TestOutboundCallAndGetTask:
    def setup_method(self):
        tools._card_cache.clear()

    def test_call_sends_return_immediately_and_prints_task_id(self, monkeypatch):
        monkeypatch.setattr(tools, "_load_config", _peer_cfg)
        monkeypatch.setattr(tools, "_http_get_json", lambda url, h, t: None)
        captured = {}

        def fake_post(url, body, headers, timeout):
            captured["body"] = body
            ctx = body["params"]["message"]["contextId"]
            return protocol.jsonrpc_result(
                body["id"],
                {"task": protocol.build_task("task-99", ctx, protocol.STATE_WORKING)},
            )

        monkeypatch.setattr(tools, "_http_post_json", fake_post)
        out = tools.a2a_call({
            "agent": "fpga",
            "message": "run synth",
            "return_immediately": True,
        })
        assert captured["body"]["params"]["configuration"]["returnImmediately"] is True
        assert "task-99" in out
        assert "a2a_get_task" in out
        assert "working" in out

    def test_call_string_true_counts_as_flag(self, monkeypatch):
        monkeypatch.setattr(tools, "_load_config", _peer_cfg)
        monkeypatch.setattr(tools, "_http_get_json", lambda url, h, t: None)
        captured = {}

        def fake_post(url, body, headers, timeout):
            captured["flag"] = (body["params"].get("configuration") or {}).get(
                "returnImmediately")
            ctx = body["params"]["message"]["contextId"]
            return protocol.jsonrpc_result(
                body["id"],
                protocol.build_task("t1", ctx, protocol.STATE_WORKING),
            )

        monkeypatch.setattr(tools, "_http_post_json", fake_post)
        tools.a2a_call({
            "agent": "fpga",
            "message": "go",
            "return_immediately": "true",
        })
        assert captured["flag"] is True

    def test_default_call_does_not_set_the_flag(self, monkeypatch):
        monkeypatch.setattr(tools, "_load_config", _peer_cfg)
        monkeypatch.setattr(tools, "_http_get_json", lambda url, h, t: None)
        captured = {}

        def fake_post(url, body, headers, timeout):
            captured["params"] = body["params"]
            ctx = body["params"]["message"]["contextId"]
            return protocol.jsonrpc_result(
                body["id"],
                protocol.build_task("t1", ctx, protocol.STATE_COMPLETED, "ok"),
            )

        monkeypatch.setattr(tools, "_http_post_json", fake_post)
        out = tools.a2a_call({"agent": "fpga", "message": "ping"})
        assert "configuration" not in captured["params"]
        assert "ok" in out
        assert "completed" in out

    def test_get_task_polls_completed_reply(self, monkeypatch):
        monkeypatch.setattr(tools, "_load_config", _peer_cfg)
        monkeypatch.setattr(tools, "_http_get_json", lambda url, h, t: None)
        captured = {}

        def fake_post(url, body, headers, timeout):
            captured["body"] = body
            return protocol.jsonrpc_result(
                body["id"],
                protocol.build_task("task-99", "ctx-1", protocol.STATE_COMPLETED, "bitfile ready"),
            )

        monkeypatch.setattr(tools, "_http_post_json", fake_post)
        out = tools.a2a_get_task({"agent": "fpga", "task_id": "task-99"})
        assert captured["body"]["method"] == "GetTask"
        assert captured["body"]["params"]["id"] == "task-99"
        assert captured["body"]["params"]["taskId"] == "task-99"
        assert "bitfile ready" in out
        assert "completed" in out
        assert "task-99" in out

    def test_get_task_still_running(self, monkeypatch):
        monkeypatch.setattr(tools, "_load_config", _peer_cfg)
        monkeypatch.setattr(tools, "_http_get_json", lambda url, h, t: None)

        def fake_post(url, body, headers, timeout):
            return protocol.jsonrpc_result(
                body["id"],
                protocol.build_task("task-99", "ctx-1", protocol.STATE_WORKING),
            )

        monkeypatch.setattr(tools, "_http_post_json", fake_post)
        out = tools.a2a_get_task({"agent": "fpga", "task_id": "task-99"})
        assert "working" in out
        assert "a2a_get_task" in out

    def test_get_task_requires_args(self):
        assert "required" in tools.a2a_get_task({"agent": "", "task_id": "x"})
        assert "required" in tools.a2a_get_task({"agent": "fpga", "task_id": ""})

    def test_get_task_unknown_peer(self, monkeypatch):
        monkeypatch.setattr(tools, "_load_config", lambda: {"a2a_agents": {}})
        out = tools.a2a_get_task({"agent": "ghost", "task_id": "t1"})
        assert "unknown agent" in out

    def test_get_task_accepts_taskId_alias(self, monkeypatch):
        monkeypatch.setattr(tools, "_load_config", _peer_cfg)
        monkeypatch.setattr(tools, "_http_get_json", lambda url, h, t: None)
        captured = {}

        def fake_post(url, body, headers, timeout):
            captured["id"] = body["params"]["id"]
            return protocol.jsonrpc_result(
                body["id"],
                protocol.build_task("t-alt", "c", protocol.STATE_COMPLETED, "done"),
            )

        monkeypatch.setattr(tools, "_http_post_json", fake_post)
        out = tools.a2a_get_task({"agent": "fpga", "taskId": "t-alt"})
        assert captured["id"] == "t-alt"
        assert "done" in out

    def test_get_task_reuses_card_within_ttl(self, monkeypatch):
        monkeypatch.setattr(tools, "_load_config", _peer_cfg)
        gets = []

        def fake_get(url, h, t):
            gets.append(url)
            return {"name": "fpga", "url": "http://fpga.local:9900"}

        def fake_post(url, body, headers, timeout):
            return protocol.jsonrpc_result(
                body["id"],
                protocol.build_task("task-99", "ctx-1", protocol.STATE_WORKING),
            )

        monkeypatch.setattr(tools, "_http_get_json", fake_get)
        monkeypatch.setattr(tools, "_http_post_json", fake_post)
        tools.a2a_get_task({"agent": "fpga", "task_id": "task-99"})
        tools.a2a_get_task({"agent": "fpga", "task_id": "task-99"})
        assert len(gets) == 1

    def test_get_task_refetches_card_after_ttl(self, monkeypatch):
        monkeypatch.setattr(tools, "_load_config", _peer_cfg)
        monkeypatch.setattr(tools, "_CARD_CACHE_TTL", 0)
        gets = []

        def fake_get(url, h, t):
            gets.append(url)
            return {"name": "fpga", "url": "http://fpga.local:9900"}

        def fake_post(url, body, headers, timeout):
            return protocol.jsonrpc_result(
                body["id"],
                protocol.build_task("task-99", "ctx-1", protocol.STATE_WORKING),
            )

        monkeypatch.setattr(tools, "_http_get_json", fake_get)
        monkeypatch.setattr(tools, "_http_post_json", fake_post)
        tools.a2a_get_task({"agent": "fpga", "task_id": "task-99"})
        tools.a2a_get_task({"agent": "fpga", "task_id": "task-99"})
        assert len(gets) == 2

def test_peer_polling_records_each_outcome_once(tmp_path, monkeypatch):
    import json
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    reply = {"state": protocol.STATE_INPUT_REQUIRED, "text": "Which target?"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{}')

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            task = protocol.build_task("shared-id", "shared-context", reply["state"], reply["text"])
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(protocol.jsonrpc_result(body["id"], task)).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    initial = protocol.metrics.inbound_total
    try:
        assert "Which target?" in tools.a2a_call({"agent": url, "message": "Build"})
        with ThreadPoolExecutor(max_workers=4) as pool:
            outputs = list(pool.map(lambda _: tools.a2a_get_task({"agent": url, "task_id": "shared-id"}), range(8)))
        assert all("Which target?" in text for text in outputs)
        assert protocol.metrics.inbound_total == initial + 1
        assert "Which target?" in tools.a2a_call({"agent": url, "message": "Not sure", "context_id": "shared-context"})
        assert "Which target?" in tools.a2a_get_task({"agent": url, "task_id": "shared-id"})
        assert protocol.metrics.inbound_total == initial + 2
        reply.update(state=protocol.STATE_COMPLETED, text="Built target")
        for _ in range(3):
            assert "Built target" in tools.a2a_get_task({"agent": url, "task_id": "shared-id"})
        rows = [row for row in protocol.load_conversation("shared-context") if row["role"] == "agent"]
        assert [row["text"] for row in rows] == ["Which target?", "Which target?", "Built target"]
        assert protocol.metrics.inbound_total == initial + 3
        monkeypatch.setattr(tools, "_configured_peers", lambda: {"other": {"url": url, "tenant": "other"}})
        assert "Built target" in tools.a2a_get_task({"agent": "other", "task_id": "shared-id"})
        assert protocol.metrics.inbound_total == initial + 4
        assert len([row for row in protocol.load_conversation("shared-context") if row["role"] == "agent"]) == 4
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("routed", [False, True])
def test_background_completion_preserves_profile(tmp_path, monkeypatch, routed):
    import json
    import threading
    from hermes_constants import get_hermes_home, set_hermes_home_override, reset_hermes_home_override

    profile = tmp_path / "profile"
    process_home = tmp_path / "process"
    monkeypatch.setenv("HERMES_HOME", str(process_home))
    token = set_hermes_home_override(profile)
    release, done = threading.Event(), threading.Event()
    adapter = _bare_adapter()
    monkeypatch.setattr(adapter, "_send_push_notification", lambda *_args: done.set())
    future = None
    try:
        if routed:
            def forward(*_args):
                assert get_hermes_home() == profile
                assert release.wait(5)
                return "profile reply", protocol.STATE_COMPLETED
            monkeypatch.setattr(adapter, "_forward_to_profile", forward)
            agent = {"local": False, "profile": "other", "slug": "other", "tenant": "other"}
        else:
            pending, future = TestInboundReturnImmediately()._pending(adapter, "profile-task", "profile-context")
            monkeypatch.setattr(adapter, "_prepare_task", lambda *args, **kwargs: (None, pending))
            agent = None
        response = adapter._rpc_message_send(1, {
            "message": protocol.text_message(protocol.ROLE_USER, "work", context_id="profile-context"),
            "configuration": {"returnImmediately": True}}, "peer", agent=agent)
        assert response["result"]["status"]["state"] == protocol.STATE_WORKING
        if future:
            future.set_result((protocol.STATE_COMPLETED, "profile reply"))
        release.set()
        assert done.wait(5)
        rows = protocol.load_conversation("profile-context")
        assert rows[-1]["text"] == "profile reply"
        audit = [json.loads(line) for line in (profile / "a2a_audit.jsonl").read_text().splitlines()]
        assert audit[-1]["direction"] == "outbound"
        assert not (process_home / "a2a_audit.jsonl").exists()
        assert not (process_home / "a2a_conversations").exists()
        if routed:
            # Streaming uses ordinary preparation even if the request includes the async option.
            task, pending = adapter._prepare_task({
                "message": protocol.text_message(protocol.ROLE_USER, "stream", context_id="stream-context"),
                "configuration": {"returnImmediately": True}}, "peer", agent=agent)
            assert task["status"]["state"] == protocol.STATE_COMPLETED
            assert pending is None
    finally:
        release.set()
        reset_hermes_home_override(token)

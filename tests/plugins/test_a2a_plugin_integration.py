"""Tests for the A2A (Agent-to-Agent) platform plugin — protocol v1.0.

Covers security primitives (peer-token identity, injection filtering,
redaction), v1.0 protocol shapes (Agent Card, Task, Part, roles, error codes),
the client tools (with HTTP mocked), adapter RPC handlers driven directly
(no HTTP), and real end-to-end inbound round-trips against a live http.server
with a mocked agent handler.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import socket
import threading
import urllib.error
import urllib.request
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest

from plugins.platforms.a2a import protocol, security, tools


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port
def _bare_adapter():
    from plugins.platforms.a2a.adapter import A2AAdapter
    from gateway.config import PlatformConfig
    return A2AAdapter(PlatformConfig(enabled=True))

class TestTaskRpcHandlers:
    def test_tasks_get_unknown_uses_spec_error_code(self):
        adapter = _bare_adapter()
        resp = adapter._rpc_tasks_get(1, {"taskId": "ghost"})
        assert resp["error"]["code"] == protocol.ERR_TASK_NOT_FOUND

    def test_tasks_get_returns_completed_task(self):
        adapter = _bare_adapter()
        adapter.tasks.create("task-done", "ctx-d", "peer")
        adapter.tasks.complete("task-done", protocol.STATE_COMPLETED, "answer")
        resp = adapter._rpc_tasks_get(1, {"taskId": "task-done"})
        task = resp["result"]
        assert task["status"]["state"] == "TASK_STATE_COMPLETED"
        assert protocol.extract_text(task["artifacts"][0]) == "answer"

    def test_tasks_cancel_resets_turns_for_context(self):
        """Cancel must reset anti-loop turns for the task's CONTEXT (the old
        code passed the task_id into a context-keyed map — silent no-op)."""
        adapter = _bare_adapter()
        for _ in range(4):
            adapter._turns.track("ctx-loopy")
        adapter.tasks.create("task-c", "ctx-loopy", "peer")
        resp = adapter._rpc_tasks_cancel(1, {"taskId": "task-c"})
        assert resp["result"]["status"]["state"] == "TASK_STATE_CANCELED"
        # Turn counter went back to zero: next track() is turn 1.
        assert adapter._turns.track("ctx-loopy") == 1

    def test_cancel_terminal_task_not_cancelable(self):
        adapter = _bare_adapter()
        adapter.tasks.create("task-t", "ctx-t", "peer")
        adapter.tasks.complete("task-t", protocol.STATE_COMPLETED, "done")
        resp = adapter._rpc_tasks_cancel(1, {"taskId": "task-t"})
        assert resp["error"]["code"] == protocol.ERR_TASK_NOT_CANCELABLE

    def test_cancel_unknown_task(self):
        adapter = _bare_adapter()
        resp = adapter._rpc_tasks_cancel(1, {"taskId": "ghost"})
        assert resp["error"]["code"] == protocol.ERR_TASK_NOT_FOUND

    def test_tasks_list_filters_by_context(self):
        adapter = _bare_adapter()
        adapter.tasks.create("t1", "ctx-a", "p")
        adapter.tasks.create("t2", "ctx-b", "p")
        adapter.tasks.complete("t1", protocol.STATE_COMPLETED, "x")
        resp = adapter._rpc_tasks_list(1, {"contextId": "ctx-a"})
        tasks = resp["result"]["tasks"]
        assert [t["id"] for t in tasks] == ["t1"]

    def test_tasks_list_filters_by_status_and_paginates(self):
        adapter = _bare_adapter()
        for i in range(5):
            adapter.tasks.create(f"tl-{i}", "ctx-l", "p")
            adapter.tasks.complete(f"tl-{i}", protocol.STATE_COMPLETED, "x")
        resp = adapter._rpc_tasks_list(1, {
            "contextId": "ctx-l", "status": "TASK_STATE_COMPLETED", "pageSize": 2})
        result = resp["result"]
        assert len(result["tasks"]) == 2
        assert result["nextPageToken"] == "2"
        resp2 = adapter._rpc_tasks_list(1, {
            "contextId": "ctx-l", "status": "TASK_STATE_COMPLETED",
            "pageSize": 2, "pageToken": result["nextPageToken"]})
        assert len(resp2["result"]["tasks"]) == 2
        ids = {t["id"] for t in result["tasks"]} | {t["id"] for t in resp2["result"]["tasks"]}
        assert len(ids) == 4  # no overlap between pages

    def test_push_config_create_returns_config_id(self):
        adapter = _bare_adapter()
        adapter.tasks.create("task-p", "ctx-p", "peer")
        resp = adapter._rpc_push_config_create(1, {
            "taskId": "task-p",
            "pushNotificationConfig": {"url": "https://example.com/hook"},
        })
        cfg = resp["result"]
        assert cfg["configId"].startswith("cfg-")
        assert cfg["createdAt"]
        assert cfg["pushNotificationConfig"]["url"] == "https://example.com/hook"

    def test_push_config_create_unknown_task(self):
        adapter = _bare_adapter()
        resp = adapter._rpc_push_config_create(1, {
            "taskId": "ghost", "pushNotificationConfig": {"url": "https://x/h"}})
        assert resp["error"]["code"] == protocol.ERR_TASK_NOT_FOUND

    def test_push_config_create_requires_url(self):
        adapter = _bare_adapter()
        resp = adapter._rpc_push_config_create(1, {"taskId": "t"})
        assert resp["error"]["code"] == protocol.ERR_INVALID_PARAMS

    def test_push_config_get_returns_stored_config(self):
        """GetTaskPushNotificationConfig retrieves a config after create."""
        adapter = _bare_adapter()
        adapter.tasks.create("task-g", "ctx-g", "peer")
        adapter._rpc_push_config_create(1, {
            "taskId": "task-g",
            "pushNotificationConfig": {"url": "https://example.com/hook"},
        })
        resp = adapter._rpc_push_config_get(1, {"taskId": "task-g"})
        cfg = resp["result"]
        assert cfg["pushNotificationConfig"]["url"] == "https://example.com/hook"
        assert cfg["configId"].startswith("cfg-")

    def test_push_config_get_by_config_id(self):
        """Get with a specific configId returns the matching config."""
        adapter = _bare_adapter()
        adapter.tasks.create("task-g2", "ctx-g2", "peer")
        create_resp = adapter._rpc_push_config_create(1, {
            "taskId": "task-g2",
            "pushNotificationConfig": {"url": "https://example.com/hook"},
        })
        config_id = create_resp["result"]["configId"]
        resp = adapter._rpc_push_config_get(1, {"taskId": "task-g2", "id": config_id})
        assert resp["result"]["configId"] == config_id

    def test_push_config_get_wrong_config_id_returns_error(self):
        """Get with wrong configId returns not-found error."""
        adapter = _bare_adapter()
        adapter.tasks.create("task-g3", "ctx-g3", "peer")
        adapter._rpc_push_config_create(1, {
            "taskId": "task-g3",
            "pushNotificationConfig": {"url": "https://example.com/hook"},
        })
        resp = adapter._rpc_push_config_get(1, {"taskId": "task-g3", "id": "cfg-wrong"})
        assert resp["error"]["code"] == protocol.ERR_TASK_NOT_FOUND

    def test_push_config_get_unknown_task(self):
        """Get for non-existent task returns not-found."""
        adapter = _bare_adapter()
        resp = adapter._rpc_push_config_get(1, {"taskId": "ghost"})
        assert resp["error"]["code"] == protocol.ERR_TASK_NOT_FOUND

    def test_push_config_get_requires_task_id(self):
        """Get without taskId returns invalid-params."""
        adapter = _bare_adapter()
        resp = adapter._rpc_push_config_get(1, {})
        assert resp["error"]["code"] == protocol.ERR_INVALID_PARAMS

    def test_push_config_list_returns_configs(self):
        """ListTaskPushNotificationConfigs returns all configs for a task."""
        adapter = _bare_adapter()
        adapter.tasks.create("task-l", "ctx-l", "peer")
        adapter._rpc_push_config_create(1, {
            "taskId": "task-l",
            "pushNotificationConfig": {"url": "https://example.com/hook"},
        })
        resp = adapter._rpc_push_config_list(1, {"taskId": "task-l"})
        configs = resp["result"]["configs"]
        assert len(configs) == 1
        assert configs[0]["pushNotificationConfig"]["url"] == "https://example.com/hook"

    def test_push_config_list_empty_for_task_without_config(self):
        """List returns empty array for a task with no push config."""
        adapter = _bare_adapter()
        adapter.tasks.create("task-l2", "ctx-l2", "peer")
        resp = adapter._rpc_push_config_list(1, {"taskId": "task-l2"})
        assert resp["result"]["configs"] == []

    def test_push_config_delete_removes_config(self):
        """DeleteTaskPushNotificationConfig removes the push config."""
        adapter = _bare_adapter()
        adapter.tasks.create("task-d", "ctx-d", "peer")
        adapter._rpc_push_config_create(1, {
            "taskId": "task-d",
            "pushNotificationConfig": {"url": "https://example.com/hook"},
        })
        # Delete
        resp = adapter._rpc_push_config_delete(1, {"taskId": "task-d"})
        assert resp["result"]["deleted"] is True
        # Get now fails
        resp2 = adapter._rpc_push_config_get(1, {"taskId": "task-d"})
        assert resp2["error"]["code"] == protocol.ERR_TASK_NOT_FOUND

    def test_push_config_delete_unknown_task(self):
        """Delete for non-existent task returns not-found."""
        adapter = _bare_adapter()
        resp = adapter._rpc_push_config_delete(1, {"taskId": "ghost"})
        assert resp["error"]["code"] == protocol.ERR_TASK_NOT_FOUND

    def test_push_config_delete_by_config_id(self):
        """Delete with a specific configId only deletes the matching config."""
        adapter = _bare_adapter()
        adapter.tasks.create("task-d2", "ctx-d2", "peer")
        create_resp = adapter._rpc_push_config_create(1, {
            "taskId": "task-d2",
            "pushNotificationConfig": {"url": "https://example.com/hook"},
        })
        config_id = create_resp["result"]["configId"]
        resp = adapter._rpc_push_config_delete(1, {"taskId": "task-d2", "id": config_id})
        assert resp["result"]["deleted"] is True

    def test_push_config_delete_wrong_config_id(self):
        """Delete with wrong configId returns not-found."""
        adapter = _bare_adapter()
        adapter.tasks.create("task-d3", "ctx-d3", "peer")
        adapter._rpc_push_config_create(1, {
            "taskId": "task-d3",
            "pushNotificationConfig": {"url": "https://example.com/hook"},
        })
        resp = adapter._rpc_push_config_delete(1, {"taskId": "task-d3", "id": "cfg-wrong"})
        assert resp["error"]["code"] == protocol.ERR_TASK_NOT_FOUND


# --------------------------------------------------------------------------
# End-to-end inbound round-trip (real http.server + mocked agent)
# --------------------------------------------------------------------------

def _make_live_adapter(monkeypatch, reply_fn=None):
    """Create an adapter on a free port with a mocked agent handler.

    ``reply_fn(event) -> Optional[str]`` returns the agent's reply (None =
    never reply). Returns (adapter, base_url).
    """
    from plugins.platforms.a2a.adapter import A2AAdapter
    from gateway.config import PlatformConfig

    port = _free_port()
    monkeypatch.setenv("A2A_PORT", str(port))

    # A scoped secondary profile ignores the process env (#100382); pass the
    # port through config.extra so both construction paths bind the same port.
    adapter = A2AAdapter(PlatformConfig(enabled=True, extra={"port": port}))

    async def fake_handle_message(event):
        if reply_fn is None:
            reply = "ECHO: " + event.text
        else:
            reply = reply_fn(event)
        if reply is not None:
            await adapter.send(event.source.chat_id, reply, metadata={"notify": True})

    adapter.handle_message = fake_handle_message  # type: ignore
    adapter._message_handler = object()  # non-None so dispatch proceeds
    return adapter, f"http://127.0.0.1:{port}"


def _get_json(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())


def _post_json(url, body, headers=None):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **(headers or {})}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())


def _send_body(text, ctx="", extra_params=None):
    msg = protocol.text_message(protocol.ROLE_USER, text, context_id=ctx)
    params = {"message": msg}
    if extra_params:
        params.update(extra_params)
    return {"jsonrpc": "2.0", "id": "1", "method": "message/send", "params": params}


@pytest.mark.integration
class TestInboundRoundTrip:
    def test_live_server_card_and_message_send(self, monkeypatch):
        """Start the real adapter server, hit the Agent Card, then send a task
        and verify the mocked agent's reply comes back as a v1.0 Task."""
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
        adapter, base = _make_live_adapter(monkeypatch)

        async def run():
            assert await adapter.connect() is True

            card = await asyncio.to_thread(_get_json, base + "/.well-known/agent.json")
            assert card["name"]
            assert card["supportedInterfaces"][0]["protocolVersion"] == "1.0"
            assert "security" not in card  # localhost-only, no auth advertised

            resp = await asyncio.to_thread(_post_json, base + "/", _send_body("hello agent"))
            assert resp["id"] == "1"
            task = resp["result"]
            assert task["status"]["state"] == "TASK_STATE_COMPLETED"
            reply = protocol.extract_text(task["artifacts"][0])
            assert "ECHO:" in reply
            assert "hello agent" in reply  # framed text still contains the task

            # 3) tasks/get finds the COMPLETED task (task store, not popped)
            get_resp = await asyncio.to_thread(_post_json, base + "/", {
                "jsonrpc": "2.0", "id": "2", "method": "tasks/get",
                "params": {"taskId": task["id"]},
            })
            assert get_resp["result"]["status"]["state"] == "TASK_STATE_COMPLETED"
            assert protocol.extract_text(get_resp["result"]["artifacts"][0]) == reply

            # 4) tasks/list sees it too
            list_resp = await asyncio.to_thread(_post_json, base + "/", {
                "jsonrpc": "2.0", "id": "3", "method": "tasks/list",
                "params": {"contextId": task["contextId"]},
            })
            assert any(t["id"] == task["id"] for t in list_resp["result"]["tasks"])

            await adapter.disconnect()

        asyncio.run(run())

    def test_mixed_parts_delivered_to_agent(self, monkeypatch):
        """A message with text + file + data Parts delivers all content to the
        agent — file URLs and data JSON are rendered into the text stream."""
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)

        received = {}

        def reply_fn(event):
            received["text"] = event.text
            return "got it"

        adapter, base = _make_live_adapter(monkeypatch, reply_fn=reply_fn)

        async def run():
            assert await adapter.connect() is True
            msg = protocol.message_with_parts(
                protocol.ROLE_USER,
                [
                    protocol.text_part("Please process these:"),
                    protocol.file_part(url="https://example.com/report.pdf",
                                       filename="report.pdf", media_type="application/pdf"),
                    protocol.data_part({"title": "Q3", "pages": 42}, "application/json"),
                ],
                context_id="ctx-mixed",
            )
            resp = await asyncio.to_thread(_post_json, base + "/", {
                "jsonrpc": "2.0", "id": "1", "method": "message/send",
                "params": {"message": msg},
            })
            assert resp["result"]["status"]["state"] == "TASK_STATE_COMPLETED"
            # The agent received all three parts rendered into text
            assert "Please process these:" in received["text"]
            assert "https://example.com/report.pdf" in received["text"]
            assert "report.pdf" in received["text"]
            assert "Q3" in received["text"]
            assert "42" in received["text"]
            await adapter.disconnect()

        asyncio.run(run())

    def test_push_config_crud_over_http(self, monkeypatch):
        """Full push notification config CRUD over real HTTP."""
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
        adapter, base = _make_live_adapter(monkeypatch)

        async def run():
            assert await adapter.connect() is True
            # Create a task first by sending a message (will get a task id back)
            resp = await asyncio.to_thread(_post_json, base + "/",
                                            _send_body("hello", ctx="ctx-crud"))
            task_id = resp["result"]["id"]

            # CREATE
            r = await asyncio.to_thread(_post_json, base + "/", {
                "jsonrpc": "2.0", "id": "2", "method": "tasks/pushNotificationConfig/create",
                "params": {"taskId": task_id,
                           "pushNotificationConfig": {"url": "https://example.com/hook"}},
            })
            assert r["result"]["configId"].startswith("cfg-")
            assert r["result"]["pushNotificationConfig"]["url"] == "https://example.com/hook"
            config_id = r["result"]["configId"]

            # GET
            r = await asyncio.to_thread(_post_json, base + "/", {
                "jsonrpc": "2.0", "id": "3", "method": "tasks/pushNotificationConfig/get",
                "params": {"taskId": task_id},
            })
            assert r["result"]["configId"] == config_id

            # LIST
            r = await asyncio.to_thread(_post_json, base + "/", {
                "jsonrpc": "2.0", "id": "4", "method": "tasks/pushNotificationConfig/list",
                "params": {"taskId": task_id},
            })
            assert len(r["result"]["configs"]) == 1

            # DELETE
            r = await asyncio.to_thread(_post_json, base + "/", {
                "jsonrpc": "2.0", "id": "5", "method": "tasks/pushNotificationConfig/delete",
                "params": {"taskId": task_id},
            })
            assert r["result"]["deleted"] is True

            # GET after delete → not found
            r = await asyncio.to_thread(_post_json, base + "/", {
                "jsonrpc": "2.0", "id": "6", "method": "tasks/pushNotificationConfig/get",
                "params": {"taskId": task_id},
            })
            assert r["error"]["code"] == protocol.ERR_TASK_NOT_FOUND

            await adapter.disconnect()

        asyncio.run(run())

    def test_unknown_method_error(self, monkeypatch):
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
        adapter, base = _make_live_adapter(monkeypatch)

        async def run():
            assert await adapter.connect() is True
            resp = await asyncio.to_thread(_post_json, base + "/", {
                "jsonrpc": "2.0", "id": "9", "method": "bogus/method", "params": {}})
            assert resp["error"]["code"] == protocol.ERR_METHOD_NOT_FOUND
            await adapter.disconnect()

        asyncio.run(run())

    def test_input_required_state_reachable(self, monkeypatch):
        """An agent reply starting with [INPUT_REQUIRED] maps to the v1.0
        input-required state with the question in status.message."""
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
        adapter, base = _make_live_adapter(
            monkeypatch, reply_fn=lambda e: "[INPUT_REQUIRED] Which repository do you mean?")

        async def run():
            assert await adapter.connect() is True
            resp = await asyncio.to_thread(_post_json, base + "/", _send_body("review the code"))
            task = resp["result"]
            assert task["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
            question = protocol.extract_text(task["status"]["message"])
            assert "Which repository" in question
            assert "[INPUT_REQUIRED]" not in question
            assert "artifacts" not in task
            await adapter.disconnect()

        asyncio.run(run())

    def test_timeout_returns_failed_not_completed(self, monkeypatch):
        """When the agent never replies, the task must FAIL (and count as a
        failure), not report success."""
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
        monkeypatch.setenv("A2A_REPLY_TIMEOUT", "1")
        adapter, base = _make_live_adapter(monkeypatch, reply_fn=lambda e: None)

        async def run():
            assert await adapter.connect() is True
            failed_before = protocol.metrics.tasks_failed
            completed_before = protocol.metrics.tasks_completed
            resp = await asyncio.to_thread(_post_json, base + "/", _send_body("are you there"))
            task = resp["result"]
            assert task["status"]["state"] == "TASK_STATE_FAILED"
            assert protocol.metrics.tasks_failed == failed_before + 1
            assert protocol.metrics.tasks_completed == completed_before
            # The task store agrees.
            rec = adapter.tasks.get(task["id"])
            assert rec["state"] == "TASK_STATE_FAILED"
            await adapter.disconnect()

        asyncio.run(run())

    def test_connect_accepts_gateway_reconnect_kwarg(self, monkeypatch):
        """Gateway reconnection passes is_reconnect=... to every adapter connect()."""
        monkeypatch.setenv("A2A_BEARER_TOKEN", "topsecret")
        monkeypatch.setenv("A2A_HOST", "127.0.0.1")
        adapter, _base = _make_live_adapter(monkeypatch)

        async def run():
            assert await adapter.connect(is_reconnect=True) is True
            await adapter.disconnect()

        asyncio.run(run())

    def test_auth_required_when_token_set(self, monkeypatch):
        monkeypatch.setenv("A2A_BEARER_TOKEN", "topsecret")
        monkeypatch.setenv("A2A_HOST", "127.0.0.1")
        adapter, base = _make_live_adapter(monkeypatch)

        async def run():
            assert await adapter.connect() is True
            # Card should now advertise auth.
            card = await asyncio.to_thread(_get_json, base + "/.well-known/agent.json")
            assert card["security"] == [{"bearer": []}]

            # POST without auth → 401 with our custom (non-spec-reserved) code.
            def _post_unauth():
                try:
                    _post_json(base + "/", _send_body("x"))
                    raise AssertionError("expected 401")
                except urllib.error.HTTPError as e:
                    assert e.code == 401
                    return json.loads(e.read().decode())

            err = await asyncio.to_thread(_post_unauth)
            assert err["error"]["code"] == protocol.ERR_UNAUTHORIZED

            # POST with the token succeeds.
            resp = await asyncio.to_thread(
                _post_json, base + "/", _send_body("hello"),
                {"Authorization": "Bearer topsecret"})
            assert resp["result"]["status"]["state"] == "TASK_STATE_COMPLETED"

            await adapter.disconnect()

        asyncio.run(run())

    def test_peer_token_identity_used_for_framing(self, monkeypatch):
        """The authenticated peer-token name (not anything in the body) is the
        identity the agent sees in the privacy frame."""
        monkeypatch.setenv("A2A_PEER_TOKENS", "alice:tok-alice")
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        monkeypatch.setenv("A2A_HOST", "127.0.0.1")

        seen = {}

        def reply_fn(event):
            seen["text"] = event.text
            seen["user"] = event.source.user_id
            return "ok"

        adapter, base = _make_live_adapter(monkeypatch, reply_fn=reply_fn)

        async def run():
            assert await adapter.connect() is True
            body = _send_body("do a thing")
            # An attacker-controlled 'peer' field in params must be ignored.
            body["params"]["peer"] = "the-operator"
            resp = await asyncio.to_thread(
                _post_json, base + "/", body, {"Authorization": "Bearer tok-alice"})
            assert resp["result"]["status"]["state"] == "TASK_STATE_COMPLETED"
            assert seen["user"] == "alice"
            assert "'alice'" in seen["text"]
            assert "the-operator" not in seen["text"]
            await adapter.disconnect()

        asyncio.run(run())

    def test_multiplex_adapter_keeps_profile_scoped_peer_tokens(self, monkeypatch):
        """A secondary listener must not authenticate with the default profile's tokens."""
        from agent.secret_scope import (
            reset_secret_scope,
            set_multiplex_active,
            set_secret_scope,
        )

        monkeypatch.setenv("A2A_PEER_TOKENS", "default:default-token")
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        monkeypatch.setenv("A2A_HOST", "127.0.0.1")

        set_multiplex_active(True)
        scope_token = set_secret_scope(
            {"A2A_PEER_TOKENS": "secondary:secondary-token"}
        )
        try:
            adapter, base = _make_live_adapter(monkeypatch)
        finally:
            reset_secret_scope(scope_token)

        async def run():
            try:
                assert await adapter.connect() is True
                response = await asyncio.to_thread(
                    _post_json,
                    base + "/",
                    _send_body("profile-scoped auth"),
                    {"Authorization": "Bearer secondary-token"},
                )
                assert response["result"]["status"]["state"] == "TASK_STATE_COMPLETED"

                with pytest.raises(urllib.error.HTTPError) as exc_info:
                    await asyncio.to_thread(
                        _post_json,
                        base + "/",
                        _send_body("wrong profile"),
                        {"Authorization": "Bearer default-token"},
                    )
                assert exc_info.value.code == 401
            finally:
                await adapter.disconnect()

        try:
            asyncio.run(run())
        finally:
            set_multiplex_active(False)


# --------------------------------------------------------------------------
# Push notifications end-to-end (inline config in message/send)
# --------------------------------------------------------------------------

@pytest.mark.integration
class TestPushNotificationEndToEnd:
    def test_inline_push_config_delivers_stream_response(self, monkeypatch):
        """message/send carrying configuration.taskPushNotificationConfig gets
        a signed v1.0 StreamResponse POSTed to the callback on completion."""
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
        monkeypatch.setenv("A2A_PUSH_SECRET", "push-secret-1")
        monkeypatch.setenv("A2A_REPLY_TIMEOUT", "15")

        callbacks: list[tuple[dict, str]] = []
        cb_lock = threading.Lock()
        received_evt = threading.Event()

        class _Hook(BaseHTTPRequestHandler):
            def log_message(self, *a):  # noqa: A002
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length).decode())
                sig = self.headers.get("X-A2A-Signature", "")
                with cb_lock:
                    callbacks.append((body, sig))
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()
                received_evt.set()

        hook_port = _free_port()
        hook_server = HTTPServer(("127.0.0.1", hook_port), _Hook)
        hook_thread = threading.Thread(target=hook_server.serve_forever, daemon=True)
        hook_thread.start()
        hook_url = f"http://127.0.0.1:{hook_port}/hook"

        hold_started = threading.Event()
        allow_reply = threading.Event()

        from plugins.platforms.a2a.adapter import A2AAdapter
        from gateway.config import PlatformConfig

        port = _free_port()
        monkeypatch.setenv("A2A_PORT", str(port))
        adapter = A2AAdapter(PlatformConfig(enabled=True, extra={"port": port}))

        async def controlled_handle(event):
            hold_started.set()
            await asyncio.to_thread(lambda: allow_reply.wait(timeout=10))
            if not allow_reply.is_set():
                await adapter.send(event.source.chat_id, "ECHO: timeout", metadata={"notify": True})
                return
            await adapter.send(event.source.chat_id, "ECHO: " + event.text, metadata={"notify": True})

        adapter.handle_message = controlled_handle  # type: ignore
        adapter._message_handler = object()
        base = f"http://127.0.0.1:{port}"

        async def run():
            assert await adapter.connect() is True
            from plugins.platforms.a2a.a2a_persistence import _task_ledger_path
            from plugins.platforms.a2a.protocol import TaskStore
            ledger_path = _task_ledger_path()
            body = _send_body("ping with push", extra_params={
                "configuration": {
                    "taskPushNotificationConfig": {
                        "url": hook_url,
                    },
                },
            })
            req_task = asyncio.create_task(asyncio.to_thread(_post_json, base + "/", body))
            assert await asyncio.to_thread(lambda: hold_started.wait(timeout=5)), "agent handler never started — WORKING not held"
            for _ in range(30):
                await asyncio.sleep(0.1)
                if ledger_path.exists():
                    try:
                        data = json.loads(ledger_path.read_text())
                        for tid, rec in data.items():
                            if rec.get("push_url") == hook_url and rec.get("state") == protocol.STATE_WORKING:
                                break
                        else:
                            continue
                        break
                    except Exception:
                        continue
            else:
                assert False, "durable ledger WORKING record not found"
            data = json.loads(ledger_path.read_text())
            task_id = None
            config_id = None
            context_id = None
            rec_work = None
            for tid, rec in data.items():
                if rec.get("push_url") == hook_url:
                    task_id = tid
                    config_id = rec.get("push_config_id")
                    context_id = rec.get("context_id")
                    rec_work = rec
                    break
            assert task_id and config_id and context_id and rec_work
            assert rec_work["push_url"] == hook_url
            assert isinstance(config_id, str) and config_id.startswith("cfg-") and len(config_id) == 16
            assert rec_work["agent_slug"] == "" and rec_work["tenant"] == ""
            assert rec_work["state"] == protocol.STATE_WORKING
            fresh = TaskStore()
            cnt = fresh.restore(ledger_path)
            assert cnt >= 1
            got = fresh.get(task_id, "", "")
            assert got is not None and got["push_url"] == hook_url and got["push_config_id"] == config_id
            assert got["agent_slug"] == "" and got["tenant"] == ""
            assert fresh.get(task_id, "wrong-slug", "") is None
            assert fresh.get(task_id, "", "wrong-tenant") is None
            cfg = fresh.get_push_config(task_id, config_id, "", "")
            assert cfg is not None and cfg["pushNotificationConfig"]["url"] == hook_url
            assert fresh.get_push_config(task_id, config_id, "wrong", "") is None
            assert fresh.get_push_config(task_id, "cfg-000000000000", "", "") is None
            allow_reply.set()
            resp = await req_task
            task = resp["result"]
            assert task["status"]["state"] == "TASK_STATE_COMPLETED"
            assert task["id"] == task_id
            assert task["contextId"] == context_id
            data2 = json.loads(ledger_path.read_text())
            rec_done = data2.get(task_id)
            assert rec_done is not None
            assert rec_done["state"] == protocol.STATE_COMPLETED
            assert rec_done["push_url"] == hook_url
            assert rec_done["push_config_id"] == config_id
            assert rec_done["agent_slug"] == "" and rec_done["tenant"] == ""
            fresh2 = TaskStore()
            fresh2.restore(ledger_path)
            got2 = fresh2.get(task_id, "", "")
            assert got2 is not None
            assert got2["push_url"] == hook_url and got2["push_config_id"] == config_id
            assert got2["agent_slug"] == "" and got2["tenant"] == ""
            assert fresh2.get(task_id, "wrong", "") is None
            assert await asyncio.to_thread(lambda: received_evt.wait(timeout=5)), "push callback never received"
            await asyncio.sleep(0.3)
            with cb_lock:
                cbs = list(callbacks)
            assert len(cbs) == 1, f"expected exactly one callback, got {len(cbs)}"
            payload, signature = cbs[0]
            assert "statusUpdate" in payload
            su = payload["statusUpdate"]
            assert su["taskId"] == task_id
            assert su["contextId"] == context_id
            assert su["status"]["state"] == "TASK_STATE_COMPLETED"
            assert "ECHO:" in protocol.extract_text(su["status"]["message"])
            assert "ping with push" in protocol.extract_text(su["status"]["message"])
            expected = hmac.new(
                b"push-secret-1",
                json.dumps(payload, sort_keys=True, ensure_ascii=False).encode(),
                hashlib.sha256,
            ).hexdigest()
            assert signature == expected
            cand_dup = dict(got2)
            cand_dup["state"] = protocol.STATE_COMPLETED
            cand_dup["reply"] = got2.get("reply", "")
            cand_dup["completed_at"] = got2.get("completed_at") or __import__("time").time()
            outcome = fresh2.publish_durable(ledger_path, task_id, cand_dup)
            assert outcome.published and not outcome.newly_published
            await asyncio.sleep(0.5)
            with cb_lock:
                assert len(callbacks) == 1, "duplicate terminal publication produced second callback"
            await adapter.disconnect()

        try:
            asyncio.run(run())
        finally:
            hook_server.shutdown()
            hook_server.server_close()
            hook_thread.join(timeout=2)
            try:
                asyncio.run(adapter.disconnect())
            except Exception:
                pass
def test_agent_card_can_advertise_tenant():
    card = protocol.build_agent_card(
        name="tenant-agent",
        url="http://localhost:9900/research/",
        description="test",
        tenant="research",
    )
    assert card["supportedInterfaces"][0]["tenant"] == "research"


class TestMultiAgentRouting:
    def test_path_routed_agent_card_uses_prefix_and_canonical_path(self, monkeypatch):
        from plugins.platforms.a2a.adapter import A2AAdapter
        from gateway.config import PlatformConfig
        from tools.registry import registry

        # Pin the shared tool registry so the dynamic card skills are
        # deterministic: the card advertises registered ∩ configured
        # toolsets, and must not depend on ambient registrations left by
        # other tests importing tools.* modules (e.g. tools.web_tools
        # registers the 'web' toolset at import time).
        monkeypatch.setattr(registry, "get_registered_toolset_names",
                            lambda: ["web", "research"])
        monkeypatch.setattr(registry, "get_tool_names_for_toolset",
                            lambda ts: {"web": ["web_search"], "research": ["research_synthesize"]}[ts])

        adapter = A2AAdapter(PlatformConfig(enabled=True, extra={
            "agents": {
                "research": {
                    "profile": "research",
                    "name": "Research Agent",
                    "description": "Research specialist",
                    "capabilities": ["web", "research"],
                }
            }
        }))

        route = adapter._route_for_path("/research/.well-known/agent-card.json")
        assert route["agent"]["slug"] == "research"
        assert route["subpath"] == "/.well-known/agent-card.json"

        card = adapter._build_card("http://agents.example.com/", agent=route["agent"])
        assert card["name"] == "Research Agent"
        assert card["supportedInterfaces"][0]["url"] == "http://agents.example.com/research/"
        assert card["supportedInterfaces"][0]["tenant"] == "research"
        assert {s["name"] for s in card["skills"]} == {"research", "web"}

    def test_tenant_routing_selects_agent_without_path_prefix(self):
        from plugins.platforms.a2a.adapter import A2AAdapter
        from gateway.config import PlatformConfig

        adapter = A2AAdapter(PlatformConfig(enabled=True, extra={
            "agents": {
                "dev": {"profile": "dev", "tenant": "dev-team", "capabilities": ["code"]}
            }
        }))
        route = adapter._route_for_request("/", {"tenant": "dev-team"})
        assert route["agent"]["slug"] == "dev"

    def test_tenant_mismatch_is_rejected(self):
        from plugins.platforms.a2a.adapter import A2AAdapter
        from gateway.config import PlatformConfig

        adapter = A2AAdapter(PlatformConfig(enabled=True, extra={
            "agents": {"dev": {"profile": "dev", "tenant": "dev-team"}}
        }))
        route = adapter._route_for_request("/dev/", {"tenant": "research"})
        assert "error" in route

    def test_forwarded_profile_task_completes_in_task_store(self, monkeypatch):
        from plugins.platforms.a2a.adapter import A2AAdapter
        from gateway.config import PlatformConfig

        adapter = A2AAdapter(PlatformConfig(enabled=True, extra={
            "agents": {"dev": {"profile": "dev", "tenant": "dev"}}
        }))
        agent = adapter._agents["dev"]

        def fake_forward(agent_arg, peer, context_id, framed_text, task_id=None):
            assert agent_arg["slug"] == "dev"
            assert peer == "peer-x"
            assert "hello" in framed_text
            assert task_id  # session thread identity rides the forward
            return "dev reply", protocol.STATE_COMPLETED

        adapter._forward_to_profile = fake_forward  # type: ignore
        terminal, pending = adapter._prepare_task(
            {"tenant": "dev", "message": protocol.text_message(protocol.ROLE_USER, "hello", context_id="ctx-dev")},
            "peer-x",
            agent=agent,
        )
        assert pending is None
        assert terminal["status"]["state"] == protocol.STATE_COMPLETED
        assert protocol.extract_text(terminal["artifacts"][0]) == "dev reply"
        assert adapter.tasks.get(terminal["id"])["state"] == protocol.STATE_COMPLETED


class TestClientTenantAndDiscovery:
    def test_rpc_body_echoes_tenant_from_agent_card(self, monkeypatch):
        posted = {}

        def fake_get(url, headers, timeout, *a, **kw):
            assert url.endswith("/.well-known/agent-card.json")
            return protocol.build_agent_card(
                name="dev",
                url="http://peer.example/dev/",
                description="dev",
                tenant="dev-team",
            )

        def fake_post(url, body, headers, timeout, **kw):
            posted["url"] = url
            posted["body"] = body
            return {"jsonrpc": "2.0", "id": body["id"], "result": {"task": protocol.build_task(
                "task-1", "ctx-1", protocol.STATE_COMPLETED, "ok"
            )}}

        monkeypatch.setattr(tools, "_http_get_json", fake_get)
        monkeypatch.setattr(tools, "_http_post_json", fake_post)
        reply, _ctx, _state = tools._send_task(
            "dev", {"url": "http://peer.example", "auth": {}, "timeout": 5}, "hello", "ctx-1"
        )
        assert reply == "ok"
        assert posted["url"] == "http://peer.example/dev/"
        assert posted["body"]["params"]["tenant"] == "dev-team"

    def test_discovery_falls_back_to_legacy_agent_json(self, monkeypatch):
        calls = []

        def fake_get(url, headers, timeout, *a, **kw):
            calls.append(url)
            if url.endswith("agent-card.json"):
                raise urllib.error.HTTPError(url, 404, "not found", {}, None)
            return protocol.build_agent_card(name="legacy", url="http://legacy/", description="legacy")

        monkeypatch.setattr(tools, "_http_get_json", fake_get)
        out = tools.a2a_discover({"url": "http://legacy"})
        assert "Agent: legacy" in out
        assert calls[0].endswith("/.well-known/agent-card.json")
        assert calls[1].endswith("/.well-known/agent.json")





# --------------------------------------------------------------------------
# Multiplex secondary-profile scope (construction-time config leak)
# --------------------------------------------------------------------------
#
# __init__'s port/advertised-toolsets reads and _load_served_agents's
# description default all previously read raw A2A_* env vars unconditionally.
# Under a multiplexed secondary profile, os.environ holds the DEFAULT
# profile's YAML-to-env bridge output — a secondary profile with its own
# (different, or absent) A2A config would silently borrow the default
# profile's port, toolset advertisement, agent name, or Agent Card
# description. Mirrors the Buzz/SimpleX fix for #98738.

_A2A_ENV_VARS = (
    "A2A_PORT",
    "A2A_AGENT_NAME",
    "A2A_ADVERTISED_TOOLSETS",
    "A2A_AGENT_DESCRIPTION",
)


@pytest.fixture(autouse=True)
def _clean_a2a_construction_env(monkeypatch):
    """Keep the new multiplex tests hermetic regardless of ambient env."""
    for var in _A2A_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    yield


@pytest.fixture
def multiplex_scope():
    """Install multiplex + a secondary-profile secret scope; restore after."""
    tokens = []

    def install(scope=None):
        from agent.secret_scope import set_multiplex_active, set_secret_scope

        set_multiplex_active(True)
        tokens.append(set_secret_scope(scope or {}))
        return tokens[-1]

    yield install

    from agent.secret_scope import reset_secret_scope, set_multiplex_active

    for token in reversed(tokens):
        reset_secret_scope(token)
    set_multiplex_active(False)


@pytest.fixture
def default_profile_env(monkeypatch):
    """The default profile's YAML-to-env bridge output in os.environ."""
    monkeypatch.setenv("A2A_PORT", "9111")
    monkeypatch.setenv("A2A_AGENT_NAME", "default-profile-agent")
    monkeypatch.setenv("A2A_ADVERTISED_TOOLSETS", "default-only-toolset")
    monkeypatch.setenv("A2A_AGENT_DESCRIPTION", "Default profile's own agent.")


class TestMultiplexConstructionScope:

    def test_secondary_profile_never_borrows_default_profile_env(
        self, multiplex_scope, default_profile_env
    ):
        """The secondary profile's own config is authoritative; keys absent
        from it fall to the module defaults, never to the default profile's
        bridged A2A_* env values."""
        from plugins.platforms.a2a.adapter import A2AAdapter, _DEFAULT_PORT
        from gateway.config import PlatformConfig

        multiplex_scope()
        assert A2AAdapter(PlatformConfig(enabled=True, extra={"port": 9222})).port == 9222

        adapter = A2AAdapter(PlatformConfig(enabled=True, extra={}))
        assert adapter.port == _DEFAULT_PORT
        assert adapter.agent_name != "default-profile-agent"
        assert adapter._agents[""]["description"] == (
            "Hermes Agent — a general-purpose agent reachable over A2A."
        )

    def test_default_profile_unscoped_keeps_env_precedence(
        self, monkeypatch, default_profile_env
    ):
        """Multiplex ON but no scope (the DEFAULT profile constructs
        unscoped): env is its own bridge output and still wins."""
        from agent.secret_scope import set_multiplex_active
        from plugins.platforms.a2a.adapter import A2AAdapter
        from gateway.config import PlatformConfig

        set_multiplex_active(True)
        try:
            adapter = A2AAdapter(PlatformConfig(enabled=True, extra={}))
        finally:
            set_multiplex_active(False)
        assert adapter.port == 9111
        assert adapter.agent_name == "default-profile-agent"
        assert adapter._agents[""]["description"] == "Default profile's own agent."

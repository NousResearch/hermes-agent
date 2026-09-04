"""Tests for A2A V1 spec regression fixes — extracted from test_a2a_plugin.py.

Covers V1 method routing, cross-tenant isolation, push config scoping,
malformed params handling, health endpoint auth, reserved path rejection,
and profile forwarding.
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import urllib.request

import pytest

from plugins.platforms.a2a import protocol, security, tools


def _free_port() -> int:
    """Pick an ephemeral TCP port."""
    import socket as _sock
    with _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_live_adapter(monkeypatch, reply_fn=None):
    """Create an adapter on a free port with a mocked agent handler."""
    from plugins.platforms.a2a.adapter import A2AAdapter
    from gateway.config import PlatformConfig

    port = _free_port()
    monkeypatch.setenv("A2A_PORT", str(port))
    adapter = A2AAdapter(PlatformConfig(enabled=True, extra={"port": port}))

    async def fake_handle_message(event):
        if reply_fn is None:
            reply = "ECHO: " + event.text
        else:
            reply = reply_fn(event)
        if reply is not None:
            await adapter.send(event.source.chat_id, reply, metadata={"notify": True})

    adapter.handle_message = fake_handle_message  # type: ignore
    adapter._message_handler = object()
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


class TestV1SpecRegressionFixes:
    def test_rpc_send_message_v1_returns_send_message_response_wrapper(self, monkeypatch):
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
        adapter, base = _make_live_adapter(monkeypatch)

        async def run():
            assert await adapter.connect() is True
            body = _send_body("hello v1")
            body["method"] = "SendMessage"
            resp = await asyncio.to_thread(_post_json, base + "/", body, {"A2A-Version": "1.0"})
            assert resp["id"] == "1"
            assert set(resp["result"].keys()) == {"task"}
            task = resp["result"]["task"]
            assert task["status"]["state"] == protocol.STATE_COMPLETED
            assert "hello v1" in protocol.extract_text(task["artifacts"][0])
            get_resp = await asyncio.to_thread(_post_json, base + "/", {
                "jsonrpc": "2.0", "id": "2", "method": "GetTask",
                "params": {"id": task["id"]},
            }, {"A2A-Version": "1.0"})
            assert get_resp["result"]["id"] == task["id"]
            list_resp = await asyncio.to_thread(_post_json, base + "/", {
                "jsonrpc": "2.0", "id": "3", "method": "ListTasks",
                "params": {"contextId": task["contextId"], "pageSize": 10},
            }, {"A2A-Version": "1.0"})
            assert list_resp["result"]["nextPageToken"] == ""
            assert list_resp["result"]["pageSize"] == 10
            assert list_resp["result"]["totalSize"] >= 1
            assert "artifacts" not in list_resp["result"]["tasks"][0]
            await adapter.disconnect()

        asyncio.run(run())

    def test_client_sends_v1_method_and_unwraps_response(self, monkeypatch):
        posted = {}

        def fake_get(url, headers, timeout, *a, **kw):
            return protocol.build_agent_card(
                name="dev", url="http://peer.example/dev/", description="dev", tenant="dev-team")

        def fake_post(url, body, headers, timeout, **kw):
            posted["headers"] = headers
            posted["body"] = body
            return {"jsonrpc": "2.0", "id": body["id"], "result": {"task": protocol.build_task(
                "task-1", "ctx-1", protocol.STATE_COMPLETED, "ok")}}

        monkeypatch.setattr(tools, "_http_get_json", fake_get)
        monkeypatch.setattr(tools, "_http_post_json", fake_post)
        reply, _ctx, state = tools._send_task(
            "dev", {"url": "http://peer.example", "auth": {}, "timeout": 5}, "hello", "ctx-1")
        assert reply == "ok"
        assert state == protocol.STATE_COMPLETED
        assert posted["body"]["method"] == "SendMessage"
        assert posted["body"]["params"]["tenant"] == "dev-team"

    def test_cross_tenant_task_access_is_hidden(self):
        from plugins.platforms.a2a.adapter import A2AAdapter
        from gateway.config import PlatformConfig

        adapter = A2AAdapter(PlatformConfig(enabled=True, extra={
            "agents": {
                "research": {"profile": "research", "tenant": "research"},
                "dev": {"profile": "dev", "tenant": "dev"},
            }
        }))
        research = adapter._agents["research"]
        dev = adapter._agents["dev"]
        adapter.tasks.create("task-r", "ctx-r", "peer", *adapter._scope_for_agent(research))
        adapter.tasks.complete("task-r", protocol.STATE_COMPLETED, "secret")
        assert adapter._rpc_tasks_get(1, {"id": "task-r", "tenant": "research"}, agent=research)["result"]["id"] == "task-r"
        assert adapter._rpc_tasks_get(2, {"id": "task-r", "tenant": "dev"}, agent=dev)["error"]["code"] == protocol.ERR_TASK_NOT_FOUND
        assert adapter._rpc_tasks_cancel(3, {"id": "task-r", "tenant": "dev"}, agent=dev)["error"]["code"] == protocol.ERR_TASK_NOT_FOUND
        list_resp = adapter._rpc_tasks_list(4, {"tenant": "dev"}, agent=dev)
        assert list_resp["result"]["tasks"] == []

    def test_push_config_is_tenant_scoped(self):
        from plugins.platforms.a2a.adapter import A2AAdapter
        from gateway.config import PlatformConfig

        adapter = A2AAdapter(PlatformConfig(enabled=True, extra={
            "agents": {
                "research": {"profile": "research", "tenant": "research"},
                "dev": {"profile": "dev", "tenant": "dev"},
            }
        }))
        research = adapter._agents["research"]
        dev = adapter._agents["dev"]
        adapter.tasks.create("task-r", "ctx-r", "peer", *adapter._scope_for_agent(research))
        ok = adapter._rpc_push_config_create(1, {
            "taskId": "task-r", "tenant": "research",
            "pushNotificationConfig": {"url": "https://example.com/hook"},
        }, agent=research)
        assert ok["result"]["configId"].startswith("cfg-")
        hidden = adapter._rpc_push_config_get(2, {"taskId": "task-r", "tenant": "dev"}, agent=dev)
        assert hidden["error"]["code"] == protocol.ERR_TASK_NOT_FOUND

    def test_malformed_params_returns_jsonrpc_error_not_500(self, monkeypatch):
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
        adapter, base = _make_live_adapter(monkeypatch)

        async def run():
            assert await adapter.connect() is True
            resp = await asyncio.to_thread(_post_json, base + "/", {
                "jsonrpc": "2.0", "id": "bad", "method": "GetTask", "params": []})
            assert resp["error"]["code"] == protocol.ERR_INVALID_PARAMS
            await adapter.disconnect()

        asyncio.run(run())

    def test_remote_health_does_not_leak_served_agents_without_auth(self, monkeypatch):
        monkeypatch.setenv("A2A_BEARER_TOKEN", "secret")
        monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
        adapter, base = _make_live_adapter(monkeypatch)

        async def run():
            assert await adapter.connect() is True
            payload = await asyncio.to_thread(_get_json, base + "/health")
            assert payload["status"] == "ok"
            assert "served_agents" not in payload
            payload2 = await asyncio.to_thread(_get_json, base + "/health", {"Authorization": "Bearer secret"})
            assert "served_agents" in payload2
            await adapter.disconnect()

        asyncio.run(run())

    def test_reserved_paths_and_duplicate_tenants_are_ignored(self):
        from plugins.platforms.a2a.adapter import A2AAdapter
        from gateway.config import PlatformConfig

        adapter = A2AAdapter(PlatformConfig(enabled=True, extra={
            "agents": {
                "bad": {"path": "health", "profile": "bad", "tenant": "bad"},
                "one": {"profile": "one", "tenant": "same"},
                "two": {"profile": "two", "tenant": "same"},
            }
        }))
        assert "bad" not in adapter._agents
        assert "one" in adapter._agents
        assert "two" not in adapter._agents

    def test_forward_to_profile_first_contact_creates_then_resumes_fake_hermes(self, monkeypatch, tmp_path):
        from plugins.platforms.a2a.adapter import A2AAdapter
        from gateway.config import PlatformConfig

        profile_home = tmp_path / "profile"
        profile_home.mkdir()
        db = profile_home / "state.db"
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, started_at REAL, title TEXT)")
        con.commit(); con.close()

        fakebin = tmp_path / "bin"
        fakebin.mkdir()
        calls = tmp_path / "calls.jsonl"
        hermes = fakebin / "hermes"
        hermes.write_text("""#!/usr/bin/env python3
import json, os, sqlite3, sys, time
calls = os.environ['FAKE_HERMES_CALLS']
with open(calls, 'a') as f:
    f.write(json.dumps(sys.argv[1:]) + '\\n')
home = os.environ['HERMES_HOME']
con = sqlite3.connect(os.path.join(home, 'state.db'))
if '--resume' not in sys.argv:
    con.execute('INSERT INTO sessions (id, source, started_at, title) VALUES (?, ?, ?, ?)', ('sess-1', 'a2a', time.time(), None))
    con.commit()
print('fake reply')
""")
        hermes.chmod(0o755)
        monkeypatch.setenv("PATH", str(fakebin) + os.pathsep + os.environ.get("PATH", ""))
        monkeypatch.setenv("FAKE_HERMES_CALLS", str(calls))
        monkeypatch.setattr("plugins.platforms.a2a.adapter._profile_home", lambda profile: str(profile_home))

        adapter = A2AAdapter(PlatformConfig(enabled=True, extra={
            "agents": {"dev": {"profile": "dev", "tenant": "dev", "timeout": 5}}
        }))
        agent = adapter._agents["dev"]
        reply, state = adapter._forward_to_profile(agent, "peer", "ctx/unsafe value", "hello", "task-1")
        assert (reply, state) == ("fake reply", protocol.STATE_COMPLETED)
        reply2, state2 = adapter._forward_to_profile(agent, "peer", "ctx/unsafe value", "again", "task-2")
        assert (reply2, state2) == ("fake reply", protocol.STATE_COMPLETED)
        argv_lines = [json.loads(line) for line in calls.read_text().splitlines()]
        assert "--resume" not in argv_lines[0]
        assert argv_lines[1][argv_lines[1].index("--resume") + 1] == "sess-1"
        con = sqlite3.connect(db)
        title = con.execute("SELECT title FROM sessions WHERE id='sess-1'").fetchone()[0]
        con.close()
        assert title == "a2a-dev-ctx-unsafe-value"

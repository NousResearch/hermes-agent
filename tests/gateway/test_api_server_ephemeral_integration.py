"""End-to-end privacy proof for ephemeral API-server room turns."""

import json
import os
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.base import PlatformConfig
from tests.gateway.test_api_server import _create_app


def _database_rows(path: Path):
    if not path.exists():
        return {}
    with sqlite3.connect(path) as connection:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        return {
            table: rows
            for table in tables
            if (rows := connection.execute(f'SELECT * FROM "{table}"').fetchall())
        }


class _ProviderHandler(BaseHTTPRequestHandler):
    captured = []

    def do_POST(self):  # noqa: N802 - stdlib handler contract
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        type(self).captured.append(payload)
        if payload.get("stream"):
            chunks = [
                {
                    "id": "chatcmpl-ephemeral-proof",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "mock-runtime-model",
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": "deterministic room reply"},
                        "finish_reason": None,
                    }],
                },
                {
                    "id": "chatcmpl-ephemeral-proof",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "mock-runtime-model",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
                },
            ]
            body = b"".join(
                f"data: {json.dumps(chunk)}\n\n".encode() for chunk in chunks
            ) + b"data: [DONE]\n\n"
            content_type = "text/event-stream"
        else:
            body = json.dumps({
                "id": "chatcmpl-ephemeral-proof",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "mock-runtime-model",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "deterministic room reply"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            }).encode()
            content_type = "application/json"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *_args):
        return


@pytest.mark.asyncio
async def test_ephemeral_turn_reads_identity_and_memory_without_writing_stores(monkeypatch):
    home = Path(os.environ["HERMES_HOME"])
    memories = home / "memories"
    memories.mkdir(parents=True, exist_ok=True)
    soul = home / "SOUL.md"
    memory = memories / "MEMORY.md"
    user = memories / "USER.md"
    soul.write_text("Identity marker: AGENT_IDENTITY_PROOF\n", encoding="utf-8")
    memory.write_text("- Memory marker: EXISTING_MEMORY_PROOF\n", encoding="utf-8")
    user.write_text("- User marker: EXISTING_USER_PROOF\n", encoding="utf-8")
    (home / "config.yaml").write_text(
        "memory:\n"
        "  memory_enabled: true\n"
        "  user_profile_enabled: true\n"
        "  provider: lifecycle-tripwire\n"
        "sessions:\n"
        "  write_json_snapshots: true\n",
        encoding="utf-8",
    )
    before = {path: path.read_bytes() for path in (soul, memory, user)}

    external_provider_loads = []

    def load_external_provider(name):
        external_provider_loads.append(name)
        return None

    monkeypatch.setattr("plugins.memory.load_memory_provider", load_external_provider)

    _ProviderHandler.captured = []
    provider = ThreadingHTTPServer(("127.0.0.1", 0), _ProviderHandler)
    thread = threading.Thread(target=provider.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{provider.server_port}/v1"

    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "custom",
            "base_url": base_url,
            "api_key": "test-only",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "mock-runtime-model")
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_reasoning_config",
        staticmethod(lambda: {}),
    )
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_fallback_model",
        staticmethod(lambda: None),
    )
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_: set())
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "room-profile")

    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    created_agents = []
    create_agent = adapter._create_agent

    def capture_agent(*args, **kwargs):
        agent = create_agent(*args, **kwargs)
        created_agents.append(agent)
        return agent

    monkeypatch.setattr(adapter, "_create_agent", capture_agent)
    app = _create_app(adapter)
    state_db = home / "state.db"
    database_before = _database_rows(state_db)
    try:
        async with TestClient(TestServer(app)) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={"X-Hermes-Ephemeral": "true"},
                json={
                    "model": "hermes-agent",
                    "messages": [
                        {"role": "user", "content": "ROOM_TRANSCRIPT_PROOF"},
                    ],
                },
            )
            assert response.status == 200, await response.text()
            payload = await response.json()
            assert payload["choices"][0]["message"]["content"] == "deterministic room reply"
            assert payload["hermes"] == {
                "ephemeral": True,
                "persistence": "disabled",
                "memory": {
                    "local": "read-only",
                    "external_provider": "disabled",
                },
                "tools": "disabled",
                "profile": "room-profile",
                "requested_model": "hermes-agent",
                "provider": "custom",
                "model": "mock-runtime-model",
            }
    finally:
        provider.shutdown()
        provider.server_close()
        thread.join(timeout=2)

    assert _ProviderHandler.captured
    assert len(created_agents) == 1
    assert created_agents[0].tools == []
    assert created_agents[0].valid_tool_names == set()
    provider_request = next(
        payload for payload in _ProviderHandler.captured if "messages" in payload
    )
    assert "tools" not in provider_request
    provider_messages = provider_request["messages"]
    serialized = json.dumps(provider_messages)
    assert "AGENT_IDENTITY_PROOF" in serialized
    assert "EXISTING_MEMORY_PROOF" in serialized
    assert "EXISTING_USER_PROOF" in serialized
    assert "ROOM_TRANSCRIPT_PROOF" in serialized

    assert {path: path.read_bytes() for path in before} == before
    assert _database_rows(state_db) == database_before
    assert not list(home.glob("sessions/*.json"))
    assert external_provider_loads == []

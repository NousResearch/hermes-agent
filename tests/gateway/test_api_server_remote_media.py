"""API-server MEDIA: tags pointing into a remote terminal sandbox are inlined as data URLs.

On an ssh/modal/daytona backend the agent writes files on the sandbox host, which the gateway cannot
read, so ``_resolve_media_to_data_urls`` returned the raw ``MEDIA:/path`` to the HTTP client. It now
takes the same remote-fetch fallback as the messaging lane (#466) — from the key the turn's sandbox
was registered under, because the turn's session context is already cleared when the route resolves
media (#130875).

Stubbed one layer below ``_run_agent`` so the real session binding, its clearing and each route's
call site are what is exercised; the agent registers its sandbox through the real resolver, the way
the terminal tool does.
"""

import base64
import hashlib
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("aiohttp")

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB

REMOTE_PNG = "/home/agent/out/chart.png"
STORED_SESSION = "sess-remote-media"
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQAB"
    "h6FO1AAAAABJRU5ErkJggg==")
_API_KEY = "sk-test-remote-media"
_DATA_URL = re.compile(r"data:image/png;base64,([A-Za-z0-9+/=]+)")


class _FakeSandbox:
    """Stands in for a live SSHEnvironment holding one image the gateway host cannot see."""

    _remote_home = "/home/agent"

    def __init__(self, payload: bytes):
        self.payload = payload
        self.fetched: list = []

    def fetch_realpath(self, remote_path):
        return remote_path

    def fetch_file(self, remote_path, local_dest, *, max_bytes):
        self.fetched.append(remote_path)
        Path(local_dest).write_bytes(self.payload)


@pytest.fixture
def served(monkeypatch, tmp_path):
    from tools import terminal_tool

    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    cache = tmp_path / "cache" / "documents"
    monkeypatch.setattr("gateway.platforms.base.DOCUMENT_CACHE_DIR", cache)
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (cache,))
    sandbox = _FakeSandbox(_PNG)
    # A CLI or cron task's environment under the context-less key must never be the one consulted.
    unrelated = _FakeSandbox(b"not this one")
    monkeypatch.setitem(terminal_tool._active_environments, "default", unrelated)

    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": _API_KEY}))
    db = SessionDB(tmp_path / "state.db")
    db.create_session(session_id=STORED_SESSION, source="api_server", model="m")
    adapter._session_db = db

    def _create_agent(**kw):
        agent = MagicMock()
        agent.session_id = kw.get("session_id")
        agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0

        def _run_conversation(**run_kw):
            key = terminal_tool._resolve_container_task_id(run_kw["task_id"])
            monkeypatch.setitem(terminal_tool._active_environments, key, sandbox)
            return {"final_response": f"Here it is: MEDIA:{REMOTE_PNG}", "messages": [], "api_calls": 1}

        agent.run_conversation.side_effect = _run_conversation
        return agent

    adapter._create_agent = _create_agent
    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    yield app, sandbox, unrelated
    db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("path, body", [
    ("/v1/chat/completions", {"model": "hermes-agent", "messages": [{"role": "user", "content": "chart"}]}),
    ("/v1/responses", {"model": "hermes-agent", "input": "chart"}),
    (f"/api/sessions/{STORED_SESSION}/chat", {"message": "chart"}),
])
@pytest.mark.parametrize("session_key_header", [None, "openwebui:chat-42"])
async def test_sandbox_image_reaches_the_client_as_the_same_bytes(served, path, body, session_key_header):
    app, sandbox, unrelated = served
    headers = {"Authorization": f"Bearer {_API_KEY}"}
    if session_key_header:
        headers["X-Hermes-Session-Key"] = session_key_header

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(path, json=body, headers=headers)
        text = await resp.text()

    assert resp.status == 200, text
    inlined = _DATA_URL.findall(text)
    assert [hashlib.sha256(base64.b64decode(b)).hexdigest() for b in inlined] == [hashlib.sha256(_PNG).hexdigest()]
    assert f"MEDIA:{REMOTE_PNG}" not in text
    assert sandbox.fetched == [REMOTE_PNG]
    assert unrelated.fetched == []

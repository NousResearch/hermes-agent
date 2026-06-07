"""End-to-end front<->worker transport over a real loopback socket.

Unit tests inject the worker HTTP transport.  These wire the real
``WorkerClient`` (its own aiohttp client) to a real ``APIServerAdapter`` running
on an aiohttp ``TestServer`` (a real listening socket), with only the agent's
model turn stubbed.  This exercises the actual ``POST /v1/runs`` ->
``message.delta``/``response.media``/``run.completed`` SSE relay,
``continue_session`` history rehydration, and reset, over the wire.
"""

from unittest.mock import MagicMock, patch

import pytest
from aiohttp.test_utils import TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.worker_client import WorkerClient


def _app(adapter):
    from aiohttp import web

    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    app.router.add_post("/v1/runs/{run_id}/approval", adapter._handle_run_approval)
    app.router.add_delete("/api/sessions/{session_id}", adapter._handle_delete_session)
    return app


def _fake_agent(reply, captured_history=None):
    agent = MagicMock()

    def _run(user_message=None, conversation_history=None, task_id=None, **kw):
        if captured_history is not None:
            captured_history.append(list(conversation_history or []))
        return {"final_response": reply}

    agent.run_conversation.side_effect = _run
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    return agent


class _Collect:
    def __init__(self):
        self.deltas = []

    def on_delta(self, t):
        self.deltas.append(t)


@pytest.fixture
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MEDIA_SPOOL", str(tmp_path / "spool"))


@pytest.mark.asyncio
async def test_real_socket_dispatch_roundtrip(_isolated_home):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "k"}))
    server = TestServer(_app(adapter))
    await server.start_server()
    try:
        client = WorkerClient(f"http://127.0.0.1:{server.port}", "k")
        with patch.object(adapter, "_create_agent", return_value=_fake_agent("hi from worker")):
            result = await client.dispatch(input="hello", consumer=_Collect())
        assert result["output"] == "hi from worker"
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_real_socket_continue_session_rehydrates(_isolated_home):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "k"}))
    db = adapter._ensure_session_db()
    sid = "agent:coder:telegram:dm:42"
    db.create_session(sid, "test")
    db.append_message(sid, "user", "remember the number 7")
    db.append_message(sid, "assistant", "noted")

    captured = []
    server = TestServer(_app(adapter))
    await server.start_server()
    try:
        client = WorkerClient(f"http://127.0.0.1:{server.port}", "k")
        with patch.object(adapter, "_create_agent", return_value=_fake_agent("ok", captured)):
            await client.dispatch(
                input="what number?", consumer=_Collect(),
                session_id=sid, continue_session=True,
            )
        # The worker rehydrated its own transcript from state.db and handed it
        # to the agent — the prior turn is present.
        assert captured and any(
            "remember the number 7" in str(m.get("content", "")) for m in captured[0]
        )
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_real_socket_outbound_media_emitted(_isolated_home, tmp_path):
    img = tmp_path / "pic.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n fake")
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "k"}))
    server = TestServer(_app(adapter))
    await server.start_server()
    try:
        client = WorkerClient(f"http://127.0.0.1:{server.port}", "k")
        seen = []

        async def media_handler(ev):
            seen.append(ev)

        with patch.object(adapter, "_create_agent", return_value=_fake_agent(f"here you go MEDIA:{img}")):
            result = await client.dispatch(input="send pic", consumer=_Collect(), media_handler=media_handler)
        # The MEDIA: tag was minted to a ref and emitted as response.media; the
        # delivered text is tag-free.
        assert seen and seen[0]["media"] and "ref" in seen[0]["media"][0]
        assert "MEDIA:" not in result["output"]
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_real_socket_reset_unknown_session_is_idempotent(_isolated_home):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "k"}))
    server = TestServer(_app(adapter))
    await server.start_server()
    try:
        client = WorkerClient(f"http://127.0.0.1:{server.port}", "k")
        # No such session yet -> worker returns 404 -> reset is a no-op, not an error.
        await client.reset_session("agent:coder:telegram:dm:never")
    finally:
        await server.close()

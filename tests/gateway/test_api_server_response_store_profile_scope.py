"""/v1/responses state belongs to the routed profile (#84253).

A multiplexed API server mirrors every route under /p/<profile>/ with that profile's key, but kept ONE
ResponseStore at the launch home. Conversation names are client-chosen ("main", "my-project"), so another
profile's key could chain onto a profile's conversation (reading its transcript, instructions and session),
become its tip, and GET or DELETE its responses. Real profile homes, prefix middleware and route table; only
the agent run is faked.
"""
from __future__ import annotations

import pytest

from agent import secret_scope as ss

ALICE, BOB, DEFAULT = "a" * 40, "b" * 40, "d" * 40


@pytest.fixture(autouse=True)
def _reset_multiplex():
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


async def _response_id(resp) -> str:
    """The id from a JSON body, or from the SSE stream's response.created event."""
    import json
    if resp.content_type == "application/json":
        return (await resp.json())["id"]
    for line in (await resp.text()).splitlines():
        if line.startswith("data:") and '"response"' in line:
            return json.loads(line[5:])["response"]["id"]
    raise AssertionError("no response id in the stream")


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
async def test_a_profile_key_never_reaches_another_profiles_responses(tmp_path, monkeypatch, stream):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    from gateway.config import GatewayConfig, PlatformConfig
    from gateway.platforms import api_server as api
    from gateway.platforms.api_server import APIServerAdapter

    home = tmp_path / ".hermes"
    for name, key in (("alice", ALICE), ("bob", BOB)):
        (home / "profiles" / name).mkdir(parents=True)
        (home / "profiles" / name / ".env").write_text(f"API_SERVER_KEY={key}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": DEFAULT}))
    adapter.gateway_runner = type("_Runner", (), {"config": GatewayConfig(multiplex_profiles=True)})()
    ss.set_multiplex_active(True)
    runs = []

    async def fake_run_agent(**kw):
        who = api._api_request_profile.get()
        runs.append({"profile": who, "history": kw.get("conversation_history"),
                     "instructions": kw.get("ephemeral_system_prompt"), "session_id": kw.get("session_id")})
        return ({"final_response": f"ok from {who}", "messages": [], "api_calls": 1,
                 "session_id": kw.get("session_id")}, {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})

    adapter._run_agent = fake_run_agent
    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)
        app.router.add_route(method, f"/p/{{profile}}{path}", handler)
    app["api_server_adapter"] = adapter

    def auth(key):
        return {"Authorization": f"Bearer {key}"}

    try:
        async with TestClient(TestServer(app)) as client:
            first = await client.post("/p/alice/v1/responses", headers=auth(ALICE), json={
                "input": "SECRET-ALICE", "conversation": "main", "instructions": "ALICE PRIVATE PROMPT",
                "stream": stream})
            alice_id = await _response_id(first)
            alice_session = runs[-1]["session_id"]

            bob = await client.post("/p/bob/v1/responses", headers=auth(BOB), json={
                "input": "what did I say?", "conversation": "main", "stream": stream})
            assert bob.status == 200
            await bob.read()
            assert runs[-1]["history"] == [] and runs[-1]["instructions"] is None
            assert runs[-1]["session_id"] != alice_session
            assert (await client.get(f"/p/bob/v1/responses/{alice_id}", headers=auth(BOB))).status == 404
            assert (await client.delete(f"/p/bob/v1/responses/{alice_id}", headers=auth(BOB))).status == 404

            await (await client.post("/p/alice/v1/responses", headers=auth(ALICE), json={
                "input": "next", "conversation": "main", "stream": stream})).read()
            assert "what did I say?" not in str(runs[-1]["history"])
            assert "SECRET-ALICE" in str(runs[-1]["history"])
            assert (await client.get(f"/p/alice/v1/responses/{alice_id}", headers=auth(ALICE))).status == 200
    finally:
        await adapter.disconnect()

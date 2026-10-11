"""#136472: a peer-DM turn that ran and failed must be distinguishable from an answered one.

``POST /api/sessions/{id}/chat`` serves a 200 completion for a failed turn (the request really was
served and the message IS in the peer's transcript), so before this the provider's failure
paragraph was indistinguishable from a reply and ``hermes peer dm`` exited 0. The completion object
now carries the additive ``failed`` / ``error`` / ``failure_reason`` keys — the same typed verdict
``result_retry_action`` reads for its retry policy — and only for a failed turn.

Only the model turn is faked (``_create_agent``); the route, the store and ``_run_agent`` are the
real ones, same harness as ``test_peer_dm_transient_retry.py``.
"""

from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB

SESSION_ID = "peer_dm_failed_signal"
DM = "disk status?"

# Permanent (auth): the policy never re-runs it, so exactly one attempt reaches the payload below.
AUTH_WALL = {
    "final_response": "Provider authentication failed: invalid api key",
    "failed": True,
    "completed": False,
    "error": "Error code: 401 - invalid_api_key",
    "failure_reason": "auth",
    "messages": [],
    "api_calls": 1,
}


def _ok(text: str) -> dict:
    return {
        "final_response": text,
        "failed": False,
        "completed": True,
        "messages": [],
        "api_calls": 1,
    }


def _app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    return app


def _adapter(tmp_path) -> tuple[APIServerAdapter, str]:
    db = SessionDB(tmp_path / "state.db")
    sid = db.create_session(SESSION_ID, "api_server")
    db.append_message(sid, "user", content=DM)
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = db
    return adapter, sid


def _fake_agent(outcome: dict) -> MagicMock:
    agent = MagicMock()
    agent.session_id = SESSION_ID
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0

    def _run(user_message=None, conversation_history=None, task_id=None, **_kwargs):
        return dict(outcome)

    agent.run_conversation.side_effect = _run
    return agent


@pytest.mark.asyncio
async def test_a_failed_turn_completes_200_but_carries_the_failure_verdict(tmp_path):
    adapter, sid = _adapter(tmp_path)
    app = _app(adapter)

    with patch.object(
        adapter, "_create_agent", side_effect=lambda **_kw: _fake_agent(AUTH_WALL)
    ):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(f"/api/sessions/{sid}/chat", json={"message": DM})
            body = await resp.json()

    assert resp.status == 200, (
        "the request was served; a non-2xx would break other consumers"
    )
    assert body["object"] == "hermes.session.chat.completion"
    assert body["failed"] is True
    assert body["error"] == AUTH_WALL["error"]
    assert body["failure_reason"] == AUTH_WALL["failure_reason"]
    assert body["message"]["content"] == AUTH_WALL["final_response"], (
        "the failure text stays the reply body for consumers that only read message.content"
    )


@pytest.mark.asyncio
async def test_an_answered_turn_carries_no_failure_keys(tmp_path):
    adapter, sid = _adapter(tmp_path)
    app = _app(adapter)

    with patch.object(
        adapter, "_create_agent", side_effect=lambda **_kw: _fake_agent(_ok("fine"))
    ):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(f"/api/sessions/{sid}/chat", json={"message": DM})
            body = await resp.json()

    assert resp.status == 200
    assert body["message"]["content"] == "fine"
    assert (
        "failed" not in body and "error" not in body and "failure_reason" not in body
    ), (
        "the additive keys appear only for a failed turn, so their absence stays an answer"
    )

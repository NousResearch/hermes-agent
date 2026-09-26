"""#115325: the peer-DM transport (``POST /api/sessions/{id}/chat``) retries a transiently failed
turn once, resuming the DM row the failed attempt already persisted.

``hermes peer dm`` is the third Bot-DM transport. The local (``tools.bot_mode_dm``) and relayed
(``tui_gateway.methods_bot_relay``) lanes both re-run a transiently failed turn once and resume the
row the failed attempt left as the transcript's unanswered tail (``tools/bot_failure_reasons``
``retry_action`` / ``RESUME_UNANSWERED_TURN``); the peer lane ran the turn once and handed whatever
came back — the provider's 429 paragraph — to the sender as the reply.

Only the model turn is faked (``_create_agent`` hands out recording agents); the route, the store and
``_run_agent`` are the real ones, so both halves of the contract are asserted on what the retry was
actually handed: the persisted row adopted as this turn's user message, and no second attempt for a
failure the policy never retries.
"""

from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB

SESSION_ID = "peer_dm_retry"
DM = "disk status?"

# Transient: the policy's own class (429 / rate limit).
RATE_LIMIT = {
    "final_response": "Rate limited by the provider. API call failed after 3 retries: 429 Too Many Requests",
    "failed": True, "completed": False, "error": "429 Too Many Requests",
    "failure_reason": "rate_limit", "messages": [], "api_calls": 3,
}
# Permanent: a wall no re-run can fix.
AUTH_WALL = {
    "final_response": "Provider authentication failed: invalid api key",
    "failed": True, "completed": False, "error": "Error code: 401 - invalid_api_key",
    "failure_reason": "auth", "messages": [], "api_calls": 1,
}


def _ok(text: str) -> dict:
    return {"final_response": text, "failed": False, "completed": True,
            "messages": [], "api_calls": 1}


def _app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    return app


def _adapter(tmp_path) -> tuple[APIServerAdapter, str]:
    """A real adapter over a real store, holding the DM row a failed attempt's turn-start persist
    would have left as the transcript's unanswered tail."""
    db = SessionDB(tmp_path / "state.db")
    sid = db.create_session(SESSION_ID, "api_server")
    db.append_message(sid, "user", content=DM)
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = db
    return adapter, sid


def _fake_agent(seen: list, outcome: dict) -> MagicMock:
    """Records what each turn was handed; ``_pending_cli_user_message`` is the adopted row."""
    agent = MagicMock()
    agent.session_id = SESSION_ID
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0

    def _run(user_message=None, conversation_history=None, task_id=None, **_kwargs):
        seen.append({
            "message": user_message,
            "history": list(conversation_history or []),
            "resumed": getattr(agent, "_pending_cli_user_message", None),
        })
        return dict(outcome)

    agent.run_conversation.side_effect = _run
    return agent


@pytest.mark.asyncio
async def test_peer_dm_retries_a_transient_turn_once_and_still_reports_a_permanent_one(tmp_path):
    adapter, sid = _adapter(tmp_path)
    app = _app(adapter)

    # ① Transient failure -> re-run once, resuming the persisted DM row, and the recovered reply is
    #    what the peer client receives.
    seen: list = []
    agents = [_fake_agent(seen, RATE_LIMIT), _fake_agent(seen, _ok("2+2 is 4"))]
    with patch.object(adapter, "_create_agent", side_effect=lambda **_kw: agents.pop(0)):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(f"/api/sessions/{sid}/chat", json={"message": DM})
            body = await resp.json()

    assert resp.status == 200
    assert len(seen) == 2, "a transiently failed peer DM must be re-run exactly once"
    assert body["message"]["content"] == "2+2 is 4"
    retry = seen[1]
    assert retry["message"] == DM, "the re-run replays the same DM"
    resumed = retry["resumed"]
    assert isinstance(resumed, dict) and resumed.get("content") == DM, (
        "the re-run must resume the row the failed attempt persisted as the pending user message")
    assert resumed.get("_db_persisted") is True, "the resumed row is the durable one, not a copy"
    assert all(row.get("content") != DM for row in retry["history"]), (
        "the resumed row leaves the history, so the turn appends no second copy of the DM")

    # ② Permanent failure -> reported exactly as before: one attempt, its error copy as the reply.
    seen2: list = []
    walls = [_fake_agent(seen2, AUTH_WALL), _fake_agent(seen2, AUTH_WALL)]
    with patch.object(adapter, "_create_agent", side_effect=lambda **_kw: walls.pop(0)):
        async with TestClient(TestServer(app)) as cli:
            resp2 = await cli.post(f"/api/sessions/{sid}/chat", json={"message": DM})
            body2 = await resp2.json()

    assert resp2.status == 200
    assert len(seen2) == 1, "auth is never auto-retried: it cannot be fixed by a re-run"
    assert body2["message"]["content"] == AUTH_WALL["final_response"]

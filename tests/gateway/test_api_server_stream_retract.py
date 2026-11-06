"""The API server tells its stream clients when streamed text is withdrawn (#142333).

A model call that restarts after deltas were delivered (content-filter stop, malformed
response, provider failover) fires the agent's ``stream_retract_callback``. The session chat
stream says ``assistant.retract {message_id}`` and ``/v1/runs`` says ``message.retract``,
each before the restarted reply's first delta, so a client that renders deltas drops the
dead text instead of showing a refusal followed by the real answer. Clients that ignore
unknown events are unaffected.
"""

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB
from tests.gateway.test_api_server_runs import _create_runs_app
from tests.gateway.test_session_api import _create_session_app


@pytest.fixture
def session_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        yield db
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()


@pytest.fixture
def adapter(session_db):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = session_db
    return adapter


def _fake_create_agent(script):
    """An agent whose turn replays ``script`` against the callbacks the server handed it."""

    def create_agent(**kwargs):
        delta, retract = kwargs["stream_delta_callback"], kwargs["stream_retract_callback"]
        agent = MagicMock()

        def run_conversation(**_kw):
            for kind, text in script:
                (delta(text) if kind == "delta" else retract())
            return {"final_response": "Here is the sermon.", "messages": [], "api_calls": 2}

        agent.run_conversation.side_effect = run_conversation
        agent.provider, agent.model = "openrouter", "deepseek/deepseek-v4-pro"
        agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
        return agent

    return create_agent


SCRIPT = [("delta", "I'm sorry, I cannot help with that"), ("retract", None), ("delta", "Here is the sermon.")]


@pytest.mark.asyncio
async def test_session_stream_retracts_the_dead_text_before_the_fallbacks_delta(adapter, session_db, monkeypatch):
    session_id = session_db.create_session("retract-session", "api_server")
    monkeypatch.setattr("gateway.run._load_gateway_config", dict)
    app = _create_session_app(adapter)
    with patch.object(adapter, "_create_agent", side_effect=_fake_create_agent(SCRIPT)):
        async with TestClient(TestServer(app)) as cli:
            body = await (await cli.post(f"/api/sessions/{session_id}/chat/stream", json={"message": "go"})).text()

    frames = [ln for ln in body.splitlines() if ln.startswith("event: ") or ln.startswith("data: ")]
    events = []
    for i, line in enumerate(frames):
        if line.startswith("event: "):
            events.append((line[7:], json.loads(frames[i + 1][6:])))
    names = [n for n, _ in events]
    assert names.index("assistant.delta") < names.index("assistant.retract") < names.index("assistant.completed"), names
    deltas = [d for n, d in events if n == "assistant.delta"]
    retract = next(d for n, d in events if n == "assistant.retract")
    assert retract["message_id"] == deltas[0]["message_id"]
    assert [d["delta"] for d in deltas] == ["I'm sorry, I cannot help with that", "Here is the sermon."]
    # The retract sits between the two deltas, not after the reply.
    assert names[names.index("assistant.retract") - 1] == "assistant.delta"
    assert names[names.index("assistant.retract") + 1] == "assistant.delta"


@pytest.mark.asyncio
async def test_runs_stream_retracts_between_the_deltas(adapter):
    from concurrent.futures import ThreadPoolExecutor

    class CompletedWorkerExecutor(ThreadPoolExecutor):
        def submit(self, fn, /, *args, **kwargs):
            future = super().submit(fn, *args, **kwargs)
            future.exception(timeout=10)  # deterministic ordering, real thread callbacks
            return future

    asyncio.get_running_loop().set_default_executor(CompletedWorkerExecutor(max_workers=1))
    app = _create_runs_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_create_agent", side_effect=_fake_create_agent(SCRIPT)):
            resp = await cli.post("/v1/runs", json={"input": "hello"})
            run_id = (await resp.json())["run_id"]
            body = await (await cli.get(f"/v1/runs/{run_id}/events")).text()

    events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
    assert [e["event"] for e in events] == ["message.delta", "message.retract", "message.delta", "run.completed"], events
    assert events[1]["run_id"] == run_id
    assert [e["delta"] for e in events if e["event"] == "message.delta"] == [
        "I'm sorry, I cannot help with that", "Here is the sermon."]


def test_canonical_turn_observers_receive_the_retract():
    """On the session-authority path the agent's retract reaches the admission's API observers
    (the session stream and runs register ``stream_retract_callback`` like their delta sink)."""
    from gateway.session_api_turn import _OBSERVER_KEYS, publish_api_event, wire_api_observers

    assert "stream_retract_callback" in _OBSERVER_KEYS
    observer_retract = MagicMock()
    authority = SimpleNamespace(
        sessions={"s1": SimpleNamespace(event_stream=SimpleNamespace(
            execution={"admission_id": "a1"}, lock=threading.Lock()))},
        api_observers={"a1": [{"stream_retract_callback": observer_retract}]},
        check_approval_generation=lambda *_a: None,
    )
    agent = SimpleNamespace(tool_progress_callback=None, status_callback=None, interim_assistant_callback=None,
                            stream_retract_callback=None)
    wire_api_observers(agent, (authority, "s1", 1), False)

    agent.stream_retract_callback()
    observer_retract.assert_called_once_with()

    # A worker that publishes the event by name reaches the same sink.
    assert publish_api_event(authority, "s1", "message.retract", {}) is True
    assert observer_retract.call_count == 2

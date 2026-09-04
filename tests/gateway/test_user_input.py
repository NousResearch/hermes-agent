"""Gateway-independent delivery contracts for Hermes-native user input."""

from __future__ import annotations

import asyncio
import json
import threading


def _agent(model_active=False, executing_tools=False, turn_id="turn-1"):
    from types import SimpleNamespace

    events = []
    agent = SimpleNamespace(
        _current_turn_id=turn_id,
        _model_request_active=threading.Event(),
        _executing_tools=executing_tools,
    )
    if model_active:
        agent._model_request_active.set()

    def redirect(text):
        events.append(("redirect", text))
        return True

    def steer(text):
        events.append(("steer", text))
        return True

    agent.redirect = redirect
    agent.steer = steer
    agent.events = events
    return agent


def test_model_active_answer_redirects_without_steering():
    from tools.user_input_tool import deliver_answer_to_agent

    agent = _agent(model_active=True)
    assert deliver_answer_to_agent(agent, "turn-1", {"choice": "stable"}) == "redirected"
    assert agent.events == [("redirect", json.dumps({"choice": "stable"}, ensure_ascii=False))]


def test_tool_active_answer_uses_steering_boundary():
    from tools.user_input_tool import deliver_answer_to_agent

    agent = _agent(executing_tools=True)
    assert deliver_answer_to_agent(agent, "turn-1", {"choice": "stable"}) == "steered"
    assert agent.events == [("steer", json.dumps({"choice": "stable"}, ensure_ascii=False))]


def test_answer_for_stale_turn_is_deferred_without_queueing():
    from tools.user_input_tool import deliver_answer_to_agent

    agent = _agent(turn_id="new-turn")
    assert deliver_answer_to_agent(agent, "old-turn", {"choice": "stable"}) == "deferred"
    assert agent.events == []


def test_answer_after_turn_slot_is_cleared_is_deferred_without_queueing():
    from tools.user_input_tool import deliver_answer_to_agent

    agent = _agent(turn_id="finished-turn")
    agent._inflight_turn_id = None
    assert deliver_answer_to_agent(agent, "finished-turn", {"choice": "stable"}) == "deferred"
    assert agent.events == []


def test_base_adapter_user_input_fallback_is_explicit_and_structured():
    from gateway.platforms.base import BasePlatformAdapter, SendResult

    class Adapter(BasePlatformAdapter):
        async def connect(self, *args, **kwargs):
            return None

        async def disconnect(self, *args, **kwargs):
            return None

        async def get_chat_info(self, *args, **kwargs):
            return None

        async def send(self, *args, **kwargs):
            return SendResult(success=True)

    adapter = Adapter.__new__(Adapter)
    sent = []

    async def send(*, chat_id, content, metadata=None, **kwargs):
        sent.append((chat_id, content, metadata))
        return SendResult(success=True)

    adapter.send = send
    request = {
        "request_id": "request-1",
        "context": "Choose a compatibility target",
        "questions": [{
            "id": "target",
            "text": "Which target?",
            "options": ["stable", "beta"],
            "allow_free_text": False,
            "default": "stable",
        }],
    }
    result = asyncio.run(adapter.send_user_input("chat-1", request, "session-1"))

    assert result.success is True
    assert sent and sent[0][0] == "chat-1"
    content = sent[0][1]
    assert "/answer request-1" in content
    assert "target: Which target?" in content
    assert "1. stable" in content
    assert "2. beta" in content


def test_gateway_answer_envelope_resolves_active_turn_without_new_turn(tmp_path):
    import time
    from types import SimpleNamespace

    from gateway.run import GatewayRunner
    from tools.user_input_tool import request_user_input
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    db.create_session("gateway-session", source="telegram")
    try:
        request = json.loads(request_user_input(
            questions=[{"id": "target", "text": "Which target?", "options": ["stable"], "default": "stable"}],
            session_id="gateway-session", turn_id="turn-1", session_db=db,
            now=time.time(),
        ))
        delivered = []
        agent = _agent(executing_tools=True, turn_id="turn-1")
        agent.session_id = "gateway-session"
        agent.steer = lambda text: delivered.append(text) or True

        runner = object.__new__(GatewayRunner)
        runner._peek_session_state = lambda _key: SimpleNamespace(
            turn=SimpleNamespace(agent=agent)
        )
        runner._session_db = db
        event = SimpleNamespace(
            text=f"/answer {request['request_id']} {{\"target\": \"stable\"}}",
        )
        result = asyncio.run(runner._hm_user_input_reply(event, None, "route-key"))

        assert result == ""
        assert delivered == [json.dumps({"target": "stable"}, ensure_ascii=False)]
        assert db.get_pending_user_input(request["request_id"], session_id="gateway-session")["status"] == "answered"
    finally:
        db.close()

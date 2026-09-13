"""Opt-in notifications preserve the ordinary response and their initiating turn."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import acp
import pytest
from acp.agent.router import build_agent_router

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager
from agent import background_review


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [None, 1, True, "1", 2])
@pytest.mark.parametrize("streamed", [False, True])
async def test_commentary_is_opt_in_additive_and_restores_callbacks(version, streamed, monkeypatch):
    manager = SessionManager(agent_factory=lambda: MagicMock(name="Agent"))
    adapter = HermesACPAgent(manager)
    router = build_agent_router(adapter)
    conn = MagicMock(spec=acp.Client)
    conn.session_update = AsyncMock()
    adapter._conn = conn
    init = await router("initialize", {
        "protocolVersion": acp.PROTOCOL_VERSION,
        "clientCapabilities": {"_meta": {"hermes": {"messagePhases": version}}},
    }, False)
    init = init.model_dump(by_alias=True, exclude_none=True)
    enabled = type(version) is int and version == 1
    assert init["agentCapabilities"].get("_meta", {}).get("hermes") == (
        {"messagePhases": 1} if enabled else None
    )
    session = await adapter.new_session(cwd=".")
    state = manager.get_session(session.session_id)
    state.agent.session_id = session.session_id
    prior_interim, prior_review = lambda text, **kw: None, lambda text: None
    state.agent.interim_assistant_callback = prior_interim
    state.agent.background_review_callback = prior_review
    conn.session_update.reset_mock()
    token = str(uuid4())
    captured = []

    def run(**kwargs):
        if streamed:
            state.agent.stream_delta_callback("Working. Done.")
        if state.agent.interim_assistant_callback:
            state.agent.interim_assistant_callback("Working.", already_streamed=streamed)
            state.agent.interim_assistant_callback("Working.", already_streamed=streamed)
        captured.append(state.agent.background_review_callback)
        return {"final_response": "Working. Done.", "messages": []}

    monkeypatch.setattr(adapter, "_run_agent_turn", run)
    monkeypatch.setattr(adapter, "_send_usage_update", AsyncMock())
    await router("session/prompt", {
        "sessionId": session.session_id,
        "prompt": [{"type": "text", "text": "Do it"}],
        "_meta": {"hermes": {"turnId": token}},
    }, False)
    assert state.agent.interim_assistant_callback is prior_interim
    assert state.agent.background_review_callback is prior_review
    # A finished turn's callback can report late without reading a later turn's callback.
    if captured[0]:
        await asyncio.to_thread(captured[0], "Self-improvement review: Skill 'example' patched")
    updates = [call.args[1] if len(call.args) > 1 else call.kwargs["update"]
               for call in conn.session_update.call_args_list]
    chunks = [update.model_dump(by_alias=True, exclude_none=True)
              for update in updates if update.session_update == "agent_message_chunk"]
    raw = [chunk["content"]["text"] for chunk in chunks if "_meta" not in chunk]
    assert raw == ["Working. Done."]
    notices = [chunk for chunk in chunks if "_meta" in chunk]
    assert len(notices) == (3 if enabled else 0)
    if enabled:
        assert len({chunk["messageId"] for chunk in notices}) == 3
        assert [chunk["_meta"]["hermes"]["source"] for chunk in notices] == [
            "assistant", "assistant", "background_review",
        ]
        assert all(chunk["_meta"]["hermes"]["turnId"] == token for chunk in notices)
        assert all(chunk["_meta"]["hermes"]["phase"] == "commentary" for chunk in notices)


@pytest.mark.parametrize("initially_enabled", [False, True])
def test_background_review_uses_callback_captured_before_thread_start(initially_enabled, monkeypatch):
    delivered, wrong_owner = [], []
    parent = SimpleNamespace(
        background_review_callback=delivered.append if initially_enabled else None,
        _safe_print=lambda text: None, memory_notifications="on",
    )
    monkeypatch.setattr(background_review, "_parent_can_emit_tool_calls", lambda agent: True)
    monkeypatch.setattr(background_review, "_set_thread_approval_callback", lambda cb: None)
    monkeypatch.setattr(background_review, "_run_review_fork", lambda *args: None)
    monkeypatch.setattr(background_review, "summarize_background_review_actions", lambda *a, **kw: ["Skill 'example' patched"])
    monkeypatch.setattr(background_review, "_log_review_completion", lambda *a: None)
    target, _ = background_review.spawn_background_review_thread(
        parent, [], review_skills=True, task_cfg={},
    )
    parent.background_review_callback = wrong_owner.append
    target()
    assert wrong_owner == []
    assert bool(delivered) is initially_enabled


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["error", "cancel", "queued", "reconnect"])
async def test_turn_callbacks_do_not_escape_errors_cancellation_or_queued_prompts(outcome, monkeypatch):
    manager = SessionManager(agent_factory=lambda: MagicMock(name="Agent"))
    adapter = HermesACPAgent(manager)
    router = build_agent_router(adapter)
    adapter._conn = MagicMock(spec=acp.Client)
    adapter._conn.session_update = AsyncMock()
    await router("initialize", {
        "protocolVersion": acp.PROTOCOL_VERSION,
        "clientCapabilities": {"_meta": {"hermes": {"messagePhases": 1}}},
    }, False)
    session = await adapter.new_session(cwd=".")
    state = manager.get_session(session.session_id)
    prior = lambda text, **kwargs: None
    state.agent.interim_assistant_callback = prior
    state.agent.background_review_callback = prior
    state.agent.session_id = session.session_id
    callbacks = []

    def run(**kwargs):
        callbacks.append(state.agent.interim_assistant_callback)
        if len(callbacks) == 1:
            if outcome == "reconnect":
                assert state.agent.interim_assistant_callback is None
                assert state.agent.background_review_callback is None
            else:
                assert state.agent.interim_assistant_callback is not prior
            if outcome == "error":
                raise RuntimeError("provider failed")
            if outcome == "cancel":
                state.cancel_event.set()
            if outcome == "queued":
                state.queued_prompts.append("queued without a client token")
        else:
            # A queued admission with no original client token must not inherit
            # the first owner's destination or generate correlated notices.
            assert state.agent.interim_assistant_callback is None
            assert state.agent.background_review_callback is None
        return {"final_response": "done", "messages": []}

    monkeypatch.setattr(adapter, "_run_agent_turn", run)
    monkeypatch.setattr(adapter, "_send_usage_update", AsyncMock())
    if outcome == "reconnect":
        adapter.on_connect(adapter._conn)
    response = await router("session/prompt", {
        "sessionId": session.session_id,
        "prompt": [{"type": "text", "text": "Do it"}],
        "_meta": {"hermes": {"turnId": str(uuid4())}},
    }, False)
    assert state.agent.interim_assistant_callback is prior
    assert state.agent.background_review_callback is prior
    assert len(callbacks) == (2 if outcome == "queued" else 1)
    if outcome == "queued":
        assert callbacks[1] is None
    assert not state.is_running
    if outcome == "reconnect":
        assert callbacks == [None]
    if outcome == "cancel":
        assert response.stop_reason == "cancelled"


@pytest.mark.parametrize("commentary_enabled", [False, True])
def test_codex_commentary_consumer_preserves_analysis_reasoning_and_final_text(commentary_enabled):
    from agent.codex_runtime import _consume_codex_event_stream

    events = []
    items = []
    for phase, text in [("commentary", "Working."), ("analysis", "Private analysis."), ("final_answer", "Done.")]:
        item = {"id": phase, "type": "message", "phase": phase,
                "content": [{"type": "output_text", "text": text}]}
        items.append(item)
        events.extend([
            {"type": "response.output_item.added", "item": item},
            {"type": "response.output_text.delta", "delta": text},
            {"type": "response.output_item.done", "item": item},
        ])
    events.extend([
        {"type": "response.reasoning_text.delta", "delta": "Private reasoning."},
        {"type": "response.completed", "response": {"status": "completed"}},
    ])
    commentary, reasoning, output = [], [], []
    response = _consume_codex_event_stream(
        events, model="fixture-model", on_text_delta=output.append,
        on_reasoning_delta=reasoning.append,
        on_commentary_message=commentary.append if commentary_enabled else None,
    )
    assert output == ["Done."]
    assert response.output_text == "Done."
    assert response.output == items
    assert commentary == (["Working."] if commentary_enabled else [])
    assert reasoning == (["Private analysis.", "Private reasoning."] if commentary_enabled
                         else ["Working.", "Private analysis.", "Private reasoning."])

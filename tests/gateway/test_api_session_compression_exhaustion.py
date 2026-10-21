"""An API conversation whose turn exhausts compression stays addressable, and the session-chat
door's re-run continues the failed attempt's row (#115325) on the admitted path too, never past
completed tool work."""
from types import SimpleNamespace

import pytest

from gateway.session_api_turn import admit_api_turn

DM = "disk status?"


@pytest.mark.asyncio
async def test_api_session_keeps_its_id_after_compression_exhaustion(monkeypatch):
    """The messaging auto-reset rotated the API route to a fresh id, so every later request to the
    id the caller addresses was refused (``admission_conflict``; the session-chat door's 500)."""
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.run import GatewayRunner
    from gateway.session_authority import SessionAuthority, initialize_session_authority

    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)
    runner = GatewayRunner(GatewayConfig())
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter.gateway_runner = runner
    runner.adapters = {Platform.API_SERVER: adapter}
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="fixture")
    adapter._session_db = authority.db
    try:
        admit_api_turn(adapter, session_id="dm1", user_message=DM, conversation_history=[])
        source = authority.sessions["dm1"].source
        route = runner.session_store._generate_session_key(source)
        exhausted = {"failed": True, "compression_exhausted": True, "failure_reason": "context_overflow",
                     "messages": []}
        response, entry = await runner._hmwa_compression_exhaustion_reset(
            exhausted, "copy", runner.session_store.lookup_by_session_key(route), route, source)
        admit_api_turn(adapter, session_id="dm1", user_message="next", conversation_history=[])
        assert (response, entry.session_id) == ("copy", "dm1")
    finally:
        adapter._response_store.close()
        adapter._run_idempotency_store.close()
        authority.db.close()


def test_session_chat_rerun_continues_the_failed_attempts_row(api, owner):
    """The re-run's history ends with the DM its failed attempt persisted: the turn adopts that row
    instead of sending (and storing) a second copy. A fresh admission never adopts."""
    from gateway.run_turn_runner import TurnRunner
    from gateway.session_api_turn import api_execution, prepare_api_execution
    from gateway.turn_context import TurnContext

    history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "ok"},
               {"role": "user", "content": DM}]
    loaded = {}
    for rerun in (True, False):
        _, ref, row = admit_api_turn(api, session_id=f"dm-{rerun}", user_message=DM,
                                     conversation_history=history, resume_unanswered_turn=rerun)
        agent = SimpleNamespace()
        turn = TurnRunner(SimpleNamespace(session_authority=None),
                          TurnContext(message=DM, session_id=ref.session_id, session_key="api"))
        token = api_execution.set(prepare_api_execution(owner, ref, row["payload"]))
        try:
            rows, _, _ = turn._load_turn_history(agent, False)
        finally:
            api_execution.reset(token)
        loaded[rerun] = [r["content"] for r in rows], getattr(agent, "_pending_cli_user_message", None)
    rows, adopted = loaded[True]
    assert rows == ["earlier", "ok"] and adopted["content"] == DM
    assert loaded[False] == (["earlier", "ok", DM], None)


def test_admitted_retry_preserves_completed_tool_evidence(api, owner):
    """A 429/5xx re-run after a tool completed keeps ``user → assistant(tool_calls) → tool`` in view
    (as on main), so it can't repeat the side effect or answer without its result."""
    from gateway.run_turn_runner import TurnRunner
    from gateway.session_api_turn import api_execution, prepare_api_execution
    from gateway.turn_context import TurnContext
    from tools.bot_failure_reasons import RETRY_RESUME, result_retry_action

    sid = "retry-completed-tool"
    message = "Create exactly one draft and return its ID."
    owner.db.create_session(sid, "api_server")
    owner.db.append_message(sid, "user", content=message)
    owner.db.append_message(sid, "assistant", content=None, tool_calls=[
        {"id": "call_1", "type": "function", "function": {"name": "create_draft", "arguments": "{}"}}])
    owner.db.append_message(sid, "tool", content="Created draft 73.", tool_call_id="call_1")
    # Realistic state after the next provider request exhausts its retries.
    assert owner.db.latest_conversation_role(sid) == "tool"
    assert result_retry_action({"failed": True, "failure_reason": "rate_limit",
                                "error": "429 Too Many Requests"}) == RETRY_RESUME

    _, ref, row = admit_api_turn(api, session_id=sid, user_message=message,
                                 conversation_history=owner.db.get_messages_as_conversation(sid),
                                 resume_unanswered_turn=True)
    turn = TurnRunner(SimpleNamespace(session_authority=None),
                      TurnContext(message=message, session_id=sid, session_key="api"))
    token = api_execution.set(prepare_api_execution(owner, ref, row["payload"]))
    try:
        history, _, _ = turn._load_turn_history(SimpleNamespace(), False)
    finally:
        api_execution.reset(token)
    assert any(m.get("role") == "tool" and m.get("content") == "Created draft 73." for m in history)

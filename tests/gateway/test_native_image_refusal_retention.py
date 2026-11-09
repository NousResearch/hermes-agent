"""Native refusal artifacts survive actual managed/TUI owner retention and a DB reopen."""
import asyncio
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from agent.turn_failure_copy import CONTENT_REFUSED_METADATA_KEY, CONTENT_REFUSED_NOTICE
from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import AsyncSessionStore, SessionSource, SessionStore
from gateway.session_managed_worker import _finalize_managed_refusal
from hermes_state import SessionDB
from tests.agent.native_responses_image_support import image_item, media_paths, response, text_item

pytest_plugins = ("tests.agent.native_responses_image_support",)


@pytest.mark.parametrize("owner", ["managed", "tui"])
@pytest.mark.parametrize("held,scenario", [
    (False, "current"), (True, "current"), (False, "prior"), (True, "prior"), (False, "delivered"),
])
@pytest.mark.parametrize("rewrite", ["noop", "replace", "failure"])
def test_native_notice_survives_real_owner_and_reopen(make_agent, monkeypatch, tmp_path, owner, held, scenario, rewrite):
    from agent.turn_finalizer import HELD_TRANSFORM_FAILED_REPLY

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    entry = store.get_or_create_session(SessionSource(platform=Platform.TELEGRAM, chat_id="refusal-owner", user_id="u"))
    db = store._db_for_session_id(entry.session_id)
    agent = make_agent(native=True)
    agent.session_id, agent._session_db = entry.session_id, db
    agent._ensure_db_session()
    deliveries, seen = [], []
    if scenario == "delivered":
        agent.interim_assistant_callback = lambda text, **kw: deliveries.append(text)

    def transform(**kw):
        seen.append(kw["response_text"])
        if rewrite == "failure":
            raise RuntimeError("synthetic transform failure")
        return "Reviewed retained refusal." if rewrite == "replace" else None

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda name, **kw: (
        [transform(**kw)] if name == "transform_llm_output" else []
    ))
    monkeypatch.setattr("agent.conversation_loop.held_stream_transforms", lambda: (transform,) if held else ())
    blocked = response(text_item("Synthetic refusal."), *([image_item()] if scenario == "current" else []),
                       status="incomplete", response_id="refused")
    blocked.incomplete_details = NS(reason="content_filter")
    pending = iter(([response(image_item(), status="incomplete")] if scenario != "current" else []) + [blocked])
    monkeypatch.setattr(agent, "_interruptible_api_call", lambda *a, **k: next(pending))
    monkeypatch.setattr(agent, "_interruptible_streaming_api_call", lambda *a, **k: next(pending))
    try:
        result = agent.run_conversation("Synthetic refused image request.")
        assert result["failure_reason"] == "content_policy_blocked" and result["messages"] == []
        if owner == "managed":
            runner = object.__new__(GatewayRunner)
            runner.session_store, runner._async_session_store = store, AsyncSessionStore(store)
            runner._session_db = db
            runner._evict_cached_agent = lambda *a, **k: None
            from agent.managed_worker import accept_result, result_frame
            accepted, _ = accept_result(result_frame(result))
            asyncio.run(_finalize_managed_refusal(NS(runner=runner), agent.session_id, accepted))
        else:
            import tui_gateway.server as srv
            monkeypatch.setattr(srv, "render_message", lambda *a: None)
            session = {"agent": agent, "pending_title": None, "session_key": entry.session_key,
                       "history_lock": nullcontext()}
            st = srv._TurnRun(agent, None, None, False, history=[], result=result,
                              prompt_text="Synthetic refused image request.")
            payload, _, _ = srv._complete_turn_payload(session, st, None, 80)
            assert "refused_turn_artifacts" not in payload
            srv._event_frame("message.complete", "sid", payload)
        assert db.get_messages_as_conversation(agent.session_id) == []
        display = db.get_messages_as_conversation(agent.session_id, include_compacted=True)
        notice = display[-1]["display_metadata"][CONTENT_REFUSED_METADATA_KEY]["notice"]
        expected = (HELD_TRANSFORM_FAILED_REPLY if held and rewrite == "failure"
                    else "Reviewed retained refusal." if rewrite == "replace" else result["final_response"])
        assert expected in notice, "The real owner discarded the permitted native final"
        paths = media_paths(notice)
        visible_image = scenario == "delivered" or (rewrite != "replace" and not (held and rewrite == "failure"))
        assert len(paths) == int(visible_image), "Durable display must preserve allowed refs only"
        assert all(p.is_file() for p in paths)
        assert len(seen) == 1
        if scenario == "delivered":
            assert len([t for t in deliveries if media_paths(t)]) == 1
            assert not media_paths(result["final_response"]), "Do not resend an already-delivered image"
        reopened = SessionDB(db.db_path)
        try:
            assert reopened.get_messages_as_conversation(agent.session_id) == []
            restored = reopened.get_messages_as_conversation(agent.session_id, include_compacted=True)
            assert restored[-1]["display_metadata"][CONTENT_REFUSED_METADATA_KEY]["notice"] == notice
        finally:
            reopened.close()
    finally:
        db.close()


def test_no_image_managed_refusal_keeps_the_generic_notice(tmp_path):
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    entry = store.get_or_create_session(SessionSource(platform=Platform.TELEGRAM, chat_id="no-image", user_id="u"))
    db = store._db_for_session_id(entry.session_id)
    runner = object.__new__(GatewayRunner)
    runner.session_store, runner._async_session_store = store, AsyncSessionStore(store)
    runner._session_db = db
    runner._evict_cached_agent = lambda *a, **k: None
    try:
        boundary = db.append_message(entry.session_id, "user", "Synthetic refused request.")
        result = {"failure_reason": "content_policy_blocked", "final_response": "Raw provider refusal detail.",
                  "refused_turn_boundary_row_id": boundary}
        asyncio.run(_finalize_managed_refusal(NS(runner=runner), entry.session_id, result))
        display = db.get_messages_as_conversation(entry.session_id, include_compacted=True)
        assert display[-1]["display_metadata"][CONTENT_REFUSED_METADATA_KEY]["notice"] == CONTENT_REFUSED_NOTICE
    finally:
        db.close()

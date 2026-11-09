"""A muted wake keeps durable evidence while hiding only its new presentation rows."""
from types import SimpleNamespace

from agent.notification_presentation import notification_turn
from agent.session_persistence import _db_flush_collect
from hermes_state import SessionDB
from tui_gateway.server import _history_to_messages


def test_muted_wake_preserves_history_and_new_evidence(tmp_path):
    old = {"role": "user", "content": "human request"}
    diagnostic = {"role": "user", "content": "worker failed"}
    reply = {"role": "assistant", "content": "model diagnostic echo"}
    messages = [old, diagnostic, reply]
    agent = SimpleNamespace(session_id="session", _last_flushed_db_idx=0)
    with notification_turn(agent, muted=True):
        rows, _ = _db_flush_collect(agent, messages, [old])
    assert "display_kind" not in old
    assert [row["content"] for row in rows] == ["worker failed", "model diagnostic echo"]
    assert all(row["display_kind"] == "hidden" for row in rows)
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("session", source="tui")
        db.append_messages_batch("session", rows)
        stored = db.get_messages("session")
    assert [row["content"] for row in stored] == ["worker failed", "model diagnostic echo"]
    assert _history_to_messages(stored) == []
    next_result = {"role": "assistant", "content": "requested result"}
    rows, _ = _db_flush_collect(agent, [next_result], [])
    assert rows[0]["display_kind"] is None


# ---------------------------------------------------------------------------
# A committed turn's persist receipt: agent flag -> finalize result ->
# SERVER-GENERATED X-Hermes-Turn-Persisted response header.
# ---------------------------------------------------------------------------


class _GovF25cAgent:
    """A finalize_turn-shaped agent (mirrors the finalize test harness) that also drives the REAL
    session-persistence funnel for its per-turn persist flag. ``persist`` invokes the unbound
    ``SessionPersistenceMixin._persist_session`` so the flag is set by the source line under test
    (not by this stub); ``begin_turn`` mirrors the turn-facade per-turn reset. finalize_turn clears the
    flag to None before its final persist, so the simplified ``_persist_session`` capture below
    mirrors the real funnel and sets the flag from ``self._committed`` — the value the final persist
    writes (or None if that step raises before it) is what finalize exposes."""

    def __init__(self):
        self.max_iterations = 90
        self.iteration_budget = SimpleNamespace(remaining=10, used=1, max_total=90)
        self.quiet_mode = True
        self.model = "test-model"
        self.provider = "test-provider"
        self.base_url = ""
        self.session_id = "sess-test"
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0)
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_estimated_cost_usd = 0
        self.session_cost_status = "unknown"
        self.session_cost_source = "test"
        self._tool_guardrail_halt_decision = None
        self._interrupt_message = None
        self._response_was_previewed = True
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self.valid_tool_names = []
        self.persisted_messages = None
        self._persist_user_message_idx = None
        self._persist_user_message_override = None
        self._persist_user_message_timestamp = None
        # persist-funnel seams (real _persist_session consumes these)
        self._session_db = None
        self._persist_disabled = True
        self._inflight_turn_id = None
        self._inflight_turn_session_id = None
        self._committed = None

    # ---- finalize_turn harness ----
    def _handle_max_iterations(self, messages, api_call_count):
        raise AssertionError("not expected")

    def _emit_status(self, *_a, **_k):
        pass

    def _safe_print(self, *_a, **_k):
        pass

    def _save_trajectory(self, *_a, **_k):
        pass

    def _cleanup_task_resources(self, *_a, **_k):
        pass

    def _drop_trailing_empty_response_scaffolding(self, messages):
        pass

    def _persist_session(self, messages, conversation_history):
        self.persisted_messages = [dict(m) for m in messages]
        # Mirror the real funnel: the final flush records the per-turn receipt. finalize_turn resets
        # the flag to None before reaching here, so if this step is never called (an earlier
        # finalize sub-step raised) the receipt stays None instead of a stale True.
        self._last_turn_persisted = bool(self._committed)

    def _apply_persist_user_message_override(self, messages):
        idx = self._persist_user_message_idx
        override = self._persist_user_message_override
        if idx is not None and override is not None:
            messages[idx]["content"] = override

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **_k):
        pass

    # ---- persist-receipt seams under test ----
    def _flush_messages_to_session_db(self, messages, conversation_history):
        return self._committed

    def begin_turn(self):
        # mirrors the turn-facade per-turn reset (run_conversation sets this None each turn)
        self._last_turn_persisted = None

    def persist(self, *, committed):
        from agent.session_persistence import SessionPersistenceMixin
        self._committed = committed
        SessionPersistenceMixin._persist_session(self, [], [])


def _finalize(turn_finalizer, agent):
    return turn_finalizer.finalize_turn(
        agent, final_response="Done.", api_call_count=1, interrupted=False, failed=False,
        messages=[{"role": "user", "content": "do it"}], conversation_history=[],
        effective_task_id="task", turn_id="turn", user_message="do it",
        original_user_message="do it", _should_review_memory=False,
        _turn_exit_reason="fallback_prior_turn_content")


def _post_chat(*, persisted, request_headers):
    """POST the REAL /v1/chat/completions non-stream route with _run_agent mocked to return a result
    carrying turn_persisted=<persisted>; return (status, response_headers)."""
    import asyncio
    from unittest.mock import AsyncMock, patch

    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import (
        APIServerAdapter, cors_middleware, security_headers_middleware,
    )

    async def _drive():
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
        mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
        app = web.Application(middlewares=mws)
        app["api_server_adapter"] = adapter
        app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "ok", "messages": [], "turn_persisted": persisted},
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    "/v1/chat/completions", headers=request_headers or {},
                    json={"model": "hermes-agent",
                          "messages": [{"role": "user", "content": "hi"}], "stream": False})
                await resp.read()
                return resp.status, dict(resp.headers)

    return asyncio.run(_drive())


def test_ac_gov_f25_c_5():
    """A committed turn sets _last_turn_persisted True; finalize_turn exposes turn_persisted; the chat route emits a SERVER-GENERATED X-Hermes-Turn-Persisted, never reflected."""
    from agent import turn_finalizer

    agent = _GovF25cAgent()
    agent.begin_turn()
    assert agent._last_turn_persisted is None
    agent.persist(committed=True)
    assert agent._last_turn_persisted is True
    assert _finalize(turn_finalizer, agent)["turn_persisted"] is True

    agent.begin_turn()
    agent.persist(committed=False)
    assert agent._last_turn_persisted is False
    assert _finalize(turn_finalizer, agent)["turn_persisted"] is False

    # server-generated response header, NOT reflected from a forged request header
    _status_ok, headers_ok = _post_chat(persisted=True,
                                         request_headers={"X-Hermes-Turn-Persisted": "true"})
    assert headers_ok["X-Hermes-Turn-Persisted"] == "true"
    _status_forged, headers_forged = _post_chat(persisted=False,
                                                 request_headers={"X-Hermes-Turn-Persisted": "true"})
    assert headers_forged["X-Hermes-Turn-Persisted"] == "false"  # server ignores the forged request value


def test_final_persist_failure_clears_stale_receipt():
    """If the final persist step raises after an earlier mid-turn flush committed, finalize reports
    turn_persisted None (not a stale True) — a wake caller must never ack an uncommitted turn."""
    from agent import turn_finalizer

    agent = _GovF25cAgent()
    agent.begin_turn()
    agent.persist(committed=True)
    assert agent._last_turn_persisted is True

    def _boom(*_a, **_k):
        raise RuntimeError("final persist failed")

    agent._persist_session = _boom
    assert _finalize(turn_finalizer, agent)["turn_persisted"] is None

"""Catalog resolution must stay dynamic without translating cancellation wire metadata."""
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent import i18n
from agent.turn_failure_copy import failed_turn_notice, interrupted_waiting_for_model
from agent.turn_retry_state import TurnRetryState
from agent.turn_truncation import recover_from_truncation


@pytest.fixture(autouse=True)
def reset_catalogs():
    i18n.reset_language_cache()
    yield
    i18n.reset_language_cache()


@pytest.mark.parametrize("lang", ["zh", "ja", "fr"])
def test_real_catalog_notices_and_cancellation_consumers(monkeypatch, lang):
    from acp_adapter.server import HermesACPAgent
    from acp_adapter.session import SessionState
    from gateway.run import _sanitize_gateway_final_response
    from tui_gateway import server as tui_server  # registers prompt-turn runtime bindings
    _turn_outcome = tui_server._turn_outcome

    monkeypatch.setenv("HERMES_LANGUAGE", lang)
    assert failed_turn_notice([]) == i18n.t("turn_failure.notice", lang=lang)
    assert failed_turn_notice([]) != i18n.t("turn_failure.notice", lang="en")
    # Waiting metadata is deliberately a stable protocol string, not localized prose.
    sentinel = interrupted_waiting_for_model(1.7)
    assert sentinel != i18n.t("turn_failure.interrupted_waiting", elapsed="1.7", lang=lang)
    assert _sanitize_gateway_final_response("telegram", sentinel + "\ud800") == ""
    result = {"final_response": sentinel, "interrupted": True}
    assert _turn_outcome(result)[:2] == ("", "interrupted")

    server = HermesACPAgent.__new__(HermesACPAgent)
    server._send_usage_update = AsyncMock()
    server._drain_queued_prompts = AsyncMock()
    state = SessionState(session_id="s", agent=SimpleNamespace(session_id="h"),
                         history=[], cancel_event=None, is_running=True,
                         queued_prompts=[], runtime_lock=threading.Lock())
    conn = SimpleNamespace(session_update=AsyncMock())
    asyncio.run(server._finish_turn(state, "s", conn, result, "h", False))
    conn.session_update.assert_not_called()
    # Partial/ordinary prose is not discarded merely because the turn was interrupted.
    assert _turn_outcome({"final_response": "partial answer", "interrupted": True})[0] == "partial answer"


@pytest.mark.parametrize("path", ["first", "rollback", "tool"])
def test_truncation_resolves_language_after_import(monkeypatch, path):
    # The module is already imported above; each recovery must use the current catalog.
    for lang in ("en", "zh", "ja", "en"):
        monkeypatch.setenv("HERMES_LANGUAGE", lang)
        i18n.reset_language_cache()
        agent = MagicMock()
        agent.api_mode = "chat_completions" if path == "tool" else "unsupported"
        agent.provider = "openrouter"
        agent.model = "test/model"
        agent.context_length = 128000
        response = SimpleNamespace(id="normal", usage=None, choices=[SimpleNamespace(
            message=SimpleNamespace(content="some text", tool_calls=[SimpleNamespace(
                function=SimpleNamespace(name="terminal", arguments="{incomplete"))] if path == "tool" else None),
            finish_reason="length")])
        messages = [{"role": "user", "content": "go"}]
        if path == "rollback":
            messages.append({"role": "assistant", "content": "previous answer"})
        verdict = recover_from_truncation(
            agent, response, "length", TurnRetryState(), messages=messages,
            conversation_history=None, api_kwargs={}, api_call_count=1,
            effective_task_id="test", current_turn_user_idx=0, length_continue_retries=0,
            truncated_response_parts=[], truncated_tool_call_retries=4, retry_count=0,
            compression_attempts=0,
        )
        assert verdict.result["final_response"] == i18n.t("turn_failure.site.truncated", lang=lang)

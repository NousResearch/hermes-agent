"""731: a content-policy refusal hands the caller the CLEAN pre-turn history, never the compaction-
poisoned ``conversation_history``.

After an in-place compaction this turn, the loop's ``conversation_history`` flush baseline is
rewritten to ``list(messages)`` — the WHOLE working set, including the refused request and its tool
rounds. Returning that as ``result["messages"]`` would re-poison the CLI/ACP/TUI in-memory history,
so the next prompt re-trips the deterministic safety filter even after the durable transcript is
cleaned. The refusal handlers must rebuild the clean history from the real turn boundary and carry
the refused run's durable boundary row for the owner to drop.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

from hermes_agent.agent.context_compressor import _DB_PERSISTED_MARKER
from hermes_agent.agent.turn_retry_state import TurnRetryState
from hermes_agent.agent.turn_truncation import handle_content_policy_refusal


def _agent():
    from hermes_agent.agent.transports import get_transport

    agent = MagicMock()
    agent.api_mode, agent.provider, agent.model, agent.log_prefix = "anthropic_messages", "anthropic", "claude", ""
    agent._is_anthropic_oauth = False
    agent._get_transport.return_value = get_transport("anthropic_messages")
    agent._has_pending_fallback.return_value = False
    agent._try_activate_fallback.return_value = False
    agent._extract_reasoning.return_value = ""
    agent._session_db = None
    return agent


def _refusal_response():
    return SimpleNamespace(
        content=[], stop_reason="refusal", usage=None, model="claude",
        stop_details={"type": "refusal", "category": "general_harms", "explanation": "nope"})


def test_refusal_returns_clean_pre_turn_history_despite_a_poisoned_baseline():
    # The refused turn (user ask + its tool round) starts at index 2.
    refused_user = {"role": "user", "content": "the request the filter blocked",
                    "_row_id": 55, _DB_PERSISTED_MARKER: True}
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        refused_user,
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function",
         "function": {"name": "shell", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "ran"},
    ]
    # In-place compaction rewrote the flush baseline to the whole poisoned working set.
    poisoned_history = list(messages)

    verdict = handle_content_policy_refusal(
        _agent(), _refusal_response(), TurnRetryState(), thinking_spinner=None, messages=messages,
        api_messages=[], api_kwargs={}, active_system_prompt=None, conversation_history=poisoned_history,
        api_call_count=1, effective_task_id="t", turn_id="u", api_request_id="r", api_start_time=0.0,
        retry_count=0, max_retries=0, current_turn_user_idx=2)

    assert verdict.action == "return"
    # The returned history is the pre-turn boundary: the refused ask + tool round are dropped.
    assert verdict.result["messages"] == messages[:2]
    assert all(m.get("content") != "the request the filter blocked" for m in verdict.result["messages"])
    # The durable boundary row rides the result so the owner drops the EXACT refused run.
    assert verdict.result["refused_turn_boundary_row_id"] == 55


def test_refusal_without_a_turn_index_falls_back_to_the_conversation_history():
    # No current_turn_user_idx (e.g. a runtime that doesn't thread it): fall back to the baseline list.
    history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "ok"}]
    verdict = handle_content_policy_refusal(
        _agent(), _refusal_response(), TurnRetryState(), thinking_spinner=None, messages=[],
        api_messages=[], api_kwargs={}, active_system_prompt=None, conversation_history=history,
        api_call_count=1, effective_task_id="t", turn_id="u", api_request_id="r", api_start_time=0.0,
        retry_count=0, max_retries=0)
    assert verdict.action == "return"
    assert verdict.result["messages"] == history
    assert "refused_turn_boundary_row_id" not in verdict.result

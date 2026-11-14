"""messages.token_count carries each response's own completion figure (#136851).

The column shipped with the schema but no runtime path ever populated it. The figure is
bound where the row's source object is produced (``normalize_response_for_agent``), not in
agent state: truncation recovery builds its rows BEFORE the usage fold runs, so an
agent-level stash is one response behind. ``build_assistant_message`` stamps the row from
that object and ``_db_flush_row`` carries it into state.db (the db layer has always read
it). ``None`` means the provider reported no usage (unknown) and leaves the column NULL;
a reported ``0`` is a real figure and is stamped.
"""
import time
from types import SimpleNamespace


def _agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from run_agent import AIAgent
    return AIAgent(api_key="k", base_url="https://inference-api.nousresearch.com/v1", provider="nous",
                   api_mode="chat_completions", model="anthropic/claude-fable-5.1", session_id="t",
                   platform="cli", quiet_mode=True, skip_context_files=True, skip_memory=True,
                   save_trajectories=False, enabled_toolsets=["file"])


def _wire_response(completion, *, finish_reason="stop", content="hello", prompt=62_889, cached=34_283):
    """A Chat Completions-shaped provider response; ``completion=None`` means NO usage
    reported at all, while ``completion=0`` is an explicitly reported zero."""
    usage = None if completion is None else SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached, cache_write_tokens=0),
        completion_tokens_details=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(
            index=0, finish_reason=finish_reason,
            message=SimpleNamespace(role="assistant", content=content, tool_calls=None,
                                    reasoning=None, reasoning_content=None,
                                    reasoning_details=None, model_extra={}),
        )],
        usage=usage, id="gen-test", model="anthropic/claude-fable-5.1",
    )


def _build_row(agent, response, finish_reason="stop"):
    from agent.turn_truncation import normalize_response_for_agent
    return agent._build_assistant_message(
        normalize_response_for_agent(agent, response), finish_reason
    )


def _check_response(agent, response, messages, *, api_call_count=1):
    """Drive the real post-call entry point (response check → refusal/truncation → usage)."""
    from agent.turn_response_check import check_api_response
    from agent.turn_retry_state import TurnRetryState

    return check_api_response(
        agent, response=response, _retry=TurnRetryState(), thinking_spinner=None,
        messages=messages, api_messages=[], api_kwargs={}, active_system_prompt=None,
        conversation_history=[], finish_reason=None, retry_count=0, max_retries=3,
        compression_attempts=0, max_compression_attempts=3, length_continue_retries=0,
        truncated_response_parts=[], truncated_tool_call_retries=0, current_turn_user_idx=0,
        api_call_count=api_call_count, api_request_id=f"req-{api_call_count}",
        api_start_time=time.time(), effective_task_id=None, turn_id="turn-1",
        _preflight_compression_blocked=False, _last_preflight_pressure=None,
    )


def _interim_rows(messages):
    return [m for m in messages if m.get("_length_continuation_fragment")]


def test_each_row_binds_to_its_own_response_in_the_real_chain(tmp_path, monkeypatch):
    """Response A (1,204) then a truncated response B (777): B's interim row is built
    inside recovery, BEFORE the usage fold, and must still carry 777 — not A's figure."""
    a = _agent(tmp_path, monkeypatch)
    messages = [{"role": "user", "content": "hi"}]
    try:
        resp_a = _wire_response(1_204, content="first answer")
        assert _check_response(a, resp_a, messages).action == "break"
        row_a = _build_row(a, resp_a)

        resp_b = _wire_response(777, finish_reason="length", content="truncated answer")
        assert _check_response(a, resp_b, messages, api_call_count=2).action == "break"

        # Session billing is untouched: only the completed call was folded. The
        # truncated reply was not billed before this fix either.
        assert a.session_api_calls == 1
        assert a.session_completion_tokens == 1_204
    finally:
        a.close()

    assert row_a["token_count"] == 1_204
    assert [m["token_count"] for m in _interim_rows(messages)] == [777]


def test_truncated_response_without_usage_stays_unknown(tmp_path, monkeypatch):
    """A usage-less truncated response must not inherit the previous response's figure:
    the row is stamped unknown (column NULL), not A's number and not 0."""
    a = _agent(tmp_path, monkeypatch)
    messages = [{"role": "user", "content": "hi"}]
    try:
        resp_a = _wire_response(1_204, content="first answer")
        assert _check_response(a, resp_a, messages).action == "break"
        _build_row(a, resp_a)

        resp_b = _wire_response(None, finish_reason="length", content="truncated answer")
        assert _check_response(a, resp_b, messages, api_call_count=2).action == "break"
    finally:
        a.close()

    interim = _interim_rows(messages)
    assert len(interim) == 1
    assert "token_count" not in interim[0]


def test_first_response_truncated_binds_its_own_usage(tmp_path, monkeypatch):
    """No earlier response exists: the first truncated reply's row carries its own figure."""
    a = _agent(tmp_path, monkeypatch)
    messages = [{"role": "user", "content": "hi"}]
    try:
        resp = _wire_response(999, finish_reason="length", content="cut short")
        assert _check_response(a, resp, messages).action == "break"
    finally:
        a.close()

    assert [m["token_count"] for m in _interim_rows(messages)] == [999]


class _FakeMoaClient:
    def __init__(self, advisor_usage):
        self._advisor_usage = advisor_usage
        self.reference_consumed = 0
        self.trace_flushes = 0
        self.last_aggregator_slot = None

    def consume_reference_usage(self):
        self.reference_consumed += 1
        return self._advisor_usage, None

    def consume_and_save_trace(self, *_args, **_kwargs):
        self.trace_flushes += 1


def test_moa_row_uses_aggregator_output_session_keeps_folded_total(tmp_path, monkeypatch):
    """MoA: the row shows what the aggregator itself produced (300), while the session
    ledger keeps the advisor-folded total (300 + 900) and consumes advisor usage once."""
    from agent.usage_pricing import CanonicalUsage

    a = _agent(tmp_path, monkeypatch)
    moa = _FakeMoaClient(CanonicalUsage(input_tokens=4_000, output_tokens=900))
    original_client = a.client
    a.client = moa
    messages = [{"role": "user", "content": "hi"}]
    try:
        resp = _wire_response(300, content="aggregator answer")
        assert _check_response(a, resp, messages).action == "break"
        row = _build_row(a, resp)
        assert a.session_completion_tokens == 1_200
        assert moa.reference_consumed == 1
        assert moa.trace_flushes == 1
    finally:
        a.client = original_client
        a.close()

    assert row["token_count"] == 300


def test_reported_zero_is_stamped_missing_usage_is_omitted(tmp_path, monkeypatch):
    a = _agent(tmp_path, monkeypatch)
    try:
        zero_row = _build_row(a, _wire_response(0, content="zero tokens"))
        missing_row = _build_row(a, _wire_response(None, content="usage unavailable"))
    finally:
        a.close()

    assert zero_row["token_count"] == 0
    assert "token_count" not in missing_row


def test_row_without_response_metadata_is_not_stamped(tmp_path, monkeypatch):
    """A synthesized message that never came from a provider response (scripted prelude,
    replay shims) has no figure to bind — it must stay unknown, never borrow agent state."""
    a = _agent(tmp_path, monkeypatch)
    try:
        msg = a._build_assistant_message(
            SimpleNamespace(content="hi", tool_calls=None, finish_reason="stop",
                            reasoning=None, reasoning_content=None, reasoning_details=None),
            "stop",
        )
    finally:
        a.close()
    assert "token_count" not in msg


def test_stamp_survives_the_db_flush_row(tmp_path, monkeypatch):
    a = _agent(tmp_path, monkeypatch)
    try:
        msg = _build_row(a, _wire_response(1_204))
        from agent.session_persistence import _db_flush_row
        row = _db_flush_row(a, msg, is_current_turn_user=False)
        user_row = _db_flush_row(a, {"role": "user", "content": "hi", "timestamp": 1.0},
                                 is_current_turn_user=True)
    finally:
        a.close()
    assert row["token_count"] == 1_204
    # rows the stamp never touched stay NULL, exactly as before
    assert user_row["token_count"] is None

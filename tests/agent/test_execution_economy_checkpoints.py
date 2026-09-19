"""Execution-economy checkpoints (issue #110129): opt-in, advisory, never interrupting.

Covers the contract the feature must keep: off by default (byte-identical pipeline), the two
detectors (consecutive single-call rounds, normalized-equivalent repeated calls), the per-turn
notice cap + cooldown, refusal to touch a durable (``_DB_PERSISTED_MARKER``) row, interrupt
safety, multimodal results, per-turn reset, and logs that carry reason/count/tool names only.
"""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from agent.execution_economy import (
    COOLDOWN_ROUNDS,
    MAX_NOTICES_PER_TURN,
    REPEAT_ROUND_THRESHOLD,
    SINGLE_TOOL_ROUNDS,
    SINGLE_TOOL_ROUND_THRESHOLD,
    inject_execution_economy_checkpoint,
    observe_tool_round,
)


def _agent(tmp_path, monkeypatch, enabled):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        f"agent:\n  execution_economy_checkpoints: {str(enabled).lower()}\n", encoding="utf-8"
    )
    from run_agent import AIAgent
    from hermes_state import SessionDB
    from model_tools import _clear_tool_defs_cache
    from tools.registry import invalidate_check_fn_cache

    invalidate_check_fn_cache()
    _clear_tool_defs_cache()
    return AIAgent(session_db=SessionDB(db_path=tmp_path / "proof.db"),
                   model="test-model", provider="openai-compat", api_key="test",
                   base_url="http://127.0.0.1:1/v1", max_iterations=8,
                   quiet_mode=True, skip_context_files=True, skip_memory=True)


@pytest.fixture
def agent(tmp_path, monkeypatch):
    from agent.turn_context import _reset_per_turn_agent_state

    a = _agent(tmp_path, monkeypatch, True)
    _reset_per_turn_agent_state(a)
    yield a
    a._session_db.close()


def _call(name, args, call_id="t1"):
    return SimpleNamespace(
        id=call_id, type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def _round(agent, messages, calls, content="result"):
    """One turn-loop tool round as production runs it: observe, append the result, flush."""
    from agent.tool_executor import _flush_session_db_after_tool_progress

    observe_tool_round(agent, calls)
    messages.append({"role": "tool", "tool_call_id": calls[0].id,
                     "name": calls[0].function.name, "content": deepcopy(content)})
    assert _flush_session_db_after_tool_progress(agent, messages, stage="round") is True
    return messages[-1]


def _single_call_rounds(agent, messages, count, *, args=None, content="result"):
    for i in range(count):
        call_args = args(i) if callable(args) else (args or {"path": f"f{i}.txt"})
        calls = [_call("read_file", call_args, call_id=f"t{i}")]
        _round(agent, messages, calls, content=content)
    return messages


def _persisted_text(agent):
    return " ".join(str(row["content"]) for row in agent._session_db.get_messages(agent.session_id))


def test_disabled_leaves_the_pipeline_untouched(tmp_path, monkeypatch):
    """Off (the default) writes no notice anywhere; on, the same rounds do."""
    from agent.turn_context import _reset_per_turn_agent_state

    for enabled in (False, True):
        home = tmp_path / f"cfg-{enabled}"
        home.mkdir()
        a = _agent(home, monkeypatch, enabled)
        try:
            assert a._execution_economy_checkpoints is enabled
            _reset_per_turn_agent_state(a)
            assert (a._execution_economy is not None) is enabled
            before = (a.iteration_budget.max_total, a.iteration_budget.used)

            messages = []
            _single_call_rounds(a, messages, SINGLE_TOOL_ROUND_THRESHOLD)
            text = str(messages[-1]["content"])
            assert ("[hermes note:" in text) is enabled
            assert ("hermes note" in _persisted_text(a)) is enabled
            # Advisory only: budget, transcript shape and the payload itself are untouched.
            assert (a.iteration_budget.max_total, a.iteration_budget.used) == before
            assert [m["role"] for m in messages] == ["tool"] * SINGLE_TOOL_ROUND_THRESHOLD
            assert text.startswith("result") and "result" in text
        finally:
            a._session_db.close()


def test_single_call_rounds_threshold_then_cooldown_then_cap(agent):
    messages = []
    rounds = 3 * COOLDOWN_ROUNDS + SINGLE_TOOL_ROUND_THRESHOLD
    _single_call_rounds(agent, messages, rounds)

    noticed = [i for i, msg in enumerate(messages, 1) if "hermes note" in str(msg["content"])]
    assert noticed == [SINGLE_TOOL_ROUND_THRESHOLD,
                       SINGLE_TOOL_ROUND_THRESHOLD + COOLDOWN_ROUNDS]
    assert len(noticed) == MAX_NOTICES_PER_TURN  # cap: never more than the per-turn limit
    assert "single call" in str(messages[noticed[0] - 1]["content"])
    assert messages[noticed[1] - 1]["content"].count("hermes note") == 1
    # Nothing else in the transcript moved.
    assert all(m["role"] == "tool" for m in messages)


def test_normalized_equivalent_repeats_fire_repeated_calls(agent):
    """Reformatted JSON and reordered keys are the same normalized call."""
    variants = ['{"path": "a.txt"}', '{ "path": "a.txt" }', '{"path":"a.txt"}']
    assert len(variants) >= REPEAT_ROUND_THRESHOLD
    for i, raw in enumerate(variants):
        call = _call("read_file", {}, call_id=f"t{i}")
        call.function.arguments = raw
        observe_tool_round(agent, [call])
    pending = agent._execution_economy.pending
    assert pending is not None and pending[0] == "repeated_calls"

    messages = [{"role": "tool", "tool_call_id": "t2", "name": "read_file", "content": "same"}]
    from agent.tool_executor import _flush_session_db_after_tool_progress

    assert _flush_session_db_after_tool_progress(agent, messages, stage="round") is True
    text = str(messages[-1]["content"])
    assert "repeated the same normalized call set (read_file)" in text
    assert f"last {REPEAT_ROUND_THRESHOLD} rounds" in text
    assert "same" in text

    # A different round resets the streak: no notice pending without a fresh repetition.
    observe_tool_round(agent, [_call("read_file", {"path": "b.txt"}, call_id="t9"),
                               _call("read_file", {"path": "c.txt"}, call_id="t9")])
    assert agent._execution_economy.pending is None


def test_durable_row_is_never_rewritten_and_stale_pending_is_dropped(agent):
    from agent.context_compressor import _DB_PERSISTED_MARKER

    messages = []
    _single_call_rounds(agent, messages, SINGLE_TOOL_ROUND_THRESHOLD)
    durable = messages[-1]
    assert "hermes note" in str(durable["content"])
    snapshot = deepcopy(durable)

    # A notice that finds only durable rows is refused, not force-written.
    durable[_DB_PERSISTED_MARKER] = True
    agent._execution_economy.pending = (SINGLE_TOOL_ROUNDS, SINGLE_TOOL_ROUND_THRESHOLD, "read_file")
    assert inject_execution_economy_checkpoint(agent, messages) is False
    assert durable == snapshot
    assert durable[_DB_PERSISTED_MARKER] is True

    # The refused notice is dropped at the next round, never carried onto a later result.
    observe_tool_round(agent, [_call("read_file", {"path": "x"}, call_id="tx"),
                               _call("read_file", {"path": "y"}, call_id="ty")])
    assert agent._execution_economy.pending is None


def test_notice_lands_on_the_newest_unpersisted_tool_row(agent):
    """Durable rows and non-tool rows are skipped in favour of the live tail."""
    from agent.context_compressor import _DB_PERSISTED_MARKER

    durable = {"role": "tool", "tool_call_id": "old", "content": "old result",
               _DB_PERSISTED_MARKER: True}
    fresh = {"role": "tool", "tool_call_id": "new", "content": "new result"}
    messages = [durable, {"role": "assistant", "content": "notes"}, fresh]
    agent._execution_economy.pending = (SINGLE_TOOL_ROUNDS, SINGLE_TOOL_ROUND_THRESHOLD, "read_file")

    assert inject_execution_economy_checkpoint(agent, messages) is True
    assert fresh["content"].startswith("new result") and "[hermes note:" in fresh["content"]
    assert durable["content"] == "old result"


def test_interrupt_never_annotates(agent):
    messages = []
    _single_call_rounds(agent, messages, SINGLE_TOOL_ROUND_THRESHOLD - 1)
    agent._interrupt_requested = True
    _round(agent, messages, [_call("read_file", {"path": "f"}, call_id="tf")])
    assert "hermes note" not in str(messages[-1]["content"])
    assert "hermes note" not in _persisted_text(agent)

    # The dropped notice is not carried into a later (post-interrupt) round.
    agent._interrupt_requested = False
    _round(agent, messages, [_call("read_file", {"path": "other"}, call_id="to"),
                             _call("read_file", {"path": "other2"}, call_id="to2")])
    assert "hermes note" not in str(messages[-1]["content"])
    assert "hermes note" not in _persisted_text(agent)


def test_multimodal_result_gets_a_text_block(agent):
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    messages = []
    _single_call_rounds(agent, messages, SINGLE_TOOL_ROUND_THRESHOLD,
                        content=[deepcopy(image)])
    content = messages[-1]["content"]
    assert isinstance(content, list)
    assert content[0] == image  # image block untouched
    assert content[-1]["type"] == "text"
    assert "hermes note" in content[-1]["text"]
    assert "hermes note" in _persisted_text(agent)


def test_state_resets_each_turn(agent):
    from agent.turn_context import _reset_per_turn_agent_state

    messages = []
    _single_call_rounds(agent, messages, SINGLE_TOOL_ROUND_THRESHOLD)
    assert "hermes note" in str(messages[-1]["content"])
    assert agent._execution_economy.notices == 1

    _reset_per_turn_agent_state(agent)  # turn boundary
    state = agent._execution_economy
    assert (state.rounds, state.notices, state.single_tool_streak, state.pending) == (0, 0, 0, None)
    messages = []
    _single_call_rounds(agent, messages, SINGLE_TOOL_ROUND_THRESHOLD)
    assert "hermes note" in str(messages[-1]["content"])  # rearmed next turn


def test_logs_carry_reason_count_and_tool_names_only(agent, caplog):
    secret = "SECRET_QUERY_STRING_abcdef"
    with caplog.at_level("INFO", logger="agent.conversation_loop"):
        messages = []
        _single_call_rounds(agent, messages, SINGLE_TOOL_ROUND_THRESHOLD,
                            args=lambda i: {"query": f"{secret}-{i}"})
    line = "\n".join(record.getMessage() for record in caplog.records)
    assert "execution economy checkpoint appended" in line
    assert "reason=single_tool_rounds" in line
    assert f"count={SINGLE_TOOL_ROUND_THRESHOLD}" in line
    assert "tool=read_file" in line
    assert secret not in line


def test_round_hook_is_wired_into_the_tool_execution_funnel(tmp_path, monkeypatch):
    """Production seam: AIAgent._execute_tool_calls observes every round."""
    from unittest.mock import patch

    from agent.turn_context import _reset_per_turn_agent_state

    a = _agent(tmp_path, monkeypatch, True)
    try:
        _reset_per_turn_agent_state(a)
        messages = []
        with patch("model_tools.handle_function_call", return_value="search result"):
            for i in range(SINGLE_TOOL_ROUND_THRESHOLD):
                calls = [_call("web_search", {"q": f"test-{i}"}, call_id=f"c{i}")]
                messages.append({
                    "role": "assistant", "content": "",
                    "tool_calls": [{"id": f"c{i}", "type": "function",
                                    "function": {"name": "web_search", "arguments": f'{{"q": "test-{i}"}}'}}],
                })
                a._execute_tool_calls(SimpleNamespace(tool_calls=calls), messages, "task-1")
        assert a._execution_economy.rounds == SINGLE_TOOL_ROUND_THRESHOLD
        assert "tool rounds in a row each issued a single call" in str(messages[-1]["content"])
        assert "hermes note" in _persisted_text(a)
        assert "search result" in str(messages[-1]["content"])
    finally:
        a._session_db.close()

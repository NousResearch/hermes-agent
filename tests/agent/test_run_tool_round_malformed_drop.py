"""Polarity tests for the run_tool_round malformed-args drop (RCA 2026-09-17).

A sentinel-args call must be error-resulted and NOT executed; valid calls in
the same batch must still execute. Mirrors the mixed-invalid-batch pattern.
"""
import json
from types import SimpleNamespace

import pytest


class _StubAgent:
    quiet_mode = True
    verbose_logging = False
    valid_tool_names = {"memory", "terminal"}
    session_id = "test-session"

    def _vprint(self, *_a, **_k):
        pass

    def _deduplicate_tool_calls(self, tcs):
        return tcs

    def _cap_delegate_task_calls(self, tcs):
        return tcs

    def _flush_messages_to_session_db(self, *a, **k):
        return True

    def _emit_interim_assistant_message(self, *_a, **_k):
        pass

    def _touch_activity(self, *_a, **_k):
        pass

    def _vprint(self, *_a, **_k):
        pass

    stream_delta_callback = None
    _incremental_persistence_failed = False
    _tool_guardrail_halt_decision = None

    class _Budget:
        def refund(self):
            pass

    iteration_budget = _Budget()

    def _stream_needs_break(self):  # attribute set by the loop; harmless callable default
        return None

    def _execute_tool_calls(self, assistant_message, messages, effective_task_id, api_call_count):
        for tc in assistant_message.tool_calls:
            self._executed.append(tc.function.name)

    _executed = []


def _tc(name, arguments, cid="call_1"):
    return SimpleNamespace(
        id=cid, type="function", extra_content=None,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _sentinel(name="memory", cid="call_1"):
    return _tc(name, json.dumps({"__hermes_malformed_tool_arguments__": True}), cid)


@pytest.fixture
def harness(monkeypatch):
    from agent import turn_tool_round as ttr

    agent = _StubAgent()
    agent._executed = []

    monkeypatch.setattr(ttr, "validate_tool_calls", lambda *a, **k: SimpleNamespace(action="proceed", mixed_invalid_batch=False))
    monkeypatch.setattr(ttr, "stage_tool_call_message", lambda agent, **k: (k["assistant_message"], False))
    monkeypatch.setattr(ttr, "append_message", lambda msgs, m: msgs.append(m))
    def _compress_stub(agent, **k):
        return SimpleNamespace(
            messages=k.get("messages"), active_system_prompt=k.get("active_system_prompt"),
            conversation_history=k.get("conversation_history"), compression_attempts=k.get("compression_attempts"),
            final_response=k.get("final_response"), turn_exit_reason="unknown", end_turn=False,
            current_turn_user_idx=k.get("current_turn_user_idx", 0),
        )

    monkeypatch.setattr(ttr, "compress_after_tool_results", _compress_stub)
    return agent, ttr


def _run_round(agent, tcs, messages, ttr):
    verdict = ttr.run_tool_round(
        agent,
        assistant_message=SimpleNamespace(tool_calls=list(tcs)),
        finish_reason="tool_calls",
        messages=messages,
        conversation_history=[],
        api_call_count=1,
        effective_task_id="t",
        user_message={},
        system_message={},
        active_system_prompt=None,
        compression_attempts=0,
        max_compression_attempts=3,
        final_response=None,
        failed=False,
        _turn_exit_reason="unknown",
        truncated_tool_call_retries=0,
        current_turn_user_idx=0,
        )
    return verdict


def test_sentinel_dropped_and_errored(harness):
    agent, ttr = harness
    messages = []
    _run_round(agent, [_sentinel()], messages, ttr)
    tool_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "NOT executed" in tool_msgs[0]["content"]
    assert "Re-issue" in tool_msgs[0]["content"]
    assert agent._executed == [], f"sentinel call was executed: {agent._executed}"


def test_sentinel_and_valid_batch_valid_survives(harness):
    agent, ttr = harness
    messages = []
    good = _tc("terminal", json.dumps({"command": "ls"}), cid="call_2")
    _run_round(agent, [_sentinel(cid="call_1"), good], messages, ttr)
    tool_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"]
    assert len(tool_msgs) == 1  # the sentinel error result
    assert "NOT executed" in tool_msgs[0]["content"]
    assert agent._executed == ["terminal"], f"expected valid call to run, got {agent._executed}"


def test_normal_empty_object_args_still_execute(harness):
    agent, ttr = harness
    messages = []
    _run_round(agent, [_tc("memory", "{}")], messages, ttr)
    assert agent._executed == ["memory"], "legit empty-object args must still execute"

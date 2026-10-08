"""Post-gate footer seam: ``append_turn_footer`` + context-usage kwargs.

Invariants under test:
- the gate (``before_turn_end``) evaluates the transformed body WITHOUT the footer;
- the footer hook fires after the gate, once per turn, with the gate's evaluated outcome for
  that exact text, and may only append;
- delivered text == in-memory row == SQLite row (#44239);
- no subscriber => byte-identical behaviour; a crashing footer fails open.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import plugins
from hermes_state import SessionDB
from run_agent import AIAgent


def _fake_completion(texts, prompt_tokens=101_000):
    texts = list(texts)

    def create(**kwargs):
        text = texts.pop(0) if len(texts) > 1 else texts[0]
        msg = SimpleNamespace(content=text, tool_calls=None, reasoning=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=5,
                                  total_tokens=prompt_tokens + 5),
            model="fake/model",
        )
    return create


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir()
    db = SessionDB(db_path=tmp_path / ".hermes" / "state.db")
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", model="fake/model",
            quiet_mode=True, skip_context_files=True, skip_memory=True, platform="telegram",
            session_id="sess-footer", session_db=db,
        )
    agent.client = MagicMock()
    agent._intent_ack_continuation = False
    agent._stall_guards = False
    manager = plugins.PluginManager()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    yield agent, db, manager
    db.close()


def _stored(db):
    return [r["content"] for r in db.get_messages("sess-footer") if r["role"] == "assistant"]


def test_footer_appends_after_gate_with_exact_outcome_and_persists_identically(env):
    agent, db, manager = env
    agent.context_compressor.threshold_tokens = 128_000
    events = []

    def transform(response_text, **kw):
        events.append(("transform", response_text, kw.get("prompt_tokens"), kw.get("context_threshold_tokens")))
        return response_text + " [t]"

    def gate(final_response, turn_id, **_):
        events.append(("gate", final_response))
        return {"gate_outcome": {"action": "deliver", "claims": {"reproduced": 3}, "block_count": 0,
                                 "escalation_reason": None}}

    def footer(response_text, turn_id, gate_outcome, prompt_tokens, context_threshold_tokens, **_):
        events.append(("footer", response_text, gate_outcome, prompt_tokens, context_threshold_tokens))
        return "\n\n-- footer"

    manager._hooks["transform_llm_output"] = [transform]
    manager._hooks["before_turn_end"] = [gate]
    manager._hooks["append_turn_footer"] = [footer]
    agent.client.chat.completions.create = _fake_completion(["Body."])

    result = agent.run_conversation("hi")

    assert [e[0] for e in events] == ["transform", "gate", "footer"]
    # Gate judged the transformed body only; the footer never reached the gate.
    assert events[1] == ("gate", "Body. [t]")
    assert events[2][1] == "Body. [t]"
    assert events[2][2] == {"action": "deliver", "claims": {"reproduced": 3}, "block_count": 0,
                            "escalation_reason": None}
    # Real compressor figures: the reading it compares, and its computed threshold.
    assert events[2][3] == agent.context_compressor.last_prompt_tokens == 101_000
    assert events[2][4] == agent.context_compressor.threshold_tokens == 128_000
    assert result["final_response"] == "Body. [t]\n\n-- footer"
    last = next(m for m in reversed(result["messages"]) if m.get("role") == "assistant")
    assert last["content"] == result["final_response"]
    assert _stored(db) == [result["final_response"]]
    # Streaming surfaces must deliver the suffix (gateway edits/sends on response_transformed).
    assert result["response_transformed"] is True
    assert result["pre_transform_response"] == "Body."


def test_transform_kwargs_carry_context_usage_without_footer_subscriber(env):
    agent, db, manager = env
    seen = {}
    manager._hooks["transform_llm_output"] = [lambda response_text, **kw: seen.update(kw)]
    agent.client.chat.completions.create = _fake_completion(["Body."], prompt_tokens=4242)
    agent.run_conversation("hi")
    assert seen["prompt_tokens"] == 4242
    assert seen["context_threshold_tokens"] == agent.context_compressor.threshold_tokens > 0


def test_rework_fires_footer_once_with_final_candidates_outcome(env):
    agent, db, manager = env
    gate_calls, footer_calls = [], []

    def gate(final_response, already_blocked, **_):
        gate_calls.append(final_response)
        if len(gate_calls) == 1:
            return {"action": "block", "message": "fix it",
                    "gate_outcome": {"action": "block", "block_count": 1}}
        return {"gate_outcome": {"action": "deliver", "block_count": 1}}

    def footer(response_text, gate_outcome, **_):
        footer_calls.append((response_text, gate_outcome))
        return "\n-- f"

    manager._hooks["before_turn_end"] = [gate]
    manager._hooks["append_turn_footer"] = [footer]
    agent.client.chat.completions.create = _fake_completion(["draft", "fixed"])

    result = agent.run_conversation("hi")

    assert gate_calls == ["draft", "fixed"]
    assert footer_calls == [("fixed", {"action": "deliver", "block_count": 1})]
    assert result["final_response"] == "fixed\n-- f"
    assert _stored(db) == ["fixed\n-- f"]


def test_finalize_does_not_reaudit_footed_text(env):
    agent, db, manager = env
    gate_calls = []
    manager._hooks["before_turn_end"] = [lambda final_response, **_: gate_calls.append(final_response)]
    manager._hooks["append_turn_footer"] = [lambda **_: "\n-- f"]
    agent.client.chat.completions.create = _fake_completion(["Body."])
    result = agent.run_conversation("hi")
    assert result["final_response"] == "Body.\n-- f"
    assert gate_calls == ["Body."]


def test_no_subscriber_is_byte_identical(env):
    agent, db, manager = env
    agent.client.chat.completions.create = _fake_completion(["Body."])
    result = agent.run_conversation("hi")
    assert result["final_response"] == "Body."
    assert result["response_transformed"] is False
    assert _stored(db) == ["Body."]


def test_crashing_or_blank_footer_fails_open(env):
    agent, db, manager = env

    def boom(**_):
        raise RuntimeError("footer crashed")

    manager._hooks["append_turn_footer"] = [boom, lambda **_: "   "]
    agent.client.chat.completions.create = _fake_completion(["Body."])
    result = agent.run_conversation("hi")
    assert result["final_response"] == "Body."
    assert _stored(db) == ["Body."]


def test_gate_outcome_absent_without_gate_and_non_json_outcome_ignored(env):
    agent, db, manager = env
    outcomes = []
    manager._hooks["append_turn_footer"] = [lambda gate_outcome, **_: outcomes.append(gate_outcome)]
    agent.client.chat.completions.create = _fake_completion(["Body."])
    agent.run_conversation("hi")
    manager._hooks["before_turn_end"] = [lambda **_: {"gate_outcome": {"bad": object()}}]
    agent.run_conversation("again")
    assert outcomes == [None, None]


def test_recovery_path_footer_gets_no_gate_outcome_and_persists(env, monkeypatch):
    from agent.turn_finalizer import finalize_turn
    import json

    agent, db, manager = env
    seen = []
    manager._hooks["append_turn_footer"] = [lambda gate_outcome, **_: seen.append(gate_outcome) or "\n-- f"]
    agent._persist_session = lambda *a, **k: None
    agent._current_turn_id = "turn-r"
    messages = [
        {"role": "user", "content": "do a thing"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"ok": True})},
    ]
    result = finalize_turn(
        agent, final_response="RECOVERED", api_call_count=1, interrupted=False, failed=False,
        messages=messages, conversation_history=None, effective_task_id="task-1", turn_id="turn-r",
        user_message="do a thing", original_user_message="do a thing", _should_review_memory=False,
        _turn_exit_reason="partial_stream_recovery",
    )
    assert seen == [None]
    assert messages[-1]["content"] == "RECOVERED\n-- f"
    assert result["final_response"].startswith("RECOVERED\n-- f")


def test_shell_hooks_refuse_footer_event():
    assert "append_turn_footer" in plugins.VALID_HOOKS
    assert "append_turn_footer" in plugins.SHELL_UNSUPPORTED_HOOKS

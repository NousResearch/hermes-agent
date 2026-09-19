"""Premature-action-promise stop guard (visible-text Telegram regression).

Root cause: a text-only ``finish_reason=stop`` whose text TAILS on an
immediate-action announcement ("Both writes landed. Running the full
verification battery now — …:", "Two real bugs caught … Fixing all of it now.",
"Right … Moving them into the profile … all as writes now, then one
verification run:", "Applying them correctly instead, plus fixing the gate's
artifact-timestamp parsing:") ended the turn with no tool call.
``trailing_continue_intent`` only matches explicit first-person tails
("let me now", "I'll now"), ``_looks_like_codex_intermediate_ack`` requires
``i['’]ll|i will|let me`` (and config ``tool_use_enforcement="auto"`` keeps
it off the chat_completions path), and the kanban guard is kanban-only — the
gerund/imperative announcements match NONE of them.

Fix: a narrow, policy-only sibling guard (agent/promise_stop.py) consulted by
``agent.turn_stop_gates.apply_stop_gates`` AFTER verify-on-stop / pre_verify /
kanban, gated on the ``agent.promise_stop_guard`` agent attribute. Bounded
continuation (max 2 per turn) rides the established nudge pattern: the answer
is kept as the budget-exhaustion fallback, the nudge is the synthetic user
row, and at exhaustion the turn must not present the promise as a clean
completion.

Heuristic limitations are deliberately NOT asserted here beyond the sibling
negatives: this is a conservative surface-shape detector, not semantics.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.turn_stop_gates import apply_stop_gates

from agent.promise_stop import (
    PROMISE_STOP_SYNTHETIC_FLAG,
    build_promise_stop_nudge,
    promise_stop_nudge_text,
)

# Representative Telegram candidates (finish_reason=stop, no tools).
OBSERVED = [
    "Both writes landed. Running the full verification battery now — controls, "
    "regression, and the post-delivery gate against today's real artifacts:",
    "Two real bugs caught in the harness. Fixing all of it now.",
    "Right. The skill belongs in the profile. Moving them into the profile and "
    "the skill, all as writes now, then one verification run:",
    "Applying them correctly instead, plus fixing the gate's artifact-timestamp "
    "parsing (it silently reviewed 0 files):",
]


def _agent(**attrs):
    base = {
        "_verification_stop_nudges": 0,
        "_pre_verify_nudges": 0,
        "_kanban_stop_nudges": 0,
        "_promise_stop_nudges": 0,
        "valid_tool_names": ["terminal", "write_file"],
        "_persist_disabled": False,
    }
    base.update(attrs)
    agent = SimpleNamespace(**base)
    agent._emit_interim_assistant_message = lambda *_a, **_k: None
    agent._flush_messages_to_session_db = lambda *_a, **_k: None
    agent._interim_content_was_streamed = lambda _c: False
    agent._emit_status = lambda *_a, **_k: None
    agent._emit_diagnostic_status = lambda *_a, **_k: None
    return agent


def _apply(agent, text, *, messages=None, api_call_count=3):
    """Run the gate chain exactly as ``finish_text_response`` does."""
    final_msg = {
        "role": "assistant", "content": text, "finish_reason": "stop", "reasoning": None,
    }
    messages = messages if messages is not None else [
        {"role": "user", "content": "do the task"},
    ]
    return apply_stop_gates(
        agent, final_msg, final_response=text, messages=messages,
        conversation_history=None, pending_verification_response=None,
        pending_verification_response_previewed=False,
        api_call_count=api_call_count,
    )


# ── observed candidates reproduce the bug (builder layer) ───────────────────


@pytest.mark.parametrize("text", OBSERVED)
def test_observed_promise_tail_requests_bounded_continuation(text):
    nudge = build_promise_stop_nudge(final_text=text, attempts=0)
    assert nudge is not None
    assert nudge.startswith("[System:")  # stable synthetic system-style nudge


# ── the same candidates through the gate chain (wiring layer) ───────────────


@pytest.mark.parametrize("text", OBSERVED)
def test_gate_wiring_continues_turn_on_observed_promise(text):
    agent = _agent()
    messages = [{"role": "user", "content": "run the battery"}]
    verdict = _apply(agent, text, messages=messages)
    assert verdict.continue_turn is True
    assert verdict.final_response is None
    # The withheld answer rides as the budget-exhaustion fallback.
    assert verdict.pending_verification_response == text
    # Role alternation holds: … assistant(answer) → user(nudge, synthetic).
    assert messages[-1]["role"] == "user"
    assert messages[-1][PROMISE_STOP_SYNTHETIC_FLAG] is True
    assert messages[-1]["content"].startswith("[System:")
    assert messages[-2]["role"] == "assistant"
    assert messages[-2]["content"] == text
    # Gate provenance: distinct finish_reason + counter, like the sibling gates.
    assert messages[-2]["finish_reason"] == "promise_unfulfilled"
    assert agent._promise_stop_nudges == 1


def test_gate_wiring_respects_other_gates_priority():
    """The guard is consulted after verify-on-stop / pre_verify / kanban."""
    agent = _agent()
    with (
        patch.dict("os.environ", {"HERMES_VERIFY_ON_STOP": "1"}),
        patch("agent.verification_stop.build_verify_on_stop_nudge", return_value="verify it"),
    ):
        verdict = _apply(agent, OBSERVED[1])
    assert verdict.continue_turn is True
    # verify-on-stop consumed the turn; the promise counter stayed untouched.
    assert agent._verification_stop_nudges == 1
    assert agent._promise_stop_nudges == 0

    agent2 = _agent()
    msgs2 = [{"role": "user", "content": "do it"}]
    with patch.dict("os.environ", {"HERMES_KANBAN_TASK": "t-1"}):
        verdict2 = _apply(agent2, OBSERVED[1], messages=msgs2)
    assert verdict2.continue_turn is True
    assert agent2._kanban_stop_nudges == 1
    assert agent2._promise_stop_nudges == 0
    assert msgs2[-1]["content"].lower().count("kanban") >= 1


# ── budget: bounded, never more than the sibling gates ──────────────────────


@pytest.mark.parametrize("attempts,expected", [(0, True), (1, True), (2, False)])
def test_continuation_is_bounded_per_turn(attempts, expected):
    got = build_promise_stop_nudge(final_text=OBSERVED[1], attempts=attempts)
    assert (got is not None) is expected


def test_no_tool_names_means_no_tools_available_guard_inactive():
    """No valid tool names → the model literally can't act → must not nudge."""
    agent = _agent(valid_tool_names=[])
    verdict = _apply(agent, OBSERVED[1])
    assert verdict.continue_turn is False
    assert verdict.final_response == OBSERVED[1]


def test_config_attr_off_disables_the_guard():
    agent = _agent(promise_stop_guard=False)
    verdict = _apply(agent, OBSERVED[1])
    assert verdict.continue_turn is False


def test_config_attr_on_forces_the_guard():
    agent = _agent(promise_stop_guard=True)
    verdict = _apply(agent, OBSERVED[1])
    assert verdict.continue_turn is True


# ── sibling negatives: questions / plans / quotes / reports / background ─────


@pytest.mark.parametrize("text", [
    # Question, not a promise.
    "Should I run the verification battery now?",
    "Which of the three suites do you want me to run?",
    # Plan / option listing, not a committed next action.
    "We can either run the battery now or defer it to CI.",
    "I could run the full battery now.",
    "You'll want the battery run against today's artifacts.",
    # Quoted third-party text carrying the promise shape.
    "The ticket said: 'Running the full verification battery now:' — that is "
    "what it reported, not what I am doing.",
    # Completed report describing finished work.
    "Verification battery ran green: 41/41 passed. All done.",
    "The gate reviewed 12 files and emitted the report. It is finished.",
    # Completed report that ENDS on a done-word despite a promise word.
    "Fixing all of it is done — every bug is caught and verified, so it is complete.",
    # Background-delegated phrasing ("is running" / "will run" progressive
    # after a copula is background state, not the assistant's own next action).
    "The subagent is running the tests now and will report back.",
    "Delegation is dispatched. The subagent will run the tests now.",
    # Explicit background markers.
    "Background process started; the polling continues.",
    "The delegated task will check it now.",
    # Quoted tail: the promise is the final quote, not the assistant's voice.
    "> Running the tests now.",
    "```python\nRunning the tests now.\n```",
    # Nothing promised at all.
    "The answer is 42.",
    "",
    None,
])
def test_negative_text_stops_cleanly(text):
    assert build_promise_stop_nudge(final_text=text, attempts=0) is None


def test_long_substantive_reply_is_not_treated_as_dangling():
    text = ("Here is the full analysis. " * 40) + "Verification completed: all checks passed."
    assert build_promise_stop_nudge(final_text=text, attempts=0) is None


def test_delegation_only_surface_disables_guard():
    """A delegate-only toolset owns no direct actions: narrating "the subagent
    will run it now" is the correct end state, so a re-prompt would be noise."""
    agent = _agent(valid_tool_names=["delegate_task"])
    verdict = _apply(agent, OBSERVED[1])
    assert verdict.continue_turn is False
    assert verdict.final_response == OBSERVED[1]


# ── nudge content is a byte-stable constant per turn (prompt-cache safe) ────


def test_nudge_text_is_stable_across_attempts():
    first = build_promise_stop_nudge(final_text=OBSERVED[1], attempts=0)
    second = build_promise_stop_nudge(final_text=OBSERVED[2], attempts=1)
    assert first == second
    assert first == promise_stop_nudge_text()


# ── gate wiring: consulted through its origin module, after the siblings ────


def test_gate_chain_consults_promise_builder_at_origin_module():
    """apply_stop_gates imports the builder lazily from ``agent.promise_stop``
    (the repo's patch-where-production-reads seam): patching the origin module
    flips the verdict, proving the wiring into the gate chain."""
    agent = _agent()
    messages = [{"role": "user", "content": "do it"}]
    text = "The answer is 42."  # builder would refuse this on its own
    with patch("agent.promise_stop.build_promise_stop_nudge", return_value="[System: act]"):
        verdict = _apply(agent, text, messages=messages)
    assert verdict.continue_turn is True
    assert agent._promise_stop_nudges == 1
    assert messages[-1][PROMISE_STOP_SYNTHETIC_FLAG] is True


def test_gate_chain_stops_before_promise_when_no_builder_result():
    agent = _agent()
    with patch("agent.promise_stop.build_promise_stop_nudge", return_value=None):
        verdict = _apply(agent, OBSERVED[0])
    assert verdict.continue_turn is False
    assert verdict.final_response == OBSERVED[0]
    assert agent._promise_stop_nudges == 0


# ── synthetic-metadata registration (behaviour, not source reading) ────────


def test_nudge_row_is_registered_ephemeral_and_non_user_context():
    """The nudge flag must be known to the persistence skip-list, the
    compression real-user-message classifier, and the finalizer's
    continuation drop — asserted on the live tables, not on source text."""
    from agent.conversation_compression import _is_real_user_message
    from agent.session_persistence import _is_ephemeral_scaffolding
    from agent.turn_finalizer import _drop_verification_continuation_scaffolding

    nudge_row = {
        "role": "user", "content": promise_stop_nudge_text(),
        PROMISE_STOP_SYNTHETIC_FLAG: True,
    }
    assert _is_ephemeral_scaffolding(nudge_row) is True  # never durable
    assert _is_real_user_message(nudge_row) is False  # never human intent
    rows = [nudge_row, {"role": "assistant", "content": "x"}]
    _drop_verification_continuation_scaffolding(rows)  # stripped from live/returned history
    assert nudge_row not in rows
    # A human row is untouched by all three.
    human = {"role": "user", "content": "please run it"}
    assert _is_ephemeral_scaffolding(human) is False
    assert _is_real_user_message(human) is True


# ── exhausted budget: the promise must not read as a clean completion ───────


def test_exhaustion_never_marks_the_candidate_row_as_plain_stop():
    """Once the per-turn budget is spent, the gate must not stamp a further
    candidate with the ordinary terminal ``finish_reason=stop`` shape: the
    turn either ends as a clean text response (no stamp) or the candidate row
    is dropped by the finalizer scaffolding drop — never a silent success."""
    agent = _agent()
    text = OBSERVED[1]
    messages = [{"role": "user", "content": "do it"}]
    # Spend the budget through the gate's own counter, as the loop would.
    agent._promise_stop_nudges = 2
    verdict = _apply(agent, text, messages=messages)
    assert verdict.continue_turn is False
    assert verdict.final_response == text
    assert agent._promise_stop_nudges == 2  # untouched by a refused gate


# ── end-to-end: real loop continuation + finalizer scaffolding contract ──────


def _response(content):
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model="test/model",
        usage=None,
    )


@pytest.fixture
def loop_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        instance = AIAgent(
            session_id="promise-guard-test",
            api_key="test-key",
            base_url="https://example.invalid/v1",
            provider="openai-compat",
            model="test/model",
            max_iterations=4,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    instance._cached_system_prompt = "stable test prompt"
    instance._session_db = None
    instance.save_trajectories = False
    instance.compression_enabled = False
    instance._cleanup_task_resources = lambda *_a, **_kw: None
    instance._save_trajectory = lambda *_a, **_kw: None
    instance.valid_tool_names = ["terminal", "write_file"]
    return instance


def test_loop_continues_after_promise_and_delivers_real_completion(loop_agent):
    """E2E: a promised-then-fulfilled turn continues, the nudge is stripped
    from the returned transcript, and roles alternate."""
    answers = iter([
        _response("Two real bugs caught in the harness. Fixing all of it now."),
        _response("Fixes complete: the gate now parses timestamps and 41 tests pass."),
    ])
    loop_agent._interruptible_api_call = lambda _kwargs: next(answers)
    with (
        patch("hermes_cli.plugins.has_hook", return_value=False),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = loop_agent.run_conversation("fix the gate")

    assert result["final_response"] == (
        "Fixes complete: the gate now parses timestamps and 41 tests pass."
    )
    assert result["api_calls"] == 2
    roles = [m["role"] for m in result["messages"]]
    # The synthetic nudge is dropped by the finalizer scaffolding drop; the
    # assistant interim that answered with the promise survives.
    assert roles == ["user", "assistant", "assistant"] or roles == ["user", "assistant"]
    assert all(
        PROMISE_STOP_SYNTHETIC_FLAG not in m for m in result["messages"]
    )
    assert result.get("completed") is True


def test_loop_exhaustion_surfaces_incomplete_promise_not_clean_success(loop_agent):
    """E2E: the model keeps only promising; after the bounded continuations
    the turn must not present the promise as a clean completion."""
    def _promising(_kwargs):
        return _response("Both writes landed. Running the full verification battery now:")

    loop_agent._interruptible_api_call = _promising
    with (
        patch("hermes_cli.plugins.has_hook", return_value=False),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = loop_agent.run_conversation("run the battery")

    # 1 initial call + the two bounded continuations, then the gate stops.
    assert result["api_calls"] == 3
    assert result["final_response"] and "Running the full verification battery" in (
        result["final_response"]
    )
    # The promise is NOT presented as a clean completion: an explicit
    # incomplete signal rides the result and the user-visible text.
    assert result.get("action_promise_unfulfilled") is True
    assert result.get("completed") is False
    assert "⚠" in result["final_response"]
    assert "[System:" not in result["final_response"]  # raw nudge never leaks


def test_promise_budget_resets_between_turns(loop_agent):
    """Per-turn reset contract: turn 1 exhausts the two-continuation budget,
    yet turn 2 still gets its own (the counter spans a turn, not the session)."""
    answers = iter([
        _response("Two real bugs caught in the harness. Fixing all of it now."),
        _response("Both writes landed. Running the verification battery now:"),
        _response("Fixes complete."),
        _response("Both writes landed. Running the verification battery now:"),
        _response("Battery green: 41 passed."),
    ])
    loop_agent._interruptible_api_call = lambda _kwargs: next(answers)
    with (
        patch("hermes_cli.plugins.has_hook", return_value=False),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        first = loop_agent.run_conversation("fix the bugs")
        second = loop_agent.run_conversation("run the battery")

    # Turn 1: promise → nudge → promise → nudge → clean answer (3 calls, budget spent).
    assert first["api_calls"] == 3
    # Turn 2: with a session-wide counter this would be 1 call (budget already spent);
    # the per-turn reset gives it its own continuation.
    assert second["api_calls"] == 2
    assert second["final_response"] == "Battery green: 41 passed."
    assert second.get("completed") is True

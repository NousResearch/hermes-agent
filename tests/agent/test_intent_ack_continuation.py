"""Intent-ack continuation gate + detector behavior.

Covers the config-driven generalization of the codex intent-ack continuation
(issue #27881): the historical ``codex_responses``-only path is byte-stable
under the default ``"auto"`` mode, while an explicit ``true``/model-list opt-in
extends the "you announced an action but called no tool — keep going" nudge to
every api_mode and relaxes the codebase/workspace requirement so general
autonomous workflows ("I'll run a health check on the server") are caught.

Also covers the #74604 progress-narration stall: present-progressive narration
("I am now compiling the complete answer.") must continue the turn both before
any tool work (the ack detector) and AFTER real tool work, where the
``post_work_narration_intent`` guard in ``agent.turn_final_response`` fires —
exercised end-to-end through ``run_conversation`` with a mocked client.

These are invariant assertions about how the mode string and the detector
gates relate, not snapshots of the marker lists.
"""

from unittest.mock import MagicMock, patch

from types import SimpleNamespace
from typing import Union

import pytest

from agent.agent_runtime_helpers import (
    intent_ack_continuation_mode,
    looks_like_codex_intermediate_ack,
)


def _agent(
    mode: Union[str, bool, list] = "auto",
    api_mode="chat_completions",
    model="anthropic/claude-sonnet-4",
):
    # _strip_think_blocks is a no-op for these plain-text fixtures.
    return SimpleNamespace(
        _intent_ack_continuation=mode,
        api_mode=api_mode,
        model=model,
        _strip_think_blocks=lambda c: c,
    )


# The reporter's exact repro (#27881): server-ops task, no filesystem reference.
REPRO_USER = (
    "check the current status of the server, grab the latest error logs, "
    "and let me know if there's anything critical"
)
REPRO_ACK = "I will start by running a health check command on the server to see its current status."

# The codex-coding case the detector was originally built for.
CODE_USER = "review the codebase in /app"
CODE_ACK = "Let me inspect the repository files first."


# ── mode resolution ────────────────────────────────────────────────────────




def test_true_is_all_api_modes():
    for am in ("chat_completions", "anthropic", "codex_responses"):
        assert intent_ack_continuation_mode(_agent(True, am)) == "all"
    for s in ("true", "always", "yes", "on", "ON"):
        assert intent_ack_continuation_mode(_agent(s, "chat_completions")) == "all"








def test_unrecognized_attr_defaults_like_auto():
    """A missing/unrecognized ``_intent_ack_continuation`` value falls back to the
    historical auto behavior: codex-only scope, off elsewhere."""
    bare = SimpleNamespace(api_mode="chat_completions", model="x", _strip_think_blocks=lambda c: c)
    assert intent_ack_continuation_mode(bare) == "off"
    bare_codex = SimpleNamespace(api_mode="codex_responses", model="x", _strip_think_blocks=lambda c: c)
    assert intent_ack_continuation_mode(bare_codex) == "codex_only"


# ── detector: workspace requirement ─────────────────────────────────────────




def test_multipart_user_message_does_not_crash_on_workspace_path():
    """#9562: vision requests forward ``user_message`` as a multi-part list.

    The OpenAI-compat API server passes the raw ``content`` field straight
    through for vision turns, so ``user_message`` reaches the detector as
    ``[{type:"text",...}, {type:"image_url",...}]``. The ``require_workspace``
    path flattened it with ``(user_message or "").strip()`` — a truthy list
    survived and ``.strip()`` raised ``AttributeError``, killing the turn.
    The text part still has to drive workspace detection.
    """
    a = _agent("auto", "codex_responses")
    multipart = [
        {"type": "text", "text": CODE_USER},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    msgs = [{"role": "user", "content": multipart}]
    # No crash, and the text part ("review the codebase in /app") still
    # satisfies the workspace requirement so the ack fires.
    assert looks_like_codex_intermediate_ack(
        a, multipart, CODE_ACK, msgs, require_workspace=True
    )


def test_all_path_drops_workspace_requirement():
    """The #27881 fix: opted-in turns catch non-codebase intent acks."""
    a = _agent(True, "chat_completions")
    msgs = [{"role": "user", "content": REPRO_USER}]
    assert looks_like_codex_intermediate_ack(
        a, REPRO_USER, REPRO_ACK, msgs, require_workspace=False
    )


# ── detector: guardrails that hold regardless of workspace ───────────────────


# ── detector: progress narration patterns (issue #74604) ────────────────────

NARRATION_USER = "What is the full analysis of the dataset?"
NARRATION_ACK = "I am now compiling the complete answer."


def test_progress_narration_detected_as_intermediate_ack():
    """Issue #74604: 'I am now compiling the complete answer.' is a progress
    narration that should trigger a continuation, not end the turn. The
    detector must catch 'i am now' + 'compiling' as a future-ack + action.
    """
    a = _agent(True, "chat_completions")
    msgs = [{"role": "user", "content": NARRATION_USER}]
    assert looks_like_codex_intermediate_ack(
        a, NARRATION_USER, NARRATION_ACK, msgs, require_workspace=False
    )


def test_progress_narration_with_generating():
    """'I'm currently generating the report.' should also be caught."""
    a = _agent(True, "chat_completions")
    msgs = [{"role": "user", "content": "Create the report"}]
    assert looks_like_codex_intermediate_ack(
        a, "Create the report",
        "I'm currently generating the report.",
        msgs, require_workspace=False,
    )

# ── detector: concise final answers remain terminal (#74604) ───────────────

CONCISE_ANSWERS = [
    "The server is healthy. All services are running normally.",
    "42",
    "Done. The file has been updated.",
    "Based on the analysis, the root cause is a missing index on the users table.",
]


@pytest.mark.parametrize("answer", CONCISE_ANSWERS)
def test_concise_final_answer_not_detected_as_intermediate_ack(answer):
    """A valid concise final answer with finish_reason=stop must NOT trigger
    a continuation — it should remain terminal. This is the guard against
    over-filtering that GottZ's triage on #76013 asked for.
    """
    a = _agent(True, "chat_completions")
    msgs = [{"role": "user", "content": "What is the answer?"}]
    assert not looks_like_codex_intermediate_ack(
        a, "What is the answer?", answer, msgs, require_workspace=False
    )


# ── #74604 end-to-end: the REAL loop continues a narrated stop ──────────────
# (mirrors test_degenerate_final_recovery.py: a real AIAgent with a mocked
# OpenAI client; the re-prompt must come from the actual loop wiring, not a
# hand-copied decision — an inline reimplementation stays green even if the
# loop never calls the detector, and it was a real-loop run that exposed the
# post-tool-work silence the pre-tool-work-only detector had).


@pytest.fixture()
def loop_agent():
    """AIAgent with a mocked OpenAI client (mirrors test_degenerate_final_recovery)."""
    from run_agent import AIAgent
    from tests.agent.test_run_agent import _make_tool_defs
    with (
        patch("model_tools.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.tool_delay = 0
        agent.compression_enabled = False
        agent.save_trajectories = False
        # Explicit opt-in: the loop tests exercise the mechanism, not the
        # transport-scoped default (chat_completions would otherwise be off).
        agent._intent_ack_continuation = True
        return agent


def _run(agent, stages):
    agent.client.chat.completions.create.side_effect = stages
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("model_tools.handle_function_call", return_value="ok"),
    ):
        return agent.run_conversation("analyze the dataset")


def _tool_round(call_id):
    from tests.agent.test_run_agent import _mock_response, _mock_tool_call
    return _mock_response(
        content="", finish_reason="tool_calls",
        tool_calls=[_mock_tool_call(name="web_search", arguments="{}", call_id=call_id)],
    )


def _final(text):
    from tests.agent.test_run_agent import _mock_response
    return _mock_response(content=text, finish_reason="stop")


def _user_rows_sent(agent, call_index):
    kwargs = agent.client.chat.completions.create.call_args_list[call_index].kwargs
    return [m["content"] for m in kwargs["messages"] if m.get("role") == "user"]


NARRATION = "I am now compiling the complete answer."


def test_loop_continues_narrated_stop_before_tool_work(loop_agent):
    """#74604 pre-tool-work shape: the model answers with the progress narration
    alone (no tool call) and the REAL loop must re-prompt instead of exiting."""
    result = _run(loop_agent, [
        _final(NARRATION),
        _final("The full analysis shows three defects."),
    ])

    assert loop_agent.client.chat.completions.create.call_count == 2, (
        "A narrated stop must trigger a continuation (second API call), not end the turn."
    )
    nudge = _user_rows_sent(loop_agent, 1)[-1]
    assert "continue" in nudge.lower()
    assert "The full analysis" in result["final_response"]


def test_loop_continues_narrated_stop_after_tool_work(loop_agent):
    """#74604's reported shape: ~35 tool calls, then the narration, then stop.
    The ack detector is gated off once tool rows exist, so this can only pass
    through the post-work narration guard — a regression here means the loop
    delivered the narration as the final answer."""
    result = _run(loop_agent, [
        _tool_round(f"call_{n:02d}") for n in range(35)
    ] + [
        _final(NARRATION),
        _final("The compiled analysis: root cause is a missing index."),
    ])

    assert loop_agent.client.chat.completions.create.call_count == 37, (
        "A post-work narrated stop must trigger a continuation, not end the turn "
        "with the narration as the answer."
    )
    nudge = _user_rows_sent(loop_agent, 36)[-1]
    assert "continue" in nudge.lower()
    assert "root cause is a missing index" in result["final_response"]


def test_loop_narration_path_is_bounded_and_terminates(loop_agent):
    """A model that keeps narrating instead of delivering ends after the bounded
    continuation counter (2) — the second narration itself becomes the answer
    (only 1 re-prompt left before the cap), never a loop."""
    result = _run(loop_agent, [
        _tool_round("call_1"),
        _final(NARRATION),
        _final("I'm currently generating the report."),
    ])

    # 1 tool round + 1 narrated stop + 1 continuation call after the first
    # nudge = 3; the second narrated stop exhausts the 2-continuation cap and
    # ends the turn with that narration as the answer.
    assert loop_agent.client.chat.completions.create.call_count == 3
    assert result["final_response"] == "I'm currently generating the report."


def test_loop_concise_answer_after_tool_work_remains_terminal(loop_agent):
    """The guardrail: a real concise answer after tool work stays terminal — the
    post-work narration guard must not re-prompt completed answers."""
    result = _run(loop_agent, [
        _tool_round("call_1"),
        _final("The dataset has 3 defects; details above."),
    ])

    assert loop_agent.client.chat.completions.create.call_count == 2
    assert "3 defects" in result["final_response"]


def test_loop_narration_ignored_when_stall_guards_disabled(loop_agent):
    """agent.stall_guards: false turns the narration guard off — the narrated
    stop is delivered as-is (a config escape hatch for false positives)."""
    loop_agent._stall_guards = False
    result = _run(loop_agent, [
        _tool_round("call_1"),
        _final(NARRATION),
    ])

    assert loop_agent.client.chat.completions.create.call_count == 2
    assert result["final_response"] == NARRATION








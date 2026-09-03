"""Intent-ack continuation gate + detector behavior.

Covers the config-driven generalization of the codex intent-ack continuation
(issue #27881): the historical ``codex_responses``-only path is byte-stable
under the default ``"auto"`` mode, while an explicit ``true``/model-list opt-in
extends the "you announced an action but called no tool — keep going" nudge to
every api_mode and relaxes the codebase/workspace requirement so general
autonomous workflows ("I'll run a health check on the server") are caught.

These are invariant assertions about how the mode string and the detector
gates relate, not snapshots of the marker lists.
"""

from types import SimpleNamespace
from typing import Union

import pytest

from agent.agent_runtime_helpers import (
    intent_ack_continuation_enabled,
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








def test_missing_attr_defaults_to_auto():
    bare = SimpleNamespace(api_mode="chat_completions", model="x", _strip_think_blocks=lambda c: c)
    assert intent_ack_continuation_mode(bare) == "off"
    bare_codex = SimpleNamespace(api_mode="codex_responses", model="x", _strip_think_blocks=lambda c: c)
    assert intent_ack_continuation_mode(bare_codex) == "codex_only"


def test_enabled_is_mode_not_off():
    assert intent_ack_continuation_enabled(_agent(True, "chat_completions")) is True
    assert intent_ack_continuation_enabled(_agent("auto", "codex_responses")) is True
    assert intent_ack_continuation_enabled(_agent("auto", "chat_completions")) is False
    assert intent_ack_continuation_enabled(_agent(False, "codex_responses")) is False


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


def test_finish_reason_stop_narration_triggers_continuation():
    """Issue #74604: a progress narration with finish_reason=stop (no tool
    calls) must trigger a continuation, not end the turn. This simulates the
    conversation loop's decision at agent/conversation_loop.py:6713 — when
    finish_reason is 'stop', the loop checks _looks_like_codex_intermediate_ack
    and continues if True.

    This test exercises the detector with the exact pattern from the bug
    report: the model narrates progress ("I am now compiling the complete
    answer.") and stops without calling any tool. The detector must return
    True so the loop continues the turn.
    """
    a = _agent(True, "chat_completions")
    user_msg = "What is the full analysis of the dataset?"
    narration = "I am now compiling the complete answer."
    msgs = [{"role": "user", "content": user_msg}]

    # Simulate the loop's decision: finish_reason=stop, no tool_calls,
    # check if the response looks like an intermediate ack.
    finish_reason = "stop"
    has_tool_calls = False

    should_continue = (
        finish_reason == "stop"
        and not has_tool_calls
        and looks_like_codex_intermediate_ack(
            a, user_msg, narration, msgs, require_workspace=False
        )
    )

    assert should_continue, (
        "Progress narration with finish_reason=stop should trigger continuation, "
        "not end the turn"
    )


def test_finish_reason_stop_concise_answer_remains_terminal():
    """The complement of the above: a concise final answer with
    finish_reason=stop must NOT trigger a continuation. The loop should
    break and treat it as the final answer.
    """
    a = _agent(True, "chat_completions")
    user_msg = "What is the answer?"
    answer = "The answer is 42."
    msgs = [{"role": "user", "content": user_msg}]

    finish_reason = "stop"
    has_tool_calls = False

    should_continue = (
        finish_reason == "stop"
        and not has_tool_calls
        and looks_like_codex_intermediate_ack(
            a, user_msg, answer, msgs, require_workspace=False
        )
    )

    assert not should_continue, (
        "A concise final answer with finish_reason=stop should remain terminal, "
        "not trigger a continuation"
    )








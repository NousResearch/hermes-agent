"""End-to-end regression coverage for verification budget exhaustion (#61631, #65919 §7)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _response(content="composed report"):
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model="test/model",
        usage=None,
    )


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        instance = AIAgent(
            session_id="verify-budget-test",
            api_key="test-key",
            base_url="https://example.invalid/v1",
            provider="openai-compat",
            model="test/model",
            max_iterations=1,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    instance._cached_system_prompt = "stable test prompt"
    instance._session_db = None
    instance._session_json_enabled = False
    instance.save_trajectories = False
    instance.compression_enabled = False
    instance._cleanup_task_resources = lambda *_a, **_kw: None
    instance._save_trajectory = lambda *_a, **_kw: None
    return instance


def _assert_pending_response_survives(agent, result):
    assert result["final_response"] == "composed report"
    assert result["turn_exit_reason"] == "max_iterations_reached(1/1)"
    assert result["completed"] is False
    assert agent._handle_max_iterations.call_count == 0
    # The nudge is stripped by _drop_verification_continuation_scaffolding,
    # so the role sequence is [user, assistant] — the candidate is the
    # tail and matches final_response so it is not duplicated. (#65919 §7)
    assert [message["role"] for message in result["messages"]] == [
        "user",
        "assistant",
    ]


def test_verify_on_stop_preserves_composed_report_at_budget_limit(agent, monkeypatch):
    def model_call(_api_kwargs):
        agent._turn_file_mutation_paths = {"changed.py"}
        return _response()

    agent._interruptible_api_call = model_call
    agent._handle_max_iterations = MagicMock(return_value="replacement summary")
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "1")

    with (
        patch("agent.verification_stop.build_verify_on_stop_nudge", return_value="verify it"),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = agent.run_conversation("edit changed.py")

    _assert_pending_response_survives(agent, result)
    # The assistant response persists (it is real, unflagged content).
    assert not result["messages"][1].get("_verification_stop_synthetic")


def test_pre_verify_preserves_composed_report_at_budget_limit(agent, monkeypatch):
    def model_call(_api_kwargs):
        agent._turn_file_mutation_paths = {"changed.py"}
        return _response()

    agent._interruptible_api_call = model_call
    agent._handle_max_iterations = MagicMock(return_value="replacement summary")
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")

    with (
        patch("hermes_cli.plugins.has_hook", side_effect=lambda name: name == "pre_verify"),
        patch(
            "hermes_cli.plugins.get_pre_verify_continue_message",
            return_value="run project tests",
        ),
        patch("agent.verify_hooks.max_verify_nudges", return_value=2),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = agent.run_conversation("edit changed.py")

    _assert_pending_response_survives(agent, result)
    # The assistant response persists (it is real, unflagged content).
    assert not result["messages"][1].get("_pre_verify_synthetic")


def test_intermediate_ack_uses_summary_instead_of_premature_text(agent, monkeypatch):
    agent.valid_tool_names = ["web_search"]
    agent._intent_ack_continuation = True
    agent._looks_like_codex_intermediate_ack = MagicMock(return_value=True)
    agent._interruptible_api_call = lambda _kwargs: _response("I'll inspect the files now")
    agent._handle_max_iterations = MagicMock(return_value="verified summary.")
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")

    with (
        patch("hermes_cli.plugins.has_hook", return_value=False),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = agent.run_conversation("inspect /tmp/project")

    assert result["final_response"] == "verified summary."
    assert result["turn_exit_reason"] == "max_iterations_reached(1/1)"
    agent._handle_max_iterations.assert_called_once()


def test_later_verified_response_supersedes_pending_report(agent, monkeypatch):
    agent.max_iterations = 2
    agent.iteration_budget.max_total = 2
    answers = iter([_response("premature report"), _response("verified final report")])
    agent._interruptible_api_call = lambda _kwargs: next(answers)
    agent._handle_max_iterations = MagicMock(return_value="replacement summary")
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "1")

    with (
        patch(
            "agent.verification_stop.build_verify_on_stop_nudge",
            side_effect=["verify it", None],
        ),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = agent.run_conversation("edit changed.py")

    assert result["final_response"] == "verified final report"
    assert result["turn_exit_reason"] == "text_response(finish_reason=stop)"
    assert result["completed"] is True
    agent._handle_max_iterations.assert_not_called()


def test_multiple_verification_retries_publish_each_candidate_once(agent, monkeypatch):
    """Multiple verification retries should publish each candidate once, in order."""
    agent.max_iterations = 3
    agent.iteration_budget.max_total = 3
    answers = iter([
        _response("candidate one"),
        _response("candidate two"),
        _response("candidate three"),
    ])
    agent._interruptible_api_call = lambda _kwargs: next(answers)
    agent._handle_max_iterations = MagicMock(return_value="replacement summary")
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "1")

    # Three nudges, then None (so the third candidate is the final response).
    nudge_side_effects = ["verify it", "verify it", None]

    emitted = []
    agent.interim_assistant_callback = lambda text, **kw: emitted.append(text)

    with (
        patch(
            "agent.verification_stop.build_verify_on_stop_nudge",
            side_effect=nudge_side_effects,
        ),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = agent.run_conversation("edit changed.py")

    # Each candidate was emitted as an interim message, in order.
    assert emitted == ["candidate one", "candidate two"]
    # The final response is the last candidate.
    assert result["final_response"] == "candidate three"
    assert result["turn_exit_reason"] == "text_response(finish_reason=stop)"
    assert result["completed"] is True
    agent._handle_max_iterations.assert_not_called()


def test_verification_false_finalizes_candidate_once(agent, monkeypatch):
    """When verification returns false/exception, the candidate is finalized once."""
    agent._interruptible_api_call = lambda _kwargs: _response("the answer")
    agent._handle_max_iterations = MagicMock(return_value="replacement summary")
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "1")

    emitted = []
    agent.interim_assistant_callback = lambda text, **kw: emitted.append(text)

    with (
        # build_verify_on_stop_nudge raises — simulates verification check failure
        patch(
            "agent.verification_stop.build_verify_on_stop_nudge",
            side_effect=RuntimeError("verify check crashed"),
        ),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = agent.run_conversation("edit changed.py")

    # No interim emission because verification did not run (exception path
    # sets _verify_nudge = None, so the candidate becomes the final response
    # without an interim emission).
    assert result["final_response"] == "the answer"
    assert result["completed"] is True
    agent._handle_max_iterations.assert_not_called()


def test_verify_on_stop_emits_interim_response_to_ui(agent, monkeypatch):
    """The verify-on-stop path must emit the full response to the UI callback.

    With no streaming set up in this test, _interim_content_was_streamed
    returns False, so already_streamed is False — the callback reports
    content the UI has not seen yet.
    """
    agent._interruptible_api_call = lambda _kwargs: _response("composed report")
    agent._handle_max_iterations = MagicMock(return_value="replacement summary")
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "1")

    callback_calls = []

    def capture_callback(text, *, already_streamed=None):
        callback_calls.append({"text": text, "already_streamed": already_streamed})

    agent.interim_assistant_callback = capture_callback

    with (
        patch("agent.verification_stop.build_verify_on_stop_nudge", return_value="verify it"),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = agent.run_conversation("edit changed.py")

    # The callback was called with the full response text and already_streamed=False
    assert len(callback_calls) == 1
    assert callback_calls[0]["text"] == "composed report"
    assert callback_calls[0]["already_streamed"] is False

    # The candidate persists as the final response.
    assert result["final_response"] == "composed report"


def test_streamed_interim_then_different_summary_not_marked_previewed(agent, monkeypatch):
    """Ordinary interim narration followed by a different non-streamed summary.

    The model streams "I'll inspect the files now" as an intermediate ack.
    _emit_interim_assistant_message is called for this ordinary narration,
    which must NOT set _response_was_previewed. Then _handle_max_iterations
    produces a different summary through the non-streaming Chat Completions
    path. The final result must NOT be marked as previewed — the interim was
    unrelated mid-turn commentary, not the final response — so the CLI renders
    the summary instead of suppressing it. (#65919 review: response-loss blocker)
    """
    agent.valid_tool_names = ["web_search"]
    agent._intent_ack_continuation = True
    agent._looks_like_codex_intermediate_ack = MagicMock(return_value=True)
    agent._interruptible_api_call = lambda _kwargs: _response("I'll inspect the files now")
    agent._handle_max_iterations = MagicMock(return_value="Here is the summary of what I found.")
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")

    emitted = []
    agent.interim_assistant_callback = lambda text, **kw: emitted.append(text)

    with (
        patch("hermes_cli.plugins.has_hook", return_value=False),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = agent.run_conversation("inspect /tmp/project")

    # The final response is the different summary from _handle_max_iterations.
    assert result["final_response"] == "Here is the summary of what I found."
    # CRITICAL: response_previewed must be False — the interim narration was
    # NOT the final response, so the CLI must render the summary.
    assert result["response_previewed"] is False


def test_streamed_verification_candidate_reused_marked_previewed(agent, monkeypatch):
    """Verification candidate reused at budget exhaustion is marked previewed.

    The model streams a verification candidate that is already streamed as
    interim content. The continuation budget is exhausted, so the finalizer
    reuses the pending verification candidate as the final response. The result
    must be marked as previewed so the CLI/desktop settle it once instead of
    duplicating. (#65919 review)
    """
    agent._interruptible_api_call = lambda _kwargs: _response("composed report")
    agent._handle_max_iterations = MagicMock(return_value="replacement summary")
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "1")

    agent._turn_file_mutation_paths = {"changed.py"}

    callback_calls = []

    def capture_callback(text, *, already_streamed=None):
        callback_calls.append({"text": text, "already_streamed": already_streamed})

    agent.interim_assistant_callback = capture_callback

    # Simulate that the candidate text was already streamed. The streaming
    # buffer is cleared after the response is processed, so mock the check
    # directly — this is the condition the test validates: when the candidate
    # was streamed, the previewed flag propagates to the finalizer.
    with (
        patch.object(agent, "_interim_content_was_streamed", return_value=True),
        patch("agent.verification_stop.build_verify_on_stop_nudge", return_value="verify it"),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = agent.run_conversation("edit changed.py")

    # The candidate was already streamed, so the callback reports already_streamed=True.
    assert len(callback_calls) == 1
    assert callback_calls[0]["already_streamed"] is True
    # The candidate is reused as the final response.
    assert result["final_response"] == "composed report"
    # CRITICAL: response_previewed must be True — the reused candidate was
    # streamed as interim content, so the CLI/desktop settle it once.
    assert result["response_previewed"] is True

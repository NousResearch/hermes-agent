"""End-to-end regression coverage for verification budget exhaustion (#61631)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB
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


def _attach_real_db(agent, tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(agent.session_id, source="test", model=agent.model)
    agent._session_db = db
    agent._session_db_created = True
    return db


def _assistant_contents(db, session_id):
    return [
        row["content"]
        for row in db.get_messages(session_id)
        if row["role"] == "assistant"
    ]


def _assert_pending_response_survives(agent, result):
    assert result["final_response"] == "composed report"
    assert result["turn_exit_reason"] == "max_iterations_reached(1/1)"
    assert result["completed"] is False
    assert agent._handle_max_iterations.call_count == 0
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
    # The published candidate is durable and becomes the budget fallback.
    # Its one-request continuation nudge is removed before finalization, so the
    # same assistant content is not appended a second time.
    assert "_verification_stop_synthetic" not in result["messages"][1]
    assert result["messages"][1]["content"] == "composed report"


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
    # The pre-verify path has the same exact-once finalizer contract.
    assert "_pre_verify_synthetic" not in result["messages"][1]
    assert result["messages"][1]["content"] == "composed report"


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


def test_multiple_verification_retries_publish_each_candidate_once_in_order(
    agent, monkeypatch, tmp_path
):
    """Every retry candidate is one durable row and one commentary event.

    The terminal answer remains the normal final response; it is persisted once
    by the finalizer and is not re-emitted as interim commentary.
    """
    db = _attach_real_db(agent, tmp_path)
    agent.max_iterations = 3
    agent.iteration_budget.max_total = 3
    answers = iter(
        [
            _response("candidate one"),
            _response("candidate two"),
            _response("verified final report"),
        ]
    )
    request_roles = []

    def model_call(api_kwargs):
        agent._turn_file_mutation_paths = {"changed.py"}
        request_roles.append([message["role"] for message in api_kwargs["messages"]])
        return next(answers)

    emitted = []
    agent._interruptible_api_call = model_call
    agent.interim_assistant_callback = (
        lambda text, **_kwargs: emitted.append(text)
    )
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "1")

    with (
        patch(
            "agent.verification_stop.build_verify_on_stop_nudge",
            side_effect=["verify once", "verify again", None],
        ),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = agent.run_conversation("edit changed.py")

    assert result["final_response"] == "verified final report"
    assert result["completed"] is True
    assert emitted == ["candidate one", "candidate two"]
    assert _assistant_contents(db, agent.session_id) == [
        "candidate one",
        "candidate two",
        "verified final report",
    ]
    assert request_roles == [
        ["system", "user"],
        ["system", "user", "assistant", "user"],
        ["system", "user", "assistant", "user", "assistant", "user"],
    ]
    assert not any(
        message.get("_verification_stop_synthetic")
        or message.get("_pre_verify_synthetic")
        for message in result["messages"]
    )

    # Resume/replay keeps all three durable rows distinct, while protocol
    # repair removes superseded provisional candidates from the model history.
    replay = db.get_messages_as_conversation(agent.session_id)
    agent._repair_message_sequence(replay)
    assert [
        message["content"]
        for message in replay
        if message["role"] == "assistant"
    ] == ["verified final report"]
    assert _assistant_contents(db, agent.session_id) == [
        "candidate one",
        "candidate two",
        "verified final report",
    ]
    db.close()


@pytest.mark.parametrize("verification_outcome", [False, RuntimeError("verifier crashed")])
def test_verification_false_or_exception_finalizes_candidate_once(
    agent, monkeypatch, tmp_path, verification_outcome
):
    """A failed verifier decision happens before interim publication.

    False and exceptions both fail open: the candidate becomes the one terminal
    response, so the return value and durable transcript cannot disagree.
    """
    db = _attach_real_db(agent, tmp_path)
    emitted = []

    def model_call(_api_kwargs):
        agent._turn_file_mutation_paths = {"changed.py"}
        return _response("candidate after verifier failure")

    agent._interruptible_api_call = model_call
    agent.interim_assistant_callback = (
        lambda text, **_kwargs: emitted.append(text)
    )
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "1")
    verifier = (
        patch(
            "agent.verification_stop.build_verify_on_stop_nudge",
            side_effect=verification_outcome,
        )
        if isinstance(verification_outcome, Exception)
        else patch(
            "agent.verification_stop.build_verify_on_stop_nudge",
            return_value=verification_outcome,
        )
    )

    with verifier, patch("hermes_cli.plugins.invoke_hook", return_value=[]):
        result = agent.run_conversation("edit changed.py")

    assert result["final_response"] == "candidate after verifier failure"
    assert result["completed"] is True
    assert emitted == []
    assert _assistant_contents(db, agent.session_id) == [
        "candidate after verifier failure"
    ]
    db.close()


def test_verify_on_stop_emits_interim_response_to_ui(agent, monkeypatch):
    """The full assistant response must reach the UI when verification stop
    triggers — not just the terse post-verification reply. (#62657)

    When the response was already streamed (Desktop gateway), the callback
    must still send it as a standalone commentary bubble (force_display=True)
    so it survives the subsequent verification messages.
    """
    emitted = []
    call_kwargs = []
    def _capture(text, **kw):
        emitted.append(text)
        call_kwargs.append(kw)

    agent.interim_assistant_callback = _capture

    def model_call(_api_kwargs):
        agent._turn_file_mutation_paths = {"changed.py"}
        return _response("full detailed report with tables and analysis")

    agent._interruptible_api_call = model_call
    agent._handle_max_iterations = MagicMock(return_value="replacement summary")
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "1")

    with (
        patch("agent.verification_stop.build_verify_on_stop_nudge", return_value="verify it"),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = agent.run_conversation("edit changed.py")

    # The full response was emitted to the UI callback.
    assert any("full detailed report" in e for e in emitted), (
        f"expected full response in emitted messages, got: {emitted}"
    )
    # force_display=True → already_streamed=False so the gateway calls
    # on_commentary() (standalone bubble), not on_segment_break() (which
    # just finalizes the streaming buffer and gets overwritten).
    assert any(kw.get("already_streamed") is False for kw in call_kwargs), (
        f"expected already_streamed=False in at least one call, got: {call_kwargs}"
    )
    # The response was also persisted to the in-memory message list without
    # the synthetic flag.
    assistant_msgs = [m for m in result["messages"] if m["role"] == "assistant"]
    assert any(
        "full detailed report" in (m.get("content") or "")
        and "_verification_stop_synthetic" not in m
        for m in assistant_msgs
    )

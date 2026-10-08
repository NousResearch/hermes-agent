"""Manual primary-provider fallback: one submitted route, one turn, no silent cascade."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.error_classifier import FailoverReason
from agent.manual_fallback import ManualFallbackStopped
from run_agent import AIAgent


ROUTES = [
    {"provider": "custom", "model": "backup-a", "base_url": "https://a.example/v1"},
    {"provider": "custom", "model": "backup-b", "base_url": "https://b.example/v1"},
]


def _agent(*, routes=None, callback=None, interactive=True, auto=False):
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key", base_url="https://primary.example/v1", provider="custom",
            model="primary", quiet_mode=True, skip_context_files=True, skip_memory=True,
            skip_background_review=True,
            fallback_model=ROUTES if routes is None else routes,
            fallback_auto_activate=auto, fallback_selection_interactive=interactive,
            clarify_callback=callback,
        )
        agent.client = MagicMock()
        return agent


def _attempt(agent, **kwargs):
    try:
        return agent._try_activate_fallback(**kwargs)
    except ManualFallbackStopped:
        return False


def _choose(index):
    def callback(questions):
        question = questions[0]
        assert question["multi_select"] is False
        return {"outcome": "submitted", "answers": {question["qid"]: question["choices"][index]}}
    return callback


def _client(url="https://b.example/v1"):
    return MagicMock(base_url=url, api_key="test-fallback")


def test_only_submitted_route_is_resolved_and_cannot_cascade():
    callback = MagicMock(side_effect=_choose(1))
    agent = _agent(callback=callback)
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(_client(), None)) as resolve:
        assert _attempt(agent, reason=FailoverReason.rate_limit) is True
        assert _attempt(agent, reason=FailoverReason.rate_limit) is False
    assert (agent.provider, agent.model) == ("custom", "backup-b")
    assert resolve.call_count == callback.call_count == 1
    assert resolve.call_args.kwargs["explicit_base_url"] == "https://b.example/v1"
    assert agent._fallback_chain == ROUTES
    assert not agent._has_pending_fallback()


@pytest.mark.parametrize("outcome", ["cancelled", "timed_out", "undelivered", "submitted"])
@pytest.mark.parametrize("answer", [None, "", "not offered", ["backup-a"]])
def test_no_answer_or_unsubmitted_answer_authorizes_no_route(outcome, answer):
    callback = MagicMock(return_value={"outcome": outcome, "answers": {"fallback_route": answer}})
    agent = _agent(callback=callback)
    with patch("agent.auxiliary_client.resolve_provider_client") as resolve:
        assert _attempt(agent) is False
        assert _attempt(agent) is False
    resolve.assert_not_called()
    callback.assert_called_once()


@pytest.mark.parametrize("reply", [None, "1", [], {}, {"outcome": "submitted", "answers": []}])
def test_malformed_callback_reply_fails_closed(reply):
    agent = _agent(callback=MagicMock(return_value=reply))
    with patch("agent.auxiliary_client.resolve_provider_client") as resolve:
        assert _attempt(agent) is False
    resolve.assert_not_called()


def test_cancelled_partial_valid_answer_does_not_count_as_consent():
    def callback(questions):
        reply = _choose(0)(questions)
        reply["outcome"] = "cancelled"
        return reply
    agent = _agent(callback=callback)
    with patch("agent.auxiliary_client.resolve_provider_client") as resolve:
        assert _attempt(agent) is False
    resolve.assert_not_called()


@pytest.mark.parametrize("routes,interactive", [([], True), (ROUTES, False)])
def test_empty_or_noninteractive_never_prompts(routes, interactive):
    callback = MagicMock(side_effect=_choose(0))
    agent = _agent(routes=routes, callback=callback, interactive=interactive)
    with patch("agent.auxiliary_client.resolve_provider_client") as resolve:
        assert agent._has_pending_fallback() is False
        assert _attempt(agent) is False
    callback.assert_not_called()
    resolve.assert_not_called()


def test_failed_selected_resolution_never_tries_another_route():
    agent = _agent(callback=_choose(0))
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(None, None)) as resolve:
        assert _attempt(agent) is False
        assert _attempt(agent) is False
    assert resolve.call_count == 1
    assert agent._fallback_chain == ROUTES


def test_safe_origin_and_neutral_labels():
    route = {"provider": "custom", "model": "backup", "base_url": "https://user:secret@[2001:db8::1]:8443/token/v1?key=x#secret"}
    callback = MagicMock(return_value={"outcome": "cancelled", "answers": {}})
    agent = _agent(routes=[route], callback=callback)
    assert not _attempt(agent)
    choices = callback.call_args.args[0][0]["choices"]
    assert choices == ["Continue with backup via custom (https://[2001:db8::1]:8443)"]


@pytest.mark.parametrize("routes", [
    [{"provider": "custom", "model": "m", "base_url": f"https://b.example/path/{i}"} for i in range(2)],
    [{"provider": "custom", "model": f"m-{i}", "base_url": "https://b.example"} for i in range(5)],
])
def test_ambiguous_or_oversized_choices_do_not_prompt(routes):
    callback = MagicMock(side_effect=_choose(0))
    agent = _agent(routes=routes, callback=callback)
    with patch("agent.auxiliary_client.resolve_provider_client") as resolve:
        assert not _attempt(agent)
    callback.assert_not_called()
    resolve.assert_not_called()


def test_callback_attached_after_construction_is_supported():
    agent = _agent(callback=None, interactive=None)
    assert not agent._has_pending_fallback()
    agent.clarify_callback = _choose(0)
    assert agent._has_pending_fallback()


@pytest.mark.parametrize("gate", ["cooldown", "reset", "entitlement", "non_chat"])
def test_manual_route_expires_despite_automatic_restore_gates(gate):
    agent = _agent(callback=_choose(0))
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(_client(), None)):
        assert _attempt(agent)
    if gate == "cooldown":
        agent._rate_limited_until = float("inf")
    if gate == "entitlement":
        agent._entitlement_rejected_models = {("custom", "primary")}
    if gate == "non_chat":
        agent._primary_runtime["model"] = "acme/text-to-image"
    with (
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.fallback_cooldown.primary_reset_gate_blocks", return_value=(gate == "reset", None, False)),
    ):
        assert agent._restore_primary_runtime()
    assert agent.model == agent._primary_runtime["model"]
    assert agent.base_url == "https://primary.example/v1"


def test_cancelled_selection_cannot_enter_automatic_recovery_ladder():
    from agent.turn_recovery_autorecover import ladder_eligible
    agent = _agent(callback=MagicMock(return_value={"outcome": "cancelled", "answers": {}}))
    agent._auto_recovery_cycles = 5
    assert not _attempt(agent)
    assert not ladder_eligible(agent, SimpleNamespace(reason=FailoverReason.timeout))


def test_short_primary_reset_defers_prompt_without_consuming_consent():
    agent = _agent(callback=MagicMock(side_effect=_choose(0)))
    with patch("agent.fallback_cooldown.switch_deferred_by_reset", return_value=True):
        assert not _attempt(agent, reason=FailoverReason.rate_limit, reset_at=1)
    agent.clarify_callback.assert_not_called()
    assert agent._has_pending_fallback()


def test_partial_binding_failure_restores_primary_before_returning():
    agent = _agent(callback=_choose(0))
    with (
        patch("agent.auxiliary_client.resolve_provider_client", return_value=(_client(), None)),
        patch("agent.client_lifecycle._swap_fallback_clients", side_effect=RuntimeError("bind failed")),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        assert not _attempt(agent)
    assert (agent.provider, agent.model, agent.base_url) == ("custom", "primary", "https://primary.example/v1")
    assert agent._fallback_manual_declined


def test_required_restore_failure_never_publishes_previous_route():
    from agent.manual_fallback import prepare_turn_runtime
    agent = _agent(callback=_choose(0))
    agent._fallback_activated = True
    agent._fallback_manual_selected_index = 0
    publish = MagicMock()
    with patch.object(agent, "_restore_primary_runtime", return_value=False):
        with pytest.raises(RuntimeError, match="refusing to start"):
            prepare_turn_runtime(agent, publish)
    publish.assert_not_called()
    assert agent._fallback_manual_selected_index == 0


def test_required_publication_failure_clears_context_without_affecting_other_context():
    import contextvars
    from agent.auxiliary_client import _runtime_main_value, set_runtime_main
    from agent.turn_context import _publish_runtime_main
    from agent.manual_fallback import prepare_turn_runtime

    other = contextvars.Context()
    other.run(set_runtime_main, "other-provider", "other-model")
    agent = _agent(callback=_choose(0))
    agent._fallback_activated = True
    agent._fallback_manual_selected_index = 0
    with (
        patch.object(agent, "_restore_primary_runtime", return_value=True),
        patch("agent.auxiliary_client.set_runtime_main", side_effect=RuntimeError("publish failed")),
    ):
        with pytest.raises(RuntimeError, match="publish failed"):
            prepare_turn_runtime(agent, _publish_runtime_main)
    assert _runtime_main_value("provider") == ""
    assert other.run(_runtime_main_value, "provider") == "other-provider"


def test_real_two_turn_loop_restores_primary_and_rearms_selection():
    class RateLimitError(Exception):
        status_code = 429

        def __init__(self):
            super().__init__("rate limit exceeded")
            self.response = SimpleNamespace(headers={})
            self.body = {"error": {"message": "rate limit exceeded"}}

    callback = MagicMock(side_effect=_choose(1))
    agent = _agent(callback=callback)
    agent._api_max_retries = 1
    calls = []

    def api_call(_kwargs):
        calls.append((agent.provider, agent.model))
        if agent.model == "primary":
            raise RateLimitError()
        message = SimpleNamespace(content="Fallback answer.", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")],
                               model=agent.model, usage=None)

    with (
        patch.object(agent, "_interruptible_api_call", side_effect=api_call),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.agent_runtime_helpers.time.sleep"),
        patch("agent.auxiliary_client.resolve_provider_client", return_value=(_client(), None)),
        patch("agent.model_metadata.get_model_context_length", return_value=200_000),
    ):
        first = agent.run_conversation("First turn")
        second = agent.run_conversation("Second turn", conversation_history=first["messages"])
    assert first["completed"] and second["completed"]
    assert calls == [("custom", "primary"), ("custom", "backup-b")] * 2
    assert callback.call_count == 2


def _run_failure_turn(agent, call, resolve):
    with (
        patch.object(agent, "_interruptible_api_call", side_effect=call),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.agent_runtime_helpers.time.sleep"),
        patch("agent.auxiliary_client.resolve_provider_client", side_effect=resolve),
        patch("agent.model_metadata.get_model_context_length", return_value=200_000),
    ):
        return agent.run_conversation("A test turn")


def _provider_error(code):
    error = RuntimeError(f"Error code: {code} - provider unavailable")
    error.status_code = code
    error.response = SimpleNamespace(headers={})
    error.body = {"error": {"message": str(error)}}
    return error


@pytest.mark.parametrize("code", [401, 429, 500])
def test_real_loop_cancellation_is_interrupted_not_provider_failure(code):
    callback = MagicMock(return_value={"outcome": "cancelled", "answers": {}})
    agent = _agent(callback=callback)
    agent._api_max_retries = 1
    call = MagicMock(side_effect=_provider_error(code))
    resolve = MagicMock()
    result = _run_failure_turn(agent, call, resolve)
    assert result["interrupted"] is True
    assert result["completed"] is False
    assert not result.get("failed")
    assert "failure_retryable" not in result
    assert "cancelled" in result["final_response"]
    assert call.call_count == callback.call_count == 1
    resolve.assert_not_called()


def test_failed_selection_during_voice_turn_does_not_send_to_session_primary():
    agent = _agent(callback=_choose(0))
    agent._voice_turn_pending = True
    agent._api_max_retries = 1
    calls = []

    def api_call(_kwargs):
        calls.append((agent.model, agent.base_url))
        if agent.model == "voice-model":
            raise _provider_error(429)
        message = SimpleNamespace(content="Unselected primary answer", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=None)

    def resolve(_provider, *, model, **_kwargs):
        return (_client("https://voice.example/v1"), model) if model == "voice-model" else (None, None)

    with patch("agent.auxiliary_task_config._get_auxiliary_task_config", return_value={
        "provider": "custom", "model": "voice-model", "base_url": "https://voice.example/v1",
    }):
        result = _run_failure_turn(agent, api_call, resolve)
    assert calls == [("voice-model", "https://voice.example/v1")]
    assert result["failed"] and not result["completed"]
    assert not result["failure_retryable"]
    assert agent.model == "primary"  # voice cleanup still restores the session after the stopped turn


def test_no_client_failure_preserves_preexisting_voice_route_until_turn_cleanup():
    from agent.route_binding import bind_route_entry
    agent = _agent(callback=_choose(0))
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(_client("https://voice.example/v1"), None)):
        assert bind_route_entry(agent, {"provider": "custom", "model": "voice-model"}, "custom", "voice-model")
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(None, None)):
        with pytest.raises(ManualFallbackStopped, match="could not be activated"):
            agent._try_activate_fallback()
    assert agent.model == "voice-model"
    assert agent.base_url == "https://voice.example/v1"


def test_stop_racing_submitted_selection_is_settled_and_does_not_poison_next_turn():
    agent = _agent()
    agent._api_max_retries = 1

    def callback(questions):
        agent.interrupt()
        return _choose(0)(questions)

    agent.clarify_callback = callback
    call = MagicMock(side_effect=_provider_error(429))
    resolve = MagicMock()
    result = _run_failure_turn(agent, call, resolve)
    assert result["interrupted"] and not result.get("failed")
    assert not agent._interrupt_requested
    assert call.call_count == 1
    resolve.assert_not_called()


def test_redirect_racing_selection_is_applied_without_selecting_a_route():
    agent = _agent()
    agent._api_max_retries = 1
    calls = []

    def callback(questions):
        # A real redirect is admitted while the model-request marker is set.
        agent._model_request_active.set()
        try:
            assert agent.redirect("Use this correction")
        finally:
            agent._model_request_active.clear()
        return _choose(0)(questions)

    def call(kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise _provider_error(429)
        message = SimpleNamespace(content="Corrected answer", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=None)

    agent.clarify_callback = callback
    resolve = MagicMock()
    result = _run_failure_turn(agent, call, resolve)
    assert result["completed"] and not result.get("interrupted")
    assert len(calls) == 2
    assert "Use this correction" in str(calls[1]["messages"])
    resolve.assert_not_called()


def test_manual_policy_cannot_relabel_automatic_startup_fallback_as_primary():
    from agent.manual_fallback import prepare_turn_runtime
    agent = _agent(auto=True)
    agent._fallback_bootstrap_active = True
    agent._fallback_auto_activate = False
    publish = MagicMock()
    with patch.object(agent, "_restore_primary_runtime") as restore:
        with pytest.raises(RuntimeError, match="primary was unavailable at startup"):
            prepare_turn_runtime(agent, publish)
    restore.assert_not_called()
    publish.assert_not_called()

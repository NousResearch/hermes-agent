"""Regression tests for the deterministic local-processing retry boundary."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _compiled_raise(filename: str):
    """Build a raiser whose traceback has the requested module filename."""
    namespace = {}
    exec(
        compile(
            "def raise_error():\n"
            "    raise TypeError(\"expected string or bytes-like object, got 'list'\")\n",
            filename,
            "exec",
        ),
        namespace,
    )
    return namespace["raise_error"]


def _compiled_call(filename: str, target):
    """Build a wrapper whose traceback has the requested module filename."""
    namespace = {"target": target}
    exec(
        compile(
            "def call_target():\n"
            "    return target()\n",
            filename,
            "exec",
        ),
        namespace,
    )
    return namespace["call_target"]


def _capture_exception(callback):
    try:
        callback()
    except Exception as error:
        return error
    raise AssertionError("callback did not raise")


def _raise_nonlocal_type_error(*_args, **_kwargs):
    raise TypeError("expected string or bytes-like object, got 'list'")


def _response(content="provider response"):
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _make_agent(tmp_path, monkeypatch, max_iterations=4) -> AIAgent:
    hermes_home = tmp_path / ".hermes"
    (hermes_home / "logs").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    with (
        patch("run_agent._hermes_home", hermes_home),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://example.com/v1",
            provider="openai",
            api_mode="chat_completions",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    agent.client = MagicMock()
    agent.client.chat.completions.create.return_value = _response()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.max_iterations = max_iterations
    return agent


def test_allowlisted_local_helper_traceback_is_classified_local():
    from agent.conversation_loop import _is_local_processing_error

    error = _capture_exception(_compiled_raise("agent_runtime_helpers.py"))

    assert _is_local_processing_error(error) is True


def test_api_helper_traceback_is_not_local_even_with_shared_module_frame():
    from agent.conversation_loop import _is_local_processing_error

    api_helper = _compiled_raise("chat_completion_helpers.py")
    shared_helper = _compiled_call("message_content.py", api_helper)
    error = _capture_exception(shared_helper)

    assert _is_local_processing_error(error) is False


def test_same_type_and_message_without_allowlisted_traceback_is_not_local():
    from agent.conversation_loop import _is_local_processing_error

    error = _capture_exception(_raise_nonlocal_type_error)

    assert _is_local_processing_error(error) is False


def test_outer_loop_stops_after_one_response_for_local_processing_error(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path, monkeypatch)
    local_helper = _compiled_raise("message_content.py")

    def raise_local_processing_error(*_args, **_kwargs):
        local_helper()

    with (
        patch(
            "agent.conversation_loop.has_incomplete_scratchpad",
            side_effect=raise_local_processing_error,
        ),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("inspect this image")

    assert agent.client.chat.completions.create.call_count == 1
    assert result["api_calls"] == 1
    assert result["turn_exit_reason"].startswith("local_processing_error(")


def test_outer_loop_retries_nonlocal_repeated_error_until_bound(tmp_path, monkeypatch):
    agent = _make_agent(tmp_path, monkeypatch, max_iterations=4)

    with (
        patch(
            "agent.conversation_loop.has_incomplete_scratchpad",
            side_effect=_raise_nonlocal_type_error,
        ),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("inspect this image")

    assert agent.client.chat.completions.create.call_count == agent.max_iterations - 1
    assert result["api_calls"] == agent.max_iterations - 1
    assert result["turn_exit_reason"].startswith("error_near_max_iterations(")

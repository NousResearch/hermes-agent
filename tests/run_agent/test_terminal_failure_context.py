"""Terminal API failure messages carry provider/model/endpoint context.

Direct coverage for both changed terminal paths in ``run_conversation``
(#66352 review): the non-retryable abort and the retries-exhausted path
append a "(provider: …, model: …, endpoint: …)" line to ``final_response``,
with config-sized fields clipped, while the structured ``error`` field keeps
the bare summary (cron notifications and CLI output are unchanged).
"""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_agent(base_url="https://api.openai.com/v1", model="gpt-5.5") -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url=base_url,
            provider="openai",
            api_mode="chat_completions",
            model=model,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    a.client = MagicMock()
    a._cached_system_prompt = "You are helpful."
    a._use_prompt_caching = False
    a.tool_delay = 0
    a.compression_enabled = False
    a.save_trajectories = False
    return a


def _run(agent):
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("time.sleep"),
    ):
        return agent.run_conversation("hello")


def test_non_retryable_final_response_has_context_and_bare_error_field():
    agent = _make_agent()
    err = Exception("Invalid request")
    err.status_code = 403
    agent.client.chat.completions.create.side_effect = err

    result = _run(agent)

    assert agent.client.chat.completions.create.called
    assert result.get("failed") is True
    final = result.get("final_response") or ""
    assert "(provider: openai, model: gpt-5.5, endpoint: https://api.openai.com/v1)" in final
    # The structured error field stays bare — no context suffix.
    error = result.get("error") or ""
    assert "(provider:" not in error


def test_retries_exhausted_final_response_has_context_and_bare_error_field():
    agent = _make_agent()
    agent._api_max_retries = 1
    err = Exception("Internal server error")
    err.status_code = 500
    agent.client.chat.completions.create.side_effect = err

    result = _run(agent)

    assert agent.client.chat.completions.create.called
    final = result.get("final_response") or ""
    assert "API call failed after 1 retries:" in final
    assert "(provider: openai, model: gpt-5.5, endpoint: https://api.openai.com/v1)" in final
    error = result.get("error") or ""
    assert "(provider:" not in error


def test_context_fields_are_clipped_for_pathological_config():
    """Config-sized fields must not produce an unbounded context line."""
    giant = "https://" + "a" * 500 + ".example.com/v1"
    agent = _make_agent(base_url=giant, model="m" * 300)
    err = Exception("Invalid request")
    err.status_code = 403
    agent.client.chat.completions.create.side_effect = err

    result = _run(agent)

    final = result.get("final_response") or ""
    context_line = next(
        (line for line in final.splitlines() if line.startswith("(provider:")), ""
    )
    assert context_line, final
    assert "…" in context_line
    # Each field is clipped to ~120 chars; the whole line stays bounded.
    assert len(context_line) < 450
    assert giant not in final

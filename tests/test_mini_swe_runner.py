from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _gpt_turns(result):
    return [t["value"] for t in result["conversations"] if t["from"] == "gpt"]


def test_run_task_captures_reasoning_on_final_message():
    """Reasoning models' chain-of-thought must land in the trajectory.

    _convert_to_hermes_format wraps a message's ``reasoning`` in
    ``<think>...</think>``, but run_task never copied ``reasoning`` off the
    assistant message, so that branch was dead and every emitted trajectory
    silently dropped the reasoning (and batch_runner discards zero-reasoning
    samples downstream).
    """
    with patch("openai.OpenAI") as mock_openai:
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content="done",
                tool_calls=[],
                reasoning="first I inspect, then I answer",
            ))]
        )
        mock_openai.return_value = client

        from mini_swe_runner import MiniSWERunner

        runner = MiniSWERunner(
            model="deepseek-reasoner",
            base_url="https://openrouter.ai/api/v1",
            api_key="test-key",
            env_type="local",
            max_iterations=1,
        )
        runner._create_env = MagicMock()
        runner._cleanup_env = MagicMock()

        result = runner.run_task("2+2")

    assert result["completed"] is True
    assert any("<think>first I inspect, then I answer</think>" in v for v in _gpt_turns(result))


def test_run_task_captures_reasoning_content_on_tool_call_message():
    """The ``.reasoning_content`` alias (DeepSeek/Kimi) is captured on a
    tool-call turn too."""
    tool_call = SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name="terminal", arguments='{"command": "echo hi"}'),
    )
    with patch("openai.OpenAI") as mock_openai:
        client = MagicMock()
        responses = [
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content=None,
                tool_calls=[tool_call],
                reasoning_content="I will run the command",
            ))]),
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content="all set", tool_calls=[],
            ))]),
        ]
        client.chat.completions.create.side_effect = responses
        mock_openai.return_value = client

        from mini_swe_runner import MiniSWERunner

        runner = MiniSWERunner(
            model="kimi-k2-thinking",
            base_url="https://openrouter.ai/api/v1",
            api_key="test-key",
            env_type="local",
            max_iterations=2,
        )
        runner._create_env = MagicMock()
        runner._cleanup_env = MagicMock()
        runner._execute_command = MagicMock(
            return_value={"output": "hi", "exit_code": 0, "error": ""}
        )

        result = runner.run_task("say hi")

    assert any("<think>I will run the command</think>" in v for v in _gpt_turns(result))


def test_run_task_kimi_omits_temperature():
    """Kimi models should NOT have client-side temperature overrides.

    The Kimi gateway selects the correct temperature server-side.
    """
    with patch("openai.OpenAI") as mock_openai:
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=[]))]
        )
        mock_openai.return_value = client

        from mini_swe_runner import MiniSWERunner

        runner = MiniSWERunner(
            model="kimi-for-coding",
            base_url="https://api.kimi.com/coding/v1",
            api_key="test-key",
            env_type="local",
            max_iterations=1,
        )
        runner._create_env = MagicMock()
        runner._cleanup_env = MagicMock()

        result = runner.run_task("2+2")

    assert result["completed"] is True
    assert "temperature" not in client.chat.completions.create.call_args.kwargs


def test_run_task_public_moonshot_kimi_k2_5_omits_temperature():
    """kimi-k2.5 on the public Moonshot API should not get a forced temperature."""
    with patch("openai.OpenAI") as mock_openai:
        client = MagicMock()
        client.base_url = "https://api.moonshot.ai/v1"
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=[]))]
        )
        mock_openai.return_value = client

        from mini_swe_runner import MiniSWERunner

        runner = MiniSWERunner(
            model="kimi-k2.5",
            base_url="https://api.moonshot.ai/v1",
            api_key="test-key",
            env_type="local",
            max_iterations=1,
        )
        runner._create_env = MagicMock()
        runner._cleanup_env = MagicMock()

        result = runner.run_task("2+2")

    assert result["completed"] is True
    assert "temperature" not in client.chat.completions.create.call_args.kwargs

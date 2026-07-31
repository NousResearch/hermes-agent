from types import SimpleNamespace
from unittest.mock import MagicMock, patch


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


def test_run_task_bounds_tool_result_before_second_model_call():
    tool_call = SimpleNamespace(
        id="call_large",
        type="function",
        function=SimpleNamespace(
            name="terminal",
            arguments='{"command":"produce-large-output"}',
        ),
    )
    first = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="", tool_calls=[tool_call])
            )
        ]
    )
    second = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="done", tool_calls=[])
            )
        ]
    )

    with patch("openai.OpenAI") as mock_openai:
        client = MagicMock()
        client.chat.completions.create.side_effect = [first, second]
        mock_openai.return_value = client

        from mini_swe_runner import MiniSWERunner

        runner = MiniSWERunner(
            model="test-model",
            base_url="https://example.invalid/v1",
            api_key="test-key",
            env_type="local",
            max_iterations=2,
        )
        runner._create_env = MagicMock()
        runner._cleanup_env = MagicMock()
        runner._execute_command = MagicMock(
            return_value={
                "output": "[]" * 8_000,
                "exit_code": 0,
                "error": None,
            }
        )

        runner.run_task("run the command")

    second_messages = client.chat.completions.create.call_args_list[1].kwargs["messages"]
    tool_message = next(msg for msg in second_messages if msg.get("role") == "tool")
    assert len(tool_message["content"].encode("utf-8")) <= 10_000
    assert "_tool_result_budget" not in tool_message


def test_runner_exposes_call_local_result_budget_schema():
    with patch("openai.OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()

        from mini_swe_runner import MiniSWERunner

        runner = MiniSWERunner(
            model="test-model",
            base_url="https://example.invalid/v1",
            api_key="test-key",
        )

    budget = runner.tools[0]["function"]["parameters"]["properties"][
        "result_token_limit"
    ]
    assert budget["default"] == 10_000
    assert budget["maximum"] == 32_000

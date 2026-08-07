import json

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


def _extract_tool_responses(trajectory):
    """Parse the JSON object inside each <tool_response> of the tool turn(s)."""
    responses = []
    for turn in trajectory:
        if turn.get("from") != "tool":
            continue
        for block in turn["value"].split("<tool_response>"):
            block = block.replace("</tool_response>", "").strip()
            if block:
                responses.append(json.loads(block))
    return responses


def test_convert_matches_tool_responses_by_id_not_position():
    """Tool responses must be paired to their call by ``tool_call_id``.

    The internal message list can carry a ``None``/non-dict ``tool_calls``
    entry (the conversion loop explicitly skips those), and providers may
    return tool responses out of order. Pairing responses to calls positionally
    therefore both raised ``TypeError`` on the ``None`` entry and labelled
    responses with the wrong tool name. Matching by id fixes both.
    """
    from mini_swe_runner import MiniSWERunner

    # Build the instance without going through __init__ (no network client),
    # since _convert_to_hermes_format only needs ``self.tools``.
    runner = MiniSWERunner.__new__(MiniSWERunner)
    runner.tools = []

    messages = [
        {"role": "user", "content": "do the task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                None,  # malformed entry — previously crashed positional indexing
                {"id": "call_a", "function": {"name": "terminal", "arguments": "{}"}},
                {"id": "call_b", "function": {"name": "browser", "arguments": "{}"}},
            ],
        },
        # Responses arrive in the opposite order to the calls.
        {"role": "tool", "tool_call_id": "call_b", "content": "B-result"},
        {"role": "tool", "tool_call_id": "call_a", "content": "A-result"},
    ]

    trajectory = runner._convert_to_hermes_format(messages, "do the task", True)

    responses = _extract_tool_responses(trajectory)
    by_id = {r["tool_call_id"]: r["name"] for r in responses}

    # Each response is labelled with the name of the call it actually answers,
    # regardless of arrival order, and no response is mislabelled "unknown".
    assert by_id == {"call_b": "browser", "call_a": "terminal"}


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

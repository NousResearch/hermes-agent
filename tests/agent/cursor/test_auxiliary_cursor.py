from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.auxiliary_client import CursorAuxiliaryClient, call_llm


def test_cursor_auxiliary_client_routes_through_agent_api():
    client = CursorAuxiliaryClient("token", "default", "https://api2.cursor.sh")
    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content="Short title here",
                    tool_calls=None,
                ),
                finish_reason="stop",
            )
        ],
        model="default",
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )

    with patch(
        "agent.cursor.stream_client.run_cursor_agent_turn",
        return_value=fake_response,
    ) as mock_turn:
        response = client.chat.completions.create(
            model="default",
            messages=[
                {"role": "system", "content": "Generate a title"},
                {"role": "user", "content": "User: hi\n\nAssistant: hello"},
            ],
            max_tokens=50,
        )

    assert response.choices[0].message.content == "Short title here"
    mock_turn.assert_called_once()
    assert mock_turn.call_args.kwargs["tools"] is None
    assert mock_turn.call_args.kwargs["api_key"] == "token"


def test_call_llm_title_generation_uses_cursor_main_runtime():
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Greeting"))],
    )

    with (
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", "", "", "", ""),
        ),
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(fake_client, "claude-4-sonnet"),
        ) as mock_resolve,
    ):
        result = call_llm(
            task="title_generation",
            messages=[{"role": "user", "content": "hello"}],
            main_runtime={
                "provider": "cursor",
                "model": "claude-4-sonnet",
                "base_url": "https://api2.cursor.sh",
                "api_key": "cursor-token",
                "api_mode": "cursor_agent",
            },
        )

    assert result.choices[0].message.content == "Greeting"
    assert mock_resolve.call_args.kwargs["main_runtime"]["provider"] == "cursor"

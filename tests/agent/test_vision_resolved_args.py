"""Test that call_llm vision path passes resolved provider args, not raw ones."""

from unittest.mock import patch, MagicMock


def test_vision_call_uses_resolved_provider_args():
    """Resolved provider/model/key/url from config must reach resolve_vision_provider_client."""
    from agent.auxiliary_client import call_llm

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="description"))],
        usage=MagicMock(prompt_tokens=10, completion_tokens=5),
    )

    with patch(
        "agent.auxiliary_client._resolve_task_provider_model",
        return_value=("my-resolved-provider", "my-resolved-model", "http://resolved", "resolved-key", "chat_completions"),
    ), patch(
        "agent.auxiliary_client.resolve_vision_provider_client",
        return_value=("my-resolved-provider", fake_client, "my-resolved-model"),
    ) as mock_vision:
        call_llm(
            "vision",
            provider="raw-provider",
            model="raw-model",
            base_url="http://raw",
            api_key="raw-key",
            messages=[{"role": "user", "content": "describe this"}],
        )

    # The resolved values must be passed, not the raw call_llm arguments
    call_args = mock_vision.call_args
    assert call_args.kwargs["provider"] == "my-resolved-provider"
    assert call_args.kwargs["model"] == "my-resolved-model"
    assert call_args.kwargs["base_url"] == "http://resolved"
    assert call_args.kwargs["api_key"] == "resolved-key"


def test_vision_base_url_override_keeps_explicit_provider():
    """Explicit provider should still drive credential resolution with custom base_url."""
    from agent.auxiliary_client import resolve_vision_provider_client

    fake_client = MagicMock()
    with patch(
        "agent.auxiliary_client._resolve_task_provider_model",
        return_value=(
            "zai",
            "glm-4v",
            "https://open.bigmodel.cn/api/paas/v4",
            None,
            "chat_completions",
        ),
    ), patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(fake_client, "glm-4v"),
    ) as mock_resolve:
        provider, client, model = resolve_vision_provider_client()

    assert provider == "zai"
    assert client is fake_client
    assert model == "glm-4v"
    assert mock_resolve.call_args.args[0] == "zai"
    assert mock_resolve.call_args.kwargs["explicit_base_url"] == "https://open.bigmodel.cn/api/paas/v4"


def test_vision_named_custom_provider_honors_explicit_api_key():
    """Explicit api_key configured on auxiliary.vision must reach the named custom provider client."""
    from agent.auxiliary_client import resolve_vision_provider_client

    fake_client = MagicMock()
    with patch(
        "agent.auxiliary_client._resolve_task_provider_model",
        return_value=(
            "my-custom-vision",
            "vision-model",
            None,
            "sk-explicit-task-key",
            "chat_completions",
        ),
    ), patch(
        "agent.auxiliary_client._get_cached_client",
        return_value=(fake_client, "vision-model"),
    ) as mock_cached:
        provider, client, model = resolve_vision_provider_client(
            provider="my-custom-vision",
            model="vision-model",
            api_key="sk-explicit-task-key",
        )

    assert provider == "my-custom-vision"
    assert client is fake_client
    assert model == "vision-model"
    assert mock_cached.call_args.kwargs["api_key"] == "sk-explicit-task-key"


def test_resolve_provider_client_named_custom_honors_explicit_api_key():
    """Named custom provider resolution in resolve_provider_client must use explicit_api_key if supplied."""
    from agent.auxiliary_client import resolve_provider_client

    custom_entry = {
        "name": "my-custom-provider",
        "base_url": "https://custom.api.example.com/v1",
        "api_mode": "chat_completions",
        "model": "custom-model",
    }

    with patch("hermes_cli.runtime_provider._get_named_custom_provider", return_value=custom_entry), patch(
        "agent.auxiliary_client._create_openai_client"
    ) as mock_create:
        mock_create.return_value = MagicMock()
        resolve_provider_client("my-custom-provider", explicit_api_key="sk-explicit-override-key")

    assert mock_create.call_args.kwargs["api_key"] == "sk-explicit-override-key"


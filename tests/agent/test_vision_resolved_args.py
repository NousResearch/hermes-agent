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


def test_zai_vision_prefers_active_coding_plan_endpoint():
    """The active Z.ai Coding Plan session must precede pay-as-you-go URLs."""
    from agent.auxiliary_client import resolve_vision_provider_client

    client = MagicMock()
    runtime = {
        "provider": "zai",
        "model": "glm-5.3-flash",
        "base_url": "https://api.z.ai/api/coding/paas/v4",
        "api_key": "coding-plan-key",
    }
    with patch(
        "agent.auxiliary_client._resolve_task_provider_model",
        return_value=("zai", "glm-5.3-flash", None, None, None),
    ), patch(
        "agent.auxiliary_client._get_cached_client",
        return_value=(client, "glm-5.3-flash"),
    ) as mock_cached:
        provider, resolved_client, model = resolve_vision_provider_client(main_runtime=runtime)

    assert (provider, resolved_client, model) == ("zai", client, "glm-5.3-flash")
    assert mock_cached.call_args.kwargs["base_url"] == runtime["base_url"]
    assert mock_cached.call_args.kwargs["api_mode"] == "chat_completions"


def test_zai_vision_falls_back_to_pay_as_you_go_urls_after_coding_plan():
    """An unavailable active Coding Plan endpoint keeps the legacy fallback ordering."""
    from agent.auxiliary_client import resolve_vision_provider_client

    client = MagicMock()
    runtime = {
        "provider": "zai",
        "model": "glm-5.3-flash",
        "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
        "api_key": "coding-plan-key",
    }
    with patch(
        "agent.auxiliary_client._resolve_task_provider_model",
        return_value=("zai", "glm-5.3-flash", None, None, None),
    ), patch(
        "agent.auxiliary_client._get_cached_client",
        side_effect=[(None, None), (client, "glm-5.3-flash")],
    ) as mock_cached:
        provider, resolved_client, model = resolve_vision_provider_client(main_runtime=runtime)

    assert (provider, resolved_client, model) == ("zai", client, "glm-5.3-flash")
    assert [call.kwargs["base_url"] for call in mock_cached.call_args_list] == [
        runtime["base_url"],
        "https://open.bigmodel.cn/api/paas/v4",
    ]

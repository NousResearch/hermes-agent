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


def test_vision_api_key_only_path_threads_api_key_into_cache_call():
    """Regression for #64242 (re-scoped per sweeper review).

    When ``auxiliary.vision`` (or another task) is configured with a provider
    + api_key but NO base_url, ``_resolve_task_provider_model`` returns
    ``(provider, model, None, api_key, mode)``. ``resolve_vision_provider_client``
    skips the ``if resolved_base_url:`` branch (line 5594) and falls through
    to the bottom ``_get_cached_client`` call (line 5772).

    Without threading ``api_key`` into that call, two profiles configured
    with the same provider+model but DIFFERENT api_keys collide in the cache
    (the cache key defaults to empty string for api_key). The fix passes
    ``resolved_api_key`` through so each profile gets its own cache entry.
    """
    from agent.auxiliary_client import resolve_vision_provider_client

    fake_client = MagicMock()
    captured_calls: list[dict] = []

    def _capture_get_cached_client(*args, **kwargs):
        captured_calls.append(kwargs)
        return fake_client, "resolved-model"

    with patch(
        "agent.auxiliary_client._resolve_task_provider_model",
        return_value=(
            "custom",
            "some-vision-model",
            None,                # resolved_base_url — None (the API-key-only path)
            "sk_profile_A",     # resolved_api_key
            "chat_completions",
        ),
    ), patch(
        "agent.auxiliary_client._get_cached_client",
        side_effect=_capture_get_cached_client,
    ):
        provider, client, model = resolve_vision_provider_client(
            api_key="sk_profile_A",
        )

    assert provider == "custom"
    assert client is fake_client
    # Exactly one _get_cached_client call, and it MUST carry the resolved api_key.
    # A None/empty api_key here is the collision bug: two profiles sharing one
    # provider+model would resolve to the same cache entry, and the second
    # profile's vision call would authenticate with the first profile's key.
    assert len(captured_calls) == 1, captured_calls
    assert captured_calls[0]["api_key"] == "sk_profile_A", (
        f"_get_cached_client was called with api_key="
        f"{captured_calls[0]['api_key']!r}; expected 'sk_profile_A'. "
        "This is the cache-collision vector: api_key-only configs (no base_url) "
        "must thread api_key into the cache key or profiles cross-authenticate."
    )
    # And base_url is passed as None (not silently dropped), so the cache key
    # is well-formed even on the api-key-only path.
    assert captured_calls[0]["base_url"] is None


def test_vision_api_key_only_cache_keys_differentiate_by_api_key():
    """Two consecutive resolve_vision_provider_client calls with the same
    provider+model but DIFFERENT api_keys must produce DIFFERENT cache keys,
    so profile A's cached client is not handed back to profile B's call.

    This is the actual collision the #64242 fix prevents — verified at the
    cache-key layer rather than just the call-args layer."""
    from agent.auxiliary_client import _client_cache_key

    key_a = _client_cache_key(
        "custom", async_mode=False, base_url=None,
        api_key="sk_A", api_mode="chat_completions",
        main_runtime=None, is_vision=True, model="some-vision-model",
    )
    key_b = _client_cache_key(
        "custom", async_mode=False, base_url=None,
        api_key="sk_B", api_mode="chat_completions",
        main_runtime=None, is_vision=True, model="some-vision-model",
    )
    key_empty = _client_cache_key(
        "custom", async_mode=False, base_url=None,
        api_key=None, api_mode="chat_completions",
        main_runtime=None, is_vision=True, model="some-vision-model",
    )

    assert key_a != key_b, "different api_keys must produce different cache keys"
    # Sanity: the empty-api_key path is what the pre-fix code produced. If the
    # fix stops threading api_key through, both profile A and profile B would
    # hit this empty-key entry and cross-authenticate.
    assert key_empty != key_a
    assert key_empty != key_b

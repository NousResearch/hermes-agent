"""Attribution default_headers applied per provider via base-URL detection."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


@patch("run_agent.OpenAI")
def test_openrouter_base_url_applies_or_headers(mock_openai):
    mock_openai.return_value = MagicMock()
    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )

    agent._apply_client_headers_for_base_url("https://openrouter.ai/api/v1")

    headers = agent._client_kwargs["default_headers"]
    assert headers["HTTP-Referer"] == "https://hermes-agent.nousresearch.com"
    assert headers["X-Title"] == "Hermes Agent"


@patch("run_agent.OpenAI")
def test_routermint_base_url_applies_user_agent_header(mock_openai):
    mock_openai.return_value = MagicMock()
    agent = AIAgent(
        api_key="test-key",
        base_url="https://api.routermint.com/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )

    agent._apply_client_headers_for_base_url("https://api.routermint.com/v1")

    headers = agent._client_kwargs["default_headers"]
    assert headers["User-Agent"].startswith("HermesAgent/")


@patch("run_agent.OpenAI")
def test_nvidia_cloud_base_url_applies_billing_origin_header(mock_openai):
    mock_openai.return_value = MagicMock()
    agent = AIAgent(
        api_key="test-key",
        base_url="https://integrate.api.nvidia.com/v1",
        model="nvidia/test-model",
        provider="nvidia",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )

    assert agent._client_kwargs["default_headers"]["X-BILLING-INVOKE-ORIGIN"] == "HermesAgent"

    agent._apply_client_headers_for_base_url("https://integrate.api.nvidia.com/v1")

    headers = agent._client_kwargs["default_headers"]
    assert headers["X-BILLING-INVOKE-ORIGIN"] == "HermesAgent"


@patch("run_agent.OpenAI")
def test_nvidia_local_base_url_does_not_apply_billing_origin_header(mock_openai):
    mock_openai.return_value = MagicMock()
    agent = AIAgent(
        api_key="test-key",
        base_url="https://integrate.api.nvidia.com/v1",
        model="nvidia/test-model",
        provider="nvidia",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._client_kwargs["default_headers"] = {
        "X-BILLING-INVOKE-ORIGIN": "HermesAgent",
    }

    agent._apply_client_headers_for_base_url("http://localhost:8000/v1")

    assert "default_headers" not in agent._client_kwargs


@patch("run_agent.OpenAI")
def test_routed_client_preserves_openai_sdk_custom_headers(mock_openai):
    mock_openai.return_value = MagicMock()
    routed_client = SimpleNamespace(
        api_key="test-key",
        base_url="https://integrate.api.nvidia.com/v1",
        _custom_headers={"X-BILLING-INVOKE-ORIGIN": "HermesAgent"},
    )

    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(
        routed_client,
        "nvidia/test-model",
    )):
        agent = AIAgent(
            provider="nvidia",
            model="nvidia/test-model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    headers = agent._client_kwargs["default_headers"]
    assert headers["X-BILLING-INVOKE-ORIGIN"] == "HermesAgent"


@patch("run_agent.OpenAI")
def test_routed_client_preserves_openai_sdk_default_headers(mock_openai):
    mock_openai.return_value = MagicMock()
    routed_client = SimpleNamespace(
        api_key="test-key",
        base_url="https://api.githubcopilot.com",
        default_headers={"copilot-integration-id": "vscode-chat"},
    )

    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(
        routed_client,
        "claude-opus-4.7",
    )):
        agent = AIAgent(
            provider="copilot",
            model="claude-opus-4.7",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    headers = agent._client_kwargs["default_headers"]
    assert headers["copilot-integration-id"] == "vscode-chat"


@patch("run_agent.OpenAI")
def test_gmi_base_url_picks_up_profile_user_agent(mock_openai):
    """GMI declares User-Agent on its ProviderProfile.default_headers.

    The ``_apply_client_headers_for_base_url`` else-branch looks up the
    provider profile and applies its default_headers, so no GMI-specific
    branch is needed in run_agent.
    """
    mock_openai.return_value = MagicMock()
    agent = AIAgent(
        api_key="test-key",
        base_url="https://api.gmi-serving.com/v1",
        model="test/model",
        provider="gmi",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )

    agent._apply_client_headers_for_base_url("https://api.gmi-serving.com/v1")

    headers = agent._client_kwargs["default_headers"]
    assert headers["User-Agent"].startswith("HermesAgent/")


@patch("run_agent.OpenAI")
def test_unknown_base_url_clears_default_headers(mock_openai):
    mock_openai.return_value = MagicMock()
    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._client_kwargs["default_headers"] = {"X-Stale": "yes"}

    agent._apply_client_headers_for_base_url("https://api.example.com/v1")

    assert "default_headers" not in agent._client_kwargs


@patch("run_agent.OpenAI")
def test_openrouter_headers_include_response_cache_when_enabled(mock_openai):
    """When openrouter.response_cache is True, the cache header is injected."""
    mock_openai.return_value = MagicMock()
    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )

    with patch("hermes_cli.config.load_config", return_value={
        "openrouter": {"response_cache": True, "response_cache_ttl": 600},
    }):
        agent._apply_client_headers_for_base_url("https://openrouter.ai/api/v1")

    headers = agent._client_kwargs["default_headers"]
    assert headers["HTTP-Referer"] == "https://hermes-agent.nousresearch.com"
    assert headers["X-OpenRouter-Cache"] == "true"
    assert headers["X-OpenRouter-Cache-TTL"] == "600"


# ---------------------------------------------------------------------------
# model.default_headers — user-configured overrides (#40033)
# ---------------------------------------------------------------------------


@patch("run_agent.OpenAI")
def test_user_default_headers_override_sdk_user_agent(mock_openai):
    """``model.default_headers`` lets a custom endpoint swap the OpenAI SDK
    User-Agent that some gateways/WAFs reject (the #40033 reproduction)."""
    mock_openai.return_value = MagicMock()
    agent = AIAgent(
        api_key="test-key",
        base_url="http://localhost:8080/v1",
        model="my-custom-model",
        provider="custom",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )

    with patch("hermes_cli.config.load_config", return_value={
        "model": {"default_headers": {"User-Agent": "curl/8.7.1", "X-Extra": "1"}},
    }):
        agent._apply_client_headers_for_base_url("http://localhost:8080/v1")

    headers = agent._client_kwargs["default_headers"]
    assert headers["User-Agent"] == "curl/8.7.1"
    assert headers["X-Extra"] == "1"


@patch("run_agent.OpenAI")
def test_user_default_headers_win_over_provider_defaults(mock_openai):
    """User headers take precedence but leave untouched provider defaults intact."""
    mock_openai.return_value = MagicMock()
    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )

    with patch("hermes_cli.config.load_config", return_value={
        "model": {"default_headers": {"X-Title": "MyApp"}},
    }):
        agent._apply_client_headers_for_base_url("https://openrouter.ai/api/v1")

    headers = agent._client_kwargs["default_headers"]
    assert headers["X-Title"] == "MyApp"  # user override wins
    assert headers["HTTP-Referer"] == "https://hermes-agent.nousresearch.com"  # default preserved


@patch("run_agent.OpenAI")
def test_no_user_default_headers_leaves_provider_defaults_untouched(mock_openai):
    mock_openai.return_value = MagicMock()
    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )

    with patch("hermes_cli.config.load_config", return_value={"model": {}}):
        agent._apply_client_headers_for_base_url("https://openrouter.ai/api/v1")

    headers = agent._client_kwargs["default_headers"]
    assert headers["HTTP-Referer"] == "https://hermes-agent.nousresearch.com"
    assert "User-Agent" not in headers  # nothing injected when unconfigured


@patch("run_agent.OpenAI")
def test_user_default_headers_skipped_for_anthropic_mode(mock_openai):
    """Anthropic/Bedrock modes don't use the OpenAI client — never touched."""
    mock_openai.return_value = MagicMock()
    agent = AIAgent(
        api_key="test-key",
        base_url="http://localhost:8080/v1",
        model="my-custom-model",
        provider="custom",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "anthropic_messages"
    agent._client_kwargs = {}

    with patch("hermes_cli.config.load_config", return_value={
        "model": {"default_headers": {"User-Agent": "curl/8.7.1"}},
    }):
        agent._apply_user_default_headers()

    assert "default_headers" not in agent._client_kwargs


@patch("run_agent.OpenAI")
def test_openrouter_headers_no_cache_when_disabled(mock_openai):
    """When openrouter.response_cache is False, no cache headers are sent."""
    mock_openai.return_value = MagicMock()
    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )

    with patch("hermes_cli.config.load_config", return_value={
        "openrouter": {"response_cache": False},
    }):
        agent._apply_client_headers_for_base_url("https://openrouter.ai/api/v1")

    headers = agent._client_kwargs["default_headers"]
    assert headers["HTTP-Referer"] == "https://hermes-agent.nousresearch.com"
    assert "X-OpenRouter-Cache" not in headers
    assert "X-OpenRouter-Cache-TTL" not in headers


@patch("run_agent.OpenAI")
def test_copilot_enterprise_base_url_applies_copilot_default_headers(mock_openai):
    """Enterprise Copilot endpoints (api.<tenant>.githubcopilot.com) must apply
    the same default_headers — including Copilot-Integration-Id: vscode-chat —
    as the default api.githubcopilot.com endpoint. Without this, the upstream
    sees the request as integrator 'zed' or 'copilot-language-server' and
    rejects it with a 400 error for many models (regression seen May 2026)."""
    mock_openai.return_value = MagicMock()
    agent = AIAgent(
        api_key="test-key",
        base_url="https://api.enterprise.githubcopilot.com",
        model="claude-opus-4.6-1m",
        provider="copilot",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )

    agent._apply_client_headers_for_base_url("https://api.enterprise.githubcopilot.com")

    headers = agent._client_kwargs.get("default_headers", {})
    # Lookup is case-insensitive — normalize for the assertion.
    lc = {k.lower(): v for k, v in headers.items()}
    assert lc.get("copilot-integration-id") == "vscode-chat", (
        f"enterprise Copilot endpoint must carry Copilot-Integration-Id=vscode-chat; got {headers}"
    )


@patch("run_agent.OpenAI")
def test_copilot_codex_responses_path_carries_integration_id(mock_openai):
    """Copilot requests via codex_responses api_mode must carry
    Copilot-Integration-Id: vscode-chat on every outbound request.

    Without this header, GitHub attributes the request to integrator
    'copilot-language-server' (whose model whitelist is much smaller than
    'vscode-chat') and rejects claude-opus-4.8, gpt-5.x-full, etc. with
    HTTP 400. The header must be present in _client_kwargs so the per-request
    client built by _create_request_openai_client inherits it for both
    chat_completions and codex_responses paths.
    """
    mock_openai.return_value = MagicMock()
    agent = AIAgent(
        api_key="test-key",
        base_url="https://api.githubcopilot.com",
        model="gpt-5.4",
        provider="copilot",
        api_mode="codex_responses",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )

    headers = agent._client_kwargs.get("default_headers", {})
    lc = {k.lower(): v for k, v in headers.items()}
    assert lc.get("copilot-integration-id") == "vscode-chat", (
        f"codex_responses path must carry Copilot-Integration-Id=vscode-chat; got {headers}"
    )


def test_copilot_default_headers_fallback_includes_integration_id():
    """The copilot_default_headers() ImportError fallback must include
    Copilot-Integration-Id: vscode-chat.

    When hermes_cli.copilot_auth cannot be imported (partial install,
    module load order), the fallback dict is used by agent_init.py and
    run_agent.py to set default_headers. Omitting Copilot-Integration-Id
    here causes the same HTTP 400 integrator attribution bug as missing
    it in the primary path.
    """
    import builtins
    real_import = builtins.__import__

    def _block_copilot_auth(name, *args, **kwargs):
        if name == "hermes_cli.copilot_auth":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    from hermes_cli.models import copilot_default_headers

    with patch.object(builtins, "__import__", side_effect=_block_copilot_auth):
        headers = copilot_default_headers()

    lc = {k.lower(): v for k, v in headers.items()}
    assert lc.get("copilot-integration-id") == "vscode-chat", (
        f"fallback copilot_default_headers must include Copilot-Integration-Id; got {headers}"
    )

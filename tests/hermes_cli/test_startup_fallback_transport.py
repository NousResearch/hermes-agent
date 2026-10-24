"""Startup fallback must preserve its own wire on CLI, messaging/oneshot, cron and TUI/Desktop."""

import pytest

from hermes_cli import runtime_provider
from hermes_cli.auth import AuthError
from hermes_cli.config import atomic_config_write


@pytest.mark.parametrize("surface", ["shared", "cli", "cron", "tui"])
@pytest.mark.parametrize("field, value, expected", [
    ("api_mode", "codex_responses", "codex_responses"),
    ("api_mode", "anthropic_messages", "anthropic_messages"),
    ("api_mode", "chat_completions", "chat_completions"),
    ("transport", "responses", "codex_responses"),
    ("api_mode", "invalid-mode", "chat_completions"),
    ("api_mode", None, "chat_completions"),
])
def test_startup_fallback_transport(tmp_path, monkeypatch, surface, field, value, expected):
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "fixture-azure-key")
    entry = {"provider": "azure-foundry", "model": "custom-deployment",
             "base_url": "https://fixture.openai.azure.com/openai/v1", field: value}
    config = {"model": {"provider": "openai-codex", "default": "gpt-6-astra"},
              "fallback_providers": [entry]}
    runtime = _resolve_startup(config, surface, tmp_path, monkeypatch)
    assert runtime["api_mode"] == expected
    assert runtime["provider"] == "azure-foundry"
    expected_url = entry["base_url"].removesuffix("/v1") if expected == "anthropic_messages" else entry["base_url"]
    assert runtime["base_url"] == expected_url
    assert runtime["api_key"] == "fixture-azure-key"


def _resolve_startup(config, surface, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_write(tmp_path / "config.yaml", config)
    entry = config["fallback_providers"][0]
    resolve = runtime_provider.resolve_runtime_provider

    def primary_unavailable(**kwargs):
        if kwargs.get("target_model") != entry["model"]:
            raise AuthError("fixture primary unavailable")
        return resolve(**kwargs)

    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", primary_unavailable)
    if surface == "shared":
        runtime, selected = runtime_provider.resolve_runtime_with_fallback(config)
        assert selected["model"] == entry["model"]
    elif surface == "cli":
        from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin
        monkeypatch.setattr("cli._cprint", lambda *a, **k: None)
        shell = CLIAgentSetupMixin.__new__(CLIAgentSetupMixin)
        shell._fallback_model = [entry]
        runtime = shell._resolve_fallback_runtime(AuthError("fixture primary unavailable"))
        assert shell.model == entry["model"]
    elif surface == "cron":
        from cron.scheduler import _CronJobConfig, _resolve_job_runtime
        jc = _CronJobConfig(cfg=config, model=config["model"]["default"], model_cfg=config["model"], cron_default_provider="")
        runtime, selected = _resolve_job_runtime({}, "fixture-job", jc)
        assert selected == entry["model"]
    else:
        from tui_gateway import server
        monkeypatch.setattr(server, "_hermes_home", tmp_path)
        resolution = server._resolve_runtime_with_fallback()
        runtime = resolution.runtime
        assert resolution.selected_model == entry["model"]
    return runtime


@pytest.mark.parametrize("surface", ["shared", "cli", "cron", "tui"])
@pytest.mark.parametrize("auth_mode", ["api_key", "entra_id"])
def test_azure_entry_mode_precedes_endpoint_normalization(tmp_path, monkeypatch, surface, auth_mode):
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "fixture-azure-key")
    token_provider = lambda: "fixture-token"
    monkeypatch.setattr("hermes_cli.runtime_provider_backends._azure_entra_credentials", lambda cfg: token_provider)
    entry = {"provider": "azure-foundry", "model": "custom-deployment",
             "base_url": "https://fixture.openai.azure.com/openai/v1", "api_mode": "codex_responses"}
    config = {"model": {"provider": "azure-foundry", "default": "claude-primary",
                        "base_url": "https://fixture.services.ai.azure.com/anthropic",
                        "api_mode": "anthropic_messages", "auth_mode": auth_mode}, "fallback_providers": [entry]}
    runtime = _resolve_startup(config, surface, tmp_path, monkeypatch)
    assert (runtime["api_mode"], runtime["base_url"]) == ("codex_responses", entry["base_url"])
    assert runtime["auth_mode"] == auth_mode
    assert runtime["api_key"] == (token_provider if auth_mode == "entra_id" else "fixture-azure-key")


@pytest.mark.parametrize("surface", ["shared", "cli", "cron", "tui"])
@pytest.mark.parametrize("mode, expected", [
    (None, "anthropic_messages"), ("responses", "codex_responses"), ("chat_completions", "chat_completions"),
])
def test_named_provider_transport_inheritance_and_override(tmp_path, monkeypatch, surface, mode, expected):
    monkeypatch.setenv("FIXTURE_RELAY_KEY", "fixture-relay-key")
    entry = {"provider": "fixture-relay", "model": "fallback-model"}
    if mode:
        entry["transport"] = mode
    base_url = "https://relay.example.test/v1"
    config = {"model": {"provider": "openai-codex", "default": "primary-model"},
              "providers": {"fixture-relay": {"base_url": base_url, "api_mode": "anthropic_messages",
                                              "key_env": "FIXTURE_RELAY_KEY"}},
              "fallback_providers": [entry]}
    runtime = _resolve_startup(config, surface, tmp_path, monkeypatch)
    assert (runtime["api_mode"], runtime["base_url"], runtime["api_key"]) == (expected, base_url, "fixture-relay-key")


@pytest.mark.parametrize("source", ["explicit", "pool", "env"])
@pytest.mark.parametrize("mode", ["codex_responses", "chat_completions", "anthropic_messages"])
def test_registry_entry_mode_precedes_endpoint_normalization(tmp_path, monkeypatch, source, mode):
    from agent.credential_pool import CredentialPool, PooledCredential
    base_url = "https://relay.example.test/v1"
    key = f"fixture-{source}-key"
    entry = {"provider": "opencode-go", "model": "qwen-fixture", "api_mode": mode}
    if source == "explicit":
        entry.update(api_key=key, base_url=base_url)
    monkeypatch.setenv("OPENCODE_GO_API_KEY", key)
    credentials = [PooledCredential(
        provider="opencode-go", id="fixture", label="fixture", auth_type="api_key", priority=0,
        source="fixture-pool", access_token=key, base_url=base_url)] if source == "pool" else []
    pool = CredentialPool("opencode-go", credentials)
    monkeypatch.setattr(runtime_provider, "load_pool", lambda provider: pool)
    config = {"model": {"provider": "opencode-go", "default": "primary-model",
                        "base_url": base_url, "api_mode": "anthropic_messages"},
              "fallback_providers": [entry]}
    runtime = _resolve_startup(config, "shared", tmp_path, monkeypatch)
    expected_url = base_url.removesuffix("/v1") if mode == "anthropic_messages" else base_url
    assert (runtime["api_mode"], runtime["base_url"], runtime["api_key"]) == (mode, expected_url, key)
    if source == "pool":
        assert runtime["credential_pool"] is pool


@pytest.mark.parametrize("surface", ["shared", "cli", "cron", "tui"])
@pytest.mark.parametrize("source", ["config", "pool"])
@pytest.mark.parametrize("provider", ["opencode-go-bridge", "opencode-zen-bridge"])
@pytest.mark.parametrize("model, mode, path", [
    ("grok-4.5", "codex_responses", "responses"),
    ("deepseek-v4-flash", "chat_completions", "chat/completions"),
])
def test_named_opencode_entry_keeps_normalized_request_path(
    tmp_path, monkeypatch, surface, source, provider, model, mode, path,
):
    import httpx
    from openai import OpenAI
    from agent.credential_pool import CredentialPool, PooledCredential

    key = f"fixture-{source}-key"
    credentials = [PooledCredential(
        provider=provider, id="fixture", label="fixture", auth_type="api_key", priority=0,
        source="fixture-pool", access_token=key)] if source == "pool" else []
    pool = CredentialPool(provider, credentials)
    monkeypatch.setattr(runtime_provider, "load_pool", lambda provider: pool)
    config = {
        "model": {"provider": "openai-codex", "default": "primary-model"},
        "providers": {provider: {"base_url": "https://opencode.ai/zen/go", "api_key": "fixture-config-key"}},
        "fallback_providers": [{"provider": provider, "model": model, "api_mode": mode}],
    }
    runtime = _resolve_startup(config, surface, tmp_path, monkeypatch)
    assert (runtime["api_mode"], runtime["api_key"]) == (mode, key)
    if source == "pool":
        assert runtime["credential_pool"] is pool
    requests = []

    def reply(request):
        requests.append(str(request.url))
        return httpx.Response(200, json={})

    with OpenAI(
        api_key=runtime["api_key"], base_url=runtime["base_url"],
        http_client=httpx.Client(transport=httpx.MockTransport(reply)),
    ) as client:
        if mode == "codex_responses":
            client.responses.create(model=model, input="offline fixture")
        else:
            client.chat.completions.create(model=model, messages=[{"role": "user", "content": "offline fixture"}])
    assert requests == [f"https://opencode.ai/zen/go/v1/{path}"]


@pytest.mark.parametrize("surface", ["shared", "cli", "cron", "tui"])
@pytest.mark.parametrize("explicit_mode", [False, True])
@pytest.mark.parametrize("base_url", [
    "https://relay.example.test/v1", "https://opencode.ai/zen/go/v1",
])
def test_named_anthropic_fallback_keeps_auth_headers(
    tmp_path, monkeypatch, surface, explicit_mode, base_url,
):
    import anthropic
    import httpx
    from agent.anthropic_adapter import build_anthropic_client
    from agent.credential_pool import CredentialPool

    provider, model = "opencode-go-bridge", "minimax-m2.7"
    entry = {"provider": provider, "model": model}
    if explicit_mode:
        entry["api_mode"] = "anthropic_messages"
    config = {
        "model": {"provider": "openai-codex", "default": "primary-model"},
        "providers": {provider: {
            "base_url": base_url, "api_mode": "anthropic_messages", "api_key": "fixture-relay-key",
            "extra_headers": {"X-Relay-Access": "fixture-relay-access"},
        }},
        "fallback_providers": [entry],
    }
    monkeypatch.setattr(runtime_provider, "load_pool", lambda name: CredentialPool(name, []))
    runtime = _resolve_startup(config, surface, tmp_path, monkeypatch)
    assert runtime["api_mode"] == "anthropic_messages"
    requests = []

    def reply(request):
        requests.append(request)
        return httpx.Response(200, json={
            "id": "fixture", "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "text", "text": "offline"}], "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })

    sdk = anthropic.Anthropic

    def offline_sdk(**kwargs):
        return sdk(http_client=httpx.Client(transport=httpx.MockTransport(reply)), **kwargs)

    monkeypatch.setattr(anthropic, "Anthropic", offline_sdk)
    with build_anthropic_client(runtime["api_key"], runtime["base_url"]) as client:
        client.messages.create(model=model, max_tokens=1, messages=[{"role": "user", "content": "offline"}])
    assert len(requests) == 1
    assert str(requests[0].url) == f"{base_url}/messages"
    assert requests[0].headers.get("X-Relay-Access") == "fixture-relay-access"


@pytest.mark.parametrize("surface", ["shared", "cli", "cron", "tui"])
@pytest.mark.parametrize("explicit_mode", [False, True])
@pytest.mark.parametrize("pool_style", ["durable", "legacy"])
def test_named_anthropic_fallback_pool_survives_agent_init(
    tmp_path, monkeypatch, surface, explicit_mode, pool_style,
):
    from types import SimpleNamespace
    from agent.agent_init import _finalize_routing
    from agent.credential_pool import CredentialPool, PooledCredential

    provider, model = "opencode-go-bridge", "minimax-m2.7"
    pool_key = provider if pool_style == "durable" else f"custom:{provider}"
    pool = CredentialPool(pool_key, [PooledCredential(
        provider=pool_key, id="fixture", label="fixture", auth_type="api_key", priority=0,
        source="fixture", access_token="fixture-pool-key")])
    empty = CredentialPool("empty", [])
    monkeypatch.setattr(runtime_provider, "load_pool", lambda name: pool if name == pool_key else empty)
    entry = {"provider": provider, "model": model}
    if explicit_mode:
        entry["api_mode"] = "anthropic_messages"
    config = {
        "model": {"provider": "openai-codex", "default": "primary-model"},
        "providers": {provider: {
            "base_url": "https://opencode.ai/zen/go/v1", "api_mode": "anthropic_messages",
        }},
        "fallback_providers": [entry],
    }
    runtime = _resolve_startup(config, surface, tmp_path, monkeypatch)
    assert runtime["api_mode"] == "anthropic_messages"
    assert runtime["credential_pool"] is pool
    agent = SimpleNamespace(
        provider=runtime["provider"], base_url=runtime["base_url"], model=model, api_mode=runtime["api_mode"],
        _credential_pool=pool, _get_transport=lambda: None, _is_openrouter_url=lambda: False)
    _finalize_routing(agent, runtime["api_mode"], pool)
    assert agent._credential_pool is pool


@pytest.mark.parametrize("surface", ["shared", "cli", "cron", "tui"])
@pytest.mark.parametrize("mode, model, path", [
    ("chat_completions", "deepseek-v4-flash", "chat/completions"),
    ("codex_responses", "grok-4.5", "responses"),
])
@pytest.mark.parametrize("pool_style", ["durable", "legacy"])
@pytest.mark.parametrize("explicit_mode", [False, True])
@pytest.mark.parametrize("suffix", ["", "/", "/v1"])
def test_named_openai_fallback_keeps_auth_through_rotation(
    tmp_path, monkeypatch, surface, mode, model, path, pool_style, explicit_mode, suffix,
):
    import httpx
    from openai import OpenAI
    from agent.agent_init import _apply_openai_header_policy, _finalize_routing
    from agent.credential_pool import PooledCredential, credential_pool_entry_serves_endpoint
    from hermes_cli.auth import write_credential_pool
    from run_agent import AIAgent

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider, base_url = "opencode-go-bridge", f"https://opencode.ai/zen/go{suffix}"
    pool_key = provider if pool_style == "durable" else f"custom:{provider}"
    rows = [PooledCredential(
        provider=pool_key, id=f"fixture-{i}", label="fixture", auth_type="api_key", priority=i,
        source="manual:fixture", access_token=f"fixture-pool-key-{i}", base_url=base_url,
    ) for i in range(2)]
    write_credential_pool(pool_key, [row.to_dict() for row in rows])
    entry = {"provider": provider, "model": model}
    if explicit_mode:
        entry["api_mode"] = mode
    config = {
        "model": {"provider": "openai-codex", "default": "primary-model"},
        "providers": {provider: {
            "base_url": base_url, "api_mode": mode,
            "extra_headers": {"X-Relay-Access": "fixture-access"}, "ssl_verify": False,
        }},
        "fallback_providers": [entry],
    }
    runtime = _resolve_startup(config, surface, tmp_path, monkeypatch)
    pool = runtime["credential_pool"]
    assert pool.provider == pool_key
    assert runtime["api_mode"] == mode
    agent = AIAgent.__new__(AIAgent)
    agent.provider, agent.model, agent.api_mode = runtime["provider"], model, mode
    agent.api_key, agent.base_url = runtime["api_key"], runtime["base_url"]
    agent._credential_pool, agent._transport_cache = pool, {}
    monkeypatch.setattr(agent, "_get_transport", lambda: None)
    agent._client_kwargs = {"api_key": agent.api_key, "base_url": agent.base_url}
    _apply_openai_header_policy(agent, agent._client_kwargs)
    _finalize_routing(agent, mode, pool)
    requests = []

    def reply(request):
        requests.append(request)
        return httpx.Response(200, json={})

    def request_with_current_credentials(**kwargs):
        client_kwargs = {k: v for k, v in agent._client_kwargs.items() if k != "ssl_verify"}
        assert agent._client_kwargs.get("ssl_verify") is False
        client_kwargs["http_client"] = httpx.Client(transport=httpx.MockTransport(reply))
        with OpenAI(**client_kwargs) as client:
            if mode == "codex_responses":
                client.responses.create(model=model, input="offline")
            else:
                client.chat.completions.create(model=model, messages=[{"role": "user", "content": "offline"}])

    request_with_current_credentials()
    assert (
        str(requests[-1].url), requests[-1].headers.get("X-Relay-Access"), agent._credential_pool,
    ) == (f"https://opencode.ai/zen/go/v1/{path}", "fixture-access", pool)
    assert credential_pool_entry_serves_endpoint(rows[1], agent.base_url)
    monkeypatch.setattr(agent, "_replace_primary_openai_client", request_with_current_credentials)
    assert agent._swap_credential(rows[1])
    assert (
        str(requests[-1].url), requests[-1].headers.get("X-Relay-Access"),
        requests[-1].headers.get("Authorization"), agent.base_url, agent._credential_pool,
    ) == (
        f"https://opencode.ai/zen/go/v1/{path}", "fixture-access",
        "Bearer fixture-pool-key-1", "https://opencode.ai/zen/go/v1", pool,
    )

from hermes_cli.models_catalog_static import CANONICAL_PROVIDERS


def test_claude_subscription_and_direct_api_are_distinct_model_choices():
    choices = {entry.slug: entry for entry in CANONICAL_PROVIDERS}

    assert choices["claude-subscription"].label == "Claude Subscription"
    assert choices["claude-api"].label == "Anthropic API"

    from hermes_cli.models_catalog_static import PROVIDER_GROUPS, _PROVIDER_MODELS
    from hermes_cli.auth import resolve_provider
    assert PROVIDER_GROUPS["claude"][2] == ["claude-subscription", "claude-api"]
    assert _PROVIDER_MODELS["claude-subscription"] == _PROVIDER_MODELS["anthropic"]
    assert _PROVIDER_MODELS["claude-api"] == _PROVIDER_MODELS["anthropic"]
    assert resolve_provider("claude-subscription") == "claude-subscription"
    assert resolve_provider("claude-api") == "claude-api"


def test_picker_lists_source_specific_choices_when_each_credential_exists(monkeypatch):
    from hermes_cli import inventory
    import hermes_cli.model_switch_providers as switch_providers
    from hermes_cli.model_switch import list_authenticated_providers

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-test-only")
    monkeypatch.setattr(inventory, "_anthropic_oauth_credentials_present", lambda: True)
    monkeypatch.setattr(
        switch_providers, "_live_or_curated_ids",
        lambda _provider, *_args, **_kwargs: ["claude-sonnet-4-6"])

    rows = list_authenticated_providers(current_provider="openrouter", max_models=50)
    by_slug = {row["slug"]: row for row in rows}

    assert by_slug["claude-api"]["name"] == "Anthropic API"
    assert by_slug["claude-subscription"]["name"] == "Claude Subscription"
    assert by_slug["claude-api"]["models"] == ["claude-sonnet-4-6"]
    assert by_slug["claude-subscription"]["models"] == ["claude-sonnet-4-6"]


def test_picker_availability_is_independent_per_credential_source(monkeypatch):
    from hermes_cli import inventory
    import hermes_cli.model_switch_providers as switch_providers
    from hermes_cli.model_switch import list_authenticated_providers

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-test-only")
    monkeypatch.setattr(inventory, "_anthropic_oauth_credentials_present", lambda: False)
    monkeypatch.setattr(
        switch_providers, "_live_or_curated_ids",
        lambda _provider, *_args, **_kwargs: ["claude-sonnet-4-6"])

    api_rows = list_authenticated_providers(current_provider="openrouter", max_models=50)
    assert "claude-api" in {row["slug"] for row in api_rows}
    assert "claude-subscription" not in {row["slug"] for row in api_rows}

    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.setattr(inventory, "_anthropic_oauth_credentials_present", lambda: True)
    subscription_rows = list_authenticated_providers(current_provider="openrouter", max_models=50)
    assert "claude-subscription" in {row["slug"] for row in subscription_rows}
    assert "claude-api" not in {row["slug"] for row in subscription_rows}


def test_direct_api_choice_uses_native_messages_api_and_explicit_key():
    from hermes_cli.runtime_provider import resolve_runtime_provider

    runtime = resolve_runtime_provider(
        requested="claude-api",
        explicit_api_key="sk-ant-api-test-only",
    )

    assert runtime["provider"] == "anthropic"
    assert runtime["api_mode"] == "anthropic_messages"
    assert runtime["base_url"] == "https://api.anthropic.com"
    assert runtime["api_key"] == "sk-ant-api-test-only"
    assert runtime["requested_provider"] == "claude-api"
    assert runtime["source"] == "anthropic_api_key"


def test_direct_api_choice_reads_only_the_api_key_secret(monkeypatch):
    from hermes_cli import runtime_provider

    monkeypatch.setattr(
        runtime_provider, "get_secret_str",
        lambda name, *_args, **_kwargs: "sk-ant-api-test-only" if name == "ANTHROPIC_API_KEY" else "",
    )
    runtime = runtime_provider.resolve_runtime_provider(requested="claude-api")

    assert runtime["api_key"] == "sk-ant-api-test-only"
    assert runtime["source"] == "anthropic_api_key"


def test_direct_api_choice_never_falls_back_to_subscription_oauth(monkeypatch):
    import pytest
    import agent.anthropic_credentials as credentials
    from hermes_cli import runtime_provider
    from hermes_cli.auth import AuthError

    monkeypatch.setattr(runtime_provider, "get_secret_str", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(credentials, "resolve_anthropic_token", lambda **_kwargs: "sk-ant-oat01-subscription-test")

    with pytest.raises(AuthError, match="requires ANTHROPIC_API_KEY"):
        runtime_provider.resolve_runtime_provider(requested="claude-api")


def test_subscription_resolver_skips_api_keys_in_ambiguous_token_env(monkeypatch):
    import agent.anthropic_credentials as credentials

    monkeypatch.setattr(
        credentials, "_first_env",
        lambda name: "sk-ant-api-test-only" if name == "ANTHROPIC_TOKEN" else "",
    )
    monkeypatch.setattr(credentials, "_resolve_anthropic_pool_token", lambda **_kwargs: "sk-ant-oat01-test")
    monkeypatch.setattr(credentials, "_available_anthropic_token", lambda token, _model: token)

    assert credentials.resolve_anthropic_oauth_token() == "sk-ant-oat01-test"


def test_subscription_choice_uses_oauth_not_api_key(monkeypatch):
    import agent.anthropic_credentials as credentials
    from hermes_cli import runtime_provider

    monkeypatch.setattr(runtime_provider, "get_secret_str", lambda *_args, **_kwargs: "sk-ant-api-test-only")
    monkeypatch.setattr(credentials, "resolve_anthropic_oauth_token", lambda **_kwargs: "sk-ant-oat01-subscription-test", raising=False)

    runtime = runtime_provider.resolve_runtime_provider(requested="claude-subscription")

    assert runtime["provider"] == "anthropic"
    assert runtime["api_mode"] == "anthropic_messages"
    assert runtime["api_key"] == "sk-ant-oat01-subscription-test"
    assert runtime["requested_provider"] == "claude-subscription"
    assert runtime["source"] == "anthropic_subscription_oauth"


def test_subscription_choice_rejects_api_key_as_explicit_credential():
    import pytest
    from hermes_cli.auth import AuthError
    from hermes_cli.runtime_provider import resolve_runtime_provider

    with pytest.raises(AuthError, match="requires an OAuth credential"):
        resolve_runtime_provider(requested="claude-subscription", explicit_api_key="sk-ant-api-test-only")


def test_source_choices_reject_non_anthropic_endpoint():
    import pytest
    from hermes_cli.auth import AuthError
    from hermes_cli.runtime_provider import resolve_runtime_provider

    with pytest.raises(AuthError, match="native Messages API endpoint"):
        resolve_runtime_provider(
            requested="claude-api", explicit_api_key="sk-ant-api-test-only",
            explicit_base_url="https://other-cloud.example/v1")

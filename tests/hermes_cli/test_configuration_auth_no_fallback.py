"""Configuration AuthError codes must not walk fallback_model / fallback_providers.

Resolve-time missing-credential errors (``missing_api_key``,
``no_provider_configured``, ``invalid_provider``) are operator misconfig, not
transient auth. Falling through to another paid provider silently spends the
wrong account. Rate-limit / uncoded / non-AuthError paths stay fail-open.
"""

from unittest.mock import patch

import pytest

from hermes_cli.auth import (
    AUTH_ERROR_CATEGORY_MISSING_CREDENTIAL,
    AuthError,
    CODEX_RATE_LIMITED_CODE,
)
from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin


FALLBACK_RUNTIME = {
    "provider": "gemini",
    "api_key": "fallback-key",
    "base_url": "https://generativelanguage.googleapis.com/v1beta",
}


def _fallback_resolve(**kwargs):
    """Succeed for any resolve so a leaked fallback walk is observable."""
    return dict(FALLBACK_RUNTIME, requested=kwargs.get("requested"))


@pytest.mark.parametrize(
    "error,expected",
    [
        (
            AuthError(
                "No Codex credentials",
                code="codex_auth_missing",
                category=AUTH_ERROR_CATEGORY_MISSING_CREDENTIAL,
            ),
            False,
        ),
        # Compatibility for producers not yet migrated to the semantic category.
        (AuthError("No usable credentials", code="missing_api_key"), False),
        (AuthError("No inference provider configured.", code="no_provider_configured"), False),
        (AuthError("Unknown provider 'nope'.", code="invalid_provider"), False),
        (AuthError("quota", code=CODEX_RATE_LIMITED_CODE), True),
        (AuthError("runtime 401", code="invalid_token", relogin_required=True), True),
        (AuthError("runtime 403", code="forbidden"), True),
        (AuthError("x"), True),
        (AuthError("x", code=None), True),
        (ValueError("nope"), True),
    ],
)
def test_should_try_fallback_on_auth_error_table(error, expected):
    from hermes_cli.auth import should_try_fallback_on_auth_error

    assert should_try_fallback_on_auth_error(error) is expected


class _FallbackCLI(CLIAgentSetupMixin):
    def __init__(self):
        self._fallback_model = [{"provider": "gemini", "model": "gemini-2.5-flash"}]
        self.requested_provider = "openai"
        self.model = "gpt-4"


def test_cli_missing_api_key_does_not_resolve_fallback():
    """CLI must not consult the fallback chain on missing_api_key."""
    cli = _FallbackCLI()
    requested = []

    def fake_resolve(**kwargs):
        requested.append(kwargs.get("requested"))
        return _fallback_resolve(**kwargs)

    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=fake_resolve), \
         patch("cli._cprint", lambda *a, **k: None):
        result = cli._resolve_fallback_runtime(
            AuthError("No usable credentials", code="missing_api_key")
        )

    assert result is None
    assert requested == []
    assert cli.requested_provider == "openai"
    assert cli.model == "gpt-4"


def test_cli_uncoded_auth_error_still_tries_fallback():
    """Fail-open CONTROL: AuthError without a code still walks fallback."""
    cli = _FallbackCLI()
    requested = []

    def fake_resolve(**kwargs):
        requested.append(kwargs.get("requested"))
        return _fallback_resolve(**kwargs)

    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=fake_resolve), \
         patch("cli._cprint", lambda *a, **k: None):
        result = cli._resolve_fallback_runtime(
            AuthError("Codex token refresh failed with status 401")
        )

    assert result is not None
    assert result["provider"] == "gemini"
    assert requested == ["gemini"]
    assert cli.requested_provider == "gemini"
    assert cli.model == "gemini-2.5-flash"


def test_tui_empty_codex_oauth_resolver_does_not_walk_fallback(monkeypatch):
    """The real empty-store Codex resolver emits semantic missing-credential and stops fallback."""
    from hermes_cli import auth_codex, runtime_provider
    from tui_gateway import server

    requested = []
    fallback_loads = []

    monkeypatch.setattr(runtime_provider, "_get_model_config", lambda: {"provider": "openai-codex"})
    monkeypatch.setattr(runtime_provider, "load_pool", lambda _provider: None)
    monkeypatch.setattr(auth_codex, "_load_auth_store_maybe_locked", lambda _lock=True: {})
    monkeypatch.setattr(auth_codex, "_pool_codex_access_token", lambda: "")
    monkeypatch.setattr(auth_codex, "_codex_pool_rate_limit_status", lambda: None)

    def fake_resolve_provider(requested_provider, **_kwargs):
        requested.append(requested_provider)
        if requested_provider == "openai-codex":
            return "openai-codex"
        pytest.fail(f"fallback resolver called for {requested_provider}")

    monkeypatch.setattr(runtime_provider, "resolve_provider", fake_resolve_provider)

    def fallback_model():
        fallback_loads.append(True)
        return [{"provider": "gemini", "model": "gemini-2.5-flash"}]

    monkeypatch.setattr(server, "_load_fallback_model", fallback_model)

    with pytest.raises(AuthError) as exc_info:
        server._resolve_runtime_with_fallback({"requested": "openai-codex"})

    assert exc_info.value.category == AUTH_ERROR_CATEGORY_MISSING_CREDENTIAL
    assert exc_info.value.code == "codex_auth_missing"
    assert requested == ["openai-codex"]
    assert fallback_loads == []


def test_tui_expired_minimax_without_refresh_does_not_walk_fallback(monkeypatch):
    """The real MiniMax resolver must not spend a fallback when re-login is required."""
    from hermes_cli import auth as auth_mod
    from hermes_cli import runtime_provider
    from tui_gateway import server

    requested = []
    fallback_loads = []
    minimax_state = {
        "access_token": "expired-token",
        "expires_at": "2000-01-01T00:00:00+00:00",
        "inference_base_url": "https://api.minimax.io/anthropic",
    }

    monkeypatch.setattr(runtime_provider, "_get_model_config", lambda: {"provider": "minimax-oauth"})
    monkeypatch.setattr(runtime_provider, "load_pool", lambda _provider: None)
    monkeypatch.setattr(auth_mod, "get_provider_auth_state", lambda _provider: minimax_state)

    def fake_resolve_provider(requested_provider, **_kwargs):
        requested.append(requested_provider)
        if requested_provider == "minimax-oauth":
            return "minimax-oauth"
        pytest.fail(f"fallback resolver called for {requested_provider}")

    monkeypatch.setattr(runtime_provider, "resolve_provider", fake_resolve_provider)

    def fallback_model():
        fallback_loads.append(True)
        return [{"provider": "gemini", "model": "gemini-2.5-flash"}]

    monkeypatch.setattr(server, "_load_fallback_model", fallback_model)

    with pytest.raises(AuthError) as exc_info:
        server._resolve_runtime_with_fallback({"requested": "minimax-oauth"})

    assert exc_info.value.code == "no_refresh_token"
    assert exc_info.value.category == AUTH_ERROR_CATEGORY_MISSING_CREDENTIAL
    assert requested == ["minimax-oauth"]
    assert fallback_loads == []


def test_minimax_token_provider_missing_access_token_is_configuration_error(monkeypatch):
    """A token disappearing after initial resolution also requires re-login, not fallback."""
    from hermes_cli import auth_minimax

    states = iter((
        {
            "access_token": "initial-token",
            "inference_base_url": "https://api.minimax.io/anthropic",
        },
        {},
    ))
    monkeypatch.setattr(auth_minimax, "_minimax_fresh_state", lambda: next(states))

    runtime = auth_minimax.resolve_minimax_oauth_runtime_credentials(as_token_provider=True)
    with pytest.raises(AuthError) as exc_info:
        runtime["api_key"]()

    assert exc_info.value.code == "no_access_token"
    assert exc_info.value.category == AUTH_ERROR_CATEGORY_MISSING_CREDENTIAL


def test_codex_pool_cooldown_is_not_missing_credential(monkeypatch):
    """Existing but unavailable pool material must retain legal fallback semantics."""
    from hermes_cli import auth_codex
    from hermes_cli.auth import should_try_fallback_on_auth_error

    monkeypatch.setattr(auth_codex, "_load_auth_store_maybe_locked", lambda _lock=True: {})
    monkeypatch.setattr(auth_codex, "_pool_codex_access_token", lambda: "")
    monkeypatch.setattr(auth_codex, "_codex_pool_rate_limit_status", lambda: None)
    monkeypatch.setattr(auth_codex, "_read_codex_pool_entries", lambda: [{"access_token": "cooled-token"}])

    with pytest.raises(AuthError) as exc_info:
        auth_codex.resolve_codex_runtime_credentials()

    assert exc_info.value.category is None
    assert should_try_fallback_on_auth_error(exc_info.value) is True

    monkeypatch.setattr(auth_codex, "_read_codex_pool_entries", lambda: [{"label": "empty-shell"}])
    with pytest.raises(AuthError) as empty_exc:
        auth_codex.resolve_codex_runtime_credentials()

    assert empty_exc.value.category == AUTH_ERROR_CATEGORY_MISSING_CREDENTIAL
    assert should_try_fallback_on_auth_error(empty_exc.value) is False


def test_auto_codex_empty_store_does_not_spend_openrouter_fallback(monkeypatch):
    """Never-configured Codex must fail closed before a paid auto fallback."""
    from hermes_cli import auth_codex, runtime_provider

    monkeypatch.setattr(runtime_provider, "_get_model_config", lambda: {})
    monkeypatch.setattr(runtime_provider, "_resolve_requested_shortcuts", lambda *_args: None)
    monkeypatch.setattr(runtime_provider, "_resolve_named_custom_runtime", lambda **_kwargs: None)
    monkeypatch.setattr(runtime_provider, "_local_endpoint_bypass", lambda *_args: None)
    monkeypatch.setattr(runtime_provider, "resolve_provider", lambda *_args, **_kwargs: "openai-codex")
    monkeypatch.setattr(runtime_provider, "load_pool", lambda _provider: None)
    monkeypatch.setattr(auth_codex, "_load_auth_store_maybe_locked", lambda _lock=True: {})
    monkeypatch.setattr(auth_codex, "_pool_codex_access_token", lambda: "")
    monkeypatch.setattr(auth_codex, "_codex_pool_rate_limit_status", lambda: None)
    monkeypatch.setattr(
        runtime_provider,
        "_openrouter_fallback",
        lambda *_args: {"provider": "openrouter", "api_key": "fallback-key"},
    )

    with pytest.raises(AuthError) as exc_info:
        runtime_provider.resolve_runtime_provider(requested="auto")
    assert exc_info.value.category == AUTH_ERROR_CATEGORY_MISSING_CREDENTIAL


def test_quarantined_oauth_states_keep_fallback_semantics(monkeypatch):
    """Terminal quarantine proves credentials existed, so empty state is not initial absence."""
    from hermes_cli import auth as auth_mod
    from hermes_cli import auth_minimax, auth_xai
    from hermes_cli.auth import should_try_fallback_on_auth_error
    from hermes_cli.auth_nous import _NousRuntimeResolve

    marker = {"code": "invalid_grant", "relogin_required": True}
    xai_store = {
        "providers": {
            "xai-oauth": {
                "tokens": {"access_token": "revoked", "refresh_token": "burned"},
            },
        },
    }
    monkeypatch.setattr(auth_mod, "_load_auth_store", lambda: xai_store)
    monkeypatch.setattr(auth_mod, "_save_auth_store", lambda _store: None)
    auth_xai._quarantine_xai_oauth_tokens(
        AuthError(
            "revoked",
            provider="xai-oauth",
            code="xai_refresh_failed",
            relogin_required=True,
        )
    )
    monkeypatch.setattr(
        auth_xai,
        "_load_auth_store_maybe_locked",
        lambda _lock=True: xai_store,
    )
    with pytest.raises(AuthError) as xai_exc:
        auth_xai._read_xai_oauth_tokens()
    assert getattr(xai_exc.value, "category", None) is None
    assert should_try_fallback_on_auth_error(xai_exc.value) is True

    minimax_state = {"last_auth_error": marker}
    monkeypatch.setattr(auth_mod, "get_provider_auth_state", lambda _provider: minimax_state)
    with pytest.raises(AuthError) as minimax_exc:
        auth_minimax._minimax_fresh_state()
    assert getattr(minimax_exc.value, "category", None) is None
    assert should_try_fallback_on_auth_error(minimax_exc.value) is True

    nous_state = {"last_auth_error": marker}
    nous_run = _NousRuntimeResolve(
        {"providers": {"nous": nous_state}},
        nous_state,
        None,
        force_refresh=False,
        stale_access_token=None,
        timeout_seconds=1,
    )
    monkeypatch.setattr(nous_run, "merge_shared", lambda: False)
    with pytest.raises(AuthError) as nous_exc:
        nous_run.ensure_usable_access_token(None)
    assert getattr(nous_exc.value, "category", None) is None
    assert should_try_fallback_on_auth_error(nous_exc.value) is True


def test_producer_level_missing_credential_categories(tmp_path, monkeypatch):
    """Key resolve-time producers emit the semantic category, not provider-specific code guesses."""
    from hermes_cli import auth as auth_mod
    from hermes_cli import auth_minimax, auth_qwen, auth_xai, runtime_provider
    from hermes_cli.auth_nous import _NousRuntimeResolve

    monkeypatch.setattr(auth_xai, "_load_auth_store_maybe_locked", lambda _lock=True: {})
    with pytest.raises(AuthError) as xai_exc:
        auth_xai._read_xai_oauth_tokens()
    assert getattr(xai_exc.value, "category", None) == AUTH_ERROR_CATEGORY_MISSING_CREDENTIAL

    monkeypatch.setattr(auth_mod, "get_provider_auth_state", lambda _provider: None)
    with pytest.raises(AuthError) as minimax_exc:
        auth_minimax._minimax_fresh_state()
    assert getattr(minimax_exc.value, "category", None) == AUTH_ERROR_CATEGORY_MISSING_CREDENTIAL

    nous_run = _NousRuntimeResolve(
        {},
        {},
        None,
        force_refresh=False,
        stale_access_token=None,
        timeout_seconds=1,
    )
    monkeypatch.setattr(nous_run, "merge_shared", lambda: False)
    with pytest.raises(AuthError) as nous_exc:
        nous_run.ensure_usable_access_token(None)
    assert getattr(nous_exc.value, "category", None) == AUTH_ERROR_CATEGORY_MISSING_CREDENTIAL

    missing_qwen = tmp_path / "missing-qwen-oauth.json"
    monkeypatch.setattr(auth_mod, "_qwen_cli_auth_path", lambda: missing_qwen)
    with pytest.raises(AuthError) as qwen_exc:
        auth_qwen.resolve_qwen_runtime_credentials()
    assert getattr(qwen_exc.value, "category", None) == AUTH_ERROR_CATEGORY_MISSING_CREDENTIAL

    monkeypatch.setattr("agent.vertex_adapter.get_vertex_config", lambda: ("", ""))
    with pytest.raises(AuthError) as vertex_exc:
        runtime_provider._resolve_vertex_runtime("vertex")
    assert getattr(vertex_exc.value, "category", None) == AUTH_ERROR_CATEGORY_MISSING_CREDENTIAL

    monkeypatch.setattr("hermes_cli.config.get_env_value", lambda _name: "")
    monkeypatch.setattr("hermes_cli.runtime_provider_backends.get_secret_str", lambda *_a, **_k: "")
    with pytest.raises(AuthError) as azure_exc:
        runtime_provider._resolve_azure_foundry_runtime(
            requested_provider="azure-foundry",
            model_cfg={
                "provider": "azure-foundry",
                "base_url": "https://example.services.ai.azure.com",
            },
        )
    assert getattr(azure_exc.value, "category", None) == AUTH_ERROR_CATEGORY_MISSING_CREDENTIAL


def test_category_checks_tolerate_custom_auth_error_without_attribute():
    """Older/custom AuthError subclasses may not expose the new semantic field."""
    from hermes_cli import runtime_provider
    from hermes_cli.auth import should_try_fallback_on_auth_error

    error = AuthError("legacy custom error")
    del error.category

    assert should_try_fallback_on_auth_error(error) is True
    assert runtime_provider._resolve_rung(
        "auto",
        lambda: (_ for _ in ()).throw(error),
    ) is None


def test_auto_rung_propagates_missing_category(monkeypatch):
    """Auto must not convert credential absence into a paid provider fallback."""
    from hermes_cli import auth as auth_mod
    from hermes_cli import runtime_provider

    missing = AuthError(
        "missing",
        category=AUTH_ERROR_CATEGORY_MISSING_CREDENTIAL,
    )
    with pytest.raises(AuthError):
        runtime_provider._resolve_rung("auto", lambda: (_ for _ in ()).throw(missing))
    with pytest.raises(AuthError):
        runtime_provider._resolve_rung("anthropic", lambda: (_ for _ in ()).throw(missing))

    monkeypatch.setattr(
        "agent.anthropic_credentials.resolve_anthropic_token",
        lambda *a, **k: "",
    )
    with pytest.raises(AuthError):
        runtime_provider._resolve_rung(
            "auto",
            lambda: runtime_provider._anthropic_env_runtime("auto", {}),
        )

    monkeypatch.setattr(
        auth_mod,
        "resolve_minimax_oauth_runtime_credentials",
        lambda: (_ for _ in ()).throw(
            AuthError("missing", category=AUTH_ERROR_CATEGORY_MISSING_CREDENTIAL)
        ),
    )
    with pytest.raises(AuthError):
        runtime_provider._resolve_rung(
            "auto",
            lambda: runtime_provider._minimax_oauth_runtime("minimax-oauth", "auto"),
        )


@pytest.mark.parametrize("provider", ["anthropic", "minimax-oauth"])
def test_auto_runtime_ladder_stops_after_provider_missing(provider, monkeypatch):
    """A missing configured provider must not consume the OpenRouter fallback."""
    from hermes_cli import auth as auth_mod
    from hermes_cli import runtime_provider

    monkeypatch.setattr(runtime_provider, "_resolve_requested_shortcuts", lambda *_args: None)
    monkeypatch.setattr(runtime_provider, "_resolve_named_custom_runtime", lambda **_kwargs: None)
    monkeypatch.setattr(runtime_provider, "_local_endpoint_bypass", lambda *_args: None)
    monkeypatch.setattr(runtime_provider, "resolve_provider", lambda *_args, **_kwargs: provider)
    monkeypatch.setattr(runtime_provider, "_get_model_config", lambda: {})
    monkeypatch.setattr(runtime_provider, "load_pool", lambda _provider: None)
    monkeypatch.setattr(
        runtime_provider,
        "_openrouter_fallback",
        lambda *_args: {"provider": "openrouter", "api_key": "fallback-key"},
    )
    monkeypatch.setattr(
        "agent.anthropic_credentials.resolve_anthropic_token",
        lambda *a, **k: "",
    )
    monkeypatch.setattr(
        auth_mod,
        "resolve_minimax_oauth_runtime_credentials",
        lambda: (_ for _ in ()).throw(
            AuthError("missing", category=AUTH_ERROR_CATEGORY_MISSING_CREDENTIAL)
        ),
    )

    with pytest.raises(AuthError) as exc_info:
        runtime_provider.resolve_runtime_provider(requested="auto")
    assert exc_info.value.category == AUTH_ERROR_CATEGORY_MISSING_CREDENTIAL


def test_auto_azure_shortcut_missing_category_does_not_fallback(monkeypatch):
    """A missing Azure shortcut credential must not spend an explicit paid route."""
    from hermes_cli import runtime_provider

    def missing_shortcut(*_args):
        raise AuthError(
            "Azure shortcut has no credential",
            category=AUTH_ERROR_CATEGORY_MISSING_CREDENTIAL,
        )

    monkeypatch.setattr(runtime_provider, "_resolve_requested_shortcuts", missing_shortcut)
    monkeypatch.setattr(runtime_provider, "_resolve_named_custom_runtime", lambda **_kwargs: None)
    monkeypatch.setattr(runtime_provider, "resolve_provider", lambda *_args, **_kwargs: "openrouter")
    monkeypatch.setattr(runtime_provider, "_get_model_config", lambda: {})
    monkeypatch.setattr(runtime_provider, "load_pool", lambda _provider: None)
    monkeypatch.setattr(
        runtime_provider,
        "_resolve_explicit_runtime",
        lambda **_kwargs: {"provider": "openrouter", "api_key": "explicit-route"},
    )

    with pytest.raises(AuthError) as exc_info:
        runtime_provider.resolve_runtime_provider(
            requested="auto",
            explicit_base_url="https://example.services.ai.azure.com",
        )
    assert exc_info.value.category == AUTH_ERROR_CATEGORY_MISSING_CREDENTIAL


def test_azure_explicit_shortcut_preserves_preexisting_empty_key_behavior(monkeypatch):
    """Keep the Azure Anthropic shortcut behavior unchanged; this PR only classifies producers."""
    from hermes_cli import runtime_provider

    monkeypatch.setattr(runtime_provider, "get_secret_str", lambda *_a, **_k: "")
    runtime = runtime_provider._resolve_requested_shortcuts(
        "anthropic",
        None,
        "https://example.services.ai.azure.com",
        None,
    )

    assert runtime["provider"] == "anthropic"
    assert runtime["api_key"] == ""
    assert runtime_provider._resolve_requested_shortcuts(
        "auto",
        None,
        "https://example.services.ai.azure.com",
        None,
    ) is None


def test_tui_missing_api_key_reraises_without_walking_chain(monkeypatch):
    """TUI resolve-time missing_api_key must re-raise, not switch provider."""
    from tui_gateway import server

    requested = []

    def fake_resolve(**kwargs):
        requested.append(kwargs.get("requested"))
        if kwargs.get("requested") == "openai":
            raise AuthError("No usable credentials", code="missing_api_key")
        return _fallback_resolve(**kwargs)

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider", fake_resolve
    )
    monkeypatch.setattr(
        server,
        "_load_fallback_model",
        lambda: [{"provider": "gemini", "model": "gemini-2.5-flash"}],
    )

    with pytest.raises(AuthError) as exc_info:
        server._resolve_runtime_with_fallback({"requested": "openai"})

    assert exc_info.value.code == "missing_api_key"
    assert requested == ["openai"]


def test_cron_missing_api_key_does_not_walk_fallback():
    """Cron must raise the formatted primary error, not swap to fallback."""
    from cron.scheduler import _CronJobConfig, _resolve_job_runtime

    jc = _CronJobConfig(
        cfg={"fallback_providers": [{"provider": "gemini", "model": "gemini-2.5-flash"}]},
        model="gpt-4",
        model_cfg={"provider": "openai"},
        cron_default_provider="",
    )
    requested = []

    def fake_resolve(**kwargs):
        requested.append(kwargs.get("requested"))
        if kwargs.get("requested") == "openai":
            raise AuthError("No usable credentials", code="missing_api_key")
        return _fallback_resolve(**kwargs)

    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=fake_resolve):
        with pytest.raises(RuntimeError) as exc_info:
            _resolve_job_runtime({"provider": "openai"}, "job-1", jc)

    assert isinstance(exc_info.value.__cause__, AuthError)
    assert exc_info.value.__cause__.code == "missing_api_key"
    assert requested == ["openai"]

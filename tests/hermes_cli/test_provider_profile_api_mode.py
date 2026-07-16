"""Regression tests: a registered ProviderProfile's ``api_mode`` must flow into
api_mode resolution for plugin providers.

Bug: ``providers/`` (ProviderProfile registry) and ``hermes_cli.providers``
(ProviderDef + HERMES_OVERLAYS) are two separate registries. A plugin provider
under ``plugins/model-providers/<name>/`` declares its wire protocol via
``ProviderProfile.api_mode``, but ``determine_api_mode()`` only consulted
models.dev + HERMES_OVERLAYS, so a plugin provider that speaks
``codex_responses`` or ``anthropic_messages`` on a base URL that URL heuristics
don't recognize (e.g. a loopback signing proxy) silently fell through to
``chat_completions``.

The bug manifested in two resolution paths, both covered here:
 1. ``hermes_cli.providers.determine_api_mode`` (CLI / model-switch)
 2. ``hermes_cli.runtime_provider`` generic branch (``--provider`` one-shot)
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from providers import register_provider
from providers.base import ProviderProfile


_TEST_PROVIDERS = (
    ("bm-test-responses", "codex_responses", "http://127.0.0.1:54321/openai/v1"),
    ("bm-test-anthropic", "anthropic_messages", "http://127.0.0.1:54321/mantle"),
    ("bm-test-chat", "chat_completions", "http://127.0.0.1:54321/v1"),
)


@pytest.fixture(autouse=True)
def _register_test_profiles():
    """Register throwaway plugin-style profiles for the duration of a test.

    Cleans up afterwards so these synthetic names don't leak into the
    process-global registry and perturb provider-count assertions in other
    test modules that share the same interpreter (xdist worker).
    """
    import providers as _pkg

    added_names = []
    added_aliases = []
    for name, api_mode, base_url in _TEST_PROVIDERS:
        prof = ProviderProfile(
            name=name,
            api_mode=api_mode,
            env_vars=("BM_TEST_KEY",),
            base_url=base_url,
            auth_type="api_key",
        )
        # Ensure discovery has run before we mutate, so our entries aren't
        # wiped by a later lazy _discover_providers().
        _pkg._discover_providers()
        register_provider(prof)
        added_names.append(name)
        added_aliases.extend(prof.aliases)
    try:
        yield
    finally:
        for name in added_names:
            _pkg._REGISTRY.pop(name, None)
        for alias in added_aliases:
            _pkg._ALIASES.pop(alias, None)


class TestDetermineApiModeUsesProfile:
    """determine_api_mode() must honor a registered profile's api_mode when its
    built-in tables (models.dev + HERMES_OVERLAYS) don't know the provider."""

    def test_responses_profile_resolves_codex(self):
        from hermes_cli.providers import determine_api_mode
        assert determine_api_mode("bm-test-responses", "") == "codex_responses"

    def test_anthropic_profile_resolves_messages(self):
        from hermes_cli.providers import determine_api_mode
        assert determine_api_mode("bm-test-anthropic", "") == "anthropic_messages"

    def test_chat_profile_resolves_chat(self):
        from hermes_cli.providers import determine_api_mode
        assert determine_api_mode("bm-test-chat", "") == "chat_completions"

    def test_no_base_url_uses_profile_mode(self):
        # The common plugin case: --provider with no base_url override. The
        # profile's declared codex_responses must win over the chat default.
        from hermes_cli.providers import determine_api_mode
        assert determine_api_mode("bm-test-responses", "") == "codex_responses"

    def test_profile_own_base_url_uses_profile_mode(self):
        # base_url equal to the profile's OWN declared endpoint (e.g. the
        # loopback signing proxy the plugin registered) → profile mode wins.
        # This is the exact bedrock-mantle case: the auto-extended registry
        # feeds the runtime resolver the profile's own loopback base_url.
        from hermes_cli.providers import determine_api_mode
        assert determine_api_mode(
            "bm-test-responses", "http://127.0.0.1:54321/openai/v1"
        ) == "codex_responses"

    def test_different_base_url_override_defers_to_default(self):
        # A user base_url override to a DIFFERENT endpoint (matches no
        # heuristic, differs from the profile's declared base_url) opts out of
        # the profile mode — mirrors MiniMax's /v1 opt-out of its default
        # /anthropic route. Must fall back to chat_completions.
        from hermes_cli.providers import determine_api_mode
        assert determine_api_mode(
            "bm-test-responses", "https://some-other-proxy.example.test/v1"
        ) == "chat_completions"

    def test_url_heuristic_still_wins_over_profile(self):
        # A user base_url override that IS recognized (/anthropic suffix) must
        # take precedence over the profile's declared api_mode.
        from hermes_cli.providers import determine_api_mode
        assert determine_api_mode(
            "bm-test-responses", "https://gateway.example.test/anthropic"
        ) == "anthropic_messages"

    def test_unknown_provider_still_defaults_chat(self):
        from hermes_cli.providers import determine_api_mode
        assert determine_api_mode("totally-unregistered-xyz", "") == "chat_completions"


class TestProviderProfileApiModeHelper:
    def test_helper_returns_declared_mode(self):
        from hermes_cli.providers import _provider_profile_api_mode
        assert _provider_profile_api_mode("bm-test-responses") == "codex_responses"

    def test_helper_empty_for_unknown(self):
        from hermes_cli.providers import _provider_profile_api_mode
        assert _provider_profile_api_mode("totally-unregistered-xyz") == ""

    def test_helper_defers_on_different_base_url(self):
        from hermes_cli.providers import _provider_profile_api_mode
        assert _provider_profile_api_mode(
            "bm-test-responses", "https://other.example.test/v1"
        ) == ""

    def test_helper_applies_on_own_base_url(self):
        from hermes_cli.providers import _provider_profile_api_mode
        assert _provider_profile_api_mode(
            "bm-test-responses", "http://127.0.0.1:54321/openai/v1"
        ) == "codex_responses"

    def test_runtime_provider_helper_delegates(self):
        # runtime_provider's helper must return the same value (single source
        # of truth via lazy import).
        from hermes_cli.runtime_provider import _provider_profile_api_mode as rt_impl
        assert rt_impl("bm-test-anthropic") == "anthropic_messages"


class TestRuntimeResolutionHonorsProfile:
    """E2E: the --provider one-shot path (resolve_runtime_provider) must route
    a plugin provider to its declared api_mode. This is the exact path the
    bedrock-mantle plugin's GPT-5 sibling depends on."""

    @pytest.fixture(autouse=True)
    def _hermetic_runtime(self, monkeypatch):
        """Isolate resolve_runtime_provider from ambient state.

        These are E2E tests over the real resolution chain, so anything the
        chain reads from the developer's / CI's environment can divert it away
        from the profile-fallback branch we're asserting. Two concrete leaks
        this neutralizes (both observed failing under a full-suite run, passing
        in isolation):

          * ``load_pool`` — an earlier test in the session can leave the
            credential-pool path primed so a ``bm-test-*`` provider resolves via
            a pooled entry instead of the generic branch. Force empty pools.
          * ambient provider credentials — a real ``OPENAI_API_KEY`` /
            ``OPENAI_BASE_URL`` (or OpenRouter/Anthropic) in ``~/.hermes/.env``
            lets the env-var credential path resolve the synthetic provider as a
            real one *before* the ProviderProfile helper is consulted, so the
            asserted api_mode silently degrades to ``chat_completions``.

        Clearing them makes the generic ProviderProfile branch the deterministic
        landing point regardless of test ordering or host env.
        """
        import hermes_cli.runtime_provider as rp

        monkeypatch.setattr(
            rp, "load_pool", lambda _provider: SimpleNamespace(has_credentials=lambda: False)
        )
        monkeypatch.setattr(rp, "_get_model_config", lambda: {})
        for _var in (
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "OPENROUTER_API_KEY",
            "OPENROUTER_BASE_URL",
            "ANTHROPIC_API_KEY",
        ):
            monkeypatch.delenv(_var, raising=False)
        monkeypatch.setenv("BM_TEST_KEY", "x")

        # The synthetic bm-test-* profiles are api_key providers with no real
        # backing credential store, so the resolver's fail-closed key guard
        # (raise AuthError on an empty key) fires before api_mode is computed.
        # Return a usable key + the provider's own default endpoint so the
        # resolution chain proceeds to the ProviderProfile api_mode branch we're
        # asserting — the value under test — without a live credential backend.
        import providers as _pkg

        def _fake_api_key_creds(provider, *a, **k):
            prof = _pkg.get_provider_profile(provider)
            return {
                "provider": provider,
                "api_key": "bm-test-key",
                "base_url": (getattr(prof, "base_url", "") if prof else "") or "",
                "source": "test",
            }

        monkeypatch.setattr(rp, "resolve_api_key_provider_credentials", _fake_api_key_creds)

    def test_resolve_runtime_provider_uses_profile_api_mode(self, monkeypatch):
        import hermes_cli.runtime_provider as rp

        monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "bm-test-responses")

        resolved = rp.resolve_runtime_provider(requested="bm-test-responses")

        assert resolved["provider"] == "bm-test-responses"
        assert resolved["api_mode"] == "codex_responses"

    def test_resolve_runtime_provider_anthropic_profile(self, monkeypatch):
        import hermes_cli.runtime_provider as rp

        monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "bm-test-anthropic")

        resolved = rp.resolve_runtime_provider(requested="bm-test-anthropic")

        assert resolved["api_mode"] == "anthropic_messages"

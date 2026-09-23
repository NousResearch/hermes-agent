"""Tests for provider-aware `/model` validation in hermes_cli.models."""

import pytest
from unittest.mock import MagicMock, patch

from hermes_cli.models import azure_foundry_model_api_mode, copilot_model_api_mode, fetch_github_model_catalog, curated_models_for_provider, fetch_api_models, github_model_reasoning_efforts, normalize_copilot_model_id, normalize_opencode_model_id, normalize_provider, opencode_model_api_mode, parse_model_input, probe_api_models, provider_label, provider_model_ids
from hermes_cli.models_local import fetch_lmstudio_models
from hermes_cli.models_validate import validate_requested_model


# -- helpers -----------------------------------------------------------------

FAKE_API_MODELS = [
    "anthropic/claude-opus-4.6",
    "anthropic/claude-sonnet-4.5",
    "openai/gpt-5.4-pro",
    "openai/gpt-5.4",
    "google/gemini-3-pro-preview",
]


def _validate(model, provider="openrouter", api_models=FAKE_API_MODELS, **kw):
    """Shortcut: call validate_requested_model with mocked API."""
    probe_payload = {
        "models": api_models,
        "probed_url": "http://localhost:11434/v1/models",
        "resolved_base_url": kw.get("base_url", "") or "http://localhost:11434/v1",
        "suggested_base_url": None,
        "used_fallback": False,
    }
    with patch("hermes_cli.models.fetch_api_models", return_value=api_models), \
         patch("hermes_cli.models.probe_api_models", return_value=probe_payload):
        return validate_requested_model(model, provider, **kw)


# -- parse_model_input -------------------------------------------------------

class TestParseModelInput:
    def test_plain_model_keeps_current_provider(self):
        provider, model = parse_model_input("anthropic/claude-sonnet-4.5", "openrouter")
        assert provider == "openrouter"
        assert model == "anthropic/claude-sonnet-4.5"


# -- curated_models_for_provider ---------------------------------------------

class TestCuratedModelsForProvider:
    def test_openrouter_returns_curated_list(self):
        with patch(
            "hermes_cli.models.fetch_openrouter_models",
            return_value=[
                ("anthropic/claude-opus-4.6", "recommended"),
                ("qwen/qwen3.6-plus", ""),
            ],
        ):
            models = curated_models_for_provider("openrouter")
        assert len(models) > 0
        assert any("claude" in m[0] for m in models)

    def test_unknown_provider_returns_empty(self):
        assert curated_models_for_provider("totally-unknown") == []

    def test_live_catalog_projected_to_tuples_else_static_fallback(self):
        with patch("hermes_cli.models.provider_model_ids", return_value=["m-live"]):
            assert curated_models_for_provider("nous") == [("m-live", "")]
        with patch("hermes_cli.models.provider_model_ids", return_value=[]), patch.dict(
            "hermes_cli.models._PROVIDER_MODELS", {"nous": ["m-static"]}
        ):
            assert curated_models_for_provider("nous") == [("m-static", "")]


# -- normalize_provider ------------------------------------------------------

class TestNormalizeProvider:

    def test_known_aliases(self):
        assert normalize_provider("glm") == "zai"
        assert normalize_provider("kimi") == "kimi-coding"
        assert normalize_provider("moonshot") == "kimi-coding"
        assert normalize_provider("step") == "stepfun"
        assert normalize_provider("github-copilot") == "copilot"


class TestProviderLabel:
    def test_known_labels_and_auto(self):
        assert provider_label("anthropic") == "Anthropic"
        assert provider_label("kimi") == "Kimi / Kimi Coding Plan"
        assert provider_label("stepfun") == "StepFun Step Plan"
        assert provider_label("copilot") == "GitHub Copilot"
        assert provider_label("copilot-acp") == "GitHub Copilot ACP"
        assert provider_label("auto") == "Auto"


# -- provider_model_ids ------------------------------------------------------

class TestProviderModelIds:


    def test_stepfun_prefers_live_catalog(self):
        with patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={"api_key": "***", "base_url": "https://api.stepfun.com/step_plan/v1"},
        ), patch(
            "hermes_cli.models.fetch_api_models",
            return_value=["step-3.5-flash", "step-3-agent-lite"],
        ):
            assert provider_model_ids("stepfun") == ["step-3.5-flash", "step-3-agent-lite"]


    def test_anthropic_provider_uses_configured_base_url_for_live_catalog(self):
        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"data": [{"id": "enterprise-claude"}]}'

        with patch(
            "hermes_cli.config.load_config",
            return_value={
                "model": {
                    "provider": "anthropic",
                    "base_url": "http://localhost:6655/anthropic/v1",
                    "api_key": "proxy-key",
                }
            },
        ), patch(
            "hermes_cli.models._urlopen_model_catalog_request",
            return_value=_Resp(),
        ) as mock_urlopen:
            assert provider_model_ids("anthropic") == ["enterprise-claude"]

        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://localhost:6655/anthropic/v1/models?limit=1000"
        assert req.get_header("X-api-key") == "proxy-key"

    def test_custom_provider_passes_anthropic_mode_for_versioned_proxy_catalog(self):
        with patch(
            "hermes_cli.config.load_config",
            return_value={
                "model": {
                    "provider": "custom",
                    "base_url": "http://localhost:6655/anthropic/v1",
                    "api_key": "proxy-key",
                }
            },
        ), patch(
            "hermes_cli.models.fetch_api_models",
            return_value=["enterprise-claude"],
        ) as mock_fetch:
            assert provider_model_ids("custom") == ["enterprise-claude"]

        mock_fetch.assert_called_once_with(
            "proxy-key",
            "http://localhost:6655/anthropic/v1",
            api_mode="anthropic_messages",
        )


# -- fetch_api_models --------------------------------------------------------

class TestFetchApiModels:
    def test_returns_none_when_no_base_url(self):
        assert fetch_api_models("key", None) is None


    def test_probe_api_models_tries_v1_fallback(self):
        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"data": [{"id": "local-model"}]}'

        calls = []

        def _fake_urlopen(req, timeout=5.0):
            calls.append(req.full_url)
            if req.full_url.endswith("/v1/models"):
                return _Resp()
            raise Exception("404")

        with patch("hermes_cli.models._urlopen_model_catalog_request", side_effect=_fake_urlopen):
            probe = probe_api_models("key", "http://localhost:8000")

        assert calls == ["http://localhost:8000/models", "http://localhost:8000/v1/models"]
        assert probe["models"] == ["local-model"]
        assert probe["resolved_base_url"] == "http://localhost:8000/v1"
        assert probe["used_fallback"] is True

    def test_probe_api_models_uses_copilot_catalog(self):
        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"data": [{"id": "gpt-5.4", "model_picker_enabled": true, "supported_endpoints": ["/responses"], "capabilities": {"type": "chat", "supports": {"reasoning_effort": ["low", "medium", "high"]}}}, {"id": "claude-sonnet-4.6", "model_picker_enabled": true, "supported_endpoints": ["/chat/completions"], "capabilities": {"type": "chat", "supports": {"reasoning_effort": ["low", "medium", "high"]}}}, {"id": "text-embedding-3-small", "model_picker_enabled": true, "capabilities": {"type": "embedding"}}]}'

        with patch("hermes_cli.models._urlopen_model_catalog_request", return_value=_Resp()) as mock_urlopen:
            probe = probe_api_models("gh-token", "https://api.githubcopilot.com")

        assert mock_urlopen.call_args[0][0].full_url == "https://api.githubcopilot.com/models"
        assert probe["models"] == ["gpt-5.4", "claude-sonnet-4.6"]
        assert probe["resolved_base_url"] == "https://api.githubcopilot.com"
        assert probe["used_fallback"] is False


def _reset_reasoning_token_cache(models_mod):
    """Clear the reasoning-lookup token cache.

    The cache is module-level and deliberately long-lived in production, so
    tests must reset it or they leak a resolved token into each other and
    become order-dependent.
    """
    models_mod._copilot_reasoning_token_cache.clear()
    with models_mod._copilot_reasoning_token_locks_guard:
        models_mod._copilot_reasoning_token_locks.clear()


class TestGithubReasoningEfforts:
    @pytest.fixture(autouse=True)
    def _isolate_token_cache(self):
        """Keep the module-level token cache from leaking between tests.

        Reset before AND after: a test that resolves a real token would
        otherwise satisfy a later test's cache assertion for free.
        """
        import hermes_cli.models as models_mod

        _reset_reasoning_token_cache(models_mod)
        yield
        _reset_reasoning_token_cache(models_mod)

    def test_gpt5_supports_minimal_to_high(self):
        catalog = [{
            "id": "gpt-5.4",
            "capabilities": {"type": "chat", "supports": {"reasoning_effort": ["low", "medium", "high"]}},
            "supported_endpoints": ["/responses"],
        }]
        assert github_model_reasoning_efforts("gpt-5.4", catalog=catalog) == [
            "low",
            "medium",
            "high",
        ]

    def test_explicit_catalog_is_authoritative_for_non_gpt_families(self):
        catalog = [
            {
                "id": "claude-opus-5",
                "capabilities": {
                    "type": "chat",
                    "supports": {"reasoning_effort": ["low", "medium", "high", "xhigh", "max"]},
                },
            },
            {
                "id": "gemini-3.6-flash",
                "capabilities": {
                    "type": "chat",
                    "supports": {"reasoning_effort": ["minimal", "low", "medium", "high"]},
                },
            },
            {
                "id": "grok-4.6",
                "capabilities": {
                    "type": "chat",
                    "supports": {"reasoning_effort": ["low", "medium", "high", "xhigh"]},
                },
            },
            {
                "id": "claude-haiku-4.5",
                "capabilities": {"type": "chat", "supports": {}},
            },
        ]
        assert github_model_reasoning_efforts("claude-opus-5", catalog=catalog) == [
            "low", "medium", "high", "xhigh", "max",
        ]
        assert github_model_reasoning_efforts("gemini-3.6-flash", catalog=catalog) == [
            "minimal", "low", "medium", "high",
        ]
        assert github_model_reasoning_efforts("grok-4.6", catalog=catalog) == [
            "low", "medium", "high", "xhigh",
        ]
        assert github_model_reasoning_efforts("claude-haiku-4.5", catalog=catalog) == []

    def test_missing_catalog_auto_resolves_token_and_uses_live_catalog(self):
        catalog = [{
            "id": "claude-opus-5",
            "capabilities": {
                "type": "chat",
                "supports": {"reasoning_effort": ["low", "medium", "high", "xhigh", "max"]},
            },
        }]
        with patch(
            "hermes_cli.models._resolve_copilot_catalog_api_key",
            return_value="tok",
        ) as mock_key, patch(
            "hermes_cli.models.fetch_github_model_catalog",
            return_value=catalog,
        ) as mock_fetch:
            assert github_model_reasoning_efforts("claude-opus-5") == [
                "low", "medium", "high", "xhigh", "max",
            ]
        mock_key.assert_called_once()
        mock_fetch.assert_called_once()

    def test_explicit_catalog_skips_token_resolve(self):
        catalog = [{
            "id": "claude-opus-5",
            "capabilities": {
                "type": "chat",
                "supports": {"reasoning_effort": ["low", "medium", "high"]},
            },
        }]
        with patch(
            "hermes_cli.models._resolve_copilot_catalog_api_key",
            return_value="tok",
        ) as mock_key:
            assert github_model_reasoning_efforts(
                "claude-opus-5", catalog=catalog
            ) == ["low", "medium", "high"]
        mock_key.assert_not_called()

    def test_token_resolution_reaches_the_credential_pool(self, monkeypatch):
        """A pool-only Copilot credential must reach the reasoning lookup.

        The token helper resolves credentials in two stages: the provider
        credentials (env vars / `gh auth token`), then `read_credential_pool`.
        Only the second stage covers users whose token lives in
        `credential_pool.copilot[]` (added by `hermes auth add copilot`).

        This asserts the observable behaviour — with the first stage empty,
        a pool entry still ends up as the api_key the catalog is fetched
        with. Any regression that bypasses the pool-aware path (including a
        duplicate helper definition shadowing it) fails here, because the
        catalog is then fetched with no credential and the model resolves
        through the offline fallback instead of the catalog ladder.
        """
        import hermes_cli.models as models_mod

        # Stage 1 yields nothing; stage 2 (the pool) holds the only token.
        monkeypatch.setattr(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            lambda _provider: {},
            raising=False,
        )
        monkeypatch.setattr(
            "hermes_cli.auth.read_credential_pool",
            lambda _provider: [{"access_token": "gho_pool_only"}],
            raising=False,
        )
        monkeypatch.setattr(
            "hermes_cli.copilot_auth.validate_copilot_token",
            lambda _token: (True, ""),
            raising=False,
        )
        monkeypatch.setattr(
            "hermes_cli.copilot_auth.exchange_copilot_token",
            lambda token, **_kw: (f"exchanged::{token}", 9e9),
            raising=False,
        )
        _reset_reasoning_token_cache(models_mod)

        seen_keys = []

        def _capture(api_key=None, **_kw):
            seen_keys.append(api_key)
            return [{
                "id": "claude-opus-5",
                "capabilities": {
                    "type": "chat",
                    "supports": {"reasoning_effort": ["low", "medium", "high"]},
                },
            }]

        monkeypatch.setattr(models_mod, "fetch_github_model_catalog", _capture)

        efforts = models_mod.github_model_reasoning_efforts("claude-opus-5")

        assert seen_keys and seen_keys[0], (
            "the catalog was fetched with no credential — the pool-aware "
            "resolution path was bypassed"
        )
        assert "gho_pool_only" in seen_keys[0], (
            f"expected the pool token to reach the catalog fetch, got {seen_keys[0]!r}"
        )
        # And the catalog ladder wins, rather than the offline fallback.
        assert efforts == ["low", "medium", "high"]

    def test_resolved_token_is_cached_across_calls(self, monkeypatch):
        """The token must not be re-resolved on every lookup.

        `github_model_reasoning_efforts` runs on hot paths (the composer's
        reasoning chip, the agent's per-turn send gates). Token resolution
        can shell out to `gh auth token` and exchange pool candidates, so
        re-resolving per call adds a large fixed cost to every turn.
        """
        import hermes_cli.models as models_mod

        calls = []
        monkeypatch.setattr(
            models_mod,
            "_resolve_copilot_catalog_api_key",
            lambda: (calls.append(1), "tok")[1],
        )
        monkeypatch.setattr(
            models_mod,
            "fetch_github_model_catalog",
            lambda **_kw: [{
                "id": "claude-opus-5",
                "capabilities": {
                    "type": "chat",
                    "supports": {"reasoning_effort": ["low", "medium", "high"]},
                },
            }],
        )
        _reset_reasoning_token_cache(models_mod)

        for _ in range(5):
            models_mod.github_model_reasoning_efforts("claude-opus-5")

        assert len(calls) == 1, (
            f"token resolved {len(calls)} times for 5 lookups — expected 1 "
            "(the result must be cached)"
        )

    @pytest.mark.parametrize("resolved_token", ["tok", ""])
    def test_concurrent_callers_share_one_profile_resolution(
        self, monkeypatch, resolved_token
    ):
        """Positive and negative refreshes are single-flight per profile."""
        import threading

        from hermes_constants import hermes_home_key
        import hermes_cli.models as models_mod

        worker_count = 8
        calls = []
        release_resolver = threading.Event()
        resolver_started = threading.Event()
        all_lock_requests_arrived = threading.Event()
        lock_request_barrier = threading.Barrier(
            worker_count, action=all_lock_requests_arrived.set
        )

        real_lock_for = models_mod._copilot_reasoning_token_lock_for
        profile_key = hermes_home_key()
        assert real_lock_for(profile_key) is real_lock_for(profile_key)

        def _lock_for(key):
            lock = real_lock_for(key)
            lock_request_barrier.wait(timeout=5)
            return lock

        def _resolve():
            calls.append(threading.get_ident())
            resolver_started.set()
            release_resolver.wait(timeout=5)
            return resolved_token

        monkeypatch.setattr(models_mod, "_copilot_reasoning_token_lock_for", _lock_for)
        monkeypatch.setattr(models_mod, "_resolve_copilot_catalog_api_key", _resolve)
        results = []
        threads = [
            threading.Thread(target=lambda: results.append(
                models_mod._cached_copilot_reasoning_token()))
            for _ in range(worker_count)
        ]
        for thread in threads:
            thread.start()
        assert all_lock_requests_arrived.wait(timeout=5)
        assert resolver_started.wait(timeout=5)
        release_resolver.set()
        for thread in threads:
            thread.join(timeout=5)

        assert not any(thread.is_alive() for thread in threads)
        assert len(calls) == 1
        assert results == [resolved_token] * worker_count

    def test_different_profiles_resolve_without_cross_profile_blocking(
        self, monkeypatch, tmp_path
    ):
        """One profile's slow credential lookup must not serialize another."""
        import threading

        from hermes_constants import (
            hermes_home_key,
            reset_hermes_home_override,
            set_hermes_home_override,
        )
        import hermes_cli.models as models_mod

        home_a = tmp_path / "profile-a"
        home_b = tmp_path / "profile-b"
        home_a.mkdir()
        home_b.mkdir()
        key_a = hermes_home_key(home_a)
        entered_a = threading.Event()
        release_a = threading.Event()
        finished_b = threading.Event()
        results = {}

        def _resolve():
            key = hermes_home_key()
            if key == key_a:
                entered_a.set()
                release_a.wait(timeout=5)
                return "token-a"
            finished_b.set()
            return "token-b"

        monkeypatch.setattr(models_mod, "_resolve_copilot_catalog_api_key", _resolve)

        def _under(label, home):
            override = set_hermes_home_override(home)
            try:
                results[label] = models_mod._cached_copilot_reasoning_token()
            finally:
                reset_hermes_home_override(override)

        thread_a = threading.Thread(target=_under, args=("a", home_a))
        thread_b = threading.Thread(target=_under, args=("b", home_b))
        thread_a.start()
        assert entered_a.wait(timeout=2)
        thread_b.start()
        try:
            assert finished_b.wait(timeout=2)
        finally:
            release_a.set()
        thread_a.join(timeout=5)
        thread_b.join(timeout=5)

        assert not thread_a.is_alive()
        assert not thread_b.is_alive()
        assert results == {"a": "token-a", "b": "token-b"}

    def test_token_cache_ttl_starts_after_resolution_completes(self, monkeypatch):
        """A slow fresh resolution must receive its full cache lifetime."""
        import hermes_cli.models as models_mod

        clock = [0.0]
        calls = []

        def _resolve():
            calls.append(1)
            clock[0] = 299.0
            return "tok"

        monkeypatch.setattr(models_mod.time, "monotonic", lambda: clock[0])
        monkeypatch.setattr(models_mod, "_resolve_copilot_catalog_api_key", _resolve)

        assert models_mod._cached_copilot_reasoning_token() == "tok"
        clock[0] = 300.0
        assert models_mod._cached_copilot_reasoning_token() == "tok"
        assert len(calls) == 1

    @pytest.mark.parametrize(
        "first_token,second_token",
        [
            ("gho_profile_a", "gho_profile_b"),
            (None, "gho_profile_b"),
        ],
    )
    def test_resolved_token_cache_is_scoped_by_profile(
        self, monkeypatch, tmp_path, first_token, second_token
    ):
        """Each real profile pool owns its positive or negative cache result.

        Gateway workers multiplex profile homes in one process by changing the
        context-local Hermes home. A process-global cache lets the first profile
        donate its Copilot account (or its missing-credential result) to every
        profile served within the TTL.
        """
        import json

        from hermes_constants import (
            hermes_home_key,
            reset_hermes_home_override,
            set_hermes_home_override,
        )
        import hermes_cli.models as models_mod

        homes = [tmp_path / "profile-a", tmp_path / "profile-b"]
        raw_tokens = [first_token or "not-a-copilot-token", second_token]
        for home, raw_token in zip(homes, raw_tokens):
            home.mkdir()
            (home / "auth.json").write_text(
                json.dumps({
                    "version": 1,
                    "providers": {},
                    "credential_pool": {
                        "copilot": [{"access_token": raw_token}],
                    },
                }),
                encoding="utf-8",
            )

        # Preserve the profile-aware credential-pool path. Only remove the
        # process-global credential source and the external token exchange.
        monkeypatch.setattr(models_mod, "_api_key_credentials", lambda _provider: ("", ""))
        monkeypatch.setattr(models_mod, "_copilot_cli_config_tokens", lambda: [])
        exchanged = []

        def _exchange(raw_token):
            exchanged.append(raw_token)
            return f"exchanged::{raw_token}", 9e9, "https://api.githubcopilot.com"

        monkeypatch.setattr("hermes_cli.copilot_auth.exchange_copilot_token", _exchange)

        fetched_with = []

        def _fetch_for_token(api_key=None, **_kwargs):
            fetched_with.append(api_key)
            if not api_key:
                return None
            level = "low" if api_key.endswith("profile_a") else "high"
            return [{
                "id": "claude-opus-5",
                "capabilities": {
                    "type": "chat",
                    "supports": {"reasoning_effort": [level]},
                },
            }]

        monkeypatch.setattr(models_mod, "fetch_github_model_catalog", _fetch_for_token)

        real_resolver = models_mod._resolve_copilot_catalog_api_key
        resolved_homes = []

        def _tracked_resolver():
            resolved_homes.append(hermes_home_key())
            return real_resolver()

        monkeypatch.setattr(models_mod, "_resolve_copilot_catalog_api_key", _tracked_resolver)

        def _under(home):
            override = set_hermes_home_override(home)
            try:
                return models_mod.github_model_reasoning_efforts("claude-opus-5")
            finally:
                reset_hermes_home_override(override)

        expected_a = ["low"] if first_token else ["low", "medium", "high"]
        assert _under(homes[0]) == expected_a
        assert _under(homes[1]) == ["high"]
        # Switching back must reuse A's own token bucket and account catalog.
        assert _under(homes[0]) == expected_a
        assert resolved_homes == [
            hermes_home_key(homes[0]),
            hermes_home_key(homes[1]),
        ]
        assert exchanged == ([first_token] if first_token else []) + [second_token]
        expected_fetches = [
            "exchanged::gho_profile_a",
            "exchanged::gho_profile_b",
            "exchanged::gho_profile_a",
        ] if first_token else ["exchanged::gho_profile_b"]
        assert fetched_with == expected_fetches

    def test_offline_fallback_covers_known_copilot_reasoning_families(self):
        with patch(
            "hermes_cli.models._resolve_copilot_catalog_api_key",
            return_value="",
        ), patch(
            "hermes_cli.models.fetch_github_model_catalog",
            return_value=None,
        ):
            for model_id in (
                "claude-opus-5",
                "claude-opus-4.6",
                "claude-sonnet-4.6",
                "claude-sonnet-5",
                "gemini-3.6-flash",
                "gemini-3.7-flash",
                "grok-4.5",
                "grok-4.6",
                "mai-code-1.1-flash",
            ):
                efforts = github_model_reasoning_efforts(model_id)
                assert "medium" in efforts, model_id
                assert "high" in efforts, model_id
            assert github_model_reasoning_efforts("claude-haiku-4.5") == []
            assert github_model_reasoning_efforts("gpt-5.5") == [
                "minimal", "low", "medium", "high",
            ]

    @pytest.mark.parametrize(
        "model_id,expected",
        [
            # Explicit major.minor at or above the adaptive-thinking line.
            ("claude-opus-4.6", True),
            ("claude-opus-4.7", True),
            ("claude-opus-4.8", True),
            ("claude-sonnet-4.6", True),
            # Bare major >= 5 — no minor component at all.
            ("claude-opus-5", True),
            ("claude-sonnet-5", True),
            ("claude-5", True),
            # A size qualifier after the version must not change the verdict.
            # Haiku having no ladder is a fact about the 4.5 generation, not
            # about the name "haiku", so a hypothetical 5-series haiku follows
            # its version like every other model.
            ("claude-5-haiku", True),
            # Below the line: explicit minor < 6 on major 4.
            ("claude-haiku-4.5", False),
            ("claude-sonnet-4.5", False),
            ("claude-opus-4.1", False),
            # Bare major 4 — no minor, and 4 < 5.
            ("claude-4", False),
            ("claude-opus-4", False),
            # Claude 3 family.
            ("claude-3-opus", False),
            ("claude-3-5-sonnet", False),
            ("claude-3.7-sonnet", False),
            # Date-stamped builds must read as their major, never as a minor.
            ("claude-opus-4-20250514", False),
            ("claude-sonnet-4-20250514", False),
            # Unversioned ids: the generation cannot be established, so the
            # fallback must fail closed rather than guess.
            ("claude-sonnet", False),
            ("claude-opus", False),
            ("claude", False),
        ],
    )
    def test_offline_claude_version_boundaries(self, model_id, expected):
        """Pin the offline Claude gate at every boundary.

        This is the fallback used when the live catalog is unreachable, so a
        wrong verdict here is user-visible: too permissive offers a control
        the provider rejects with a 400, too strict hides a working one.
        """
        with patch(
            "hermes_cli.models._resolve_copilot_catalog_api_key",
            return_value="",
        ), patch(
            "hermes_cli.models.fetch_github_model_catalog",
            return_value=None,
        ):
            efforts = github_model_reasoning_efforts(model_id)

        if expected:
            assert efforts == ["low", "medium", "high"], (
                f"{model_id} should expose the conservative offline ladder"
            )
        else:
            assert efforts == [], (
                f"{model_id} must not expose a ladder offline — its generation "
                "is either below the adaptive-thinking line or unknown"
            )


class TestCopilotNormalization:

    def test_copilot_api_mode_gpt5_uses_responses(self):
        """GPT-5+ models should use Responses API (matching opencode)."""
        assert copilot_model_api_mode("gpt-5.4") == "codex_responses"
        assert copilot_model_api_mode("gpt-5.4-mini") == "codex_responses"
        assert copilot_model_api_mode("gpt-5.3-codex") == "codex_responses"
        assert copilot_model_api_mode("gpt-5.2-codex") == "codex_responses"
        assert copilot_model_api_mode("gpt-5.2") == "codex_responses"





    def test_opencode_go_api_modes_match_docs(self):
        assert opencode_model_api_mode("opencode-go", "glm-5.1") == "chat_completions"
        assert opencode_model_api_mode("opencode-go", "opencode-go/glm-5.1") == "chat_completions"
        assert opencode_model_api_mode("opencode-go", "glm-5") == "chat_completions"
        assert opencode_model_api_mode("opencode-go", "opencode-go/glm-5") == "chat_completions"
        assert opencode_model_api_mode("opencode-go", "kimi-k2.5") == "chat_completions"
        assert opencode_model_api_mode("opencode-go", "opencode-go/kimi-k2.5") == "chat_completions"
        assert opencode_model_api_mode("opencode-go", "minimax-m2.5") == "anthropic_messages"
        assert opencode_model_api_mode("opencode-go", "opencode-go/minimax-m2.5") == "anthropic_messages"
        assert opencode_model_api_mode("opencode-go", "qwen3.7-max") == "anthropic_messages"
        assert opencode_model_api_mode("opencode-go", "opencode-go/qwen3.7-max") == "anthropic_messages"
        # All Qwen models on Go route via /v1/messages (Go endpoint table).
        assert opencode_model_api_mode("opencode-go", "qwen3.7-plus") == "anthropic_messages"
        assert opencode_model_api_mode("opencode-go", "qwen3.6-plus") == "anthropic_messages"
        # DeepSeek / MiMo on Go are OpenAI-compatible chat completions.
        assert opencode_model_api_mode("opencode-go", "deepseek-v4-pro") == "chat_completions"
        assert opencode_model_api_mode("opencode-go", "deepseek-v4-flash") == "chat_completions"
        assert opencode_model_api_mode("opencode-go", "mimo-v2.5") == "chat_completions"
        assert opencode_model_api_mode("opencode-go", "kimi-k2.7-code") == "chat_completions"
        assert opencode_model_api_mode("opencode-go", "glm-5.2") == "chat_completions"
        assert opencode_model_api_mode("opencode-go", "minimax-m3") == "anthropic_messages"
        # Union Alpha is exposed through /v1/messages on both relays.
        assert opencode_model_api_mode("opencode-go", "union-alpha") == "anthropic_messages"
        assert opencode_model_api_mode("opencode-zen", "opencode-zen/union-alpha") == "anthropic_messages"
        # GPT models on Go are Responses-only (Go endpoint table).
        assert opencode_model_api_mode("opencode-go", "gpt-5.6-luna") == "codex_responses"
        assert opencode_model_api_mode("opencode-go", "opencode-go/gpt-5.6-luna") == "codex_responses"
        # Muse Spark on Go is Responses-only. chat/completions returns HTTP 503.
        assert opencode_model_api_mode("opencode-go", "muse-spark-1.2-contributor") == "codex_responses"
        assert opencode_model_api_mode("opencode-go", "opencode-go/muse-spark-1.2-contributor") == "codex_responses"
        assert opencode_model_api_mode("opencode-go", "muse-spark-1.2") == "codex_responses"
        # Zen serves the standard Muse Spark variant on /v1/responses too.
        assert opencode_model_api_mode("opencode-zen", "muse-spark-1.2") == "codex_responses"
        assert opencode_model_api_mode("opencode-zen", "opencode-zen/muse-spark-1.2") == "codex_responses"
        # Grok models route via /v1/responses on both Zen and Go
        # (Zen/Go endpoint tables).
        assert opencode_model_api_mode("opencode-go", "grok-4.5") == "codex_responses"
        assert opencode_model_api_mode("opencode-go", "opencode-go/grok-4.5") == "codex_responses"
        assert opencode_model_api_mode("opencode-zen", "grok-4.6") == "codex_responses"
        assert opencode_model_api_mode("opencode-zen", "grok-4.5") == "codex_responses"
        assert opencode_model_api_mode("opencode-zen", "grok-build-0.1") == "codex_responses"
        # Ox Alpha (x-preview-f-free) on Zen is OpenAI-compatible
        # chat/completions per the Zen endpoint table.
        assert opencode_model_api_mode("opencode-zen", "x-preview-f-free") == "chat_completions"
        assert opencode_model_api_mode("opencode-zen", "opencode-zen/x-preview-f-free") == "chat_completions"
        # Other free-tier Zen models are chat/completions too.
        assert opencode_model_api_mode("opencode-zen", "mimo-v2.5-free") == "chat_completions"
        assert opencode_model_api_mode("opencode-zen", "nemotron-3.5-lightning-free") == "chat_completions"
        # Hy3 on Go is chat/completions (Go endpoint table).
        assert opencode_model_api_mode("opencode-go", "hy3") == "chat_completions"
        # New Go models keep their family routing: GLM chat/completions,
        # Qwen anthropic_messages.
        assert opencode_model_api_mode("opencode-go", "glm-5.3") == "chat_completions"
        assert opencode_model_api_mode("opencode-go", "glm-5.3-flash") == "chat_completions"
        assert opencode_model_api_mode("opencode-go", "qwen3.8-max") == "anthropic_messages"
        # Custom opencode-go-* providers route according to opencode-go rules
        # (family-prefix providers, issue #85589).
        assert opencode_model_api_mode("opencode-go-bridge", "grok-4.5") == "codex_responses"
        assert opencode_model_api_mode("opencode-go-bridge", "opencode-go-bridge/grok-4.5") == "codex_responses"
        assert opencode_model_api_mode("opencode-go-bridge", "minimax-m2.5") == "anthropic_messages"
        assert opencode_model_api_mode("opencode-go-bridge", "deepseek-v4-flash") == "chat_completions"
        # Case-insensitive provider ID handling (e.g. OpenCode-Go-Bridge).
        assert opencode_model_api_mode("OpenCode-Go-Bridge", "grok-4.5") == "codex_responses"
        assert opencode_model_api_mode("OpenCode-Go-Bridge", "minimax-m2.5") == "anthropic_messages"
        # Custom opencode-zen-* providers route according to opencode-zen rules.
        assert opencode_model_api_mode("opencode-zen-custom", "claude-3-5-sonnet") == "anthropic_messages"
        assert opencode_model_api_mode("opencode-zen-custom", "gpt-5") == "codex_responses"
        assert opencode_model_api_mode("opencode-zen-custom", "grok-4.5") == "codex_responses"
        assert opencode_model_api_mode("OpenCode-Zen-Custom", "claude-3-7-sonnet") == "anthropic_messages"


class TestNormalizeOpencodeBaseUrl:
    """Symmetric /v1 normalization for OpenCode Zen / Go base URLs.

    Regression for the 'only minimax works on opencode-go' bug: switching into
    an anthropic-routed model strips /v1 from the base URL and that stripped
    URL gets persisted to model.base_url; every later chat_completions model
    (glm, deepseek, kimi) then POSTed to https://opencode.ai/zen/go/chat/completions
    — a 404 (the marketing site).  The normalizer must heal a stripped URL.
    """

    def test_strips_v1_for_anthropic_messages(self):
        from hermes_cli.models import normalize_opencode_base_url
        assert normalize_opencode_base_url(
            "opencode-go", "anthropic_messages", "https://opencode.ai/zen/go/v1"
        ) == "https://opencode.ai/zen/go"
        assert normalize_opencode_base_url(
            "opencode-zen", "anthropic_messages", "https://opencode.ai/zen/v1/"
        ) == "https://opencode.ai/zen"


    def test_non_opencode_provider_untouched(self):
        from hermes_cli.models import normalize_opencode_base_url
        assert normalize_opencode_base_url(
            "openrouter", "chat_completions", "https://openrouter.ai/api"
        ) == "https://openrouter.ai/api"


class TestNormalizeOpencodeBaseUrlFamilyPath:
    """A carried-over base_url is healed on the FAMILY path segment (``/zen`` vs ``/zen/go``), not
    just ``/v1`` (#112600): ``model.base_url`` pinned to the Zen relay survived a switch to
    ``opencode-go`` and every request 401'd ("Model mimo-v2.5 is not supported")."""

    @pytest.mark.parametrize("provider, api_mode, url, expected", [
        ("opencode-go", "chat_completions", "https://opencode.ai/zen/v1", "https://opencode.ai/zen/go/v1"),
        ("opencode-zen", "chat_completions", "https://opencode.ai/zen/go/v1", "https://opencode.ai/zen/v1"),
        # Family healed first, then the /v1 strip for the Anthropic SDK — both apply.
        ("opencode-go", "anthropic_messages", "https://opencode.ai/zen/v1", "https://opencode.ai/zen/go"),
        ("opencode-zen", "anthropic_messages", "https://opencode.ai/zen/go", "https://opencode.ai/zen"),
        # A self-hosted OPENCODE_*_BASE_URL proxy has no family path to rewrite.
        ("opencode-go", "chat_completions", "https://gateway.internal.example/zen/v1", "https://gateway.internal.example/zen/v1"),
        # Non-/zen paths on the real host keep the pre-existing /v1 behaviour.
        ("opencode-go", "chat_completions", "https://opencode.ai/api", "https://opencode.ai/api/v1"),
        # A custom provider merely NAMED after a family declared its relay explicitly: no family
        # heal, but still the family's /v1 handling.
        ("opencode-zen-bridge", "chat_completions", "https://opencode.ai/zen/go/v1", "https://opencode.ai/zen/go/v1"),
        ("opencode-go-bridge", "chat_completions", "https://opencode.ai/zen/go", "https://opencode.ai/zen/go/v1"),
        # The host check is on the hostname, so a port does not defeat the heal; query survives.
        ("opencode-go", "chat_completions", "https://opencode.ai:443/zen/v1", "https://opencode.ai:443/zen/go/v1"),
        ("opencode-go", "anthropic_messages", "https://opencode.ai/zen/v1?x=1", "https://opencode.ai/zen/go?x=1"),
    ])
    def test_family_path_follows_the_resolved_provider(self, provider, api_mode, url, expected):
        from hermes_cli.models import normalize_opencode_base_url
        assert normalize_opencode_base_url(provider, api_mode, url) == expected


class TestAzureFoundryModelApiMode:
    """Azure Foundry deploys GPT-5.x / codex / o-series as Responses-API-only.

    Azure returns ``400 "The requested operation is unsupported."`` when
    /chat/completions is called against these deployments.  Verified in the
    wild by a user debug bundle on 2026-04-26: gpt-5.3-codex failed with
    that exact payload while gpt-4o-pure worked on the same endpoint.
    """

    def test_gpt5_family_uses_responses(self):
        assert azure_foundry_model_api_mode("gpt-5") == "codex_responses"
        assert azure_foundry_model_api_mode("gpt-5.3") == "codex_responses"
        assert azure_foundry_model_api_mode("gpt-5.4") == "codex_responses"
        assert azure_foundry_model_api_mode("gpt-5-codex") == "codex_responses"
        assert azure_foundry_model_api_mode("gpt-5.3-codex") == "codex_responses"
        # gpt-5-mini exceptions are Copilot-specific; Azure deploys the whole
        # gpt-5 family on Responses API uniformly.
        assert azure_foundry_model_api_mode("gpt-5-mini") == "codex_responses"

    def test_codex_family_uses_responses(self):
        assert azure_foundry_model_api_mode("codex") == "codex_responses"
        assert azure_foundry_model_api_mode("codex-mini") == "codex_responses"


    def test_gpt4_family_returns_none(self):
        """GPT-4, GPT-4o, etc. speak chat completions on Azure."""
        assert azure_foundry_model_api_mode("gpt-4") is None
        assert azure_foundry_model_api_mode("gpt-4o") is None
        assert azure_foundry_model_api_mode("gpt-4o-pure") is None
        assert azure_foundry_model_api_mode("gpt-4o-mini") is None
        assert azure_foundry_model_api_mode("gpt-4-turbo") is None
        assert azure_foundry_model_api_mode("gpt-4.1") is None
        assert azure_foundry_model_api_mode("gpt-3.5-turbo") is None


# -- validate — format checks -----------------------------------------------

class TestValidateFormatChecks:
    def test_empty_model_rejected(self):
        result = _validate("")
        assert result["accepted"] is False
        assert "empty" in result["message"]


    def test_no_slash_model_still_probes_api(self):
        result = _validate("gpt-5.4", api_models=["gpt-5.4", "gpt-5.4-pro"])
        assert result["accepted"] is True
        assert result["persist"] is True

    def test_no_slash_model_rejected_if_not_in_api(self):
        result = _validate("gpt-5.4", api_models=["openai/gpt-5.4"])
        assert result["accepted"] is False
        assert result["persist"] is False
        assert "not found" in result["message"]


# -- validate — API found ----------------------------------------------------


# -- validate — API not found ------------------------------------------------

class TestValidateApiNotFound:

    def test_not_listed_rejects_with_suggestions(self):
        """A near-miss on an aggregator listing is rejected with the listed sibling offered, never
        silently swapped in (the user asked for 4.5, not 4.6)."""
        result = _validate("anthropic/claude-opus-4.5")
        assert result["accepted"] is False
        assert "corrected_model" not in result
        assert "anthropic/claude-opus-4.6" in result["message"]


# -- validate — API unreachable — soft-accept via catalog or warning --------

class TestValidateApiFallback:
    """When /models is unreachable, the validator must accept the model (with
    a warning) rather than reject it outright — otherwise provider switches
    fail in the gateway for any provider whose /models endpoint is down or
    doesn't exist (e.g. opencode-go returns 404 HTML).

    Two paths:
      1. Provider has a curated catalog (``_PROVIDER_MODELS`` / live fetch):
         validate against it (recognized=True for known models,
         recognized=False with 'Note:' for unknown).
      2. Provider has no catalog: accept with a generic 'Note:' warning.

    In both cases ``accepted`` and ``persist`` must be True so the gateway can
    write the ``_session_model_overrides`` entry.
    """






    def test_fetch_lmstudio_models_filters_embedding_type(self):
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = False
        mock_resp.read.return_value = (
            b'{"models":['
            b'{"key":"publisher/chat-model","id":"publisher/chat-model","type":"llm"},'
            b'{"key":"publisher/embed-model","id":"publisher/embed-model","type":"embedding"}'
            b']}'
        )

        with patch("hermes_cli.models._urlopen_model_catalog_request", return_value=mock_resp):
            models = fetch_lmstudio_models(base_url="http://localhost:1234/v1")

        assert models == ["publisher/chat-model"]



    def test_validate_lmstudio_distinguishes_auth_failure(self):
        import urllib.error

        http_error = urllib.error.HTTPError(
            url="http://localhost:1234/api/v1/models",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=None,
        )

        with patch("hermes_cli.models._urlopen_model_catalog_request", side_effect=http_error):
            result = validate_requested_model(
                "publisher/chat-model",
                "lmstudio",
                base_url="http://localhost:1234/v1",
            )

        assert result["accepted"] is False
        assert "401" in result["message"]
        assert "LM_API_KEY" in result["message"]



# -- validate — the requested id is never rewritten -----------------------------

class TestRequestedIdIsNeverRewritten:
    """A selected id that is merely CLOSE to a catalog entry is the user's choice (a newer release,
    a dated snapshot, a qualifier), never a typo to "fix": the verdict may warn or reject, but no
    branch may return a different model under the user's label."""

    @pytest.mark.parametrize("requested, listing", [
        ("deepseek-v4.1-flash", ["deepseek-v4-flash-0731", "deepseek-v4-flash"]),   # custom endpoint (#mao)
        ("gemini-3.8-flash", ["gemini-3.6-flash", "gemini-3.6-pro"]),               # version bump (#101975)
        ("gpt5.3-codex", ["gpt-5.4", "gpt-5.3-codex"]),                             # genuine typo
    ])
    def test_live_listing_near_miss_keeps_requested_id(self, requested, listing):
        for provider, base_url in (("custom:hyper", "http://127.0.0.1:1/v1"), ("openrouter", None)):
            result = _validate(requested, provider, api_models=listing, base_url=base_url)
            assert "corrected_model" not in result
            assert result["recognized"] is False
            assert "Similar models" in (result["message"] or "") or listing[-1] in (result["message"] or "")

    def test_static_catalog_near_miss_keeps_requested_id(self):
        codex_models = ["gpt-5.4-mini", "gpt-5.4", "gpt-5.3-codex"]
        with patch("hermes_cli.models.provider_model_ids", return_value=codex_models):
            result = validate_requested_model("gpt5.3-codex", "openai-codex")
        assert "corrected_model" not in result
        assert result["recognized"] is False
        assert "gpt-5.3-codex" in result["message"]  # offered as a suggestion, not applied


class TestValidateCodex900kVariants:
    """`-900k` is a Hermes picker convention: valid variants come from the
    catalog; ineligible aliases are hard-rejected BEFORE the hidden-slug
    soft-accept (#92797 review)."""

    _CATALOG = ["gpt-5.6-sol", "gpt-5.6-sol-900k", "gpt-5.5", "gpt-5.4-mini"]

    def test_catalog_listed_variant_accepted(self):
        with patch("hermes_cli.models.provider_model_ids", return_value=self._CATALOG):
            result = validate_requested_model("gpt-5.6-sol-900k", "openai-codex")
        assert result["accepted"] is True
        assert result["recognized"] is True

    @pytest.mark.parametrize("alias", ["gpt-5.5-900k", "gpt-5.4-mini-900k", "gpt-5.6-sol-pro-900k"])
    def test_ineligible_900k_alias_rejected_not_soft_accepted(self, alias):
        with patch("hermes_cli.models.provider_model_ids", return_value=self._CATALOG):
            result = validate_requested_model(alias, "openai-codex")
        assert result["accepted"] is False
        assert result["persist"] is False
        assert "272K" in result["message"]

    def test_valid_variant_missing_from_catalog_still_accepted(self):
        """A verified variant not yet in the (possibly stale) catalog is
        accepted via the eligibility predicate, not the soft-accept."""
        with patch("hermes_cli.models.provider_model_ids", return_value=["gpt-5.6-sol"]):
            result = validate_requested_model("gpt-5.6-sol-900k", "openai-codex")
        assert result["accepted"] is True


# -- probe_api_models — Cloudflare UA mitigation --------------------------------

class TestProbeApiModelsUserAgent:
    """Probing custom /v1/models must send a Hermes User-Agent.

    Some custom Claude proxies (e.g. ``packyapi.com``) sit behind Cloudflare with
    Browser Integrity Check enabled. The default ``Python-urllib/3.x`` signature
    is rejected with HTTP 403 ``error code: 1010``, which ``probe_api_models``
    swallowed into ``{"models": None}``, surfacing to users as a misleading
    "Could not reach the ... API to validate ..." error — even though the
    endpoint is reachable and the listing exists.
    """

    def _make_mock_response(self, body: bytes):
        from unittest.mock import MagicMock
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read = MagicMock(return_value=body)
        return mock_resp

    def test_probe_sends_hermes_user_agent(self):
        from unittest.mock import patch

        body = b'{"data":[{"id":"claude-opus-4.7"}]}'
        with patch(
            "hermes_cli.models._urlopen_model_catalog_request",
            return_value=self._make_mock_response(body),
        ) as mock_urlopen:
            result = probe_api_models("sk-test", "https://example.com/v1")

        assert result["models"] == ["claude-opus-4.7"]
        # The urlopen call receives a Request object as its first positional arg
        req = mock_urlopen.call_args[0][0]
        ua = req.get_header("User-agent")  # urllib title-cases header names
        assert ua, "probe_api_models must send a User-Agent header"
        assert ua.startswith("hermes-cli/"), (
            f"User-Agent must advertise hermes-cli, got {ua!r}"
        )
        # Must not fall back to urllib's default — that's what Cloudflare 1010 blocks.
        assert not ua.startswith("Python-urllib")

    def test_probe_user_agent_sent_without_api_key(self):
        """UA must be present even for endpoints that don't need auth."""
        from unittest.mock import patch

        body = b'{"data":[]}'
        with patch(
            "hermes_cli.models._urlopen_model_catalog_request",
            return_value=self._make_mock_response(body),
        ) as mock_urlopen:
            probe_api_models(None, "https://example.com/v1")

        req = mock_urlopen.call_args[0][0]
        ua = req.get_header("User-agent")
        assert ua and ua.startswith("hermes-cli/")
        # No Authorization was set, but UA must still be present.
        assert req.get_header("Authorization") is None




# -- validate — OpenRouter routing-variant suffixes (:nitro / :floor / ...) ----

class TestValidateOpenRouterVariantSuffixes:
    """OpenRouter's `:nitro`, `:floor`, `:exacto`, `:online` are request-time
    routing modifiers, not catalog models — /models lists only the base id.
    Validation must accept `base:variant` when `base` is listed, preserve the
    suffixed id (no auto-correct stripping the routing opt-in), and still
    reject variants on unknown bases and unknown suffixes."""

    _LISTING = [
        "~x-ai/grok-latest",
        "x-ai/grok-4.6",
        "deepseek/deepseek-v4-flash",
        "thinkingmachines/inkling:free",
    ]

    def _validate(self, model):
        return _validate(model, "openrouter", api_models=self._LISTING)

    @pytest.mark.parametrize("suffix", ["nitro", "floor", "exacto", "online"])
    def test_variant_on_listed_base_accepted_unmodified(self, suffix):
        result = self._validate(f"~x-ai/grok-latest:{suffix}")
        assert result["accepted"] is True
        assert result["recognized"] is True
        assert result.get("corrected_model") is None
        assert result["message"] is None

    def test_variant_not_fuzzy_corrected_to_base(self):
        """The old failure mode: get_close_matches would 'fix' model:nitro
        to the bare base id and silently drop the routing behavior."""
        result = self._validate("x-ai/grok-4.6:nitro")
        assert result["accepted"] is True
        assert result.get("corrected_model") is None

    def test_variant_on_unknown_base_rejected(self):
        result = self._validate("x-ai/notreal-model:nitro")
        assert result["accepted"] is False

    def test_unknown_suffix_keeps_old_behavior(self):
        result = self._validate("x-ai/grok-4.6:bogus")
        assert result["accepted"] is False

    def test_free_sku_still_direct_matched(self):
        """`:free` SKUs ARE catalog entries; direct membership handles them."""
        result = self._validate("thinkingmachines/inkling:free")
        assert result["accepted"] is True
        assert result.get("corrected_model") is None

    def test_variant_uppercase_suffix_accepted(self):
        result = self._validate("x-ai/grok-4.6:NITRO")
        assert result["accepted"] is True
        assert result.get("corrected_model") is None

    def test_non_openrouter_provider_unaffected(self):
        """The variant carve-out is OpenRouter-only; other providers keep
        their existing behavior for colon-suffixed names."""
        result = _validate(
            "x-ai/grok-4.6:nitro",
            "groq",
            api_models=["x-ai/grok-4.6"],
        )
        assert result.get("corrected_model") != "x-ai/grok-4.6:nitro"

    def test_static_catalog_fallback_accepts_variant(self):
        """Gateway path: /models unreachable → static catalog validates the
        base id and preserves the suffix."""
        with patch("hermes_cli.models.fetch_api_models", return_value=None), \
             patch(
                 "hermes_cli.models.provider_model_ids",
                 return_value=["x-ai/grok-4.6", "anthropic/claude-opus-4.6"],
             ):
            result = validate_requested_model(
                "x-ai/grok-4.6:floor",
                "openrouter",
                base_url="https://openrouter.ai/api/v1",
            )
        assert result["accepted"] is True
        assert result["recognized"] is True
        assert result.get("corrected_model") is None


class TestValidateRequestedModelNousPortalRecommendations:
    """Regression tests for issue #71312: the Nous Telegram picker (and any
    other messaging-platform /model validation, since they all share
    validate_requested_model()) rejected models that are live Nous Portal
    recommendations (/api/nous/recommended-models) but not yet in the
    hardcoded curated catalog -- even though `hermes chat` already accepts
    these via union_with_portal_free/paid_recommendations() at model-list
    build time. The per-message validation path now checks the same Portal
    feed as a fallback tier before rejecting, so Telegram/CLI agree.
    """

    PORTAL_PAYLOAD = {
        "freeRecommendedModels": [
            {"modelName": "inclusionai/ling-3.0-flash:free"},
        ],
        "paidRecommendedModels": [
            {"modelName": "inclusionai/ling-3.0-pro"},
        ],
    }

    def _validate_nous(self, model, api_models=None, portal_payload=None, portal_raises=False):
        api_models = api_models if api_models is not None else ["inclusionai/ling-2.6-flash"]
        probe_payload = {
            "models": api_models,
            "probed_url": "https://portal.nousresearch.com/v1/models",
            "resolved_base_url": "https://portal.nousresearch.com/v1",
            "suggested_base_url": None,
            "used_fallback": False,
        }

        def _fetch_portal(*a, **kw):
            if portal_raises:
                raise RuntimeError("portal unreachable")
            return portal_payload if portal_payload is not None else self.PORTAL_PAYLOAD

        with patch("hermes_cli.models.fetch_api_models", return_value=api_models), \
             patch("hermes_cli.models.probe_api_models", return_value=probe_payload), \
             patch("hermes_cli.models.fetch_nous_recommended_models", side_effect=_fetch_portal), \
             patch("hermes_cli.models._resolve_nous_portal_url", return_value="https://portal.nousresearch.com"), \
             patch("hermes_cli.models._model_in_provider_catalog", return_value=False):
            return validate_requested_model(model, "nous")

    def test_free_portal_recommendation_accepted(self):
        """The exact scenario from #71312: a free-tier Portal recommendation
        missing from the curated catalog and the live /v1/models listing
        must be accepted, not rejected."""
        result = self._validate_nous("inclusionai/ling-3.0-flash:free")
        assert result["accepted"] is True
        assert result["persist"] is True
        assert "Portal recommendation" in (result["message"] or "")

    def test_paid_portal_recommendation_accepted(self):
        result = self._validate_nous("inclusionai/ling-3.0-pro")
        assert result["accepted"] is True

    def test_model_absent_from_portal_and_catalog_still_rejected(self):
        """A model that's genuinely nowhere (not live, not curated, not a
        Portal recommendation) must still be rejected -- this fallback
        tier must not make validation permissive for everything."""
        result = self._validate_nous("totally-made-up-model-xyz")
        assert result["accepted"] is False
        assert result["recognized"] is False

    def test_portal_fetch_failure_falls_through_to_rejection_not_crash(self):
        """A network/parse failure fetching the Portal feed must not crash
        validation -- it degrades to the existing rejection path."""
        result = self._validate_nous(
            "inclusionai/ling-3.0-flash:free", portal_raises=True
        )
        assert result["accepted"] is False  # fails closed, doesn't crash

    def test_non_string_model_name_entries_ignored(self):
        """Malformed Portal entries (non-string / empty modelName) must be
        skipped via _extract_model_name -- never stringified into garbage
        matches (e.g. an int modelName 5 must not accept a model named "5")."""
        payload = {
            "freeRecommendedModels": [
                {"modelName": 5},
                {"modelName": ""},
                {"modelName": None},
                "not-a-dict",
                {"modelName": "inclusionai/ling-3.0-flash:free"},
            ],
            "paidRecommendedModels": [],
        }
        assert self._validate_nous("5", portal_payload=payload)["accepted"] is False
        result = self._validate_nous(
            "inclusionai/ling-3.0-flash:free", portal_payload=payload
        )
        assert result["accepted"] is True

    def test_non_nous_provider_does_not_consult_portal_feed(self):
        """This fallback tier is Nous-specific; a non-Nous provider must
        not have its rejection changed by (or trigger a call to) the Nous
        Portal feed."""
        probe_payload = {
            "models": ["some/other-model"],
            "probed_url": "https://api.example.com/v1/models",
            "resolved_base_url": "https://api.example.com/v1",
            "suggested_base_url": None,
            "used_fallback": False,
        }
        with patch("hermes_cli.models.fetch_api_models", return_value=["some/other-model"]), \
             patch("hermes_cli.models.probe_api_models", return_value=probe_payload), \
             patch("hermes_cli.models.fetch_nous_recommended_models") as mock_portal, \
             patch("hermes_cli.models._model_in_provider_catalog", return_value=False):
            result = validate_requested_model("inclusionai/ling-3.0-flash:free", "openrouter")
        mock_portal.assert_not_called()
        assert result["accepted"] is False

    def test_curated_catalog_hit_short_circuits_before_portal_check(self):
        """When the curated-catalog fallback already accepts the model, the
        Portal feed should not need to be consulted at all (cheaper, and
        avoids an unnecessary network call on the common path)."""
        api_models = ["inclusionai/ling-2.6-flash"]
        probe_payload = {
            "models": api_models, "probed_url": "x", "resolved_base_url": "x",
            "suggested_base_url": None, "used_fallback": False,
        }
        with patch("hermes_cli.models.fetch_api_models", return_value=api_models), \
             patch("hermes_cli.models.probe_api_models", return_value=probe_payload), \
             patch("hermes_cli.models._model_in_provider_catalog", return_value=True), \
             patch("hermes_cli.models.fetch_nous_recommended_models") as mock_portal:
            result = validate_requested_model("inclusionai/ling-2.6-flash", "nous")
        mock_portal.assert_not_called()
        assert result["accepted"] is True


# -- validate — custom endpoint fallback when /models is unreachable (#12220) --

class TestValidateCustomUnreachableFallback:
    """A custom proxy without GET /models must not brick `/model` switches (#12220)."""

    def _validate(self, model, provider, models, **kw):
        probe = {"models": models, "probed_url": "http://localhost:8000/v1/models",
                 "resolved_base_url": "http://localhost:8000/v1", "suggested_base_url": None, "used_fallback": False}
        with patch("hermes_cli.models.probe_api_models", return_value=probe):
            return validate_requested_model(model, provider, api_key="k", base_url="http://localhost:8000/v1", **kw)

    @pytest.mark.parametrize("provider", ["custom", "custom:myproxy"])
    @pytest.mark.parametrize("api_mode", ["chat_completions", "anthropic_messages"])
    def test_unreachable_catalog_persists_unverified_for_chat_modes(self, provider, api_mode):
        result = self._validate("my-proxy-model", provider, models=None, api_mode=api_mode)
        assert (result["accepted"], result["persist"], result["recognized"]) == (True, True, False)
        assert "accepted without verification" in result["message"]

    @pytest.mark.parametrize("api_mode", [None, "codex_responses"])
    def test_unreachable_catalog_still_rejects_other_api_modes(self, api_mode):
        result = self._validate("my-proxy-model", "custom", models=None, api_mode=api_mode)
        assert result["accepted"] is False
        assert "was not saved" in result["message"]
        # A reachable catalog keeps authoritative validation regardless of mode.
        assert self._validate("my-model", "custom", models=["my-model"], api_mode="chat_completions")["recognized"] is True

    def test_anthropic_messages_reachable_listing_without_slug_is_not_called_unimplemented(self):
        """A listing that answered 200 but lacks the slug must not be described as a proxy that
        'does not implement GET /v1/models'; it names the alias candidates instead (#111436)."""
        result = self._validate("kimi-k3", "kimi-coding", models=["k3", "k3-turbo"], api_mode="anthropic_messages")
        assert (result["accepted"], result["persist"], result["recognized"]) == (True, True, False)
        assert "do not implement" not in result["message"]
        assert "not named in this endpoint's model listing" in result["message"]
        assert "`k3`" in result["message"]
        # Case-only spelling differences are a match, not a warning.
        assert self._validate("K3", "kimi-coding", models=["k3"], api_mode="anthropic_messages")["recognized"] is True
        # The unreachable-listing wording is unchanged.
        unreachable = self._validate("kimi-k3", "kimi-coding", models=None, api_mode="anthropic_messages")
        assert "do not implement GET /v1/models" in unreachable["message"]


# -- validate — profile-owned catalog is authoritative (#116667) ----------------

class TestProfileCatalogAuthoritative:
    """A profile that points ``models_url`` at its own catalog endpoint (relay re-shape,
    #101705) owns validation: the generic ``{base_url}/models`` listing may expose a
    different product line that this deployment cannot serve, so a miss in the
    profile-owned catalog must not be turned into an acceptance by that generic
    listing (#116667). Unavailable/empty profile catalogs keep the generic fallback."""

    @pytest.fixture
    def relay_profile(self, monkeypatch):
        import providers
        from providers.base import ProviderProfile

        profile = ProviderProfile(
            name="relay-owned-catalog", auth_type="api_key",
            env_vars=("RELAY_TEST_KEY",), base_url="https://relay.example.invalid/v1",
            models_url="https://relay.example.invalid/catalog",
            fallback_models=("plan/model-1",),
        )
        monkeypatch.setitem(providers._REGISTRY, profile.name, profile)
        return profile

    def test_model_only_in_generic_listing_is_rejected(self, relay_profile):
        """Profile catalog: ["plan/model-1"]; generic /models: ["other-vendor/model-a"];
        requested "other-vendor/model-a" — the generic hit must not accept it."""
        result = _validate(
            "other-vendor/model-a", "relay-owned-catalog", api_models=["other-vendor/model-a"])
        assert result["accepted"] is False
        assert result["persist"] is False
        assert "plan/model-1" in result["message"]

    def test_unavailable_profile_catalog_keeps_generic_fallback(self, relay_profile):
        """The profile's own catalog unreachable/empty (e.g. no credentials yet):
        the generic listing stays the validator, as before (#116667's carve-out)."""
        with patch("hermes_cli.models.fetch_api_models", return_value=["other-vendor/model-a"]), \
             patch("hermes_cli.models.provider_model_ids", return_value=[]):
            result = validate_requested_model(
                "other-vendor/model-a", "relay-owned-catalog", api_key="k")
        assert result["accepted"] is True
        assert result["recognized"] is True

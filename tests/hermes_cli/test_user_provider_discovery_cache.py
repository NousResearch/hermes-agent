"""Disk-cached live-model discovery for user-defined ``providers:`` endpoints.

A named user provider (e.g. a local OpenAI-compatible proxy) exposes its full
catalog on ``/v1/models``. The picker probes it live, but before the cache
layer existed the result was thrown away: GUI picker opens (which pass
``probe_custom_providers=False`` so offline endpoints can't block the dialog)
fell back to the single configured ``default_model`` — the dashboard showed
"1 models" for an endpoint that serves a dozen.

``cached_fetch_api_models`` persists probed catalogs in
``provider_models_cache.json`` (same file / TTL / refresh path as the
built-in providers) so cache-only picker opens still render the full list.
"""

import pytest

from hermes_cli.model_switch import list_authenticated_providers


_ENDPOINT = "http://127.0.0.1:55990/v1"
_LIVE_MODELS = [
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v4-pro",
    "MiniMaxAI/MiniMax-M3",
    "XiaomiMiMo/MiMo-V2.5-Pro",
]


def _user_providers() -> dict:
    return {
        "commandcode": {
            "base_url": _ENDPOINT,
            "api_key": "user_test_key",
            "default_model": "deepseek/deepseek-v4-flash",
        }
    }


def _commandcode_row(providers: list[dict]) -> dict | None:
    return next(
        (p for p in providers if p.get("is_user_defined") and p["slug"] == "commandcode"),
        None,
    )


class TestCachedFetchApiModels:
    def test_probe_populates_cache_then_cache_only_read_serves_it(self, monkeypatch):
        from hermes_cli.models import cached_fetch_api_models

        monkeypatch.setattr(
            "hermes_cli.models.fetch_api_models",
            lambda *a, **k: list(_LIVE_MODELS),
        )
        assert cached_fetch_api_models("key-1", _ENDPOINT, probe=True) == _LIVE_MODELS

        # Cache-only read must never touch the network.
        monkeypatch.setattr(
            "hermes_cli.models.fetch_api_models",
            lambda *a, **k: pytest.fail("network fetch in cache-only mode"),
        )
        assert cached_fetch_api_models("key-1", _ENDPOINT, probe=False) == _LIVE_MODELS

    def test_cache_only_read_without_prior_probe_returns_empty(self, monkeypatch):
        from hermes_cli.models import cached_fetch_api_models

        monkeypatch.setattr(
            "hermes_cli.models.fetch_api_models",
            lambda *a, **k: pytest.fail("network fetch in cache-only mode"),
        )
        assert cached_fetch_api_models("key-1", _ENDPOINT, probe=False) == []

    def test_api_key_change_invalidates_cached_entry(self, monkeypatch):
        from hermes_cli.models import cached_fetch_api_models

        monkeypatch.setattr(
            "hermes_cli.models.fetch_api_models",
            lambda *a, **k: list(_LIVE_MODELS),
        )
        cached_fetch_api_models("key-1", _ENDPOINT, probe=True)

        # Different credential fingerprint → the old entry must not leak.
        monkeypatch.setattr(
            "hermes_cli.models.fetch_api_models", lambda *a, **k: None
        )
        assert cached_fetch_api_models("key-2", _ENDPOINT, probe=False) == []

    def test_failed_probe_falls_back_to_stale_entry(self, monkeypatch):
        from hermes_cli.models import cached_fetch_api_models

        monkeypatch.setattr(
            "hermes_cli.models.fetch_api_models",
            lambda *a, **k: list(_LIVE_MODELS),
        )
        cached_fetch_api_models("key-1", _ENDPOINT, probe=True)

        monkeypatch.setattr(
            "hermes_cli.models.fetch_api_models", lambda *a, **k: None
        )
        # Even with a zero TTL (entry counts as stale), a failed live fetch
        # serves the stale catalog — stale beats no data on a flaky network.
        assert (
            cached_fetch_api_models("key-1", _ENDPOINT, probe=True, ttl_seconds=0)
            == _LIVE_MODELS
        )


class TestPickerUsesCachedCatalog:
    def test_probe_open_then_cache_only_open_keeps_full_catalog(self, monkeypatch):
        """The dashboard flow: a probing open (refresh) discovers the live
        catalog; subsequent cache-only opens must keep showing it instead of
        collapsing back to the single default_model."""
        monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
        monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
        monkeypatch.setattr(
            "hermes_cli.models.fetch_api_models",
            lambda *a, **k: list(_LIVE_MODELS),
        )

        probed = list_authenticated_providers(
            current_provider="commandcode",
            user_providers=_user_providers(),
            custom_providers=[],
            probe_custom_providers=True,
        )
        row = _commandcode_row(probed)
        assert row is not None
        assert row["total_models"] == len(_LIVE_MODELS)

        monkeypatch.setattr(
            "hermes_cli.models.fetch_api_models",
            lambda *a, **k: pytest.fail("network fetch in cache-only picker open"),
        )
        cached = list_authenticated_providers(
            current_provider="commandcode",
            user_providers=_user_providers(),
            custom_providers=[],
            probe_custom_providers=False,
        )
        row = _commandcode_row(cached)
        assert row is not None
        assert row["models"] == _LIVE_MODELS
        assert row["total_models"] == len(_LIVE_MODELS)

    def test_cache_only_open_without_cache_shows_configured_default(self, monkeypatch):
        """Cold start with no cache: cache-only opens degrade to the
        configured default_model and never touch the network."""
        monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
        monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
        monkeypatch.setattr(
            "hermes_cli.models.fetch_api_models",
            lambda *a, **k: pytest.fail("network fetch in cache-only picker open"),
        )

        providers = list_authenticated_providers(
            current_provider="commandcode",
            user_providers=_user_providers(),
            custom_providers=[],
            probe_custom_providers=False,
        )
        row = _commandcode_row(providers)
        assert row is not None
        assert row["models"] == ["deepseek/deepseek-v4-flash"]

    def test_explicit_models_without_api_key_skip_discovery(self, monkeypatch):
        """A keyless entry narrowing a public endpoint to an explicit
        ``models:`` subset keeps that subset — no live discovery, no cache."""
        monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
        monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})
        monkeypatch.setattr(
            "hermes_cli.models.fetch_api_models",
            lambda *a, **k: pytest.fail("discovery must be skipped"),
        )

        providers = list_authenticated_providers(
            current_provider="local-subset",
            user_providers={
                "local-subset": {
                    "base_url": "http://localhost:11434/v1",
                    "models": ["model-a", "model-b"],
                }
            },
            custom_providers=[],
            probe_custom_providers=True,
        )
        row = next(
            (p for p in providers if p.get("is_user_defined") and p["slug"] == "local-subset"),
            None,
        )
        assert row is not None
        assert row["models"] == ["model-a", "model-b"]

"""`model.base_url` relays must be probed instead of the vendor's canonical host (#121387)."""

from types import SimpleNamespace

import pytest

import hermes_cli.models as models


class _RecordingProfile:
    auth_type = "api_key"
    fallback_models = []

    def __init__(self):
        self.calls = []

    def fetch_models(self, api_key=None, base_url=None):
        self.calls.append((api_key, base_url))
        return ["relay-only-model"]


@pytest.mark.parametrize("provider", ["openai-codex", "copilot", "nous", "opencode-zen", "openrouter"])
def test_canonical_url_preserves_native_catalog_but_other_paths_stay_relays(monkeypatch, provider):
    from hermes_cli.config import atomic_config_write
    from hermes_constants import get_hermes_home
    from providers import get_provider_profile

    canonical = get_provider_profile(provider).base_url
    config_path = get_hermes_home() / "config.yaml"
    calls = []

    def native_catalog(slug, force_refresh):
        calls.append(slug)
        return ["native-catalog-model"]

    monkeypatch.setitem(models._PROVIDER_CATALOG_FETCHERS, provider, native_catalog)
    monkeypatch.setattr(models, "_relay_model_catalog", lambda *args: None)
    scheme, rest = canonical.split("://", 1)
    host, _, path = rest.partition("/")
    # Same route spelled differently: scheme/host case, explicit default port, trailing slash.
    shouted = f"{scheme.upper()}://{host.upper()}:443{'/' + path if path else ''}/"
    for base_url, expected in [(canonical, [provider]), (shouted, [provider]), (canonical + "/relay", [])]:
        calls.clear()
        atomic_config_write(config_path, {"model": {"provider": provider, "base_url": base_url}})
        catalog = models.provider_model_ids(provider, force_refresh=True)
        assert calls == expected
        assert ("native-catalog-model" in catalog) == bool(expected)


def test_relay_base_url_is_probed_for_configured_provider(monkeypatch):
    monkeypatch.setattr(
        models,
        "_get_model_config_dict",
        lambda: {
            "provider": "deepseek",
            "base_url": "http://127.0.0.1:9001/deepseek/v1",
        },
    )
    profile = _RecordingProfile()
    import providers

    monkeypatch.setattr(providers, "get_provider_profile", lambda name: profile)
    monkeypatch.setattr(models, "_api_key_credentials", lambda name: (None, None))
    monkeypatch.setattr(
        models,
        "_PROVIDER_CATALOG_FETCHERS",
        {
            "deepseek": lambda n, f: (_ for _ in ()).throw(
                AssertionError("canonical host touched")
            )
        },
    )

    assert models.provider_model_ids("deepseek", force_refresh=True) == [
        "relay-only-model"
    ]
    assert profile.calls == [(None, "http://127.0.0.1:9001/deepseek/v1")]


def test_base_url_for_a_different_provider_is_ignored(monkeypatch):
    monkeypatch.setattr(
        models,
        "_get_model_config_dict",
        lambda: {
            "provider": "deepseek",
            "base_url": "http://127.0.0.1:9001/deepseek/v1",
        },
    )
    assert models._configured_relay_base_url("openai") == ""


@pytest.mark.parametrize(
    "probe_result",
    [
        pytest.param("raise", id="relay-hangs-or-errors"),
        pytest.param([], id="relay-404-empty-catalog"),
    ],
)
def test_a_failed_relay_probe_never_falls_back_to_the_vendor_host(monkeypatch, probe_result):
    """Egress sentinel: a configured relay is TERMINAL for live catalog egress.

    A relay that 404s, hangs or returns nothing must degrade to the LOCAL curated list.
    Falling through to the canonical fetcher would send the provider credential to exactly
    the vendor host the user routed away from, recreating #121387 on the failure path.
    """
    monkeypatch.setattr(
        models,
        "_get_model_config_dict",
        lambda: {
            "provider": "deepseek",
            "base_url": "http://127.0.0.1:9001/deepseek/v1",
        },
    )

    def _probe(**kwargs):
        if probe_result == "raise":
            raise RuntimeError("relay down")
        return probe_result

    import providers

    monkeypatch.setattr(
        providers,
        "get_provider_profile",
        lambda name: SimpleNamespace(
            auth_type="api_key", fetch_models=_probe, fallback_models=[]
        ),
    )
    monkeypatch.setattr(models, "_api_key_credentials", lambda name: (None, None))

    # Sentinels on EVERY live-egress seam below the relay probe.
    def _egress(*a, **k):
        raise AssertionError("vendor host touched after a failed relay probe")

    monkeypatch.setattr(models, "_PROVIDER_CATALOG_FETCHERS", {"deepseek": _egress})
    monkeypatch.setattr(models, "_profile_live_catalog", _egress)
    monkeypatch.setattr(models, "_merge_with_models_dev", _egress)
    monkeypatch.setattr(models, "_PROVIDER_MODELS", {"deepseek": ["curated-local"]})

    # Degrades locally to the curated list, with no vendor egress at all.
    assert models.provider_model_ids("deepseek") == ["curated-local"]

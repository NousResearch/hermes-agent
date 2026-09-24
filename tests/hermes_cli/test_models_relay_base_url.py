"""`model.base_url` relays must be probed instead of the vendor's canonical host (#121387)."""

from types import SimpleNamespace

import hermes_cli.models as models


class _RecordingProfile:
    auth_type = "api_key"
    fallback_models = []

    def __init__(self):
        self.calls = []

    def fetch_models(self, api_key=None, base_url=None):
        self.calls.append((api_key, base_url))
        return ["relay-only-model"]


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


def test_failed_relay_probe_falls_through_to_canonical_fetchers(monkeypatch):
    monkeypatch.setattr(
        models,
        "_get_model_config_dict",
        lambda: {
            "provider": "deepseek",
            "base_url": "http://127.0.0.1:9001/deepseek/v1",
        },
    )

    def _boom(**kwargs):
        raise RuntimeError("relay down")

    import providers

    monkeypatch.setattr(
        providers,
        "get_provider_profile",
        lambda name: SimpleNamespace(
            auth_type="api_key", fetch_models=_boom, fallback_models=[]
        ),
    )
    monkeypatch.setattr(models, "_api_key_credentials", lambda name: (None, None))
    monkeypatch.setattr(
        models,
        "_PROVIDER_CATALOG_FETCHERS",
        {"deepseek": lambda n, f: ["canonical-model"]},
    )

    assert models.provider_model_ids("deepseek") == ["canonical-model"]

"""``::``-namespaced model ids keep a configured provider name (#148755)."""

import providers
from agent.model_metadata import _strip_provider_prefix
from providers.base import ProviderProfile


def test_double_colon_model_id_keeps_provider_name(monkeypatch):
    # TensorZero ids look like "tensorzero::function_name::variant"; with a provider named
    # "tensorzero" configured, stripping on the first colon left ":function_name::variant".
    monkeypatch.setattr(providers, "_REGISTRY", {})
    monkeypatch.setattr(providers, "_ALIASES", {})
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None)
    monkeypatch.setattr(providers, "_discovered", True)
    providers.register_provider(ProviderProfile(name="tensorzero"))

    assert _strip_provider_prefix("tensorzero::function_name::mechanical") == (
        "tensorzero::function_name::mechanical"
    )
    assert _strip_provider_prefix("tensorzero:my-model") == "my-model"

"""Provider SDK contract for ephemeral, row-bound runtime credentials."""

import pytest

from providers import base


def test_runtime_credential_preserves_pair_without_disclosing_secrets():
    credential_type = getattr(base, "PoolRuntimeCredential", None)
    assert credential_type is not None, "provider SDK must expose PoolRuntimeCredential"
    credential = credential_type("runtime-secret", "https://account.example/v1")
    assert credential.api_key == "runtime-secret"
    assert credential.base_url == "https://account.example/v1"
    assert "runtime-secret" not in repr(credential)
    assert credential_type("runtime-secret", None).base_url is None


@pytest.mark.parametrize("api_key,base_url", [(None, None), (123, None), ("key", 123), ("key", {})])
def test_runtime_credential_rejects_non_string_fields(api_key, base_url):
    credential_type = getattr(base, "PoolRuntimeCredential", None)
    assert credential_type is not None, "provider SDK must expose PoolRuntimeCredential"
    with pytest.raises(TypeError):
        credential_type(api_key, base_url)


def test_legacy_pool_profile_defaults_are_non_disruptive():
    profile = base.ProviderProfile(name="legacy")
    entry = object()
    assert profile.prepare_pool_entry(entry, "original-key") == "original-key"
    refresh = getattr(profile, "refresh_pool_runtime_credentials", None)
    assert callable(refresh), "SDK must offer a selected-row runtime refresh hook"
    assert refresh(entry, reason="unauthorized") is None
    assert profile.credential_pool_base_urls == ()


def test_pool_routing_hints_are_constructor_data_not_profile_mutation():
    import inspect

    parameters = inspect.signature(base.ProviderProfile).parameters
    assert "credential_pool_base_urls" in parameters, "pool routing hints must be constructor data"
    urls = ("https://gateway.example/v1", "https://account.example/v1")
    profile = base.ProviderProfile(name="routed", credential_pool_base_urls=urls)
    assert profile.credential_pool_base_urls == urls

"""ProviderProfile declaration contract for Phase 5.2."""

import pytest

import providers
from providers.base import ProviderProfile


def test_bundled_profiles_have_complete_declarations():
    for profile in providers.list_providers():
        assert profile.display_name, f"{profile.name}: missing display_name"
        assert profile.description, f"{profile.name}: missing description"
        assert profile.auth_type, f"{profile.name}: missing auth_type"
        assert profile.base_url_env_var not in profile.env_vars, (
            f"{profile.name}: endpoint variable leaked into credential env_vars"
        )
        if profile.name != "custom":
            assert profile.base_url or profile.base_url_env_var, f"{profile.name}: missing endpoint declaration"
        if profile.auth_type == "api_key" and profile.name != "custom":
            assert profile.env_vars, f"{profile.name}: api_key profile has no credential env_vars"


def test_api_key_profile_env_contract_matches_auth_projection():
    from hermes_cli.provider_auth import get_provider_config

    for profile in providers.list_providers():
        config = get_provider_config(profile.name)
        if config is None or profile.auth_type != "api_key" or config.auth_type != "api_key":
            continue
        assert tuple(profile.env_vars) == tuple(config.api_key_env_vars or ()), profile.name
        assert (profile.base_url_env_var or "") == (config.base_url_env_var or ""), profile.name


def test_url_suffix_does_not_define_env_var_role(monkeypatch):
    import hermes_cli.provider_auth as provider_auth
    import hermes_cli.providers as cli_providers

    profile = ProviderProfile(
        name="declaration-probe",
        display_name="Declaration Probe",
        description="Contract probe",
        env_vars=("CREDENTIAL_URL",),
        base_url="https://probe.invalid/v1",
        base_url_env_var="ENDPOINT_OVERRIDE",
    )

    monkeypatch.setattr(provider_auth, "get_provider_profile", lambda _name: profile)
    config = provider_auth.get_provider_config("declaration-probe")
    assert config is not None
    assert config.api_key_env_vars == ("CREDENTIAL_URL",)
    assert config.base_url_env_var == "ENDPOINT_OVERRIDE"

    monkeypatch.setattr(cli_providers, "_get_provider_profile", lambda _name: profile)
    resolved = cli_providers._profile_resolved_provider("declaration-probe")
    assert resolved is not None
    assert resolved.env_vars == ("CREDENTIAL_URL",)
    assert resolved.base_url_env_var == "ENDPOINT_OVERRIDE"


def test_endpoint_env_may_not_overlap_credential_env_vars():
    with pytest.raises(ValueError, match="base_url_env_var"):
        ProviderProfile(
            name="invalid-provider",
            env_vars=("SHARED_VAR",),
            base_url_env_var="SHARED_VAR",
        )

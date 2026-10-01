"""Tests for the unified provider catalog (hermes_cli.provider_catalog).

These are invariant tests, not snapshots: they assert the parity *contract*
between the canonical provider registry, what ``hermes model`` shows, and what the
catalog exposes, plus how each provider's ``auth_type`` maps to a desktop tab —
never a specific provider count or a frozen vendor list (both change over time).
"""

from hermes_cli.provider_catalog import (
    PROVIDER_PICKER_ORDER,
    provider_catalog,
    provider_catalog_by_slug,
    provider_slugs,
)



def test_catalog_matches_effective_registry_and_presentation_order():
    from providers import list_providers

    profiles = list_providers()
    profile_slugs = {profile.name for profile in profiles}
    slugs = provider_slugs()
    assert set(slugs) == profile_slugs

    policy = [slug for slug in PROVIDER_PICKER_ORDER if slug in profile_slugs]
    actual_policy = [slug for slug in slugs if slug in PROVIDER_PICKER_ORDER]
    assert actual_policy == policy


def test_catalogued_builtins_have_authoritative_profiles():
    """Former catalogue-only rows now carry their own complete ProviderProfile metadata."""
    from providers import get_provider_profile

    by = provider_catalog_by_slug()
    for slug in ("lmstudio", "openai-api", "tencent-tokenhub", "xai-oauth"):
        profile = get_provider_profile(slug)
        assert profile is not None, f"{slug} has no ProviderProfile"
        assert profile.display_name and profile.description
        assert by[slug].label == profile.display_name
        assert by[slug].description == profile.description

def test_copilot_surfaces_as_a_provider_with_its_own_token_var():
    """Regression for the reported bug: a GitHub Copilot login showed up under
    tools, never as a provider, because the shared GITHUB_TOKEN is tool-category.

    Copilot authenticates via the `copilot`/api_key path, so it belongs on the
    keys tab — but its PRIMARY credential var must be the provider-owned
    COPILOT_GITHUB_TOKEN, not the shared tool-category GITHUB_TOKEN. That is what
    lets the desktop render Copilot as its own provider card.
    """
    by = provider_catalog_by_slug()
    assert "copilot" in by
    d = by["copilot"]
    assert d.tab == "keys"
    assert d.api_key_env_vars, "Copilot must expose a credential env var"
    assert d.api_key_env_vars[0] == "COPILOT_GITHUB_TOKEN", (
        "Copilot's primary var must be the provider-owned token, not shared GITHUB_TOKEN"
    )

def test_api_key_providers_expose_a_credential_env_var():
    """Every keys-tab provider that authenticates via a pasted API key must
    surface at least one env var to write the key into (otherwise the GUI can't
    configure it).

    Exemptions: ``aws_sdk`` (bedrock — uses AWS_REGION/AWS_PROFILE) and the
    ``custom`` bring-your-own-endpoint pseudo-provider (configured inline via
    the ``local-endpoint`` flow).
    """
    exempt = {"custom"}
    for d in provider_catalog():
        if d.auth_type == "api_key" and d.slug not in exempt:
            assert d.api_key_env_vars, f"{d.slug} is api_key but exposes no env var"

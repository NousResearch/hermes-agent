"""Tests for the unified provider catalog (hermes_cli.provider_catalog).

These are invariant tests, not snapshots: they assert the parity *contract*
between what ``hermes model`` shows (``CANONICAL_PROVIDERS``) and what the
catalog exposes, plus how each provider's ``auth_type`` maps to a desktop tab —
never a specific provider count or a frozen vendor list (both change over time).
"""

from hermes_cli.models import CANONICAL_PROVIDERS
from hermes_cli.provider_catalog import (
    ProviderDescriptor,
    provider_catalog,
    provider_catalog_by_slug,
    tab_for_auth_type,
)


def test_catalog_covers_every_hermes_model_provider():
    """PARITY CONTRACT: the catalog == the `hermes model` universe."""
    slugs = {d.slug for d in provider_catalog()}
    for entry in CANONICAL_PROVIDERS:
        assert entry.slug in slugs, (
            f"{entry.slug} is shown in `hermes model` but missing from provider_catalog()"
        )


def test_catalog_has_no_providers_outside_hermes_model():
    """The catalog must not invent providers `hermes model` doesn't show."""
    canonical = {e.slug for e in CANONICAL_PROVIDERS}
    for d in provider_catalog():
        assert d.slug in canonical, f"{d.slug} in catalog but not in CANONICAL_PROVIDERS"


def test_every_descriptor_lands_on_exactly_one_known_tab():
    for d in provider_catalog():
        assert d.tab in {"keys", "accounts"}, f"{d.slug} has bad tab {d.tab!r}"


def test_descriptor_count_matches_canonical():
    """One descriptor per canonical entry (no dupes, no drops)."""
    cat = provider_catalog()
    assert len(cat) == len(CANONICAL_PROVIDERS)
    assert len({d.slug for d in cat}) == len(cat)


def test_profileless_providers_still_present():
    """Providers without a ProviderProfile must still resolve via fallbacks.

    lmstudio / openai-api / tencent-tokenhub / xai-oauth have no profile on
    main; they exist only as registry + canonical entries. The catalog must
    not require a profile to include a provider.
    """
    by = provider_catalog_by_slug()
    for slug in ("lmstudio", "openai-api", "tencent-tokenhub", "xai-oauth"):
        assert slug in by, f"{slug} dropped from catalog (profile-less provider)"
        assert by[slug].label, f"{slug} has empty label despite canonical fallback"
        assert by[slug].description, f"{slug} has empty description despite fallback"


def test_api_key_providers_route_to_keys_oauth_to_accounts():
    by = provider_catalog_by_slug()
    # api_key → keys
    assert by["kilocode"].tab == "keys"
    assert by["openai-api"].tab == "keys"
    # account / sign-in flows → accounts
    assert by["google-gemini-cli"].tab == "accounts"
    assert by["copilot-acp"].tab == "accounts"


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


def test_bedrock_routes_to_keys():
    """Bedrock is aws_sdk (AWS_REGION/AWS_PROFILE), configured on the keys tab."""
    by = provider_catalog_by_slug()
    assert by["bedrock"].tab == "keys"


def test_api_key_providers_expose_a_credential_env_var():
    """Every keys-tab provider that authenticates via a pasted API key must
    surface at least one env var to write the key into (otherwise the GUI can't
    configure it).

    Exemptions: ``aws_sdk`` (bedrock — uses AWS_REGION/AWS_PROFILE) and the
    ``custom`` bring-your-own-endpoint pseudo-provider, which is configured
    inline via the local-endpoint flow rather than a fixed env var.
    """
    exempt = {"custom"}
    for d in provider_catalog():
        if d.auth_type == "api_key" and d.slug not in exempt:
            assert d.api_key_env_vars, f"{d.slug} is api_key but exposes no env var"


def test_order_mirrors_canonical_declaration():
    cat = provider_catalog()
    assert [d.order for d in cat] == list(range(len(cat)))
    assert [d.slug for d in cat] == [e.slug for e in CANONICAL_PROVIDERS]


def test_descriptors_are_provider_descriptor_instances():
    for d in provider_catalog():
        assert isinstance(d, ProviderDescriptor)


def test_tab_for_auth_type_helper():
    assert tab_for_auth_type("api_key") == "keys"
    assert tab_for_auth_type("aws_sdk") == "keys"
    assert tab_for_auth_type("oauth_external") == "accounts"
    assert tab_for_auth_type("oauth_device_code") == "accounts"
    assert tab_for_auth_type("copilot") == "accounts"
    assert tab_for_auth_type("external_process") == "accounts"

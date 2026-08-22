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

    Exemptions: ``aws_sdk`` (bedrock — uses AWS_REGION/AWS_PROFILE), the
    ``custom`` bring-your-own-endpoint pseudo-provider (configured inline via
    the local-endpoint flow), and keyless providers (``d.keyless`` — e.g.
    opencode-free, served anonymously: there is no credential to write).
    """
    exempt = {"custom"}
    for d in provider_catalog():
        if d.auth_type == "api_key" and d.slug not in exempt and not d.keyless:
            assert d.api_key_env_vars, f"{d.slug} is api_key but exposes no env var"




def test_tab_for_auth_type_helper():
    assert tab_for_auth_type("api_key") == "keys"
    assert tab_for_auth_type("aws_sdk") == "keys"
    assert tab_for_auth_type("oauth_external") == "accounts"
    assert tab_for_auth_type("oauth_device_code") == "accounts"
    assert tab_for_auth_type("copilot") == "accounts"
    assert tab_for_auth_type("external_process") == "accounts"

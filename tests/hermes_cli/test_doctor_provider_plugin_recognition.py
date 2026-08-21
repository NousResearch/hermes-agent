"""Doctor must recognise providers declared by provider plugins.

Provider plugins register a ``ProviderProfile`` with the ``providers/``
registry; they never appear in the static model catalog in
``hermes_cli/providers.py``. Doctor's ``model.provider`` validation used to
resolve names against that catalog alone, so a provider authored per the
documented plugin recipe failed with "not a recognised provider" while being
listed among the known providers in the very same message.

These assert the relationship — "registered profile ⇒ recognised" — rather
than snapshotting which providers happen to ship today.
"""

import pytest

from hermes_cli.doctor import resolve_registered_provider_id


@pytest.fixture
def registered_profile():
    """Register a throwaway provider profile.

    No cleanup needed: the suite runs one subprocess per test file, so the
    process-global provider registry cannot leak into another file.
    """
    from providers import register_provider
    from providers.base import ProviderProfile

    profile = ProviderProfile(
        name="doctor-test-provider",
        aliases=("dtp",),
        base_url="http://127.0.0.1:9/v1",
        auth_type="api_key",
    )
    register_provider(profile)
    return profile


def test_registered_provider_is_recognised(registered_profile):
    assert resolve_registered_provider_id("doctor-test-provider") == "doctor-test-provider"


def test_alias_resolves_to_canonical_name(registered_profile):
    assert resolve_registered_provider_id("dtp") == "doctor-test-provider"


def test_lookup_is_case_insensitive(registered_profile):
    assert resolve_registered_provider_id("  Doctor-Test-Provider  ") == "doctor-test-provider"


def test_unregistered_provider_is_not_recognised():
    assert resolve_registered_provider_id("no-such-provider-anywhere") is None


@pytest.mark.parametrize("value", ["", "   ", None])
def test_blank_input_is_not_recognised(value):
    assert resolve_registered_provider_id(value) is None


def test_every_bundled_provider_passes_the_check():
    """Invariant: anything the registry lists must satisfy the validation.

    This is what actually broke — the doctor's own "known providers" list is
    built from this registry, so a provider it advertises must never be
    rejected as unknown.
    """
    from providers import list_providers

    registered = list_providers()
    assert registered, "provider discovery returned nothing — registry is not wired"
    for profile in registered:
        assert resolve_registered_provider_id(profile.name) == profile.name

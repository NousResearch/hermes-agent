"""``resolve_provider`` must accept named custom providers (``custom:<name>``).

Named entries from ``custom_providers`` never appear in ``PROVIDER_REGISTRY``,
so every config path routing through ``resolve_provider`` — MoA reference and
aggregator models, ``fallback_providers`` — used to die at agent init with
"Unknown provider 'custom:<name>'".
"""

import pytest

from hermes_cli.auth import AuthError, resolve_provider


@pytest.mark.parametrize(
    "provider",
    ["custom:grok-gate", "custom:qwen-gate", "custom:dario", "custom:deepseek-gate"],
)
def test_named_custom_provider_resolves_to_itself(provider):
    """The name survives verbatim: callers key off the specific entry."""
    assert resolve_provider(provider) == provider


def test_named_custom_is_not_collapsed_to_bare_custom():
    """Collapsing would reroute a named provider onto the generic custom path.

    ``runtime_provider`` looks the name up to pick that provider's own
    ``base_url``/``api_key``; bare "custom" loses the endpoint identity.
    """
    assert resolve_provider("custom:grok-gate") != "custom"


def test_named_custom_matches_is_runtime_provider_routable():
    """The two resolution paths in this module must agree on the same shape."""
    from hermes_cli.auth import is_runtime_provider_routable

    assert is_runtime_provider_routable("custom:grok-gate") is True
    assert resolve_provider("custom:grok-gate") == "custom:grok-gate"


def test_bare_custom_still_resolves():
    """The pre-existing bare-``custom`` contract is untouched."""
    assert resolve_provider("custom") == "custom"


@pytest.mark.parametrize("provider", ["custom:", "custom:   "])
def test_empty_custom_suffix_still_raises(provider):
    """An empty name is not a provider identity — fail closed."""
    with pytest.raises(AuthError):
        resolve_provider(provider)


def test_unknown_provider_still_raises():
    """Unrelated garbage must not slip through the new branch."""
    with pytest.raises(AuthError):
        resolve_provider("totally-bogus-provider")

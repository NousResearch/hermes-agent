"""Tests for issue #65793 — OpenRouter missing from PROVIDER_REGISTRY.

The desktop model picker drops OpenRouter because
is_provider_explicitly_configured("openrouter") returns False —
the env-var check consults PROVIDER_REGISTRY which had no openrouter entry.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from hermes_cli.auth import PROVIDER_REGISTRY, is_provider_explicitly_configured


# --------------------------------------------------------------------------- #
# PROVIDER_REGISTRY has openrouter
# --------------------------------------------------------------------------- #


def test_openrouter_in_registry():
    """OpenRouter must be in PROVIDER_REGISTRY."""
    assert "openrouter" in PROVIDER_REGISTRY, (
        "OpenRouter must be in PROVIDER_REGISTRY — see issue #65793"
    )


def test_openrouter_registry_entry_fields():
    """The openrouter entry must have correct fields."""
    pconfig = PROVIDER_REGISTRY["openrouter"]
    assert pconfig.auth_type == "api_key"
    assert "OPENROUTER_API_KEY" in pconfig.api_key_env_vars
    assert pconfig.inference_base_url  # non-empty
    assert "openrouter" in pconfig.inference_base_url.lower()


# --------------------------------------------------------------------------- #
# is_provider_explicitly_configured detects OpenRouter via env var
# --------------------------------------------------------------------------- #


def test_is_explicitly_configured_openrouter_with_env_key():
    """When OPENROUTER_API_KEY is set, is_provider_explicitly_configured
    must return True for openrouter."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-v1-fake-key-for-test"}):
        result = is_provider_explicitly_configured("openrouter")
    assert result is True, (
        "is_provider_explicitly_configured('openrouter') must return True "
        "when OPENROUTER_API_KEY is set — see issue #65793"
    )


def test_is_explicitly_configured_openrouter_without_env_key():
    """When no OPENROUTER_API_KEY is set and no config/auth.json, returns False."""
    # Make sure no env key, no auth.json, no config.yaml pointing to openrouter
    env_without_key = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
    with patch.dict(os.environ, env_without_key, clear=True), \
         patch("hermes_cli.auth._load_auth_store", return_value={}), \
         patch("hermes_cli.config.load_config", return_value={}):
        result = is_provider_explicitly_configured("openrouter")
    assert result is False


# --------------------------------------------------------------------------- #
# resolve_provider still works with openrouter in registry
# --------------------------------------------------------------------------- #


def test_resolve_provider_openrouter_still_resolves():
    """Adding openrouter to PROVIDER_REGISTRY must not break resolve_provider().

    resolve_provider() short-circuits with `if normalized == "openrouter":
    return "openrouter"` before the registry check, so the addition is safe.
    """
    from hermes_cli.auth import resolve_provider

    # Direct resolution
    result = resolve_provider("openrouter")
    assert result == "openrouter"


def test_resolve_provider_unknown_still_raises():
    """Unknown providers must still raise AuthError (not silently resolve)."""
    from hermes_cli.auth import resolve_provider, AuthError

    with pytest.raises(AuthError):
        resolve_provider("definitely-not-a-real-provider")


# --------------------------------------------------------------------------- #
# Auto-extend doesn't duplicate openrouter
# --------------------------------------------------------------------------- #


def test_auto_extend_skips_openrouter():
    """The auto-extend loop must not create a duplicate openrouter entry."""
    # The skip set on line 462 no longer includes openrouter, but since
    # openrouter IS already in PROVIDER_REGISTRY (we added it above), the
    # `if _pp.name in PROVIDER_REGISTRY: continue` check on line 453 handles it.
    # Verify there's exactly one entry:
    import hermes_cli.auth as auth_mod
    # Count how many times "openrouter" appears as a key
    # (it's a dict so duplicates are impossible, but verify the auto-extend
    # didn't create a second ProviderConfig with different fields)
    entry = PROVIDER_REGISTRY["openrouter"]
    assert entry.api_key_env_vars == ("OPENROUTER_API_KEY",)
    assert entry.name == "OpenRouter"
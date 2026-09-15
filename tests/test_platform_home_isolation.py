"""A test run must never resolve — let alone clobber — the operator's real Hermes home.

``hermes_constants.get_default_hermes_root()`` prefers the platform-native home whenever
``HERMES_HOME`` is empty or sits *under* it. The per-test ``HERMES_HOME`` is
``<basetemp>/hermes_test``, so a pytest basetemp inside the operator's Hermes home silently turns
that sandbox back into the real root and any test writing through ``get_profile_dir("default")``
lands in the live install. The autouse ``_bind_platform_native_home`` fixture in tests/conftest.py
redirects the native base in exactly that shape; these tests pin both sides of that boundary.
"""
from __future__ import annotations

import os
from pathlib import Path

import hermes_constants

# Captured at collection time, before any per-test fixture can redirect the resolver.
_OPERATOR_PLATFORM_HOME = Path(hermes_constants._get_platform_default_hermes_home()).resolve()


def test_default_profile_root_is_never_the_operator_home():
    """Whatever basetemp was used, the resolved default root must not be the live install."""
    assert hermes_constants.get_default_hermes_root().resolve() != _OPERATOR_PLATFORM_HOME


def test_native_binding_applies_only_to_the_inverted_shape():
    """The autouse binding must stay out of the way unless HERMES_HOME sits under the native home.

    Tests that deliberately unset ``HERMES_HOME`` and patch ``Path.home()`` assert native/profile
    resolution; redirecting the native base for them changes what they measure. Only the inversion
    (per-test home *under* the operator's native home) is guarded.
    """
    env_home = os.environ.get("HERMES_HOME", "").strip()
    bound = Path(hermes_constants._get_platform_default_hermes_home()).resolve()
    inverted = bool(env_home) and Path(env_home).resolve().is_relative_to(_OPERATOR_PLATFORM_HOME)
    if inverted:
        assert bound != _OPERATOR_PLATFORM_HOME, "inverted shape must be redirected"
        assert "platform-native-home" in str(bound)
    else:
        assert bound == _OPERATOR_PLATFORM_HOME, "non-inverted shape must keep the real resolver"

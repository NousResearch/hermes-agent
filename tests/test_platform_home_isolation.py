"""A test run must never resolve — let alone clobber — the operator's real Hermes home.

``hermes_constants.get_default_hermes_root()`` prefers the platform-native home whenever
``HERMES_HOME`` is empty or sits *under* it. The per-test ``HERMES_HOME`` is
``<basetemp>/hermes_test``, so a pytest basetemp inside the operator's Hermes home silently turns
that sandbox back into the real root and any test writing through ``get_profile_dir("default")``
lands in the live install. The autouse ``_bind_platform_native_home`` fixture in tests/conftest.py
redirects the native base in exactly that shape; these tests pin both sides of that boundary.
"""
from __future__ import annotations

from pathlib import Path

import hermes_constants
from tests.conftest import _hermes_home_under_native_home

# Captured at collection time, before any per-test fixture can redirect the resolver.
_OPERATOR_PLATFORM_HOME = hermes_constants._get_platform_default_hermes_home().resolve()


def test_default_profile_root_is_never_the_operator_home():
    """Tripwire: bites only when basetemp sits under the native home (the shape the fixture guards)."""
    assert hermes_constants.get_default_hermes_root().resolve() != _OPERATOR_PLATFORM_HOME


def test_only_a_home_under_the_native_root_is_redirected(tmp_path):
    """The autouse binding fires for the inverted shape and stays out of every other one."""
    native = tmp_path / "native-home"
    assert _hermes_home_under_native_home(str(native / "cache" / "pytest-0" / "hermes_test"), native)
    assert not _hermes_home_under_native_home(str(tmp_path / "elsewhere" / "hermes_test"), native)
    assert not _hermes_home_under_native_home("", native)

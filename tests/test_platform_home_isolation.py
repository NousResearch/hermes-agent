"""A test run must never resolve — let alone write to — the operator's real Hermes home.

``hermes_constants.get_default_hermes_root()`` prefers the platform-native home whenever
``HERMES_HOME`` is empty or sits *under* it. The per-test ``HERMES_HOME`` is
``<basetemp>/hermes_test``, so a pytest basetemp inside the operator's Hermes home silently turned
that sandbox back into the real root and ``tests/hermes_cli/test_profiles.py`` wrote its fixtures
into the live install (``config.yaml`` -> ``"ok"``, ``.env`` -> ``KEY=val``,
``MEMORY.md`` -> ``remember this``).

These invariants pin the autouse ``_bind_platform_native_home`` binding: delete it and both tests
fail, because the session would resolve the operator's actual home.
"""
from __future__ import annotations

from pathlib import Path

import hermes_constants


def test_session_binds_the_platform_native_home_away_from_the_operator(operator_platform_home):
    assert operator_platform_home is not None, "conftest could not read the platform-native home"
    bound = Path(hermes_constants._get_platform_default_hermes_home()).resolve()
    assert bound != operator_platform_home, (
        "the test session resolves the operator's real Hermes home "
        f"({operator_platform_home}); _bind_platform_native_home must redirect it"
    )


def test_default_profile_dir_is_not_the_operators_live_install(operator_platform_home):
    from hermes_cli.profiles import get_profile_dir

    resolved = get_profile_dir("default").resolve()
    assert resolved != operator_platform_home
    assert operator_platform_home not in resolved.parents

"""A test run must never resolve — let alone clobber — the operator's real Hermes home.

``hermes_constants.get_default_hermes_root()`` prefers the platform-native home whenever
``HERMES_HOME`` is empty or sits *under* it. The per-test ``HERMES_HOME`` is
``<basetemp>/hermes_test``, so a pytest basetemp inside the operator's Hermes home silently turned
that sandbox back into the real root and ``tests/hermes_cli/test_profiles.py`` wrote its fixtures
into the live install (``config.yaml`` -> ``"ok"``, ``.env`` -> ``KEY=val``,
``memories/MEMORY.md`` -> ``remember this``).

These tests pin the two halves of the guard separately: the resolution boundary (the default
profile root is never the operator's) and the damage predicate (it sees a real clobber, and it
stays quiet on a synthetic home whose files are legitimately small).
"""
from __future__ import annotations

import os
from pathlib import Path

import hermes_constants


def _entry(size: int, digest: str, keys: int | None = None) -> dict:
    entry = {"size": size, "sha256": digest}
    if keys is not None:
        entry["keys"] = keys
    return entry


def test_default_profile_root_is_never_the_operator_home(operator_platform_home):
    """Whatever basetemp was used, the resolved default root must not be the live install."""
    assert operator_platform_home is not None, "conftest could not read the platform-native home"
    assert hermes_constants.get_default_hermes_root().resolve() != operator_platform_home


def test_native_binding_applies_only_to_the_inverted_shape(operator_platform_home):
    """The autouse binding must stay out of the way unless HERMES_HOME sits under the native home.

    Tests that deliberately unset ``HERMES_HOME`` and patch ``Path.home()`` assert native/profile
    resolution; redirecting the native base for them changes what they measure. Only the inversion
    (per-test home *under* the operator's native home) is guarded.
    """
    env_home = os.environ.get("HERMES_HOME", "").strip()
    bound = Path(hermes_constants._get_platform_default_hermes_home()).resolve()
    inverted = bool(env_home) and Path(env_home).resolve().is_relative_to(operator_platform_home)
    if inverted:
        assert bound != operator_platform_home, "inverted shape must be redirected"
        assert "platform-native-home" in str(bound)
    else:
        assert bound == operator_platform_home, "non-inverted shape must keep the real resolver"


def test_damage_detector_sees_a_memory_store_overwrite(home_damage_detector):
    """The store path (memories/MEMORY.md) is what a clobber actually destroys — watch it."""
    before = {"memories/MEMORY.md": _entry(258_765, "a" * 64), "config.yaml": _entry(13_823, "b" * 64)}
    after = {"memories/MEMORY.md": _entry(13, "c" * 64), "config.yaml": _entry(13_823, "b" * 64)}

    findings = home_damage_detector(before, after)

    assert findings, "a populated memories/MEMORY.md replaced by 'remember this' must be detected"
    assert any("memories/MEMORY.md" in finding for finding in findings)


def test_damage_detector_ignores_an_untouched_small_config(home_damage_detector):
    """A synthetic home with a legitimately tiny config must not fail every per-file session."""
    tiny = _entry(26, "d" * 64)
    snapshot = {"config.yaml": tiny, "SOUL.md": _entry(9, "e" * 64), ".env": _entry(8, "f" * 64, keys=1)}

    assert home_damage_detector(snapshot, dict(snapshot)) == []


def test_damage_detector_ignores_a_legitimate_rewrite(home_damage_detector):
    """The operator's own agent rewrites these files while the suite runs — that is not damage."""
    before = {"config.yaml": _entry(13_823, "a" * 64), ".env": _entry(7_400, "b" * 64, keys=94)}
    after = {"config.yaml": _entry(14_855, "c" * 64), ".env": _entry(7_406, "d" * 64, keys=94)}

    assert home_damage_detector(before, after) == []


def test_damage_detector_reports_config_and_env_collapse(home_damage_detector):
    before = {"config.yaml": _entry(13_823, "a" * 64), ".env": _entry(7_400, "b" * 64, keys=94)}
    after = {"config.yaml": _entry(2, "c" * 64), ".env": _entry(9, "d" * 64, keys=1)}

    findings = home_damage_detector(before, after)

    assert any("config.yaml" in finding for finding in findings)
    assert any(".env" in finding for finding in findings)

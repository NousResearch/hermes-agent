"""Regression: profile-home resolvers must be legacy-authoritative (#18594).

A profile-dispatched child sets ``HERMES_HOME=<profile>`` explicitly while
``HT_HOME`` may be a *stale* value inherited from the parent: ``mirror_brand_env``
fills ``HT_HOME`` in the parent, and every profile spawner copies ``os.environ``
(``env={**os.environ, "HERMES_HOME": <profile>}``) without updating ``HT_HOME``.
The legacy ``HERMES_*`` value must therefore win — otherwise
``_profile_home_path`` / ``_iter_real_home_candidates`` disagree with
``get_hermes_home()`` (legacy-authoritative via ``resolve_env``) *within the same
process*, and the subprocess HOME contract is computed against the wrong profile
home. Guards against a regression of the hand-rolled HT-first ordering.
"""

import hermes_constants


def test_profile_home_path_prefers_legacy_over_stale_ht_home(tmp_path):
    # Explicit HERMES_HOME points at the active profile; a stale inherited
    # HT_HOME points at the parent/default home. Both profile-home dirs exist,
    # so a wrong (HT-first) precedence returns the wrong *non-None* directory.
    profile = tmp_path / "profiles" / "coder"
    (profile / "home").mkdir(parents=True)
    parent = tmp_path / "parent"
    (parent / "home").mkdir(parents=True)
    env = {"HT_HOME": str(parent), "HERMES_HOME": str(profile)}
    assert hermes_constants._profile_home_path(env) == str(profile / "home")


def test_profile_home_path_uses_ht_home_when_legacy_absent(tmp_path):
    # New-name-only callers still work: HT_HOME applies when HERMES_HOME is unset.
    home = tmp_path / "htonly"
    (home / "home").mkdir(parents=True)
    env = {"HT_HOME": str(home)}
    assert hermes_constants._profile_home_path(env) == str(home / "home")


def test_iter_real_home_candidates_prefers_legacy_over_stale_ht_real_home():
    # HERMES_REAL_HOME (explicitly set by a spawner) must outrank a stale
    # inherited HT_REAL_HOME, so external CLIs read the right OS-user home.
    env = {"HT_REAL_HOME": "/stale/parent", "HERMES_REAL_HOME": "/correct/child"}
    candidates = hermes_constants._iter_real_home_candidates(env)
    assert candidates[0] == "/correct/child"


def test_iter_real_home_candidates_uses_ht_real_home_when_legacy_absent():
    env = {"HT_REAL_HOME": "/new/only"}
    candidates = hermes_constants._iter_real_home_candidates(env)
    assert candidates[0] == "/new/only"

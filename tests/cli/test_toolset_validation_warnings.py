"""Tests for hermes_cli.toolset_validation warning paths (see #59547).

Covers the three branches the validator must handle when a toolset name in
``platform_toolsets`` is rejected by ``is_valid_toolset``:

1. The name is in ``known_plugin_toolsets[platform]`` → "plugin disabled or
   uninstalled" warning, *not* the generic "unknown toolset" wording, and
   *not* the ``hermes-<platform>`` typo guess.
2. The name is genuinely unknown → falls through to the existing generic
   "unknown toolset" warning (with the ``hermes-<platform>`` hint when that
   name happens to be valid).
3. The name belongs to an enabled/loaded plugin → no warning at all.

Plus edge cases for malformed / missing ``known_plugin_toolsets`` so the new
parameter never breaks the pre-#59547 behavior.
"""

import pytest

from hermes_cli.toolset_validation import validate_platform_toolsets

# A representative registry snapshot. ``hermes-cli`` is the canonical "would
# have been a typo" name that #38798 surfaced; ``spotify_plugin_toolset`` is
# a plugin toolset that exists in ``known_plugin_toolsets`` but has been
# "removed" from the live registry in the tests below (i.e. its plugin got
# disabled or uninstalled). ``web`` is an enabled, loaded toolset.
_KNOWN = {
    "hermes-cli",
    "hermes-telegram",
    "hermes-discord",
    "terminal",
    "web",
}


def _is_valid(name):
    return name in _KNOWN


# ---------------------------------------------------------------------------
# Branch 1 — name is in known_plugin_toolsets for the same platform → plugin
# warning, NOT the generic typo wording. #59547.
# ---------------------------------------------------------------------------


def test_known_plugin_toolset_now_missing_emits_plugin_disabled_warning():
    """The exact #59547 reproduction: a plugin toolset that used to be valid
    for this platform is no longer in the live registry."""
    cfg = {"cli": ["spotify_plugin_toolset", "web"]}
    known = {"cli": ["spotify_plugin_toolset"]}
    warnings = validate_platform_toolsets(cfg, _is_valid, known)

    # ``web`` is valid, so only the plugin warning fires — no zero-valid noise.
    assert len(warnings) == 1
    msg = warnings[0]
    assert "plugin is disabled or uninstalled" in msg
    assert "plugins.enabled" in msg
    # The generic typo wording and the misleading `hermes-cli` guess must
    # NOT appear, since neither would point the user at the real fix.
    assert "unknown toolset" not in msg
    assert "did you mean" not in msg


def test_known_plugin_toolset_for_one_platform_does_not_leak_to_another():
    """A name that's only known for ``telegram`` must still be flagged as a
    genuine typo for ``cli`` — the cross-reference is platform-scoped."""
    cfg = {"cli": ["spotify_plugin_toolset", "web"]}
    known = {"telegram": ["spotify_plugin_toolset"]}
    warnings = validate_platform_toolsets(cfg, _is_valid, known)

    # Falls back to the existing generic "unknown toolset" branch. ``web`` is
    # valid so no zero-valid noise; only the typo warning fires.
    assert len(warnings) == 1
    assert "unknown toolset 'spotify_plugin_toolset'" in warnings[0]


def test_known_plugin_toolset_disabled_warning_does_not_pair_with_typo_guess():
    """The `hermes-<platform>` typo guess is irrelevant for a known-but-now-
    missing plugin, so it must not be appended to the plugin warning."""
    cfg = {"cli": ["spotify_plugin_toolset"]}
    known = {"cli": ["spotify_plugin_toolset"]}
    warnings = validate_platform_toolsets(cfg, _is_valid, known)

    assert all("did you mean" not in w for w in warnings)


# ---------------------------------------------------------------------------
# Branch 2 — genuine typo / unknown name → falls through to the existing
# generic warning. No regression on #38798 behavior.
# ---------------------------------------------------------------------------


def test_unknown_toolset_falls_through_to_generic_warning():
    cfg = {"cli": ["hermes"]}  # the canonical #38798 corruption shape
    known = {"cli": ["some_other_plugin"]}
    warnings = validate_platform_toolsets(cfg, _is_valid, known)

    unknown = [w for w in warnings if "unknown toolset 'hermes'" in w]
    assert len(unknown) == 1
    assert "did you mean 'hermes-cli'?" in unknown[0]


def test_unknown_toolset_without_known_map_keeps_generic_warning():
    """Omitting the new parameter entirely must reproduce pre-#59547 output."""
    cfg = {"cli": ["hermes"]}
    warnings = validate_platform_toolsets(cfg, _is_valid)
    unknown = [w for w in warnings if "unknown toolset 'hermes'" in w]
    assert len(unknown) == 1
    assert "did you mean 'hermes-cli'?" in unknown[0]


def test_unknown_toolset_when_known_map_lacks_the_name():
    """A non-empty known map that doesn't contain the offending name must not
    turn a typo into a plugin warning."""
    cfg = {"cli": ["hermes"]}
    known = {"cli": ["unrelated_plugin"]}
    warnings = validate_platform_toolsets(cfg, _is_valid, known)

    unknown = [w for w in warnings if "unknown toolset 'hermes'" in w]
    assert len(unknown) == 1
    assert "did you mean 'hermes-cli'?" in unknown[0]


# ---------------------------------------------------------------------------
# Branch 3 — toolset is genuinely valid → no warning at all.
# ---------------------------------------------------------------------------


def test_known_enabled_toolset_in_known_plugin_map_produces_no_warning():
    cfg = {"cli": ["web"], "telegram": ["hermes-telegram"]}
    known = {"cli": ["web", "spotify_plugin_toolset"]}
    warnings = validate_platform_toolsets(cfg, _is_valid, known)
    assert warnings == []


def test_known_enabled_toolset_without_known_map_produces_no_warning():
    """Pre-#59547 regression guard: a valid toolset with no known map at all
    must not produce a warning."""
    cfg = {"cli": ["hermes-cli"]}
    warnings = validate_platform_toolsets(cfg, _is_valid)
    assert warnings == []


# ---------------------------------------------------------------------------
# Edge cases — malformed / missing known_plugin_toolsets. None of these
# should turn into an exception or change pre-#59547 behavior.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [None, [], "spotify_plugin_toolset", 42, {"cli": "not-a-list"}, {"cli": 7}],
)
def test_malformed_known_plugin_toolsets_falls_back_to_generic_behavior(value):
    """Anything that isn't a {platform: [toolset, ...]} mapping should be
    treated as "no data" and reproduce pre-#59547 output."""
    cfg = {"cli": ["hermes"]}
    warnings = validate_platform_toolsets(cfg, _is_valid, value)

    unknown = [w for w in warnings if "unknown toolset 'hermes'" in w]
    assert len(unknown) == 1
    assert "did you mean 'hermes-cli'?" in unknown[0]


def test_empty_dict_known_plugin_toolsets_falls_back_to_generic_behavior():
    cfg = {"cli": ["hermes"]}
    warnings = validate_platform_toolsets(cfg, _is_valid, {})
    unknown = [w for w in warnings if "unknown toolset 'hermes'" in w]
    assert len(unknown) == 1


# ---------------------------------------------------------------------------
# Mixed entries — the validator must produce one warning per offending name
# and still respect the zero-valid-toolsets safety net from #38798.
# ---------------------------------------------------------------------------


def test_mixed_unknown_typo_and_known_plugin_disabled():
    cfg = {
        "cli": ["hermes", "spotify_plugin_toolset"],
        "telegram": ["hermes-telegram"],
    }
    known = {"cli": ["spotify_plugin_toolset"]}
    warnings = validate_platform_toolsets(cfg, _is_valid, known)

    # One valid entry exists, so no "zero valid toolsets" warning.
    assert not any("zero valid toolsets" in w for w in warnings)

    plugin_warnings = [w for w in warnings if "plugin is disabled" in w]
    typo_warnings = [w for w in warnings if "unknown toolset" in w]
    assert len(plugin_warnings) == 1
    assert len(typo_warnings) == 1
    assert "spotify_plugin_toolset" in plugin_warnings[0]
    assert "hermes" in typo_warnings[0]


def test_zero_valid_state_still_fires_when_all_references_are_stale():
    """If every name in a platform is a known-but-now-missing plugin, the
    zero-valid safety net from #38798 must still surface."""
    cfg = {"cli": ["plugin_a", "plugin_b"]}
    known = {"cli": ["plugin_a", "plugin_b"]}
    warnings = validate_platform_toolsets(cfg, _is_valid, known)

    assert sum("plugin is disabled" in w for w in warnings) == 2
    assert any("zero valid toolsets" in w for w in warnings)
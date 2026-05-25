"""Tests for agent.lmstudio_reasoning — resolve_lmstudio_effort()."""

from __future__ import annotations

import pytest

from agent.lmstudio_reasoning import resolve_lmstudio_effort


# ---------------------------------------------------------------------------
# No reasoning_config — default effort
# ---------------------------------------------------------------------------
class TestNoReasoningConfig:
    """When reasoning_config is None, effort defaults to 'medium'."""

    def test_none_reasoning_config_returns_medium(self):
        assert resolve_lmstudio_effort(None, None) == "medium"

    def test_none_reasoning_config_with_allowed_clamped(self):
        assert resolve_lmstudio_effort(None, ["low", "medium", "high"]) == "medium"

    def test_none_reasoning_config_with_allowed_missing_medium_returns_none(self):
        assert resolve_lmstudio_effort(None, ["low", "high"]) is None

    def test_none_reasoning_config_with_empty_allowed(self):
        assert resolve_lmstudio_effort(None, []) == "medium"

    def test_non_dict_reasoning_config_returns_medium(self):
        """If reasoning_config is not a dict, fall back to default."""
        assert resolve_lmstudio_effort("not_a_dict", None) == "medium"  # type: ignore[arg-type]
        assert resolve_lmstudio_effort(42, None) == "medium"  # type: ignore[arg-type]
        assert resolve_lmstudio_effort([], None) == "medium"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# reasoning_config enabled=False → "none"
# ---------------------------------------------------------------------------
class TestEnabledFalse:
    def test_enabled_false_no_allowed(self):
        assert resolve_lmstudio_effort({"enabled": False}, None) == "none"

    def test_enabled_false_with_allowed_containing_none(self):
        assert resolve_lmstudio_effort({"enabled": False}, ["none", "medium"]) == "none"

    def test_enabled_false_with_allowed_missing_none(self):
        """If allowed_options excludes 'none', return None."""
        assert resolve_lmstudio_effort({"enabled": False}, ["medium", "high"]) is None


# ---------------------------------------------------------------------------
# Explicit effort values
# ---------------------------------------------------------------------------
class TestExplicitEffort:
    def test_effort_high(self):
        assert resolve_lmstudio_effort({"effort": "high"}, None) == "high"

    def test_effort_minimal(self):
        assert resolve_lmstudio_effort({"effort": "minimal"}, None) == "minimal"

    def test_effort_low(self):
        assert resolve_lmstudio_effort({"effort": "low"}, None) == "low"

    def test_effort_xhigh(self):
        assert resolve_lmstudio_effort({"effort": "xhigh"}, None) == "xhigh"

    def test_effort_none_literal(self):
        """Literal 'none' string as effort."""
        assert resolve_lmstudio_effort({"effort": "none"}, None) == "none"

    def test_effort_invalid_falls_back_to_medium(self):
        """Invalid effort value falls back to default 'medium'."""
        assert resolve_lmstudio_effort({"effort": "extreme"}, None) == "medium"
        assert resolve_lmstudio_effort({"effort": "super"}, None) == "medium"
        assert resolve_lmstudio_effort({"effort": "banana"}, None) == "medium"


# ---------------------------------------------------------------------------
# Case-insensitivity in effort
# ---------------------------------------------------------------------------
class TestEffortCaseInsensitivity:
    def test_upper_case(self):
        assert resolve_lmstudio_effort({"effort": "HIGH"}, None) == "high"

    def test_mixed_case(self):
        assert resolve_lmstudio_effort({"effort": "MeDiUm"}, None) == "medium"

    def test_leading_trailing_whitespace(self):
        assert resolve_lmstudio_effort({"effort": "  high  "}, None) == "high"


# ---------------------------------------------------------------------------
# Empty / falsy effort field
# ---------------------------------------------------------------------------
class TestEmptyEffort:
    def test_effort_empty_string(self):
        """Empty effort string strips to empty and falls through to default."""
        assert resolve_lmstudio_effort({"effort": ""}, None) == "medium"

    def test_effort_whitespace_only(self):
        assert resolve_lmstudio_effort({"effort": "   "}, None) == "medium"

    def test_effort_missing_key(self):
        """Missing 'effort' key — default 'medium'."""
        assert resolve_lmstudio_effort({}, None) == "medium"


# ---------------------------------------------------------------------------
# LM Studio aliases: "off" → "none", "on" → "medium"
# ---------------------------------------------------------------------------
class TestLmstudioAliases:
    def test_alias_off_maps_to_none(self):
        assert resolve_lmstudio_effort({"effort": "off"}, None) == "none"

    def test_alias_on_maps_to_medium(self):
        assert resolve_lmstudio_effort({"effort": "on"}, None) == "medium"

    def test_alias_off_with_allowed(self):
        """off→none, and 'none' is in allowed → returns 'none'."""
        assert resolve_lmstudio_effort({"effort": "off"}, ["none", "medium"]) == "none"

    def test_alias_off_blocked_by_allowed(self):
        """off→none, but 'none' not in allowed → None."""
        assert resolve_lmstudio_effort({"effort": "off"}, ["medium", "high"]) is None

    def test_alias_on_with_allowed(self):
        """on→medium, and 'medium' is in allowed → returns 'medium'."""
        assert resolve_lmstudio_effort({"effort": "on"}, ["low", "medium", "high"]) == "medium"

    def test_alias_on_blocked_by_allowed(self):
        """on→medium, but 'medium' not in allowed → None."""
        assert resolve_lmstudio_effort({"effort": "on"}, ["low", "high"]) is None


# ---------------------------------------------------------------------------
# Allowed options clamping
# ---------------------------------------------------------------------------
class TestAllowedOptionsClamping:
    def test_effort_in_allowed_returns_effort(self):
        assert resolve_lmstudio_effort({"effort": "high"}, ["low", "medium", "high"]) == "high"

    def test_effort_not_in_allowed_returns_none(self):
        assert resolve_lmstudio_effort({"effort": "high"}, ["low", "medium"]) is None

    def test_allowed_options_accepts_aliases(self):
        """allowed_options with 'off'/'on' get mapped before checking."""
        # enabled=False → effort="none". allowed=["off"] → mapped to {"none"}
        # "none" is in {"none"} → returns "none"
        assert resolve_lmstudio_effort({"enabled": False}, ["off"]) == "none"

    def test_allowed_options_accepts_aliases_blocked(self):
        # effort="medium" (default, no reasoning_config). allowed=["off"] → {"none"}
        # "medium" not in {"none"} → None
        assert resolve_lmstudio_effort(None, ["off"]) is None

    def test_falsy_allowed_skips_clamping(self):
        """When allowed_options is falsy (None, []), clamping is skipped."""
        assert resolve_lmstudio_effort({"effort": "high"}, None) == "high"
        assert resolve_lmstudio_effort({"effort": "high"}, []) == "high"

    def test_allowed_as_tuple_not_list(self):
        """allowed_options as tuple (not list) — still iterable, clamping works."""
        assert resolve_lmstudio_effort({"effort": "medium"}, ("low", "medium", "high")) == "medium"
        assert resolve_lmstudio_effort({"effort": "high"}, ("low", "medium")) is None


# ---------------------------------------------------------------------------
# Integration-style: enabled=True + effort combinations
# ---------------------------------------------------------------------------
class TestEnabledTrueWithEffort:
    """When enabled=True, effort is taken from the config (not forced to 'none')."""

    def test_enabled_true_effort_high(self):
        assert resolve_lmstudio_effort({"enabled": True, "effort": "high"}, None) == "high"

    def test_enabled_true_effort_default(self):
        """enabled=True but no effort → default 'medium'."""
        assert resolve_lmstudio_effort({"enabled": True}, None) == "medium"

    def test_enabled_true_effort_clamped(self):
        assert resolve_lmstudio_effort(
            {"enabled": True, "effort": "high"}, ["low"]
        ) is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_none_everywhere(self):
        assert resolve_lmstudio_effort(None, None) == "medium"

    def test_both_clamping_layers_together(self):
        """Alias in both reasoning_config AND allowed_options."""
        # effort="off" → "none", allowed=["off","on"] → {"none","medium"}
        # "none" is in {"none","medium"} → "none"
        assert resolve_lmstudio_effort({"effort": "off"}, ["off", "on"]) == "none"

    def test_enabled_false_overrides_effort(self):
        """enabled=False takes priority over any explicit effort."""
        assert resolve_lmstudio_effort(
            {"enabled": False, "effort": "high"}, None
        ) == "none"

    def test_effort_none_value(self):
        """effort=None (Python None, not string 'none')."""
        # reasoning_config.get("effort") returns None, or "" becomes ""
        # which is falsy → falls back to "medium"
        assert resolve_lmstudio_effort({"effort": None}, None) == "medium"  # type: ignore[dict-item]

    def test_allowed_with_unknown_values(self):
        """Unknown values in allowed_options are passed through unchanged."""
        # "medium" (default) is in {"low","weird","medium"} → "medium"
        assert resolve_lmstudio_effort(None, ["low", "weird", "medium"]) == "medium"

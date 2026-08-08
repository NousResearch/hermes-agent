"""Unit tests for hermes_cli.toolset_validation (see #38798).

Pure logic — the validity predicate is injected, so these tests need neither the
tool registry nor a running Hermes.
"""

import pytest

from hermes_cli.toolset_validation import (
    clean_platform_toolsets,
    validate_platform_toolsets,
)

# A representative set of real toolset names. `hermes` is deliberately absent —
# that is the corruption #38798 reported (`hermes-cli` rewritten to `hermes`).
_KNOWN = {
    "hermes-cli",
    "hermes-telegram",
    "hermes-discord",
    "terminal",
    "web",
}


def _is_valid(name):
    return name in _KNOWN




def test_38798_corruption_warns_and_suggests_correct_name():
    # The exact reported shape: cli holds 'hermes' instead of 'hermes-cli'.
    warnings = validate_platform_toolsets({"cli": ["hermes"]}, _is_valid)
    unknown = [w for w in warnings if "unknown toolset 'hermes'" in w]
    assert len(unknown) == 1
    # Actionable: points at the valid name the entry should have been.
    assert "did you mean 'hermes-cli'?" in unknown[0]
    # And the zero-valid-toolsets safety net fires.
    assert any("zero valid toolsets" in w for w in warnings)


def test_mixed_valid_and_invalid_flags_only_the_invalid():
    cfg = {"cli": ["hermes-cli"], "discord": ["bogus"]}
    warnings = validate_platform_toolsets(cfg, _is_valid)
    # One valid entry exists, so no zero-valid warning.
    assert not any("zero valid toolsets" in w for w in warnings)
    assert len(warnings) == 1
    assert "platform 'discord'" in warnings[0]
    assert "unknown toolset 'bogus'" in warnings[0]


def test_clean_platform_toolsets_drops_invalid_entries_and_empty_overrides():
    cfg = {"cli": ["terminal", "bogus"], "discord": ["wrong"], "web": []}

    cleaned, warnings, changed = clean_platform_toolsets(cfg, _is_valid)

    assert changed is True
    assert cleaned == {"cli": ["terminal"], "web": []}
    assert len(warnings) == 2
    assert any("platform 'cli'" in warning for warning in warnings)
    assert any("platform 'discord'" in warning for warning in warnings)


def test_clean_platform_toolsets_can_preserve_dynamic_and_non_removable_names():
    cfg = {"cli": ["mcp-github", "messaging", "future-toolset"]}

    cleaned, warnings, changed = clean_platform_toolsets(
        cfg,
        _is_valid,
        extra_valid_names={"mcp-github"},
        removable_names={"messaging"},
    )

    assert changed is True
    assert cleaned == {"cli": ["mcp-github", "future-toolset"]}
    assert len(warnings) == 2
    assert any("unknown toolset 'messaging'" in warning for warning in warnings)
    assert any("unknown toolset 'future-toolset'" in warning for warning in warnings)


def test_validate_platform_toolsets_accepts_dynamic_valid_names():
    warnings = validate_platform_toolsets(
        {"cli": ["mcp-github"]},
        _is_valid,
        extra_valid_names={"mcp-github"},
    )

    assert warnings == []


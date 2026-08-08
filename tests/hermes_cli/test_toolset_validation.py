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


# ---------------------------------------------------------------------------
# clean_platform_toolsets — auto-cleanup for #76847
# ---------------------------------------------------------------------------

def test_clean_drops_stale_entry_and_keeps_siblings():
    # The #76847 shape: legacy 'messaging' toolset inside the cli list.
    cfg = {"cli": ["web", "messaging", "terminal"]}
    assert clean_platform_toolsets(cfg, _is_valid) is True
    assert cfg == {"cli": ["web", "terminal"]}


def test_clean_removes_platform_key_when_every_entry_is_stale():
    cfg = {"cli": ["messaging"]}
    assert clean_platform_toolsets(cfg, _is_valid) is True
    assert cfg == {}


def test_clean_preserves_all_valid_config():
    cfg = {"cli": ["hermes-cli", "terminal"], "telegram": ["hermes-telegram"]}
    assert clean_platform_toolsets(cfg, _is_valid) is False
    assert cfg == {"cli": ["hermes-cli", "terminal"], "telegram": ["hermes-telegram"]}


def test_clean_handles_scalar_and_non_string_entries():
    # Scalar string entry that is stale is dropped; non-string list items are
    # tolerated exactly like validate_platform_toolsets tolerates them.
    cfg = {"cli": "messaging", "web": ["web", 123, None]}
    assert clean_platform_toolsets(cfg, _is_valid) is True
    assert cfg == {"web": ["web", 123, None]}


def test_clean_is_noop_for_non_dict():
    assert clean_platform_toolsets(None, _is_valid) is False
    assert clean_platform_toolsets("messaging", _is_valid) is False
    assert clean_platform_toolsets([], _is_valid) is False





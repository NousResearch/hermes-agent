"""Unit tests for hermes_cli.toolset_validation (see #38798).

Pure logic — the validity predicate is injected, so these tests need neither the
tool registry nor a running Hermes.
"""

import pytest

from hermes_cli.toolset_validation import validate_platform_toolsets

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






def test_empty_list_on_a_platform_warns_even_when_others_are_valid():
    # The #89050 shape: the active platform is wiped to [] while every other
    # platform stays populated. The global zero-valid-toolsets net does not fire
    # (telegram/discord are valid), so without a per-platform check this config
    # produces no warning at all and the agent silently starts with no tools.
    cfg = {
        "cli": [],
        "telegram": ["hermes-telegram"],
        "discord": ["hermes-discord"],
    }
    warnings = validate_platform_toolsets(cfg, _is_valid)

    empty = [w for w in warnings if "empty toolset list" in w]
    assert len(empty) == 1
    assert "platform 'cli'" in empty[0]
    assert "no tools" in empty[0]
    # Populated platforms must not be implicated.
    assert "telegram" not in empty[0]
    # The global net is genuinely suppressed here — the per-platform warning is
    # the only thing standing between the user and a silent zero-tool agent.
    assert not any("zero valid toolsets" in w for w in warnings)


def test_empty_list_warns_for_each_affected_platform():
    cfg = {"cli": [], "discord": [], "telegram": ["hermes-telegram"]}
    warnings = validate_platform_toolsets(cfg, _is_valid)

    empty = [w for w in warnings if "empty toolset list" in w]
    assert len(empty) == 2
    assert {"cli", "discord"} == {
        p for p in ("cli", "discord") if any(f"platform '{p}'" in w for w in empty)
    }


def test_empty_list_does_not_mask_unknown_names_on_other_platforms():
    cfg = {"cli": [], "discord": ["bogus"]}
    warnings = validate_platform_toolsets(cfg, _is_valid)

    assert any("empty toolset list" in w and "platform 'cli'" in w for w in warnings)
    assert any("unknown toolset 'bogus'" in w for w in warnings)
    # Nothing valid anywhere, so the global net still fires too.
    assert any("zero valid toolsets" in w for w in warnings)


def test_populated_platforms_produce_no_empty_list_warning():
    cfg = {"cli": ["hermes-cli"], "telegram": ["hermes-telegram"]}
    warnings = validate_platform_toolsets(cfg, _is_valid)
    assert warnings == []

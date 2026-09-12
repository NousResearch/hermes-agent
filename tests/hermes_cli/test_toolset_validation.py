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
    "hermes-whatsapp",
    "discord",
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
    assert any(
        "platform 'discord'" in w and "unknown toolset 'bogus'" in w
        for w in warnings
    )
    assert any(
        "platform 'discord'" in w and "no valid toolsets" in w
        for w in warnings
    )






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


def test_null_platform_warns_even_when_others_are_valid():
    # YAML's ``cli:`` parses as None. The resolver treats it as absent and uses
    # the platform default, so the warning must not claim that tools disappear.
    cfg = {"cli": None, "telegram": ["hermes-telegram"]}
    warnings = validate_platform_toolsets(cfg, _is_valid)

    assert any(
        "platform 'cli'" in w
        and "null toolset value" in w
        and "falling back to 'hermes-cli'" in w
        for w in warnings
    )
    assert not any("zero valid toolsets" in w for w in warnings)


def test_null_platform_uses_canonical_default_for_alias():
    cfg = {"whatsapp_cloud": None, "telegram": ["hermes-telegram"]}
    warnings = validate_platform_toolsets(cfg, _is_valid)

    assert any(
        "platform 'whatsapp_cloud'" in w
        and "falling back to 'hermes-whatsapp'" in w
        for w in warnings
    )
    assert not any("zero valid toolsets" in w for w in warnings)


def test_scalar_platform_value_warns_but_uses_platform_default():
    cfg = {"cli": "bogus", "telegram": ["hermes-telegram"]}
    warnings = validate_platform_toolsets(cfg, _is_valid)

    assert any(
        "platform 'cli'" in w
        and "invalid toolset value 'bogus'" in w
        and "falling back to 'hermes-cli'" in w
        for w in warnings
    )
    assert not any("zero valid toolsets" in w for w in warnings)


def test_platform_restricted_toolset_warns_when_other_platform_is_valid():
    cfg = {"telegram": ["discord"], "cli": ["hermes-cli"]}
    warnings = validate_platform_toolsets(cfg, _is_valid)

    assert any(
        "platform 'telegram'" in w
        and "toolset 'discord'" in w
        and "not available" in w
        for w in warnings
    )
    assert not any("zero valid toolsets" in w for w in warnings)


def test_null_plugin_platform_uses_synthetic_default():
    from gateway.platform_registry import PlatformEntry, platform_registry
    from toolsets import resolve_toolset

    platform = "toolset_validation_plugin"
    platform_registry.register(
        PlatformEntry(
            name=platform,
            label="Toolset Validation Plugin",
            adapter_factory=lambda _config: object(),
            check_fn=lambda: True,
        )
    )
    try:
        cfg = {platform: None, "telegram": ["hermes-telegram"]}
        warnings = validate_platform_toolsets(cfg, _is_valid)

        assert resolve_toolset(f"hermes-{platform}")
        assert any(
            f"platform '{platform}'" in w
            and f"falling back to 'hermes-{platform}'" in w
            for w in warnings
        )
        assert not any("zero valid toolsets" in w for w in warnings)
    finally:
        platform_registry.unregister(platform)


def test_all_invalid_platform_warns_even_when_others_are_valid():
    cfg = {"cli": ["bogus"], "telegram": ["hermes-telegram"]}
    warnings = validate_platform_toolsets(cfg, _is_valid)

    assert any("unknown toolset 'bogus'" in w for w in warnings)
    assert any(
        "platform 'cli'" in w and "no valid toolsets" in w for w in warnings
    )
    assert not any("zero valid toolsets" in w for w in warnings)


def test_populated_platforms_produce_no_empty_list_warning():
    cfg = {"cli": ["hermes-cli"], "telegram": ["hermes-telegram"]}
    warnings = validate_platform_toolsets(cfg, _is_valid)
    assert warnings == []


# --- MCP-awareness (#78102): platform_toolsets legitimately holds MCP server names ---
# (per-platform MCP allowlist; see tools_config._merge_mcp_servers / _save_platform_tools).


def test_bare_mcp_server_name_accepted_via_predicate():
    cfg = {"telegram": ["linear", "web"]}
    warnings = validate_platform_toolsets(
        cfg, _is_valid, is_valid_mcp_server=lambda n: n in {"linear"})
    # 'linear' is not a toolset, but the MCP predicate vouches for it: no warning at all.
    assert not any("telegram" in w for w in warnings)


def test_prefixed_mcp_server_name_accepted_via_predicate():
    cfg = {"cli": ["mcp-codegraph", "hermes-cli"]}
    warnings = validate_platform_toolsets(
        cfg, _is_valid, is_valid_mcp_server=lambda n: n in {"mcp-codegraph"})
    assert not any("unknown toolset 'mcp-codegraph'" in w for w in warnings)
    assert not any("zero valid toolsets" in w for w in warnings)


def test_mcp_only_platform_list_does_not_trip_zero_valid_safety_net():
    # Both names are MCP allowlist entries, neither is a toolset: the platform has tools
    # (the MCP allowlist), so neither the per-platform nor the global net may fire.
    cfg = {"telegram": ["linear", "fellow"]}
    warnings = validate_platform_toolsets(
        cfg, _is_valid, is_valid_mcp_server=lambda n: n in {"linear", "fellow"})
    assert warnings == []


def test_mixed_list_flags_only_the_genuinely_unknown_name():
    cfg = {"cli": ["hermes-cli", "linear", "bogus"]}
    warnings = validate_platform_toolsets(
        cfg, _is_valid, is_valid_mcp_server=lambda n: n in {"linear"})
    assert [w for w in warnings if "unknown toolset" in w] == [
        w for w in warnings if "'bogus'" in w
    ]
    assert len(warnings) == 1


def test_mcp_names_still_warn_without_predicate():
    # Legacy callers pass no predicate (explicit None = the default): behavior is
    # unchanged, MCP names keep warning.
    cfg = {"telegram": ["linear"]}
    warnings = validate_platform_toolsets(cfg, _is_valid, is_valid_mcp_server=None)
    assert any("unknown toolset 'linear'" in w for w in warnings)


def test_config_validator_accepts_mcp_names_and_no_mcp_sentinel(monkeypatch):
    # Caller-level: _warn_invalid_platform_toolsets must build the MCP predicate from the
    # raw config, so bare MCP names and the no_mcp sentinel pass without warnings.
    import hermes_cli.config as config_mod

    cfg = {
        "mcp_servers": {
            "linear": {"transport": "stdio"},
            "fellow": {"transport": "stdio"},
        },
        "platform_toolsets": {"telegram": ["web", "linear", "fellow", "no_mcp"]},
    }
    monkeypatch.setattr(config_mod, "read_raw_config", lambda: cfg)
    results = {"warnings": []}
    config_mod._warn_invalid_platform_toolsets(results, quiet=True)
    assert results["warnings"] == []


def test_config_validator_still_flags_bogus_name_alongside_mcp_names(monkeypatch):
    # The predicate widens acceptance, never blinds it: an unknown name still warns.
    import hermes_cli.config as config_mod

    cfg = {
        "mcp_servers": {
            "linear": {"transport": "stdio"},
            "fellow": {"transport": "stdio"},
        },
        "platform_toolsets": {"telegram": ["web", "linear", "fellow", "no_mcp", "bogus"]},
    }
    monkeypatch.setattr(config_mod, "read_raw_config", lambda: cfg)
    results = {"warnings": []}
    config_mod._warn_invalid_platform_toolsets(results, quiet=True)
    assert len(results["warnings"]) == 1
    assert "unknown toolset 'bogus'" in results["warnings"][0]


def test_config_validator_flags_mcp_name_of_disabled_server(monkeypatch):
    # A disabled MCP server's name is not in the allowlist: the warning is genuinely
    # useful there (its tools are silently absent at runtime), so it must fire.
    import hermes_cli.config as config_mod

    cfg = {
        "mcp_servers": {
            "linear": {"transport": "stdio", "enabled": False},
        },
        "platform_toolsets": {"telegram": ["linear"]},
    }
    monkeypatch.setattr(config_mod, "read_raw_config", lambda: cfg)
    results = {"warnings": []}
    config_mod._warn_invalid_platform_toolsets(results, quiet=True)
    # The unknown-name warning fires (plus the zero-valid safety nets, since nothing
    # in the list is valid) — the disabled server is not vouched for.
    assert any("unknown toolset 'linear'" in w for w in results["warnings"])

"""Boundary contracts for malformed ``platform_toolsets`` list members."""

import pytest
import yaml

from hermes_cli.toolset_validation import validate_platform_toolsets


@pytest.mark.parametrize("invalid_member", [None, 42], ids=["null", "integer"])
def test_malformed_list_member_is_reported_alongside_a_valid_toolset(invalid_member):
    """Every ignored list member stays visible even when another member is valid."""
    warnings = validate_platform_toolsets(
        {"cli": ["hermes-cli", invalid_member]},
        lambda name: name == "hermes-cli",
    )

    assert any(
        "platform 'cli'" in warning
        and "invalid" in warning
        and "toolset value" in warning
        for warning in warnings
    )
    assert not any("no valid toolsets" in warning for warning in warnings)


@pytest.mark.parametrize("invalid_member", [None, 42], ids=["null", "integer"])
def test_real_config_warning_path_matches_runtime_resolution(
    invalid_member, tmp_path, monkeypatch
):
    """Raw YAML validation exposes every malformed member that runtime retains."""
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"platform_toolsets": {"cli": ["hermes-cli", invalid_member]}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from hermes_cli.config import _warn_invalid_platform_toolsets, read_raw_config
    from hermes_cli.tools_config import _get_platform_tools

    results = {"warnings": []}
    _warn_invalid_platform_toolsets(results, quiet=True)

    assert any(
        "platform 'cli'" in warning
        and "invalid" in warning
        and "toolset value" in warning
        for warning in results["warnings"]
    )
    assert not any("no valid toolsets" in warning for warning in results["warnings"])

    enabled = _get_platform_tools(
        read_raw_config(), "cli", include_default_mcp_servers=False
    )
    assert str(invalid_member) in enabled
    assert "web" in enabled


def test_list_literal_parser_path_remains_validated_as_a_list():
    """The later list-literal compatibility parser still validates decoded names."""
    warnings = validate_platform_toolsets(
        {"cli": '["web", "terminal"]'},
        lambda name: name in {"web", "terminal"},
    )

    assert warnings == []

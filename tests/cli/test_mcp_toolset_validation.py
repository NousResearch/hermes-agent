from copy import deepcopy

import cli


def _init_cli_and_capture_warnings(monkeypatch, toolsets):
    cfg = deepcopy(cli.CLI_CONFIG)
    cfg["model"] = {"default": "stub-model", "provider": "auto", "base_url": ""}
    cfg.setdefault("agent", {})["disabled_toolsets"] = []
    cfg.setdefault("display", {})
    cfg["mcp_servers"] = {
        "agentmail": {"enabled": True},
        "notion-findit": {"enabled": True},
    }

    warnings = []
    monkeypatch.setattr(cli, "CLI_CONFIG", cfg)
    monkeypatch.setattr(cli, "validate_toolset", lambda name: name == "terminal")
    monkeypatch.setattr(
        cli.HermesCLI,
        "_console_print",
        lambda self, message, *args, **kwargs: warnings.append(str(message)),
    )

    cli.HermesCLI(
        model="stub-model",
        provider="auto",
        toolsets=toolsets,
        compact=True,
    )
    return warnings


def test_cli_startup_accepts_mcp_prefixed_toolsets_before_discovery(monkeypatch):
    warnings = _init_cli_and_capture_warnings(
        monkeypatch,
        ["terminal", "mcp-agentmail", "mcp-notion-findit"],
    )

    assert warnings == []


def test_cli_startup_still_warns_for_non_mcp_unknown_toolsets(monkeypatch):
    warnings = _init_cli_and_capture_warnings(
        monkeypatch,
        ["terminal", "mcp-agentmail", "not-real"],
    )

    assert warnings == ["[bold red]Warning: Unknown toolsets: not-real[/]"]


def test_mcp_toolset_aliases_skips_non_mapping_entries():
    """Non-mapping server entries are skipped to avoid validating toolsets
    that discovery cannot register."""
    from hermes_cli.mcp_toolsets import mcp_toolset_aliases_for_servers

    result = mcp_toolset_aliases_for_servers({
        "good": {"enabled": True},
        "bad": "just-a-string",
        "also-bad": 42,
    })
    assert result == {"good", "mcp-good"}


def test_split_configured_mcp_toolset_aliases_skips_non_mapping():
    from hermes_cli.mcp_toolsets import split_configured_mcp_toolset_aliases

    enabled, disabled = split_configured_mcp_toolset_aliases({
        "active": {"enabled": True},
        "inactive": {"enabled": False},
        "malformed": "not-a-dict",
    })
    assert enabled == {"active", "mcp-active"}
    assert disabled == {"inactive", "mcp-inactive"}

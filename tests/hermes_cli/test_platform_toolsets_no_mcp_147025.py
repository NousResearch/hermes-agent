"""Tests for #147025: an empty ``platform_toolsets.<platform>`` list is not a lockdown.

``_merge_mcp_servers``' include-all fallback hands a platform with an explicitly saved empty
list every globally enabled MCP server, so "no built-in toolsets" silently means "87 MCP tools
reachable from an untrusted channel". The reader-side validator and the session-side resolver
must both say what the platform actually gets, and the ``no_mcp`` sentinel — the correct
lockdown — must resolve to zero tools without a false "unknown toolset" warning.
"""

import logging

import hermes_cli.tools_config as _tc
from hermes_cli.toolset_validation import validate_platform_toolsets

_KNOWN = {"hermes-cli", "hermes-email", "clarify", "terminal"}


def _is_valid(name):
    return name in _KNOWN


_MCP = {"cua-driver", "evernote"}


def _inherited(_platform):
    return sorted(_MCP)


# ── validator (config check / config migrate) ──────────────────────────────────


def test_empty_list_with_inherited_mcp_names_the_servers_and_sends_no_mcp():
    warnings = validate_platform_toolsets(
        {"email": []}, _is_valid, inherited_mcp_servers=_inherited)
    empty = [w for w in warnings if "empty toolset list" in w]
    assert len(empty) == 1
    # The old message claimed "no tools" while the platform really inherited 87 MCP tools.
    assert "no tools on this platform" not in empty[0]
    assert "cua-driver" in empty[0] and "evernote" in empty[0]
    assert "no_mcp" in empty[0]


def test_empty_list_without_mcp_still_claims_no_tools():
    warnings = validate_platform_toolsets(
        {"email": []}, _is_valid, inherited_mcp_servers=lambda _p: [])
    empty = [w for w in warnings if "empty toolset list" in w]
    assert len(empty) == 1
    assert "no tools on this platform" in empty[0]


def test_empty_list_without_the_callable_keeps_the_old_message():
    # Callers that do not surface an MCP view (pure tests, injected validators) keep
    # the historical wording untouched.
    warnings = validate_platform_toolsets({"email": []}, _is_valid)
    empty = [w for w in warnings if "empty toolset list" in w]
    assert len(empty) == 1
    assert "no tools on this platform" in empty[0]


def test_inherited_mcp_servers_failure_degrades_to_old_message():
    def _boom(_platform):
        raise RuntimeError("secret backend down")
    warnings = validate_platform_toolsets(
        {"email": []}, _is_valid, inherited_mcp_servers=_boom)
    empty = [w for w in warnings if "empty toolset list" in w]
    assert len(empty) == 1
    assert "no tools on this platform" in empty[0]


def test_no_mcp_sentinel_produces_no_warnings():
    # The correct lockdown must not trip the false "unknown toolset" / "no valid
    # toolsets" warnings the issue reported alongside the empty list. Uses the
    # real wiring (saved resolver), where the sentinel is recognized.
    from hermes_cli.toolset_validation import platform_toolset_diagnostics
    for locked in (["no_mcp"], ["clarify", "no_mcp"]):
        config = {"mcp_servers": {"cua-driver": {"command": "x", "enabled": True}},
                  "platform_toolsets": {"email": locked}}
        warnings = platform_toolset_diagnostics(config)
        assert not [w for w in warnings if "unknown toolset" in w and "no_mcp" in w], warnings
        assert not [w for w in warnings if "no valid toolsets" in w], warnings


# ── runtime resolution (_get_platform_tools) ───────────────────────────────────


_CONFIG = {
    "mcp_servers": {
        "cua-driver": {"command": "cua-driver", "enabled": True},
        "evernote": {"command": "evernote", "enabled": True},
    },
    "platform_toolsets": {"email": []},
}


def test_runtime_empty_list_warns_once_per_platform(caplog):
    _tc._warned_empty_platform_mcp.discard("email")
    with caplog.at_level(logging.WARNING, logger="hermes_cli.tools_config"):
        _tc._get_platform_tools(_CONFIG, "email")
        _tc._get_platform_tools(_CONFIG, "email")
    messages = [r.getMessage() for r in caplog.records
                if r.levelno >= logging.WARNING and "#147025" in r.getMessage()]
    assert len(messages) == 1, messages
    assert "cua-driver" in messages[0] and "evernote" in messages[0]
    assert "no_mcp" in messages[0]


def test_runtime_no_mcp_sentinel_resolves_zero_and_stays_silent(caplog):
    _tc._warned_empty_platform_mcp.discard("email")
    config = {"mcp_servers": _CONFIG["mcp_servers"], "platform_toolsets": {"email": ["no_mcp"]}}
    with caplog.at_level(logging.WARNING, logger="hermes_cli.tools_config"):
        resolved = _tc._get_platform_tools(config, "email")
    assert not [r for r in caplog.records
                if r.levelno >= logging.WARNING and "#147025" in r.getMessage()]
    # No MCP server names survived the no_mcp sentinel.
    assert not resolved & {"cua-driver", "evernote", "mcp-cua-driver", "mcp-evernote"}


def test_runtime_nonempty_list_without_no_mcp_still_inherits(caplog):
    _tc._warned_empty_platform_mcp.discard("email")
    config = {"mcp_servers": _CONFIG["mcp_servers"], "platform_toolsets": {"email": ["clarify"]}}
    with caplog.at_level(logging.WARNING, logger="hermes_cli.tools_config"):
        resolved = _tc._get_platform_tools(config, "email")
    assert not [r for r in caplog.records
                if r.levelno >= logging.WARNING and "#147025" in r.getMessage()]
    # Documented include-all fallback: a non-empty list with no server names still
    # inherits every enabled server — only ``no_mcp`` excludes them.
    assert resolved & {"cua-driver", "evernote"} == {"cua-driver", "evernote"}


def test_runtime_disabled_servers_only_warn_for_enabled(caplog):
    _tc._warned_empty_platform_mcp.discard("email")
    config = {
        "mcp_servers": {
            "cua-driver": {"command": "cua-driver", "enabled": True},
            "evernote": {"command": "evernote", "enabled": False},
        },
        "platform_toolsets": {"email": []},
    }
    with caplog.at_level(logging.WARNING, logger="hermes_cli.tools_config"):
        _tc._get_platform_tools(config, "email")
    messages = [r.getMessage() for r in caplog.records
                if r.levelno >= logging.WARNING and "#147025" in r.getMessage()]
    assert len(messages) == 1
    assert "cua-driver" in messages[0]
    assert "evernote" not in messages[0]

"""Regression tests for removed dead config keys.

This file guards against accidental re-introduction of config keys that were
documented or declared at some point but never actually wired up to read code.
Future dead-config regressions can accumulate here.
"""


def test_delegation_default_toolsets_removed_from_cli_config():
    """delegation.default_toolsets was dead config — never read by
    _load_config() or anywhere else. Removed in M0.5.

    Guards against accidental re-introduction in cli.py's CLI_CONFIG default
    dict. If this test fails, someone re-added the key without wiring it up
    to _load_config() in tools/delegate_tool.py.
    """
    from cli import CLI_CONFIG

    delegation_cfg = CLI_CONFIG.get("delegation", {})
    assert "default_toolsets" not in delegation_cfg, (
        "delegation.default_toolsets was removed in M0.5 because it was "
        "never read. Do not re-add it; use tools/delegate_tool.py's "
        "DEFAULT_TOOLSETS module constant or wire a new config key through "
        "_load_config()."
    )

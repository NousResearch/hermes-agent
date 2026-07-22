"""Regression tests for removed dead config keys.

This file guards against accidental re-introduction of config keys that were
documented or declared at some point but never actually wired up to read code.
Future dead-config regressions can accumulate here.
"""

import inspect


def test_delegation_default_toolsets_removed_from_cli_config():
    """delegation.default_toolsets was dead config — never read by
    _load_config() or anywhere else. Removed.

    Guards against accidental re-introduction in cli.py's CLI_CONFIG default
    dict. If this test fails, someone re-added the key without wiring it up
    to _load_config() in tools/delegate_tool.py.

    We inspect the source of load_cli_config() instead of asserting on the
    runtime CLI_CONFIG dict because CLI_CONFIG is populated by deep-merging
    the user's ~/.hermes/config.yaml over the defaults (cli.py:359-366).
    A contributor who still has the legacy key set in their own config
    would cause a false failure, and HERMES_HOME patching via conftest
    doesn't help because cli._hermes_home is frozen at module import time
    (cli.py:76) — before any autouse fixture can fire. Source inspection
    sidesteps all of that: it tests the defaults literal directly.
    """
    from cli import load_cli_config

    source = inspect.getsource(load_cli_config)
    assert '"default_toolsets"' not in source, (
        "delegation.default_toolsets was removed because it was never read. "
        "Do not re-add it to cli.py's CLI_CONFIG default dict; "
        "use tools/delegate_tool.py's DEFAULT_TOOLSETS module constant or "
        "wire a new config key through _load_config()."
    )


def test_delegation_include_tool_trace_declared_and_wired():
    """delegation.include_tool_trace must stay declared in BOTH default
    config blocks (hermes_cli/config.py DEFAULT_CONFIG and cli.py's legacy
    load_cli_config() defaults — _load_config() in tools/delegate_tool.py can
    fall back to either) with the same default (True), and must actually be
    read by tools/delegate_tool.py so it never becomes dead config.

    cli.py is checked via source inspection for the same reason as the test
    above: CLI_CONFIG is deep-merged with the contributor's own
    ~/.hermes/config.yaml at import time. DEFAULT_CONFIG is a plain module
    constant, so a runtime assert is safe there.
    """
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["delegation"]["include_tool_trace"] is True, (
        "delegation.include_tool_trace default must be True (include the "
        "trace) in hermes_cli/config.py's DEFAULT_CONFIG."
    )

    from cli import load_cli_config

    source = inspect.getsource(load_cli_config)
    assert '"include_tool_trace": True' in source, (
        "delegation.include_tool_trace missing (or default changed) in "
        "cli.py's legacy CLI_CONFIG defaults — keep it mirrored with "
        "hermes_cli/config.py DEFAULT_CONFIG (default True)."
    )

    from tools.delegate_tool import _get_include_tool_trace

    reader_source = inspect.getsource(_get_include_tool_trace)
    assert '"include_tool_trace"' in reader_source, (
        "tools/delegate_tool.py no longer reads delegation.include_tool_trace"
        " — the config key would be dead. Wire it back through "
        "_get_include_tool_trace()/_load_config() or remove it everywhere."
    )

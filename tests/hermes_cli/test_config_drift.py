"""Regression tests for removed dead config keys.

This file guards against accidental re-introduction of config keys that were
documented or declared at some point but never actually wired up to read code.
Future dead-config regressions can accumulate here.
"""

import inspect


def test_delegation_default_toolsets_removed_from_cli_config():
    """delegation.default_toolsets was dead config — never read by
    _load_config() or anywhere else. Removed in M0.5.

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

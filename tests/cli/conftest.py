"""Shared fixtures for tests/cli.

Several test modules here build a HermesCLI via ``importlib.reload(cli)``
with prompt_toolkit stubbed out in ``sys.modules`` (the ``_make_cli``
pattern from test_cli_init.py). ``importlib.reload()`` re-executes cli.py
into the SAME module dict, so those MagicMock bindings (``_pt_print``,
``_PT_ANSI``, ...) survive the ``patch.dict`` context and silently break
any later test that needs cli's real prompt_toolkit machinery — e.g.
``cli._cprint`` output vanishes into a MagicMock and capsys sees nothing.

The autouse fixture below restores the real bindings at each module
boundary by re-reloading cli (with the real prompt_toolkit back in
sys.modules) whenever the pollution is detected.
"""

import importlib
import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True, scope="module")
def _unpollute_cli_module():
    yield
    cli_mod = sys.modules.get("cli")
    if cli_mod is not None and isinstance(
        getattr(cli_mod, "_pt_print", None), MagicMock
    ):
        importlib.reload(cli_mod)

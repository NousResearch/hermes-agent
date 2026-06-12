"""Fork-specific tests for Kamell's patched-main.

These test durable fork behavior that lives outside upstream's test suite.
Keep this file small — only tests for patches that aren't (yet) upstream.
"""

import importlib
import sys


def _main_mod():
    """Import hermes_cli.main fresh to avoid stale module state."""
    if "hermes_cli.main" in sys.modules:
        return sys.modules["hermes_cli.main"]
    return importlib.import_module("hermes_cli.main")


# ── _tui_initial_skin_env ──────────────────────────────────────────


def test_tui_initial_skin_env_serializes_configured_skin() -> None:
    raw = _main_mod()._tui_initial_skin_env({"display": {"skin": "mono"}})

    assert raw
    assert '"name":"mono"' in raw
    assert '"colors"' in raw
    assert '"branding"' in raw

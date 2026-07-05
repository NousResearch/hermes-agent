"""Shared helpers for the apex_overlay seam tests.

The plugin-register tests execute the bundled apex-overlay plugin's
``register()``. That hook applies EVERY seam — so a per-file test that mocked
only its own seam's ``apply`` would let the *other* seams monkey-patch the
process for real, contaminating later test files when the whole directory runs
in one interpreter (the ``pytest tests/apex_overlay`` gate). This helper stubs
all seam ``apply()``s and reports which ones register() invoked.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

_SEAM_MODULES = (
    "provider_filter",
    "models_dev_fast",
    "gateway_bootstrap",
    "feishu_supervisor",
    # hc-401 seams (ported cloud build-time patches).
    "web_tools_extract",
    "stt_no_lazy_install",
    "first_turn_ack",
    "cn_im_messages",
)


def run_plugin_register_with_stubbed_seams(module_alias: str) -> set[str]:
    """Execute the apex-overlay plugin's register() with every seam stubbed.

    Returns the set of seam module names whose ``apply()`` register() called.
    ``module_alias`` keeps each caller's synthetic plugin-module name unique.
    """
    import apex_overlay

    plugin_init = (
        Path(__file__).resolve().parents[2]
        / "plugins" / "apex-overlay" / "__init__.py"
    )
    assert plugin_init.exists(), "apex-overlay plugin __init__.py missing"

    spec = importlib.util.spec_from_file_location(module_alias, plugin_init)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "register"), "plugin must expose register(ctx)"

    called: set[str] = set()
    patches = []
    for name in _SEAM_MODULES:
        seam_mod = __import__(f"apex_overlay.{name}", fromlist=[name])
        patches.append(
            patch.object(
                seam_mod, "apply",
                (lambda n: (lambda: called.add(n) or True))(name),
            )
        )
    try:
        for p in patches:
            p.start()
        mod.register(ctx=None)
    finally:
        for p in patches:
            p.stop()
    return called

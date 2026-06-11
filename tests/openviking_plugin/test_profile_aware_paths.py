"""Verify that OpenViking plugin state paths honor the HERMES_HOME env var
when the modules are imported under an active profile.

This guards against regression of the egilewski review blocker
(reported 2026-06-10 22:46 UTC): the new finalizer/registry code was
hardcoding ~/.hermes instead of the active profile, so a Codex review
run with HERMES_HOME set to a sandbox path still saw registry paths
in the user's real ~/.hermes/ directory.

The fix was to replace os.path.expanduser("~/.hermes") and ~/
literals with str(get_hermes_home()), which honors HERMES_HOME. But
because the constants are captured at import time, we MUST re-import
the modules under the new env to verify the path actually changed.
"""
import importlib
import os
import sys

import pytest


def _import_ov_module(name, hermes_home):
    """Force a clean import of an OpenViking module under HERMES_HOME.

    Uses a fresh subprocess-via-import dance:
      1. Save current modules of the plugin
      2. Set HERMES_HOME in os.environ
      3. Re-import — module-level constants re-evaluate

    NOTE: We rely on hermes_constants.get_hermes_home() reading
    os.environ at call time, AND on the plugin's top-level line
    ``_HERMES_HOME = str(get_hermes_home())`` re-running on import.
    Python caches imported modules in sys.modules, so we MUST
    evict them first.
    """
    old_home = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = hermes_home
    # Evict the plugin's modules AND hermes_constants (since the
    # plugin imports `from hermes_constants import get_hermes_home`).
    to_evict = [
        k for k in list(sys.modules)
        if k.startswith("plugins.memory.openviking")
        or k == "hermes_constants"
        or k == "agent.redact"
    ]
    for k in to_evict:
        sys.modules.pop(k, None)
    try:
        mod = importlib.import_module(name)
        return mod
    finally:
        if old_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = old_home


def test_registry_session_db_path_hermes_home_override(tmp_path):
    """registry._SESSION_DB_PATH must point under HERMES_HOME."""
    sandbox = tmp_path / "sandbox_home"
    sandbox.mkdir()
    reg = _import_ov_module(
        "plugins.memory.openviking.registry", str(sandbox)
    )
    assert reg._SESSION_DB_PATH.startswith(str(sandbox) + os.sep), (
        f"registry._SESSION_DB_PATH = {reg._SESSION_DB_PATH!r} is not "
        f"under HERMES_HOME={sandbox!r}"
    )
    assert reg._SESSION_DB_PATH.endswith("openviking-sessions.db"), (
        f"registry._SESSION_DB_PATH = {reg._SESSION_DB_PATH!r} lost its filename"
    )


def test_finalizer_paths_hermes_home_override(tmp_path):
    """finalizer._STATE_DB, _REPAIR_DIR, _RECOVERY_DIR must point under HERMES_HOME."""
    sandbox = tmp_path / "sandbox_home_2"
    sandbox.mkdir()
    fin = _import_ov_module(
        "plugins.memory.openviking.finalizer", str(sandbox)
    )
    assert fin._STATE_DB.startswith(str(sandbox) + os.sep)
    assert fin._REPAIR_DIR.startswith(str(sandbox) + os.sep)
    assert fin._RECOVERY_DIR.startswith(str(sandbox) + os.sep)
    assert fin._STATE_DB.endswith("state.db")
    assert fin._REPAIR_DIR.endswith("openviking-repair")
    assert fin._RECOVERY_DIR.endswith("openviking-recovery")

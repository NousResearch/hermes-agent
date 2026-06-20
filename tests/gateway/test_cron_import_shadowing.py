"""Test that cron.scheduler_provider imports are not shadowed by plugins/cron/.

Platform adapters (discord, raft) insert ``plugins/`` into ``sys.path[0]``
at import time.  This shadows the real ``cron/`` package with
``plugins/cron/`` (the provider discovery module), which does NOT contain
``scheduler_provider.py``.  If ``cron.scheduler_provider`` is imported *after*
the adapters pollute sys.path, Python raises ``ModuleNotFoundError``.

The fix: import ``cron.scheduler_provider`` at module level in
``gateway/run.py``, before any ``gateway.platforms.*`` import that can
trigger adapter loading.  This ensures the correct module is cached in
``sys.modules`` first.

See: https://github.com/NousResearch/hermes-agent/issues/49410
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PLUGINS_DIR = REPO_ROOT / "plugins"


class TestCronImportShadowing:
    """Verify that cron imports succeed even after plugins/ hits sys.path."""

    def test_cron_scheduler_provider_importable_after_plugins_path_insert(self):
        """After inserting plugins/ into sys.path (as adapters do),
        ``from cron.scheduler_provider import resolve_cron_scheduler``
        must still resolve to the real module, not fail with
        ModuleNotFoundError because plugins/cron/ shadows cron/."""

        # Ensure the real cron module is cached first (as our fix does).
        from cron.scheduler_provider import resolve_cron_scheduler, InProcessCronScheduler

        # Now simulate what platform adapters do at import time:
        # insert plugins/ at sys.path[0].
        plugins_path = str(PLUGINS_DIR)
        inserted = False
        if plugins_path not in sys.path:
            sys.path.insert(0, plugins_path)
            inserted = True

        try:
            # This must NOT raise ModuleNotFoundError.
            # If cron.scheduler_provider is already in sys.modules,
            # Python reuses the cached module — no shadow.
            from cron.scheduler_provider import resolve_cron_scheduler as rcs
            assert callable(rcs), "resolve_cron_scheduler should be callable"

            # Verify we got the real module, not the plugins/cron package.
            import cron.scheduler_provider as csp
            assert hasattr(csp, "InProcessCronScheduler"), (
                "cron.scheduler_provider should have InProcessCronScheduler"
            )
            assert hasattr(csp, "resolve_cron_scheduler"), (
                "cron.scheduler_provider should have resolve_cron_scheduler"
            )
        finally:
            if inserted:
                sys.path.remove(plugins_path)

    def test_plugins_cron_does_not_have_scheduler_provider(self):
        """Confirm that plugins/cron/ (the shadow package) does NOT contain
        scheduler_provider — this is the root cause of the crash."""

        # plugins/cron/__init__.py exists but has no scheduler_provider submodule
        plugins_cron_init = PLUGINS_DIR / "cron" / "__init__.py"
        assert plugins_cron_init.exists(), "plugins/cron/__init__.py should exist"

        plugins_cron_sp = PLUGINS_DIR / "cron" / "scheduler_provider.py"
        assert not plugins_cron_sp.exists(), (
            "plugins/cron/ should NOT contain scheduler_provider.py"
        )

    def test_module_level_import_before_platform_base(self):
        """Verify that gateway/run.py imports cron.scheduler_provider at
        module level BEFORE gateway.platforms.base (which triggers adapter
        loading that mutates sys.path)."""

        import gateway.run as gw_mod
        source = inspect_getsource(gw_mod)

        # Find the line numbers of key imports
        lines = source.splitlines()
        cron_import_line = None
        platform_base_import_line = None

        for i, line in enumerate(lines):
            stripped = line.strip()
            if "from cron.scheduler_provider import" in stripped:
                cron_import_line = i
            if "from gateway.platforms.base import" in stripped:
                platform_base_import_line = i

        assert cron_import_line is not None, (
            "gateway/run.py must have a module-level import of cron.scheduler_provider"
        )
        assert platform_base_import_line is not None, (
            "gateway/run.py must import gateway.platforms.base"
        )
        assert cron_import_line < platform_base_import_line, (
            f"cron.scheduler_provider import (line {cron_import_line}) must come "
            f"before gateway.platforms.base import (line {platform_base_import_line}) "
            f"to prevent sys.path shadowing"
        )


def inspect_getsource(module):
    """Get module source without requiring the inspect module to re-read."""
    import inspect
    return inspect.getsource(module)
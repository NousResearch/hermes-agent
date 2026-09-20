"""Regression tests for the ``tools_config`` leg of the post-update config reload.

``hermes update`` runs config migrations in the updater process — the PRE-pull
process — after refreshing the config modules from the pulled tree. Migrations
import ``hermes_cli.tools_config`` helpers at call time, so a ``tools_config``
cached in ``sys.modules`` from before the pull makes the import fail with
``ImportError`` even though the symbol exists in the freshly-written file.

History:
- #111271: the v45 migration (``_migrate_to_45``) imports ``_configurable_keys``
  from ``hermes_cli.tools_config``; on a v0.20.6 → v0.21.3 update the cached
  pre-pull module lacked that newly-added symbol, the migration aborted with
  "cannot import name '_configurable_keys'", and the config silently stayed at
  the old version while the code and the gateway moved on (update exit 1).
"""

from __future__ import annotations

import sys

import pytest

_RELOADED_MODULES = (
    "hermes_cli.config_defaults", "hermes_cli.config", "hermes_cli.tools_config",
    "hermes_cli.config_migrations", "hermes_cli._subprocess_compat", "hermes_cli.dashboard_procs")


@pytest.fixture(autouse=True)
def _restore_config_modules_after_reload():
    """Restore the config-family module namespaces after these tests.

    ``importlib.reload()`` re-executes a module IN PLACE (same module object,
    so restoring ``sys.modules`` entries is useless) and rebinds its top-level
    names — classes like ``hermes_cli.config.InvalidUserConfigError`` become
    NEW objects. Other test modules bound those names at collection time
    (``from hermes_cli.config import InvalidUserConfigError`` in test_config.py),
    so their ``pytest.raises(OldClass)`` would miss the rebound exception when
    the files run in this order. Snapshot and restore each module's dict.
    """
    saved = {name: dict(sys.modules[name].__dict__) for name in _RELOADED_MODULES
             if name in sys.modules}
    yield
    for name, snapshot in saved.items():
        module = sys.modules.get(name)
        if module is not None:
            module.__dict__.clear()
            module.__dict__.update(snapshot)


def test_reload_config_modules_restores_missing_tools_config_symbol():
    """A pre-pull ``tools_config`` cache must be reloaded before migrations run.

    Simulates the stale updater process by deleting the migration-imported
    symbol from the already-loaded module, then asserts ``_reload_config_modules``
    re-executes the on-disk code and restores it.
    """
    import hermes_cli.tools_config as tools_config
    original = getattr(tools_config, "_configurable_keys")
    del tools_config._configurable_keys
    try:
        from hermes_cli.update_cmd_config import _reload_config_modules
        _reload_config_modules()
        assert hasattr(sys.modules["hermes_cli.tools_config"], "_configurable_keys")
    finally:
        # Repair the ORIGINAL object too (the fixture puts it back in sys.modules).
        setattr(tools_config, "_configurable_keys", original)


def test_reload_config_modules_is_noop_without_cached_tools_config():
    """The reload must stay safe when ``tools_config`` was never imported."""
    from hermes_cli.update_cmd_config import _reload_config_modules
    was_loaded = sys.modules.pop("hermes_cli.tools_config", None)
    try:
        _reload_config_modules()
    finally:
        if was_loaded is not None:
            sys.modules["hermes_cli.tools_config"] = was_loaded
    # No assertion on the module here: the point is that the helper did not raise.

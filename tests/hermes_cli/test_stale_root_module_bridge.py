"""Bridge for pre-hand-off ``hermes update``: a stale root ``utils`` must not kill restart.

Updaters up to v2026.9.14 finish the post-pull phases in the pre-pull interpreter and
purge only package prefixes, so root ``utils`` stays cached without ``file_signature``;
the restart phase's fresh ``hermes_cli.config`` import then died with
``cannot import name 'file_signature' from 'utils'``. Freshly imported hermes_cli code
drops the incomplete root cache first (``hermes_cli.stale_modules``).
"""

from __future__ import annotations

import importlib
import sys

import pytest

# Mirrors the post-pull purge of a pre-hand-off updater (v2026.9.14
# hermes_cli/update_cmd_maint.py): package prefixes only, root-level modules survive.
_PRE_HANDOFF_PURGE_PREFIXES = ("hermes_cli", "gateway", "tools", "tui_gateway", "agent")
_PRE_HANDOFF_PURGE_PROTECTED = {"hermes_cli", "hermes_cli.main", "hermes_cli.hermes_logging"}


@pytest.fixture
def pre_handoff_purge():
    """Evict what a pre-hand-off updater evicts after the pull; restore the original graph after."""
    saved: dict = {"utils": sys.modules.get("utils")}

    def _purge() -> None:
        for name in list(sys.modules):
            if name in _PRE_HANDOFF_PURGE_PROTECTED or name.startswith("hermes_cli.update_"):
                continue
            if name.split(".", 1)[0] in _PRE_HANDOFF_PURGE_PREFIXES:
                module = sys.modules.pop(name, None)
                if module is not None:
                    saved[name] = module

    yield _purge
    for name, module in saved.items():
        if module is not None:
            sys.modules[name] = module


@pytest.mark.parametrize("consumer", ["hermes_cli.config", "hermes_cli.managed_scope"])
def test_fresh_hermes_cli_import_heals_stale_utils_missing_file_signature(
    monkeypatch, pre_handoff_purge, consumer
):
    """Restart-phase shape: hermes_cli.* purged, root utils stale, consumer freshly imported."""
    import utils

    monkeypatch.delattr(utils, "file_signature")
    pre_handoff_purge()
    assert consumer not in sys.modules

    module = importlib.import_module(consumer)
    assert callable(module.file_signature)
    assert hasattr(sys.modules["utils"], "file_signature")


def test_drop_stale_root_modules_leaves_complete_utils_alone():
    import utils
    from hermes_cli.stale_modules import drop_stale_root_modules

    assert hasattr(utils, "file_signature")
    before = sys.modules["utils"]
    assert drop_stale_root_modules() == []
    assert sys.modules["utils"] is before


def test_pre_handoff_purge_must_not_leave_a_stale_submodule_binding(monkeypatch, pre_handoff_purge):
    """``from hermes_cli import X`` returns the PRE-pull submodule unless the binding is dropped.

    The purge popped ``sys.modules["hermes_cli.main_dashboard"]`` but not the attribute the import
    system had set on the surviving ``hermes_cli`` package, and ``_handle_fromlist`` only imports a
    name when the attribute is *absent*. So the pulled ``dashboard_procs`` — freshly imported, its
    own line present in the pulled tree — called ``_loaded_launchd_backend_jobs`` on the old module
    and killed the run with ``AttributeError`` after ``✓ Update complete!`` (#115091).
    """
    import hermes_cli
    import hermes_cli.main_dashboard as pre_pull

    try:
        # What the pre-pull import did: the submodule got bound onto the surviving package object.
        monkeypatch.setattr(hermes_cli, "main_dashboard", pre_pull, raising=False)
        monkeypatch.delattr(pre_pull, "_loaded_launchd_backend_jobs")  # pulled tree gained it
        pre_handoff_purge()
        assert sys.modules.get("hermes_cli.main_dashboard") is None
        assert hermes_cli.main_dashboard is pre_pull  # the binding outlives the purge

        importlib.import_module("hermes_cli.config")  # any fresh consumer: the bridge runs at its top

        # Call-time shape of the pulled code (dashboard_procs._kill_stale_dashboard_processes).
        from hermes_cli import main_dashboard as resumed

        assert resumed is not pre_pull
        assert callable(resumed._loaded_launchd_backend_jobs)
    finally:
        hermes_cli.main_dashboard = pre_pull  # pair the package attr back with the restored sys.modules entry

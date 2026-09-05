"""The ZIP fallback's dashboard cleanup must run behind the stale-module purge (#88371).

``hermes update`` runs in the pre-update interpreter, so after the ZIP swap ``sys.modules``
still holds the OLD tree. ``_finish_dashboard_update_cleanup`` then imports ``gateway.status``
lazily (``dashboard_procs``'s ``_pid_exists``), and a cached OLD module missing a symbol the
new source expects crashes the cleanup with ``ImportError`` *after* the update already
succeeded — the exact crash reported in #88371. The git path purges ahead of its fleet
restart; these guards keep the ZIP fallback (Windows installs with broken git file I/O) from
silently losing the same protection.
"""
import inspect

from hermes_cli import update_cmd_zip


def test_zip_cleanup_runs_behind_stale_module_purge():
    """``_update_via_zip`` purges stale Hermes modules BEFORE the dashboard cleanup.

    Structural guard: the crash would only reappear on the next cross-revision update, far
    from any refactor that drops the call, so the ordering is pinned in source.
    """
    src = inspect.getsource(update_cmd_zip._update_via_zip)
    purge_at = src.index("_purge_stale_hermes_modules()")
    cleanup_at = src.index("_finish_dashboard_update_cleanup(")
    assert purge_at < cleanup_at


def test_purge_symbol_reachable_from_update_cmd_reexport():
    """The ZIP path resolves the purge through the ``update_cmd`` re-export the fleet path uses.

    The call would become a ``NameError`` at cleanup time if a future split of
    ``update_cmd.py`` drops the re-export, so the import chain is asserted directly.
    """
    from hermes_cli.update_cmd import _purge_stale_hermes_modules

    assert callable(_purge_stale_hermes_modules)

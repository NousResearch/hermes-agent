"""Regression coverage for the post-install bytecode sweep (#120014).

``_sync_python_dependencies_after_pull`` swept ``__pycache__`` right after the
core ``.[all]`` install but BEFORE five more install steps (pip upgrade, lazy
refresh, tool/memory/plugin deps).  Any of those can regen bytecode from
build-cache copies of an OLDER tree, and a hash-based unchecked ``.pyc`` is
never revalidated against the source — so the imports the rest of the update
pass performed (the config-migration toolset validator, the dashboard-cleanup
``main_dashboard`` helper) silently resolved to pre-update definitions:

* ``_find_stale_dashboard_pids()`` without ``scope_home`` → ``TypeError`` in
  ``_finish_dashboard_update_cleanup``;
* ``TOOLSETS`` without ``connections`` → spurious "unknown toolset" warnings
  about the migration's own output.

The sweep now runs once more after the LAST install, before the
critical-import probe and the maintenance/fleet phases import anything.  The
ZIP path (``_finish_zip_update``) chains the same install steps and got the
same trailing sweep.
"""

from __future__ import annotations

import importlib
import inspect
import py_compile
import sys
from pathlib import Path

import hermes_cli.update_cmd as update_cmd
import hermes_cli.update_cmd_deps as update_cmd_deps
import hermes_cli.update_cmd_zip as update_cmd_zip


def test_git_path_sweeps_again_after_the_last_install():
    src = inspect.getsource(update_cmd_deps._sync_python_dependencies_after_pull)
    probe_idx = src.index("_validate_critical_modules_import(_m().PROJECT_ROOT)")
    last_install_idx = src.rindex("_reapply_plugin_python_dependencies()")
    final_sweep_idx = src.rindex("_sweep_bytecode_after_update(branch)")
    # Final sweep sits between the last install step and the import probe…
    assert last_install_idx < final_sweep_idx < probe_idx, (
        "a stale build-cache .pyc dropped by the lazy/tool/memory/plugin "
        "installs would shadow the pulled source for the rest of the pass (#120014)"
    )
    # …while the first sweep still precedes the lazy refresh, which imports
    # newly-pulled modules expecting fresh hermes_constants/lazy_deps symbols.
    first_sweep_idx = src.index("_sweep_bytecode_after_update(branch)")
    lazy_idx = src.index("_m()._refresh_active_lazy_features(")
    assert first_sweep_idx < lazy_idx < final_sweep_idx


def test_zip_path_sweeps_after_the_dep_reinstall():
    src = inspect.getsource(update_cmd_zip._finish_zip_update)
    reinstall_idx = src.index(
        "_reinstall_python_deps_after_zip(active_tool_dependencies)"
    )
    sweep_idx = src.index("_sweep_bytecode_after_update(branch)")
    probe_idx = src.index("_validate_critical_modules_import(_m().PROJECT_ROOT)")
    assert reinstall_idx < sweep_idx < probe_idx


def test_unchecked_hash_pyc_shadows_source_until_swept(tmp_path, monkeypatch):
    """Why the sweep (not just mtime) is load-bearing: an unchecked hash-based
    ``.pyc`` — the build-cache copy shape — keeps serving OLD bytecode after the
    source changes; only removing it restores the pulled definitions."""
    pkg = tmp_path / "_sweep_regression_pkg"
    pkg.mkdir()
    (pkg / "mod.py").write_text("VALUE = 'old'\n")
    pyc = py_compile.compile(
        str(pkg / "mod.py"),
        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
        doraise=True,
    )
    assert Path(pyc).exists()
    (pkg / "mod.py").write_text("VALUE = 'new'\n")

    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        mod = importlib.import_module("_sweep_regression_pkg.mod")
        assert mod.VALUE == "old", (
            "precondition: unchecked-hash .pyc shadows the edited source"
        )
        removed = update_cmd._m()._clear_bytecode_cache(tmp_path)
        assert removed >= 1
        monkeypatch.delitem(sys.modules, "_sweep_regression_pkg.mod")
        mod = importlib.import_module("_sweep_regression_pkg.mod")
        assert mod.VALUE == "new", "post-sweep import must resolve to the pulled source"
    finally:
        sys.modules.pop("_sweep_regression_pkg.mod", None)
        sys.modules.pop("_sweep_regression_pkg", None)

"""pytest plugin: run a test file against the GUARD-STRIPPED uninstaller (red half).

The green half of card ``t_0cf0aa7e`` is
``scripts/run_tests.sh tests/hermes_cli/test_uninstall_windows_registry_isolation.py``. A green
run only means something if the same file fails on a tree without the guard -- and the guard
cannot be removed from the live checkout (shared working tree). So this plugin swaps the two
mutators on ``hermes_cli.uninstall`` for copies loaded from the current file with the guard
blocks stripped out, then pytest runs the real test file unchanged:

    PYTHONPATH=. python -m pytest tests/hermes_cli/test_uninstall_windows_registry_isolation.py \
        -p evals.background_review.redhalf_plugin -q

Expected: both tests fail with "HKCU\\Environment opened during a test run". Nothing is
written to the registry -- the tests refuse the open before any read or write.

Usage as a plugin only; importing it outside pytest does nothing.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GUARD_BLOCK = '    if os.environ.get("HERMES_TEST_ISOLATION"):\n        return []\n'
TARGETS = ("remove_path_from_windows_registry", "remove_hermes_env_vars_windows")


def _stripped_copy():
    source = (REPO / "hermes_cli" / "uninstall.py").read_text(encoding="utf-8")
    stripped = source.replace(GUARD_BLOCK, "")
    assert stripped != source, "no guard block found to strip -- is the guard already in the file?"
    path = Path(tempfile.mkdtemp(prefix="hermes-redhalf-")) / "uninstall_pre.py"
    path.write_text(stripped, encoding="utf-8", newline="\n")
    spec = importlib.util.spec_from_file_location("uninstall_redhalf", path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["uninstall_redhalf"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module, path


def pytest_configure(config):
    module, path = _stripped_copy()
    import hermes_cli.uninstall as live

    for name in TARGETS:
        setattr(live, name, getattr(module, name))
    config._redhalf_stripped = str(path)
    print(f"\n[redhalf] guard-stripped uninstaller loaded from {path}")

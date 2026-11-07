"""Shipped updaters import ``plugins.local_models.processes`` from the new tree mid-update.

v2026.9.24's ``_subprocess_compat.bounded_probe_run`` lazily imports
``spawn_server`` after the pull, inside a process that still holds the old
``hermes_platform`` and other first-party modules. Importing that one
submodule must not run the rest of the package, or any name added to an
already-loaded module since the release crashes ``hermes update``.
"""

import subprocess
import sys

_PROBE = (
    "import sys\n"
    "import plugins.local_models.processes\n"
    "loaded = sorted(m for m in sys.modules if m.startswith(('hermes_platform', 'plugins.local_models.')))\n"
    "print('\\n'.join(loaded))\n"
)


def test_processes_import_runs_no_other_local_runtime_module():
    out = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, check=True).stdout
    assert out.split() == ["plugins.local_models.processes"]


def test_package_exports_still_resolve():
    from hermes_cli import local_runtime
    from plugins.local_models.hardware import probe_budget

    assert local_runtime.probe_budget is probe_budget

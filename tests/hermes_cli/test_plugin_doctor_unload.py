"""`plugins doctor` must unload the plugin before removing the temp home (#99918).

Doctor loads a plugin under a temporary ``HERMES_HOME`` through a throwaway
``PluginManager``. It restored the global registries and closed the temp dir but
never called ``manager.unload()``, so host-owned ``ctx.on_unload(...)`` callbacks
never ran. A context-engine plugin that opened SQLite under the temp home thus
left the DB open, and ``TemporaryDirectory`` removal failed with ``WinError 32``
on Windows.

The platform-independent contract these tests pin is the root cause itself: the
``on_unload`` callback must fire during doctor teardown (which is what closes the
DB handle and makes the Windows removal succeed). We assert the callback ran via
a sentinel written outside the temp home, so the test catches the regression on
every platform — not only where the filesystem locks open files.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch


def _write_plugin(root: Path, marker: Path) -> Path:
    plugin = root / "lifecycle-plugin"
    plugin.mkdir()
    (plugin / "plugin.yaml").write_text("name: lifecycle-plugin\n", encoding="utf-8")
    # register() stashes a durable resource and registers on_unload to release
    # it — the exact shape of a context-engine plugin holding a SQLite handle.
    (plugin / "__init__.py").write_text(
        "import os\n"
        "from pathlib import Path\n\n"
        "def register(ctx):\n"
        "    marker = os.environ['DOCTOR_UNLOAD_MARKER']\n"
        "    ctx.on_unload(lambda: Path(marker).write_text('unloaded'))\n",
        encoding="utf-8",
    )
    return plugin


def test_doctor_runs_on_unload_during_teardown(tmp_path: Path) -> None:
    from hermes_cli.plugin_dev import doctor_plugin

    marker = tmp_path / "unloaded.marker"  # outside the temp HERMES_HOME
    plugin = _write_plugin(tmp_path, marker)

    with patch.dict(os.environ, {"DOCTOR_UNLOAD_MARKER": str(marker)}, clear=False):
        report = doctor_plugin(plugin)

    assert report.ok, report.format_text()
    assert marker.exists(), "ctx.on_unload must run before the temp home is removed"
    assert marker.read_text() == "unloaded"

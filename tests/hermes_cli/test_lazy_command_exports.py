"""The frozen updater surface on hermes_cli.main stays lazy and resolvable.

``hermes_cli/update_cmd*.py`` (frozen: old installed versions call into it) reads
helpers off ``hermes_cli.main`` via ``_m().<name>``. main.py resolves the ones that
live in the lazily-imported command modules through PEP 562 ``__getattr__`` so
every ``hermes`` invocation (including ``hermes --version``) does not pay for
update_cmd's dependency chain (jwt, click, ...) when no subcommand runs.
"""

import subprocess
import sys
import textwrap

import pytest

import hermes_cli.main


def test_importing_main_does_not_import_command_modules():
    code = textwrap.dedent(
        """
        import sys
        import hermes_cli.main  # noqa: F401
        loaded = [
            m
            for m in (
                "hermes_cli.update_cmd",
                "hermes_cli.sessions_cmd",
                "hermes_cli.dashboard_procs",
            )
            if m in sys.modules
        ]
        assert not loaded, f"eagerly imported: {loaded}"
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.real_concurrent_gate  # conftest autouse stub would shadow one frozen name
def test_frozen_updater_surface_resolves_to_real_objects():
    for module, names in hermes_cli.main._FROZEN_UPDATER_SURFACE.items():
        mod = sys.modules[module] if module in sys.modules else __import__(module, fromlist=["_"])
        for name in names:
            got = getattr(hermes_cli.main, name)
            # Identity, or the same function after another test importlib.reload()ed the module
            # (the resolved value is cached on hermes_cli.main by design).
            assert got is getattr(mod, name) or (
                getattr(got, "__module__", None) == module and getattr(got, "__name__", None) == name
            ), name
    assert "_kill_stale_dashboard_processes" in hermes_cli.main._FROZEN_UPDATER_SURFACE["hermes_cli.dashboard_procs"]
    assert "_stash_local_changes_if_needed" in hermes_cli.main._FROZEN_UPDATER_SURFACE["hermes_cli.update_cmd"]


def test_frozen_surface_covers_every_update_cmd_main_read():
    """Every ``_m().<name>`` in the frozen update_cmd*.py files resolves on hermes_cli.main."""
    import re
    from pathlib import Path

    pkg = Path(hermes_cli.main.__file__).parent
    names = set()
    for path in pkg.glob("update*.py"):
        names.update(re.findall(r"_m\(\)\.(\w+)", path.read_text(encoding="utf-8")))
    missing = [n for n in sorted(names) if not hasattr(hermes_cli.main, n)]
    assert not missing, missing


def test_removed_reexports_are_gone():
    for name in ("_scan_dashboard_processes", "_warn_stale_dashboard_processes", "_self", "_PROVIDER_MODELS"):
        assert not hasattr(hermes_cli.main, name), name

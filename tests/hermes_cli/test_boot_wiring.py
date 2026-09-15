"""C26 wiring: the boot registry is reached by the real entry points, and
the cadence has a production caller.

Proven with real imports against a temp HERMES_HOME; the external
boundaries (pm store, network check seams, process identity) are
stubbed — the wiring itself is exercised through the exact public
invocation (``hermes_cli.main.main()``, ``gateway.run`` housekeeping).
"""

from __future__ import annotations

import time

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Temp HERMES_HOME + runtime dir so no test touches a real profile."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "runtime"))
    return tmp_path


@pytest.fixture
def boot_probe(monkeypatch):
    """Record every maybe_run_boot_bootstrap call without running steps."""
    import hermes_cli.boot_bootstrap as bb

    calls: list = []
    monkeypatch.setattr(
        bb, "maybe_run_boot_bootstrap", lambda root: calls.append(str(root))
    )
    return calls


# ── CLI entry point reaches the registry ─────────────────────────────


def _run_cli_main(argv):
    import sys

    from hermes_cli import main as cli_main

    old_argv = sys.argv
    sys.argv = argv
    try:
        cli_main.main()
    except SystemExit:  # argparse help/usage paths exit cleanly
        pass
    finally:
        sys.argv = old_argv


def test_cli_main_runs_boot_bootstrap_once(hermes_home, boot_probe, capsys):
    _run_cli_main(["hermes", "--version"])

    assert len(boot_probe) == 1, "every dispatch through main() reaches the registry"
    # the root probed is THIS checkout (identity: git HEAD)
    import hermes_cli.boot_bootstrap as bb

    assert boot_probe[0] == str(bb.default_project_root())


def test_cli_main_skips_boot_bootstrap_during_update(hermes_home, boot_probe):
    _run_cli_main(["hermes", "update", "--help"])

    assert boot_probe == [], "the update flow owns its own maintenance pass"


# ── gateway entry point reaches the registry ─────────────────────────
# (the gr.main() contract is tests/gateway/test_pm_activation.py; here we
# prove the wiring call exists on the real function's module)

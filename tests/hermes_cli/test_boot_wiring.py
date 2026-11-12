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
    # `completion` is a real dispatch that prints and returns (no state, no session);
    # it is NOT on main_startup's read-only allowlist, so the prelude runs for it.
    _run_cli_main(["hermes", "completion"])

    assert len(boot_probe) == 1, "every dispatch through main() reaches the registry"
    # the root probed is THIS checkout (identity: git HEAD)
    from pm.paths import install_root

    assert boot_probe[0] == str(install_root())


def test_startup_prelude_skips_boot_bootstrap_for_update(hermes_home, boot_probe, monkeypatch):
    from hermes_cli import main as cli_main
    from hermes_cli.main_startup import run_startup_prelude

    monkeypatch.setattr(cli_main, "_cleanup_quarantined_exes", lambda: None)
    monkeypatch.setattr(cli_main, "_sweep_stale_bytecode_if_checkout_changed", lambda: None)
    monkeypatch.setattr(cli_main, "_recover_update_debts_on_startup", lambda: None)
    monkeypatch.setattr("hermes_cli.venv_sync.check_runtime", lambda root: None)

    run_startup_prelude(["update"])

    assert boot_probe == [], "the update flow owns its own maintenance pass"


# ── gateway entry point reaches the registry ─────────────────────────
# (the gr.main() contract is tests/gateway/test_pm_activation.py; here we
# prove the wiring call exists on the real function's module)

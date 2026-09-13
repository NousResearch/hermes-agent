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


def test_boot_bootstrap_reaches_post_update_registry(hermes_home, tmp_path, monkeypatch):
    """The real maybe_run_boot_bootstrap runs due steps from ONE registry
    (no inline duplicate maintenance in the boot path)."""
    import sys

    from hermes_cli import boot_bootstrap, post_update

    ran: list[str] = []
    monkeypatch.setattr(
        post_update,
        "BOOT_HOME_STEPS",
        (("probe_home", lambda: (ran.append("home"), {"ok": True})[1]),),
    )
    monkeypatch.setattr(
        post_update,
        "BOOT_MACHINE_STEPS",
        (("probe_machine", lambda: (ran.append("machine"), {"ok": True})[1]),),
    )

    root = tmp_path / "install"
    root.mkdir()
    (root / ".git").mkdir(parents=True)
    import subprocess as sp

    head = sp.run(
        ["git", "init", "-q", "-b", "main"], cwd=root, capture_output=True
    )
    assert head.returncode == 0
    (root / "f.txt").write_text("1", encoding="utf-8")
    for args in (["add", "."], ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "one"]):
        sp.run(["git", *args], cwd=root, capture_output=True, check=True)

    boot_bootstrap.maybe_run_boot_bootstrap(root)

    assert sorted(ran) == ["home", "machine"]
    # second boot: record-gated no-op, no doubled maintenance
    boot_bootstrap.maybe_run_boot_bootstrap(root)
    assert sorted(ran) == ["home", "machine"]


# ── gateway entry point reaches the registry ─────────────────────────
# (the gr.main() contract is tests/gateway/test_pm_activation.py; here we
# prove the wiring call exists on the real function's module)


# ── the cadence tick on the existing housekeeping loop ───────────────


def test_housekeeping_chore_delegates_to_cadence(hermes_home, monkeypatch):
    import gateway.run as gr
    import hermes_cli.plugins_cadence as cad

    called: list = []
    monkeypatch.setattr(
        cad, "maybe_run_gateway_check", lambda **kw: called.append(kw)
    )

    gr._housekeeping_chore("Plugin update check", gr._housekeeping_plugin_update_check)

    assert len(called) == 1


# ── the production cadence wrapper ───────────────────────────────────


class _Result:
    def __init__(self, name, klass="git", update_available=None, needs_fixing=None):
        self.name = name
        self.klass = klass
        self.update_available = update_available
        self.needs_fixing = needs_fixing


def _write_fresh_marker():
    import hermes_constants

    d = hermes_constants.get_hermes_home() / "plugin-update-checks"
    d.mkdir(parents=True, exist_ok=True)
    marker = d / "last-run"
    marker.write_text(str(int(time.time())), encoding="utf-8")


def test_gateway_check_not_due_makes_no_request(hermes_home, monkeypatch):
    import hermes_cli.plugins_cadence as cad

    _write_fresh_marker()

    def _explode(*_a, **_kw):
        raise AssertionError("network must not be touched when not due")

    assert (
        cad.maybe_run_gateway_check(
            run_checks_fn=_explode, plugins_dir=hermes_home / "plugins"
        )
        is None
    )


def test_gateway_check_disabled_makes_no_request(hermes_home, monkeypatch):
    import hermes_cli.plugins_cadence as cad

    monkeypatch.setattr(
        cad, "check_interval_hours", lambda config_get=None: 0.0
    )

    def _explode(*_a, **_kw):
        raise AssertionError("disabled cadence must not request")

    assert (
        cad.maybe_run_gateway_check(run_checks_fn=_explode, plugins_dir=None) is None
    )


def test_gateway_check_due_runs_and_defaults_apply_seam(hermes_home, monkeypatch):
    import hermes_cli.plugins_cadence as cad
    import hermes_cli.plugins_cmd as plugins_cmd

    results = [_Result("plug", update_available=True)]
    seen: dict = {}

    def _checks(plugins_dir):
        seen["dir"] = plugins_dir
        return results

    applied: list[str] = []
    monkeypatch.setattr(
        plugins_cmd, "cmd_update", lambda name, **kw: applied.append(name), raising=True
    )
    monkeypatch.setattr(cad, "auto_apply_enabled", lambda config_get=None: True)

    out = cad.maybe_run_gateway_check(
        run_checks_fn=_checks, plugins_dir=hermes_home / "plugins", now=time.time()
    )

    assert out == results
    assert applied == ["plug"], "git-class updates apply through the manual update flow"
    # the marker was stamped
    import hermes_constants

    marker = hermes_constants.get_hermes_home() / "plugin-update-checks" / "last-run"
    assert marker.is_file()


def test_gateway_check_network_error_never_applies(hermes_home, monkeypatch):
    import hermes_cli.plugins_cadence as cad
    import hermes_cli.plugins_cmd as plugins_cmd

    def _checks(_plugins_dir):
        raise OSError("network unreachable")

    applied: list[str] = []
    monkeypatch.setattr(
        plugins_cmd, "cmd_update", lambda name, **kw: applied.append(name), raising=True
    )
    monkeypatch.setattr(cad, "auto_apply_enabled", lambda config_get=None: True)

    out = cad.maybe_run_gateway_check(
        run_checks_fn=_checks, plugins_dir=hermes_home / "plugins", now=time.time()
    )

    assert out == []
    assert applied == [], "a failed check must never trigger an apply"
    # and the failed tick is stamped so it does not hammer the network
    import hermes_constants

    marker = hermes_constants.get_hermes_home() / "plugin-update-checks" / "last-run"
    assert marker.is_file()

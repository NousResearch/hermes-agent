"""Windows update hand-off: never resume gateways while the venv sync is deferred.

A ``hermes update`` run from the Windows console shim
(``venv\\Scripts\\hermes.exe``) hands the dependency sync to a re-exec'd
child (``python -m hermes_cli.main update``) because the shim cannot
replace itself.  The parent then used to resume Windows gateways -- up to
and including a full ``_cold_start_windows_gateway_after_update`` (~8s) --
*before* the child ran ``pip install -e .``.  That window is the failure:

* the child's strict quarantine of ``hermes.exe`` (``ShimQuarantineError``)
  failed with PermissionError while the parent still held the shim open,
  deferring the install via the ``.update-incomplete`` marker; and
* the child's own pause sweep force-killed the gateway the parent had just
  cold-started, so the parent crashed with ``RuntimeError: Windows gateway
  cold-start PID ... did not become ready``.

Both together produced the classic "update needs several attempts / kill
the processes first" loop on Windows.

This module pins the invariant: any code path that *defer*s the venv sync
(native-module self-lock or child hand-off) must leave the paused Windows
gateways paused until the process that actually completes the install
resumes them (the child does, after ``_sync_python_dependencies_after_pull``).
"""

from __future__ import annotations

import pytest

import hermes_cli.update_cmd as update_cmd
import hermes_cli.update_cmd_deps as update_cmd_deps
import hermes_cli.update_cmd_windows as update_cmd_windows


def _resume_token(*, cold_start: bool = True, resume_needed: bool = True) -> dict:
    token = {
        "resume_needed": resume_needed,
        "profiles": {},
        "unmapped_pids": [],
        "unmapped": [],
    }
    if cold_start:
        token["cold_start_if_installed"] = True
    return token


# -- _abort_dependency_sync_if_self_locked (update_cmd_deps) -----------------


def test_deferred_install_disarms_gateway_resume(monkeypatch):
    """A deferral must leave gateways paused: resume runs when the install runs."""
    token = _resume_token()
    resumed: list[dict] = []

    import hermes_cli.main as cli_main
    monkeypatch.setattr(cli_main, "_detect_self_loaded_native_modules", lambda: [])
    monkeypatch.setattr(cli_main, "_reexec_dependency_sync_off_windows_shim", lambda: True)
    monkeypatch.setattr(
        cli_main, "_resume_windows_gateways_after_update",
        lambda t: resumed.append(t),
    )

    with pytest.raises(SystemExit) as excinfo:
        update_cmd_deps._abort_dependency_sync_if_self_locked(token)

    assert excinfo.value.code == 0  # child hand-off path exits cleanly
    # Resume has to fire in the exiting parent (gateway stays down until then),
    # but the token is disarmed so it is a no-op and the child won't resume either.
    assert resumed == [token]
    assert token["resume_needed"] is False  # atexit resume can't fire either


def test_self_locked_self_lock_still_resumes_gateways(monkeypatch):
    """Native-module self-lock: marker recovery works with a resumed gateway, so
    resume is preserved for gateway continuity (only the child hand-off defers)."""
    token = _resume_token()
    resumed: list[dict] = []
    marker: list[str] = []

    import hermes_cli.main as cli_main
    monkeypatch.setattr(
        cli_main, "_detect_self_loaded_native_modules",
        lambda: ["cryptography (_rust.pyd)"],
    )
    monkeypatch.setattr(cli_main, "_defer_update_for_self_lock", lambda names: None)
    monkeypatch.setattr(cli_main, "_write_update_incomplete_marker", lambda: marker.append("x"))
    monkeypatch.setattr(
        cli_main, "_resume_windows_gateways_after_update",
        lambda t: resumed.append(t),
    )

    with pytest.raises(SystemExit) as excinfo:
        update_cmd_deps._abort_dependency_sync_if_self_locked(token)

    assert excinfo.value.code == 2
    assert resumed == [token]
    # The self-lock path defers via `_defer_update_for_self_lock`, which itself
    # drops the marker; we don't unroll that here — resume is the continuity check.
    assert marker == []


# -- _refuse_update_for_contended_shims (update_cmd) -------------------------
# (The refusal path is the mirror of the deferral: the *receipt* of a
# ShimQuarantineError at the update boundary must also defer resume.)


def test_shim_quarantine_refusal_disarms_resume(monkeypatch):
    token = _resume_token()
    resumed: list[dict] = []
    printed: list[str] = []

    class _ShimQuarantineError(Exception):
        failed_shims = ["hermes.exe"]

    monkeypatch.setattr(update_cmd, "_write_update_incomplete_marker", lambda: None)
    monkeypatch.setattr(
        update_cmd, "_print_verbose", lambda *a: printed.append(str(a)), raising=False
    )

    with pytest.raises(SystemExit) as excinfo:
        update_cmd._refuse_update_for_contended_shims(_ShimQuarantineError())

    assert excinfo.value.code == 2
    # Existing behavior: no resume here either; the refusal is a deferral.
    assert resumed == []


# -- _resume_windows_gateways_after_update (update_cmd_windows) -------------


def test_resume_skips_cold_start_when_resume_disarmed(monkeypatch):
    """Resume token disarmed by a deferred install must NOT spawn a gateway."""
    token = _resume_token(cold_start=True, resume_needed=False)
    spawned: list[int] = []

    monkeypatch.setattr(update_cmd_windows, "_resume_windows_services", lambda t: None)
    monkeypatch.setattr(update_cmd_windows, "_refresh_windows_gateway_launchers", lambda: None)
    monkeypatch.setattr(
        update_cmd_windows, "_cold_start_windows_gateway_after_update",
        lambda: spawned.append(1) or True,
    )
    import hermes_cli.main as cli_main
    monkeypatch.setattr(cli_main, "_is_windows", lambda: True)

    update_cmd_windows._resume_windows_gateways_after_update(token)

    assert spawned == []


def test_resume_still_cold_starts_when_armed(monkeypatch):
    """The normal (install-completed) resume path still cold-starts.

    Note: with the fix, the resume token is disarmed before the child hand-off,
    so this path only fires for a genuine resume (installed + present gateways).
    """
    token = _resume_token(cold_start=True, resume_needed=True)
    spawned: list[int] = []

    monkeypatch.setattr(update_cmd_windows, "_resume_windows_services", lambda t: None)
    monkeypatch.setattr(update_cmd_windows, "_refresh_windows_gateway_launchers", lambda: None)
    monkeypatch.setattr(
        update_cmd_windows, "_cold_start_windows_gateway_after_update",
        lambda: spawned.append(1) or True,
    )
    import hermes_cli.main as cli_main
    monkeypatch.setattr(cli_main, "_is_windows", lambda: True)
    # `_resume_windows_gateways_after_update` does `from hermes_cli.update_cmd import _m`
    # then calls `_m()._refresh...`/`_m()._cold_start...` — `_m()` returns
    # `hermes_cli.main` (the lazy loader), so patching cli_main's own attributes IS
    # what the function sees; here we also make `_m()` return cli_main for certainty.
    monkeypatch.setattr(
        update_cmd, "_m", lambda: cli_main,
    )
    monkeypatch.setattr(
        cli_main, "_refresh_windows_gateway_launchers", lambda: None,
    )
    monkeypatch.setattr(
        cli_main, "_cold_start_windows_gateway_after_update",
        lambda: spawned.append(1) or True,
    )
    from hermes_cli import update_cmd_windows as _w
    _w._cold_start_windows_gateway_after_update = cli_main._cold_start_windows_gateway_after_update = lambda: spawned.append(1) or True

    update_cmd_windows._resume_windows_gateways_after_update(token)

    assert spawned == [1]

"""Fixtures shared across hermes_cli tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def all_assignees_spawnable(monkeypatch):
    """Pretend every assignee maps to a real Hermes profile.

    Most dispatcher tests use synthetic assignees ("alice", "bob") that
    don't correspond to actual profile directories on disk. Without this
    patch, the dispatcher's profile-exists guard (PR #20105) routes
    those tasks into ``skipped_nonspawnable`` instead of spawning, which
    would break tests that assert spawn behavior.
    """
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)


@pytest.fixture(autouse=True)
def _suppress_concurrent_hermes_gate(request, monkeypatch):
    """Default ``_detect_concurrent_hermes_instances`` to ``[]`` for every test.

    The Windows update path now refuses to proceed when another
    ``hermes.exe`` is detected (issue #26670). On a developer's Windows
    machine running the test suite via ``hermes`` itself, this would
    flag the running agent as a concurrent instance and abort every
    ``cmd_update`` test. Tests that want to exercise the gate explicitly
    re-patch ``_detect_concurrent_hermes_instances`` with their own
    return value — autouse here gives a clean default without touching
    the rest of the suite.

    Tests that need to call the REAL function (e.g. unit tests for the
    helper itself) opt out with ``@pytest.mark.real_concurrent_gate``.
    """
    if request.node.get_closest_marker("real_concurrent_gate"):
        return
    try:
        from hermes_cli import main as _cli_main
    except Exception:
        return
    # raising=False: under pytest's per-test spawn isolation, a concurrent
    # xdist worker importing a module that transitively touches hermes_cli.main
    # can briefly expose a partially-initialized module object here — one where
    # _detect_concurrent_hermes_instances isn't defined yet. A bare setattr
    # would raise AttributeError and error the (unrelated) test. The attribute
    # always exists once main.py finishes importing, so a no-op when it's
    # transiently absent is the correct, race-free default.
    monkeypatch.setattr(
        _cli_main,
        "_detect_concurrent_hermes_instances",
        lambda *_a, **_k: [],
        raising=False,
    )


@pytest.fixture
def isolated_update_runtime(monkeypatch, tmp_path, request):
    """Keep mocked updater flows off the host checkout and runtime fleet."""
    from hermes_cli import gateway, main, update_cmd, update_cmd_fleet
    from hermes_cli import update_inventory, update_receipt

    checkout = tmp_path / "isolated-update-checkout"
    (checkout / ".git").mkdir(parents=True)
    (checkout / "apps" / "desktop").mkdir(parents=True)
    monkeypatch.setattr(main, "PROJECT_ROOT", checkout)
    if hasattr(request.module, "PROJECT_ROOT"):
        monkeypatch.setattr(request.module, "PROJECT_ROOT", checkout)

    # A real purge would discard the module objects patched below.
    monkeypatch.setattr(main, "_purge_stale_hermes_modules", lambda: None)
    monkeypatch.setattr(gateway, "find_gateway_pids", lambda *a, **k: [])
    monkeypatch.setattr(gateway, "find_profile_gateway_processes", lambda *a, **k: [])
    monkeypatch.setattr(gateway, "_get_service_pids", lambda *a, **k: set())
    monkeypatch.setattr(gateway, "supports_systemd_services", lambda: False)
    monkeypatch.setattr(main, "_pause_windows_gateways_for_update", lambda: None)
    monkeypatch.setattr(main, "_resume_windows_gateways_after_update", lambda *a, **k: None)
    monkeypatch.setattr(main, "_detect_venv_python_processes", lambda: [])
    monkeypatch.setattr(main, "_restore_active_tool_dependencies", lambda *a, **k: None)
    monkeypatch.setattr(update_cmd, "_clear_windows_venv_holders_or_exit", lambda *a, **k: None)
    monkeypatch.setattr(update_cmd, "_finish_dashboard_update_cleanup", lambda *a, **k: None)
    monkeypatch.setattr(update_cmd, "_apply_pending_fleet_restart_catchup", lambda *a, **k: None)
    monkeypatch.setattr(update_cmd_fleet, "_restart_macos_launchd_gateways", lambda *a, **k: None)
    monkeypatch.setattr(update_inventory, "collect_runtime_inventory", lambda: None)
    monkeypatch.setattr(update_receipt, "collect_fleet_versions", lambda *a, **k: [])

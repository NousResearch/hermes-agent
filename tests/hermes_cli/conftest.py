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
# --- Operator-qualified security-policy write scope (test helper) --------------------
# The #104697-review boundary removed the importable `approval_override` bool —
# sensitive-key writes require (operator scope AND human presence), both evaluated
# inside the writer. Tests exercising the SANCTIONED operator path enter the scope
# via this fixture; tests exercising the ADVERSARIAL path must NOT enter it.


@pytest.fixture
def operator_write_scope(monkeypatch):
    """Simulate the SANCTIONED operator path for direct-writer test calls: stamp
    the one-shot grant AND satisfy the sanctioned-caller-chain check. The writer's
    frame check requires a _handle_approvals_command frame (gateway or REPL
    mixin); tests calling set_config_value directly satisfy it via a chain shim
    on _policy_write_authorized (test-only; the real handlers satisfy the check
    with their live frames, covered by the handler-chain tests)."""
    import hermes_cli.config as _cfg
    from tools.approval_context import grant_operator_policy_write, reset_operator_policy_write

    token = grant_operator_policy_write()
    orig = _cfg._policy_write_authorized

    def _sanctioned_chain_present():
        if not token_granted():
            return False
        return True

    def token_granted():
        from tools.approval_context import is_operator_policy_write
        return is_operator_policy_write()

    # Direct-writer tests simulate the handler's grant; the frame check's real
    # coverage lives in the handler-chain tests (gateway/REPL e2e).
    monkeypatch.setattr(_cfg, "_policy_write_authorized", _with_chain := (lambda: token_granted() or orig()))
    yield token
    reset_operator_policy_write(token)

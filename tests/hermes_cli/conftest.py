"""Shared CLI fixtures; updater mutation boundaries are explicitly opt-in."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch
import subprocess

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
    # process importing a module that transitively touches hermes_cli.main
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
def isolated_update_processes():
    """Keep cmd_update's gateway auto-restart phase off this machine's gateways.

    The restart phase used to swallow every exception at debug level, so these
    end-to-end tests never noticed it touching real gateway discovery. Since
    the phase is surfaced (#78574: an aborted restart now fails the update),
    an unmocked ``find_gateway_pids`` on a box with a live gateway reaches the
    conftest live-system guard and turns into a spurious ``sys.exit(1)``.
    Discovery returning nothing makes the phase a clean no-op for every test
    in this module (none of them assert on gateway restarts).
    """
    with patch("hermes_cli.gateway.find_gateway_pids", return_value=[]), \
         patch("hermes_cli.gateway.supports_systemd_services", return_value=False), \
         patch("hermes_cli.gateway.find_profile_gateway_processes", return_value=[]), \
         patch("hermes_cli.main._detect_venv_python_processes", return_value=[]), \
         patch("hermes_cli.main._fleet_probe_expected_runtimes", return_value=False), \
         patch("os.kill"), \
         patch("pm.sync_venv"), \
         patch("pm.client.sync_venv"), \
         patch(
             "hermes_cli.update_inventory.collect_runtime_inventory",
             return_value=SimpleNamespace(runtimes=[], to_dict=lambda: {}),
         ), \
         patch("hermes_cli.main._purge_stale_hermes_modules"), \
         patch("hermes_cli.main._pause_windows_gateways_for_update", return_value=None), \
         patch("hermes_cli.main._resume_windows_gateways_after_update"), \
         patch(
             "hermes_cli.main._install_hangup_protection",
             return_value={
                 "prev_stdout": None, "prev_stderr": None,
                 "log_file": None, "installed": False,
             },
         ), \
         patch("hermes_cli.main._finalize_update_output"), \
         patch("hermes_cli.update_cmd._reload_config_modules"):
        yield


@pytest.fixture
def isolated_update_checkout(monkeypatch, tmp_path):
    """Keep the updater on an isolated checkout and intercept the web build's Popen path."""
    import hermes_cli.main as cli_main
    from hermes_cli import main_web_build

    (tmp_path / ".git").mkdir(exist_ok=True)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", tmp_path, raising=False)
    noop_build = lambda *args, **kwargs: True  # noqa: E731
    monkeypatch.setattr(cli_main, "_build_web_ui", noop_build, raising=False)
    monkeypatch.setattr(main_web_build, "_build_web_ui", noop_build, raising=False)
    monkeypatch.setattr(
        main_web_build, "_web_ui_build_needed", lambda *a, **k: False, raising=False
    )
    fake_npm = lambda *a, **k: subprocess.CompletedProcess(  # noqa: E731
        [], 0, stdout="", stderr=""
    )
    monkeypatch.setattr(
        main_web_build, "_run_npm_install_deterministic", fake_npm, raising=False
    )

    # Tests that exercise ZIP fallback must override this tripwire explicitly.
    def _no_zip_fallback(*args, **kwargs):
        pytest.fail(
            "test reached _update_via_zip — the git checkout path was not "
            "isolated correctly (missing tmp .git or unexpected fallback)"
        )

    monkeypatch.setattr("hermes_cli.update_cmd._update_via_zip", _no_zip_fallback)

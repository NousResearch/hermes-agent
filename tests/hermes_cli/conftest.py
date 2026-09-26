"""Fixtures shared across hermes_cli tests."""

from __future__ import annotations
from gateway import systemd_runtime

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


@pytest.fixture(autouse=True)
def _inline_post_swap_handoff(request, monkeypatch):
    """Run the post-swap tail in-process instead of re-executing ``hermes update --post-swap``.

    ``_apply_pulled_update`` / ``_update_via_zip`` hand the rest of the run to a child
    interpreter on the pulled tree. A mocked updater flow must not spawn that child (it would
    run a real dependency sync against the worktree), so the tail runs here through the same
    payload round-trip — every step stays patchable and the payload shape is still exercised.
    Tests of the hand-off itself opt out with ``@pytest.mark.real_post_swap_handoff``.
    """
    if request.node.get_closest_marker("real_post_swap_handoff"):
        return
    try:
        from hermes_cli import update_cmd, update_receipt
    except Exception:
        return

    def _inline(args, **payload_kwargs):
        payload = update_cmd._post_swap_payload(**payload_kwargs)
        if payload["receipt"]:
            update_receipt.resume_update_receipt(payload["receipt"])
        update_cmd._execute_post_swap(payload, args, payload_kwargs["gateway_mode"])

    monkeypatch.setattr(update_cmd, "_hand_off_post_swap", _inline, raising=False)


@pytest.fixture(autouse=True)
def _discharge_host_update_obligation():
    """Start and end every ``hermes_cli`` test with NO host update-restart obligation.

    The record is host-scoped on purpose (one multiplexer per host), so it lives in the
    per-OS-USER host state dir — not in the per-test ``HERMES_HOME``. The root conftest pins
    that dir per test only when the caller supplied no ``HERMES_GATEWAY_LOCK_DIR`` (#118097
    keeps the documented override working), so with one set every test in a file shares it and
    a test that arms the obligation makes the next one read a restart it never owed. Clearing
    the record — rather than re-pinning the dir — leaves that override rule untouched.
    """

    def _clear() -> None:
        try:
            from hermes_cli.update_host_obligation import clear_host_obligation

            clear_host_obligation()
        except Exception:
            # Import/env failure here must never error an unrelated test.
            pass

    _clear()
    yield
    _clear()


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

    monkeypatch.setattr(gateway, "find_gateway_pids", lambda *a, **k: [])
    monkeypatch.setattr(gateway, "find_profile_gateway_processes", lambda *a, **k: [])
    monkeypatch.setattr(gateway, "_get_service_pids", lambda *a, **k: set())
    monkeypatch.setattr(systemd_runtime, "supports_services", lambda: False)
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


# ---- prompt_toolkit / capsys isolation ----
# ``cli._cprint`` renders through ``prompt_toolkit.print_formatted_text``,
# which — when called with no explicit ``output=`` — lazily creates an
# ``Output`` from ``sys.stdout`` **and caches it on the process-global default
# ``AppSession``** (``prompt_toolkit.application.current._current_app_session``,
# a ``ContextVar`` with a module-level default). The cache is keyed to nothing
# and never re-reads ``sys.stdout``.
#
# Under pytest, ``capsys`` swaps ``sys.stdout`` for a fresh buffer per test.
# So the first CLI test that emits through ``_cprint`` (e.g. one exercising
# ``/queue``, which prints a "Queued: …" line) locks prompt_toolkit's cached
# output onto *its* captured stdout. Every later ``capsys`` test that asserts
# on ``_cprint`` output then reads an empty buffer, because the render went to
# the first test's now-dead capture target. That is the mechanism behind the
# order-dependent ``test_resume_quiet_stderr`` failure: it passes in isolation
# and in its own file, but fails in a full ``tests/cli`` run.
#
# Reset the cached output before every CLI test so each one re-creates a fresh
# prompt_toolkit ``Output`` bound to its own ``sys.stdout`` on first use. This
# is a no-op when prompt_toolkit isn't importable and cheap otherwise (the
# property re-creates lazily).


@pytest.fixture(autouse=True)
def _reset_prompt_toolkit_output_cache():
    """Clear prompt_toolkit's cached AppSession output around each CLI test.

    See the module docstring for the capsys/prompt_toolkit interaction this
    guards against.
    """

    def _clear() -> None:
        try:
            from prompt_toolkit.application.current import get_app_session

            get_app_session()._output = None
        except Exception:
            # prompt_toolkit not importable / internal shape changed — the
            # tests that rely on this simply keep their prior behavior.
            pass

    _clear()
    yield
    _clear()

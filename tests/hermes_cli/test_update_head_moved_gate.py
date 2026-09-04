"""Tests for the post-pull HEAD-movement gate in ``hermes update``.

Issue #79678: a detached/pinned checkout can report "N new commit(s)"
against origin, run the ff-only merge successfully, and still sit on the
old commit afterward (the branch-switch step re-detaches to the raw SHA).
Before this guard ``hermes update`` printed "✓ Code updated!" and
reinstalled deps + rebuilt the desktop app against the stale tree — no
error, no warning. The gate compares the pre-pull and post-pull HEAD SHA
and fails loudly when the update was a no-op.
"""

from types import SimpleNamespace

import os

import pytest

from hermes_cli import main as hermes_main
import hermes_cli.main_web_build as main_web_build
import hermes_cli.main_install_repair as main_install_repair
from hermes_cli import update_cmd


@pytest.fixture(autouse=True)
def _isolate_venv_holders(monkeypatch):
    """The update flow's venv-holder guard sees the live gateway processes on
    a dev machine and aborts with SystemExit 2 before reaching the HEAD-move
    gate under test.  Isolate it so the test exercises the intended path."""
    monkeypatch.setattr(hermes_main, "_detect_venv_python_processes", lambda: [])


def _make_head_moved_side_effect(pre_sha="abc123", post_sha="def456"):
    """Simulate git commands where HEAD advances from pre_sha to post_sha."""
    calls = {"n": 0}

    def side_effect(cmd, **kwargs):
        joined = " ".join(str(c) for c in cmd)

        # git rev-parse --abbrev-ref HEAD  (get current branch)
        if "rev-parse" in joined and "--abbrev-ref" in joined:
            return SimpleNamespace(returncode=0, stdout="main\n", stderr="")

        # git rev-list HEAD..origin/main --count  (behind count)
        if "rev-list" in joined:
            return SimpleNamespace(returncode=0, stdout="3\n", stderr="")

        # git rev-parse HEAD  — pre-pull capture (first call) sees pre_sha;
        # post-pull capture (second call) sees post_sha. get_version_info
        # is mocked in _patch_update_deps so the startup banner makes no
        # rev-parse calls of its own.
        if joined.endswith("rev-parse HEAD"):
            if calls["n"] < 1:
                calls["n"] += 1
                return SimpleNamespace(returncode=0, stdout=f"{pre_sha}\n", stderr="")
            return SimpleNamespace(returncode=0, stdout=f"{post_sha}\n", stderr="")

        # Everything else (merge, checkout, etc.) succeeds quietly.
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return side_effect


def _make_head_pinned_side_effect(sha="abc123"):
    """Simulate a detached checkout pinned to ``sha``: HEAD never moves."""

    def side_effect(cmd, **kwargs):
        joined = " ".join(str(c) for c in cmd)

        if "rev-parse" in joined and "--abbrev-ref" in joined:
            return SimpleNamespace(returncode=0, stdout="HEAD\n", stderr="")

        if "rev-list" in joined:
            return SimpleNamespace(returncode=0, stdout="3\n", stderr="")

        if joined.endswith("rev-parse HEAD"):
            return SimpleNamespace(returncode=0, stdout=f"{sha}\n", stderr="")

        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return side_effect


def _patch_update_deps(monkeypatch, tmp_path, run_side_effect):
    """Patch the hermes_cli.main helpers ``_cmd_update_impl`` touches.

    ``_m()`` in update_cmd.py lazily returns hermes_cli.main, so patching
    attributes on that module is the canonical test surface (matches
    tests/hermes_cli/test_cmd_update.py).
    """
    monkeypatch.setattr(hermes_main.subprocess, "run", run_side_effect)
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)
    (tmp_path / ".git").mkdir()  # pass the "is a git repo" gate
    monkeypatch.setattr(
        hermes_main, "_resolve_update_branch", lambda args: "main"
    )
    monkeypatch.setattr(hermes_main, "_is_windows", lambda: False)
    monkeypatch.setattr(main_install_repair, "_is_windows", lambda: False)
    monkeypatch.setattr(
        hermes_main, "_get_origin_url",
        lambda *a, **k: "https://github.com/NousResearch/hermes-agent.git",
    )
    monkeypatch.setattr(update_cmd, "_is_fork", lambda *a, **k: False)
    monkeypatch.setattr(
        hermes_main, "_stash_local_changes_if_needed", lambda *a, **k: None
    )
    monkeypatch.setattr(hermes_main, "_clear_bytecode_cache", lambda *a, **k: 0)
    monkeypatch.setattr(
        hermes_main, "_record_bytecode_fingerprint", lambda *a, **k: None
    )
    monkeypatch.setattr(
        main_web_build, "_record_bytecode_fingerprint", lambda *a, **k: None
    )
    monkeypatch.setattr(
        hermes_main, "_run_pre_update_backup", lambda *a, **k: None
    )
    monkeypatch.setattr(
        hermes_main, "_pause_windows_gateways_for_update", lambda: None
    )
    monkeypatch.setattr(
        hermes_main, "_resume_windows_gateways_after_update", lambda *a, **k: None
    )
    # Short-circuit the long tail: dependency install + desktop build.
    monkeypatch.setattr(hermes_main, "_write_update_incomplete_marker", lambda: None)
    monkeypatch.setattr(hermes_main, "_clear_update_incomplete_marker", lambda: None)
    monkeypatch.setattr(
        hermes_main,
        "_install_hangup_protection",
        lambda gateway_mode=False: {
            "prev_stdout": None, "prev_stderr": None,
            "log_file": None, "installed": False,
        },
    )
    monkeypatch.setattr(hermes_main, "_finalize_update_output", lambda *a, **k: None)
    # _check_and_apply_config_migration → _run_migrate_config_fresh →
    # _reload_config_modules() force-reloads hermes_cli.config via
    # importlib, replacing the module object pytest's capsys + the config
    # tests' patches target. No-op the reload so the config module stays
    # stable for later tests in the same process.
    monkeypatch.setattr(
        "hermes_cli.update_cmd._reload_config_modules",
        lambda *a, **k: None,
    )
    # The startup version-info probe runs git rev-parse HEAD (and the
    # result is cached per-process, so whether it runs depends on test
    # order). Mock it so the head-moved mock's call counting only sees the
    # update flow's own pre/post captures.
    monkeypatch.setattr(
        "hermes_cli.version_info.get_version_info",
        lambda *a, **k: SimpleNamespace(),
    )
    # Upstream refactor moved _clear_update_incomplete_marker into
    # main_install_repair; no-op it there too so the marker machinery
    # stays inert on both import paths.
    monkeypatch.setattr(main_install_repair, "_clear_update_incomplete_marker", lambda: None)
    # Gateway restart path (called after a successful update).
    monkeypatch.setattr(update_cmd, "_finish_dashboard_update_cleanup", lambda *a, **k: None)
    # Keep the (now surfaced — #78574) gateway auto-restart phase away from
    # this machine's real gateways: discovery returns nothing, systemd is
    # unsupported, so the phase is a clean no-op for both snapshots.
    import hermes_cli.gateway as hermes_gateway

    monkeypatch.setattr(
        hermes_gateway,
        "find_gateway_pids",
        lambda *a, **k: [],
    )
    # The gateway restart phase also probes per-profile processes; return
    # nothing so no real pid reaches the drain/kill loop (the conftest
    # live-system guard would block the os.kill).
    monkeypatch.setattr(
        hermes_gateway, "find_profile_gateway_processes", lambda *a, **k: []
    )
    # pm-era update flow: the venv re-sync after a pull runs
    # `pm.ensure.sync_venv`, which needs a real uv/venv. The test env has
    # neither, so short-circuit it — the head-moved gate is what's under
    # test, not dependency sync. (pm.ensure the package attribute is the
    # `ensure` function; the module is reached via importlib.)
    import importlib

    _pm_ensure_mod = importlib.import_module("pm.ensure")

    monkeypatch.setattr(_pm_ensure_mod, "sync_venv", lambda *a, **k: None)
    # Fleet-restart verification (#93406): no gateways are expected (the
    # pid probe above returns nothing), so the post-restart version matrix
    # must not demand rows or fail the update.
    monkeypatch.setattr(
        hermes_main,
        "_fleet_probe_expected_runtimes",
        lambda *a, **k: False,
    )
    # The gateway restart phase imports discovery functions fresh AFTER
    # _purge_stale_hermes_modules drops every cached module (the update
    # reloads code in-place), so monkeypatching hermes_cli.gateway
    # attributes is lost. The phase may still attempt os.kill on real
    # pids; stub os.kill so the conftest live-system guard never fires —
    # this test asserts the head-moved gate, not process teardown.
    import os as _os_mod

    monkeypatch.setattr(_os_mod, "kill", lambda *a, **k: None)
    monkeypatch.setattr(
        hermes_gateway, "kill_gateway_processes", lambda *a, **k: []
    )
    monkeypatch.setattr(
        hermes_gateway, "supports_systemd_services", lambda: False
    )
    monkeypatch.setattr(
        hermes_gateway, "find_profile_gateway_processes", lambda *a, **k: []
    )
    # _restart_gateways_after_update purges and re-imports hermes_cli.gateway,
    # which undoes the discovery mocks above (the fresh import finds the live
    # gateways on this dev machine).  Neutralize the kill itself so the phase
    # is a no-op regardless of what the re-imported discovery returns.
    monkeypatch.setattr(os, "kill", lambda *a, **k: None)
    # ...and stop the purge from evicting the mocked module in the first place,
    # so find_gateway_pids stays [] and the fleet-restart phase is a clean no-op.
    monkeypatch.setattr(hermes_main, "_purge_stale_hermes_modules", lambda: None)


def test_update_success_when_head_moves(monkeypatch, tmp_path, capsys):
    """When the pull advances HEAD, the update proceeds normally."""
    args = SimpleNamespace(branch=None, yes=False, force=False, force_venv=False)
    _patch_update_deps(monkeypatch, tmp_path, _make_head_moved_side_effect())

    hermes_main.cmd_update(args)  # completes normally (no SystemExit)

    out = capsys.readouterr().out
    assert "✓ Code updated!" in out
    assert "Code did not move" not in out


def test_update_fails_loudly_when_head_pinned(monkeypatch, tmp_path, capsys):
    """A detached/pinned HEAD that never moves must fail loudly, not print
    '✓ Code updated!' against the stale tree."""
    args = SimpleNamespace(branch=None, yes=False, force=False, force_venv=False)
    _patch_update_deps(monkeypatch, tmp_path, _make_head_pinned_side_effect())

    with pytest.raises(SystemExit) as exc_info:
        hermes_main.cmd_update(args)

    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "Code did not move" in out
    assert "✓ Code updated!" not in out
    assert "checkout main" in out

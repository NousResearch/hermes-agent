from pathlib import Path
from subprocess import CalledProcessError
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli import config as hermes_config
from hermes_cli import main as hermes_main


# ---------------------------------------------------------------------------
# Managed-uv compatibility for tests that patch shutil.which
# ---------------------------------------------------------------------------
# The production code now uses ``ensure_uv()`` / ``update_managed_uv()``
# instead of ``shutil.which("uv")``.  Many tests in this file patch
# ``shutil.which`` to control whether uv is "available" — these autouse
# fixtures make the managed_uv functions delegate to the patched
# ``shutil.which`` so the existing test setup keeps working without
# per-test changes.
@pytest.fixture(autouse=True)
def _patch_managed_uv(request):
    """Make managed_uv helpers follow shutil.which mocking in tests."""
    import shutil

    # resolve_uv delegates to shutil.which("uv") so that test patches
    # on shutil.which flow through naturally.
    def _fake_resolve_uv(**kwargs):
        return shutil.which("uv")

    def _fake_ensure_uv(**kwargs):
        return shutil.which("uv")

    def _fake_update_managed_uv(**kwargs):
        return None  # never actually self-update in tests

    with patch("hermes_cli.managed_uv.resolve_uv", side_effect=_fake_resolve_uv), \
         patch("hermes_cli.managed_uv.ensure_uv", side_effect=_fake_ensure_uv), \
         patch("hermes_cli.managed_uv.update_managed_uv", side_effect=_fake_update_managed_uv):
        yield


@pytest.fixture(autouse=True)
def _patch_gateway_discovery():
    """Keep cmd_update's gateway auto-restart phase off this machine's gateways.

    Tests in this file that reach the full success path (e.g. the #87694
    orphan-history rescue-ref tests) would otherwise hit real gateway
    discovery: an unmocked ``find_gateway_pids`` on a box with a live gateway
    reaches the conftest live-system guard and turns into a spurious
    ``sys.exit(1)`` (#78574). Discovery returning nothing makes the phase a
    clean no-op — none of the tests here assert on gateway restarts.
    """
    with patch("hermes_cli.gateway.find_gateway_pids", return_value=[]), \
         patch("hermes_cli.gateway.supports_systemd_services", return_value=False), \
         patch("hermes_cli.gateway.find_profile_gateway_processes", return_value=[]):
        yield













# ---------------------------------------------------------------------------
# Update uses .[all] with fallback to .
# ---------------------------------------------------------------------------

def _setup_update_mocks(monkeypatch, tmp_path):
    """Common setup for cmd_update tests."""
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(hermes_main, "_stash_local_changes_if_needed", lambda *a, **kw: None)
    monkeypatch.setattr(hermes_main, "_restore_stashed_changes", lambda *a, **kw: True)
    monkeypatch.setattr(hermes_config, "get_missing_env_vars", lambda required_only=True: [])
    monkeypatch.setattr(hermes_config, "get_missing_config_fields", lambda: [])
    monkeypatch.setattr(hermes_config, "check_config_version", lambda: (5, 5))
    monkeypatch.setattr(hermes_config, "migrate_config", lambda **kw: {"env_added": [], "config_added": []})
    monkeypatch.setattr(hermes_main, "_upgrade_pip_before_lazy_refresh", lambda *a, **kw: None)
    monkeypatch.setattr(hermes_main, "_refresh_active_lazy_features", lambda *a, **kw: True)




def test_refresh_active_memory_provider_dependencies_reinstalls_active_provider(monkeypatch):
    """#53272/#70636: update must re-run the active provider's dep install."""
    recorded = []

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"memory": {"provider": "mem0"}},
    )
    monkeypatch.setattr(
        "hermes_cli.memory_setup._install_dependencies",
        lambda provider_name, force=False: recorded.append((provider_name, force)),
    )

    hermes_main._refresh_active_memory_provider_dependencies()

    assert recorded == [("mem0", True)]




def test_reload_updated_runtime_modules_restores_new_hermes_constants_symbol(monkeypatch):
    """A pre-pull module object missing a new helper is repaired by reload."""
    import hermes_constants

    monkeypatch.delattr(hermes_constants, "apply_subprocess_home_env", raising=False)
    assert not hasattr(hermes_constants, "apply_subprocess_home_env")

    hermes_main._reload_updated_runtime_modules()

    assert callable(hermes_constants.apply_subprocess_home_env)






# ---------------------------------------------------------------------------
# ff-only fallback to reset --hard on diverged history
# ---------------------------------------------------------------------------

def _make_update_side_effect(
    current_branch="main",
    commit_count="3",
    ff_only_fails=False,
    reset_fails=False,
    fetch_fails=False,
    fetch_stderr="",
    merge_base_exists=True,
    update_ref_fails=False,
    pre_pull_sha_unavailable=False,
    existing_rescue_refs=None,
):
    """Build a subprocess.run side_effect for cmd_update tests.

    ``merge_base_exists`` controls the ``git merge-base HEAD origin/<branch>``
    probe used by the ff-only-fallback orphan-history guard (#87694): True
    (default) simulates ordinary divergence (a common ancestor exists, e.g.
    upstream force-push), False simulates orphan/unrelated-history divergence
    (no common ancestor at all).

    ``update_ref_fails`` simulates ``git update-ref`` itself failing (disk
    full, permissions) when writing the orphan rescue ref.

    ``pre_pull_sha_unavailable`` simulates ``_capture_head_sha`` being unable
    to resolve HEAD before the pull (empty rev-parse output) — the rescue-ref
    guard requires a truthy ``pre_pull_sha`` and must degrade gracefully
    without one.

    ``existing_rescue_refs`` simulates the refs already present under
    ``refs/hermes-update-backups/orphan-<branch>-*`` (oldest first) so the
    ``_prune_orphan_rescue_refs`` cleanup pass has something to trim.
    """
    recorded = []
    head_sha_calls = []

    def side_effect(cmd, **kwargs):
        recorded.append(cmd)
        joined = " ".join(str(c) for c in cmd)
        if "fetch" in joined and "origin" in joined:
            if fetch_fails:
                return SimpleNamespace(stdout="", stderr=fetch_stderr, returncode=128)
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if "rev-parse" in joined and "--abbrev-ref" in joined:
            return SimpleNamespace(stdout=f"{current_branch}\n", stderr="", returncode=0)
        if "rev-parse" in joined and "HEAD" in joined:
            # First call = pre-pull HEAD, every later call = post-pull HEAD
            # (issue #79678's "did HEAD actually move" guard depends on these
            # differing after a successful reset/merge).
            head_sha_calls.append(1)
            if len(head_sha_calls) == 1:
                if pre_pull_sha_unavailable:
                    return SimpleNamespace(stdout="", stderr="", returncode=0)
                return SimpleNamespace(
                    stdout="1111111111111111111111111111111111111beef\n", stderr="", returncode=0
                )
            return SimpleNamespace(
                stdout="2222222222222222222222222222222222222cafe\n", stderr="", returncode=0
            )
        if "checkout" in joined and "main" in joined:
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if "rev-list" in joined:
            return SimpleNamespace(stdout=f"{commit_count}\n", stderr="", returncode=0)
        if "merge-base" in joined:
            if merge_base_exists:
                return SimpleNamespace(stdout="abc123deadbeef\n", stderr="", returncode=0)
            return SimpleNamespace(
                stdout="", stderr="fatal: Not a valid commit name origin/main\n", returncode=1
            )
        if "for-each-ref" in joined:
            refs = existing_rescue_refs or []
            return SimpleNamespace(stdout="\n".join(refs) + ("\n" if refs else ""), stderr="", returncode=0)
        if "update-ref" in joined and "-d" in cmd:
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if "update-ref" in joined:
            if update_ref_fails:
                return SimpleNamespace(
                    stdout="", stderr="fatal: unable to write ref\n", returncode=128
                )
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if "--ff-only" in joined:
            if ff_only_fails:
                return SimpleNamespace(
                    stdout="",
                    stderr="fatal: Not possible to fast-forward, aborting.\n",
                    returncode=128,
                )
            return SimpleNamespace(stdout="Updating abc..def\n", stderr="", returncode=0)
        if "reset" in joined and "--hard" in joined:
            if reset_fails:
                return SimpleNamespace(stdout="", stderr="error: unable to write\n", returncode=1)
            return SimpleNamespace(stdout="HEAD is now at abc123\n", stderr="", returncode=0)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return side_effect, recorded


# ---------------------------------------------------------------------------
# Non-main branch → auto-checkout main
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Fetch failure — friendly error messages
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# reset --hard failure — don't attempt stash restore
# ---------------------------------------------------------------------------

def test_cmd_update_skips_stash_restore_when_reset_fails(monkeypatch, tmp_path, capsys):
    """When reset --hard fails, stash restore is skipped with a helpful message."""
    _setup_update_mocks(monkeypatch, tmp_path)
    # Re-enable stash so it actually returns a ref
    monkeypatch.setattr(
        hermes_main, "_stash_local_changes_if_needed",
        lambda *a, **kw: "abc123deadbeef",
    )
    restore_calls = []
    monkeypatch.setattr(
        hermes_main, "_restore_stashed_changes",
        lambda *a, **kw: restore_calls.append(1) or True,
    )

    side_effect, _ = _make_update_side_effect(ff_only_fails=True, reset_fails=True)
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    with pytest.raises(SystemExit, match="1"):
        hermes_main.cmd_update(SimpleNamespace())

    # Stash restore should NOT have been called
    assert len(restore_calls) == 0

    out = capsys.readouterr().out
    assert "preserved in stash" in out


# ---------------------------------------------------------------------------
# #87694: orphan/unrelated-history divergence must be backed up to a rescue
# ref before `reset --hard` discards it (ordinary divergence is unaffected).
# ---------------------------------------------------------------------------

def test_cmd_update_orphan_history_backs_up_before_reset(monkeypatch, tmp_path, capsys):
    """No common ancestor with origin/<branch> → HEAD is parked behind a
    ``refs/hermes-update-backups/orphan-*`` ref before the reset proceeds."""
    _setup_update_mocks(monkeypatch, tmp_path)
    # Orphan-history detection is platform-agnostic; skip the unrelated
    # Windows gateway-fleet stop/restart machinery so this test only
    # exercises the merge-base guard.
    monkeypatch.setattr(hermes_main, "_is_windows", lambda: False)

    side_effect, recorded = _make_update_side_effect(
        ff_only_fails=True, merge_base_exists=False,
    )
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    hermes_main.cmd_update(SimpleNamespace())

    update_ref_calls = [c for c in recorded if "update-ref" in " ".join(str(x) for x in c)]
    assert len(update_ref_calls) == 1
    ref_name = update_ref_calls[0][update_ref_calls[0].index("update-ref") + 1]
    assert ref_name.startswith("refs/hermes-update-backups/orphan-main-")
    assert update_ref_calls[0][update_ref_calls[0].index("update-ref") + 2] == (
        "1111111111111111111111111111111111111beef"
    )
    # Ref name carries the pre-pull SHA, not just a second-resolution
    # timestamp, so two updates racing within the same second don't collide.
    assert ref_name.endswith("-111111111111")

    out = capsys.readouterr().out
    assert "orphan divergence" in out
    assert ref_name in out


def test_cmd_update_orphan_rescue_ref_write_failure_message_is_honest(monkeypatch, tmp_path, capsys):
    """When ``git update-ref`` fails, the printed message must not claim a
    backup exists — it should say the write was attempted and failed."""
    _setup_update_mocks(monkeypatch, tmp_path)
    monkeypatch.setattr(hermes_main, "_is_windows", lambda: False)

    side_effect, recorded = _make_update_side_effect(
        ff_only_fails=True, merge_base_exists=False, update_ref_fails=True,
    )
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    hermes_main.cmd_update(SimpleNamespace())

    out = capsys.readouterr().out
    assert "orphan divergence" in out
    assert "backup write failed" in out
    assert "backed up current HEAD" not in out


def test_cmd_update_orphan_rescue_refs_pruned_beyond_keep_limit(monkeypatch, tmp_path, capsys):
    """Old orphan rescue refs beyond the retention limit are deleted so a
    repeatedly corrupted install doesn't pin unbounded objects against gc."""
    _setup_update_mocks(monkeypatch, tmp_path)
    monkeypatch.setattr(hermes_main, "_is_windows", lambda: False)

    stale_refs = [
        f"refs/hermes-update-backups/orphan-main-2026010{i}-000000-abc"
        for i in range(1, hermes_main._ORPHAN_RESCUE_REFS_TO_KEEP + 3)
    ]
    side_effect, recorded = _make_update_side_effect(
        ff_only_fails=True, merge_base_exists=False, existing_rescue_refs=stale_refs,
    )
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    hermes_main.cmd_update(SimpleNamespace())

    delete_calls = [
        c for c in recorded
        if "update-ref" in " ".join(str(x) for x in c) and "-d" in c
    ]
    assert len(delete_calls) == len(stale_refs) - hermes_main._ORPHAN_RESCUE_REFS_TO_KEEP
    deleted_refs = {c[c.index("-d") + 1] for c in delete_calls}
    assert deleted_refs == set(stale_refs[: len(stale_refs) - hermes_main._ORPHAN_RESCUE_REFS_TO_KEEP])


def test_cmd_update_ordinary_divergence_skips_rescue_ref(monkeypatch, tmp_path, capsys):
    """Common ancestor still exists (e.g. upstream force-push) → no rescue
    ref, no orphan messaging, behavior identical to before #87694."""
    _setup_update_mocks(monkeypatch, tmp_path)
    monkeypatch.setattr(hermes_main, "_is_windows", lambda: False)

    side_effect, recorded = _make_update_side_effect(
        ff_only_fails=True, merge_base_exists=True,
    )
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    hermes_main.cmd_update(SimpleNamespace())

    update_ref_calls = [c for c in recorded if "update-ref" in " ".join(str(x) for x in c)]
    assert update_ref_calls == []

    out = capsys.readouterr().out
    assert "orphan divergence" not in out
    assert "Fast-forward not possible (history diverged), resetting to match remote" in out


def test_cmd_update_orphan_rescue_ref_write_failure_is_non_fatal(monkeypatch, tmp_path, capsys):
    """#87694 stress test: ``git update-ref`` itself fails (disk full,
    permissions) while parking the orphan rescue ref. The backup attempt is
    best-effort — its return code is intentionally not checked — so the
    reset must still proceed and the update must still succeed."""
    _setup_update_mocks(monkeypatch, tmp_path)
    monkeypatch.setattr(hermes_main, "_is_windows", lambda: False)

    side_effect, recorded = _make_update_side_effect(
        ff_only_fails=True, merge_base_exists=False, update_ref_fails=True,
    )
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    hermes_main.cmd_update(SimpleNamespace())

    update_ref_calls = [c for c in recorded if "update-ref" in " ".join(str(x) for x in c)]
    assert len(update_ref_calls) == 1

    reset_calls = [
        c for c in recorded if "reset" in " ".join(str(x) for x in c) and "--hard" in c
    ]
    assert len(reset_calls) == 1

    out = capsys.readouterr().out
    assert "orphan divergence" in out


def test_cmd_update_orphan_guard_skips_rescue_ref_when_pre_pull_sha_missing(
    monkeypatch, tmp_path, capsys
):
    """#87694 stress test: if capturing the pre-pull HEAD SHA itself fails
    (empty ``rev-parse HEAD`` output), the rescue-ref guard requires a
    truthy ``pre_pull_sha`` and must skip the backup rather than writing a
    ref pointing at nothing — the reset must still proceed without crashing.
    """
    _setup_update_mocks(monkeypatch, tmp_path)
    monkeypatch.setattr(hermes_main, "_is_windows", lambda: False)

    side_effect, recorded = _make_update_side_effect(
        ff_only_fails=True, merge_base_exists=False, pre_pull_sha_unavailable=True,
    )
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    hermes_main.cmd_update(SimpleNamespace())

    update_ref_calls = [c for c in recorded if "update-ref" in " ".join(str(x) for x in c)]
    assert update_ref_calls == []

    out = capsys.readouterr().out
    assert "orphan divergence" not in out


def test_cmd_update_orphan_rescue_ref_persists_when_reset_fails(monkeypatch, tmp_path, capsys):
    """#87694 stress test: even when the subsequent ``reset --hard`` itself
    fails, the rescue ref must already have been written — the backup is
    not lost just because the overall update aborts."""
    _setup_update_mocks(monkeypatch, tmp_path)
    monkeypatch.setattr(hermes_main, "_is_windows", lambda: False)

    side_effect, recorded = _make_update_side_effect(
        ff_only_fails=True, merge_base_exists=False, reset_fails=True,
    )
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    with pytest.raises(SystemExit) as exc_info:
        hermes_main.cmd_update(SimpleNamespace())
    assert exc_info.value.code == 1

    update_ref_calls = [c for c in recorded if "update-ref" in " ".join(str(x) for x in c)]
    assert len(update_ref_calls) == 1

    out = capsys.readouterr().out
    assert "orphan divergence" in out
    assert "Failed to reset to origin/main" in out


# ---------------------------------------------------------------------------
# Non-interactive update.non_interactive_local_changes setting
# (chat app / gateway): "discard" throws stashed changes away, "stash"
# (default) restores them. Interactive terminal updates ignore the setting
# and always go through the restore path.
# ---------------------------------------------------------------------------

def _setup_setting_test(monkeypatch, tmp_path, mode):
    """Common wiring: real stash returns a ref, restore + discard are
    recorded, and load_config reports the given non_interactive_local_changes
    mode."""
    _setup_update_mocks(monkeypatch, tmp_path)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(
        hermes_main, "_stash_local_changes_if_needed",
        lambda *a, **kw: "abc123deadbeef",
    )
    restore_calls = []
    discard_calls = []
    monkeypatch.setattr(
        hermes_main, "_restore_stashed_changes",
        lambda *a, **kw: restore_calls.append(1) or True,
    )
    monkeypatch.setattr(
        hermes_main, "_discard_stashed_changes",
        lambda *a, **kw: discard_calls.append(1) or True,
    )
    monkeypatch.setattr(
        hermes_config, "load_config",
        lambda *a, **kw: {"updates": {"non_interactive_local_changes": mode}},
    )
    side_effect, recorded = _make_update_side_effect()
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)
    return restore_calls, discard_calls, recorded


# ---------------------------------------------------------------------------
# --keep-stash (desktop updater): stash for the update, never re-apply.
# ---------------------------------------------------------------------------

def _setup_keep_stash_test(monkeypatch, tmp_path):
    """Wiring for --keep-stash tests: stash returns a ref; restore, discard,
    and park are all recorded."""
    _setup_update_mocks(monkeypatch, tmp_path)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(
        hermes_main, "_stash_local_changes_if_needed",
        lambda *a, **kw: "abc123deadbeef",
    )
    restore_calls = []
    discard_calls = []
    park_calls = []
    monkeypatch.setattr(
        hermes_main, "_restore_stashed_changes",
        lambda *a, **kw: restore_calls.append(1) or True,
    )
    monkeypatch.setattr(
        hermes_main, "_discard_stashed_changes",
        lambda *a, **kw: discard_calls.append(1) or True,
    )
    monkeypatch.setattr(
        hermes_main, "_park_stashed_changes",
        lambda *a, **kw: park_calls.append(a) or None,
    )
    # Keep the update flow away from the real gateway fleet on this machine —
    # a live gateway PID would trip the test-suite kill guard and turn the
    # run into exit 1 (gateway_fleet_restart_incomplete).
    monkeypatch.setattr(
        "hermes_cli.gateway.find_gateway_pids", lambda **kw: [], raising=False
    )
    return restore_calls, discard_calls, park_calls


def test_update_keep_stash_parks_instead_of_restoring(monkeypatch, tmp_path):
    """--keep-stash: after a successful update, the autostash is parked (left
    in git stash) — never re-applied, never discarded."""
    restore_calls, discard_calls, park_calls = _setup_keep_stash_test(monkeypatch, tmp_path)
    side_effect, _ = _make_update_side_effect()
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    hermes_main.cmd_update(SimpleNamespace(yes=True, keep_stash=True))

    assert len(park_calls) == 1
    assert park_calls[0][0] == "abc123deadbeef"
    assert restore_calls == []
    assert discard_calls == []


def test_update_without_keep_stash_still_restores(monkeypatch, tmp_path):
    """Regression guard: default behavior (no --keep-stash) is unchanged —
    the autostash is auto-restored under --yes."""
    restore_calls, discard_calls, park_calls = _setup_keep_stash_test(monkeypatch, tmp_path)
    side_effect, _ = _make_update_side_effect()
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    hermes_main.cmd_update(SimpleNamespace(yes=True, keep_stash=False))

    assert restore_calls == [1]
    assert park_calls == []
    assert discard_calls == []


def test_update_keep_stash_failure_path_still_preserves(monkeypatch, tmp_path, capsys):
    """--keep-stash + failed update: neither restore nor park runs; the
    existing preserved-in-stash message fires (working tree unknown)."""
    restore_calls, discard_calls, park_calls = _setup_keep_stash_test(monkeypatch, tmp_path)
    side_effect, _ = _make_update_side_effect(ff_only_fails=True, reset_fails=True)
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    with pytest.raises(SystemExit, match="1"):
        hermes_main.cmd_update(SimpleNamespace(yes=True, keep_stash=True))

    assert restore_calls == []
    assert park_calls == []
    assert discard_calls == []
    assert "preserved in stash" in capsys.readouterr().out


def test_update_parser_accepts_keep_stash():
    """The flag parses and defaults off."""
    import argparse

    from hermes_cli.subcommands.update import build_update_parser

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    build_update_parser(subparsers, cmd_update=lambda args: None)

    args = parser.parse_args(["update", "--keep-stash"])
    assert args.keep_stash is True
    args = parser.parse_args(["update"])
    assert args.keep_stash is False






def test_bootstrap_marker_not_autostashed_by_update(tmp_path):
    """#38529: the Desktop bootstrap marker must be git-ignored so that
    ``hermes update``'s ``git stash push --include-untracked`` does not sweep it
    into an autostash on every run.

    Behavioral + hermetic: build a throwaway repo that adopts the project's real
    ``.gitignore`` (the contract under test), drop the marker, and confirm the
    same stash invocation the updater uses leaves it untouched.
    """
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git not available")

    repo_gitignore = Path(hermes_main.__file__).resolve().parents[1] / ".gitignore"

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=tmp_path, capture_output=True, text=True, check=True
        )

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / ".gitignore").write_text(repo_gitignore.read_text())
    (tmp_path / "tracked.txt").write_text("x\n")
    git("add", "-A")
    git("commit", "-qm", "init")

    marker = tmp_path / ".hermes-bootstrap-complete"
    marker.write_text("")

    # Exact flags used by hermes update (hermes_cli/main.py).
    git("stash", "push", "--include-untracked", "-m", "hermes-update-autostash")

    assert marker.exists(), (
        ".hermes-bootstrap-complete was swept into the update autostash — it must "
        "be listed in .gitignore so `git stash -u` skips it (#38529)."
    )
    # It must not even register as a dirty/untracked change.
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=tmp_path, capture_output=True, text=True
    ).stdout
    assert ".hermes-bootstrap-complete" not in status


# ---------------------------------------------------------------------------
# Permission-denied autostash class: undeletable untracked files (root-owned
# packaging/ etc.) must not abort the update when the stash entry was created.
# ---------------------------------------------------------------------------






def test_update_autostash_survives_undeletable_untracked_dir(tmp_path):
    """Behavioral E2E of the whole permission-denied class with real git:
    root-owned-style undeletable untracked dir → stash succeeds, update-style
    reset works, restore round-trips, nothing lost. (#70127 follow-up)"""
    import os
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git not available")
    if os.name == "nt":
        pytest.skip("POSIX permission semantics")
    if os.geteuid() == 0:
        pytest.skip("root ignores directory write bits")

    def git(*args, check=True):
        return subprocess.run(
            ["git", *args], cwd=tmp_path, capture_output=True, text=True, check=check
        )

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "tracked.txt").write_text("v1\n")
    git("add", "-A")
    git("commit", "-qm", "init")

    (tmp_path / "tracked.txt").write_text("v2 local change\n")
    pkg = tmp_path / "packaging" / "homebrew"
    pkg.mkdir(parents=True)
    (pkg / "hermes-agent.rb").write_text("formula\n")
    os.chmod(pkg, 0o555)  # undeletable contents, like a root-owned dir
    try:
        stash_ref = hermes_main._stash_local_changes_if_needed(["git"], tmp_path)
        assert stash_ref

        # The tracked change is stashed; simulate the updater's checkout window.
        assert (tmp_path / "tracked.txt").read_text() == "v1\n"

        restored = hermes_main._restore_stashed_changes(
            ["git"], tmp_path, stash_ref, prompt_user=False
        )
        assert restored is True
        assert (tmp_path / "tracked.txt").read_text() == "v2 local change\n"
        assert (pkg / "hermes-agent.rb").read_text() == "formula\n"
    finally:
        os.chmod(pkg, 0o755)


# ---------------------------------------------------------------------------
# #87694: real-git sanity check for the merge-base premise the orphan guard
# relies on — two independently-initialized repos (no shared history) must
# report no merge-base, while an ordinary branch divergence must report one.
# ---------------------------------------------------------------------------

def test_merge_base_detects_orphan_vs_ordinary_divergence_with_real_git(tmp_path):
    """Anchors the assumption behind the #87694 orphan-history guard: `git
    merge-base` fails/empties on truly unrelated histories, and succeeds on
    ordinary (e.g. force-pushed) divergence. If a future git version changes
    this contract, this test breaks instead of the guard silently going
    inert."""
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git not available")

    def git(cwd, *args, check=True):
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=check
        )

    # Ordinary divergence: two branches of the SAME repo, one reset to an
    # earlier point then given a new commit (simulates an upstream force-push).
    ordinary = tmp_path / "ordinary"
    ordinary.mkdir()
    git(ordinary, "init", "-q", "-b", "main")
    git(ordinary, "config", "user.email", "t@example.com")
    git(ordinary, "config", "user.name", "t")
    (ordinary / "f.txt").write_text("v1\n")
    git(ordinary, "add", "-A")
    git(ordinary, "commit", "-qm", "init")
    git(ordinary, "checkout", "-qb", "origin-main")
    (ordinary / "f.txt").write_text("v2\n")
    git(ordinary, "add", "-A")
    git(ordinary, "commit", "-qm", "upstream")
    git(ordinary, "checkout", "-q", "main")
    result = git(ordinary, "merge-base", "HEAD", "origin-main", check=False)
    assert result.returncode == 0
    assert result.stdout.strip()

    # Orphan divergence: two independently-init'd repos merged as remotes,
    # sharing zero history.
    orphan = tmp_path / "orphan"
    orphan.mkdir()
    git(orphan, "init", "-q", "-b", "main")
    git(orphan, "config", "user.email", "t@example.com")
    git(orphan, "config", "user.name", "t")
    (orphan / "f.txt").write_text("local\n")
    git(orphan, "add", "-A")
    git(orphan, "commit", "-qm", "local init")

    remote = tmp_path / "orphan-remote"
    remote.mkdir()
    git(remote, "init", "-q", "-b", "main")
    git(remote, "config", "user.email", "t@example.com")
    git(remote, "config", "user.name", "t")
    (remote / "f.txt").write_text("remote\n")
    git(remote, "add", "-A")
    git(remote, "commit", "-qm", "remote init")

    git(orphan, "remote", "add", "origin", str(remote))
    git(orphan, "fetch", "-q", "origin")
    result = git(orphan, "merge-base", "HEAD", "origin/main", check=False)
    assert result.returncode != 0
    assert not result.stdout.strip()

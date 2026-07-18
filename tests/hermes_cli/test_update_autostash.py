import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from subprocess import CalledProcessError
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# The canonical runner intentionally starts from an almost-empty environment.
# Windows' ``Path.home()`` needs USERPROFILE, which that runner does not carry;
# give module import a harmless test-only Hermes root outside the checkout until
# fixtures replace it with their per-test directories. Keeping this under the
# OS temp root also prevents an interrupted test run from dirtying the repo.
_MODULE_TEST_ROOT = Path(tempfile.gettempdir()) / f"hermes-update-tests-{os.getpid()}"
os.environ.setdefault("HERMES_HOME", str(_MODULE_TEST_ROOT / "hermes-home"))
os.environ.setdefault("LOCALAPPDATA", str(_MODULE_TEST_ROOT / "local-appdata"))
os.environ.setdefault("USERPROFILE", str(_MODULE_TEST_ROOT / "user-profile"))

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
    def _fake_resolve_uv():
        return shutil.which("uv")

    def _fake_ensure_uv():
        return shutil.which("uv")

    def _fake_update_managed_uv():
        return None  # never actually self-update in tests

    with patch("hermes_cli.managed_uv.resolve_uv", side_effect=_fake_resolve_uv), \
         patch("hermes_cli.managed_uv.ensure_uv", side_effect=_fake_ensure_uv), \
         patch("hermes_cli.managed_uv.update_managed_uv", side_effect=_fake_update_managed_uv):
        yield

def test_stash_local_changes_if_needed_returns_none_when_tree_clean(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if cmd[-2:] == ["status", "--porcelain"]:
            return SimpleNamespace(stdout="", returncode=0)
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)

    stash_ref = hermes_main._stash_local_changes_if_needed(["git"], tmp_path)

    assert stash_ref is None
    assert [cmd[-2:] for cmd, _ in calls] == [["status", "--porcelain"]]


def test_stash_local_changes_if_needed_returns_specific_stash_commit(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if cmd[-2:] == ["status", "--porcelain"]:
            return SimpleNamespace(stdout=" M hermes_cli/main.py\n?? notes.txt\n", returncode=0)
        if cmd[-2:] == ["ls-files", "--unmerged"]:
            return SimpleNamespace(stdout="", returncode=0)
        if cmd[1:4] == ["stash", "push", "--include-untracked"]:
            return SimpleNamespace(stdout="Saved working directory\n", returncode=0)
        if cmd[-3:] == ["rev-parse", "--verify", "refs/stash"]:
            return SimpleNamespace(stdout="abc123\n", returncode=0)
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)

    stash_ref = hermes_main._stash_local_changes_if_needed(["git"], tmp_path)

    assert stash_ref == "abc123"
    assert calls[1][0][-2:] == ["ls-files", "--unmerged"]
    assert calls[2][0][1:4] == ["stash", "push", "--include-untracked"]
    assert calls[3][0][-3:] == ["rev-parse", "--verify", "refs/stash"]


def test_resolve_stash_selector_returns_matching_entry(monkeypatch, tmp_path):
    def fake_run(cmd, **kwargs):
        assert cmd == ["git", "stash", "list", "--format=%gd %H"]
        return SimpleNamespace(
            stdout="stash@{0} def456\nstash@{1} abc123\n",
            returncode=0,
        )

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)

    assert hermes_main._resolve_stash_selector(["git"], tmp_path, "abc123") == "stash@{1}"



def test_restore_stashed_changes_prompts_before_applying(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if cmd[1:3] == ["stash", "apply"]:
            return SimpleNamespace(stdout="applied\n", stderr="", returncode=0)
        if cmd[1:3] == ["diff", "--name-only"]:
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if cmd[1:3] == ["stash", "list"]:
            return SimpleNamespace(stdout="stash@{1} abc123\n", stderr="", returncode=0)
        if cmd[1:3] == ["stash", "drop"]:
            return SimpleNamespace(stdout="dropped\n", stderr="", returncode=0)
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)
    monkeypatch.setattr("builtins.input", lambda: "")

    restored = hermes_main._restore_stashed_changes(["git"], tmp_path, "abc123", prompt_user=True)

    assert restored is True
    assert calls[0][0] == ["git", "stash", "apply", "abc123"]
    assert calls[1][0] == ["git", "diff", "--name-only", "--diff-filter=U"]
    assert calls[2][0] == ["git", "stash", "list", "--format=%gd %H"]
    assert calls[3][0] == ["git", "stash", "drop", "stash@{1}"]
    out = capsys.readouterr().out
    assert "Restore local changes now? [Y/n]" in out
    assert "restored on top of the updated codebase" in out
    assert "git diff" in out
    assert "git status" in out


def test_restore_stashed_changes_can_skip_restore_and_keep_stash(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)
    monkeypatch.setattr("builtins.input", lambda: "n")

    restored = hermes_main._restore_stashed_changes(["git"], tmp_path, "abc123", prompt_user=True)

    assert restored is False
    assert calls == []
    out = capsys.readouterr().out
    assert "Restore local changes now? [Y/n]" in out
    assert "Your changes are still preserved in git stash." in out
    assert "git stash apply abc123" in out


def test_restore_stashed_changes_applies_without_prompt_when_disabled(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if cmd[1:3] == ["stash", "apply"]:
            return SimpleNamespace(stdout="applied\n", stderr="", returncode=0)
        if cmd[1:3] == ["diff", "--name-only"]:
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if cmd[1:3] == ["stash", "list"]:
            return SimpleNamespace(stdout="stash@{0} abc123\n", stderr="", returncode=0)
        if cmd[1:3] == ["stash", "drop"]:
            return SimpleNamespace(stdout="dropped\n", stderr="", returncode=0)
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)

    restored = hermes_main._restore_stashed_changes(["git"], tmp_path, "abc123", prompt_user=False)

    assert restored is True
    assert calls[0][0] == ["git", "stash", "apply", "abc123"]
    assert calls[1][0] == ["git", "diff", "--name-only", "--diff-filter=U"]
    assert calls[2][0] == ["git", "stash", "list", "--format=%gd %H"]
    assert calls[3][0] == ["git", "stash", "drop", "stash@{0}"]
    assert "Restore local changes now?" not in capsys.readouterr().out



def test_print_stash_cleanup_guidance_with_selector(capsys):
    hermes_main._print_stash_cleanup_guidance("abc123", "stash@{2}")

    out = capsys.readouterr().out
    assert "Check `git status` first" in out
    assert "git stash list --format='%gd %H %s'" in out
    assert "git stash drop stash@{2}" in out



def test_restore_stashed_changes_keeps_going_when_stash_entry_cannot_be_resolved(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if cmd[1:3] == ["stash", "apply"]:
            return SimpleNamespace(stdout="applied\n", stderr="", returncode=0)
        if cmd[1:3] == ["diff", "--name-only"]:
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if cmd[1:3] == ["stash", "list"]:
            return SimpleNamespace(stdout="stash@{0} def456\n", stderr="", returncode=0)
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)

    restored = hermes_main._restore_stashed_changes(["git"], tmp_path, "abc123", prompt_user=False)

    assert restored is True
    assert calls[0] == (["git", "stash", "apply", "abc123"], {"cwd": tmp_path, "capture_output": True, "text": True})
    assert calls[1] == (["git", "diff", "--name-only", "--diff-filter=U"], {"cwd": tmp_path, "capture_output": True, "text": True})
    assert calls[2] == (["git", "stash", "list", "--format=%gd %H"], {"cwd": tmp_path, "capture_output": True, "text": True, "check": True})
    out = capsys.readouterr().out
    assert "couldn't find the stash entry to drop" in out
    assert "stash was left in place" in out
    assert "Check `git status` first" in out
    assert "git stash list --format='%gd %H %s'" in out
    assert "Look for commit abc123" in out



def test_restore_stashed_changes_keeps_going_when_drop_fails(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if cmd[1:3] == ["stash", "apply"]:
            return SimpleNamespace(stdout="applied\n", stderr="", returncode=0)
        if cmd[1:3] == ["diff", "--name-only"]:
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if cmd[1:3] == ["stash", "list"]:
            return SimpleNamespace(stdout="stash@{0} abc123\n", stderr="", returncode=0)
        if cmd[1:3] == ["stash", "drop"]:
            return SimpleNamespace(stdout="", stderr="drop failed\n", returncode=1)
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)

    restored = hermes_main._restore_stashed_changes(["git"], tmp_path, "abc123", prompt_user=False)

    assert restored is True
    assert calls[3][0] == ["git", "stash", "drop", "stash@{0}"]
    out = capsys.readouterr().out
    assert "couldn't drop the saved stash entry" in out
    assert "drop failed" in out
    assert "Check `git status` first" in out
    assert "git stash list --format='%gd %H %s'" in out
    assert "git stash drop stash@{0}" in out


def test_restore_stashed_changes_always_resets_on_conflict(monkeypatch, tmp_path, capsys):
    """Conflicts always auto-reset (no prompt) and return False, even interactively.

    Leaving conflict markers in source files makes hermes unrunnable (SyntaxError).
    The stash is preserved for manual recovery; cmd_update continues normally.
    """
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if cmd[1:3] == ["stash", "apply"]:
            return SimpleNamespace(stdout="conflict output\n", stderr="conflict stderr\n", returncode=1)
        if cmd[1:3] == ["diff", "--name-only"]:
            return SimpleNamespace(stdout="hermes_cli/main.py\n", stderr="", returncode=0)
        if cmd[1:3] == ["reset", "--hard"]:
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)
    monkeypatch.setattr("builtins.input", lambda: "y")

    result = hermes_main._restore_stashed_changes(["git"], tmp_path, "abc123", prompt_user=True)

    assert result is False
    out = capsys.readouterr().out
    assert "Conflicted files:" in out
    assert "hermes_cli/main.py" in out
    assert "stashed changes are preserved" in out
    assert "Working tree reset to clean state" in out
    assert "git stash apply abc123" in out
    reset_calls = [c for c, _ in calls if c[1:3] == ["reset", "--hard"]]
    assert len(reset_calls) == 1


def test_restore_stashed_changes_auto_resets_non_interactive(monkeypatch, tmp_path, capsys):
    """Non-interactive mode auto-resets without prompting and returns False
    instead of sys.exit(1) so the update can continue (gateway /update path)."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if cmd[1:3] == ["stash", "apply"]:
            return SimpleNamespace(stdout="applied\n", stderr="", returncode=0)
        if cmd[1:3] == ["diff", "--name-only"]:
            return SimpleNamespace(stdout="cli.py\n", stderr="", returncode=0)
        if cmd[1:3] == ["reset", "--hard"]:
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)

    result = hermes_main._restore_stashed_changes(["git"], tmp_path, "abc123", prompt_user=False)

    assert result is False
    out = capsys.readouterr().out
    assert "Working tree reset to clean state" in out
    reset_calls = [c for c, _ in calls if c[1:3] == ["reset", "--hard"]]
    assert len(reset_calls) == 1


def test_stash_local_changes_if_needed_raises_when_stash_ref_missing(monkeypatch, tmp_path):
    def fake_run(cmd, **kwargs):
        if cmd[-2:] == ["status", "--porcelain"]:
            return SimpleNamespace(stdout=" M hermes_cli/main.py\n", returncode=0)
        if cmd[-2:] == ["ls-files", "--unmerged"]:
            return SimpleNamespace(stdout="", returncode=0)
        if cmd[1:4] == ["stash", "push", "--include-untracked"]:
            return SimpleNamespace(stdout="Saved working directory\n", returncode=0)
        if cmd[-3:] == ["rev-parse", "--verify", "refs/stash"]:
            raise CalledProcessError(returncode=128, cmd=cmd)
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)

    with pytest.raises(CalledProcessError):
        hermes_main._stash_local_changes_if_needed(["git"], Path(tmp_path))


def test_discard_lockfile_churn_skips_lock_when_package_json_dirty(tmp_path):
    """Intentional dependency edits update package.json and lockfile together."""
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git not available")

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=tmp_path, capture_output=True, text=True, check=True
        )

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "package.json").write_text('{"dependencies":{"a":"1"}}\n')
    (tmp_path / "package-lock.json").write_text('{"lock":"old"}\n')
    git("add", "package.json", "package-lock.json")
    git("commit", "-qm", "init")

    (tmp_path / "package.json").write_text('{"dependencies":{"a":"2"}}\n')
    (tmp_path / "package-lock.json").write_text('{"lock":"new"}\n')

    hermes_main._discard_lockfile_churn(["git"], tmp_path)

    assert (tmp_path / "package-lock.json").read_text() == '{"lock":"new"}\n'


def test_discard_lockfile_churn_restores_lock_when_package_json_clean(tmp_path):
    """Runtime npm lockfile rewrites are still discarded on managed updates."""
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git not available")

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=tmp_path, capture_output=True, text=True, check=True
        )

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "package.json").write_text('{"dependencies":{"a":"1"}}\n')
    (tmp_path / "package-lock.json").write_text('{"lock":"old"}\n')
    git("add", "package.json", "package-lock.json")
    git("commit", "-qm", "init")

    (tmp_path / "package-lock.json").write_text('{"lock":"runtime-churn"}\n')

    hermes_main._discard_lockfile_churn(["git"], tmp_path)

    assert (tmp_path / "package-lock.json").read_text() == '{"lock":"old"}\n'


# ---------------------------------------------------------------------------
# Update uses .[all] with fallback to .
# ---------------------------------------------------------------------------

def _setup_update_mocks(monkeypatch, tmp_path):
    """Common setup for cmd_update tests."""
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)
    # A unit test must never inspect or restart the machine's real Windows
    # gateway/autostart tasks. Keep the updater lifecycle inside this process;
    # dedicated gateway tests cover pause/resume separately.
    monkeypatch.setattr(
        hermes_main, "_pause_windows_gateways_for_update", lambda: None
    )
    monkeypatch.setattr(
        hermes_main, "_resume_windows_gateways_after_update", lambda _token: None
    )
    monkeypatch.setattr(hermes_main, "_stash_local_changes_if_needed", lambda *a, **kw: None)
    monkeypatch.setattr(hermes_main, "_restore_stashed_changes", lambda *a, **kw: True)
    monkeypatch.setattr(
        hermes_main, "_git_update_commit_sha", lambda *a, **kw: "abc123"
    )
    monkeypatch.setattr(hermes_config, "get_missing_env_vars", lambda required_only=True: [])
    monkeypatch.setattr(hermes_config, "get_missing_config_fields", lambda: [])
    monkeypatch.setattr(hermes_config, "check_config_version", lambda: (5, 5))
    monkeypatch.setattr(hermes_config, "migrate_config", lambda **kw: {"env_added": [], "config_added": []})
    monkeypatch.setattr(hermes_main, "_refresh_active_lazy_features", lambda: None)


def test_cmd_update_retries_optional_extras_individually_when_all_fails(monkeypatch, tmp_path, capsys):
    """When .[all] fails, update should keep base deps and retry extras individually."""
    _setup_update_mocks(monkeypatch, tmp_path)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(hermes_main, "_is_termux_env", lambda env=None: False)
    monkeypatch.setattr(hermes_main, "_load_installable_optional_extras", lambda group="all": ["matrix", "mcp"])

    recorded = []

    def fake_run(cmd, **kwargs):
        recorded.append(cmd)
        if cmd[-3:] == ["fetch", "origin", "main"]:
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if cmd[-3:] == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return SimpleNamespace(stdout="main\n", stderr="", returncode=0)
        if cmd[-3:] == ["rev-list", "HEAD..origin/main", "--count"]:
            return SimpleNamespace(stdout="1\n", stderr="", returncode=0)
        if cmd[-4:] == ["pull", "--ff-only", "origin", "main"]:
            return SimpleNamespace(stdout="Updating\n", stderr="", returncode=0)
        if cmd == ["/usr/bin/uv", "pip", "install", "-e", ".[all]"]:
            raise CalledProcessError(returncode=1, cmd=cmd)
        if cmd == ["/usr/bin/uv", "pip", "install", "-e", "."]:
            return SimpleNamespace(returncode=0)
        if cmd == ["/usr/bin/uv", "pip", "install", "-e", ".[matrix]"]:
            raise CalledProcessError(returncode=1, cmd=cmd)
        if cmd == ["/usr/bin/uv", "pip", "install", "-e", ".[mcp]"]:
            return SimpleNamespace(returncode=0)
        # Catch-all must include stdout/stderr so consumers that parse
        # output (e.g. the dashboard-restart `ps -A` scan added in the
        # updater) don't crash on AttributeError.
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)

    hermes_main.cmd_update(SimpleNamespace())

    install_cmds = [c for c in recorded if "pip" in c and "install" in c]
    assert install_cmds == [
        ["/usr/bin/uv", "pip", "install", "-e", ".[all]"],
        ["/usr/bin/uv", "pip", "install", "-e", "."],
        ["/usr/bin/uv", "pip", "install", "-e", ".[matrix]"],
        ["/usr/bin/uv", "pip", "install", "-e", ".[mcp]"],
    ]

    out = capsys.readouterr().out
    assert "retrying extras individually" in out
    assert "Reinstalled optional extras individually: mcp" in out
    assert "Skipped optional extras that still failed: matrix" in out


def test_cmd_update_succeeds_with_extras(monkeypatch, tmp_path):
    """When .[all] succeeds, no fallback should be attempted."""
    _setup_update_mocks(monkeypatch, tmp_path)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(hermes_main, "_is_termux_env", lambda env=None: False)

    recorded = []

    def fake_run(cmd, **kwargs):
        recorded.append(cmd)
        if cmd[-3:] == ["fetch", "origin", "main"]:
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if cmd[-3:] == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return SimpleNamespace(stdout="main\n", stderr="", returncode=0)
        if cmd[-3:] == ["rev-list", "HEAD..origin/main", "--count"]:
            return SimpleNamespace(stdout="1\n", stderr="", returncode=0)
        if cmd[-4:] == ["pull", "--ff-only", "origin", "main"]:
            return SimpleNamespace(stdout="Updating\n", stderr="", returncode=0)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)

    hermes_main.cmd_update(SimpleNamespace())

    install_cmds = [c for c in recorded if "pip" in c and "install" in c]
    assert len(install_cmds) == 1
    assert ".[all]" in install_cmds[0]


def test_install_with_optional_fallback_honors_custom_group(monkeypatch):
    """Termux update path should target .[termux-all] when requested."""
    calls = []
    monkeypatch.setattr(
        hermes_main,
        "_load_installable_optional_extras",
        lambda group="all": ["termux", "mcp"] if group == "termux-all" else [],
    )

    def fake_run_with_heartbeat(cmd, **kwargs):
        calls.append(cmd)
        if cmd[-1] == ".[termux-all]":
            raise CalledProcessError(returncode=1, cmd=cmd)
        return None

    monkeypatch.setattr(hermes_main, "_run_install_with_heartbeat", fake_run_with_heartbeat)

    hermes_main._install_python_dependencies_with_optional_fallback(
        ["/usr/bin/uv", "pip"],
        group="termux-all",
    )

    assert calls == [
        ["/usr/bin/uv", "pip", "install", "-e", ".[termux-all]"],
        ["/usr/bin/uv", "pip", "install", "-e", "."],
        ["/usr/bin/uv", "pip", "install", "-e", ".[termux]"],
        ["/usr/bin/uv", "pip", "install", "-e", ".[mcp]"],
    ]


def test_install_heartbeat_prints_when_dependency_install_is_silent(monkeypatch, capsys):
    """Long quiet installs should emit periodic heartbeat lines."""

    def fake_run(cmd, **kwargs):
        hermes_main._time.sleep(1.2)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)

    hermes_main._run_install_with_heartbeat(
        ["uv", "pip", "install", "-e", "."],
        heartbeat_interval_seconds=1,
    )

    out = capsys.readouterr().out
    assert "still installing dependencies" in out


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
):
    """Build a subprocess.run side_effect for cmd_update tests."""
    recorded = []

    def side_effect(cmd, **kwargs):
        recorded.append(cmd)
        joined = " ".join(str(c) for c in cmd)
        if "fetch" in joined and "origin" in joined:
            if fetch_fails:
                return SimpleNamespace(stdout="", stderr=fetch_stderr, returncode=128)
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if "rev-parse" in joined and "--abbrev-ref" in joined:
            return SimpleNamespace(stdout=f"{current_branch}\n", stderr="", returncode=0)
        if "checkout" in joined and "main" in joined:
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if "rev-list" in joined:
            return SimpleNamespace(stdout=f"{commit_count}\n", stderr="", returncode=0)
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


def test_cmd_update_falls_back_to_reset_when_ff_only_fails(monkeypatch, tmp_path, capsys):
    """When --ff-only fails (diverged history), update resets to origin/{branch}."""
    _setup_update_mocks(monkeypatch, tmp_path)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)

    side_effect, recorded = _make_update_side_effect(ff_only_fails=True)
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    hermes_main.cmd_update(SimpleNamespace())

    reset_calls = [c for c in recorded if "reset" in c and "--hard" in c]
    assert len(reset_calls) == 1
    assert reset_calls[0][-3:] == ["reset", "--hard", "origin/main"]

    out = capsys.readouterr().out
    assert "Fast-forward not possible" in out


def test_cmd_update_rebases_local_patch_stack_when_preservation_is_enabled(
    monkeypatch, tmp_path, capsys
):
    """A managed local fix must survive the update that it launches."""
    _setup_update_mocks(monkeypatch, tmp_path)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(
        hermes_main,
        "_preserve_local_update_commits",
        lambda *args: True,
    )
    monkeypatch.setattr(
        hermes_main,
        "_ensure_update_merge_base",
        lambda *args: {"merge_base": "base123", "error": None, "fetch_steps": []},
    )
    applied = []

    def fake_apply(*args, **kwargs):
        applied.append(kwargs)
        return {
            "success": True,
            "safe_to_restore_stash": True,
            "recovery_ref": "refs/hermes/update-recovery/test",
            "used_rebase": True,
            "error": None,
        }

    monkeypatch.setattr(hermes_main, "_apply_pinned_git_update", fake_apply)
    monkeypatch.setattr(
        hermes_main, "_delete_update_recovery_ref", lambda *args: True
    )

    side_effect, recorded = _make_update_side_effect(ff_only_fails=True)
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    hermes_main.cmd_update(SimpleNamespace())

    assert len(applied) == 1
    assert applied[0]["target_sha"] == "abc123"
    assert applied[0]["merge_base"] == "base123"
    assert not any("reset" in call and "--hard" in call for call in recorded)
    assert "Preserving the pinned local patch stack" in capsys.readouterr().out


def test_cmd_update_no_reset_when_ff_only_succeeds(monkeypatch, tmp_path):
    """When --ff-only succeeds, no reset is attempted."""
    _setup_update_mocks(monkeypatch, tmp_path)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)

    side_effect, recorded = _make_update_side_effect()
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    hermes_main.cmd_update(SimpleNamespace())

    reset_calls = [c for c in recorded if "reset" in c and "--hard" in c]
    assert len(reset_calls) == 0


def test_post_git_failure_never_falls_back_to_zip(monkeypatch, tmp_path, capsys):
    """A late dependency/build failure must not overwrite the updated checkout."""
    _setup_update_mocks(monkeypatch, tmp_path)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    zip_calls = []
    monkeypatch.setattr(
        hermes_main, "_update_via_zip", lambda *a, **kw: zip_calls.append(1)
    )
    monkeypatch.setattr(
        hermes_main,
        "_refresh_active_lazy_features",
        lambda: (_ for _ in ()).throw(
            CalledProcessError(returncode=17, cmd=["post-git-stage"])
        ),
    )
    side_effect, _ = _make_update_side_effect()
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    with pytest.raises(SystemExit, match="1"):
        hermes_main.cmd_update(SimpleNamespace())

    assert zip_calls == []
    out = capsys.readouterr().out
    assert "No ZIP fallback was attempted" in out


# ---------------------------------------------------------------------------
# Non-main branch → auto-checkout main
# ---------------------------------------------------------------------------

def test_cmd_update_switches_to_main_from_feature_branch(monkeypatch, tmp_path, capsys):
    """When on a feature branch, update checks out main before pulling."""
    _setup_update_mocks(monkeypatch, tmp_path)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)

    side_effect, recorded = _make_update_side_effect(current_branch="fix/something")
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    hermes_main.cmd_update(SimpleNamespace())

    checkout_calls = [c for c in recorded if "checkout" in c and "main" in c]
    assert len(checkout_calls) == 1

    out = capsys.readouterr().out
    assert "fix/something" in out
    assert "switching to main" in out


def test_cmd_update_switches_to_main_from_detached_head(monkeypatch, tmp_path, capsys):
    """When in detached HEAD state, update checks out main before pulling."""
    _setup_update_mocks(monkeypatch, tmp_path)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)

    side_effect, recorded = _make_update_side_effect(current_branch="HEAD")
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    hermes_main.cmd_update(SimpleNamespace())

    checkout_calls = [c for c in recorded if "checkout" in c and "main" in c]
    assert len(checkout_calls) == 1

    out = capsys.readouterr().out
    assert "detached HEAD" in out


def test_cmd_update_restores_stash_and_branch_when_already_up_to_date(monkeypatch, tmp_path, capsys):
    """When on a feature branch with no updates, stash is restored and branch switched back."""
    _setup_update_mocks(monkeypatch, tmp_path)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)

    # Enable stash so it returns a ref
    monkeypatch.setattr(
        hermes_main, "_stash_local_changes_if_needed",
        lambda *a, **kw: "abc123deadbeef",
    )
    events = []
    checkout_calls = []
    monkeypatch.setattr(
        hermes_main,
        "_prepare_update_checkout_for_stash",
        lambda *a, **kw: checkout_calls.append(kw) or events.append("checkout") or True,
    )
    monkeypatch.setattr(
        hermes_main, "_restore_stashed_changes",
        lambda *a, **kw: events.append("stash") or True,
    )

    side_effect, recorded = _make_update_side_effect(
        current_branch="fix/something", commit_count="0",
    )
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    hermes_main.cmd_update(SimpleNamespace())

    assert events == ["checkout", "stash"]
    assert checkout_calls[0]["original_branch"] == "fix/something"
    assert checkout_calls[0]["updated_branch"] == "main"
    assert checkout_calls[0]["update_succeeded"] is False

    out = capsys.readouterr().out
    assert "Already up to date" in out


def test_cmd_update_success_restores_feature_checkout_before_stash(
    monkeypatch, tmp_path
):
    _setup_update_mocks(monkeypatch, tmp_path)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(
        hermes_main,
        "_stash_local_changes_if_needed",
        lambda *a, **kw: "abc123deadbeef",
    )
    events = []
    checkout_calls = []
    monkeypatch.setattr(
        hermes_main,
        "_prepare_update_checkout_for_stash",
        lambda *a, **kw: checkout_calls.append(kw) or events.append("checkout") or True,
    )
    monkeypatch.setattr(
        hermes_main,
        "_restore_stashed_changes",
        lambda *a, **kw: events.append("stash") or True,
    )
    side_effect, _ = _make_update_side_effect(current_branch="fix/something")
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    hermes_main.cmd_update(SimpleNamespace())

    assert events == ["checkout", "stash"]
    assert checkout_calls[0]["original_branch"] == "fix/something"
    assert checkout_calls[0]["updated_branch"] == "main"
    assert checkout_calls[0]["update_succeeded"] is True


def test_cmd_update_boundary_change_restores_checkout_before_stash(
    monkeypatch, tmp_path
):
    _setup_update_mocks(monkeypatch, tmp_path)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(
        hermes_main,
        "_stash_local_changes_if_needed",
        lambda *a, **kw: "abc123deadbeef",
    )
    monkeypatch.setattr(
        hermes_main, "_preserve_local_update_commits", lambda *a, **kw: True
    )
    monkeypatch.setattr(
        hermes_main,
        "_ensure_update_merge_base",
        lambda *a, **kw: {"merge_base": "base123", "error": None, "fetch_steps": []},
    )
    monkeypatch.setattr(
        hermes_main,
        "_apply_pinned_git_update",
        lambda *a, **kw: {
            "success": False,
            "safe_to_restore_stash": True,
            "recovery_ref": "refs/hermes/update-recovery/boundary",
            "used_rebase": False,
            "error": "UpdaterBoundaryChanged: pinned target moved",
        },
    )
    monkeypatch.setattr(
        hermes_main, "_delete_update_recovery_ref", lambda *a, **kw: True
    )
    events = []
    monkeypatch.setattr(
        hermes_main,
        "_prepare_update_checkout_for_stash",
        lambda *a, **kw: events.append("checkout") or True,
    )
    monkeypatch.setattr(
        hermes_main,
        "_restore_stashed_changes",
        lambda *a, **kw: events.append("stash") or True,
    )
    side_effect, _ = _make_update_side_effect(current_branch="fix/something")
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    with pytest.raises(SystemExit, match="1"):
        hermes_main.cmd_update(SimpleNamespace())

    assert events == ["checkout", "stash"]


def test_cmd_update_stops_before_dependencies_when_stash_restore_fails(
    monkeypatch, tmp_path, capsys
):
    _setup_update_mocks(monkeypatch, tmp_path)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(
        hermes_main,
        "_stash_local_changes_if_needed",
        lambda *a, **kw: "abc123deadbeef",
    )
    monkeypatch.setattr(
        hermes_main, "_preserve_local_update_commits", lambda *a, **kw: True
    )
    monkeypatch.setattr(
        hermes_main,
        "_ensure_update_merge_base",
        lambda *a, **kw: {"merge_base": "base123", "error": None, "fetch_steps": []},
    )
    monkeypatch.setattr(
        hermes_main,
        "_apply_pinned_git_update",
        lambda *a, **kw: {
            "success": True,
            "safe_to_restore_stash": True,
            "recovery_ref": "refs/hermes/update-recovery/stash-conflict",
            "used_rebase": True,
            "error": None,
        },
    )
    monkeypatch.setattr(
        hermes_main, "_prepare_update_checkout_for_stash", lambda *a, **kw: True
    )
    monkeypatch.setattr(
        hermes_main, "_restore_stashed_changes", lambda *a, **kw: False
    )
    deleted_refs = []
    monkeypatch.setattr(
        hermes_main,
        "_delete_update_recovery_ref",
        lambda *a, **kw: deleted_refs.append(1) or True,
    )
    post_git_stages = []
    monkeypatch.setattr(
        hermes_main,
        "_refresh_active_lazy_features",
        lambda: post_git_stages.append(1),
    )
    side_effect, _ = _make_update_side_effect()
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    with pytest.raises(SystemExit, match="1"):
        hermes_main.cmd_update(SimpleNamespace())

    assert deleted_refs == []
    assert post_git_stages == []
    out = capsys.readouterr().out
    assert "could not be restored safely" in out
    assert "Recovery ref retained" in out


def test_cmd_update_no_checkout_when_already_on_main(monkeypatch, tmp_path):
    """When already on main, no checkout is needed."""
    _setup_update_mocks(monkeypatch, tmp_path)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)

    side_effect, recorded = _make_update_side_effect()
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    hermes_main.cmd_update(SimpleNamespace())

    checkout_calls = [c for c in recorded if "checkout" in c]
    assert len(checkout_calls) == 0


def test_cmd_update_fetch_is_scoped_to_target_branch(monkeypatch, tmp_path):
    """The update fetch must name the target branch. A bare `git fetch origin`
    pulls every ref, and this repo has thousands of auto-generated branches, so
    an unscoped fetch can stall for minutes on a non-single-branch checkout."""
    _setup_update_mocks(monkeypatch, tmp_path)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)

    side_effect, recorded = _make_update_side_effect()
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    hermes_main.cmd_update(SimpleNamespace())

    fetch_calls = [c for c in recorded if "fetch" in c]
    assert [c[-3:] for c in fetch_calls] == [["fetch", "origin", "main"]]
    assert not any(c[-2:] == ["fetch", "origin"] for c in recorded)


# ---------------------------------------------------------------------------
# Fetch failure — friendly error messages
# ---------------------------------------------------------------------------

def test_cmd_update_network_error_shows_friendly_message(monkeypatch, tmp_path, capsys):
    """Network failures during fetch show a user-friendly message."""
    _setup_update_mocks(monkeypatch, tmp_path)

    side_effect, _ = _make_update_side_effect(
        fetch_fails=True,
        fetch_stderr="fatal: unable to access 'https://...': Could not resolve host: github.com",
    )
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    with pytest.raises(SystemExit, match="1"):
        hermes_main.cmd_update(SimpleNamespace())

    out = capsys.readouterr().out
    assert "Network error" in out


def test_cmd_update_auth_error_shows_friendly_message(monkeypatch, tmp_path, capsys):
    """Auth failures during fetch show a user-friendly message."""
    _setup_update_mocks(monkeypatch, tmp_path)

    side_effect, _ = _make_update_side_effect(
        fetch_fails=True,
        fetch_stderr="fatal: Authentication failed for 'https://...'",
    )
    monkeypatch.setattr(hermes_main.subprocess, "run", side_effect)

    with pytest.raises(SystemExit, match="1"):
        hermes_main.cmd_update(SimpleNamespace())

    out = capsys.readouterr().out
    assert "Authentication failed" in out


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


def test_non_interactive_discard_throws_changes_away(monkeypatch, tmp_path):
    """Gateway/chat-app update with discard mode drops the stash, never restores."""
    restore_calls, discard_calls, _ = _setup_setting_test(monkeypatch, tmp_path, "discard")

    hermes_main.cmd_update(SimpleNamespace(gateway=True))

    assert len(discard_calls) == 1
    assert len(restore_calls) == 0


def test_non_interactive_stash_restores_changes(monkeypatch, tmp_path):
    """Gateway/chat-app update with the default stash mode restores, never discards."""
    restore_calls, discard_calls, _ = _setup_setting_test(monkeypatch, tmp_path, "stash")

    hermes_main.cmd_update(SimpleNamespace(gateway=True))

    assert len(restore_calls) == 1
    assert len(discard_calls) == 0


def test_interactive_update_ignores_discard_setting(monkeypatch, tmp_path):
    """An interactive (TTY) terminal update always restores — the discard
    setting only governs non-interactive updates."""
    restore_calls, discard_calls, _ = _setup_setting_test(monkeypatch, tmp_path, "discard")
    # Force an interactive TTY so _non_interactive_update is False even though
    # the config says discard.
    monkeypatch.setattr(hermes_main.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(hermes_main.sys.stdout, "isatty", lambda: True)

    hermes_main.cmd_update(SimpleNamespace())  # no gateway, no --yes

    assert len(restore_calls) == 1
    assert len(discard_calls) == 0


def test_non_interactive_defaults_to_stash_when_setting_absent(monkeypatch, tmp_path):
    """A config with no update section falls back to stash (safe default)."""
    restore_calls, discard_calls, _ = _setup_setting_test(monkeypatch, tmp_path, "stash")
    # Override load_config to return a config with NO update section at all.
    monkeypatch.setattr(hermes_config, "load_config", lambda *a, **kw: {"model": {}})

    hermes_main.cmd_update(SimpleNamespace(gateway=True))

    assert len(restore_calls) == 1
    assert len(discard_calls) == 0


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
# Real shallow-clone update topology
# ---------------------------------------------------------------------------


def _fixture_git(cwd: Path, *args: str, check: bool = True):
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=check,
    )


def _configure_fixture_repo(repo: Path) -> None:
    _fixture_git(repo, "config", "user.email", "hermes-update-test@example.invalid")
    _fixture_git(repo, "config", "user.name", "Fixture User")
    _fixture_git(repo, "config", "core.autocrlf", "false")


def _make_shallow_update_fixture(
    tmp_path: Path,
    *,
    remote_commit_count: int = 40,
    conflict: bool = False,
):
    if shutil.which("git") is None:
        pytest.skip("git not available")

    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    client = tmp_path / "client"
    _fixture_git(tmp_path, "init", "--bare", str(remote))
    _fixture_git(tmp_path, "init", "-b", "main", str(seed))
    _configure_fixture_repo(seed)
    (seed / "shared.txt").write_text("base\n", encoding="utf-8")
    (seed / "dirty.bin").write_bytes(b"clean\x00base\n")
    _fixture_git(seed, "add", "-A")
    _fixture_git(seed, "commit", "-m", "base")
    _fixture_git(seed, "remote", "add", "origin", remote.as_uri())
    _fixture_git(seed, "push", "-u", "origin", "main")

    _fixture_git(
        tmp_path,
        "clone",
        "--depth",
        "1",
        "--branch",
        "main",
        remote.as_uri(),
        str(client),
    )
    _configure_fixture_repo(client)
    (client / ".gitignore").write_text("ignored-runtime.bin\n", encoding="utf-8")
    if conflict:
        (client / "shared.txt").write_text("local\n", encoding="utf-8")
    else:
        (client / "local-patch.txt").write_text("local patch\n", encoding="utf-8")
    _fixture_git(client, "add", "-A")
    _fixture_git(client, "commit", "-m", "local patch")

    for index in range(remote_commit_count):
        if conflict and index == 0:
            (seed / "shared.txt").write_text("remote\n", encoding="utf-8")
        else:
            (seed / f"remote-{index:03}.txt").write_text(
                f"remote {index}\n", encoding="utf-8"
            )
        _fixture_git(seed, "add", "-A")
        _fixture_git(seed, "commit", "-m", f"remote {index}")
    _fixture_git(seed, "push", "origin", "main")

    # Match an updater that starts from the official depth-1 install and has
    # fetched only the new tip. The local patch remains connected to the old
    # shallow boundary, while origin/main is a second shallow island.
    _fixture_git(client, "fetch", "--depth", "1", "origin", "main")
    original_head = _fixture_git(client, "rev-parse", "HEAD").stdout.strip()
    target_head = _fixture_git(client, "rev-parse", "origin/main").stdout.strip()
    assert (
        _fixture_git(
            client, "merge-base", original_head, target_head, check=False
        ).returncode
        != 0
    )
    return client, original_head, target_head


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pinned_update_repairs_depth_one_history_and_restores_dirty_files(tmp_path):
    client, original_head, target_head = _make_shallow_update_fixture(tmp_path)
    dirty = client / "dirty.bin"
    untracked = client / "untracked.bin"
    dirty.write_bytes(b"dirty\x00tracked\r\n")
    untracked.write_bytes(b"untracked\x00payload\xff")
    expected = (_digest(dirty), _digest(untracked))

    stash_ref = hermes_main._stash_local_changes_if_needed(["git"], client)
    assert stash_ref
    base = hermes_main._ensure_update_merge_base(
        ["git"], client, "main", original_head, target_head
    )
    assert base["merge_base"]
    assert base["error"] is None
    assert base["fetch_steps"][:2] == ["--deepen=32", "--deepen=128"]

    outcome = hermes_main._apply_pinned_git_update(
        ["git"],
        client,
        branch="main",
        pre_update_sha=original_head,
        target_sha=target_head,
        merge_base=base["merge_base"],
        preserve_local_commits=True,
        original_branch="main",
        original_head=original_head,
    )
    assert outcome["success"], outcome
    assert outcome["used_rebase"]
    assert hermes_main._restore_stashed_changes(
        ["git"], client, stash_ref, prompt_user=False
    )
    assert hermes_main._delete_update_recovery_ref(
        ["git"], client, outcome["recovery_ref"]
    )

    assert _fixture_git(client, "rev-list", f"{target_head}..HEAD", "--count").stdout.strip() == "1"
    assert _fixture_git(client, "log", "-1", "--format=%s").stdout.strip() == "local patch"
    assert _fixture_git(client, "merge-base", target_head, "HEAD").stdout.strip() == target_head
    assert (_digest(dirty), _digest(untracked)) == expected
    assert not hermes_main._git_update_operation_paths(["git"], client)
    assert not _fixture_git(client, "diff", "--name-only", "--diff-filter=U").stdout


def test_pinned_update_conflict_rolls_back_before_restoring_autostash(tmp_path):
    client, original_head, target_head = _make_shallow_update_fixture(
        tmp_path, remote_commit_count=1, conflict=True
    )
    dirty = client / "dirty.bin"
    untracked = client / "untracked.bin"
    ignored = client / "ignored-runtime.bin"
    dirty.write_bytes(b"dirty before conflict\x00")
    untracked.write_bytes(b"untracked before conflict\xff")
    ignored.write_bytes(b"ignored runtime sentinel\x00\xfe")
    expected = (_digest(dirty), _digest(untracked), _digest(ignored))
    stash_ref = hermes_main._stash_local_changes_if_needed(["git"], client)
    assert stash_ref

    base = hermes_main._ensure_update_merge_base(
        ["git"], client, "main", original_head, target_head
    )
    assert base["merge_base"]
    outcome = hermes_main._apply_pinned_git_update(
        ["git"],
        client,
        branch="main",
        pre_update_sha=original_head,
        target_sha=target_head,
        merge_base=base["merge_base"],
        preserve_local_commits=True,
        original_branch="main",
        original_head=original_head,
    )

    assert not outcome["success"]
    assert outcome["safe_to_restore_stash"]
    assert outcome["used_rebase"]
    assert outcome["recovery_ref"]
    assert "conflict" in outcome["error"].lower()
    assert _fixture_git(client, "rev-parse", "HEAD").stdout.strip() == original_head
    assert _fixture_git(client, "branch", "--show-current").stdout.strip() == "main"
    assert not hermes_main._git_update_operation_paths(["git"], client)
    assert hermes_main._restore_stashed_changes(
        ["git"], client, stash_ref, prompt_user=False
    )
    assert (_digest(dirty), _digest(untracked), _digest(ignored)) == expected
    assert not _fixture_git(client, "diff", "--name-only", "--diff-filter=U").stdout
    assert hermes_main._delete_update_recovery_ref(
        ["git"], client, outcome["recovery_ref"]
    )


def test_unrelated_full_histories_abort_without_head_mutation(tmp_path):
    local = tmp_path / "local"
    other = tmp_path / "other"
    _fixture_git(tmp_path, "init", "-b", "main", str(local))
    _fixture_git(tmp_path, "init", "-b", "main", str(other))
    for repo, content in ((local, "local\n"), (other, "other\n")):
        _configure_fixture_repo(repo)
        (repo / "history.txt").write_text(content, encoding="utf-8")
        _fixture_git(repo, "add", "-A")
        _fixture_git(repo, "commit", "-m", content.strip())
    original_head = _fixture_git(local, "rev-parse", "HEAD").stdout.strip()
    target_head = _fixture_git(other, "rev-parse", "HEAD").stdout.strip()
    _fixture_git(local, "remote", "add", "origin", other.as_uri())
    _fixture_git(local, "fetch", "origin", "main")

    result = hermes_main._ensure_update_merge_base(
        ["git"], local, "main", original_head, target_head
    )

    assert result["merge_base"] is None
    assert "no merge base" in result["error"].lower()
    assert result["fetch_steps"] == []
    assert _fixture_git(local, "rev-parse", "HEAD").stdout.strip() == original_head
    assert not hermes_main._git_update_operation_paths(["git"], local)

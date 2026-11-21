"""The upgrade rehearsal's transcript checks (RE2 migration, stage S2c) recognise what they must.

``_install_helpers.manual_step_lines`` and ``engine_error_lines`` judge real ``hermes update``
transcripts and logs in the e2e-upgrade lane, which runs only on release and dispatch runs; their
contract is pinned here so PR CI covers it. The hint lines below are what the updaters of v0.21.6,
v0.22.7, v0.23.0 and v0.23.3 and this tree print when an update leaves work to the user, with sample
values filled in.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tests.e2e.core.upgrade import _helpers as H
from tests.e2e.core.upgrade import _install_helpers as I

MANUAL_STEPS = (
    "  ⚠ Gateway multiplex migration did not finish: boom (run `hermes gateway migrate` later)",
    "  ⚠ Update follow-up 'dependencies' did not finish: uv failed (dependencies not installed yet — the next launch retries)",
    "  ⚠ Source update completion did not finish: child exited 1",
    "⚠ Gateway restart still owed: 1 gateway(s) still running the old code (or stopped).",
    "  Run `hermes gateway restart` (or `hermes -p <profile> gateway restart` for a named",
    "  ⚠ Could not restart every paused gateway: boom. Run `hermes update` or ",
    "  ⚠ default: could not be signalled (EPERM) — restart it by hand",
    "    Restart manually: hermes gateway run",
    "hermes: source-update completion failed: boom; running with the previous dependencies — run `hermes update` to finish it",
    "hermes: a source update could not be finished automatically (3 attempts); run `hermes update` from a shell to finish it",
    "hermes: running with the previous dependencies after 3 failed syncs; next automatic retry in 5 min, or run `hermes update` to finish it",
    "hermes: a source update is unfinished; run `hermes update` from a shell to finish it",
    "  ✗ Restored local changes made the Hermes agent unexecutable.",
    "  Restore manually after fixing them: git stash apply stash@{0}",
    "Restore your changes later with: git stash apply stash@{0}",
    "To complete this update, run `hermes update` again.",
    "  Try ``hermes update`` again later once a fix lands.",
    "  ✗ Rollback failed. Recover manually with:",
    "Skipped. Run 'hermes config migrate' later to configure.",
    "Run 'hermes config migrate' manually to configure.",
    "config migration stopped at v31; run `hermes config migrate` to retry the remaining steps",
    "  ⚠ Some stopped backends could not be relaunched automatically; restart them manually (hermes serve --host <ip> --port <port>).",
    "    Unmask it and run `hermes gateway migrate` to fold them into one gateway.",
    "hermes: cannot start: the dependencies this process runs on lack google-re2; run `hermes pm install` to install them",
    "✗ Cannot update: the dependencies lack google-re2, and installing them did not finish.",
    "  Skipping upstream setup (non-interactive run).\n  Add it later with: git remote add upstream x",
)

# What a clean `hermes update --yes` prints around them, none of it owed to the user.
CLEAN = """\
→ Fetching updates...
✓ Code updated to 1a2b3c4d
  ℹ API keys require manual entry: hermes config migrate
✓ Configuration updated!
  ✓ Profile 'work': config format updated (v30 → v31)
  ✓ Gateway restart skipped: no gateway is running.
→ Migrating per-profile gateways onto one multiplexed default gateway...
✓ Migrated: the default gateway now serves 2 profiles: default, work
  ↻ resumed: this update completes the work it still owed.
\x1b[32m✓ Update complete!\x1b[0m
"""


@pytest.mark.parametrize("line", MANUAL_STEPS)
def test_every_manual_step_hint_is_a_manual_step(line):
    assert I.manual_step_lines(CLEAN + line + "\n"), line


def test_a_clean_update_transcript_owes_nothing():
    assert I.manual_step_lines(CLEAN) == []


ENGINE_ERRORS = (
    "ModuleNotFoundError: No module named 're2'",
    "  File \"/x/hermes_re/_engine.py\", line 12, in _load",
    "hermes: cannot start: the dependencies this process runs on lack google-re2; run `hermes pm install`",
    "metrics row lost: ModuleNotFoundError",
    "No module named 're2._re2'",
    "re2.error: invalid perl operator: (?<",
    "✗ Cannot update: the dependencies lack google-re2, and installing it failed",
    "  × Failed to build `google-re2==1.1.20251105`",
    "ImportError: cannot import name 'compile' from 'hermes_re'",
)
NOT_ENGINE = (
    " + google-re2==1.1.20251105",
    "Building google-re2==1.1.20251105",
    "Installed 3 packages in 12ms",
    "Prepared 2 packages in 1.2s",
    "→ Checking for updates (pre2 channel)...",
    "Downloading google-re2 (1.2MiB)",
    "Collecting google-re2==1.1.20251105",
    "Successfully installed google-re2-1.1.20251105 pybind11-2.13.6",
    "  ✓ Regex engine: google-re2 1.1.20251105 (RE2)",
    "Updated google-re2 v1.1.20251105",
)


@pytest.mark.parametrize("line", ENGINE_ERRORS)
def test_a_regex_engine_failure_is_found(line):
    assert I.engine_error_lines(f"ok\n{line}\nok\n") == [line.strip()]


@pytest.mark.parametrize("line", NOT_ENGINE)
def test_install_progress_and_plain_lines_are_not_engine_failures(line):
    assert I.engine_error_lines(line) == []


_ENV = {
    "PATH": os.environ.get("PATH", "/usr/bin:/bin"), "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_AUTHOR_NAME": "e2e", "GIT_AUTHOR_EMAIL": "e2e@example.invalid",
    "GIT_COMMITTER_NAME": "e2e", "GIT_COMMITTER_EMAIL": "e2e@example.invalid",
}


def test_a_start_carries_a_fix_only_from_the_commit_that_has_it(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo, env=_ENV, capture_output=True, text=True,
                              check=True).stdout.strip()

    git("init", "-q", "-b", "main")
    commits = []
    for name in ("old release", "the fix", "new release"):
        git("commit", "-q", "--allow-empty", "-m", name)
        commits.append(git("rev-parse", "HEAD"))
    monkeypatch.setattr(H, "WORKTREE", repo)
    monkeypatch.setitem(I.FIXED_BY, "#1", commits[1])
    assert not I.start_carries("#1", commits[0])
    assert I.start_carries("#1", commits[1]) and I.start_carries("#1", commits[2])

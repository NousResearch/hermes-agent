"""install.ps1 lands a new or rerun official install on the latest stable release.

The Windows twin of test_install_sh_stable_default: real Windows PowerShell runs
Stage-Repository against a local origin the official URL is redirected to, with the
release lookup and the pinned-Git download stubbed (the runner's git serves).
"""
from pathlib import Path
import os
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = "https://github.com/NousResearch/hermes-agent.git"

pytestmark = pytest.mark.platforms("windows")


def _git(cwd, *args, env=None):
    return subprocess.run(
        ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
         "-c", "commit.gpgsign=false", *args],
        cwd=cwd, check=True, capture_output=True, text=True, stdin=subprocess.DEVNULL, env=env,
    ).stdout.strip()


@pytest.mark.parametrize("side,start,channel,tagged,lands", [
    (False, None, "", True, "release"),        # fresh install: the release, not main's tip
    (False, None, "main", True, "tip"),        # explicit opt-out: main's tip
    (False, "before", "", True, "release"),    # rerun behind the release: moves forward to it
    (False, "tip", "", True, "tip"),           # rerun already containing it: never moves back
    (False, None, "", False, None),            # release tag unreachable: fail, publish nothing
    (False, "before", "", False, "before"),    # ... a rerun keeps its checkout instead
    # The release cut on a release branch, main past the fork point (v0.22.2's shape): main moves
    # back only through `hermes update`, whose downgrade guard the installer cannot run.
    (True, "tip", "", True, "tip"),
    (True, "tip", "stable", True, "tip"),
    (True, "before", "", True, "release"),
])
def test_install_ps1_lands_on_the_latest_stable_release(tmp_path, side, start, channel, tagged, lands):
    powershell = shutil.which("powershell")
    if not powershell:
        pytest.skip("Windows PowerShell is required")
    origin, home = tmp_path / "origin", tmp_path / "home"
    install = home / "hermes-agent"
    origin.mkdir()
    home.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _git(origin, "commit", "-q", "--allow-empty", "-m", "before")
    if side:
        _git(origin, "checkout", "-q", "-b", "release/1.2")
    _git(origin, "commit", "-q", "--allow-empty", "-m", "release")
    commits = {"before": _git(origin, "rev-parse", "HEAD~1"), "release": _git(origin, "rev-parse", "HEAD")}
    if tagged:
        _git(origin, "tag", "-a", "v1.2.3", "-m", "release")
    _git(origin, "checkout", "-q", "main")
    _git(origin, "commit", "-q", "--allow-empty", "-m", "newer")
    commits["tip"] = _git(origin, "rev-parse", "HEAD")
    gitconfig = tmp_path / "gitconfig"
    gitconfig.write_text(f'[url "{origin.as_uri()}"]\n  insteadOf = {OFFICIAL}\n', encoding="utf-8")
    env = {**os.environ, "GIT_CONFIG_GLOBAL": str(gitconfig), "NO_COLOR": "1"}
    env.pop("HERMES_INSTALL_CHANNEL", None)
    if start:
        _git(tmp_path, "clone", "-q", OFFICIAL, str(install), env=env)
        _git(install, "checkout", "-q", "--detach", commits[start], env=env)
    wrapper = tmp_path / "stage.ps1"
    wrapper.write_text(
        f". '{ROOT / 'scripts' / 'install.ps1'}' -InstallDir '{install}' -HermesHome '{home}' -Channel '{channel}'\n"
        "function Ensure-Git { return $true }\n"
        "function Get-LatestStableRelease { return 'v1.2.3' }\n"
        "Stage-Repository\n", encoding="utf-8")

    result = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(wrapper)],
        check=False, capture_output=True, text=True, encoding="utf-8", errors="replace",
        stdin=subprocess.DEVNULL, timeout=300, env=env,
    )

    if lands is None:
        assert result.returncode != 0 and not install.exists(), result.stdout + result.stderr
        return
    assert result.returncode == 0, result.stdout + result.stderr
    assert _git(install, "rev-parse", "HEAD", env=env) == commits[lands], result.stdout + result.stderr
    assert _git(install, "status", "--porcelain", env=env) == "", result.stdout + result.stderr
